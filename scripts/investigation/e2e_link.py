#!/usr/bin/env python
"""Offline end-to-end replica of the deployed DJSCC link, as close to the lab run as
possible without radios:

  Kodak image -> socket_tx (encode + interleave)
  -> djscc_tx.grc enabled blocks (robust header, BPSK x header_boost, carrier
     allocator with sync words * 4, IFFT, CP, x0.03)
  -> channel emulator (AWGN at an "emulator SNR" + switchable hardware impairments)
  -> djscc_rx.grc enabled blocks (Schmidl-Cox sync, delay/freq-mod CFO correction,
     header/payload demux, chanest with sync words * 3, simpledfe header equalizer,
     robust header parser, static payload equalizer, serializer, x header_boost)
  -> socket_rx logic (anchor/slot assignment, idle-gap flush, zero-fill, de-interleave,
     clip-mag 5, fixed CSI) -> decoder -> PSNR (slide-notebook preprocessing).

Besides PSNR it reports, per run, what the decoder actually received vs the clean
latent (effective SNR after LS gain, gain, phase drift inside a packet), packet loss
and how many images were slot-shifted by a lost first packet.

Impairments (all off by default): --cfo-hz, --sfo-ppm, --pn-linewidth (Wiener
phase noise, Hz), --iq-amp-db/--iq-phase-deg, --dc, --clip (TX amplitude clip,
relative to the frame RMS), --taps (exponential multipath, number of taps).
--cpe applies pilot-based common-phase-error correction per OFDM symbol after
the static equalizer (what the deployed RX lacks).

Run inside the `demo` env with the GNU Radio env vars, e.g.:
  python scripts/investigation/e2e_link.py --snr 0 8 16 28 --images 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import pmt
import yaml
from gnuradio import analog, blocks, digital, fft, gr
from gnuradio import deepjscc
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from jscc import load_codec  # noqa: E402

FFT, CP, PKT, NDATA = 64, 16, 960, 48
OCC = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7))
       + list(range(8, 21)) + list(range(22, 27)),)
PIL_C, PIL_S = ((-21, -7, 7, 21),), ((1, 1, 1, -1),)
DATA_BINS = np.array([k + FFT // 2 for k in OCC[0]])
PILOT_BINS = np.array([k + FFT // 2 for k in PIL_C[0]])
PILOT_SYMS = np.array(PIL_S[0], dtype=complex)
HEADER_BOOST = 2.0
SAMP_RATE = 1e6
IDLE_GAP_S = 0.3
BPSK = digital.constellation_bpsk()

_v = {b["name"]: b["parameters"]["value"] for b in
      yaml.safe_load(open(f"{ROOT}/transmitter/gnu_radio/djscc_tx.grc"))["blocks"] if b["id"] == "variable"}
SW1 = np.array(eval(_v["sync_word1"].replace("sw_multiplier*", ""), {"np": np}), dtype=float)
SW2 = np.array(eval(_v["sync_word2"].replace("sw_multiplier*", ""), {"np": np}), dtype=float)


def make_interleaver(n, slots):
    bps = n // slots
    p = np.arange(n, dtype=np.int64)
    fwd = (p % slots) * bps + (p // slots)
    inv = np.empty(n, dtype=np.int64)
    inv[fwd] = p
    return fwd, inv


# ── TX: socket_tx + djscc_tx.grc ─────────────────────────────────────────────
def gr_tx(symbols: np.ndarray, sw_tx: float, post_pad: int = 0) -> np.ndarray:
    hf = deepjscc.packet_header_ofdm_robust(OCC, n_syms=1, len_tag_key="packet_len",
        frame_len_tag_key="packet_len", num_tag_key="packet_num", bits_per_header_sym=1,
        bits_per_payload_sym=8, scramble_header=True, num_bits=16, use_fec=True)
    tb = gr.top_block()
    src = blocks.vector_source_c(symbols.astype(np.complex64).tolist(), False)
    s2t = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, PKT, "packet_len")
    c2r, f2u = blocks.complex_to_real(1), blocks.float_to_uchar()
    hgen = digital.packet_headergenerator_bb(hf.formatter(), "packet_len")
    c2s = digital.chunks_to_symbols_bc(BPSK.points(), 1)
    hb = blocks.multiply_const_cc(HEADER_BOOST)
    mux = blocks.tagged_stream_mux(gr.sizeof_gr_complex, "packet_len", 0)
    alloc = digital.ofdm_carrier_allocator_cvc(FFT, OCC, PIL_C, PIL_S, (sw_tx * SW1, sw_tx * SW2), "packet_len", True)
    ifft = fft.fft_vcc(FFT, False, (), True)
    cp = digital.ofdm_cyclic_prefixer(FFT, FFT + CP, 0, "packet_len")
    sc = blocks.multiply_const_cc(0.03)
    snk = blocks.vector_sink_c()
    tb.connect(src, s2t, c2r, f2u, hgen, c2s, hb, (mux, 0))
    tb.connect(s2t, (mux, 1))
    if post_pad:   # djscc_tx.grc fix: Burst Shaper appends post_pad zeros per frame
        shaper = digital.burst_shaper_cc([], 0, post_pad, False, "packet_len")
        tb.connect(mux, alloc, ifft, cp, shaper, sc, snk)
    else:
        tb.connect(mux, alloc, ifft, cp, sc, snk)
    tb.run()
    return np.array(snk.data(), np.complex64)


# ── channel emulator ─────────────────────────────────────────────────────────
def channel(x, active, snr_db, a, rng):
    """x: baseband incl. idle gaps; active: bool mask of transmitted samples (for the
    SNR reference = mean power of the active signal over the full sample band)."""
    y = x.astype(np.complex128)
    if a.clip > 0:                                     # TX DAC/PA amplitude clip
        rms = np.sqrt(np.mean(np.abs(y[active]) ** 2))
        m = np.abs(y) > a.clip * rms
        y[m] = y[m] / np.abs(y[m]) * a.clip * rms
    if a.taps > 1:                                     # static multipath
        pdp = np.exp(-np.arange(a.taps) / max(a.taps / 3, 1))
        h = (rng.standard_normal(a.taps) + 1j * rng.standard_normal(a.taps)) * np.sqrt(pdp / pdp.sum() / 2)
        y = np.convolve(y, h)[: len(y)]
    p_sig = np.mean(np.abs(x[active]) ** 2)
    n = np.arange(len(y))
    if a.sfo_ppm:
        t = n * (1 + a.sfo_ppm * 1e-6)
        y = np.interp(t, n, y.real) + 1j * np.interp(t, n, y.imag)
    phase = 2 * np.pi * a.cfo_hz / SAMP_RATE * n
    if a.pn_linewidth > 0:                             # Wiener phase noise (TX+RX LO)
        phase = phase + np.cumsum(rng.standard_normal(len(y)) * np.sqrt(2 * np.pi * a.pn_linewidth / SAMP_RATE))
    y = y * np.exp(1j * phase)
    # SNR = signal power / noise power inside the "useful bandwidth" snr_bw (as the Azimuth emulator
    # defines it: flat noise, SNR measured in the signal bandwidth). Noise per complex sample over the
    # whole 1 MS/s band is therefore scaled by SAMP_RATE / snr_bw (snr_bw = SAMP_RATE -> plain per-sample SNR).
    bw_scale = SAMP_RATE / (getattr(a, "snr_bw", 0) or SAMP_RATE)
    y = y + np.sqrt(p_sig / 10 ** (snr_db / 10) * bw_scale / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    if a.iq_amp_db or a.iq_phase_deg:                  # RX IQ imbalance
        g, ph = 10 ** (a.iq_amp_db / 20), np.radians(a.iq_phase_deg)
        i, q = y.real, y.imag
        y = i + 1j * g * (np.sin(ph) * i + np.cos(ph) * q)
    if a.dc:
        y = y + a.dc * np.sqrt(p_sig)
    return y.astype(np.complex64)


# ── RX: djscc_rx.grc ─────────────────────────────────────────────────────────
def gr_rx(rx: np.ndarray, sw_rx: float, sync_threshold: float = 0.65, gr_cpe: bool = False):
    hf = deepjscc.packet_header_ofdm_robust(OCC, n_syms=1, len_tag_key="packet_len",
        frame_len_tag_key="frame_len", num_tag_key="packet_num", bits_per_header_sym=1,
        bits_per_payload_sym=8, scramble_header=True, expected_packet_len=PKT, num_bits=16, use_fec=True)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"e2e_rx_{os.getpid()}.cf32")
    rx.tofile(path)
    tb = gr.top_block()
    rs = blocks.file_source(gr.sizeof_gr_complex, path, False)
    sync = digital.ofdm_sync_sc_cfb(FFT, CP, False, sync_threshold)
    dl = blocks.delay(gr.sizeof_gr_complex, FFT + CP)
    fm = analog.frequency_modulator_fc(-2.0 / FFT)
    mul = blocks.multiply_cc()
    dmx = digital.header_payload_demux(3, FFT, CP, "frame_len", "", True, gr.sizeof_gr_complex,
                                       "rx_time", int(SAMP_RATE), (), 0)
    f0 = fft.fft_vcc(FFT, True, (), True)
    ce = digital.ofdm_chanest_vcvc(sw_rx * SW1, sw_rx * SW2, 1, int(FFT / 4), 3, False)
    heq = digital.ofdm_equalizer_simpledfe(FFT, BPSK.base(), OCC, PIL_C, PIL_S)
    e0 = digital.ofdm_frame_equalizer_vcvc(heq.base(), CP, "frame_len", True, 1)
    ser0 = digital.ofdm_serializer_vcc(FFT, OCC, "frame_len")
    dec = digital.constellation_decoder_cb(BPSK.base())
    hp = digital.packet_headerparser_b(hf.base())
    f1 = fft.fft_vcc(FFT, True, (), True)
    peq = digital.ofdm_equalizer_static(FFT, OCC, PIL_C, PIL_S)
    e1 = digital.ofdm_frame_equalizer_vcvc(peq.base(), CP, "frame_len", True, 0)
    vs = blocks.vector_sink_c(FFT)
    vraw = blocks.vector_sink_c(FFT)
    tb.connect(rs, sync); tb.connect((sync, 0), fm, (mul, 1)); tb.connect(rs, dl, (mul, 0))
    tb.connect(mul, (dmx, 0)); tb.connect((sync, 1), (dmx, 1))
    tb.connect((dmx, 0), f0, ce, e0, ser0, dec, hp)
    tb.msg_connect(hp, "header_data", dmx, "header_data")
    if gr_cpe:     # djscc_rx.grc fix: embedded cfo_ramp_corrector block
        from cfo_ramp_corrector import blk as CfoRamp
        corr = CfoRamp(FFT, [int(b) for b in PILOT_BINS], True)
        tb.connect((dmx, 1), f1, e1, (corr, 0))
        tb.connect(f1, (corr, 1))
        tb.connect(corr, vs)
    else:
        tb.connect((dmx, 1), f1, e1, vs)
    tb.connect(f1, vraw)
    tb.run()
    os.remove(path)
    data = np.array(vs.data(), np.complex64).reshape(-1, FFT)
    raw = np.array(vraw.data(), np.complex64).reshape(-1, FFT)
    frames = []                                        # (packet_num, rx_sample_time, [n_sym, 64])
    tags = sorted(vs.tags(), key=lambda t: t.offset)
    starts = {}
    for t in tags:
        k = pmt.symbol_to_string(t.key)
        if k == "packet_num":
            starts.setdefault(t.offset, {})["pn"] = pmt.to_long(t.value)
        elif k == "frame_len":
            starts.setdefault(t.offset, {})["len"] = pmt.to_long(t.value)
        elif k == "rx_time":
            v = t.value
            starts.setdefault(t.offset, {})["t"] = pmt.to_uint64(pmt.tuple_ref(v, 0)) + pmt.to_double(pmt.tuple_ref(v, 1))
    for off in sorted(starts):
        d = starts[off]
        if "pn" not in d or "len" not in d:
            continue
        frames.append((d["pn"], d.get("t", np.nan), (data[off: off + d["len"]], raw[off: off + d["len"]])))
    return frames


def serialize(frame, cpe: bool):
    """ofdm_serializer_vcc (input_is_shifted) + optional residual-CFO correction.

    The static equalizer never updates data carriers, so a residual CFO shows up as
    a linear phase ramp across the packet's OFDM symbols. With cpe=True the ramp
    slope is estimated from the raw (pre-EQ) pilots, averaging the phase of
    consecutive-symbol pilot products over all symbols and pilots, then removed."""
    v, raw = frame
    if cpe and len(raw) > 1:
        p = raw[:, PILOT_BINS]
        slope = np.angle(np.sum(p[1:] * np.conj(p[:-1])))
        v = v * np.exp(-1j * slope * np.arange(len(v)))[:, None]
    return v[:, DATA_BINS].reshape(-1)[:PKT]


# ── socket_rx reassembly ─────────────────────────────────────────────────────
def socket_rx(frames, n_slots, expected, interleave, cpe, decode_fn, true_pn0):
    """Mirror receiver/socket_rx.py: anchor = first pn after an idle gap,
    slot = (pn-anchor) % n_slots, flush on last slot / image change / idle gap."""
    fwd, _ = make_interleaver(n_slots * PKT, n_slots)
    out, buf = [], np.zeros(n_slots * PKT, np.complex64)
    seen = np.zeros(n_slots, bool)
    state = {"anchor": None, "cur": None, "last_t": None, "first_true_slot": None}

    def flush():
        if not seen.any():
            return
        b = buf.copy()
        b[np.repeat(~seen, PKT)] = 0
        s = (b[fwd] if interleave else b)[:expected] * HEADER_BOOST
        m = np.abs(s) > 5.0
        s[m] = s[m] / np.abs(s[m]) * 5.0
        out.append(dict(symbols=s, seen=seen.copy(), shift=state["first_true_slot"]))
    for pn, t, vec in frames:
        if state["last_t"] is not None and np.isfinite(t) and t - state["last_t"] > IDLE_GAP_S and state["cur"] is not None:
            flush(); seen[:] = False; buf[:] = 0; state.update(cur=None, anchor=None)
        state["last_t"] = t
        if state["anchor"] is None:
            state["anchor"] = pn
            state["first_true_slot"] = (pn - true_pn0) % n_slots   # 0 = correctly anchored
        delta = pn - state["anchor"]
        slot, img = delta % n_slots, delta // n_slots
        if state["cur"] is None:
            state["cur"] = img
        elif img != state["cur"]:
            flush(); seen[:] = False; buf[:] = 0; state["cur"] = img
        buf[slot * PKT:(slot + 1) * PKT] = serialize(vec, cpe)
        seen[slot] = True
        if slot == n_slots - 1:
            flush(); seen[:] = False; buf[:] = 0; state["cur"] = None
    if state["cur"] is not None:
        flush()
    return out


def psnr(a, b):
    return 10 * np.log10(255 ** 2 / np.mean((a.astype(float) - b.astype(float)) ** 2))


_SSIM = None


def ssim(a, b):
    """Kaira SSIM (data_range=1) on HxWx3 uint8 RGB images, as in the plotting notebooks."""
    global _SSIM
    import torch
    from kaira.metrics.image import SSIM
    if _SSIM is None:
        _SSIM = SSIM(data_range=1.0).eval()
    t = lambda im: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    with torch.no_grad():
        return float(_SSIM(t(a), t(b)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi")
    ap.add_argument("--csi", type=float, default=19.0, help="fixed decoder CSI (socket_rx --snr-db)")
    ap.add_argument("--kodak", default="/Users/marcellobullo/Downloads/kodak_dataset")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 28])
    ap.add_argument("--no-interleave", action="store_true")
    ap.add_argument("--sw-tx", type=float, default=4.0)
    ap.add_argument("--sw-rx", type=float, default=3.0)
    ap.add_argument("--cpe", action="store_true")
    ap.add_argument("--cfo-hz", type=float, default=0.0)
    ap.add_argument("--sfo-ppm", type=float, default=0.0)
    ap.add_argument("--pn-linewidth", type=float, default=0.0)
    ap.add_argument("--iq-amp-db", type=float, default=0.0)
    ap.add_argument("--iq-phase-deg", type=float, default=0.0)
    ap.add_argument("--dc", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--taps", type=int, default=0)
    ap.add_argument("--frame-gap", type=int, default=0, help="idle samples between OFDM frames (TX fix)")
    ap.add_argument("--gr-gap", type=int, default=0, help="frame gap via the GNU Radio Burst Shaper (as in the fixed .grc)")
    ap.add_argument("--gr-cpe", action="store_true", help="phase-ramp fix via the GNU Radio cfo_ramp_corrector block")
    ap.add_argument("--sync-thr", type=float, default=0.65)
    ap.add_argument("--gap-s", type=float, default=0.5, help="idle air time between images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csi-follow-snr", action="store_true",
                    help="set encoder and decoder CSI to the channel SNR at every sweep point (overrides --csi)")
    ap.add_argument("--snr-bw", type=float, default=0.0,
                    help="bandwidth (Hz) in which the SNR is defined; 0 = the full 1 MS/s band. 812500 = the 52 occupied carriers")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-dir", default="", help="save every reconstructed image as <dir>/snr_<S>dB/snr<S>_<kodim#>.png")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "e2e"))
    a = ap.parse_args()
    interleave = not a.no_interleave
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    enc = load_codec(a.model, "encoder", packet_len=PKT, snr_db=a.csi)
    dec = load_codec(a.model, "decoder", packet_len=PKT, snr_db=a.csi)
    expected = enc.expected_complex_items
    paths = sorted(glob.glob(f"{a.kodak}/kodim*.png"))[: a.images]
    W, H = 768, 512
    def build():
        origs, lat, tx_stream = [], [], []
        for p in paths:
            bgr = cv2.resize(cv2.imread(p), (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            origs.append(cv2.resize(np.array(Image.open(p))[..., :3], (W, H)))  # notebook reference
            s = enc.encode(rgb.tobytes())
            lat.append(s[:expected].copy())
            tx_stream.append(s[make_interleaver(len(s), len(s) // PKT)[1]] if interleave else s)
        n_slots = len(tx_stream[0]) // PKT
        print(f"[*] {len(paths)} images, {n_slots} pkts/image, model={a.model}", flush=True)

        t0 = time.time()
        bb = gr_tx(np.concatenate(tx_stream), a.sw_tx, a.gr_gap)
        frame_samples = (2 + 1 + PKT // NDATA) * (FFT + CP) + a.gr_gap
        per_img = n_slots * frame_samples
        if len(bb) < per_img * len(paths):   # Burst Shaper omits the suffix of the stream's last frame
            bb = np.concatenate([bb, np.zeros(per_img * len(paths) - len(bb), np.complex64)])
        assert len(bb) == per_img * len(paths), (len(bb), per_img)
        if a.frame_gap:
            z = np.zeros(a.frame_gap, np.complex64)
            bb = np.concatenate([np.concatenate([bb[i*frame_samples:(i+1)*frame_samples], z]) for i in range(len(bb)//frame_samples)])
            per_img = n_slots * (frame_samples + a.frame_gap)
        gap = np.zeros(int(a.gap_s * SAMP_RATE), np.complex64)
        pieces, active = [gap], [np.zeros(len(gap), bool)]
        for k in range(len(paths)):
            pieces += [bb[k * per_img:(k + 1) * per_img], gap]
            active += [np.ones(per_img, bool), np.zeros(len(gap), bool)]
        x, act = np.concatenate(pieces), np.concatenate(active)
        print(f"[*] TX {len(x)/1e6:.2f} Ms in {time.time()-t0:.1f}s; PAPR={10*np.log10(np.max(np.abs(bb)**2)/np.mean(np.abs(bb)**2)):.1f} dB", flush=True)
        return origs, lat, n_slots, x, act

    origs, lat, n_slots, x, act = build()

    os.makedirs(a.out, exist_ok=True)
    results = []
    for snr in a.snr:
        if a.csi_follow_snr:          # feed the actual channel SNR to the encoder (SNR-adaptive models) and the decoder
            if hasattr(enc, 'set_csi'):
                enc.set_csi(snr)          # channel-blind encoders (ConvNeXt) have no CSI input
            dec.set_csi(snr)
            origs, lat, n_slots, x, act = build()
        rng = np.random.default_rng(a.seed + int(snr * 10))
        t0 = time.time()
        frames = gr_rx(channel(x, act, snr, a, rng), a.sw_rx, a.sync_thr, a.gr_cpe)
        imgs = socket_rx(frames, n_slots, expected, interleave, a.cpe, None, 0)
        rows = []
        for k, im in enumerate(imgs[: len(paths)]):
            s_rx = im["symbols"]
            # identify which image this is (robust to lost/merged images)
            cs = [abs(np.vdot(l, s_rx)) / (np.linalg.norm(l) * np.linalg.norm(s_rx) + 1e-9) for l in lat]
            i = int(np.argmax(cs))
            s = lat[i]
            ok = np.repeat(im["seen"], PKT)[make_interleaver(n_slots * PKT, n_slots)[0]][:expected] if interleave else np.repeat(im["seen"], PKT)[:expected]
            g = np.vdot(s[ok], s_rx[ok]) / np.vdot(s[ok], s[ok])
            e = s_rx[ok] - g * s[ok]
            snr_eff = 10 * np.log10(abs(g) ** 2 * np.mean(abs(s[ok]) ** 2) / np.mean(abs(e) ** 2))
            rgb = np.frombuffer(dec.decode(s_rx), np.uint8).reshape(H, W, 3)
            if a.save_dir:
                d = os.path.join(a.save_dir, f"snr_{int(snr)}dB")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"snr{int(snr)}_{i + 1:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rows.append(dict(img=os.path.basename(paths[i]), corr=float(cs[i]), psnr=float(psnr(rgb, origs[i])),
                             ssim=ssim(rgb, origs[i]),
                             gain=float(abs(g)), phase_deg=float(np.degrees(np.angle(g))), snr_eff=float(snr_eff),
                             missing=int((~im["seen"]).sum()), shifted=int(im["shift"] != 0)))
        n_img = len(rows)
        lost_imgs = len(paths) - n_img
        mean_psnr = float(np.mean([r["psnr"] for r in rows] + [0.0] * 0)) if rows else float("nan")
        res = dict(snr=snr, images_decoded=n_img, images_lost=lost_imgs, psnr=mean_psnr,
                   ssim=float(np.mean([r["ssim"] for r in rows])) if rows else float("nan"),
                   snr_eff=float(np.median([r["snr_eff"] for r in rows])) if rows else float("nan"),
                   gain=float(np.median([r["gain"] for r in rows])) if rows else float("nan"),
                   missing_pkts=float(np.mean([r["missing"] for r in rows])) if rows else float("nan"),
                   shifted_imgs=int(sum(r["shifted"] for r in rows)),
                   frames_rx=len(frames), frames_tx=n_slots * len(paths), per_image=rows,
                   secs=time.time() - t0)
        results.append(res)
        print(f"SNR {snr:5.1f} dB | PSNR {res['psnr']:6.2f} | SSIM {res['ssim']:.3f} | SNR_eff {res['snr_eff']:6.2f} dB | gain {res['gain']:.3f} | "
              f"frames {res['frames_rx']}/{res['frames_tx']} | miss/img {res['missing_pkts']:.1f} | "
              f"shifted {res['shifted_imgs']}/{n_img} | lost imgs {lost_imgs} | {res['secs']:.0f}s", flush=True)
    name = a.tag or "run"
    json.dump(dict(args=vars(a), results=results), open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
