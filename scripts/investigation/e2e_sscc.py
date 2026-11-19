#!/usr/bin/env python
"""Offline end-to-end replica of the conventional SSCC link (JPEG + LDPC + OFDM), the
counterpart of e2e_link.py for the DJSCC chain:

  Kodak image -> ConventionalEncoder (JPEG fitted to the budget + LDPC + interleaver)
  -> conventional_tx.grc blocks (repack bits, constellation map, robust header,
     carrier allocator with sync words * 4, IFFT, CP, x0.03)
  -> AWGN channel (same `channel` as e2e_link.py, SNR defined on the active signal)
  -> conventional_rx.grc blocks (Schmidl-Cox sync, header/payload demux, chanest,
     header simpledfe EQ, robust header parser, payload simpledfe EQ, serializer,
     x header_boost, soft constellation decoder with npwr = 10**(-SNR/10))
  -> socket_conventional_rx logic (anchor/slot assignment, zero-filled lost slots,
     de-interleave, soft LDPC BP decode, JPEG decode) -> PSNR / SSIM.

SNR-dependent / per-run parameters (same as the lab scripts): the modulation order and
LDPC (n,k) select the flowgraph/bit budget, and the soft demapper noise power npwr is
set from the channel SNR. `--npwr-snr` overrides the SNR used for npwr.

Output (per SNR, per (n,k,mod) combo), same layout as results/conventional:
  <out>/snr_<S>dB/ldpc_n<N>_k<K>_mod<M>/image_<id>.png  (or image_<id>_fail.png, black)
  + manifest.csv, and <out>/data.json with every per-image PSNR / SSIM.

Run inside the `demo` env with the GNU Radio env vars (see e2e_link.py).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pmt
from gnuradio import analog, blocks, digital, fft, gr
from gnuradio import deepjscc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "transmitter"))
sys.path.insert(0, os.path.join(ROOT, "receiver"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import e2e_link as E  # noqa: E402  (channel, constellation/OFDM constants, psnr, ssim)
from socket_conventional_tx import ConventionalEncoder, TxConventionalConfig  # noqa: E402
from socket_conventional_rx import LDPCBatchDecoder, decompress_to_bgr, _make_interleaver  # noqa: E402

FFT, CP, PKT = E.FFT, E.CP, E.PKT
OCC, PIL_C, PIL_S = E.OCC, E.PIL_C, E.PIL_S
SW1, SW2, HEADER_BOOST, SAMP_RATE = E.SW1, E.SW2, E.HEADER_BOOST, E.SAMP_RATE
BPSK = E.BPSK
W, H = 768, 512
MODS = {1: digital.constellation_bpsk, 2: digital.constellation_qpsk, 4: digital.constellation_16qam}


def tx_config(n, k, mod, codec="jpeg", comp_ratio=6):
    tcn = int((1 / comp_ratio) * 4 * 4 * 2 * 3)                      # as socket_conventional_tx
    target_symbols = (tcn * (H // 4) * (W // 4)) // 2
    return TxConventionalConfig(
        width=W, height=H, channel=3, codec=codec, codec_quality=48, target_bytes=0,
        fit_to_budget=True, ldpc_n=n, ldpc_k=k, bits_per_symbol=mod,
        target_complex_symbols=target_symbols, packet_len=PKT, repeat=1, repeat_interval=0.5,
        warmup_frames=0, warmup_interval=0.5, port="5557", bind_host="127.0.0.1", topic=b"",
        direct_zmq=False, interleave=True)


# ── TX: conventional_tx.grc ──────────────────────────────────────────────────
def gr_tx(byte_stream: np.ndarray, mod: int, sw_tx: float, post_pad: int = 0) -> np.ndarray:
    pm = MODS[mod]()
    hf = deepjscc.packet_header_ofdm_robust(OCC, n_syms=1, len_tag_key="packet_len",
        frame_len_tag_key="packet_len", num_tag_key="packet_num", bits_per_header_sym=1,
        bits_per_payload_sym=8, scramble_header=True, num_bits=16, use_fec=True)
    tb = gr.top_block()
    src = blocks.vector_source_b(byte_stream.astype(np.uint8).tolist(), False)
    rep = blocks.repack_bits_bb(8, mod, "", False, gr.GR_LSB_FIRST)
    c2s_p = digital.chunks_to_symbols_bc(pm.points(), 1)
    s2t = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, PKT, "packet_len")
    c2r, f2u = blocks.complex_to_real(1), blocks.float_to_uchar()
    hgen = digital.packet_headergenerator_bb(hf.formatter(), "packet_len")
    c2s_h = digital.chunks_to_symbols_bc(BPSK.points(), 1)
    hb = blocks.multiply_const_cc(HEADER_BOOST)
    mux = blocks.tagged_stream_mux(gr.sizeof_gr_complex, "packet_len", 0)
    alloc = digital.ofdm_carrier_allocator_cvc(FFT, OCC, PIL_C, PIL_S, (sw_tx * SW1, sw_tx * SW2), "packet_len", True)
    ifft = fft.fft_vcc(FFT, False, (), True)
    cp = digital.ofdm_cyclic_prefixer(FFT, FFT + CP, 0, "packet_len")
    sc = blocks.multiply_const_cc(0.03)
    snk = blocks.vector_sink_c()
    tb.connect(src, rep, c2s_p, s2t, c2r, f2u, hgen, c2s_h, hb, (mux, 0))
    tb.connect(s2t, (mux, 1))
    if post_pad:
        shaper = digital.burst_shaper_cc([], 0, post_pad, False, "packet_len")
        tb.connect(mux, alloc, ifft, cp, shaper, sc, snk)
    else:
        tb.connect(mux, alloc, ifft, cp, sc, snk)
    tb.run()
    return np.array(snk.data(), np.complex64)


# ── RX: conventional_rx.grc ──────────────────────────────────────────────────
def gr_rx(rx: np.ndarray, mod: int, npwr: float, sw_rx: float = 3.0, sync_threshold: float = 0.65,
          trigger_key: str = "ofdm_start"):
    """Returns [(packet_num, LLR float32[PKT*mod])] for every parsed frame."""
    pm = MODS[mod]()
    hf = deepjscc.packet_header_ofdm_robust(OCC, n_syms=1, len_tag_key="packet_len",
        frame_len_tag_key="frame_len", num_tag_key="packet_num", bits_per_header_sym=1,
        bits_per_payload_sym=8, scramble_header=True, expected_packet_len=PKT, num_bits=16, use_fec=True)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"e2e_sscc_rx_{os.getpid()}.cf32")
    rx.tofile(path)
    tb = gr.top_block()
    rs = blocks.file_source(gr.sizeof_gr_complex, path, False)
    sync = digital.ofdm_sync_sc_cfb(FFT, CP, False, sync_threshold)
    dl = blocks.delay(gr.sizeof_gr_complex, FFT + CP)
    fm = analog.frequency_modulator_fc(-2.0 / FFT)
    mul = blocks.multiply_cc()
    dmx = digital.header_payload_demux(3, FFT, CP, "frame_len", trigger_key, True, gr.sizeof_gr_complex,
                                       "rx_time", int(SAMP_RATE), (), 0)
    f0 = fft.fft_vcc(FFT, True, (), True)
    ce = digital.ofdm_chanest_vcvc(SW1 * sw_rx, SW2 * sw_rx, 1, 0, 3, False)
    heq = digital.ofdm_equalizer_simpledfe(FFT, BPSK.base(), OCC, PIL_C, PIL_S)
    e0 = digital.ofdm_frame_equalizer_vcvc(heq.base(), CP, "", True, 1)
    ser0 = digital.ofdm_serializer_vcc(FFT, OCC, "", "", 0, "", True)
    dec = digital.constellation_decoder_cb(BPSK.base())
    hp = digital.packet_headerparser_b(hf.base())
    f1 = fft.fft_vcc(FFT, True, (), True)
    peq = digital.ofdm_equalizer_simpledfe(FFT, pm.base(), OCC, PIL_C, PIL_S, 1)
    e1 = digital.ofdm_frame_equalizer_vcvc(peq.base(), CP, "frame_len", False, 0)
    ser1 = digital.ofdm_serializer_vcc(FFT, OCC, "frame_len", "packet_len", 1, "", True)
    boost = blocks.multiply_const_cc(HEADER_BOOST)
    soft = digital.constellation_soft_decoder_cf(pm.base(), npwr)
    snk = blocks.vector_sink_f()
    tb.connect(rs, sync); tb.connect((sync, 0), fm, (mul, 1)); tb.connect(rs, dl, (mul, 0))
    tb.connect(mul, (dmx, 0)); tb.connect((sync, 1), (dmx, 1))
    tb.connect((dmx, 0), f0, ce, e0, ser0, dec, hp)
    tb.msg_connect(hp, "header_data", dmx, "header_data")
    tb.connect((dmx, 1), f1, e1, ser1, boost, soft, snk)
    tb.run()
    os.remove(path)
    llr = np.array(snk.data(), np.float32)
    frames = []
    pns = sorted(((t.offset, pmt.to_long(t.value)) for t in snk.tags() if pmt.symbol_to_string(t.key) == "packet_num"))
    for off, pn in pns:                       # tag offsets are in float items (PKT*mod per frame)
        v = llr[off: off + PKT * mod]
        if len(v) == PKT * mod:
            frames.append((pn, v))
    return frames


# ── socket_conventional_rx reassembly + decode ───────────────────────────────
def reassemble(frames, pkt_per_img, mod):
    """Mirror receive_loop: anchor on the first packet_num after a flush, slot = delta %
    pkt_per_img, flush when the last slot arrives or the image index changes. The offline
    file has no wall-clock, so the idle-gap flush is replaced by the image-index change."""
    items = PKT * mod
    out, buf, seen = [], np.zeros(pkt_per_img * items, np.float32), np.zeros(pkt_per_img, bool)
    anchor, cur = None, None

    def flush():
        if seen.any():
            b = buf.copy()
            b[np.repeat(~seen, items)] = 0
            out.append((b, seen.copy()))
    for pn, v in frames:
        if anchor is None:
            anchor = pn
        delta = pn - anchor
        slot, img = delta % pkt_per_img, delta // pkt_per_img
        if cur is None:
            cur = img
        elif img != cur:
            flush(); seen[:] = False; buf[:] = 0; cur = img
        buf[slot * items:(slot + 1) * items] = v
        seen[slot] = True
        if slot == pkt_per_img - 1:
            flush(); seen[:] = False; buf[:] = 0; cur = None; anchor = None
    if cur is not None:
        flush()
    return out


def ldpc_jpeg_decode(buf, ldpc, img_total, mod, perm_fwd):
    rx_llrs = -buf[:img_total].astype(np.float32)
    rx_llrs = rx_llrs.reshape(-1, mod)[:, ::-1].flatten()
    msg = ldpc.decode_llrs(rx_llrs[perm_fwd])
    return decompress_to_bgr(msg.tobytes(), W, H), ldpc.last_failed_blocks, ldpc.last_total_blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kodak", default="/Users/marcellobullo/Downloads/kodak_dataset")
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 28])
    ap.add_argument("--ldpc", type=int, nargs=2, action="append", metavar=("N", "K"),
                    help="LDPC (n,k); repeat for several. Default: 2304 with k=1152,1728,1920")
    ap.add_argument("--mods", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--codec", default="jpeg")
    ap.add_argument("--bp-iters", type=int, default=50)
    ap.add_argument("--sw-tx", type=float, default=4.0)
    ap.add_argument("--sw-rx", type=float, default=3.0)
    ap.add_argument("--sync-thr", type=float, default=0.65)
    ap.add_argument("--trigger-key", default="ofdm_start", help="header_payload_demux trigger tag key (conventional_rx.grc uses ofdm_start)")
    ap.add_argument("--npwr-snr", type=float, default=None, help="SNR used for the soft-demapper npwr (default: the channel SNR)")
    ap.add_argument("--frame-gap", type=int, default=0, help="idle samples between OFDM frames via the Burst Shaper (TX fix)")
    ap.add_argument("--gap-s", type=float, default=0.5)
    # impairments handled by e2e_link.channel (all off by default)
    for k, t in [("cfo_hz", float), ("sfo_ppm", float), ("pn_linewidth", float), ("iq_amp_db", float),
                 ("iq_phase_deg", float), ("dc", float), ("clip", float), ("taps", int)]:
        ap.add_argument("--" + k.replace("_", "-"), type=t, default=0)
    ap.add_argument("--snr-bw", type=float, default=0.0, help="bandwidth (Hz) in which the SNR is defined; 0 = full 1 MS/s band")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="sscc")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    combos = [tuple(c) for c in a.ldpc] if a.ldpc else [(2304, 1152), (2304, 1728), (2304, 1920)]

    paths = sorted(glob.glob(f"{a.kodak}/kodim*.png"))[: a.images]
    frames_bgr = [cv2.imread(p) for p in paths]
    from PIL import Image
    origs = [cv2.resize(np.array(Image.open(p))[..., :3], (W, H)) for p in paths]   # notebook reference
    data = []
    os.makedirs(a.out, exist_ok=True)

    for (n, k) in combos:
        for mod in a.mods:
            cfg = tx_config(n, k, mod, a.codec)
            enc = ConventionalEncoder(cfg)
            L = enc.layout
            pkt_bytes = PKT * mod // 8
            pkt_per_img = L.ldpc_output_bytes // pkt_bytes
            img_total = L.ldpc_output_bytes * 8
            perm_fwd, _ = _make_interleaver(img_total, pkt_per_img)
            ldpc = LDPCBatchDecoder(n, k, a.bp_iters, "cpu")
            t0 = time.time()
            tx_bytes = [enc.encode(f) for f in frames_bgr]
            bb = gr_tx(np.concatenate(tx_bytes), mod, a.sw_tx, a.frame_gap)
            frame_samples = (2 + 1 + PKT // E.NDATA) * (FFT + CP) + a.frame_gap
            per_img = pkt_per_img * frame_samples
            if len(bb) < per_img * len(paths):
                bb = np.concatenate([bb, np.zeros(per_img * len(paths) - len(bb), np.complex64)])
            assert len(bb) == per_img * len(paths), (len(bb), per_img)
            gap = np.zeros(int(a.gap_s * SAMP_RATE), np.complex64)
            pieces, active = [gap], [np.zeros(len(gap), bool)]
            for i in range(len(paths)):
                pieces += [bb[i * per_img:(i + 1) * per_img], gap]
                active += [np.ones(per_img, bool), np.zeros(len(gap), bool)]
            x, act = np.concatenate(pieces), np.concatenate(active)
            print(f"[*] LDPC {k}/{n} mod{mod}: {len(paths)} imgs, {pkt_per_img} pkts/img, "
                  f"{L.num_ldpc_blocks} blocks, TX built in {time.time()-t0:.0f}s", flush=True)

            for snr in a.snr:
                t0 = time.time()
                rng = np.random.default_rng(a.seed + int(snr * 10))
                npwr = 10 ** (-(snr if a.npwr_snr is None else a.npwr_snr) / 10)
                y = E.channel(x, act, snr, a, rng)
                frames = gr_rx(y, mod, npwr, a.sw_rx, a.sync_thr, a.trigger_key)
                imgs = reassemble(frames, pkt_per_img, mod)
                od = os.path.join(a.out, f"snr_{int(snr)}dB", f"ldpc_n{n}_k{k}_mod{mod}")
                os.makedirs(od, exist_ok=True)
                manifest, ps, ss = [], [], []
                for idx in range(len(paths)):
                    if idx < len(imgs):
                        buf, seen = imgs[idx]
                        bgr, nfail, nblk = ldpc_jpeg_decode(buf, ldpc, img_total, mod, perm_fwd)
                        miss = int((~seen).sum())
                    else:
                        bgr, nfail, nblk, miss = None, -1, 0, pkt_per_img
                    ok = bgr is not None
                    if ok:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        p_, s_ = float(E.psnr(rgb, origs[idx])), E.ssim(rgb, origs[idx])
                        fname = f"image_{idx:03d}.png"
                        cv2.imwrite(os.path.join(od, fname), bgr)
                    else:
                        p_, s_ = 0.0, 0.0                         # decode failure -> floor, as in the notebook
                        fname = f"image_{idx:03d}_fail.png"
                        cv2.imwrite(os.path.join(od, fname), np.zeros((H, W, 3), np.uint8))
                    ps.append(p_); ss.append(s_)
                    manifest.append(dict(image_id=idx, filename=fname, slots_seen=pkt_per_img - miss,
                                         slots_missing=miss, decode_ok=int(ok), ldpc_failed_blocks=nfail,
                                         ldpc_blocks=nblk, psnr=p_, ssim=s_))
                with open(os.path.join(od, "manifest.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
                    w.writeheader(); w.writerows(manifest)
                row = dict(snr=snr, ldpc_n=n, ldpc_k=k, mod=mod, psnr=float(np.mean(ps)), psnr_std=float(np.std(ps)),
                           ssim=float(np.mean(ss)), ssim_std=float(np.std(ss)),
                           decode_ok=int(sum(m["decode_ok"] for m in manifest)), n_images=len(paths),
                           frames_rx=len(frames), frames_tx=pkt_per_img * len(paths),
                           per_image=manifest, secs=time.time() - t0)
                data.append(row)
                print(f"  SNR {snr:4.0f} dB | PSNR {row['psnr']:6.2f} | SSIM {row['ssim']:.3f} | "
                      f"ok {row['decode_ok']}/{len(paths)} | frames {len(frames)}/{row['frames_tx']} | {row['secs']:.0f}s", flush=True)
                json.dump(dict(args=vars(a), results=data), open(os.path.join(a.out, "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
