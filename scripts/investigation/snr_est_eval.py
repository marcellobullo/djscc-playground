#!/usr/bin/env python
"""Accuracy test of the receiver's live SNR estimation.

Runs the REAL embedded logger blocks of djscc_rx.grc (raw_noise_logger, pilot_snr_logger, chan_taps_logger, from
receiver/gnu_radio/djscc_rx_*_logger_0.py) inside the offline replica RX, collects their ZMQ messages exactly like
receiver/socket_rx.py does, and compares the resulting per-packet SNR with the TRUE post-equalization payload SNR
(known transmitted latents). Latents are random Gaussian with power 2 (the codec's P=1 per real value).

  python scripts/investigation/snr_est_eval.py --snr 4 8 12 16 20 24 28 --frames 300 [--gr-gap 16]
"""
import argparse, importlib.util, json, os, sys, threading, time
import numpy as np, pmt, zmq
from gnuradio import analog, blocks, digital, fft, gr
from gnuradio import deepjscc

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e2e_link as E
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from jscc.snr_estimation import per_packet_snr_db, pilot_stats, pooled_snr_eff_db, snr_eff_db   # the code socket_rx.py runs
ROOT = E.ROOT
FFT, CP, PKT = E.FFT, E.CP, E.PKT
DATA_IDX = np.array([k + 32 for k in (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27))) if k not in (-21, -7, 7, 21)])
PILOT_IDX = np.array([11, 25, 39, 53]); PILOT_SYM = np.array([1, 1, 1, -1], dtype=complex)
NULL_IDX = [0, 1, 2, 3, 4, 5, 59, 60, 61, 62, 63]


def load_blk(name):
    p = os.path.join(ROOT, "receiver", "gnu_radio", f"djscc_rx_{name}_0.py")
    spec = importlib.util.spec_from_file_location(f"djscc_rx_{name}_0", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.blk


def run_rx(rx, addr, sw_rx=3.0, sync_thr=0.65):
    RawL, PilL, TapL = load_blk("raw_noise_logger"), load_blk("pilot_snr_logger"), load_blk("chan_taps_logger")
    hf = deepjscc.packet_header_ofdm_robust(E.OCC, n_syms=1, len_tag_key="packet_len", frame_len_tag_key="frame_len",
        num_tag_key="packet_num", bits_per_header_sym=1, bits_per_payload_sym=8, scramble_header=True,
        expected_packet_len=PKT, num_bits=16, use_fec=True)
    path = f"/tmp/snr_eval_rx_{os.getpid()}.cf32"; rx.tofile(path)
    tb = gr.top_block()
    rs = blocks.file_source(gr.sizeof_gr_complex, path, False)
    sync = digital.ofdm_sync_sc_cfb(FFT, CP, False, sync_thr)
    dl = blocks.delay(gr.sizeof_gr_complex, FFT + CP); fm = analog.frequency_modulator_fc(-2.0 / FFT); mul = blocks.multiply_cc()
    dmx = digital.header_payload_demux(3, FFT, CP, "frame_len", "", True, gr.sizeof_gr_complex, "rx_time", int(E.SAMP_RATE), (), 0)
    f0 = fft.fft_vcc(FFT, True, (), True)
    ce = digital.ofdm_chanest_vcvc(sw_rx * E.SW1, sw_rx * E.SW2, 1, int(FFT / 4), 3, False)
    heq = digital.ofdm_equalizer_simpledfe(FFT, E.BPSK.base(), E.OCC, E.PIL_C, E.PIL_S)
    e0 = digital.ofdm_frame_equalizer_vcvc(heq.base(), CP, "frame_len", True, 1)
    ser0 = digital.ofdm_serializer_vcc(FFT, E.OCC, "frame_len"); dec = digital.constellation_decoder_cb(E.BPSK.base())
    hp = digital.packet_headerparser_b(hf.base())
    f1 = fft.fft_vcc(FFT, True, (), True)
    peq = digital.ofdm_equalizer_static(FFT, E.OCC, E.PIL_C, E.PIL_S)
    e1 = digital.ofdm_frame_equalizer_vcvc(peq.base(), CP, "frame_len", True, 0)
    vs = blocks.vector_sink_c(FFT); vraw = blocks.vector_sink_c(FFT)
    raw_l = RawL(fft_len=FFT, null_positions=NULL_IDX, out_path="/tmp/_raw.npz", snr_addr=addr)
    pil_l = PilL(fft_len=FFT, pilot_positions=[11, 25, 39, 53], pilot_symbols=[1.0, 1.0, 1.0, -1.0], out_path="/tmp/_pil.npz", snr_addr=addr)
    tap_l = TapL(fft_len=FFT, packet_len=PKT, out_path="/tmp/_h.npz", snr_addr=addr)
    tb.connect(rs, sync); tb.connect((sync, 0), fm, (mul, 1)); tb.connect(rs, dl, (mul, 0)); tb.connect(mul, (dmx, 0)); tb.connect((sync, 1), (dmx, 1))
    tb.connect((dmx, 0), f0, ce, tap_l, e0, ser0, dec, hp)
    tb.msg_connect(hp, "header_data", dmx, "header_data"); tb.msg_connect(hp, "header_data", tap_l, "header_valid")
    tb.connect((dmx, 1), f1, e1, vs); tb.connect(e1, pil_l); tb.connect(f1, raw_l); tb.connect(f1, vraw)
    tb.run(); os.remove(path)
    data = np.array(vs.data(), np.complex64).reshape(-1, FFT); rawd = np.array(vraw.data(), np.complex64).reshape(-1, FFT)
    starts = {}
    for t in sorted(vs.tags(), key=lambda t: t.offset):
        k = pmt.symbol_to_string(t.key)
        if k == "packet_num": starts.setdefault(t.offset, {})["pn"] = pmt.to_long(t.value)
        elif k == "frame_len": starts.setdefault(t.offset, {})["len"] = pmt.to_long(t.value)
    return {d["pn"]: (data[o:o + d["len"]], rawd[o:o + d["len"]]) for o, d in starts.items() if "pn" in d and "len" in d}


def est_pilot_diff(fr):
    """Candidate A: in-band noise from consecutive-symbol pilot differences (after removing the residual-CFO phase
    ramp), signal power from the equalized data carriers. No transmit-power / channel-scale calibration needed.
    Returns (sigma2_eq, p_data) -> SNR = (p_data - sigma2) / sigma2."""
    P = fr[:, PILOT_IDX]
    slope = np.angle(np.sum(P[1:] * np.conj(P[:-1])))
    Pd = P * np.exp(-1j * slope * np.arange(len(P)))[:, None]
    d = Pd[1:] - Pd[:-1]
    return float(np.mean(np.abs(d) ** 2) / 2), float(np.mean(np.abs(fr[:, DATA_IDX]) ** 2))


def est_null_raw(raw):
    """Candidate B: noise from the guard bins and signal+noise power from the data bins, both in the raw FFT domain
    (white noise, no equalizer or channel-estimate involved). Valid for a flat channel."""
    return float(np.mean(np.abs(raw[:, NULL_IDX]) ** 2)), float(np.mean(np.abs(raw[:, DATA_IDX]) ** 2))


def est_pilots_principled(fr, ps_tx=2.0, c_chest=0.5, p_pilot_tx=1.0, remove_ramp_loss=True):
    """Candidate E (proposed): everything from the equalized pilots, no assumption on the equalizer/channel-estimate scale.
      - residual-CFO phase ramp slope from consecutive-symbol pilot products, removed before measuring
      - pilot tone amplitude A = |mean over symbols|, noise sigma2 = var of consecutive differences / 2
      - SNR at the pilot tones = (A^2 - sigma2/T) / sigma2 ; data carriers carry ps_tx/p_pilot_tx times the pilot power
        -> channel SNR on the data carriers (what the receiver front-end really delivers)
      - decoder-side SNR = channel SNR degraded by the channel-estimate noise (c_chest, design constant of the sync words)
        and, if the ramp is not corrected downstream, by the phase-ramp error power  s^2 (T^2-1)/12.
    Returns (snr_channel_dB, snr_eff_dB)."""
    P = fr[:, PILOT_IDX] * PILOT_SYM.conj()[None, :]
    T = len(P)
    slope = np.angle(np.sum(P[1:] * np.conj(P[:-1])))
    Pd = P * np.exp(-1j * slope * np.arange(T))[:, None]
    s2 = float(np.mean(np.abs(Pd[1:] - Pd[:-1]) ** 2) / 2)
    A2 = float(np.mean(np.abs(Pd.mean(axis=0)) ** 2)) - s2 / T
    snr_pilot = max(A2, 1e-12) / s2
    snr_chan = snr_pilot * ps_tx / p_pilot_tx
    ramp = slope ** 2 * (T * T - 1) / 12.0 if remove_ramp_loss else 0.0
    snr_eff = 1.0 / ((1.0 + c_chest) / snr_chan + ramp)
    return 10 * np.log10(snr_chan), 10 * np.log10(snr_eff)


def snr_from(s2, p):
    return float(10 * np.log10(max(p - s2, 1e-12) / s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 4, 8, 12, 16, 20, 24, 28])
    ap.add_argument("--frames", type=int, default=300); ap.add_argument("--gr-gap", type=int, default=2)
    ap.add_argument("--cfo-hz", type=float, default=0.0); ap.add_argument("--cpe", action="store_true", help="truth = what the decoder sees with the CFO-ramp corrector (fixed chain)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--port", type=int, default=5571)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gr-channel", "snr_estimation"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    sym = ((rng.standard_normal(PKT * a.frames) + 1j * rng.standard_normal(PKT * a.frames))).astype(np.complex64)   # power 2
    bb = E.gr_tx(sym, 4.0, a.gr_gap)
    fs = (2 + 1 + PKT // E.NDATA) * (FFT + CP) + a.gr_gap
    bb = np.concatenate([bb, np.zeros(max(0, fs * a.frames - len(bb)), np.complex64)])[: fs * a.frames]
    gap = np.zeros(int(0.5 * E.SAMP_RATE), np.complex64)
    x = np.concatenate([gap, bb, gap]); act = np.concatenate([np.zeros(len(gap), bool), np.ones(len(bb), bool), np.zeros(len(gap), bool)])
    ns = argparse.Namespace(clip=0, taps=0, sfo_ppm=0, cfo_hz=a.cfo_hz, pn_linewidth=0, iq_amp_db=0, iq_phase_deg=0, dc=0, snr_bw=0)
    port = a.port; ctx = zmq.Context.instance(); results = []
    for snr in a.snr:
        sock = ctx.socket(zmq.PULL); sock.setsockopt(zmq.RCVHWM, 100000); sock.bind(f"tcp://127.0.0.1:{port}")
        msgs, stop = [], threading.Event()
        def drain():
            while not stop.is_set():
                try: msgs.append(sock.recv(flags=zmq.NOBLOCK))
                except zmq.Again: time.sleep(0.002)
        th = threading.Thread(target=drain, daemon=True); th.start()
        y = E.channel(x, act, snr, ns, np.random.default_rng(a.seed + int(snr * 10)))
        frames = run_rx(y, f"tcp://127.0.0.1:{port}")
        time.sleep(0.5); stop.set(); th.join(); 
        while True:
            try: msgs.append(sock.recv(flags=zmq.NOBLOCK))
            except zmq.Again: break
        sock.setsockopt(zmq.LINGER, 0); sock.close()
        h, raw, pil = {}, {}, {}
        for m in msgs:
            sp = pmt.deserialize_str(m); meta = pmt.car(sp)
            pn = pmt.to_long(pmt.dict_ref(meta, pmt.intern("packet_num"), pmt.from_long(-1)))
            kind = pmt.symbol_to_string(pmt.dict_ref(meta, pmt.intern("kind"), pmt.intern("?")))
            if kind == "h": h[pn] = np.array(pmt.c32vector_elements(pmt.cdr(sp)), np.complex64)
            elif kind == "raw": raw[pn] = pmt.to_double(pmt.dict_ref(meta, pmt.intern("sigma2"), pmt.from_double(np.nan)))
            elif kind == "pilots":
                g_ = lambda k: pmt.to_double(pmt.dict_ref(meta, pmt.intern(k), pmt.from_double(np.nan)))
                pil[pn] = (g_("slope"), g_("sigma2"), g_("amp2"), pmt.to_long(pmt.dict_ref(meta, pmt.intern("n_sym"), pmt.from_long(0))))
        rows = []
        agg = dict(A=[], B=[], C=[], T=[]); agg_new = []; proto = []
        for pn, (fr, rawfr) in frames.items():
            if not (0 <= pn < a.frames) or len(fr) < 2: continue
            s = sym[pn * PKT:(pn + 1) * PKT]; r = E.serialize((fr, rawfr), a.cpe)
            g = np.vdot(s, r) / np.vdot(s, s); e = r - g * s
            true = 10 * np.log10(abs(g) ** 2 * np.mean(abs(s) ** 2) / np.mean(abs(e) ** 2))
            row = dict(pn=int(pn), true=float(true), gain=float(abs(g)))
            if pn in h and pn in raw:
                npw = float(np.mean(raw[pn] / np.maximum(np.abs(h[pn][DATA_IDX]) ** 2, 1e-12)))
                row["socket_rx"] = float(10 * np.log10(1.0 / npw))                           # formula used by socket_rx.py
            if pn in pil: row["logger"] = [float(v) for v in pil[pn]]                          # (slope, sigma2, amp2, n_sym)
            if pn in pil and pil[pn][1] > 0:                                                  # NEW: logger stats -> jscc.snr_estimation
                row["new"] = snr_eff_db(pil[pn][1], pil[pn][2], pil[pn][0], pil[pn][3], ramp_corrected=a.cpe)
                agg_new.append(pil[pn])
            row["has_h"] = pn in h; row["has_raw"] = pn in raw
            sa, pa = est_pilot_diff(fr); sb, pb = est_null_raw(rawfr)
            row["A_pilot_diff"] = snr_from(sa, pa); row["B_null_raw"] = snr_from(sb, pb)
            agg["A"].append((sa, pa)); agg["B"].append((sb, pb)); agg["T"].append((np.mean(abs(e) ** 2), abs(g) ** 2 * np.mean(abs(s) ** 2)))
            # ground-truth in-band noise in the RAW FFT domain: remove the known signal with a rank-1 model G_t * H_k
            X = s.reshape(-1, E.NDATA); Y = rawfr[:, E.DATA_BINS][: len(X)]; Hk = np.ones(E.NDATA, complex)
            for _ in range(4):
                Gt = np.sum(Y * np.conj(Hk[None] * X), axis=1) / np.sum(np.abs(Hk[None] * X) ** 2, axis=1)
                Hk = np.sum(Y * np.conj(Gt[:, None] * X), axis=0) / np.sum(np.abs(Gt[:, None] * X) ** 2, axis=0)
            row["sig2_data_raw"] = float(np.mean(np.abs(Y - Gt[:, None] * Hk[None] * X) ** 2) * X.size / (X.size - len(Gt) - len(Hk)))
            row["snr_raw_true"] = float(10 * np.log10(np.mean(np.abs(Gt[:, None] * Hk[None] * X) ** 2) / row["sig2_data_raw"]))
            row.update(sa=sa, pa=pa, sb=sb, pb=pb, mse=float(np.mean(abs(e) ** 2)), g2=float(abs(g) ** 2), ps=float(np.mean(abs(s) ** 2)),
                       phase_slope=float(np.angle(np.sum(fr[1:, PILOT_IDX] * np.conj(fr[:-1, PILOT_IDX])))), pilot_amp=float(np.mean(np.abs(fr[:, PILOT_IDX]))),
                       h2=float(np.mean(np.abs(h[pn][DATA_IDX]) ** 2)) if pn in h else None, sraw=raw.get(pn))
            if pn in h and pn in raw:   # Candidate C: existing noise estimate (raw nulls / |H|^2) + MEASURED equalized data power
                npw = float(np.mean(raw[pn] / np.maximum(np.abs(h[pn][DATA_IDX]) ** 2, 1e-12)))
                row["C_nulls_taps_datapower"] = snr_from(npw, pa); agg["C"].append((npw, pa))
                T = len(fr); sl = np.angle(np.sum(fr[1:, PILOT_IDX] * np.conj(fr[:-1, PILOT_IDX])))   # residual-CFO ramp, rad/symbol
                ramp = pa * sl ** 2 * (T * T - 1) / 12.0                                             # error power left after the LS common phase
                row["D_plus_ramp"] = snr_from(npw + ramp, pa)
            ec_, ee_ = est_pilots_principled(fr, remove_ramp_loss=not a.cpe)
            row["E_chan"] = float(ec_); row["E_eff"] = float(ee_)
            st = pilot_stats(fr[:, PILOT_IDX], PILOT_SYM)
            if pn in h:
                hd = h[pn][E.DATA_BINS]
                al = np.vdot(hd, Hk) / np.vdot(hd, hd)
                row["v_true"] = float(np.mean(np.abs(Hk - al * hd) ** 2) / np.mean(np.abs(al * hd) ** 2))   # real chan-est error of the taps
                Pd = fr[:, PILOT_IDX] * np.conj(PILOT_SYM)[None]
                Pd = Pd * np.exp(-1j * st[0] * np.arange(len(Pd)))[:, None]
                q = Pd.mean(0) / h[pn][PILOT_IDX]                      # pilots divided by the receiver's own estimate
                spread = np.mean(np.abs(q - q.mean()) ** 2) * 4 / 3 - np.mean(st[1] / len(Pd) / np.abs(h[pn][PILOT_IDX]) ** 2)
                row["v_pil"] = float(max(spread, 0) / abs(q.mean()) ** 2)
                snr_ch = max(st[2], 1e-12) / st[1] * 2.0
                row["pred_true_v"] = float(10 * np.log10(1 / (1 / snr_ch + row["v_true"])))
                proto.append((st[1], st[2], row["v_pil"], row["v_true"]))
            rows.append(row)
        def stat(k):
            v = np.array([(r[k] - r["true"]) for r in rows if k in r]); return (float(v.mean()), float(v.std()), len(v)) if len(v) else (np.nan, np.nan, 0)
        sr, pl = stat("socket_rx"), stat("new"); eE = stat("E_eff"); ea, eb, ec, ed = stat("A_pilot_diff"), stat("B_null_raw"), stat("C_nulls_taps_datapower"), stat("D_plus_ramp")
        # image-level (all packets pooled in the linear domain, what the scalar-CSI decoder gets)
        ia = snr_from(*np.mean(agg["A"], axis=0)); ib = snr_from(*np.mean(agg["B"], axis=0))
        inew = pooled_snr_eff_db(agg_new, ramp_corrected=a.cpe) if agg_new else float("nan"); ic = snr_from(*np.mean(agg["C"], axis=0)); it = float(10 * np.log10(np.mean([b for _, b in agg["T"]]) / np.mean([a_ for a_, _ in agg["T"]])))
        results.append(dict(snr=snr, frames_tx=a.frames, n_frames=len(frames), n_h=len(h), n_raw=len(raw), n_pil=len(pil),
                            true_mean=float(np.mean([r["true"] for r in rows])), socket_rx=sr, pilot_logger=pl, A=ea, B=eb, C=ec, D=ed, E=eE, image_level=dict(true=it, A=ia, B=ib, C=ic, new=inew), rows=rows))
        s2m, a2m, vpm, vtm = [np.mean([x[k] for x in proto]) for k in range(4)]
        snr_ch = a2m / s2m * 2.0
        pr = lambda v: 10 * np.log10(1 / (1 / snr_ch + v))
        print(f"   channel SNR {10*np.log10(snr_ch):6.2f} | chan-est error: true {vtm:.2e} (= c {vtm*snr_ch:.2f}), from pilots/taps {vpm:.2e} (= c {vpm*snr_ch:.2f}) | "
              f"image-level: with true v {pr(vtm):6.2f} (err {pr(vtm)-it:+.2f}), with pilot v {pr(vpm):6.2f} (err {pr(vpm)-it:+.2f})")
        # code path of socket_rx.py: smoothed per-packet CSI and erasure-aware scalar CSI
        by_slot = [pil.get(i) for i in range(a.frames)]
        sm = per_packet_snr_db(by_slot, window=5, ramp_corrected=a.cpe)
        dsm = np.array([sm[r["pn"]] - r["true"] for r in rows if np.isfinite(sm[r["pn"]])])
        good = [r for r in rows if "logger" in r and r["logger"][1] > 0]
        p_lost = 1 - len(good) / a.frames
        sig_ = np.mean([r["g2"] * r["ps"] for r in good]); mse_ = np.mean([r["mse"] for r in good])
        t_er = 10 * np.log10(sig_ / ((len(good) * mse_ + (a.frames - len(good)) * sig_) / a.frames))
        e_er = pooled_snr_eff_db(agg_new, p_lost, ramp_corrected=a.cpe)
        results[-1].update(smoothed=(float(dsm.mean()), float(dsm.std())), erasure=dict(p=p_lost, truth=t_er, est=e_er))
        print(f"   socket_rx path: per-packet (smoothed w=5) bias/std {dsm.mean():+5.2f}/{dsm.std():.2f} | lost {100*p_lost:4.1f}% -> "
              f"image-level incl. erasures: truth {t_er:6.2f}  est {e_er:6.2f}  err {e_er - t_er:+.2f}")
        print(f"nominal {snr:4.0f} dB | pkts {len(frames):3d} | true post-EQ SNR {results[-1]['true_mean']:6.2f} (image-level {it:6.2f}) | "
              f"per-packet est-true bias/std:  OLD socket_rx {sr[0]:+5.2f}/{sr[1]:.2f}   NEW {pl[0]:+5.2f}/{pl[1]:.2f} | image-level NEW {inew:6.2f} (err {inew - it:+.2f})", flush=True)
    for r in results:
        rr = [x for x in r["rows"] if "E_chan" in x and "snr_raw_true" in x]
        if rr: r["E_chan_vs_channel_snr"] = (float(np.mean([x["E_chan"] - x["snr_raw_true"] for x in rr])), float(np.std([x["E_chan"] - x["snr_raw_true"] for x in rr])))
    json.dump(dict(args=vars(a), results=results), open(os.path.join(a.out, f"snr_est_gap{a.gr_gap}_cfo{a.cfo_hz:g}{'_cpe' if a.cpe else ''}_seed{a.seed}.json"), "w"))


if __name__ == "__main__":
    main()
