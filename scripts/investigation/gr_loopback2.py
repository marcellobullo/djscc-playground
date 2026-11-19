"""Baseband loopback of the deployed DJSCC OFDM chain (djscc_tx.grc -> djscc_rx.grc,
enabled blocks only, USRPs replaced by an AWGN+CFO channel). Measures what the
decoder actually receives on its ZMQ input relative to the TX latent symbols."""
import numpy as np, pmt, yaml
from gnuradio import gr, blocks, digital, fft, analog
from gnuradio import deepjscc

G = "/Users/marcellobullo/research/FINAL/github/djscc-playground"
_v = {b["name"]: b["parameters"]["value"] for b in
      yaml.safe_load(open(f"{G}/transmitter/gnu_radio/djscc_tx.grc"))["blocks"] if b["id"] == "variable"}
SW1 = np.array(eval(_v["sync_word1"].replace("sw_multiplier*", ""), {"np": np}), dtype=float)
SW2 = np.array(eval(_v["sync_word2"].replace("sw_multiplier*", ""), {"np": np}), dtype=float)
fft_len, cp_len, PKT = 64, 16, 960
occ = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7))
       + list(range(8, 21)) + list(range(22, 27)),)
pil_c, pil_s = ((-21, -7, 7, 21),), ((1, 1, 1, -1),)
bpsk = digital.constellation_bpsk()


def run(snr_db, sw_tx=4, sw_rx=3, rx_boost=2.0, cfo_hz=0.0, sfo_ppm=0.0, samp_rate=1e6,
        n_frames=30, payload_eq="static", seed=0, power=2.0):
    rng = np.random.default_rng(seed)
    pay = np.sqrt(power / 2) * (rng.standard_normal((n_frames, PKT)) + 1j * rng.standard_normal((n_frames, PKT)))
    stream = pay.reshape(-1).astype(np.complex64)
    hf_tx = deepjscc.packet_header_ofdm_robust(occ, n_syms=1, len_tag_key="packet_len",
        frame_len_tag_key="packet_len", num_tag_key="packet_num", bits_per_header_sym=1,
        bits_per_payload_sym=8, scramble_header=True, num_bits=16, use_fec=True)
    tb = gr.top_block()
    src = blocks.vector_source_c(stream.tolist(), False)
    s2t = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, PKT, "packet_len")
    c2r = blocks.complex_to_real(1); f2u = blocks.float_to_uchar()
    hgen = digital.packet_headergenerator_bb(hf_tx.formatter(), "packet_len")
    c2s = digital.chunks_to_symbols_bc(bpsk.points(), 1)
    hb = blocks.multiply_const_cc(2.0)
    mux = blocks.tagged_stream_mux(gr.sizeof_gr_complex, "packet_len", 0)
    alloc = digital.ofdm_carrier_allocator_cvc(fft_len, occ, pil_c, pil_s, (sw_tx * SW1, sw_tx * SW2), "packet_len", True)
    ifft = fft.fft_vcc(fft_len, False, (), True)
    cp = digital.ofdm_cyclic_prefixer(fft_len, fft_len + cp_len, 0, "packet_len")
    sc = blocks.multiply_const_cc(0.03)
    snk = blocks.vector_sink_c()
    tb.connect(src, s2t, c2r, f2u, hgen, c2s, hb, (mux, 0))
    tb.connect(s2t, (mux, 1))
    tb.connect(mux, alloc, ifft, cp, sc, snk)
    tb.run()
    tx = np.array(snk.data(), np.complex64)
    fr = (2 + 1 + PKT // 48) * (fft_len + cp_len)
    gap = np.zeros(300, np.complex64)
    txg = np.concatenate([np.concatenate([gap, tx[i*fr:(i+1)*fr]]) for i in range(len(tx)//fr)] + [gap] * 3)
    p = np.mean(np.abs(tx) ** 2)
    n = np.arange(len(txg))
    rx = txg * np.exp(2j * np.pi * cfo_hz / samp_rate * n)
    if sfo_ppm:
        t_new = n * (1 + sfo_ppm * 1e-6)
        rx = np.interp(t_new, n, rx.real) + 1j * np.interp(t_new, n, rx.imag)
    rx = rx + np.sqrt(p / 10 ** (snr_db / 10) / 2) * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    rx = rx.astype(np.complex64)

    hf_rx = deepjscc.packet_header_ofdm_robust(occ, n_syms=1, len_tag_key="packet_len",
        frame_len_tag_key="frame_len", num_tag_key="packet_num", bits_per_header_sym=1,
        bits_per_payload_sym=8, scramble_header=True, expected_packet_len=PKT, num_bits=16, use_fec=True)
    tb = gr.top_block()
    rs = blocks.vector_source_c(rx.tolist(), False)
    sync = digital.ofdm_sync_sc_cfb(fft_len, cp_len, False, 0.65)
    dl = blocks.delay(gr.sizeof_gr_complex, fft_len + cp_len)
    fm = analog.frequency_modulator_fc(-2.0 / fft_len)
    mul = blocks.multiply_cc()
    dmx = digital.header_payload_demux(3, fft_len, cp_len, "frame_len", "", True, gr.sizeof_gr_complex, "rx_time", int(samp_rate), (), 0)
    f0 = fft.fft_vcc(fft_len, True, (), True)
    ce = digital.ofdm_chanest_vcvc(sw_rx * SW1, sw_rx * SW2, 1, 0, 3, False)
    heq = digital.ofdm_equalizer_simpledfe(fft_len, bpsk.base(), occ, pil_c, pil_s)
    e0 = digital.ofdm_frame_equalizer_vcvc(heq.base(), cp_len, "frame_len", True, 1)
    ser0 = digital.ofdm_serializer_vcc(fft_len, occ, "frame_len")
    dec = digital.constellation_decoder_cb(bpsk.base())
    hp = digital.packet_headerparser_b(hf_rx.base())
    f1 = fft.fft_vcc(fft_len, True, (), True)
    peq = (digital.ofdm_equalizer_static(fft_len, occ, pil_c, pil_s) if payload_eq == "static"
           else digital.ofdm_equalizer_simpledfe(fft_len, digital.constellation_qpsk().base(), occ, pil_c, pil_s, 1))
    e1 = digital.ofdm_frame_equalizer_vcvc(peq.base(), cp_len, "frame_len", True, 0)
    ser1 = digital.ofdm_serializer_vcc(fft_len, occ, "frame_len", "packet_len", 1)
    boost = blocks.multiply_const_cc(rx_boost)
    ps = blocks.vector_sink_c()
    tb.connect(rs, sync); tb.connect((sync, 0), fm, (mul, 1)); tb.connect(rs, dl, (mul, 0))
    tb.connect(mul, (dmx, 0)); tb.connect((sync, 1), (dmx, 1))
    tb.connect((dmx, 0), f0, ce, e0, ser0, dec, hp)
    tb.msg_connect(hp, "header_data", dmx, "header_data")
    tb.connect((dmx, 1), f1, e1, ser1, boost, ps)
    tb.run()
    r = np.array(ps.data(), np.complex64)
    nf = len(r) // PKT
    r = r[:nf*PKT].reshape(nf, PKT)
    res = []
    for x in r:
        i = int(np.argmax(np.abs(pay.conj() @ x))); s = pay[i]
        g = np.vdot(s, x) / np.vdot(s, s)
        ph = np.unwrap([np.angle(np.vdot(s[j*48:(j+1)*48], x[j*48:(j+1)*48])) for j in range(PKT//48)])
        gs = [abs(np.vdot(s[j*48:(j+1)*48], x[j*48:(j+1)*48]) / np.vdot(s[j*48:(j+1)*48], s[j*48:(j+1)*48])) for j in range(PKT//48)]
        e = x - g * s
        res.append(dict(gain=abs(g), corr=abs(np.vdot(s, x)) / np.linalg.norm(s) / np.linalg.norm(x),
                        snr_eff_after_LS_gain=10*np.log10(np.mean(abs(g*s)**2) / np.mean(abs(e)**2)),
                        sdr_vs_clean=10*np.log10(np.mean(abs(s)**2) / np.mean(abs(x - s)**2)),
                        drift_deg=np.degrees(ph[-1] - ph[0]), gain_first=gs[0], gain_last=gs[-1]))
    return nf, {k: round(float(np.median([q[k] for q in res])), 3) for k in res[0]}


if __name__ == "__main__":
    import sys
    cases = [dict(snr_db=40, sw_rx=4, rx_boost=1.0), dict(snr_db=40), dict(snr_db=30), dict(snr_db=20), dict(snr_db=10),
             dict(snr_db=40, cfo_hz=100), dict(snr_db=40, sfo_ppm=2), dict(snr_db=40, sfo_ppm=10),
             dict(snr_db=30, payload_eq="dfe")]
    for c in cases:
        print(c, run(**c), flush=True)
