"""MIMO header MRC combiner (antenna-0-only header, 2 RX antennas).

The header is transmitted on TX antenna 0 only (antenna 1 silent), so each RX
antenna sees a clean SISO channel h[r][0] (estimated from LTF0). This 4:1 decimator
takes the per-frame preamble+header of both RX antennas (demux header output, FFT'd:
[STF, LTF0, LTF1, HDR]) and emits one maximal-ratio-combined header OFDM symbol per
frame for the standard header-decode chain (serializer -> decoder -> headerparser).
"""
import numpy as np
from gnuradio import gr


class blk(gr.decim_block):
    def __init__(self, fft_len=64, sync_word2=None):
        gr.decim_block.__init__(
            self, name="mimo_header_mrc",
            in_sig=[(np.complex64, fft_len), (np.complex64, fft_len)],
            out_sig=[(np.complex64, fft_len)], decim=4)
        self.fft_len = int(fft_len)
        ltf = np.array(sync_word2 if sync_word2 is not None else [0] * fft_len,
                       dtype=np.complex64)
        self.used = np.nonzero(ltf)[0]
        self.Lu = ltf[self.used]

    def work(self, input_items, output_items):
        in0 = input_items[0]; in1 = input_items[1]; out = output_items[0]
        n = len(out); u = self.used; Lu = self.Lu
        for f in range(n):
            b = 4 * f
            h00 = in0[b + 1][u] / Lu        # LTF0: TX ant0 -> RX0
            h10 = in1[b + 1][u] / Lu        # LTF0: TX ant0 -> RX1
            hdr0 = in0[b + 3][u]; hdr1 = in1[b + 3][u]
            num = np.conj(h00) * hdr0 + np.conj(h10) * hdr1
            den = np.abs(h00) ** 2 + np.abs(h10) ** 2 + 1e-12
            vec = np.zeros(self.fft_len, dtype=np.complex64)
            vec[u] = (num / den).astype(np.complex64)
            out[f] = vec
        return n
