"""2x2 MIMO-OFDM payload equalizer + spatial-stream merge (self-contained).

4 inputs (FFT'd OFDM symbols), from the two RX antennas' demux outputs:
    0,1 = rx0 / rx1 preamble+header (4 OFDM symbols/frame: STF, LTF0, LTF1, HDR)
    2,3 = rx0 / rx1 payload        (frame_len OFDM symbols/frame, tagged by the demux
                                     with packet_num + frame_len on the first symbol)
For each frame it estimates the full 2x2 channel H(k) from the two time-orthogonal
LTFs, builds regularized-ZF (MMSE-like) weights W(k), and recovers the two spatial
streams per OFDM symbol. The 48 data carriers of stream A then the 48 of stream B
are emitted consecutively which -- with the TX deinterleave-by-48 split -- restores
the original 960-symbol DJSCC packet order. The merged frame is re-tagged with
packet_len = frame_len*96 (=960) and the decoded packet_num.

Frames are paired (preamble<->payload) in FIFO order. A header CRC failure (rare at
the SNRs needed for usable image quality) makes the demux drop that frame's payload
while still emitting its preamble, which would desync the pairing; restart the RX if
that happens. Keeping the channel estimate internal (no inter-block message) avoids a
Python->Python async-message crash in this GNU Radio build.
"""
import numpy as np
import pmt
from gnuradio import gr


class blk(gr.basic_block):
    def __init__(self, fft_len=64, sync_word2=None, occupied_carriers=None, reg=1e-3):
        gr.basic_block.__init__(
            self, name="mimo_payload_eq",
            in_sig=[(np.complex64, fft_len)] * 4, out_sig=[np.complex64])
        self.fft_len = int(fft_len)
        self.reg = float(reg)
        self.GROUP = 4
        ltf = np.array(sync_word2 if sync_word2 is not None else [0] * fft_len,
                       dtype=np.complex64)
        self.used = np.nonzero(ltf)[0]
        self.Lu = ltf[self.used]
        upos = {int(p): i for i, p in enumerate(self.used.tolist())}
        if occupied_carriers is None:               # default-probe path (grcc): no-op
            occ = []
        elif isinstance(occupied_carriers[0], (list, tuple)):
            occ = occupied_carriers[0]
        else:
            occ = occupied_carriers
        data_pos = [c + self.fft_len // 2 for c in occ]
        self.data_idx = np.array([upos[p] for p in data_pos], dtype=np.int64)
        self.ndata = len(self.data_idx)             # 48
        self.per_sym = 2 * self.ndata if self.ndata else 96   # 96 merged out / OFDM sym
        self.set_output_multiple(self.per_sym)
        self.pn_key = pmt.intern("packet_num")
        self.fl_key = pmt.intern("frame_len")
        self.pl_key = pmt.intern("packet_len")

    def forecast(self, noutput_items, ninputs):
        nf = max(1, noutput_items // (10 * self.per_sym))
        return [self.GROUP * nf, self.GROUP * nf, 10 * nf, 10 * nf]

    def _weights(self, h00, h01, h10, h11):
        n = len(h00)
        H = np.empty((n, 2, 2), np.complex128)
        H[:, 0, 0] = h00; H[:, 0, 1] = h01
        H[:, 1, 0] = h10; H[:, 1, 1] = h11
        Hh = np.conj(np.transpose(H, (0, 2, 1)))
        G = Hh @ H + self.reg * np.eye(2)[None]
        return np.linalg.inv(G) @ Hh

    def general_work(self, input_items, output_items):
        pre0, pre1, pay0, pay1 = input_items
        out = output_items[0]
        npre = min(len(pre0), len(pre1))
        npay = min(len(pay0), len(pay1))
        u = self.used; Lu = self.Lu
        cp = 0; cpay = 0; produced = 0
        while cp + self.GROUP <= npre:
            # frame start on payload input (port 2): need frame_len tag + enough data
            fl = None; pn = -1
            for t in self.get_tags_in_window(2, cpay, cpay + 1, self.fl_key):
                fl = pmt.to_long(t.value)
            for t in self.get_tags_in_window(2, cpay, cpay + 1, self.pn_key):
                pn = pmt.to_long(t.value)
            if fl is None or cpay + fl > npay:
                break
            if produced + fl * self.per_sym > len(out):
                break
            h00 = pre0[cp + 1][u] / Lu; h10 = pre1[cp + 1][u] / Lu   # LTF0 -> ant0
            h01 = pre0[cp + 2][u] / Lu; h11 = pre1[cp + 2][u] / Lu   # LTF1 -> ant1
            W = self._weights(h00, h01, h10, h11)
            for m in range(fl):
                y0 = pay0[cpay + m][u]; y1 = pay1[cpay + m][u]
                Y = np.stack([y0, y1], axis=1)[:, :, None]
                X = (W @ Y)[:, :, 0]
                o = produced
                out[o:o + self.ndata] = X[self.data_idx, 0].astype(np.complex64)
                out[o + self.ndata:o + self.per_sym] = X[self.data_idx, 1].astype(np.complex64)
                if m == 0:
                    w0 = self.nitems_written(0) + o
                    self.add_item_tag(0, w0, self.pl_key,
                                      pmt.from_long(fl * self.per_sym))
                    self.add_item_tag(0, w0, self.pn_key, pmt.from_long(int(pn)))
                produced += self.per_sym
            cp += self.GROUP; cpay += fl
        self.consume(0, cp); self.consume(1, cp)
        self.consume(2, cpay); self.consume(3, cpay)
        return produced
