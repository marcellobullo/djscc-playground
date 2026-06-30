"""2x2 MIMO-OFDM payload equalizer + spatial-stream merge (eager-buffered, msg-gated).

Inputs:
  0,1 = rx0/rx1 preamble+header (demux header output, FFT'd): 4 OFDM symbols/frame
        [STF, LTF0, LTF1, HDR], emitted for EVERY detected frame.
  2,3 = rx0/rx1 payload (demux payload output, FFT'd): frame_len symbols, emitted only
        for frames whose header passed CRC.
Message input 'hdr' = the header parser's per-frame result (C++->Python, safe): a dict
  (valid -> packet_num/frame_len) or #f (invalid).

CRITICAL: the preamble inputs (0,1) are the SAME stream that feeds mimo_header_mrc (whose
output, decoded, produces these messages). So this block must NOT hold that shared
upstream buffer waiting for messages -- doing so starves the header chain and deadlocks.
Instead it consumes preamble groups EAGERLY into an internal queue every work() call,
then pairs them 1:1 with the header messages (1 message per detected frame): #f frames are
dropped, valid frames are equalized (2x2 ZF from the two orthogonal LTFs), merged A||B ->
960 symbols, and tagged packet_len/packet_num.
"""
import collections
import numpy as np
import pmt
from gnuradio import gr


class blk(gr.basic_block):
    def __init__(self, fft_len=64, sync_word2=None, occupied_carriers=None, reg=1e-3,
                 max_frame_len=40):
        gr.basic_block.__init__(
            self, name="mimo_payload_eq",
            in_sig=[(np.complex64, fft_len)] * 4, out_sig=[np.complex64])
        self.fft_len = int(fft_len)
        self.reg = float(reg)
        self.GROUP = 4
        self.max_fl = int(max_frame_len)
        ltf = np.array(sync_word2 if sync_word2 is not None else [0] * fft_len,
                       dtype=np.complex64)
        self.used = np.nonzero(ltf)[0]
        self.Lu = ltf[self.used]
        upos = {int(p): i for i, p in enumerate(self.used.tolist())}
        if occupied_carriers is None:
            occ = []
        elif isinstance(occupied_carriers[0], (list, tuple)):
            occ = occupied_carriers[0]
        else:
            occ = occupied_carriers
        data_pos = [c + self.fft_len // 2 for c in occ]
        self.data_idx = np.array([upos[p] for p in data_pos], dtype=np.int64)
        self.ndata = len(self.data_idx)
        self.per_sym = 2 * self.ndata if self.ndata else 96
        self.set_output_multiple(self.per_sym)
        self.pn_key = pmt.intern("packet_num")
        self.fl_key = pmt.intern("frame_len")
        self.pl_key = pmt.intern("packet_len")
        self.msgq = collections.deque()      # (valid, pn, fl) per detected frame
        self.pre_buf = collections.deque()   # (rx0_group, rx1_group) per detected frame
        self.pend = None                     # (W, pn, fl) valid frame awaiting payload
        self._dbg = 0
        self.hdr_port = pmt.intern("hdr")
        self.message_port_register_in(self.hdr_port)
        self.set_msg_handler(self.hdr_port, self._on_hdr)

    def _on_hdr(self, msg):
        try:
            if pmt.is_dict(msg):
                pn = pmt.to_long(pmt.dict_ref(msg, self.pn_key, pmt.from_long(-1)))
                fl = pmt.to_long(pmt.dict_ref(msg, self.fl_key, pmt.from_long(-1)))
                if 0 < fl <= self.max_fl:
                    self.msgq.append((True, int(pn), int(fl)))
                    return
        except Exception:
            pass
        self.msgq.append((False, -1, 0))

    def forecast(self, noutput_items, ninputs):
        # require only preamble; payload is gated internally (never block on it)
        return [self.GROUP, self.GROUP, 0, 0][:ninputs]

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
        u = self.used; Lu = self.Lu

        # 1) EAGERLY buffer whole preamble groups -> frees the shared upstream buffer
        npre = min(len(pre0), len(pre1))
        ng = npre // self.GROUP
        if ng:
            for g in range(ng):
                b = g * self.GROUP
                self.pre_buf.append((pre0[b:b + self.GROUP].copy(),
                                     pre1[b:b + self.GROUP].copy()))
            self.consume(0, ng * self.GROUP)
            self.consume(1, ng * self.GROUP)

        # 2) pair messages<->preamble, equalize valid payloads
        npay = min(len(pay0), len(pay1))
        cpay = 0; produced = 0
        while True:
            if self.pend is None:
                got = False
                while self.msgq and self.pre_buf:
                    valid, pn, fl = self.msgq.popleft()
                    p0, p1 = self.pre_buf.popleft()
                    if not valid:
                        continue
                    h00 = p0[1][u] / Lu; h10 = p1[1][u] / Lu
                    h01 = p0[2][u] / Lu; h11 = p1[2][u] / Lu
                    self.pend = (self._weights(h00, h01, h10, h11), pn, fl)
                    got = True
                    break
                if not got:
                    break
            W, pn, fl = self.pend
            if cpay + fl > npay or produced + fl * self.per_sym > len(out):
                break
            for m in range(fl):
                y0 = pay0[cpay + m][u]; y1 = pay1[cpay + m][u]
                Y = np.stack([y0, y1], axis=1)[:, :, None]
                X = (W @ Y)[:, :, 0]
                o = produced
                out[o:o + self.ndata] = X[self.data_idx, 0].astype(np.complex64)
                out[o + self.ndata:o + self.per_sym] = X[self.data_idx, 1].astype(np.complex64)
                if m == 0:
                    w0 = self.nitems_written(0) + o
                    self.add_item_tag(0, w0, self.pl_key, pmt.from_long(fl * self.per_sym))
                    self.add_item_tag(0, w0, self.pn_key, pmt.from_long(int(pn)))
                    if self._dbg < 14:
                        print("[peq] OUT pn=%d fl=%d per_sym=%d tagval=%d w0=%d "
                              "lenout=%d npay=%d prebuf=%d msgq=%d"
                              % (pn, fl, self.per_sym, fl * self.per_sym, w0,
                                 len(out), npay, len(self.pre_buf), len(self.msgq)),
                              flush=True)
                        self._dbg += 1
                produced += self.per_sym
            cpay += fl
            self.pend = None
        self.consume(2, cpay); self.consume(3, cpay)
        return produced
