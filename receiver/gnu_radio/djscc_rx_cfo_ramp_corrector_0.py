"""Residual-CFO phase-ramp corrector for the DJSCC OFDM payload.

ofdm_equalizer_static divides every data carrier by the channel estimated on the
preamble and never updates it, so the residual CFO left by Schmidl-Cox shows up
as a phase that grows linearly across the OFDM symbols of each frame. The
decoder gets that rotation straight in the latent.

This block takes the equalized payload (in0) and the raw FFT output (in1),
both fft_len vectors in the same order. It buffers each frame (a 'frame_len'
tag on in0 gives its start and length). It averages the consecutive-symbol
pilot products sum_k sum_p raw[k, p] * conj(raw[k-1, p]) over the whole frame,
takes their angle as the per-symbol phase slope, and outputs the frame with
symbol k derotated by slope * k. Latency is one frame. Tags are copied from in0
to out0 only (in1 carries the same tags), so the serializer sees each tag once.
"""

import numpy as np
import pmt
from gnuradio import gr


class blk(gr.basic_block):
    def __init__(self, fft_len=64, pilot_positions=[11, 25, 39, 53], enable=True):
        gr.basic_block.__init__(
            self,
            name='cfo_ramp_corrector',
            in_sig=[(np.complex64, fft_len), (np.complex64, fft_len)],
            out_sig=[(np.complex64, fft_len)],
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)
        self.fft_len = int(fft_len)
        self.pilot_positions = np.array(pilot_positions, dtype=np.int64)
        self.enable = bool(enable)
        self.frame_len_key = pmt.intern('frame_len')
        self._eq, self._raw, self._tags = [], [], []   # current frame
        self._need = 0                                  # symbols still missing
        self._ready = []                                # (vector, [tags]) waiting for output space

    def forecast(self, noutput_items, ninputs):
        # finished symbols waiting for output space need no new input
        return [0 if self._ready else 1] * ninputs

    def _finish_frame(self):
        if not self._eq:
            return
        eq = np.array(self._eq)
        if self.enable and len(eq) > 1:
            p = np.array(self._raw)[:, self.pilot_positions]
            slope = np.angle(np.sum(p[1:] * np.conj(p[:-1])))
            eq = eq * np.exp(-1j * slope * np.arange(len(eq)))[:, None].astype(np.complex64)
        for k in range(len(eq)):
            self._ready.append((eq[k], self._tags[k]))
        self._eq, self._raw, self._tags, self._need = [], [], [], 0

    def general_work(self, input_items, output_items):
        eq_in, raw_in, out = input_items[0], input_items[1], output_items[0]
        n_in = min(len(eq_in), len(raw_in))
        nread = self.nitems_read(0)
        tags_at = {}
        for t in self.get_tags_in_window(0, 0, n_in):
            tags_at.setdefault(int(t.offset - nread), []).append(t)
        for i in range(n_in):
            ts = tags_at.get(i, [])
            fl = [t for t in ts if pmt.eq(t.key, self.frame_len_key)]
            if fl:                               # a new frame starts: close the old one
                self._finish_frame()
                self._need = pmt.to_long(fl[0].value)
            self._eq.append(np.array(eq_in[i]))
            self._raw.append(np.array(raw_in[i]))
            self._tags.append(ts)
            self._need -= 1
            if self._need <= 0:                  # frame complete (or untagged symbol)
                self._finish_frame()
        self.consume(0, n_in)
        self.consume(1, n_in)

        n_out = min(len(out), len(self._ready))
        nw = self.nitems_written(0)
        for k in range(n_out):
            vec, ts = self._ready[k]
            out[k] = vec
            for t in ts:
                self.add_item_tag(0, nw + k, t.key, t.value, t.srcid)
        del self._ready[:n_out]
        return n_out
