"""Header-gated channel-tap logger with packet_len validation."""

import threading
import numpy as np
import pmt
import zmq
from gnuradio import gr


class blk(gr.sync_block):
    def __init__(self, fft_len=64, packet_len=960,
                 out_path='/tmp/h_dataset.npz',
                 snr_addr='tcp://127.0.0.1:5560', flush_every=200):
        gr.sync_block.__init__(
            self,
            name='chan_taps_logger',
            in_sig=[(np.complex64, fft_len)],
            out_sig=[(np.complex64, fft_len)],
        )
        self.fft_len = int(fft_len)
        self.packet_len = int(packet_len)
        self.out_path = str(out_path)
        self.flush_every = int(flush_every)
        self.tag_key = pmt.intern('ofdm_sync_chan_taps')
        self.pkt_num_key = pmt.intern('packet_num')
        self.pkt_len_key = pmt.intern('packet_len')
        self.pending = []
        self.max_pending = 8
        self.pn_list = []
        self.h_list = []
        self.n_rejected = 0
        self._lock = threading.Lock()
        self.msg_port = pmt.intern('header_valid')
        self.message_port_register_in(self.msg_port)
        self.set_msg_handler(self.msg_port, self._on_header_valid)
        self.snr_addr = str(snr_addr)
        self._ctx = None
        self._sock = None

    def _extract_int(self, msg, key):
        sentinel = pmt.from_long(-(1 << 30))
        candidates = [msg]
        try:
            if pmt.is_pair(msg):
                candidates.append(pmt.car(msg))
        except Exception:
            pass
        for cand in candidates:
            try:
                val = pmt.dict_ref(cand, key, sentinel)
                if not pmt.equal(val, sentinel):
                    return pmt.to_long(val)
            except Exception:
                continue
        return None

    def _on_header_valid(self, msg):
        pl = self._extract_int(msg, self.pkt_len_key)
        pn = self._extract_int(msg, self.pkt_num_key)
        if pl != self.packet_len or pn is None or pn < 0 or pn >= (1 << 24):
            self.n_rejected += 1
            return
        with self._lock:
            if not self.pending:
                return
            h_finalized = self.pending[-1]
            self.h_list.append(h_finalized)
            self.pn_list.append(int(pn))
            self.pending.clear()
            if len(self.pn_list) % self.flush_every == 0:
                self._flush()
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern('packet_num'), pmt.from_long(int(pn)))
        meta = pmt.dict_add(meta, pmt.intern('kind'), pmt.intern('h'))
        payload = pmt.init_c32vector(self.fft_len, h_finalized.tolist())
        self._send_nb(pmt.cons(meta, payload))

    def _flush(self):
        if not self.pn_list:
            return
        np.savez(self.out_path,
                 packet_num=np.array(self.pn_list, dtype=np.int64),
                 h=np.array(self.h_list, dtype=np.complex64))

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        out0[:] = in0
        n = len(in0)
        for tag in self.get_tags_in_window(0, 0, n, self.tag_key):
            taps = np.array(pmt.c32vector_elements(tag.value), dtype=np.complex64)
            with self._lock:
                self.pending.append(taps)
                if len(self.pending) > self.max_pending:
                    self.pending = self.pending[-self.max_pending:]
        return n

    def start(self):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SNDHWM, 1000)
        self._sock.connect(self.snr_addr)
        return True

    def _send_nb(self, pdu):
        s = self._sock
        if s is None:
            return
        try:
            s.send(pmt.serialize_str(pdu), flags=zmq.NOBLOCK)
        except (zmq.Again, zmq.ZMQError):
            pass

    def stop(self):
        with self._lock:
            self._flush()
        print('[chan_taps_logger] committed=%d rejected=%d -> %s'
              % (len(self.pn_list), self.n_rejected, self.out_path))
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
        return True
