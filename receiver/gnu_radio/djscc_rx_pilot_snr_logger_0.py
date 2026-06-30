"""Per-frame equalized-pilot logger for offline SNR estimation."""

import threading
import numpy as np
import pmt
import zmq
from gnuradio import gr


class blk(gr.sync_block):
    def __init__(self, fft_len=64,
                 pilot_positions=[11, 25, 39, 53],
                 pilot_symbols=[1.0, 1.0, 1.0, -1.0],
                 out_path='/tmp/pilot_dataset.npz',
                 snr_addr='tcp://127.0.0.1:5560',
                 flush_every=200):
        gr.sync_block.__init__(
            self,
            name='pilot_snr_logger',
            in_sig=[(np.complex64, fft_len)],
            out_sig=None,
        )
        self.fft_len = int(fft_len)
        self.pilot_positions = np.array(pilot_positions, dtype=np.int64)
        self.pilot_symbols = np.array(pilot_symbols, dtype=np.complex64)
        self.out_path = str(out_path)
        self.flush_every = int(flush_every)
        self.pkt_num_key = pmt.intern('packet_num')
        self.frame_len_key = pmt.intern('frame_len')

        self._curr_pn = None
        self._curr_fl = 0
        self._curr_consumed = 0
        self._curr_pilots = []

        self.pn_list = []
        self.mean_pilots_list = []
        self.sigma2_list = []
        self._lock = threading.Lock()
        self.snr_addr = str(snr_addr)
        self._ctx = None
        self._sock = None

    def work(self, input_items, output_items):
        in0 = input_items[0]
        n = len(in0)
        nread = self.nitems_read(0)

        tag_map = {}
        for tag in self.get_tags_in_window(0, 0, n, self.pkt_num_key):
            tag_map.setdefault(int(tag.offset - nread), {})['pn'] = pmt.to_long(tag.value)
        for tag in self.get_tags_in_window(0, 0, n, self.frame_len_key):
            tag_map.setdefault(int(tag.offset - nread), {})['fl'] = pmt.to_long(tag.value)

        for i in range(n):
            if i in tag_map:
                self._finalize_frame()
                info = tag_map[i]
                self._curr_pn = info.get('pn')
                self._curr_fl = info.get('fl', 0)
                self._curr_consumed = 0
                self._curr_pilots = []
            if self._curr_pn is not None and self._curr_consumed < self._curr_fl:
                self._curr_pilots.append(in0[i][self.pilot_positions])
                self._curr_consumed += 1
                if self._curr_consumed >= self._curr_fl:
                    self._finalize_frame()
        return n

    def _finalize_frame(self):
        if self._curr_pn is None or not self._curr_pilots:
            self._curr_pn = None
            self._curr_pilots = []
            self._curr_consumed = 0
            self._curr_fl = 0
            return
        arr = np.array(self._curr_pilots, dtype=np.complex64)
        mean_p = arr.mean(axis=0)
        n_eq = arr - self.pilot_symbols[None, :]
        sigma2 = float(np.mean(np.abs(n_eq) ** 2))
        pn_finalized = int(self._curr_pn)
        with self._lock:
            self.pn_list.append(pn_finalized)
            self.mean_pilots_list.append(mean_p)
            self.sigma2_list.append(sigma2)
            if len(self.pn_list) % self.flush_every == 0:
                self._flush()
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern('packet_num'), pmt.from_long(pn_finalized))
        meta = pmt.dict_add(meta, pmt.intern('kind'), pmt.intern('pilots'))
        meta = pmt.dict_add(meta, pmt.intern('sigma2'), pmt.from_double(sigma2))
        self._send_nb(pmt.cons(meta, pmt.PMT_NIL))
        self._curr_pn = None
        self._curr_pilots = []
        self._curr_consumed = 0
        self._curr_fl = 0

    def _flush(self):
        if not self.pn_list:
            return
        np.savez(self.out_path,
                 packet_num=np.array(self.pn_list, dtype=np.int64),
                 mean_pilots=np.array(self.mean_pilots_list, dtype=np.complex64),
                 sigma2=np.array(self.sigma2_list, dtype=np.float32))

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
        print('[pilot_snr_logger] saved %d frames -> %s'
              % (len(self.pn_list), self.out_path))
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
        return True
