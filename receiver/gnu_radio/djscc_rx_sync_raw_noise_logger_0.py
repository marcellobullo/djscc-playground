import threading
import numpy as np
import pmt
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self, fft_len=64, null_positions=[0, 1, 2, 3, 4, 5, 59, 60, 61, 62, 63], out_path='/tmp/raw_noise_dataset.npz', flush_every=200):
        gr.sync_block.__init__(self, name='raw_noise_logger', in_sig=[(np.complex64, fft_len)], out_sig=None)
        self.fft_len = int(fft_len)
        self.null_positions = np.array(null_positions, dtype=np.int64)
        self.out_path = str(out_path)
        self.flush_every = int(flush_every)
        self.pkt_num_key = pmt.intern('packet_num')
        self.frame_len_key = pmt.intern('frame_len')
        self._curr_pn = None
        self._curr_fl = 0
        self._curr_consumed = 0
        self._curr_noise_pwrs = []
        self.pn_list = []
        self.sigma2_list = []
        self._lock = threading.Lock()
        self.snr_part_port = pmt.intern('snr_part')
        self.message_port_register_out(self.snr_part_port)

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
                self._curr_noise_pwrs = []
            if self._curr_pn is not None and self._curr_consumed < self._curr_fl:
                null_syms = in0[i][self.null_positions]
                self._curr_noise_pwrs.append(np.mean(np.abs(null_syms)**2))
                self._curr_consumed += 1
                if self._curr_consumed >= self._curr_fl:
                    self._finalize_frame()
        return n

    def _finalize_frame(self):
        if self._curr_pn is None or not self._curr_noise_pwrs:
            self._curr_pn = None
            self._curr_noise_pwrs = []
            self._curr_consumed = 0
            self._curr_fl = 0
            return
        avg_sigma2 = float(np.mean(self._curr_noise_pwrs))
        pn_finalized = int(self._curr_pn)
        with self._lock:
            self.pn_list.append(pn_finalized)
            self.sigma2_list.append(avg_sigma2)
            if len(self.pn_list) % self.flush_every == 0:
                self._flush()
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern('packet_num'), pmt.from_long(pn_finalized))
        meta = pmt.dict_add(meta, pmt.intern('kind'), pmt.intern('raw'))
        meta = pmt.dict_add(meta, pmt.intern('sigma2'), pmt.from_double(avg_sigma2))
        self.message_port_pub(self.snr_part_port, pmt.cons(meta, pmt.PMT_NIL))
        self._curr_pn = None
        self._curr_noise_pwrs = []
        self._curr_consumed = 0
        self._curr_fl = 0

    def _flush(self):
        if not self.pn_list:
            return
        np.savez(self.out_path, packet_num=np.array(self.pn_list, dtype=np.int64), sigma2=np.array(self.sigma2_list, dtype=np.float32))

    def stop(self):
        with self._lock:
            self._flush()
        print('[raw_noise_logger] saved %d frames -> %s' % (len(self.pn_list), self.out_path))
        return True
