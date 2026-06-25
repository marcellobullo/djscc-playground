#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM Image Transmitter for DJSCC (2x2 MIMO)
# Author: Marcello Bullo
# Description: 2x2 spatial-multiplexing MIMO OFDM image transmitter for DJSCC
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
from gnuradio import deepjscc
from gnuradio import digital
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio import zeromq
import sip
import threading



class djscc_tx_mimo(gr.top_block, Qt.QWidget):

    def __init__(self, band=5e6, carrier_freq=2.45e9, device_address="192.168.1.68", device_address_1="192.168.1.22", samp_rate=1e6):
        gr.top_block.__init__(self, "OFDM Image Transmitter for DJSCC (2x2 MIMO)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("OFDM Image Transmitter for DJSCC (2x2 MIMO)")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "djscc_tx_mimo")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.band = band
        self.carrier_freq = carrier_freq
        self.device_address = device_address
        self.device_address_1 = device_address_1
        self.samp_rate = samp_rate

        ##################################################
        # Variables
        ##################################################
        self.packet_len = packet_len = 960
        self.fft_len = fft_len = 64
        self.zeros_word = zeros_word = [0]*fft_len
        self.tx_scale = tx_scale = 0.03/(2**0.5)
        self.tx_gain = tx_gain = 18
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.stream_len = stream_len = packet_len//2
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.length_tag_key = length_tag_key = "packet_len"
        self.header_mod = header_mod = digital.constellation_bpsk()

        ##################################################
        # Blocks
        ##################################################

        self._tx_gain_range = qtgui.Range(0, 70, 8, 18, 200)
        self._tx_gain_win = qtgui.RangeWidget(self._tx_gain_range, self.set_tx_gain, "'tx_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_gain_win)
        self.zeromq_sub_source_0 = zeromq.sub_source(gr.sizeof_gr_complex, 1, 'tcp://127.0.0.1:5556', 100, False, 5000, '', False)
        self.zeromq_sub_source_0.set_min_output_buffer(2000000)
        self.zeromq_sub_source_0.set_max_output_buffer(2000000)
        self.zero_hdr_b = blocks.multiply_const_cc(0)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("addr0="+device_address+",addr1="+device_address_1, '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,2)),
            ),
            "packet_len",
        )
        self.uhd_usrp_sink_0.set_clock_source('mimo', 1)
        self.uhd_usrp_sink_0.set_time_source('mimo', 1)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0.set_center_freq(carrier_freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_bandwidth(band, 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 0)

        self.uhd_usrp_sink_0.set_center_freq(carrier_freq, 1)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 1)
        self.uhd_usrp_sink_0.set_bandwidth(band, 1)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 1)
        self.sts_b = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, stream_len, "packet_len")
        self.sts_a = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, stream_len, "packet_len")
        self.scale_b = blocks.multiply_const_cc(tx_scale)
        self.scale_a = blocks.multiply_const_cc(tx_scale)
        self.qtgui_waterfall_sink_x_0_0 = qtgui.waterfall_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "TX MIMO After-OFDM (ant1)", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0_0.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0_0.qwidget(), Qt.QWidget)

        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_0_win)
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "TX MIMO After-OFDM (ant0)", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.qwidget(), Qt.QWidget)

        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_win)
        self.mux_b = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, "packet_len", 0)
        self.mux_a = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, "packet_len", 0)
        self.hdrgen_b = digital.packet_headergenerator_bb(deepjscc.packet_header_ofdm_wide(occupied_carriers, n_syms=1, len_tag_key="packet_len", frame_len_tag_key=length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True), "packet_len")
        self.hdrgen_a = digital.packet_headergenerator_bb(deepjscc.packet_header_ofdm_wide(occupied_carriers, n_syms=1, len_tag_key="packet_len", frame_len_tag_key=length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True), "packet_len")
        self.fft_b = fft.fft_vcc(fft_len, False, (), True, 1)
        self.fft_a = fft.fft_vcc(fft_len, False, (), True, 1)
        self.f2u_b = blocks.float_to_uchar(1, 1, 0)
        self.f2u_a = blocks.float_to_uchar(1, 1, 0)
        self.cp_b = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + int(fft_len/4),
            0,
            length_tag_key)
        self.cp_a = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + int(fft_len/4),
            0,
            length_tag_key)
        self.chunks_b = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.chunks_a = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.c2r_b = blocks.complex_to_real(1)
        self.c2r_a = blocks.complex_to_real(1)
        self.blocks_deinterleave_0 = blocks.deinterleave(gr.sizeof_gr_complex*1, 48)
        self.alloc_b = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers, pilot_carriers, pilot_symbols, (sync_word1, zeros_word, sync_word2), length_tag_key, True)
        self.alloc_a = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers, pilot_carriers, pilot_symbols, (sync_word1, sync_word2, zeros_word), length_tag_key, True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.alloc_a, 0), (self.fft_a, 0))
        self.connect((self.alloc_b, 0), (self.fft_b, 0))
        self.connect((self.blocks_deinterleave_0, 0), (self.sts_a, 0))
        self.connect((self.blocks_deinterleave_0, 1), (self.sts_b, 0))
        self.connect((self.c2r_a, 0), (self.f2u_a, 0))
        self.connect((self.c2r_b, 0), (self.f2u_b, 0))
        self.connect((self.chunks_a, 0), (self.mux_a, 0))
        self.connect((self.chunks_b, 0), (self.zero_hdr_b, 0))
        self.connect((self.cp_a, 0), (self.scale_a, 0))
        self.connect((self.cp_b, 0), (self.scale_b, 0))
        self.connect((self.f2u_a, 0), (self.hdrgen_a, 0))
        self.connect((self.f2u_b, 0), (self.hdrgen_b, 0))
        self.connect((self.fft_a, 0), (self.cp_a, 0))
        self.connect((self.fft_b, 0), (self.cp_b, 0))
        self.connect((self.hdrgen_a, 0), (self.chunks_a, 0))
        self.connect((self.hdrgen_b, 0), (self.chunks_b, 0))
        self.connect((self.mux_a, 0), (self.alloc_a, 0))
        self.connect((self.mux_b, 0), (self.alloc_b, 0))
        self.connect((self.scale_a, 0), (self.qtgui_waterfall_sink_x_0, 0))
        self.connect((self.scale_a, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.scale_b, 0), (self.qtgui_waterfall_sink_x_0_0, 0))
        self.connect((self.scale_b, 0), (self.uhd_usrp_sink_0, 1))
        self.connect((self.sts_a, 0), (self.c2r_a, 0))
        self.connect((self.sts_a, 0), (self.mux_a, 1))
        self.connect((self.sts_b, 0), (self.c2r_b, 0))
        self.connect((self.sts_b, 0), (self.mux_b, 1))
        self.connect((self.zero_hdr_b, 0), (self.mux_b, 0))
        self.connect((self.zeromq_sub_source_0, 0), (self.blocks_deinterleave_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "djscc_tx_mimo")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_band(self):
        return self.band

    def set_band(self, band):
        self.band = band
        self.uhd_usrp_sink_0.set_bandwidth(self.band, 0)
        self.uhd_usrp_sink_0.set_bandwidth(self.band, 1)

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self.uhd_usrp_sink_0.set_center_freq(self.carrier_freq, 0)
        self.uhd_usrp_sink_0.set_center_freq(self.carrier_freq, 1)

    def get_device_address(self):
        return self.device_address

    def set_device_address(self, device_address):
        self.device_address = device_address

    def get_device_address_1(self):
        return self.device_address_1

    def set_device_address_1(self, device_address_1):
        self.device_address_1 = device_address_1

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_waterfall_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_waterfall_sink_x_0_0.set_frequency_range(0, self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_stream_len(self.packet_len//2)

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_zeros_word([0]*self.fft_len)

    def get_zeros_word(self):
        return self.zeros_word

    def set_zeros_word(self, zeros_word):
        self.zeros_word = zeros_word

    def get_tx_scale(self):
        return self.tx_scale

    def set_tx_scale(self, tx_scale):
        self.tx_scale = tx_scale
        self.scale_a.set_k(self.tx_scale)
        self.scale_b.set_k(self.tx_scale)

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 1)

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_stream_len(self):
        return self.stream_len

    def set_stream_len(self, stream_len):
        self.stream_len = stream_len
        self.sts_a.set_packet_len(self.stream_len)
        self.sts_a.set_packet_len_pmt(self.stream_len)
        self.sts_b.set_packet_len(self.stream_len)
        self.sts_b.set_packet_len_pmt(self.stream_len)

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.hdrgen_a.set_header_formatter(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len", frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True))
        self.hdrgen_b.set_header_formatter(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len", frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True))

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key
        self.hdrgen_a.set_header_formatter(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len", frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True))
        self.hdrgen_b.set_header_formatter(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len", frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True))

    def get_header_mod(self):
        return self.header_mod

    def set_header_mod(self, header_mod):
        self.header_mod = header_mod



def argument_parser():
    description = '2x2 spatial-multiplexing MIMO OFDM image transmitter for DJSCC'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--band", dest="band", type=eng_float, default=eng_notation.num_to_str(float(5e6)),
        help="Set Band [default=%(default)r]")
    parser.add_argument(
        "--carrier-freq", dest="carrier_freq", type=eng_float, default=eng_notation.num_to_str(float(2.45e9)),
        help="Set Carrier Frequency [default=%(default)r]")
    parser.add_argument(
        "--device-address", dest="device_address", type=str, default="192.168.1.68",
        help="Set Device IP address  [default=%(default)r]")
    parser.add_argument(
        "--device-address-1", dest="device_address_1", type=str, default="192.168.1.22",
        help="Set Device IP address (USRP 1) [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default=eng_notation.num_to_str(float(1e6)),
        help="Set Sampling Frequency [default=%(default)r]")
    return parser


def main(top_block_cls=djscc_tx_mimo, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(band=options.band, carrier_freq=options.carrier_freq, device_address=options.device_address, device_address_1=options.device_address_1, samp_rate=options.samp_rate)

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
