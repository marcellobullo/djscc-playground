#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM Transmitter for Conventional SSCC
# Description: OFDM transmitter for conventional separated source-channel coding (SSCC)
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
import numpy as np
import os, math
import sip
import threading



class conventional_tx(gr.top_block, Qt.QWidget):

    def __init__(self, band=5e6, carrier_freq=2.45e9, device_address="192.168.1.68", mod_order=1, samp_rate=1e6):
        gr.top_block.__init__(self, "OFDM Transmitter for Conventional SSCC", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("OFDM Transmitter for Conventional SSCC")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "conventional_tx")

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
        self.mod_order = mod_order
        self.samp_rate = samp_rate

        ##################################################
        # Variables
        ##################################################
        self.sw_multiplier = sw_multiplier = 4
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.length_tag_key = length_tag_key = "packet_len"
        self.header_mod = header_mod = digital.constellation_bpsk()
        self.tx_gain = tx_gain = 18
        self.sync_word2 = sync_word2 = sw_multiplier*np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0])
        self.sync_word1 = sync_word1 = sw_multiplier*np.array([0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.])
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.payload_mod = payload_mod = {1: digital.constellation_bpsk(), 2: digital.constellation_qpsk(), 4: digital.constellation_16qam()}[mod_order]
        self.packet_length_tag_key = packet_length_tag_key = "packet_len"
        self.packet_len = packet_len = 960
        self.header_formatter_tx_wide = header_formatter_tx_wide = deepjscc.packet_header_ofdm_wide(occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key=length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True)
        self.header_formatter_tx_robust = header_formatter_tx_robust = deepjscc.packet_header_ofdm_robust(occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key=length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True, num_bits=16,use_fec=True)
        self.header_boost = header_boost = 2
        self.fft_len = fft_len = 64

        ##################################################
        # Blocks
        ##################################################

        self._tx_gain_range = qtgui.Range(0, 70, 8, 18, 200)
        self._tx_gain_win = qtgui.RangeWidget(self._tx_gain_range, self.set_tx_gain, "'tx_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_gain_win)
        self._header_boost_range = qtgui.Range(1, 10, 1, 2, 200)
        self._header_boost_win = qtgui.RangeWidget(self._header_boost_range, self.set_header_boost, "Header Boost", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._header_boost_win)
        self.zeromq_sub_source_0 = zeromq.sub_source(gr.sizeof_char, 1, 'tcp://127.0.0.1:5557', 100, False, 5000, '', False)
        self.zeromq_sub_source_0.set_min_output_buffer(2000000)
        self.zeromq_sub_source_0.set_max_output_buffer(2000000)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("addr="+device_address, '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            "packet_len",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0.set_center_freq(carrier_freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_bandwidth(band, 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 0)
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "After-OFDM", #name
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
        self.qtgui_time_sink_x_0_1_1_0_0 = qtgui.time_sink_c(
            (int(packet_len*5.1)), #size
            samp_rate, #samp_rate
            "TX: after tag ", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_1_1_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_1_1_0_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0_1_1_0_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_1_1_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_1_1_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0_1_1_0_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0_1_1_0_0.enable_grid(False)
        self.qtgui_time_sink_x_0_1_1_0_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_1_1_0_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0_1_1_0_0.enable_stem_plot(False)


        labels = ['Re', 'Im', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [2, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0_1_1_0_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0_1_1_0_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0_1_1_0_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_1_1_0_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_1_1_0_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_1_1_0_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_1_1_0_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_1_1_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_1_1_0_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0_1_1_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_1_1_0_0_win)
        self.qtgui_time_sink_x_0_1_1 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "TX: After-OFDM", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_1_1.set_update_time(0.10)
        self.qtgui_time_sink_x_0_1_1.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0_1_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_1_1.enable_tags(True)
        self.qtgui_time_sink_x_0_1_1.set_trigger_mode(qtgui.TRIG_MODE_TAG, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "packet_len")
        self.qtgui_time_sink_x_0_1_1.enable_autoscale(False)
        self.qtgui_time_sink_x_0_1_1.enable_grid(False)
        self.qtgui_time_sink_x_0_1_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_1_1.enable_control_panel(False)
        self.qtgui_time_sink_x_0_1_1.enable_stem_plot(False)


        labels = ['Re', 'Im', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [2, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0_1_1.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0_1_1.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0_1_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_1_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_1_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_1_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_1_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_1_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_1_1_win = sip.wrapinstance(self.qtgui_time_sink_x_0_1_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_1_1_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "Transmitted Constellation", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(True)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
        self.fft_vxx_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.digital_packet_headergenerator_bb_0 = digital.packet_headergenerator_bb(header_formatter_tx_robust, "packet_len")
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + int(fft_len/4),
            0,
            length_tag_key)
        self.digital_ofdm_carrier_allocator_cvc_0 = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers, pilot_carriers, pilot_symbols, (sync_word1, sync_word2), length_tag_key, True)
        self.digital_chunks_to_symbols_xx_0_0_0 = digital.chunks_to_symbols_bc(payload_mod.points(), 1)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, "packet_len", 0)
        self.blocks_stream_to_tagged_stream_0_1 = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, 1, packet_len, "packet_len")
        self.blocks_stream_to_tagged_stream_0_1.set_min_output_buffer(2000000)
        self.blocks_stream_to_tagged_stream_0_1.set_max_output_buffer(2000000)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, mod_order, '', False, gr.GR_LSB_FIRST)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_cc(header_boost)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.03)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer(2000000)
        self.blocks_multiply_const_vxx_0.set_max_output_buffer(2000000)
        self.blocks_float_to_uchar_0 = blocks.float_to_uchar(1, 1, 0)
        self.blocks_complex_to_real_0 = blocks.complex_to_real(1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_real_0, 0), (self.blocks_float_to_uchar_0, 0))
        self.connect((self.blocks_float_to_uchar_0, 0), (self.digital_packet_headergenerator_bb_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_time_sink_x_0_1_1, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_waterfall_sink_x_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_1, 0), (self.blocks_complex_to_real_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_1, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.blocks_stream_to_tagged_stream_0_1, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_1, 0), (self.qtgui_time_sink_x_0_1_1_0_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_multiply_const_vxx_1, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0_0, 0), (self.blocks_stream_to_tagged_stream_0_1, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.digital_packet_headergenerator_bb_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.zeromq_sub_source_0, 0), (self.blocks_repack_bits_bb_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "conventional_tx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_band(self):
        return self.band

    def set_band(self, band):
        self.band = band
        self.uhd_usrp_sink_0.set_bandwidth(self.band, 0)

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self.uhd_usrp_sink_0.set_center_freq(self.carrier_freq, 0)

    def get_device_address(self):
        return self.device_address

    def set_device_address(self, device_address):
        self.device_address = device_address

    def get_mod_order(self):
        return self.mod_order

    def set_mod_order(self, mod_order):
        self.mod_order = mod_order
        self.set_payload_mod({1: digital.constellation_bpsk(), 2: digital.constellation_qpsk(), 4: digital.constellation_16qam()}[self.mod_order])
        self.blocks_repack_bits_bb_0.set_k_and_l(8,self.mod_order)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_time_sink_x_0_1_1.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_0_1_1_0_0.set_samp_rate(self.samp_rate)
        self.qtgui_waterfall_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_sw_multiplier(self):
        return self.sw_multiplier

    def set_sw_multiplier(self, sw_multiplier):
        self.sw_multiplier = sw_multiplier
        self.set_sync_word1(self.sw_multiplier*np.array([0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]))
        self.set_sync_word2(self.sw_multiplier*np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]))

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_header_formatter_tx_robust(deepjscc.packet_header_ofdm_robust(self.occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True, num_bits=16,use_fec=True))
        self.set_header_formatter_tx_wide(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True))

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key
        self.set_header_formatter_tx_robust(deepjscc.packet_header_ofdm_robust(self.occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True, num_bits=16,use_fec=True))
        self.set_header_formatter_tx_wide(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key=self.length_tag_key, num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True))

    def get_header_mod(self):
        return self.header_mod

    def set_header_mod(self, header_mod):
        self.header_mod = header_mod

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_packet_length_tag_key(self):
        return self.packet_length_tag_key

    def set_packet_length_tag_key(self, packet_length_tag_key):
        self.packet_length_tag_key = packet_length_tag_key

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.blocks_stream_to_tagged_stream_0_1.set_packet_len(self.packet_len)
        self.blocks_stream_to_tagged_stream_0_1.set_packet_len_pmt(self.packet_len)

    def get_header_formatter_tx_wide(self):
        return self.header_formatter_tx_wide

    def set_header_formatter_tx_wide(self, header_formatter_tx_wide):
        self.header_formatter_tx_wide = header_formatter_tx_wide

    def get_header_formatter_tx_robust(self):
        return self.header_formatter_tx_robust

    def set_header_formatter_tx_robust(self, header_formatter_tx_robust):
        self.header_formatter_tx_robust = header_formatter_tx_robust
        self.digital_packet_headergenerator_bb_0.set_header_formatter(self.header_formatter_tx_robust)

    def get_header_boost(self):
        return self.header_boost

    def set_header_boost(self, header_boost):
        self.header_boost = header_boost
        self.blocks_multiply_const_vxx_1.set_k(self.header_boost)

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len



def argument_parser():
    description = 'OFDM transmitter for conventional separated source-channel coding (SSCC)'
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
        "--mod-order", dest="mod_order", type=intx, default=1,
        help="Set Bits Per Modulation Symbol [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default=eng_notation.num_to_str(float(1e6)),
        help="Set Sampling Frequency [default=%(default)r]")
    return parser


def main(top_block_cls=conventional_tx, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(band=options.band, carrier_freq=options.carrier_freq, device_address=options.device_address, mod_order=options.mod_order, samp_rate=options.samp_rate)

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
