#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM Image Receiver for DJSCC
# Author: Marcello Bullo
# Description: OFDM image receiver for deep joint source channel coding (DJSCC)
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
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
from gnuradio import gr, pdu
from gnuradio import uhd
import time
from gnuradio import zeromq
import djscc_rx_chan_taps_logger_0 as chan_taps_logger_0  # embedded python block
import djscc_rx_pilot_snr_logger_0 as pilot_snr_logger_0  # embedded python block
import djscc_rx_raw_noise_logger_0 as raw_noise_logger_0  # embedded python block
import sip
import threading



class djscc_rx(gr.top_block, Qt.QWidget):

    def __init__(self, band=5e6, carrier_freq=2.45e9, device_address="192.168.1.68", samp_rate=1e6):
        gr.top_block.__init__(self, "OFDM Image Receiver for DJSCC", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("OFDM Image Receiver for DJSCC")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "djscc_rx")

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
        self.samp_rate = samp_rate

        ##################################################
        # Variables
        ##################################################
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.packet_len = packet_len = 960
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.header_mod = header_mod = digital.constellation_bpsk()
        self.fft_len = fft_len = 64
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.receiver_gain = receiver_gain = 20
        self.payload_equalizer = payload_equalizer = digital.ofdm_equalizer_static(fft_len, occupied_carriers, pilot_carriers, pilot_symbols)
        self.header_formatter_rx = header_formatter_rx = deepjscc.packet_header_ofdm_wide(occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key="frame_len", num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True,  expected_packet_len=packet_len)
        self.header_equalizer = header_equalizer = digital.ofdm_equalizer_simpledfe(fft_len, header_mod.base(), occupied_carriers, pilot_carriers, pilot_symbols)

        ##################################################
        # Blocks
        ##################################################

        self._receiver_gain_range = qtgui.Range(0, 70, 5, 20, 200)
        self._receiver_gain_win = qtgui.RangeWidget(self._receiver_gain_range, self.set_receiver_gain, "'receiver_gain'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._receiver_gain_win)
        self.zeromq_push_msg_sink_1 = zeromq.push_msg_sink('tcp://127.0.0.1:5560', 100, True)
        self.zeromq_push_msg_sink_0 = zeromq.push_msg_sink('tcp://127.0.0.1:5558', 100, True)
        self.uhd_usrp_source_0_0 = uhd.usrp_source(
            ",".join(("addr="+device_address, '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0_0.set_center_freq(carrier_freq, 0)
        self.uhd_usrp_source_0_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0_0.set_bandwidth(band, 0)
        self.uhd_usrp_source_0_0.set_gain(receiver_gain, 0)
        self.raw_noise_logger_0 = raw_noise_logger_0.blk(fft_len=fft_len, null_positions=[0, 1, 2, 3, 4, 5, 59, 60, 61, 62, 63], out_path="/tmp/raw_noise_dataset.npz", flush_every=200)
        self.qtgui_waterfall_sink_x_0_RX = qtgui.waterfall_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            carrier_freq, #fc
            samp_rate, #bw
            "RX Raw IQ", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0_RX.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0_RX.enable_grid(False)
        self.qtgui_waterfall_sink_x_0_RX.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0_RX.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0_RX.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0_RX.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0_RX.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0_RX.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_RX_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0_RX.qwidget(), Qt.QWidget)

        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_RX_win)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
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
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_freq_sink_x_0_RX = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            carrier_freq, #fc
            samp_rate, #bw
            "RX Raw IQ PSD", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0_RX.set_update_time(0.10)
        self.qtgui_freq_sink_x_0_RX.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0_RX.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0_RX.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0_RX.enable_autoscale(False)
        self.qtgui_freq_sink_x_0_RX.enable_grid(False)
        self.qtgui_freq_sink_x_0_RX.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0_RX.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0_RX.enable_control_panel(False)
        self.qtgui_freq_sink_x_0_RX.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0_RX.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0_RX.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0_RX.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0_RX.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0_RX.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_RX_win = sip.wrapinstance(self.qtgui_freq_sink_x_0_RX.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_RX_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "RX", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
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
        self.pilot_snr_logger_0 = pilot_snr_logger_0.blk(fft_len=fft_len, pilot_positions=[11, 25, 39, 53], pilot_symbols=[1.0, 1.0, 1.0, -1.0], out_path="/tmp/pilot_dataset.npz", flush_every=200)
        self.pdu_tagged_stream_to_pdu_0 = pdu.tagged_stream_to_pdu(gr.types.complex_t, "packet_len")
        self.fft_vxx_1 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.digital_packet_headerparser_b_0_0 = digital.packet_headerparser_b(header_formatter_rx.base())
        self.digital_ofdm_sync_sc_cfb_0 = digital.ofdm_sync_sc_cfb(fft_len, (int(fft_len/4)), False, 0.9)
        self.digital_ofdm_serializer_vcc_payload = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, "frame_len", "packet_len", 1, '', True)
        self.digital_ofdm_serializer_vcc_header = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, '', '', 0, '', True)
        self.digital_ofdm_frame_equalizer_vcvc_1 = digital.ofdm_frame_equalizer_vcvc(payload_equalizer.base(), (int(fft_len/4)), "frame_len", True, 0)
        self.digital_ofdm_frame_equalizer_vcvc_1.set_min_output_buffer(1048576)
        self.digital_ofdm_frame_equalizer_vcvc_1.set_max_output_buffer(1048576)
        self.digital_ofdm_frame_equalizer_vcvc_0 = digital.ofdm_frame_equalizer_vcvc(header_equalizer.base(), (int(fft_len/4)), '', True, 1)
        self.digital_ofdm_chanest_vcvc_0 = digital.ofdm_chanest_vcvc(sync_word1, sync_word2, 1, 0, 3, False)
        self.digital_header_payload_demux_0 = digital.header_payload_demux(
            3,
            fft_len,
            (int(fft_len/4)),
            "frame_len",
            "",
            True,
            gr.sizeof_gr_complex,
            "rx_time",
            samp_rate,
            (),
            0)
        self.digital_header_payload_demux_0.set_min_output_buffer(2000000)
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(header_mod.base())
        self.chan_taps_logger_0 = chan_taps_logger_0.blk(fft_len=fft_len, packet_len=packet_len, out_path="/tmp/h_dataset.npz", flush_every=200)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0_1 = blocks.multiply_const_cc(1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, (fft_len+int(fft_len/4)))
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc((-2.0/fft_len))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.chan_taps_logger_0, 'snr_part'), (self.zeromq_push_msg_sink_1, 'in'))
        self.msg_connect((self.digital_packet_headerparser_b_0_0, 'header_data'), (self.chan_taps_logger_0, 'header_valid'))
        self.msg_connect((self.digital_packet_headerparser_b_0_0, 'header_data'), (self.digital_header_payload_demux_0, 'header_data'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_0, 'pdus'), (self.zeromq_push_msg_sink_0, 'in'))
        self.msg_connect((self.pilot_snr_logger_0, 'snr_part'), (self.zeromq_push_msg_sink_1, 'in'))
        self.msg_connect((self.raw_noise_logger_0, 'snr_part'), (self.zeromq_push_msg_sink_1, 'in'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_1, 0), (self.pdu_tagged_stream_to_pdu_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_1, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.chan_taps_logger_0, 0), (self.digital_ofdm_frame_equalizer_vcvc_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_packet_headerparser_b_0_0, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.fft_vxx_1, 0))
        self.connect((self.digital_ofdm_chanest_vcvc_0, 0), (self.chan_taps_logger_0, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_0, 0), (self.digital_ofdm_serializer_vcc_header, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_1, 0), (self.digital_ofdm_serializer_vcc_payload, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_1, 0), (self.pilot_snr_logger_0, 0))
        self.connect((self.digital_ofdm_serializer_vcc_header, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_ofdm_serializer_vcc_payload, 0), (self.blocks_multiply_const_vxx_0_1, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 1), (self.digital_header_payload_demux_0, 1))
        self.connect((self.fft_vxx_0_0, 0), (self.digital_ofdm_chanest_vcvc_0, 0))
        self.connect((self.fft_vxx_1, 0), (self.digital_ofdm_frame_equalizer_vcvc_1, 0))
        self.connect((self.fft_vxx_1, 0), (self.raw_noise_logger_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.digital_ofdm_sync_sc_cfb_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.qtgui_freq_sink_x_0_RX, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.qtgui_waterfall_sink_x_0_RX, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "djscc_rx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_band(self):
        return self.band

    def set_band(self, band):
        self.band = band
        self.uhd_usrp_source_0_0.set_bandwidth(self.band, 0)

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self.qtgui_freq_sink_x_0_RX.set_frequency_range(self.carrier_freq, self.samp_rate)
        self.qtgui_waterfall_sink_x_0_RX.set_frequency_range(self.carrier_freq, self.samp_rate)
        self.uhd_usrp_source_0_0.set_center_freq(self.carrier_freq, 0)

    def get_device_address(self):
        return self.device_address

    def set_device_address(self, device_address):
        self.device_address = device_address

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0_RX.set_frequency_range(self.carrier_freq, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_waterfall_sink_x_0_RX.set_frequency_range(self.carrier_freq, self.samp_rate)
        self.uhd_usrp_source_0_0.set_samp_rate(self.samp_rate)

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len, self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len, self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_header_formatter_rx(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key="frame_len", num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True,  expected_packet_len=self.packet_len))
        self.chan_taps_logger_0.packet_len = self.packet_len

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_header_formatter_rx(deepjscc.packet_header_ofdm_wide(self.occupied_carriers, n_syms=1, len_tag_key="packet_len",   frame_len_tag_key="frame_len", num_tag_key="packet_num", bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=True,  expected_packet_len=self.packet_len))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len, self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))

    def get_header_mod(self):
        return self.header_mod

    def set_header_mod(self, header_mod):
        self.header_mod = header_mod

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len, self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.analog_frequency_modulator_fc_0.set_sensitivity((-2.0/self.fft_len))
        self.blocks_delay_0.set_dly(int((self.fft_len+int(self.fft_len/4))))
        self.chan_taps_logger_0.fft_len = self.fft_len
        self.pilot_snr_logger_0.fft_len = self.fft_len
        self.raw_noise_logger_0.fft_len = self.fft_len

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_receiver_gain(self):
        return self.receiver_gain

    def set_receiver_gain(self, receiver_gain):
        self.receiver_gain = receiver_gain
        self.uhd_usrp_source_0_0.set_gain(self.receiver_gain, 0)

    def get_payload_equalizer(self):
        return self.payload_equalizer

    def set_payload_equalizer(self, payload_equalizer):
        self.payload_equalizer = payload_equalizer

    def get_header_formatter_rx(self):
        return self.header_formatter_rx

    def set_header_formatter_rx(self, header_formatter_rx):
        self.header_formatter_rx = header_formatter_rx

    def get_header_equalizer(self):
        return self.header_equalizer

    def set_header_equalizer(self, header_equalizer):
        self.header_equalizer = header_equalizer



def argument_parser():
    description = 'OFDM image receiver for deep joint source channel coding (DJSCC)'
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
        "--samp-rate", dest="samp_rate", type=eng_float, default=eng_notation.num_to_str(float(1e6)),
        help="Set Sampling Frequency [default=%(default)r]")
    return parser


def main(top_block_cls=djscc_rx, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(band=options.band, carrier_freq=options.carrier_freq, device_address=options.device_address, samp_rate=options.samp_rate)

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
