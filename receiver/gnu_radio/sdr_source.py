"""Radio selection for the GNU Radio TX/RX flowgraphs (``--device usrp|lime``).

The flowgraphs are generated from .grc files with a ``uhd.usrp_source`` block. Rather than
maintaining a second flowgraph per radio, ``uhd_compat`` returns an object that stands in for the
``uhd`` module: its ``usrp_source(...)`` / ``usrp_sink(...)`` build a SoapySDR source / sink
(LimeSDR by default) wrapped in a hierarchical block exposing the UHD calls the flowgraphs use
(set_samp_rate, set_center_freq, set_antenna, set_bandwidth, set_gain). The sink keeps the UHD
burst mode: the length tag ("packet_len") marks each burst. Everything downstream of the source is unchanged, so the
OFDM receiver, the pilot SNR logger, socket_rx.py and the SNR probe work with either radio.

LimeSDR notes: RX antenna LNAH covers ~2-2.6 GHz (2.45 GHz carrier); LNAW is the wideband input.
TX antenna: default = driver choice (as GNU Radio's LimeSDR sink block); BAND1/BAND2 naming differs
between LimeSDR models, so pass --lime-antenna only if needed. Soapy's overall Lime gain spans about
-12..61 dB (RX) / -12..64 dB (TX) and is not equivalent to a USRP gain value.
"""
from __future__ import annotations

import glob
import os
import sys

# Point SoapySDR at this environment's driver modules (e.g. <env>/lib/SoapySDR/modules0.8/libLMS7Support)
# when the plugin path is not set; otherwise SoapySDR may search a different prefix and find no driver.
if not os.environ.get("SOAPY_SDR_PLUGIN_PATH"):
    _mods = sorted(glob.glob(os.path.join(sys.prefix, "lib", "SoapySDR", "modules*")))
    if _mods:
        os.environ["SOAPY_SDR_PLUGIN_PATH"] = _mods[-1]

from gnuradio import gr, soapy  # noqa: E402


class SoapySource(gr.hier_block2):
    """SoapySDR RX source with the subset of the uhd.usrp_source API used by the flowgraphs."""

    GAIN_RANGE = (-12.0, 61.0)

    def __init__(self, device="driver=lime", antenna="LNAH", dc_correction=True):
        gr.hier_block2.__init__(self, "soapy_source", gr.io_signature(0, 0, 0),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.src = soapy.source(device, "fc32", 1, "", "", [""], [""])
        self.antenna = antenna
        self.src.set_antenna(0, antenna)
        if dc_correction:
            try:
                self.src.set_dc_offset_mode(0, True)
            except Exception:
                pass
        self.connect(self.src, self)

    def set_samp_rate(self, rate):
        self.src.set_sample_rate(0, float(rate))

    def set_center_freq(self, freq, chan=0):
        self.src.set_frequency(0, float(freq))

    def set_antenna(self, name, chan=0):
        # the flowgraph asks for the USRP port ("TX/RX"); keep the Lime antenna chosen on the CLI
        self.src.set_antenna(0, self.antenna)

    def set_bandwidth(self, bw, chan=0):
        self.src.set_bandwidth(0, float(bw))

    def set_gain(self, gain, chan=0):
        lo, hi = self.GAIN_RANGE
        self.src.set_gain(0, min(max(float(gain), lo), hi))


class SoapySink(gr.hier_block2):
    """SoapySDR TX sink with the subset of the uhd.usrp_sink API used by the flowgraphs (burst mode
    via the length tag, like the UHD sink)."""

    GAIN_RANGE = (-12.0, 64.0)

    def __init__(self, device="driver=lime", antenna="", length_tag="packet_len"):
        gr.hier_block2.__init__(self, "soapy_sink", gr.io_signature(1, 1, gr.sizeof_gr_complex),
                                gr.io_signature(0, 0, 0))
        self.snk = soapy.sink(device, "fc32", 1, "", "", [""], [""])
        self.antenna = antenna
        if antenna:
            self.snk.set_antenna(0, antenna)
        if length_tag:
            self.snk.set_length_tag_name(length_tag)
        self.connect(self, self.snk)

    def set_samp_rate(self, rate):
        self.snk.set_sample_rate(0, float(rate))

    def set_center_freq(self, freq, chan=0):
        self.snk.set_frequency(0, float(freq))

    def set_antenna(self, name, chan=0):
        if self.antenna:       # ignore the USRP port name ("TX/RX"); keep the CLI choice or driver default
            self.snk.set_antenna(0, self.antenna)

    def set_bandwidth(self, bw, chan=0):
        self.snk.set_bandwidth(0, float(bw))

    def set_gain(self, gain, chan=0):
        lo, hi = self.GAIN_RANGE
        self.snk.set_gain(0, min(max(float(gain), lo), hi))


class uhd_compat:
    """Drop-in for the ``uhd`` module inside a generated flowgraph."""

    def __init__(self, device="driver=lime", antenna="LNAH"):
        self.device, self.antenna = device, antenna

    def usrp_source(self, *args, **kwargs):
        print(f"[*] RX radio: SoapySDR '{self.device}', antenna {self.antenna}", flush=True)
        return SoapySource(self.device, self.antenna)

    def usrp_sink(self, addr=None, stream_args=None, length_tag="", *args, **kwargs):
        print(f"[*] TX radio: SoapySDR '{self.device}', antenna {self.antenna or 'driver default'}, "
              f"burst tag {length_tag!r}", flush=True)
        return SoapySink(self.device, self.antenna, length_tag)

    @staticmethod
    def stream_args(*args, **kwargs):
        return None
