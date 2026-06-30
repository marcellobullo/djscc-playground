#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 gr-deepjscc author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy
from gnuradio import gr

class PAPR(gr.sync_block):
    """
    Calculates the Peak-to-Average Power Ratio (PAPR) of a complex input vector.
    PAPR (dB) = 10 * log10 ( PeakPower / AveragePower )
    where Power is defined as abs(signal)*2 
    """
    def __init__(self, vec_len=720):
        gr.sync_block.__init__(self,
            name="PAPR",
            in_sig=[(numpy.complex64, vec_len)],
            out_sig=[numpy.float32])
        self.vec_len = vec_len
        self.dac_peak_voltage_limit = 1.8 / 2.0 # Vpp / 2

    def work(self, input_items, output_items):
        v   = input_items[0].reshape(-1)          # flatten full burst
        
        # Check DAC voltage limit
        peak_signal_voltage = numpy.max(numpy.abs(v))
        print(f"INFO: Peak Signal Voltage: {peak_signal_voltage:.2f}V < {self.dac_peak_voltage_limit:.2f}Vp")  # Debugging output
        if peak_signal_voltage > self.dac_peak_voltage_limit:
            print(f"WARNING: Signal peak voltage ({peak_signal_voltage:.2f}V) exceeds DAC limit ({self.dac_peak_voltage_limit:.2f}Vp).")

        pwr = numpy.abs(v) ** 2
        avg_power = pwr.mean()
        
        # Use a more appropriate epsilon for numerical stability
        if avg_power < 1e-10:
            papr_db = 0.0  # Assign a default value if average power is too low
        else:
            # Calculate PAPR normally if average power is significant
            papr_linear = pwr.max() / avg_power
            papr_db = 10 * numpy.log10(papr_linear)

        # avg_power_db_val = -numpy.inf  # Default to -infinity dB for zero or non-positive power
        # if avg_power > 0:  # Ensure argument to log10 is positive
        #     avg_power_db_val = 10 * numpy.log10(avg_power)
        # print(f"Average Power: {avg_power_db_val:.2f} dB")

        # peak_power_linear_val = pwr.max()
        # peak_power_db_val = -numpy.inf  # Default to -infinity dB for zero or non-positive power
        # if peak_power_linear_val > 0:  # Ensure argument to log10 is positive
        #     peak_power_db_val = 10 * numpy.log10(peak_power_linear_val)
        # print(f"Peak Power: {peak_power_db_val:.2f} dB")
        
        print(f"PAPR: {papr_db:.2f} dB")  # Debugging output
        output_items[0][0] = papr_db
        return 1 # Number of items produced on the output stream
