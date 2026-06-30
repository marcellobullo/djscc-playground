#
# Copyright 2008,2009 Free Software Foundation, Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

# The presence of this file turns this directory into a Python package

'''
This is the GNU Radio DEEPJSCC module. Place your Python package
description here (python/__init__.py).
'''
import os

# import pybind11 generated symbols into the deepjscc namespace
try:
    # this might fail if the module is python-only
    from .deepjscc_python import *
except ModuleNotFoundError:
    pass

# import any pure python here
# from .PAPR import PAPR
# from .image_eval import image_eval

# from .conv_packet_check import conv_packet_check
# from .snr_adjscc_decoder import snr_adjscc_decoder
# from .snr_estimator import snr_estimator




#
