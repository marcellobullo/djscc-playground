/*
 * Copyright 2026 Marcello Bullo
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gnuradio/deepjscc/packet_header_ofdm_robust.h>
#include <gnuradio/digital/packet_header_ofdm.h>

namespace py = pybind11;

void bind_packet_header_ofdm_robust(py::module& m)
{
    using gr::deepjscc::packet_header_ofdm_robust;

    py::class_<packet_header_ofdm_robust,
               gr::digital::packet_header_ofdm,
               std::shared_ptr<packet_header_ofdm_robust>>(
        m, "packet_header_ofdm_robust")
        .def(py::init(&packet_header_ofdm_robust::make),
             py::arg("occupied_carriers"),
             py::arg("n_syms"),
             py::arg("len_tag_key") = "packet_len",
             py::arg("frame_len_tag_key") = "frame_len",
             py::arg("num_tag_key") = "packet_num",
             py::arg("bits_per_header_sym") = 1,
             py::arg("bits_per_payload_sym") = 1,
             py::arg("scramble_header") = false,
             py::arg("expected_packet_len") = -1,
             py::arg("num_bits") = 24,
             py::arg("expected_number_packets") = 0);
}
