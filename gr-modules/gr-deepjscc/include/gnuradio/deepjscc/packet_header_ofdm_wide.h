/* -*- c++ -*- */
/*
 * Copyright 2026 Marcello Bullo
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_DEEPJSCC_PACKET_HEADER_OFDM_WIDE_H
#define INCLUDED_DEEPJSCC_PACKET_HEADER_OFDM_WIDE_H

#include <gnuradio/deepjscc/api.h>
#include <gnuradio/digital/packet_header_ofdm.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gr {
namespace deepjscc {

/*!
 * \brief OFDM packet header with a 24-bit packet_num field.
 *
 * Drop-in replacement for gr::digital::packet_header_ofdm with the
 * packet_num counter widened from 12 to 24 bits. Header layout:
 *
 *   Bits 0-11:  packet length      (12 bits)
 *   Bits 12-35: packet_num         (24 bits, wraps at 2^24)
 *   Bits 36-43: 8-bit CRC
 *   Bits 44+:   zero-padded out to header_len
 *
 * The CRC is computed over the same {len, packet_num} bytes used by the
 * upstream class, just with packet_num expanded to three bytes. Bit
 * scrambling and frame-length tag emission match
 * gr::digital::packet_header_ofdm.
 *
 * If expected_packet_len >= 0, header_parser also rejects any header whose
 * decoded length field does not equal that value. This guards the
 * downstream header_payload_demux against rare 8-bit-CRC false-passes on
 * noise that would otherwise cause sliding misalignment between the demux
 * and a fixed_frame_len equalizer.
 */
class DEEPJSCC_API packet_header_ofdm_wide : public gr::digital::packet_header_ofdm {
public:
    typedef std::shared_ptr<packet_header_ofdm_wide> sptr;

    packet_header_ofdm_wide(const std::vector<std::vector<int>>& occupied_carriers,
                            int n_syms,
                            const std::string& len_tag_key,
                            const std::string& frame_len_tag_key,
                            const std::string& num_tag_key,
                            int bits_per_header_sym,
                            int bits_per_payload_sym,
                            bool scramble_header,
                            int expected_packet_len);
    ~packet_header_ofdm_wide() override;

    bool header_formatter(long packet_len,
                          unsigned char* out,
                          const std::vector<gr::tag_t>& tags) override;
    bool header_parser(const unsigned char* in,
                       std::vector<gr::tag_t>& tags) override;

    static sptr make(const std::vector<std::vector<int>>& occupied_carriers,
                     int n_syms,
                     const std::string& len_tag_key = "packet_len",
                     const std::string& frame_len_tag_key = "frame_len",
                     const std::string& num_tag_key = "packet_num",
                     int bits_per_header_sym = 1,
                     int bits_per_payload_sym = 1,
                     bool scramble_header = false,
                     int expected_packet_len = -1);

private:
    uint32_t d_packet_number_wide;
    int d_expected_packet_len;
};

} // namespace deepjscc
} // namespace gr

#endif /* INCLUDED_DEEPJSCC_PACKET_HEADER_OFDM_WIDE_H */
