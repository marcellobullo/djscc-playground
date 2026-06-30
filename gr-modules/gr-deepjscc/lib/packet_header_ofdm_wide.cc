/* -*- c++ -*- */
/*
 * Copyright 2026 Marcello Bullo
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/deepjscc/packet_header_ofdm_wide.h>

#include <cstring>
#include <stdexcept>

namespace gr {
namespace deepjscc {

packet_header_ofdm_wide::sptr
packet_header_ofdm_wide::make(const std::vector<std::vector<int>>& occupied_carriers,
                              int n_syms,
                              const std::string& len_tag_key,
                              const std::string& frame_len_tag_key,
                              const std::string& num_tag_key,
                              int bits_per_header_sym,
                              int bits_per_payload_sym,
                              bool scramble_header,
                              int expected_packet_len)
{
    return packet_header_ofdm_wide::sptr(
        new packet_header_ofdm_wide(occupied_carriers,
                                    n_syms,
                                    len_tag_key,
                                    frame_len_tag_key,
                                    num_tag_key,
                                    bits_per_header_sym,
                                    bits_per_payload_sym,
                                    scramble_header,
                                    expected_packet_len));
}

packet_header_ofdm_wide::packet_header_ofdm_wide(
    const std::vector<std::vector<int>>& occupied_carriers,
    int n_syms,
    const std::string& len_tag_key,
    const std::string& frame_len_tag_key,
    const std::string& num_tag_key,
    int bits_per_header_sym,
    int bits_per_payload_sym,
    bool scramble_header,
    int expected_packet_len)
    : gr::digital::packet_header_ofdm(occupied_carriers,
                                      n_syms,
                                      len_tag_key,
                                      frame_len_tag_key,
                                      num_tag_key,
                                      bits_per_header_sym,
                                      bits_per_payload_sym,
                                      scramble_header),
      d_packet_number_wide(0),
      d_expected_packet_len(expected_packet_len)
{
    // 12-bit length + 24-bit packet_num + 8-bit CRC = 44 header bits.
    if ((long)d_header_len * d_bits_per_byte < 44) {
        throw std::invalid_argument(
            "packet_header_ofdm_wide: header is too short to fit 12-bit "
            "length + 24-bit packet_num + 8-bit CRC (need >= 44 header bits).");
    }
    if (expected_packet_len > 0xFFF) {
        throw std::invalid_argument(
            "packet_header_ofdm_wide: expected_packet_len exceeds the 12-bit "
            "length field (max 4095).");
    }
}

packet_header_ofdm_wide::~packet_header_ofdm_wide() {}

bool packet_header_ofdm_wide::header_formatter(long packet_len,
                                               unsigned char* out,
                                               const std::vector<gr::tag_t>& tags)
{
    packet_len &= 0x0FFF;
    uint32_t pn = d_packet_number_wide & 0xFFFFFFu;

    // CRC is computed over the byte-packed {length, packet_num} representation
    // (5 bytes: 2 for the 12-bit length, 3 for the 24-bit packet_num).
    unsigned char buffer[] = {
        (unsigned char)(packet_len & 0xFF),
        (unsigned char)((packet_len >> 8) & 0x0F),
        (unsigned char)( pn        & 0xFF),
        (unsigned char)((pn >>  8) & 0xFF),
        (unsigned char)((pn >> 16) & 0xFF),
    };
    unsigned char crc = d_crc_impl.compute(buffer, sizeof(buffer));

    std::memset(out, 0x00, d_header_len);
    long k = 0;
    for (int i = 0; i < 12 && k < d_header_len; i += d_bits_per_byte, k++) {
        out[k] = (unsigned char)((packet_len >> i) & d_mask);
    }
    for (int i = 0; i < 24 && k < d_header_len; i += d_bits_per_byte, k++) {
        out[k] = (unsigned char)((pn >> i) & d_mask);
    }
    for (int i = 0; i < 8 && k < d_header_len; i += d_bits_per_byte, k++) {
        out[k] = (unsigned char)((crc >> i) & d_mask);
    }

    // Mirror the scrambling done by gr::digital::packet_header_ofdm.
    for (long i = 0; i < d_header_len; i++) {
        out[i] ^= d_scramble_mask[i];
    }

    d_packet_number_wide = (d_packet_number_wide + 1) & 0xFFFFFFu;
    return true;
}

bool packet_header_ofdm_wide::header_parser(const unsigned char* in,
                                            std::vector<gr::tag_t>& tags)
{
    // Descramble first (mirror gr::digital::packet_header_ofdm).
    std::vector<unsigned char> dq(d_header_len, 0);
    for (long i = 0; i < d_header_len; i++) {
        dq[i] = in[i] ^ d_scramble_mask[i];
    }

    unsigned header_len = 0;
    uint32_t header_num = 0;
    long k = 0;

    for (int i = 0; i < 12 && k < d_header_len; i += d_bits_per_byte, k++) {
        header_len |= (((unsigned)dq[k]) & d_mask) << i;
    }
    if (k >= d_header_len) {
        return false;
    }
    for (int i = 0; i < 24 && k < d_header_len; i += d_bits_per_byte, k++) {
        header_num |= (((uint32_t)dq[k]) & d_mask) << i;
    }
    if (k >= d_header_len) {
        return false;
    }

    unsigned char buffer[] = {
        (unsigned char)(header_len & 0xFF),
        (unsigned char)((header_len >> 8) & 0x0F),
        (unsigned char)( header_num        & 0xFF),
        (unsigned char)((header_num >>  8) & 0xFF),
        (unsigned char)((header_num >> 16) & 0xFF),
    };
    unsigned char crc_calcd = d_crc_impl.compute(buffer, sizeof(buffer));
    for (int i = 0; i < 8 && k < d_header_len; i += d_bits_per_byte, k++) {
        if ((((unsigned)dq[k]) & d_mask) != (((unsigned)crc_calcd >> i) & d_mask)) {
            return false;
        }
    }

    // Defeat 8-bit-CRC false-passes on noise: with a fixed expected length,
    // any header whose decoded length doesn't match is treated as bogus.
    if (d_expected_packet_len >= 0 &&
        (int)header_len != d_expected_packet_len) {
        return false;
    }

    // Convert byte length to # of complex payload symbols (mirror upstream).
    int packet_len = (int)header_len * 8 / d_bits_per_payload_sym;
    if ((int)header_len * 8 % d_bits_per_payload_sym) {
        packet_len++;
    }

    gr::tag_t tag;
    tag.key = d_len_tag_key;
    tag.value = pmt::from_long(packet_len);
    tags.push_back(tag);

    if (!pmt::equal(d_num_tag_key, pmt::PMT_NIL)) {
        tag.key = d_num_tag_key;
        tag.value = pmt::from_long((long)header_num);
        tags.push_back(tag);
    }

    // Compute frame_len: # of payload OFDM symbols in this frame
    // (same math as gr::digital::packet_header_ofdm::header_parser).
    int frame_len = 0;
    size_t car = 0;
    int symbols_accounted_for = 0;
    while (symbols_accounted_for < packet_len) {
        frame_len++;
        symbols_accounted_for += d_occupied_carriers[car].size();
        car = (car + 1) % d_occupied_carriers.size();
    }
    tag.key = d_frame_len_tag_key;
    tag.value = pmt::from_long(frame_len);
    tags.push_back(tag);

    return true;
}

} // namespace deepjscc
} // namespace gr
