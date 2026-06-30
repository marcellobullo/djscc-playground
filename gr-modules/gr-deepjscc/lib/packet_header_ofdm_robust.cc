/* -*- c++ -*- */
/*
 * Copyright 2026 Marcello Bullo
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/deepjscc/packet_header_ofdm_robust.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace gr {
namespace deepjscc {

packet_header_ofdm_robust::sptr
packet_header_ofdm_robust::make(const std::vector<std::vector<int>>& occupied_carriers,
                                int n_syms,
                                const std::string& len_tag_key,
                                const std::string& frame_len_tag_key,
                                const std::string& num_tag_key,
                                int bits_per_header_sym,
                                int bits_per_payload_sym,
                                bool scramble_header,
                                int expected_packet_len,
                                int num_bits,
                                int expected_number_packets)
{
    return packet_header_ofdm_robust::sptr(
        new packet_header_ofdm_robust(occupied_carriers,
                                      n_syms,
                                      len_tag_key,
                                      frame_len_tag_key,
                                      num_tag_key,
                                      bits_per_header_sym,
                                      bits_per_payload_sym,
                                      scramble_header,
                                      expected_packet_len,
                                      num_bits,
                                      expected_number_packets));
}

packet_header_ofdm_robust::packet_header_ofdm_robust(
    const std::vector<std::vector<int>>& occupied_carriers,
    int n_syms,
    const std::string& len_tag_key,
    const std::string& frame_len_tag_key,
    const std::string& num_tag_key,
    int bits_per_header_sym,
    int bits_per_payload_sym,
    bool scramble_header,
    int expected_packet_len,
    int num_bits,
    int expected_number_packets)
    : gr::digital::packet_header_ofdm(occupied_carriers,
                                      n_syms,
                                      len_tag_key,
                                      frame_len_tag_key,
                                      num_tag_key,
                                      bits_per_header_sym,
                                      bits_per_payload_sym,
                                      scramble_header),
      d_packet_number_robust(0),
      d_expected_packet_len(expected_packet_len),
      d_num_bits(0),
      d_num_mask(0),
      d_expected_number_packets(expected_number_packets),
      d_counter_modulus(0),
      d_warned_no_expected_len(false)
{
    if (expected_number_packets > 0) {
        // Auto-size the field to the bit length of the largest id (N-1), i.e.
        // ceil(log2(N)), with a floor of 1. packet_num is transmitted modulo N.
        int needed = 0;
        for (uint32_t v = (uint32_t)(expected_number_packets - 1); v > 0; v >>= 1) {
            needed++;
        }
        if (needed < 1) {
            needed = 1;   // N == 1 -> single id, still needs one bit
        }
        if (needed > 24) {
            throw std::invalid_argument(
                "packet_header_ofdm_robust: expected_number_packets needs more "
                "than 24 bits (max 2^24).");
        }
        d_num_bits = needed;
        d_counter_modulus = (uint32_t)expected_number_packets;
    } else {
        // packet_num is restricted to 24 bits because the CRC buffer packs it
        // into 3 bytes (matching packet_header_ofdm_wide); 1 bit is the floor.
        if (num_bits < 1 || num_bits > 24) {
            throw std::invalid_argument(
                "packet_header_ofdm_robust: num_bits must be in [1, 24].");
        }
        d_num_bits = num_bits;
        d_counter_modulus = (1u << num_bits);
    }
    d_num_mask = (d_num_bits >= 32) ? 0xFFFFFFFFu : ((1u << d_num_bits) - 1u);

    // On-air header is d_num_bits packet_num + 8-bit CRC bits (the 12-bit length
    // field is not transmitted; the length is fixed and known to both sides).
    if ((long)d_header_len * d_bits_per_byte < d_num_bits + 8) {
        throw std::invalid_argument(
            "packet_header_ofdm_robust: header is too short to fit num_bits "
            "packet_num + 8-bit CRC.");
    }
    if (expected_packet_len > 0xFFF) {
        throw std::invalid_argument(
            "packet_header_ofdm_robust: expected_packet_len exceeds the 12-bit "
            "length field (max 4095).");
    }
}

packet_header_ofdm_robust::~packet_header_ofdm_robust() {}

bool packet_header_ofdm_robust::header_formatter(long packet_len,
                                                 unsigned char* out,
                                                 const std::vector<gr::tag_t>& tags)
{
    packet_len &= 0x0FFF;
    // d_packet_number_robust already lives in [0, d_counter_modulus-1], so it is
    // packet_num mod expected_number_packets when that range is enabled.
    uint32_t pn = d_packet_number_robust & d_num_mask;

    // CRC is computed over the same byte-packed {length, packet_num}
    // representation used by packet_header_ofdm_wide (5 bytes: 2 for the
    // 12-bit length, 3 for the 24-bit packet_num), so the assumed length is
    // still protected by the CRC even though it is never put on the wire.
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
    // Length bits are intentionally NOT emitted; the header carries only the
    // low num_bits of packet_num followed by the CRC.
    for (int i = 0; i < d_num_bits && k < d_header_len; i += d_bits_per_byte, k++) {
        out[k] = (unsigned char)((pn >> i) & d_mask);
    }
    for (int i = 0; i < 8 && k < d_header_len; i += d_bits_per_byte, k++) {
        out[k] = (unsigned char)((crc >> i) & d_mask);
    }

    // Mirror the scrambling done by gr::digital::packet_header_ofdm.
    for (long i = 0; i < d_header_len; i++) {
        out[i] ^= d_scramble_mask[i];
    }

    d_packet_number_robust = (d_packet_number_robust + 1) % d_counter_modulus;
    return true;
}

bool packet_header_ofdm_robust::header_parser(const unsigned char* in,
                                              std::vector<gr::tag_t>& tags)
{
    // The length is not on the wire, so a fixed expected length is mandatory.
    if (d_expected_packet_len < 0) {
        if (!d_warned_no_expected_len) {
            std::fprintf(stderr,
                         "[packet_header_ofdm_robust] expected_packet_len must be "
                         ">= 0 on the receive side; rejecting all headers.\n");
            d_warned_no_expected_len = true;
        }
        return false;
    }

    // Descramble first (mirror gr::digital::packet_header_ofdm).
    std::vector<unsigned char> dq(d_header_len, 0);
    for (long i = 0; i < d_header_len; i++) {
        dq[i] = in[i] ^ d_scramble_mask[i];
    }

    uint32_t header_num = 0;
    long k = 0;

    for (int i = 0; i < d_num_bits && k < d_header_len; i += d_bits_per_byte, k++) {
        header_num |= (((uint32_t)dq[k]) & d_mask) << i;
    }
    if (k >= d_header_len) {
        return false;
    }

    // Rebuild the CRC buffer using the fixed, known length and the decoded
    // packet_num, exactly mirroring the formatter.
    unsigned fixed_len = (unsigned)d_expected_packet_len & 0x0FFFu;
    unsigned char buffer[] = {
        (unsigned char)(fixed_len & 0xFF),
        (unsigned char)((fixed_len >> 8) & 0x0F),
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

    // With a known packet count, a valid packet_num is always in [0, N-1].
    // Anything outside is a corrupted header that slipped through the 8-bit CRC
    // -> reject. This never rejects a good frame (formatter sends pn mod N).
    if (d_expected_number_packets > 0 &&
        header_num >= (uint32_t)d_expected_number_packets) {
        return false;
    }

    unsigned header_len = fixed_len;

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
