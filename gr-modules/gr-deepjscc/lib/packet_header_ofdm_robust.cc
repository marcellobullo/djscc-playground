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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace gr {
namespace deepjscc {

namespace {

// Extended binary Golay(24,12,8) B matrix; bit (11-j) of row i is B[i][j].
// Verified independently: symmetric, 4096 distinct codewords, min distance 8.
const uint16_t kGolayB[12] = {
    0b011111111111u, 0b111011100010u, 0b110111000101u, 0b101110001011u,
    0b111100010110u, 0b111000101101u, 0b110001011011u, 0b100010110111u,
    0b100101101110u, 0b101011011100u, 0b110110111000u, 0b101101110001u,
};

// Systematic encode of a 12-bit message into a 24-bit codeword:
// bits 0..11 = message, bits 12..23 = parity (= message * B over GF(2)).
uint32_t golay24_encode(uint16_t m)
{
    uint16_t parity = 0;
    for (int j = 0; j < 12; j++) {
        int b = 0;
        for (int i = 0; i < 12; i++) {
            b ^= ((m >> i) & 1) & ((kGolayB[i] >> (11 - j)) & 1);
        }
        parity |= (uint16_t)(b << j);
    }
    return (uint32_t)(m & 0x0FFFu) | ((uint32_t)parity << 12);
}

const std::array<uint32_t, 4096>& golay24_codebook()
{
    static const std::array<uint32_t, 4096> cb = [] {
        std::array<uint32_t, 4096> t{};
        for (uint16_t m = 0; m < 4096; m++) {
            t[m] = golay24_encode(m);
        }
        return t;
    }();
    return cb;
}

// Bounded-distance-3 decode: nearest codeword wins; success iff its Hamming
// distance is <= 3 (unique for a distance-8 code). m_out gets the 12-bit
// message; ok == false flags an uncorrectable (> 3-error) codeword.
bool golay24_decode(uint32_t r, uint16_t& m_out)
{
    const auto& cb = golay24_codebook();
    r &= 0xFFFFFFu;
    int best_d = 99, best_m = 0;
    for (int m = 0; m < 4096; m++) {
        int d = __builtin_popcount(r ^ cb[m]);
        if (d < best_d) {
            best_d = d;
            best_m = m;
            if (d == 0) {
                break;
            }
        }
    }
    m_out = (uint16_t)best_m;
    return best_d <= 3;
}

} // namespace

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
                                int expected_number_packets,
                                bool use_fec)
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
                                      expected_number_packets,
                                      use_fec));
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
    int expected_number_packets,
    bool use_fec)
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
      d_use_fec(use_fec),
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

    if (d_use_fec) {
        // Golay encodes one BPSK bit per subcarrier, so the header symbols must
        // be BPSK, and there must be room for ceil((num_bits+8)/12) 24-bit words.
        if (d_bits_per_byte != 1) {
            throw std::invalid_argument(
                "packet_header_ofdm_robust: use_fec currently requires a BPSK "
                "header (bits_per_header_sym == 1).");
        }
        int n_words = (d_num_bits + 8 + 11) / 12;
        if ((long)n_words * 24 > (long)d_header_len) {
            throw std::invalid_argument(
                "packet_header_ofdm_robust: header too short for Golay FEC; "
                "reduce num_bits / expected_number_packets (with a 48-bit header "
                "symbol num_bits must be <= 16).");
        }
    } else {
        // On-air header is d_num_bits packet_num + 8-bit CRC bits (the 12-bit
        // length field is not transmitted; length is fixed and known both ends).
        if ((long)d_header_len * d_bits_per_byte < d_num_bits + 8) {
            throw std::invalid_argument(
                "packet_header_ofdm_robust: header is too short to fit num_bits "
                "packet_num + 8-bit CRC.");
        }
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
    if (d_use_fec) {
        // info word = [packet_num (d_num_bits) | CRC (8)], LSB-first, then
        // Golay(24,12)-encoded in 12-bit chunks (one BPSK bit per subcarrier).
        uint32_t info = (pn & d_num_mask) | ((uint32_t)crc << d_num_bits);
        int n_words = (d_num_bits + 8 + 11) / 12;
        long oi = 0;
        for (int w = 0; w < n_words; w++) {
            uint16_t m12 = (uint16_t)((info >> (12 * w)) & 0x0FFFu);
            uint32_t cw = golay24_encode(m12);
            for (int c = 0; c < 24 && oi < d_header_len; c++, oi++) {
                out[oi] = (unsigned char)((cw >> c) & 1);
            }
        }
    } else {
        long k = 0;
        // Length bits are intentionally NOT emitted; the header carries only the
        // low num_bits of packet_num followed by the CRC.
        for (int i = 0; i < d_num_bits && k < d_header_len; i += d_bits_per_byte, k++) {
            out[k] = (unsigned char)((pn >> i) & d_mask);
        }
        for (int i = 0; i < 8 && k < d_header_len; i += d_bits_per_byte, k++) {
            out[k] = (unsigned char)((crc >> i) & d_mask);
        }
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

    // Decoded packet_num and the fixed, known length used to rebuild the CRC
    // buffer, exactly mirroring the formatter.
    uint32_t header_num = 0;
    unsigned fixed_len = (unsigned)d_expected_packet_len & 0x0FFFu;

    if (d_use_fec) {
        // Golay-decode each 24-bit codeword back into 12 info bits; an
        // uncorrectable (> 3-error) word rejects the header outright.
        uint32_t info = 0;
        int n_words = (d_num_bits + 8 + 11) / 12;
        long di = 0;
        for (int w = 0; w < n_words; w++) {
            uint32_t r = 0;
            for (int c = 0; c < 24 && di < d_header_len; c++, di++) {
                r |= ((uint32_t)(dq[di] & 1)) << c;
            }
            uint16_t m12 = 0;
            if (!golay24_decode(r, m12)) {
                return false;
            }
            info |= ((uint32_t)m12) << (12 * w);
        }
        header_num = info & d_num_mask;
        unsigned char crc_rx = (unsigned char)((info >> d_num_bits) & 0xFFu);
        unsigned char buffer[] = {
            (unsigned char)(fixed_len & 0xFF),
            (unsigned char)((fixed_len >> 8) & 0x0F),
            (unsigned char)( header_num        & 0xFF),
            (unsigned char)((header_num >>  8) & 0xFF),
            (unsigned char)((header_num >> 16) & 0xFF),
        };
        if (crc_rx != d_crc_impl.compute(buffer, sizeof(buffer))) {
            return false;
        }
    } else {
        long k = 0;
        for (int i = 0; i < d_num_bits && k < d_header_len; i += d_bits_per_byte, k++) {
            header_num |= (((uint32_t)dq[k]) & d_mask) << i;
        }
        if (k >= d_header_len) {
            return false;
        }
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
