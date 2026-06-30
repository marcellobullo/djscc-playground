/* -*- c++ -*- */
/*
 * Copyright 2026 Marcello Bullo
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_DEEPJSCC_PACKET_HEADER_OFDM_ROBUST_H
#define INCLUDED_DEEPJSCC_PACKET_HEADER_OFDM_ROBUST_H

#include <gnuradio/deepjscc/api.h>
#include <gnuradio/digital/packet_header_ofdm.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gr {
namespace deepjscc {

/*!
 * \brief OFDM packet header for a FIXED, known packet length.
 *
 * Variant of packet_header_ofdm_wide for links where the packet length is
 * constant and already known to the receiver (expected_packet_len). The
 * 12-bit length field is therefore fully redundant and is NOT transmitted,
 * and the packet_num field is shrunk from 24 bits to a configurable num_bits.
 * Both reductions cut the on-air header size, removing bits whose corruption
 * could only ever cause a (needless) CRC failure. Header layout:
 *
 *   Bits 0 .. num_bits-1     : packet_num   (num_bits, see below)
 *   Bits num_bits .. +7      : 8-bit CRC
 *   Bits num_bits+8 .. end   : zero-padded out to header_len (never parsed)
 *
 * num_bits trades counter range for robustness: the on-air header is
 * (num_bits + 8) bits, so a packet survives iff all of those are correct.
 *
 * There are two ways to set the packet_num field width:
 *
 *  - expected_number_packets > 0 (preferred): packet_num is transmitted modulo
 *    expected_number_packets, so it is always in [0, expected_number_packets-1].
 *    num_bits is then auto-derived as ceil(log2(expected_number_packets)),
 *    clamped to [1, 24], and the explicit num_bits argument is ignored. The
 *    parser also rejects any header decoding to packet_num >=
 *    expected_number_packets (impossible for a valid frame), a free extra guard
 *    against CRC false-passes that never rejects a good packet.
 *
 *  - expected_number_packets <= 0: the explicit num_bits is used and the counter
 *    wraps at 2^num_bits with no range guard.
 *
 * Pick a range that exceeds the number of packets needing a unique id within one
 * capture; too small a range silently aliases packet_num and breaks any
 * downstream per-frame logging that keys on it.
 *
 * The CRC is still computed over the same 5-byte {len, packet_num} buffer used
 * by packet_header_ofdm_wide (packet_num masked to num_bits; its unused upper
 * bytes are zero on both sides), with the length taken from a fixed value (the
 * formatter's packet_len argument on TX, d_expected_packet_len on RX). This
 * keeps the assumed length cross-checked by the CRC while never putting the
 * length bits on the wire. Bit scrambling and frame-length tag emission match
 * gr::digital::packet_header_ofdm.
 *
 * Because the length is not transmitted, expected_packet_len MUST be set
 * (>= 0) on the parsing (receive) side; otherwise header_parser rejects every
 * header.
 *
 * use_fec adds forward error correction on top of the above. The
 * (packet_num + 8-bit CRC) info word is encoded with the extended binary
 * Golay(24,12,8) code in 12-bit chunks, one BPSK bit per subcarrier. Each
 * codeword corrects any 3 bit errors and detects 4; the CRC and the
 * packet_num >= N range check then validate the corrected word (catching the
 * rare >3-error miscorrection). FEC currently requires bits_per_header_sym == 1
 * (BPSK header) and enough header room for ceil((num_bits+8)/12) 24-bit
 * codewords, i.e. with a 48-bit header symbol num_bits <= 16. Both ends must
 * use the same use_fec / num_bits / expected_number_packets settings.
 */
class DEEPJSCC_API packet_header_ofdm_robust : public gr::digital::packet_header_ofdm {
public:
    typedef std::shared_ptr<packet_header_ofdm_robust> sptr;

    packet_header_ofdm_robust(const std::vector<std::vector<int>>& occupied_carriers,
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
                              bool use_fec);
    ~packet_header_ofdm_robust() override;

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
                     int expected_packet_len = -1,
                     int num_bits = 24,
                     int expected_number_packets = 0,
                     bool use_fec = false);

private:
    uint32_t d_packet_number_robust;
    int d_expected_packet_len;
    int d_num_bits;
    uint32_t d_num_mask;
    int d_expected_number_packets;   // <= 0 disables modulo + range guard
    uint32_t d_counter_modulus;      // wrap point: N (if >0) else 2^num_bits
    bool d_use_fec;                  // extended Golay(24,12) on packet_num+CRC
    bool d_warned_no_expected_len;
};

} // namespace deepjscc
} // namespace gr

#endif /* INCLUDED_DEEPJSCC_PACKET_HEADER_OFDM_ROBUST_H */
