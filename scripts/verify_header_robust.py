#!/usr/bin/env python3
"""Round-trip verification for deepjscc.packet_header_ofdm_robust.

Exercises the *compiled* OOT block (no radio / emulator) through the standard
GNU Radio header generator + parser, exactly as the TX/RX flowgraphs use them:

    generator: digital.packet_headergenerator_bb(<robust formatter>, "packet_len")
    parser:    digital.packet_headerparser_b(<robust formatter>.base())

Two checks:
  1. Correctness  - N packets generated then parsed must round-trip with the
     right packet_len (960), an incrementing packet_num (0..N-1), and the
     expected frame_len (ceil(960 / 48) = 20).
  2. CRC rejection - flipping a single header bit per packet must make the
     parser reject every header (publishes #f instead of a dict), proving the
     CRC still protects the (now 32-bit) header after dropping the length field.

Run inside the conda `demo` env after building/installing gr-deepjscc:
    python scripts/verify_header_robust.py
Exits non-zero on any failure so it can gate a rebuild.
"""

import sys
import time
import math

import numpy as np
import pmt
from gnuradio import gr, blocks, digital
from gnuradio import deepjscc

# --- Header geometry, matched to the TX/RX flowgraphs --------------------------
OCCUPIED_CARRIERS = (
    list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0))
    + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),
)
N_OCC = len(OCCUPIED_CARRIERS[0])          # 48 BPSK subcarriers -> 48 header bits
PACKET_LEN = 960                           # fixed, known to both sides
BITS_PER_PAYLOAD_SYM = 8
N_PACKETS = 20
EXPECTED_FRAME_LEN = math.ceil(PACKET_LEN / N_OCC)   # 20


def make_formatter(num_bits=24, expected_number_packets=0):
    """A fresh robust-header formatter (the generator mutates its counter)."""
    return deepjscc.packet_header_ofdm_robust(
        OCCUPIED_CARRIERS,
        n_syms=1,
        len_tag_key="packet_len",
        frame_len_tag_key="frame_len",
        num_tag_key="packet_num",
        bits_per_header_sym=1,                 # BPSK
        bits_per_payload_sym=BITS_PER_PAYLOAD_SYM,
        scramble_header=True,
        expected_packet_len=PACKET_LEN,
        num_bits=num_bits,
        expected_number_packets=expected_number_packets,
    )


class flip_bit(gr.sync_block):
    """Flip header bit index `pos` within every `period`-byte header block."""

    def __init__(self, period=N_OCC, pos=0):
        gr.sync_block.__init__(
            self, name="flip_bit",
            in_sig=[np.uint8], out_sig=[np.uint8])
        self.period = int(period)
        self.pos = int(pos)

    def work(self, input_items, output_items):
        out = output_items[0]
        out[:] = input_items[0]
        start = self.nitems_read(0)
        n = len(out)
        # Indices (absolute) where (offset % period) == pos.
        first = (self.pos - (start % self.period)) % self.period
        idx = np.arange(first, n, self.period)
        out[idx] ^= 1
        return n


def run_roundtrip(corrupt, num_bits=24, n_packets=N_PACKETS,
                  expected_number_packets=0):
    """Generate n_packets headers, optionally corrupt one bit each, parse them.

    Returns the list of parsed PMT messages from the header parser.
    """
    tb = gr.top_block()

    data = [0] * (n_packets * PACKET_LEN)
    src = blocks.vector_source_b(data, repeat=False)
    s2ts = blocks.stream_to_tagged_stream(
        gr.sizeof_char, 1, PACKET_LEN, "packet_len")
    hdrgen = digital.packet_headergenerator_bb(
        make_formatter(num_bits, expected_number_packets), "packet_len")
    parser_fmt = make_formatter(num_bits, expected_number_packets)  # keep alive
    parser = digital.packet_headerparser_b(parser_fmt.base())
    sink = blocks.message_debug()

    tb.connect(src, s2ts, hdrgen)
    if corrupt:
        flip = flip_bit(period=N_OCC, pos=0)
        tb.connect(hdrgen, flip, parser)
    else:
        tb.connect(hdrgen, parser)
    tb.msg_connect((parser, "header_data"), (sink, "store"))

    tb.run()
    time.sleep(0.2)   # let async header_data messages drain into the sink

    return [sink.get_message(i) for i in range(sink.num_messages())]


def parse_msg(msg):
    """Return (packet_len, packet_num, frame_len) or None if msg is a failure."""
    if not pmt.is_dict(msg):
        return None

    def get(key, default=None):
        v = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
        return pmt.to_python(v) if not pmt.eq(v, pmt.PMT_NIL) else default

    return (get("packet_len"), get("packet_num"), get("frame_len"))


def check_roundtrip(num_bits):
    """Clean round-trip + 1-bit-corruption rejection for a given num_bits."""
    ok = True
    print(f"== num_bits={num_bits}: clean round-trip ==")
    msgs = run_roundtrip(corrupt=False, num_bits=num_bits)
    parsed = [parse_msg(m) for m in msgs]
    valid = [p for p in parsed if p is not None]
    print(f"  generated={N_PACKETS}  parsed_ok={len(valid)}  "
          f"rejected={len(parsed) - len(valid)}")
    if len(valid) != N_PACKETS:
        print(f"  FAIL: expected {N_PACKETS} valid headers, got {len(valid)}")
        ok = False
    else:
        for i, (plen, pnum, flen) in enumerate(valid):
            if plen != PACKET_LEN or pnum != i or flen != EXPECTED_FRAME_LEN:
                print(f"  FAIL: packet {i}: packet_len={plen} "
                      f"packet_num={pnum} frame_len={flen} "
                      f"(want {PACKET_LEN}/{i}/{EXPECTED_FRAME_LEN})")
                ok = False
        if ok:
            print(f"  PASS: all packet_len={PACKET_LEN}, "
                  f"packet_num=0..{N_PACKETS-1}, frame_len={EXPECTED_FRAME_LEN}")

    print(f"== num_bits={num_bits}: 1-bit corruption rejection ==")
    msgs = run_roundtrip(corrupt=True, num_bits=num_bits)
    parsed = [parse_msg(m) for m in msgs]
    valid = [p for p in parsed if p is not None]
    print(f"  corrupted={N_PACKETS}  parsed_ok={len(valid)}  "
          f"rejected={len(parsed) - len(valid)}")
    if len(valid) != 0:
        print(f"  FAIL: {len(valid)} corrupted headers passed CRC (expected 0)")
        ok = False
    else:
        print("  PASS: every corrupted header was rejected by the CRC")
    return ok


def check_wrap(num_bits=8):
    """At num_bits, packet_num must wrap modulo 2**num_bits."""
    period = 2 ** num_bits
    n = period + 32                 # force at least one wrap past 2**num_bits
    print(f"== num_bits={num_bits}: counter wrap (mod {period}) ==")
    msgs = run_roundtrip(corrupt=False, num_bits=num_bits, n_packets=n)
    valid = [p for p in (parse_msg(m) for m in msgs) if p is not None]
    ok = len(valid) == n
    if not ok:
        print(f"  FAIL: expected {n} valid headers, got {len(valid)}")
    bad = [(i, pnum) for i, (_, pnum, _) in enumerate(valid)
           if pnum != i % period]
    if bad:
        ok = False
        print(f"  FAIL: {len(bad)} packet_num != i % {period}, e.g. {bad[:3]}")
    if ok:
        print(f"  PASS: {n} headers, packet_num wraps {period-1}->0 as expected")
    return ok


def check_expected_number(n_expected=205):
    """expected_number_packets mode: auto-sized field, packet_num wraps mod N."""
    n = n_expected + 32           # force a wrap past N
    auto_bits = max(1, (n_expected - 1).bit_length())
    print(f"== expected_number_packets={n_expected}: auto field + mod-N wrap ==")
    msgs = run_roundtrip(corrupt=False, n_packets=n,
                         expected_number_packets=n_expected)
    valid = [p for p in (parse_msg(m) for m in msgs) if p is not None]
    ok = len(valid) == n
    if not ok:
        print(f"  FAIL: expected {n} valid headers, got {len(valid)}")
    bad = [(i, pnum) for i, (_, pnum, _) in enumerate(valid)
           if pnum != i % n_expected]
    if bad:
        ok = False
        print(f"  FAIL: {len(bad)} packet_num != i % {n_expected}, e.g. {bad[:3]}")
    if ok:
        print(f"  PASS: {n} headers, packet_num in [0,{n_expected-1}] wraps "
              f"{n_expected-1}->0 (auto num_bits={auto_bits})")
    return ok


def main():
    ok = True
    # num_bits=24 keeps the default/backward-compatible behavior; num_bits=8 is
    # the shrunk field for single-image (<=256 packet) transfers.
    for nb in (24, 8):
        ok &= check_roundtrip(nb)
    ok &= check_wrap(8)
    # Preferred path: derive the field from the expected packet count (205).
    ok &= check_expected_number(205)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
