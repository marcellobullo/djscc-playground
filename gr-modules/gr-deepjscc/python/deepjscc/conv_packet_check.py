#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Georgios Theof.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr
import pmt

class conv_packet_check(gr.basic_block):
    """
    Fast packet checker that inserts zero packets for missing packet numbers.
    Reads 'packet_num' tags and inserts all-zero packets when sequence jumps.
    Works with byte data streams.
    """
    def __init__(self, packet_size=1024, missing_packet_size=None, add_missing_packet_tags=True):
        """
        :param packet_size: Size of each packet (bytes).
        :param missing_packet_size: Size of inserted missing packets (bytes). If None, uses packet_size.
        :param add_missing_packet_tags: Whether to add packet_num tags to inserted zero packets.
        """
        gr.basic_block.__init__(self,
            name="conv_packet_check",
            in_sig=[np.uint8],
            out_sig=[np.uint8])
        
        # Configuration parameters
        self.packet_size = packet_size
        # Determine missing packet size (user-specified or default to packet_size)
        self.missing_packet_size = missing_packet_size if missing_packet_size is not None else packet_size
        self.add_missing_packet_tags = add_missing_packet_tags
        
        # State tracking
        self.expected_packet_num = 0
        self.first_packet = True
        
        # Pre-allocate zero packet for original packet size (used for speed if needed)
        # (Not used for insertion if missing_packet_size != packet_size)
        self.zero_packet = np.zeros(self.packet_size, dtype=np.uint8)
        # Pre-allocate zero packet for missing packet size
        self.missing_zero_packet = np.zeros(self.missing_packet_size, dtype=np.uint8)
        
        print(f"ConvPacketCheck: Initialized with packet_size={packet_size}, "
              f"missing_packet_size={self.missing_packet_size}, "
              f"add_missing_packet_tags={self.add_missing_packet_tags}")

    def handle_padding_info(self, msg):
        """Handle padding information from decoder"""
        try:
            packets_padded_raw = pmt.to_python(msg)
            if isinstance(packets_padded_raw, (int, float)):
                packets_padded = int(packets_padded_raw)
                print(f"\U0001F4E5 RECEIVED PADDING INFO: {packets_padded} packets were padded by decoder")
                
                # Adjust expected packet number by the number of packets that were padded
                if packets_padded > 0:
                    self.expected_packet_num += packets_padded
                    print(f"\U0001F4DD ADJUSTED expected packet number to: {self.expected_packet_num}")
            else:
                print(f"\u274C Invalid padding info type: {type(packets_padded_raw)}")
            
        except Exception as e:
            print(f"\u274C ERROR handling padding info: {e}")

    def forecast(self, noutput_items, ninputs):
        # Request enough input to process at least one packet
        # But allow for expansion in case we need to insert zero packets
        ninput_items_required = [max(1, noutput_items // 2)] * ninputs
        return ninput_items_required

    def general_work(self, input_items, output_items):
        input_data = input_items[0]
        output_data = output_items[0]
        ninput_items = len(input_data)
        max_output_items = len(output_data)
        
        if ninput_items == 0:
            return 0
        
        # Get all packet_num tags from input stream
        tags = self.get_tags_in_window(0, 0, ninput_items)
        packet_tags = [tag for tag in tags if pmt.to_python(tag.key) == "packet_num"]
        
        # Remove duplicate packet tags at the same offset (defensive programming)
        seen_offsets = set()
        unique_packet_tags = []
        for tag in packet_tags:
            if tag.offset not in seen_offsets:
                unique_packet_tags.append(tag)
                seen_offsets.add(tag.offset)
            else:
                print(f"ConvPacketCheck: WARNING - Duplicate packet tag at offset {tag.offset}, value={pmt.to_python(tag.value)}")
        
        packet_tags = unique_packet_tags
        
        # Fast path: no packet tags => copy data and propagate other tags unchanged
        if not packet_tags:
            items_to_copy = min(ninput_items, max_output_items)
            output_data[:items_to_copy] = input_data[:items_to_copy]
            
            # Propagate all tags (no packet_num tags present)
            for tag in tags:
                new_offset = self.nitems_written(0) + (tag.offset - self.nitems_read(0))
                self.add_item_tag(0, new_offset, tag.key, tag.value)
            
            self.consume(0, items_to_copy)
            return items_to_copy
        
        # Process stream with packet checking
        input_consumed = 0
        output_produced = 0
        processed_tag_offsets = set()  # Track which tag offsets we've already processed
        
        # Sort packet_num tags by offset to process in sequence
        packet_tags.sort(key=lambda x: x.offset)
        
        for tag in packet_tags:
            tag_offset = tag.offset - self.nitems_read(0)  # Relative offset in current input buffer
            packet_num_value = pmt.to_python(tag.value)
            
            # Safe type conversion for packet number
            try:
                if isinstance(packet_num_value, (int, float)):
                    current_packet_num = int(packet_num_value)
                else:
                    current_packet_num = 0
            except (TypeError, ValueError):
                current_packet_num = 0
            
            # Copy data before this packet tag (no tags)
            if tag_offset > input_consumed:
                copy_items = min(int(tag_offset - input_consumed), max_output_items - output_produced)
                if copy_items > 0:
                    output_data[output_produced:output_produced + copy_items] = input_data[input_consumed:input_consumed + copy_items]
                    input_consumed += copy_items
                    output_produced += copy_items
            
            # Packet sequence handling: if first packet seen, set expected number
            if self.first_packet:
                self.expected_packet_num = current_packet_num
                self.first_packet = False
                print(f"ConvPacketCheck: First packet #{current_packet_num}")
            
            # Determine how many packets are missing in the sequence
            missing_packets = current_packet_num - self.expected_packet_num
            
            if missing_packets > 0:
                print(f"ConvPacketCheck: Missing {missing_packets} packets! Expected {self.expected_packet_num}, got {current_packet_num}")
                
                # Insert zero-filled packets of specified missing_packet_size for each missing packet
                for missing_num in range(self.expected_packet_num, current_packet_num):
                    if output_produced + self.missing_packet_size <= max_output_items:
                        zero_tag_offset = self.nitems_written(0) + output_produced
                        # Write zeros for missing packet
                        output_data[output_produced:output_produced + self.missing_packet_size] = self.missing_zero_packet
                        
                        # Add tags for the inserted packet if enabled
                        if self.add_missing_packet_tags:
                            # Add packet_num tag
                            packet_num_key = pmt.to_pmt("packet_num")
                            packet_num_val = pmt.to_pmt(missing_num)
                            self.add_item_tag(0, zero_tag_offset, packet_num_key, packet_num_val)
                            
                            # Add packet_len tag (use missing_packet_size)
                            packet_len_key = pmt.to_pmt("packet_len")
                            packet_len_val = pmt.to_pmt(self.missing_packet_size)
                            self.add_item_tag(0, zero_tag_offset, packet_len_key, packet_len_val)
                        
                        output_produced += self.missing_packet_size
                        print(f"ConvPacketCheck: Inserted zero packet #{missing_num} (size={self.missing_packet_size}) at offset {zero_tag_offset}")
                    else:
                        # Insufficient output buffer space; handle remaining in next call
                        break
            
            # Copy the current packet's data from input to output
            packet_start = int(tag_offset)
            packet_end = min(packet_start + self.packet_size, ninput_items)
            packet_length = packet_end - packet_start
            
            if output_produced + packet_length <= max_output_items:
                output_data[output_produced:output_produced + packet_length] = input_data[packet_start:packet_end]
                
                # Add ALL tags that belong to this packet (at this offset) to the new position
                new_tag_offset = self.nitems_written(0) + output_produced
                packet_offset_in_input = tag.offset  # Absolute offset in input stream
                
                # Find all tags at this offset and propagate them
                for input_tag in tags:
                    if input_tag.offset == packet_offset_in_input:
                        self.add_item_tag(0, new_tag_offset, input_tag.key, input_tag.value)
                
                # Mark this offset as processed so we don't duplicate tags later
                processed_tag_offsets.add(packet_offset_in_input)
                
                print(f"ConvPacketCheck: Copied packet #{current_packet_num} at offset {new_tag_offset}")
                
                output_produced += packet_length
                input_consumed = packet_end
                
                # Update expected packet counter for next iteration (only after successfully copying)
                self.expected_packet_num = current_packet_num + 1
        
        # After processing all tagged packets, copy any remaining input data
        remaining_input = ninput_items - input_consumed
        if remaining_input > 0:
            copy_items = min(remaining_input, max_output_items - output_produced)
            if copy_items > 0:
                output_data[output_produced:output_produced + copy_items] = input_data[input_consumed:input_consumed + copy_items]
                output_produced += copy_items
                input_consumed += copy_items
        
        # Propagate any remaining tags for data we've copied (excluding already processed packet offsets)
        for tag in tags:
            tag_offset = tag.offset - self.nitems_read(0)
            # Only propagate tags that:
            # 1. Are for data we actually copied (tag_offset < input_consumed)
            # 2. Haven't already been processed as part of a packet
            if tag_offset < input_consumed and tag.offset not in processed_tag_offsets:
                # Calculate new offset proportionally to account for inserted zeros
                output_offset_ratio = output_produced / input_consumed if input_consumed > 0 else 1
                new_offset = self.nitems_written(0) + int(tag_offset * output_offset_ratio)
                self.add_item_tag(0, new_offset, tag.key, tag.value)
        
        # Consume the input data and return the number of output items produced
        self.consume(0, input_consumed)
        return output_produced
