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
import os
import io
import csv
import threading
import time
import math
from datetime import datetime
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class image_eval(gr.sync_block):
    """
    A GNU Radio block that acts as a file sink for uint8 data.
    Reads bytes from input and writes them to a specified file.
    Similar structure to adjscc_decoder but for conventional compression.
    Automatically extracts image dimensions from the original image(s) for proper reconstruction.
    Supports both single image and folder dataset modes with arbitrary image sizes.
    """
    def __init__(self, filename="output.bin", original_image_path="", csv_folder="/home/gtheofil/JSCC/Data/Results",
                 buffer_size=0, padding_zeros=0, repeat=1, method="qam16_1_2", bandwidth_divisor=6,
                 tx_gain=0.0, rx_gain=0.0, distance=0.0, timeout_ms=1000, packet_size=1024, demo_mode=False):
        gr.sync_block.__init__(self,
            name="image_eval",
            in_sig=[np.uint8],
            out_sig=None)  # No output - this is a sink
        
        # Add message port for sending padding information to packet_check
        self.message_port_register_out(pmt.to_pmt("padding_info"))
        
        self.filename = filename
        self.buffer_size = buffer_size  # Target size for buffering
        self.padding_zeros = padding_zeros  # Number of padding bytes to remove
        self.original_image_path = original_image_path
        self.csv_folder = csv_folder
        self.repeat = repeat
        self.method = method
        self.bandwidth_divisor = bandwidth_divisor
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.distance = distance
        self.timeout_ms = timeout_ms
        self.packet_size = packet_size
        self.demo_mode = demo_mode
        
        # Set max repeats based on repeat value
        self.max_repeats = repeat
        
        # Repeat counter for repeat functionality
        self.repeat_count = 0
        self.max_repeats = repeat
        
        # Extract kodak_id from original_image_path (copied from decoder)
        self.kodak_id = self.extract_kodak_id_from_path(original_image_path)
        
        # Calculate bandwidth from divisor: bandwidth = 1/divisor
        self.bandwidth = f"1_{bandwidth_divisor}_bandwidth"
        
        print(f"Image Eval: Method={self.method}, Kodak ID={self.kodak_id}, Bandwidth={self.bandwidth} (1/{bandwidth_divisor})")
        
        # Image dimensions will be extracted from original image
        self.expected_width = 32  # Default fallback
        self.expected_height = 32  # Default fallback
        self.expected_channels = 3  # Default fallback
        self.file_handle = None
        
        # Timeout mechanism variables (similar to adjscc_decoder)
        self.last_data_time = None
        self.waiting_for_data = False
        self.timeout_timer = None
        self.timeout_flag = False
        self.force_processing = False
        self.force_end_of_stream = False
        
        # Extract image dimensions from original image if provided
        if self.original_image_path and os.path.exists(self.original_image_path):
            self._extract_image_dimensions(self.original_image_path)
        else:
            print(f"⚠️ Original image not found: {self.original_image_path}")
            print("⚠️ Using default dimensions: 32x32x3")
        
        # Buffer management (similar to adjscc_decoder)
        self.buffer = np.array([], dtype=np.uint8)  # Buffer to accumulate bytes
        self.bytes_processed = 0  # Track total bytes processed
        self.output_done = False  # Flag to indicate processing state
        
        # Buffer settings for sync_block (similar to adjscc_decoder)
        self.min_input_items = 0  # Allow work() to be called even without input
        self.set_output_multiple(1)  # Standard output multiple
        self.set_min_noutput_items(0)
        self.set_relative_rate(1.0)
        
        print(f"Image Eval initialized: buffer_size={self.buffer_size}, padding_zeros={self.padding_zeros}")
        print(f"Timeout={self.timeout_ms}ms, TX_gain={self.tx_gain}, RX_gain={self.rx_gain}")
        print(f"Expected image size: {self.expected_width}x{self.expected_height}x{self.expected_channels}")
        
        # CSV logging setup (only if not in demo mode)
        if not self.demo_mode:
            self.csv_log_path = self.setup_csv_logging()
        else:
            self.csv_log_path = None
            print("Demo mode enabled: No CSV logging will be performed")
        
        self._open_file()

    def extract_kodak_id_from_path(self, image_path):
        """Extract kodak ID from image path like /path/to/kodim06.png -> kodim06"""
        if not image_path:
            print("⚠️  Warning: Empty image path, defaulting to kodim01")
            return "kodim01"
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Check if it matches kodim pattern
        if filename.startswith("kodim") and len(filename) == 7:  # kodim + 2 digits
            try:
                # Validate that the last two characters are digits
                int(filename[5:7])
                print(f"Image Eval: Extracted kodak ID '{filename}' from path: {image_path}")
                return filename
            except ValueError:
                pass
        
        print(f"⚠️  Warning: Could not extract valid kodak ID from '{image_path}', defaulting to kodim01")
        return "kodim01"

    def get_structured_image_path(self):
        """Generate structured image path for experiment"""
        # In demo mode, use the original filename directly
        if self.demo_mode:
            # Use the filename parameter as the output path
            # If it doesn't have an extension, add .png
            output_path = self.filename
            if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_path = output_path + '.png'
            return output_path
            
        # Create path: images/{bandwidth}/{method}/{kodak_id}/{tx_gain}dB_tx/rep{repeat:03d}.png
        image_dir = os.path.join(
            self.csv_folder,
            "images",
            self.bandwidth,
            self.method,
            self.kodak_id,
            f"{int(self.tx_gain)}dB_tx"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(image_dir, exist_ok=True)
        
        # Generate filename with repeat number
        if self.repeat > 1:
            filename = f"rep{self.repeat_count + 1:03d}.png"
        else:
            filename = "rep001.png"
        
        return os.path.join(image_dir, filename)

    def send_padding_info(self, packets_padded):
        """Send padding information to packet_check block (similar to adjscc_decoder)"""
        try:
            # Create PMT message with number of packets that were padded
            msg = pmt.to_pmt(int(packets_padded))
            padding_info_port = pmt.to_pmt("padding_info")
            self.message_port_pub(padding_info_port, msg)
            print(f"📤 SENT PADDING INFO: {packets_padded} packets padded")
        except Exception as e:
            print(f"❌ ERROR sending padding info: {e}")

    def _extract_image_dimensions(self, image_path):
        """Extract dimensions from the original image"""
        try:
            # Load the first image to get dimensions
            img = Image.open(image_path)
            img.load()
            
            self.expected_width = img.width
            self.expected_height = img.height
            
            # Determine number of channels
            if img.mode == 'RGB':
                self.expected_channels = 3
            elif img.mode == 'RGBA':
                self.expected_channels = 4
            elif img.mode == 'L':
                self.expected_channels = 1
            else:
                # Convert to RGB and use 3 channels
                self.expected_channels = 3
                print(f"⚠️ Unknown image mode '{img.mode}', assuming RGB (3 channels)")
            
            print(f"✅ Extracted dimensions from {os.path.basename(image_path)}: {self.expected_width}x{self.expected_height}x{self.expected_channels}")
            
        except Exception as e:
            print(f"❌ Error extracting dimensions from {image_path}: {e}")
            print(f"⚠️ Using default dimensions: {self.expected_width}x{self.expected_height}x{self.expected_channels}")
    
    def setup_csv_logging(self):
        """Setup consolidated CSV file for experiment logging"""
        # Create raw CSV directory
        raw_csv_dir = os.path.join(self.csv_folder, "metrics", "raw")
        os.makedirs(raw_csv_dir, exist_ok=True)
        
        # Generate consolidated CSV filename: {method}_{bandwidth}_raw.csv (no kodak_id)
        csv_filename = f"{self.method}_{self.bandwidth}_raw.csv"
        csv_path = os.path.join(raw_csv_dir, csv_filename)
        
        # Create CSV file with experiment headers (no timestamp as requested)
        try:
            # Only create header if file doesn't exist
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        'kodak_id', 'method', 'bandwidth', 'tx_gain', 'rx_gain', 
                        'distance', 'repeat_num', 'psnr', 'ssim', 'status', 'image_path'
                    ])
                print(f"Image Eval: Created new consolidated CSV at {csv_path}")
            else:
                print(f"Image Eval: Using existing consolidated CSV at {csv_path}")
            return csv_path
        except Exception as e:
            print(f"Image Eval ERROR: Failed to create CSV log file: {e}")
            return None
    
    def log_metrics_to_csv(self, psnr_value, ssim_value, status="success", image_path=""):
        """Log PSNR and SSIM metrics to structured CSV file"""
        # Skip logging in demo mode
        if self.demo_mode or not self.csv_log_path:
            return
        
        # Calculate repeat number for logging
        repeat_num = self.repeat_count + 1
        
        # Create relative image path for CSV (remove base directory)
        relative_image_path = image_path.replace(self.csv_folder + "/", "") if image_path.startswith(self.csv_folder) else image_path
        
        try:
            with open(self.csv_log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    self.kodak_id,
                    self.method,
                    self.bandwidth,
                    self.tx_gain,
                    self.rx_gain,
                    self.distance,
                    repeat_num,
                    psnr_value,
                    ssim_value,
                    status,
                    relative_image_path
                ])
        except Exception as e:
            print(f"Image Eval ERROR: Failed to write to CSV log: {e}")

    def reset_for_next_decode(self):
        """Reset state for next decode operation (similar to adjscc_decoder)"""
        self.buffer = np.array([], dtype=np.uint8)
        self.waiting_for_data = False
        self.last_data_time = None
        self.timeout_flag = False
        self.force_processing = False
        self.force_end_of_stream = False
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def timeout_callback(self):
        """Called when timeout occurs - no new packets received for timeout_ms milliseconds (similar to adjscc_decoder)"""
        if self.waiting_for_data and len(self.buffer) > 0:
            print(f"⏰ TIMEOUT after {self.timeout_ms}ms - processing {len(self.buffer)} bytes")
            self.timeout_flag = True
            self.force_processing = True
            # Immediately process the timeout instead of waiting for next work() call
            self._process_timeout_immediately()

    def _open_file(self):
        """Open the output file for writing"""
        try:
            # Always overwrite mode - delete existing file first
            if os.path.exists(self.filename):
                os.remove(self.filename)
            
            # Create completely new file
            self.file_handle = open(self.filename, 'wb')
            self.file_handle.truncate(0)
            self.file_handle.flush()
            
            print(f"Created new file: {self.filename}")
        except Exception as e:
            print(f"Error opening file {self.filename}: {e}")
            self.file_handle = None

    def work(self, input_items, output_items):
        """Process input data with buffering similar to adjscc_decoder"""
        in0 = input_items[0]
        ninput_items = len(in0)

        # Check if timeout flag is set (from timer thread)
        timeout_occurred = self.timeout_flag

        if ninput_items == 0 and self.output_done:
            return -1
        
        # Handle end-of-stream case: no input and buffer has incomplete data
        if ninput_items == 0 and not self.output_done and len(self.buffer) > 0:
            print(f"📄 END OF STREAM: Processing remaining {len(self.buffer)} bytes")
            self._process_timeout_immediately()
            return -1
        
        # Even if no new input, check if we need to process due to timeout
        if ninput_items == 0 and self.force_processing and self.timeout_flag:
            print(f"⚡ TIMEOUT PROCESSING: {len(self.buffer)} bytes")
            self._process_timeout_immediately()
            return 0

        if not self.output_done and ninput_items > 0:
            # Update last data time and start/reset timeout
            self.last_data_time = time.time()
            
            # Add new data to buffer
            self.buffer = np.append(self.buffer, in0)
            self.waiting_for_data = True
            
            # Start timeout timer if not already running
            if self.timeout_timer:
                self.timeout_timer.cancel()
            self.timeout_timer = threading.Timer(self.timeout_ms / 1000.0, self.timeout_callback)
            self.timeout_timer.start()
            
            print(f"Buffer updated: {len(self.buffer)} bytes accumulated")

        # Check if we have enough data OR if timeout has occurred
        should_decode = False
        
        if not self.output_done:
            if self.buffer_size > 0:
                # Calculate total items needed (including padding to be removed)
                total_items_needed = self.buffer_size + self.padding_zeros
                if len(self.buffer) >= total_items_needed:
                    should_decode = True
                    print(f"Buffer size reached: {len(self.buffer)}/{total_items_needed} bytes (including {self.padding_zeros} padding) - processing")
                elif timeout_occurred and len(self.buffer) > 0:
                    should_decode = True
                    print(f"⏰ TIMEOUT: Processing {len(self.buffer)} bytes (expected {total_items_needed})")
            else:
                # No specific buffer size - process on end of stream or when no more input
                if ninput_items == 0 and len(self.buffer) > 0:
                    should_decode = True
                    print(f"End of stream: processing {len(self.buffer)} bytes")

        if should_decode:
            # Stop timeout timer
            if self.timeout_timer:
                self.timeout_timer.cancel()
                self.timeout_timer = None
            
            # Extract the required amount of data
            if self.buffer_size > 0:
                # Use buffer_size + padding_zeros bytes
                total_items_to_read = self.buffer_size + self.padding_zeros
                available_items = len(self.buffer)
                buffered_data_with_padding = self.buffer.copy()
                self.buffer = np.array([], dtype=np.uint8)  # Clear buffer
                
                # Pad with zeros to reach expected size if needed
                if available_items < total_items_to_read:
                    padding_needed = total_items_to_read - available_items
                    padding_zeros_to_add = np.zeros(padding_needed, dtype=np.uint8)
                    buffered_data_with_padding = np.append(buffered_data_with_padding, padding_zeros_to_add)
                    print(f"⚠️ Padded with {padding_needed} zeros (timeout or incomplete data)")
                    
                    # Send padding info
                    packets_padded = padding_needed // self.packet_size + (1 if padding_needed % self.packet_size > 0 else 0)
                    self.send_padding_info(packets_padded)
                else:
                    buffered_data_with_padding = buffered_data_with_padding[:total_items_to_read]
                
                # Remove padding zeros from the end if specified
                if self.padding_zeros > 0 and len(buffered_data_with_padding) >= self.padding_zeros:
                    data_to_process = buffered_data_with_padding[:-self.padding_zeros]
                    print(f"Removed {self.padding_zeros} padding bytes from end")
                else:
                    data_to_process = buffered_data_with_padding
                    if self.padding_zeros > 0:
                        print(f"Warning: Not enough data to remove {self.padding_zeros} padding bytes")
                
                print(f"Processing {len(data_to_process)} bytes (after padding removal)")
            else:
                # Process all available data
                data_to_process = self.buffer.copy()
                self.buffer = np.array([], dtype=np.uint8)  # Clear buffer
                
                # Remove padding zeros from the end if specified
                if self.padding_zeros > 0 and len(data_to_process) >= self.padding_zeros:
                    data_to_process = data_to_process[:-self.padding_zeros]
                    print(f"Removed {self.padding_zeros} padding bytes from end")
                elif self.padding_zeros > 0:
                    print(f"Warning: Not enough data to remove {self.padding_zeros} padding bytes")
                
                print(f"Processing all {len(data_to_process)} bytes (after padding removal)")

            # Process the data
            self._decode_and_save_image(data_to_process, timeout_occurred=timeout_occurred)

        if self.output_done and ninput_items == 0:
            return -1
        
        return ninput_items

    def _process_timeout_immediately(self):
        """Process timeout immediately without waiting for work() call (similar to adjscc_decoder)"""
        if not self.waiting_for_data or len(self.buffer) == 0:
            return
            
        print(f"⚡ IMMEDIATE TIMEOUT PROCESSING: {len(self.buffer)} items")
        
        # Cancel any running timeout timer
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None
        
        # Reset timeout state
        self.waiting_for_data = False
        self.last_data_time = None
        self.timeout_flag = False
        self.force_processing = False
        
        # Calculate expected data size
        if self.buffer_size > 0:
            total_items_needed = self.buffer_size + self.padding_zeros
        else:
            total_items_needed = len(self.buffer)  # Use whatever we have
        
        # Use whatever we have and pad with zeros
        available_items = len(self.buffer)
        buffered_data_with_padding = self.buffer.copy()
        self.buffer = np.array([], dtype=np.uint8)  # Clear buffer
        
        print(f"📊 IMMEDIATE TIMEOUT: Buffer cleared, now has {len(self.buffer)} items (should be 0)")
        
        # Pad with zeros to reach expected size if needed
        if available_items < total_items_needed:
            padding_needed = total_items_needed - available_items
            padding_zeros_to_add = np.zeros(padding_needed, dtype=np.uint8)
            buffered_data_with_padding = np.append(buffered_data_with_padding, padding_zeros_to_add)
            print(f"⚠️ IMMEDIATE: Padded with {padding_needed} zeros")
            
            # Send padding info
            packets_padded = padding_needed // self.packet_size + (1 if padding_needed % self.packet_size > 0 else 0)
            self.send_padding_info(packets_padded)
        else:
            buffered_data_with_padding = buffered_data_with_padding[:total_items_needed]
        
        # Remove padding zeros from end if specified
        if self.padding_zeros > 0 and len(buffered_data_with_padding) >= self.padding_zeros:
            buffered_byte_data = buffered_data_with_padding[:-self.padding_zeros]
            print(f"⚡ IMMEDIATE: Removed {self.padding_zeros} padding bytes from end")
        else:
            buffered_byte_data = buffered_data_with_padding
        
        print(f"⚡ IMMEDIATE PROCESSING: {len(buffered_byte_data)} bytes")
        
        # Process the data (similar to normal work() flow)
        self._decode_and_save_image(buffered_byte_data, timeout_occurred=True)

    def _decode_and_save_image(self, buffered_byte_data, timeout_occurred=False):
        """Decode image data and save result (similar to adjscc_decoder)"""
        try:
            output_path = self.get_structured_image_path()
            
            print(f"🔄 Processing image: {len(buffered_byte_data)} bytes")
            if timeout_occurred:
                print("⏰ Processing due to timeout")
            
            # Process the data (always assume compression)
            success = self._process_buffered_data_internal(buffered_byte_data, output_path)
            
            # Calculate metrics if we have origin image and processing was successful
            if self.original_image_path and success and os.path.exists(self.original_image_path):
                psnr_value, ssim_value = self._calculate_metrics(output_path, self.original_image_path)
                
                # Log to CSV with image path
                self.log_metrics_to_csv(psnr_value, ssim_value, 
                                      "timeout" if timeout_occurred else "success", output_path)
                
                # Print completion message similar to decoder
                status_msg = "TIMEOUT" if timeout_occurred else "NORMAL"
                repeat_info = f"[{self.repeat_count + 1}/{self.max_repeats}]" if self.repeat else "[1/1]"
                print(f"✅ {status_msg} DECODE COMPLETE {repeat_info}: {self.method}/{self.kodak_id}")
                print(f"PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}")
                
            else:
                # Log failure if we expected to have an original image
                if self.original_image_path:
                    self.log_metrics_to_csv(0.0, 0.0, "failed", output_path)
            
            # Handle completion
            self._handle_completion()
            
        except Exception as e:
            print(f"❌ Error in decode and save: {e}")
            if self.csv_log_path:
                self.log_metrics_to_csv(0.0, 0.0, "error", "")

    def stop(self):
        """Clean up when the flowgraph stops"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        return True

    def start(self):
        """Initialize when the flowgraph starts"""
        # Close any existing file handle first
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None
        
        # Clear buffer (using numpy array)
        self.buffer = np.array([], dtype=np.uint8)
        self.output_done = False
        self.bytes_processed = 0
        self.repeat_count = 0
        
        # Force recreation of the file
        self._open_file()
        return True

    def _calculate_metrics(self, output_path, origin_path):
        """Calculate PSNR and SSIM metrics (matching adjscc_decoder exactly)"""
        psnr_value = 0.0
        ssim_value = 0.0
        
        try:
            if not os.path.exists(origin_path):
                print(f"Origin image not found: {origin_path}")
                return psnr_value, ssim_value
            
            if not os.path.exists(output_path):
                print(f"Output image not found: {output_path}")
                return psnr_value, ssim_value
            
            # Load origin image using torch transforms (matching adjscc_decoder)
            transform = transforms.Compose([transforms.ToTensor()])
            origin_img = Image.open(origin_path)
            origin_img.load()
            origin_img_tensor = transform(origin_img).unsqueeze(0)
            
            # Load output image using torch transforms
            output_img = Image.open(output_path)
            output_img.load()
            
            # Check if output is an error placeholder (red image with text)
            output_array = np.array(output_img)
            if self._is_error_placeholder(output_array):
                print("📊 DETECTED ERROR PLACEHOLDER - Using minimal metrics")
                # Return very low quality metrics for error cases
                return 0.0, 0.0
            
            # Ensure both images are the same size
            if origin_img.size != output_img.size:
                print(f"Size mismatch: origin {origin_img.size}, output {output_img.size}")
                output_img = output_img.resize(origin_img.size)
            
            # Convert output to tensor
            output_img_tensor = transform(output_img).unsqueeze(0)
            
            # Calculate SSIM (matching adjscc_decoder exactly)
            ssim_loss = ssim(origin_img_tensor, output_img_tensor)
            ssim_value = ssim_loss.item()
            
            # Calculate PSNR (matching adjscc_decoder exactly)
            origin_img_squeezed = origin_img_tensor.squeeze(0)
            output_img_squeezed = output_img_tensor.squeeze(0)
            vloss = F.mse_loss(origin_img_squeezed, output_img_squeezed)
            
            if vloss > 0:
                psnr = 10 * torch.log10(1 / vloss)
                psnr_value = psnr.item()
            else:
                psnr_value = float('inf')
                
            print(f"📊 METRICS: PSNR={psnr_value:.2f}dB, SSIM={ssim_value:.4f}")
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            
        return psnr_value, ssim_value

    def _is_error_placeholder(self, img_array):
        """Check if the image is an error placeholder (mostly red)"""
        try:
            if len(img_array.shape) != 3:
                return False
                
            # Check if image is predominantly red (error placeholder)
            red_channel = img_array[:, :, 0]
            green_channel = img_array[:, :, 1] 
            blue_channel = img_array[:, :, 2]
            
            # Calculate average values
            avg_red = np.mean(red_channel)
            avg_green = np.mean(green_channel)
            avg_blue = np.mean(blue_channel)
            
            # Error placeholder should be mostly red with low green/blue
            if avg_red > 200 and avg_green < 50 and avg_blue < 50:
                return True
                
            return False
            
        except Exception:
            return False

    def _handle_completion(self):
        """Handle completion logic for repeat/continue (similar to adjscc_decoder)"""
        if self.repeat > 1:
            self.repeat_count += 1
            if self.repeat_count >= self.max_repeats:
                print(f"✅ Image Eval: All {self.max_repeats} repetitions complete")
                self.output_done = True
            else:
                print(f"🔄 Image Eval: Repeat {self.repeat_count + 1}/{self.max_repeats} - resetting for next")
                self.reset_for_next_decode()
        else:
            print("✅ Single mode: Processing complete")
            self.output_done = True

    def _process_buffered_data_internal(self, data, output_path):
        """Internal method to process buffered data and return success status"""
        try:
            # Always assume compression mode - try to decompress the data
            return self._process_compressed_data(data.tobytes(), output_path)
        except Exception as e:
            print(f"❌ Error processing buffered data: {e}")
            return False
    
    def _write_bytes_to_file(self, data_bytes, output_path):
        """Write bytes to file with proper file handling (updated signature)"""
        try:
            # Delete existing file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception as e:
                    print(f"Error deleting file: {e}")
            
            # For "none" compression type, try to interpret as raw image data
            if self.compression_type == "none":
                # Try to reshape raw bytes into image format and save as PNG
                try:
                    expected_size = self.expected_width * self.expected_height * self.expected_channels
                    if len(data_bytes) == expected_size:
                        # Reshape data into image array
                        if self.expected_channels == 3:
                            img_array = np.frombuffer(data_bytes, dtype=np.uint8).reshape(
                                self.expected_height, self.expected_width, 3)
                            img = Image.fromarray(img_array, mode='RGB')
                        elif self.expected_channels == 1:
                            img_array = np.frombuffer(data_bytes, dtype=np.uint8).reshape(
                                self.expected_height, self.expected_width)
                            img = Image.fromarray(img_array, mode='L')
                        else:
                            # Fallback to raw file writing
                            raise ValueError(f"Unsupported channel count: {self.expected_channels}")
                        
                        # Save as PNG
                        img.save(output_path, 'PNG')
                        print(f"Converted raw {self.expected_width}x{self.expected_height}x{self.expected_channels} data to PNG: {output_path}")
                        return True
                        
                    else:
                        print(f"Warning: Data size {len(data_bytes)} doesn't match expected {expected_size} for {self.expected_width}x{self.expected_height}x{self.expected_channels}")
                        # Fall through to raw file writing
                        
                except Exception as img_e:
                    print(f"Failed to convert raw data to image: {img_e}, saving as raw file")
                    # Fall through to raw file writing
            
            # Create raw file (fallback or explicit request)
            with open(output_path, 'wb') as f:
                f.write(data_bytes)
                f.flush()
            
            print(f"Wrote {len(data_bytes)} bytes to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error writing bytes to file: {e}")
            return False

    def _process_compressed_data(self, data_bytes, output_path):
        """Process compressed image data (updated signature)"""
        try:
            # Try to load as image and save directly as PNG
            img_bytes = io.BytesIO(data_bytes)
            img = Image.open(img_bytes)
            
            # Save directly as PNG to the output file
            img.save(output_path, 'PNG')
            print(f"Successfully decompressed {len(data_bytes)} bytes and saved as PNG to {output_path}")
            return True
                
        except Exception as e:
            print(f"PIL failed to decode compressed data ({len(data_bytes)} bytes): {e}")
            print("Compressed data corrupted by channel - attempting recovery methods...")
            
            # Skip saving raw binary data - proceed directly to recovery attempts
            # Recovery Method 1: Try to find JPEG/PNG headers and repair
            recovery_success = self._attempt_image_recovery(data_bytes, output_path)
            
            if not recovery_success:
                # Recovery Method 2: Create a placeholder error image with data info
                self._create_error_placeholder(data_bytes, output_path)
                
            return recovery_success

    def _attempt_image_recovery(self, data_bytes, output_path):
        """Attempt to recover corrupted image data"""
        try:
            # JPEG recovery: Look for JPEG markers
            if self._try_jpeg_recovery(data_bytes, output_path):
                return True
                
            # PNG recovery: Look for PNG signature
            if self._try_png_recovery(data_bytes, output_path):
                return True
                
            # Try ignoring errors and force loading
            if self._try_force_image_load(data_bytes, output_path):
                return True
                
            return False
            
        except Exception as e:
            print(f"Recovery attempt failed: {e}")
            return False

    def _try_jpeg_recovery(self, data_bytes, output_path):
        """Try to recover JPEG data by finding JPEG markers"""
        try:
            # Look for JPEG start marker (0xFFD8)
            jpeg_start = data_bytes.find(b'\xff\xd8')
            if jpeg_start == -1:
                return False
                
            # Look for JPEG end marker (0xFFD9) 
            jpeg_end = data_bytes.rfind(b'\xff\xd9')
            if jpeg_end == -1:
                # No end marker found, try to use all remaining data
                jpeg_data = data_bytes[jpeg_start:]
            else:
                jpeg_data = data_bytes[jpeg_start:jpeg_end+2]
            
            # Try to load the extracted JPEG data
            img_bytes = io.BytesIO(jpeg_data)
            img = Image.open(img_bytes)
            img.save(output_path, 'PNG')
            print(f"✅ JPEG recovery successful! Saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"JPEG recovery failed: {e}")
            return False

    def _try_png_recovery(self, data_bytes, output_path):
        """Try to recover PNG data by finding PNG signature"""
        try:
            # Look for PNG signature (0x89504E47)
            png_signature = b'\x89PNG\r\n\x1a\n'
            png_start = data_bytes.find(png_signature)
            if png_start == -1:
                return False
                
            # Use all data from PNG signature onwards
            png_data = data_bytes[png_start:]
            
            # Try to load the extracted PNG data
            img_bytes = io.BytesIO(png_data)
            img = Image.open(img_bytes)
            img.save(output_path, 'PNG')
            print(f"✅ PNG recovery successful! Saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"PNG recovery failed: {e}")
            return False

    def _try_force_image_load(self, data_bytes, output_path):
        """Try to force load image by ignoring some errors"""
        try:
            # Try with PIL's error tolerance
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            img_bytes = io.BytesIO(data_bytes)
            img = Image.open(img_bytes)
            img.load()  # Force loading
            img.save(output_path, 'PNG')
            print(f"✅ Force loading successful! Saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Force loading failed: {e}")
            return False

    def _create_error_placeholder(self, data_bytes, output_path):
        """Create a placeholder image showing the corruption info"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Use expected dimensions for error placeholder, with minimum size for readability
            width = max(256, self.expected_width)
            height = max(256, self.expected_height)
            img = Image.new('RGB', (width, height), color='red')
            draw = ImageDraw.Draw(img)
            
            # Try to use default font, fallback to basic if not available
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add text information
            text_lines = [
                "IMAGE CORRUPTED",
                f"Expected: {self.expected_width}x{self.expected_height}x{self.expected_channels}",
                f"Data size: {len(data_bytes)} bytes",
                f"First 4 bytes: {data_bytes[:4].hex() if len(data_bytes) >= 4 else 'N/A'}",
                f"Last 4 bytes: {data_bytes[-4:].hex() if len(data_bytes) >= 4 else 'N/A'}",
                "Check transmission quality"
            ]
            
            y = 50
            for line in text_lines:
                draw.text((10, y), line, fill='white', font=font)
                y += 30
            
            # Resize to expected dimensions if they're different
            if width != self.expected_width or height != self.expected_height:
                img = img.resize((self.expected_width, self.expected_height), Image.Resampling.LANCZOS)
            
            # Save the error placeholder
            img.save(output_path, 'PNG')
            print(f"✅ Created error placeholder image: {output_path} ({self.expected_width}x{self.expected_height})")
            
        except Exception as e:
            print(f"Failed to create error placeholder: {e}")

    def _ensure_file_handle(self):
        """Ensure file handle is valid, recreate if necessary"""
        if self.file_handle is None or not os.path.exists(self.filename):
            print(f"File {self.filename} doesn't exist or handle is invalid. Attempting to recreate...")
            self._close_file_handle()
            self._open_file()
            return self.file_handle is not None
            
        return True

    def _close_file_handle(self):
        """Safely close file handle"""
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None

    def reset_for_next_process(self):
        """Reset state for next processing cycle (similar to adjscc_decoder)"""
        self.buffer = np.array([], dtype=np.uint8)
        self.output_done = False
        self.bytes_processed = 0
        print("Reset for next processing cycle")
