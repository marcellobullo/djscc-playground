#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Georgios Theofilakos.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import pmt
import numpy as np
from gnuradio import gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import os
import csv
import time
import glob
import threading
from datetime import datetime
from compressai.layers import GDN

def get_image_files(folder_path):
    """Get all image files from a folder."""
    if not os.path.isdir(folder_path):
        # If it's a file, return it as a single-item list
        if os.path.isfile(folder_path):
            return [folder_path]
        else:
            return []
    
    # Common image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern_upper = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    # Sort for consistent ordering
    image_files.sort()
    return image_files

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
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

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


# Attention Decoder classes (from decoder.py)
class FL_De_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,out_padding,activation=None):
        super(FL_De_Module, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,output_padding=out_padding)
        self.GDN = GDN(out_channels)

        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_deconv1=self.deconv1(inputs)
        out_bn=self.GDN(out_deconv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out

class AL_CH_Module(nn.Module):
    def __init__(self,channel_size):
        super(AL_CH_Module, self).__init__()
        self.Ave_Pooling = nn.AdaptiveAvgPool2d(1)
        self.FC_1 = nn.Linear(channel_size+1,channel_size//16)
        self.FC_2 = nn.Linear(channel_size//16,channel_size)

    def forward(self, inputs,attention):
        b=inputs.shape[0]
        out_pooling=self.Ave_Pooling(inputs).view(b,-1)
        b=inputs.shape[0]
        c=inputs.shape[1]
        in_fc=torch.cat((attention,out_pooling),dim=1).float()
        out_fc_1=self.FC_1(in_fc)
        out_fc_1_relu=torch.nn.functional.relu(out_fc_1)
        out_fc_2=self.FC_2(out_fc_1_relu)
        out_fc_2_sig=torch.sigmoid(out_fc_2).view(b,c,1,1)
        out=out_fc_2_sig*inputs
        return out

class AL_De_Module(nn.Module):
    def __init__(self,channel_in_size):
        super(AL_De_Module, self).__init__()
        self.Channel_attention = AL_CH_Module(channel_in_size)

    def forward(self, inputs,attention):
        b=inputs.shape[0]
        attention=attention.view(b,-1)
        c_out=self.Channel_attention(inputs,attention)
        out=c_out
        return out

class Attention_Decoder(nn.Module):
    def __init__(self, args):
        super(Attention_Decoder, self).__init__()
        self.FL_De_Module_1 = FL_De_Module(args.tcn, 256, 5, stride=1, padding=2, out_padding=0, activation='prelu')
        self.AL_De_module_1 = AL_De_Module(256)
        self.FL_De_Module_2 = FL_De_Module(256, 256, 5, stride=1, padding=2, out_padding=0, activation='prelu')
        self.AL_De_module_2 = AL_De_Module(256)
        self.FL_De_Module_3 = FL_De_Module(256, 256, 5, stride=1, padding=2, out_padding=0, activation='prelu')
        self.AL_De_module_3 = AL_De_Module(256)
        self.FL_De_Module_4 = FL_De_Module(256, 256, 5, stride=2, padding=2, out_padding=1, activation='prelu')
        self.AL_De_module_4 = AL_De_Module(256)
        self.FL_De_Module_5 = FL_De_Module(256, 3, 9, stride=2, padding=4, out_padding=1, activation='sigmoid')

    def load_pretrained_weights(self, checkpoint_path='/home/gtheofil/JSCC/Adjscc/ADJSCC/ckpts/AWGN_rate_21_AD_JSCC_SNR_random.pth'):
        """Load decoder weights from the full model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            full_model_state = checkpoint["net"]
            
            decoder_state = {}
            for key, value in full_model_state.items():
                if key.startswith('attention_decoder.'):
                    decoder_key = key.replace('attention_decoder.', '')
                    decoder_state[decoder_key] = value
            
            self.load_state_dict(decoder_state, strict=False)
            # print(f"✅ Successfully loaded decoder weights from: {checkpoint_path}")
            print(f"📊 Loaded {len(decoder_state)} weight tensors")
            
            return True
        except Exception as e:
            print(f"❌ Error loading decoder weights: {e}")
            return False

    def forward(self, x, attention):
        decoded_1_out = self.FL_De_Module_1(x)
        attention_decoder_1_out = self.AL_De_module_1(decoded_1_out, attention)
        decoded_2_out = self.FL_De_Module_2(attention_decoder_1_out)
        attention_decoder_2_out = self.AL_De_module_2(decoded_2_out, attention)
        decoded_3_out = self.FL_De_Module_3(attention_decoder_2_out)
        attention_decoder_3_out = self.AL_De_module_3(decoded_3_out, attention)
        decoded_4_out = self.FL_De_Module_4(attention_decoder_3_out)
        attention_decoder_4_out = self.AL_De_module_4(decoded_4_out, attention)
        decoded_5_out = self.FL_De_Module_5(attention_decoder_4_out)

        return decoded_5_out


# Args class to hold required parameters
class Args:
    def __init__(self, tcn=21):
        self.tcn = tcn
        self.input_snr_min = 0
        self.input_snr_max = 20
        self.fading_flg = 0
        self.model = 'AD_JSCC'

class snr_adjscc_decoder(gr.basic_block):
    """
    Attention-based Deep JSCC Decoder Block for GNU Radio
    """
    def __init__(self,
                 model_path,
                 img_path,
                 origin_img_path,
                 tcn=21,
                 snr_db=10,
                 repeat=1,
                 tx_gain=0.0,
                 rx_gain=0.0,
                 distance=0.0,
                 csv_folder="/home/gtheofil/JSCC/Data/Results",
                 padding_zeros=0,
                 timeout_ms=1000,
                 packet_size=1024,
                 demo_mode=False
                 ):
        gr.basic_block.__init__(self,
            name="snr_adjscc_decoder",
            in_sig=[np.complex64, (np.complex64, 64)],  # Complex input + OFDM symbols (64 subcarriers)
            out_sig=None)
        
        # Add message port for sending padding information to packet_check
        self.message_port_register_out(pmt.to_pmt("padding_info"))
        
        # Parameters
        self.model_path = model_path
        self.img_path = img_path
        self.origin_img_path = origin_img_path
        self.tcn = tcn
        self.snr_db = snr_db
        self.repeat = repeat
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.distance = distance
        self.csv_folder = csv_folder
        self.padding_zeros = padding_zeros
        self.timeout_ms = timeout_ms
        self.packet_size = packet_size
        self.demo_mode = demo_mode
        
        # Get all image files from the folder/path (same as encoder)
        self.image_files = get_image_files(self.origin_img_path)
        if not self.image_files:
            print(f"Decoder ERROR: No image files found in {self.origin_img_path}")
            # Use the original path as fallback
            self.image_files = [self.origin_img_path]
        else:
            print(f"Decoder: Found {len(self.image_files)} images to expect")
            print(f"Decoder: Each image will be received {self.repeat} times")
        
        # Image processing state
        self.current_image_index = 0  # Index of current image being processed
        self.current_image_path = None  # Path to current image
        
        # Initialize current image path
        if self.image_files:
            self.current_image_path = self.image_files[0]
        
        # Extract image dimensions from first original image
        self.channel = 3  # Always RGB
        self.height, self.width = self._extract_image_dimensions(self.current_image_path or origin_img_path)
        
        # Set max repeats based on repeat value
        self.max_repeats = repeat
        
        # Hardcoded method name - modify this as needed
        self.method_name = "adjscc_feedback"
        
        # Extract kodak_id from current image path (will be updated as images advance)
        initial_kodak_id = self.extract_kodak_id_from_path(self.current_image_path or origin_img_path)
        self.kodak_id = initial_kodak_id
        print(f"Decoder: Initial kodak_id set to '{self.kodak_id}' from first image")
        
        # Calculate bandwidth automatically from tcn
        # Formula: int(tcn/4/4/2/3) -> determines 1/6 or 1/12
        bandwidth_ratio = tcn/96
        if bandwidth_ratio == 1/6:
            self.bandwidth = "1_6_bandwidth"
        elif bandwidth_ratio == 1/12:
            self.bandwidth = "1_12_bandwidth" 
        else:
            # Default fallback
            self.bandwidth = "1_6_bandwidth"
            print(f"⚠️  Warning: Calculated bandwidth ratio {bandwidth_ratio} not recognized, defaulting to 1_6_bandwidth")
        
        print(f"Decoder: Method={self.method_name}, Kodak ID={self.kodak_id}, Bandwidth={self.bandwidth} (from tcn={tcn})")
        
        # Timeout mechanism variables
        self.last_data_time = None  # Track when we last received data
        self.waiting_for_data = False
        self.timeout_timer = None
        self.timeout_flag = False
        self.force_processing = False
        self.force_end_of_stream = False
        
        # Repeat counter for repeat functionality
        self.repeat_count = 0
        self.max_repeats = repeat

        # SNR correction factor (hardcoded)
        self.correction = 4

        # Calculate expected input size using tcn directly  
        # The encoder outputs tcn * (height//4) * (width//4) real values
        # When converted to complex, we get half that number of complex values
        # The encoder then adds padding_zeros real values as additional complex values (NOT divided by 2)
        # So total complex items = (tcn * h/4 * w/4)/2 + padding_zeros
        base_complex_items = ((self.tcn * (self.height // 4) * (self.width // 4)) // 2)
        self.padding_complex_items = self.padding_zeros  # Padding is sent as-is, not divided by 2
        self.expected_input_complex_items = base_complex_items + self.padding_complex_items

        # Calculate expected number of frames based on complex items
        # Formula: expected_frames = expected_input_complex_items / 20 / 48
        # This assumes each frame contributes 20*48 = 960 complex items
        self.expected_frames = self.expected_input_complex_items // (20 * 48)
        
        print(f"Decoder Initialized: tcn={self.tcn}, Expected complex input items={self.expected_input_complex_items} (base={base_complex_items} + padding_complex={self.padding_complex_items}), Expected frames={self.expected_frames}, Padding complex to remove={self.padding_complex_items}, Last-packet timeout={self.timeout_ms}ms")
        # print(f"Decoder: This corresponds to {self.tcn * (self.height // 4) * (self.width // 4)} real values from encoder")
        
        self.buffer = np.array([], dtype=np.complex64)
        self.ofdm_buffer = []  # Buffer for OFDM symbols (64-element arrays)
        self.output_done = False
        
        # Initialize SNR estimation
        self.initialize_snr_estimation()
        
        # Initialize the new Attention Decoder
        args = Args(tcn=self.tcn)
        self.decoder_model = Attention_Decoder(args)
        
        # Load pre-trained weights
        weights_loaded = self.decoder_model.load_pretrained_weights(self.model_path)
        if not weights_loaded:
            print("⚠️ Warning: Failed to load pre-trained weights")
        
        self.decoder_model.eval()
        
        # Set up for basic_block - allow calling general_work even without input
        self.set_relative_rate(1.0)
        
        # CSV logging setup (only if not in demo mode)
        if not self.demo_mode:
            self.csv_log_path = self.setup_csv_logging()
        else:
            self.csv_log_path = None
            print("Demo mode enabled: No CSV logging will be performed")

    def compute_power_domain_snr_average(self, snr_values_db):
        """
        Compute power-domain average of SNR values in dB.
        Formula: SNR_global ≈ 10*log10(1/F * sum(10^(SNR_f_dB/10)))
        
        Args:
            snr_values_db: List or array of SNR values in dB
            
        Returns:
            float: Power-domain averaged SNR in dB
        """
        if len(snr_values_db) == 0:
            return self.snr_db  # Fallback to default
        
        # Convert dB to linear domain and compute average
        # Handle potential negative SNRs and zeros by using maximum with small epsilon
        linear_snrs = []
        for snr_db in snr_values_db:
            if snr_db <= 0:
                # For zero or negative SNR, use very small linear value
                linear_snrs.append(1e-10)  # ~-100 dB
            else:
                linear_snrs.append(10**(snr_db / 10.0))
        
        linear_average = np.mean(linear_snrs)
        
        # Convert back to dB, handle edge case where average might be very small
        if linear_average <= 1e-10:
            snr_average_db = -100.0  # Very low SNR floor
        else:
            snr_average_db = 10.0 * np.log10(linear_average)
        
        # print(f"🎯 POWER-DOMAIN CALC: {len(snr_values_db)} SNRs -> linear avg {linear_average:.6e} -> {snr_average_db:.2f} dB")
        
        return snr_average_db

    def initialize_snr_estimation(self):
        """Initialize SNR estimation parameters using hardcoded OFDM configuration"""
        # Hardcoded carrier allocation matching OFDM flowgraph
        occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + 
                           list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)))
        pilot_carriers = [-21, -7, 7, 21]
        
        # Known pilot symbols pattern
        self.pilot_symbols = np.array([1, 1, 1, -1], dtype=np.complex64)
        
        # Convert to indices for 64-point FFT
        N = 64  # FFT size
        self.occupied_indices = [(k + N) % N for k in sorted(occupied_carriers)]
        self.pilot_indices = [(k + N) % N for k in sorted(pilot_carriers)]
        self.data_indices = sorted(set(self.occupied_indices) - set(self.pilot_indices))
        
        # Frame processing parameters
        self.frame_size = 20  # Process 20 symbols per frame
        self.symbols_in_current_frame = 0
        
        # Buffers for current frame data
        self.frame_pilot_data_normalized = []
        self.frame_payload_data_normalized = []
        self.current_frame_snrs = []

    def calculate_alpha_m(self, ofdm_symbol):
        """Calculate least-squares gain/phase correction α_m"""
        pilot_values = [ofdm_symbol[idx] for idx in self.pilot_indices]
        
        if len(pilot_values) != len(self.pilot_symbols):
            return None
            
        # Calculate numerator: Σ_k∈P Z_{k,m} * S_{k,m}*
        numerator = 0.0
        for k in range(len(self.pilot_symbols)):
            Z_k = pilot_values[k]
            S_k_conj = np.conj(self.pilot_symbols[k])
            numerator += Z_k * S_k_conj
            
        # Calculate denominator: Σ_k∈P |S_{k,m}|²
        denominator = 0.0
        for k in range(len(self.pilot_symbols)):
            denominator += np.abs(self.pilot_symbols[k])**2
            
        if denominator > 1e-12:
            alpha_m = numerator / denominator
            return alpha_m
        else:
            return None

    def calculate_symbol_energy(self, normalized_data_symbols):
        """Calculate symbol energy from normalized data tones"""
        if len(normalized_data_symbols) == 0:
            return None
            
        energy_sum = 0.0
        total_symbols = 0
        
        for data_symbol_vector in normalized_data_symbols:
            for symbol in data_symbol_vector:
                energy_sum += np.abs(symbol)**2
                total_symbols += 1
        
        if total_symbols > 0:
            symbol_energy = energy_sum / total_symbols
            return symbol_energy
        else:
            return None

    def calculate_noise_variance(self, normalized_pilot_data):
        """Calculate noise variance from normalized pilots"""
        if len(normalized_pilot_data) == 0:
            return None
            
        pilot_errors_squared = 0.0
        total_pilot_comparisons = 0
        
        for pilot_data_vector in normalized_pilot_data:
            if len(pilot_data_vector) >= len(self.pilot_symbols):
                for k in range(len(self.pilot_symbols)):
                    expected_pilot = self.pilot_symbols[k]
                    normalized_received_pilot = pilot_data_vector[k]
                    
                    error_squared = np.abs(normalized_received_pilot - expected_pilot)**2
                    pilot_errors_squared += error_squared
                    total_pilot_comparisons += 1
        
        if total_pilot_comparisons > 0:
            noise_variance = pilot_errors_squared / total_pilot_comparisons
            return noise_variance
        else:
            return None

    def process_ofdm_symbol(self, ofdm_symbol):
        """Process single OFDM symbol and add to frame buffers"""
        # Calculate least-squares gain/phase correction α_m
        alpha_m = self.calculate_alpha_m(ofdm_symbol)
        
        if alpha_m is not None:
            # Normalize the whole symbol with α_m
            normalized_symbol = ofdm_symbol / alpha_m
            
            # Extract pilots and data from normalized symbol
            normalized_pilots = [normalized_symbol[idx] for idx in self.pilot_indices]
            normalized_data = [normalized_symbol[idx] for idx in self.data_indices]
        else:
            # If we can't calculate α_m, use raw data
            normalized_pilots = [ofdm_symbol[idx] for idx in self.pilot_indices]
            normalized_data = [ofdm_symbol[idx] for idx in self.data_indices]
        
        # Add to current frame buffers
        self.frame_pilot_data_normalized.append(normalized_pilots)
        self.frame_payload_data_normalized.append(normalized_data)
        self.symbols_in_current_frame += 1
        
        # Check if we've completed a frame (20 symbols)
        if self.symbols_in_current_frame >= self.frame_size:
            # Calculate SNR for this frame
            noise_var = self.calculate_noise_variance(self.frame_pilot_data_normalized)
            symbol_energy = self.calculate_symbol_energy(self.frame_payload_data_normalized)
            
            if noise_var is not None and symbol_energy is not None and noise_var > 1e-12:
                snr_db = 10 * np.log10(symbol_energy / noise_var)
                snr_db = np.clip(snr_db, -50, 50)  # Limit to reasonable range
                self.current_frame_snrs.append(snr_db)
            
            # Reset frame buffers
            self.frame_pilot_data_normalized = []
            self.frame_payload_data_normalized = []
            self.symbols_in_current_frame = 0

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
                print(f"Decoder: Extracted kodak ID '{filename}' from path: {image_path}")
                return filename
            except ValueError:
                pass
        
        print(f"⚠️  Warning: Could not extract valid kodak ID from '{image_path}', defaulting to kodim01")
        return "kodim01"
    
    def get_current_image_path(self):
        """Get the current image path to process."""
        if self.current_image_index < len(self.image_files):
            return self.image_files[self.current_image_index]
        return None
    
    def advance_to_next_image(self):
        """Advance to the next image in the sequence."""
        self.current_image_index += 1
        self.repeat_count = 0  # Reset repeat counter for new image
        if self.current_image_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_image_index]
            # Update kodak_id for the new image
            old_kodak_id = self.kodak_id
            self.kodak_id = self.extract_kodak_id_from_path(self.current_image_path)
            print(f"Decoder: Advancing to image {self.current_image_index + 1}/{len(self.image_files)}: {os.path.basename(self.current_image_path)}")
            print(f"Decoder: Kodak ID updated from '{old_kodak_id}' to '{self.kodak_id}'")
            return True
        else:
            print("Decoder: All images processed!")
            return False

    def _extract_image_dimensions(self, image_path):
        """Extract height and width from the original image"""
        try:
            if not image_path or not os.path.exists(image_path):
                print(f"⚠️ Warning: Image path not found '{image_path}', using default dimensions 32x32")
                return 32, 32
            
            # Load image to get dimensions
            img = Image.open(image_path)
            img.load()
            
            # PIL returns (width, height), but we need (height, width) for consistency
            width, height = img.size
            print(f"✅ Extracted dimensions from {os.path.basename(image_path)}: {height}x{width}")
            return height, width
            
        except Exception as e:
            print(f"❌ Error extracting dimensions from {image_path}: {e}")
            print("⚠️ Using default dimensions: 32x32")
            return 32, 32

    def get_structured_image_path(self):
        """Generate structured image path for experiment"""
        # In demo mode, use the original img_path directly
        if self.demo_mode:
            return self.img_path
            
        # Create path: images/{bandwidth}/{method}/{kodak_id}/{tx_gain}dB_tx/rep{repeat:03d}.png
        image_dir = os.path.join(
            self.csv_folder,
            "images",
            self.bandwidth,
            self.method_name,
            self.kodak_id,
            f"{int(self.tx_gain)}dB_tx"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(image_dir, exist_ok=True)
        
        # Generate filename with repeat number
        if self.repeat:
            filename = f"rep{self.repeat_count + 1:03d}.png"
        else:
            filename = "rep001.png"
        
        return os.path.join(image_dir, filename)

    def forecast(self, noutput_items, ninputs=None):
        # if ninputs is an int (sync_block style), just return a list
        if not isinstance(ninputs, list):
            return [1, 0]             # 1 complex, 0 OFDM symbols required minimum
        # otherwise (basic_block style) mutate the list in place
        ninputs[0] = 1  # at least one complex sample to schedule
        ninputs[1] = 0  # zero OFDM symbols required minimum

    def send_padding_info(self, packets_padded):
        """Send padding information to packet_check block"""
        try:
            # Create PMT message with number of packets that were padded
            msg = pmt.to_pmt(int(packets_padded))
            padding_info_port = pmt.to_pmt("padding_info")
            self.message_port_pub(padding_info_port, msg)
            print(f"📤 SENT PADDING INFO: {packets_padded} packets padded")
        except Exception as e:
            print(f"❌ ERROR sending padding info: {e}")

    def setup_csv_logging(self):
        """Setup consolidated CSV file for experiment logging"""
        # Create raw CSV directory
        raw_csv_dir = os.path.join(self.csv_folder, "metrics", "raw")
        os.makedirs(raw_csv_dir, exist_ok=True)
        
        # Generate consolidated CSV filename: {method}_{bandwidth}_raw.csv (no kodak_id)
        csv_filename = f"{self.method_name}_{self.bandwidth}_raw.csv"
        csv_path = os.path.join(raw_csv_dir, csv_filename)
        
        # Create CSV file with experiment headers (no timestamp as requested)
        try:
            # Only create header if file doesn't exist
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        'kodak_id', 'method', 'bandwidth', 'tx_gain', 'rx_gain', 
                        'distance', 'repeat_num', 'psnr', 'ssim', 'estimated_snr', 'status', 'image_path'
                    ])
                print(f"Decoder: Created new consolidated CSV at {csv_path}")
            else:
                print(f"Decoder: Using existing consolidated CSV at {csv_path}")
            return csv_path
        except Exception as e:
            print(f"Decoder ERROR: Failed to create CSV log file: {e}")
            return None
    
    def log_metrics_to_csv(self, psnr_value, ssim_value, estimated_snr, status="success", image_path=""):
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
                    self.method_name,
                    self.bandwidth,
                    self.tx_gain,
                    self.rx_gain,
                    self.distance,
                    repeat_num,
                    psnr_value,
                    ssim_value,
                    estimated_snr,
                    status,
                    relative_image_path
                ])
        except Exception as e:
            print(f"Decoder ERROR: Failed to write to CSV log: {e}")


    def reset_for_next_decode(self):
        """Reset state for next decode operation"""
        self.buffer = np.array([], dtype=np.complex64)
        self.ofdm_buffer = []  # Also reset OFDM buffer
        self.current_frame_snrs = []  # Reset frame SNRs
        self.waiting_for_data = False
        self.last_data_time = None
        self.timeout_flag = False
        self.force_processing = False
        self.force_end_of_stream = False
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def timeout_callback(self):
        """Called when timeout occurs - no new packets received for timeout_ms milliseconds"""
        if self.waiting_for_data and len(self.buffer) > 0:
            print(f"⏰ LAST PACKET TIMEOUT: No new packets for {self.timeout_ms}ms, {len(self.buffer)} items in buffer")
            
            # # Print buffer contents for debugging
            # print(f"🔍 TIMEOUT BUFFER ANALYSIS:")
            # print(f"   Buffer size: {len(self.buffer)}")
            # print(f"   Expected size: {self.expected_input_complex_items}")
            self.timeout_flag = True
            self.force_processing = True
            # Force work() to be called by creating a short-lived timer that injects minimal data
            self._force_work_call()

    def _force_work_call(self):
        """Force work() to be called by triggering GNU Radio scheduler"""
        # Unfortunately, there's no direct way to force work() call in sync_block
        # The timeout will be processed when the next data arrives
        # For now, we'll process timeout directly in this callback
        if self.timeout_flag and len(self.buffer) > 0:
            print(f"🚀 FORCING IMMEDIATE TIMEOUT PROCESSING with {len(self.buffer)} items")
            self._process_timeout_immediately()

    def _process_timeout_immediately(self):
        """Process timeout immediately without waiting for work() call"""
        if not self.waiting_for_data or len(self.buffer) == 0:
            return
            
        print(f"⚡ IMMEDIATE TIMEOUT PROCESSING: {len(self.buffer)} items")
        
        # Print detailed buffer analysis for timeout case
        # print(f"🔍 IMMEDIATE TIMEOUT BUFFER ANALYSIS:")
        # print(f"   Buffer size BEFORE processing: {len(self.buffer)}")
        # print(f"   Expected size: {self.expected_input_complex_items}")
        # Reset timeout state
        self.waiting_for_data = False
        self.last_data_time = None
        self.timeout_flag = False
        self.force_processing = False
        
        # Calculate expected data size
        total_items_needed = self.expected_input_complex_items
        
        # Use whatever we have and pad with zeros
        available_items = len(self.buffer)
        buffered_data_with_padding = self.buffer.copy()
        self.buffer = np.array([], dtype=np.complex64)  # Clear buffer
        
        print(f"📊 IMMEDIATE TIMEOUT: Buffer cleared, now has {len(self.buffer)} items (should be 0)")
        
        # Pad with zeros to reach expected size
        if available_items < total_items_needed:
            zeros_to_add = total_items_needed - available_items
            zero_padding = np.zeros(zeros_to_add, dtype=np.complex64)
            buffered_data_with_padding = np.append(buffered_data_with_padding, zero_padding)
            print(f"⚡ Padded {zeros_to_add} zeros due to immediate timeout processing")
            
            # Calculate how many packets were missing and send to packet_check
            packets_padded = zeros_to_add // self.packet_size
            if zeros_to_add % self.packet_size > 0:
                packets_padded += 1  # Round up for partial packets
            if packets_padded > 0:
                self.send_padding_info(packets_padded)
        else:
            buffered_data_with_padding = buffered_data_with_padding[:total_items_needed]
        
        # Remove padding complex values from end if specified
        if self.padding_complex_items > 0 and len(buffered_data_with_padding) >= self.padding_complex_items:
            buffered_complex_data = buffered_data_with_padding[:-self.padding_complex_items]
        else:
            buffered_complex_data = buffered_data_with_padding
        
        # Calculate expected decoder items (should be the base complex items without padding)
        expected_decoder_items = self.expected_input_complex_items - self.padding_complex_items
        if len(buffered_complex_data) != expected_decoder_items:
            if len(buffered_complex_data) < expected_decoder_items:
                zeros_needed = expected_decoder_items - len(buffered_complex_data)
                zero_padding = np.zeros(zeros_needed, dtype=np.complex64)
                buffered_complex_data = np.append(buffered_complex_data, zero_padding)
                print(f"⚡ Final padding of {zeros_needed} zeros for decoder")
            else:
                buffered_complex_data = buffered_complex_data[:expected_decoder_items]
        
        print(f"⚡ IMMEDIATE PROCESSING: {len(buffered_complex_data)} complex items")
        
        # Process the data (similar to normal work() flow)
        self._decode_and_save_image(buffered_complex_data, timeout_occurred=True)

    def _decode_and_save_image(self, buffered_complex_data, timeout_occurred=False):
        """Unified decode image data and save result function"""
        try:
            # Convert complex data to real tensor format
            real_parts = buffered_complex_data.real
            imag_parts = buffered_complex_data.imag
            channel_output = np.concatenate([real_parts, imag_parts]).astype(np.float32)
            
            # Reshape to decoder input format
            channel_tensor = torch.from_numpy(channel_output).reshape(1, self.tcn, self.height//4, self.width//4)
            
            # Create attention tensor (SNR value) - frame-based power-domain averaging
            # Use collected frame SNRs from OFDM symbol processing
            if len(self.current_frame_snrs) > 0:
                received_snrs = np.array(self.current_frame_snrs)
                print(f"🎯 RECEIVED {len(received_snrs)} FRAME SNRs from OFDM symbols, expected {self.expected_frames}")
                
                # Pad with zeros if we have fewer SNRs than expected frames
                if len(received_snrs) < self.expected_frames:
                    missing_snrs = self.expected_frames - len(received_snrs)
                    # Add zeros for missing SNRs (representing no signal/very low SNR frames)
                    padded_snrs = np.append(received_snrs, np.zeros(missing_snrs))
                    # padded_snrs = received_snrs
                    print(f"🎯 PADDED {missing_snrs} zero SNRs for missing frames")
                else:
                    # Use only the first expected_frames SNRs if we have more than expected
                    padded_snrs = received_snrs[:self.expected_frames]
                    if len(received_snrs) > self.expected_frames:
                        print(f"🎯 TRUNCATED {len(received_snrs) - self.expected_frames} excess SNRs")
                
                # Compute power-domain average
                snr_value = self.compute_power_domain_snr_average(padded_snrs)
                if snr_value > 15:
                    snr_value = 15  # Cap at 15 dB
                print(f"🎯 POWER-DOMAIN AVERAGE SNR: {snr_value+self.correction:.2f} dB (from {len(padded_snrs)} values)")
                
                # Clear frame SNRs after processing
                self.current_frame_snrs = []
            else:
                snr_value = self.snr_db
                print(f"🎯 USING DEFAULT SNR: {snr_value}")
            
            attention = torch.tensor([[snr_value+self.correction]], dtype=torch.float32)
            
            # Run decoder
            inf_start = time.time()
            with torch.no_grad():
                decoded_img = self.decoder_model(channel_tensor, attention)
            inf_time = (time.time() - inf_start) * 1000
            
            # Get paths for saving using structured format
            current_origin_path = self.get_current_image_path() or self.origin_img_path
            current_output_path = self.get_structured_image_path()
            
            # Calculate metrics
            psnr_value, ssim_value, metrics_status = self._calculate_metrics(decoded_img, current_origin_path)
            
            # Save image
            saved_image = transforms.ToPILImage()(decoded_img.squeeze(0))
            saved_image.save(current_output_path, "PNG")
            
            # Log metrics with image path
            self.log_metrics_to_csv(psnr_value, ssim_value, snr_value+self.correction, metrics_status, current_output_path)
            
            status_msg = "LAST-PACKET TIMEOUT" if timeout_occurred else "NORMAL"
            repeat_info = f"[{self.repeat_count + 1}/{self.max_repeats}]"
            print(f"✅ {status_msg} DECODE COMPLETE {repeat_info}: {self.method_name}/{self.kodak_id},\nPSNR={psnr_value:.2f}, SSIM={ssim_value:.4f} | Inference Time: {inf_time:.1f} ms")
            # print(f"📁 Saved: {os.path.basename(current_output_path)}")
            
            # Handle repeat/continue logic
            self._handle_completion()
            
        except Exception as e:
            print(f"❌ ERROR in decoding: {e}")
            # Log error with empty image path
            self.log_metrics_to_csv(0.0, 0.0, 0.0, "decode_error", "")

    def _calculate_metrics(self, decoded_img, origin_path):
        """Calculate PSNR and SSIM metrics"""
        psnr_value = 0.0
        ssim_value = 0.0
        metrics_status = "error"
        
        try:
            # Load and convert origin image to tensor
            origin_img = Image.open(origin_path)
            origin_img.load()
            transform = transforms.Compose([transforms.ToTensor()])
            # Convert to tensor and add batch dimension
            origin_img_tensor_3d = transform(origin_img)  # [3, H, W]
            if isinstance(origin_img_tensor_3d, torch.Tensor):
                origin_img_tensor = torch.unsqueeze(origin_img_tensor_3d, 0)  # [1, 3, H, W]
            else:
                raise ValueError("Failed to convert image to tensor")
            
            # Calculate SSIM
            ssim_loss = ssim(origin_img_tensor, decoded_img)
            ssim_value = ssim_loss.item()
            
            # Calculate PSNR  
            origin_img_squeezed = origin_img_tensor.squeeze(0)
            decoded_img_squeezed = decoded_img.squeeze(0)
            vloss = F.mse_loss(origin_img_squeezed, decoded_img_squeezed)
            
            if vloss > 0:
                psnr = 10 * torch.log10(1 / vloss)
                psnr_value = psnr.item()
            else:
                psnr_value = float('inf')
                
            metrics_status = "success"
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics_status = "calculation_error"
            
        return psnr_value, ssim_value, metrics_status

    def _handle_completion(self):
        """Handle completion logic for repeat functionality and image sequence"""
        # Increment repeat counter for current image
        self.repeat_count += 1
        
        if self.repeat_count < self.max_repeats:
            # Continue repeating the same image
            current_path = self.get_current_image_path()
            image_name = os.path.basename(current_path) if current_path else "Unknown"
            print(f"Decoder: Waiting for repeat of image {image_name} ({self.repeat_count + 1}/{self.max_repeats}).")
            self.output_done = False
            self.reset_for_next_decode()
        else:
            # Completed all repetitions of current image, try to advance to next image
            if self.advance_to_next_image():
                # More images to process
                print("Decoder: Moving to next image, resetting for next decode.")
                self.output_done = False
                self.reset_for_next_decode()
            else:
                # No more images to process
                print("Decoder: Completed processing all images. Marking as done.")
                self.output_done = True

    def general_work(self, input_items, output_items):
        in0 = input_items[0]  # Complex input stream
        in1 = input_items[1]  # OFDM symbols input stream (64-element arrays)
        ninput_items_complex = len(in0)
        ninput_items_ofdm = len(in1)

        # Process OFDM symbols immediately and consume them right away
        if ninput_items_ofdm > 0:
            for ofdm_symbol in in1:
                self.process_ofdm_symbol(ofdm_symbol)
            self.consume(1, ninput_items_ofdm)  # Consume OFDM port immediately

        if ninput_items_complex == 0 and self.output_done:
            # Consume any remaining complex inputs before exiting (OFDM already consumed)
            self.consume(0, ninput_items_complex)
            return -1
        
        # Check if timeout flag is set (from timer thread)
        timeout_occurred = self.timeout_flag
        

        # Even if no new input, check if we need to process due to timeout
        if ninput_items_complex == 0 and self.force_processing and self.timeout_flag:
            ninput_items_complex = 0  # Continue processing with timeout
            print("Decoder: Processing last-packet timeout with no new input data")

        # CRITICAL: Only restart timeout timer for COMPLEX data, not float data
        if not self.output_done and ninput_items_complex > 0:
            # Update last data time and reset timeout timer on every new COMPLEX data arrival
            current_time = time.time()
            self.last_data_time = current_time
            
            # Cancel existing timer if running
            if self.timeout_timer:
                self.timeout_timer.cancel()
                self.timeout_timer = None
            
            # Start new timeout timer from this moment
            if self.timeout_ms > 0:
                self.waiting_for_data = True
                self.timeout_flag = False
                self.force_processing = False
                # Start timer thread
                self.timeout_timer = threading.Timer(self.timeout_ms / 1000.0, self.timeout_callback)
                self.timeout_timer.start()
                # print(f"Decoder: Received data, restarting {self.timeout_ms}ms timeout timer")
            
            self.buffer = np.append(self.buffer, in0)

        # Check if we have enough data OR if timeout has occurred
        should_decode = False
        
        if not self.output_done:
            # Calculate how much data we need (total complex items expected from encoder)
            total_items_needed = self.expected_input_complex_items
            
            if len(self.buffer) >= total_items_needed:
                # We have enough data
                should_decode = True
                # Cancel timeout timer since we got all data
                if self.timeout_timer:
                    self.timeout_timer.cancel()
                    self.timeout_timer = None
                print(f"🎯 COMPLETE PACKET: Buffer has {len(self.buffer)} items (need {total_items_needed})")
            elif timeout_occurred and len(self.buffer) > 0:
                # Timeout occurred and we have some data
                should_decode = True
                print(f"Decoder: Processing last-packet timeout with {len(self.buffer)} items (expected {total_items_needed})")
            elif self.waiting_for_data and self.timeout_ms > 0 and self.last_data_time is not None and len(self.buffer) > 0:
                # Also check for timeout in current work() call as backup
                current_time = time.time()
                elapsed_ms = (current_time - self.last_data_time) * 1000
                
                if elapsed_ms >= self.timeout_ms:
                    timeout_occurred = True
                    should_decode = True
                    if self.timeout_timer:
                        self.timeout_timer.cancel()
                        self.timeout_timer = None
                    print(f"Decoder: Timeout detected in work() after {elapsed_ms:.1f}ms since last packet with {len(self.buffer)} items (expected {total_items_needed})")
            elif ninput_items_complex == 0 and len(self.buffer) > 0 and (not self.waiting_for_data or hasattr(self, 'force_end_of_stream') and self.force_end_of_stream):
                # End-of-stream case: no new input, have partial data, not waiting for timeout OR forced end-of-stream
                # This handles the case where packet_check filled most packets, but we need to decode the final partial packet
                should_decode = True
                timeout_occurred = True  # Treat as timeout case for padding logic
                self.force_end_of_stream = False  # Reset flag
                print(f"Decoder: End-of-stream detected with {len(self.buffer)} items (expected {total_items_needed})")

        if should_decode:
            # Reset timeout state
            self.waiting_for_data = False
            self.last_data_time = None
            
            # Debug: Show decoding trigger and buffer state
            decode_reason = "TIMEOUT" if timeout_occurred else "COMPLETE"
            # print(f"🎯 DECODE TRIGGERED ({decode_reason}): Buffer has {len(self.buffer)} items BEFORE processing")
            
            # Get the data and remove padding complex values if specified
            total_items_to_read = self.expected_input_complex_items
            
            if timeout_occurred:
                # Timeout case: use whatever we have and pad with zeros
                available_items = len(self.buffer)
                buffered_data_with_padding = self.buffer.copy()  # Use all available data
                self.buffer = np.array([], dtype=np.complex64)  # Clear buffer
                
                print(f"📊 TIMEOUT PROCESSING: Buffer cleared, now has {len(self.buffer)} items (should be 0)")
                
                # Pad with zeros to reach the expected size
                items_needed = total_items_to_read
                if available_items < items_needed:
                    zeros_to_add = items_needed - available_items
                    zero_padding = np.zeros(zeros_to_add, dtype=np.complex64)
                    buffered_data_with_padding = np.append(buffered_data_with_padding, zero_padding)
                    print(f"Decoder: Padded {zeros_to_add} zeros due to last-packet timeout (had {available_items}, needed {items_needed})")
                    
                    # Calculate how many packets were missing and send to packet_check
                    packets_padded = zeros_to_add // self.packet_size
                    if zeros_to_add % self.packet_size > 0:
                        packets_padded += 1  # Round up for partial packets
                    if packets_padded > 0:
                        self.send_padding_info(packets_padded)
                else:
                    # Truncate if we somehow have more than needed
                    buffered_data_with_padding = buffered_data_with_padding[:items_needed]
                
            else:
                # Normal case: we have enough data
                if len(self.buffer) >= total_items_to_read:
                    buffered_data_with_padding = self.buffer[:total_items_to_read]
                    self.buffer = self.buffer[total_items_to_read:]
                    print(f"📊 NORMAL PROCESSING: Processed {total_items_to_read} items, {len(self.buffer)} items remaining in buffer")
                else:
                    # Edge case: not enough data even though should_decode is True
                    self.consume(0, ninput_items_complex)
                    return 0
            
            # Remove padding complex values from the end if specified
            if self.padding_complex_items > 0 and len(buffered_data_with_padding) >= self.padding_complex_items:
                buffered_complex_data = buffered_data_with_padding[:-self.padding_complex_items]
                if not timeout_occurred:  # Only print this for normal operation
                    print(f"Decoder WORK: Removed {self.padding_complex_items} padding complex values from end of buffer")
                    # print(f"🔍 DATA ANALYSIS: Original buffer size: {len(buffered_data_with_padding)}, After padding removal: {len(buffered_complex_data)}")
            else:
                buffered_complex_data = buffered_data_with_padding
            
            # Calculate expected decoder items (should be the base complex items without padding)
            expected_decoder_items = self.expected_input_complex_items - self.padding_complex_items
            if len(buffered_complex_data) != expected_decoder_items:
                if len(buffered_complex_data) < expected_decoder_items:
                    # Pad to exact size needed by decoder
                    zeros_needed = expected_decoder_items - len(buffered_complex_data)
                    zero_padding = np.zeros(zeros_needed, dtype=np.complex64)
                    buffered_complex_data = np.append(buffered_complex_data, zero_padding)
                    print(f"Decoder: Final padding of {zeros_needed} zeros to reach decoder input size")
                else:
                    # Truncate to exact size
                    buffered_complex_data = buffered_complex_data[:expected_decoder_items]
            
            # print(f"Decoder WORK: Processing {len(buffered_complex_data)} complex items ({'last-packet timeout' if timeout_occurred else 'normal'}), remaining: {len(self.buffer)}")
            
            # Debug: Show buffer state after processing
            # print(f"📊 BUFFER STATE AFTER PROCESSING: {len(self.buffer)} items remaining (should be 0 for timeout cases)")

            # Use unified decode function
            try:
                self._decode_and_save_image(buffered_complex_data, timeout_occurred)
            except Exception as e:
                print(f"Decoder: Error during decoding: {e}")
                self.log_metrics_to_csv(0.0, 0.0, 0.0, "decode_error")
                # Consume all complex input items for general_work (OFDM already consumed)
                self.consume(0, ninput_items_complex)
                return 0

        if self.output_done and ninput_items_complex == 0:
            # Consume any remaining complex inputs before exiting (OFDM already consumed)
            self.consume(0, ninput_items_complex)
            return -1
        
        # Consume all complex input items for general_work (OFDM already consumed)
        self.consume(0, ninput_items_complex)
        return 0
