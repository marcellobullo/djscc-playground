#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Georgios Theof.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy
from gnuradio import gr
import atexit

class snr_estimator(gr.basic_block):
    """
    SNR estimator block that accepts two complex data streams:
    - pilots: known reference symbols for SNR estimation
    - payload: data symbols
    Calculates SNR using known pilot symbols and noise variance estimation.
    """
    def __init__(self, avg_power=0.00995, occupied_carriers=[], pilot_carriers=[]):
        gr.basic_block.__init__(self,
            name="snr_estimator",
            in_sig=[(numpy.complex64, 64)],  # Full OFDM symbol (64 subcarriers)
            out_sig=[numpy.float32])  # SNR output
        
        # Known pilot symbols pattern at pilot subcarriers
        self.pilot_symbols = numpy.array([1, 1, 1, -1], dtype=numpy.complex64)
        
        # Build subcarrier indices from flowgraph parameters
        if not occupied_carriers or not pilot_carriers:
            # Hardcoded carrier allocation matching your flowgraph
            occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
            pilot_carriers = ((-21, -7, 7, 21,),)
            print("Using hardcoded carrier masks from flowgraph")
        
        # Extract integer values from whatever format we get
        N = 64  # FFT size
        
        # Simple extraction: just get all the integers out
        def extract_ints(data):
            result = []
            if hasattr(data, '__iter__') and not isinstance(data, str):
                for item in data:
                    if isinstance(item, int):
                        result.append(item)
                    elif hasattr(item, '__iter__') and not isinstance(item, str):
                        result.extend(extract_ints(item))
            elif isinstance(data, int):
                result.append(data)
            return result
        
        occ_flat = extract_ints(occupied_carriers)
        pil_flat = extract_ints(pilot_carriers)
        
        self.occupied_indices = [(k + N) % N for k in sorted(occ_flat)]
        self.pilot_indices = [(k + N) % N for k in sorted(pil_flat)]
        self.data_indices = sorted(set(self.occupied_indices) - set(self.pilot_indices))
        
        # Verify indices match expected counts
        print(f"Occupied indices: {len(self.occupied_indices)} carriers")
        print(f"Pilot indices: {len(self.pilot_indices)} carriers at {self.pilot_indices}")
        print(f"Data indices: {len(self.data_indices)} carriers")
        
        if len(self.data_indices) != 48 or len(self.pilot_indices) != 4:
            print(f"WARNING: Expected 48 data + 4 pilot carriers, got {len(self.data_indices)} + {len(self.pilot_indices)}")
        
        # Average power normalization factor (configurable from GUI)
        self.avg_power = avg_power
        
        # Frame processing parameters
        self.frame_size = 20  # Process 20 symbols per frame
        self.symbols_in_current_frame = 0  # Count symbols in current frame
        
        # Initialize tracking variables
        self.noise_variance_estimates = []
        self.symbol_energy_estimates = []
        self.snr_estimates = []
        self.alpha_magnitudes = []  # Track |α_m| per symbol
        self.sample_count = 0
        self.frame_count = 0
        
        # Buffer for current frame data (normalized symbols)
        self.frame_pilot_data_normalized = []  # Store normalized pilot data
        self.frame_payload_data_normalized = []  # Store normalized payload data
        
        # Register cleanup function to print statistics when program ends
        atexit.register(self.print_statistics)

    def calculate_alpha_m(self, ofdm_symbol):
        """
        Calculate least-squares gain/phase correction α_m using the formula:
        α_m = (Σ_k∈P Z_{k,m} * S_{k,m}*) / (Σ_k∈P |S_{k,m}|²)
        
        where:
        - Z_{k,m}: received pilot symbols from OFDM symbol
        - S_{k,m}: known pilot symbols (self.pilot_symbols)
        - P: set of pilot indices {-21, -7, 7, 21} → {43, 57, 7, 21}
        """
        # Extract pilots from the full OFDM symbol
        pilot_values = [ofdm_symbol[idx] for idx in self.pilot_indices]
        
        if len(pilot_values) != len(self.pilot_symbols):
            return None
            
        # Calculate numerator: Σ_k∈P Z_{k,m} * S_{k,m}*
        numerator = 0.0
        for k in range(len(self.pilot_symbols)):
            Z_k = pilot_values[k]
            S_k_conj = numpy.conj(self.pilot_symbols[k])
            numerator += Z_k * S_k_conj
            
        # Calculate denominator: Σ_k∈P |S_{k,m}|²
        denominator = 0.0
        for k in range(len(self.pilot_symbols)):
            denominator += numpy.abs(self.pilot_symbols[k])**2
            
        if denominator > 1e-12:
            alpha_m = numerator / denominator
            return alpha_m
        else:
            return None

    def calculate_symbol_energy(self, normalized_data_symbols):
        """
        Calculate symbol energy from normalized data tones using the formula:
        Ê_s = (1/(|D|*M)) * Σ_m Σ_k∈D |Z̃^{(d)}_{k,m}|²
        
        where:
        - D is the set of data subcarrier indices
        - M is the number of payload symbols (20 per frame)
        - Z̃^{(d)}_{k,m} is the normalized received data symbol
        """
        if len(normalized_data_symbols) == 0:
            return None
            
        # Calculate sum of |Z̃^{(d)}_{k,m}|² over all normalized data tones
        energy_sum = 0.0
        total_symbols = 0
        
        # Sum over all normalized data symbols in the frame
        for data_symbol_vector in normalized_data_symbols:
            for symbol in data_symbol_vector:
                energy_sum += numpy.abs(symbol)**2
                total_symbols += 1
        
        # Average over all symbols
        if total_symbols > 0:
            symbol_energy = energy_sum / total_symbols
        else:
            return None
        
        return symbol_energy

    def calculate_noise_variance(self, normalized_pilot_data, normalized_payload_data):
        """
        Calculate noise variance from normalized pilots using the formula:
        σ̂² = (1/(|P|*M)) * Σ_m Σ_k∈P |Z̃^{(p)}_{k,m} - S_{k,m}|²
        
        where:
        - P is the set of pilot indices (4 pilots per symbol)
        - M is the number of payload symbols (20 per frame)
        - Z̃^{(p)}_{k,m} is the normalized received pilot symbol
        - S_{k,m} is the known transmitted pilot symbol
        """
        if len(normalized_pilot_data) == 0 or len(normalized_payload_data) == 0:
            return None
            
        # Calculate pilot errors for current frame using normalized pilots
        pilot_errors_squared = 0.0
        total_pilot_comparisons = 0
        
        # Sum over all normalized pilot data in the frame
        for pilot_data_vector in normalized_pilot_data:
            if len(pilot_data_vector) >= len(self.pilot_symbols):
                # Compare each normalized pilot with expected pilot
                for k in range(len(self.pilot_symbols)):
                    expected_pilot = self.pilot_symbols[k]
                    normalized_received_pilot = pilot_data_vector[k]
                    
                    # Calculate |Z̃^{(p)}_{k,m} - S_{k,m}|²
                    error_squared = numpy.abs(normalized_received_pilot - expected_pilot)**2
                    pilot_errors_squared += error_squared
                    total_pilot_comparisons += 1
        
        # FIXED: Divide by total_pilot_comparisons only (which equals |P|*M)
        # total_pilot_comparisons already equals |P| × M (4 pilots × number of symbols)
        if total_pilot_comparisons > 0:
            noise_variance = pilot_errors_squared / total_pilot_comparisons
        else:
            return None
        
        # Note: No normalization by avg_power since we're working with normalized symbols
        return noise_variance

    def print_statistics(self):
        """Print comprehensive statistics when the program ends"""
        print(f"\n=== SNR Estimator Statistics ===")
        print(f"Total symbols processed: {self.sample_count}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Pilot symbol pattern: {self.pilot_symbols}")
        print(f"Average power normalization: {self.avg_power}")
        print(f"Frame size: {self.frame_size} symbols per frame")
        print(f"Process ALL symbols (no skipping)")
        print(f"Occupied indices: {len(self.occupied_indices)} carriers")
        print(f"Pilot indices: {self.pilot_indices} ({len(self.pilot_indices)} carriers)")
        print(f"Data indices: {len(self.data_indices)} carriers")
        
        if len(self.data_indices) == 48 and len(self.pilot_indices) == 4:
            print("✓ Carrier allocation correct: 48 data + 4 pilot carriers")
        else:
            print("✗ WARNING: Unexpected carrier allocation")
        
        if self.alpha_magnitudes:
            avg_alpha_mag = numpy.mean(self.alpha_magnitudes)
            print(f"Per-symbol |α_m| mean: {avg_alpha_mag:.6f}")
            print(f"Number of α_m estimates: {len(self.alpha_magnitudes)}")
        
        if self.noise_variance_estimates:
            avg_noise_var = numpy.mean(self.noise_variance_estimates)
            print(f"Average noise variance from pilots: {avg_noise_var:.6f}")
            print(f"Number of noise variance estimates: {len(self.noise_variance_estimates)}")
        else:
            avg_noise_var = None
            
        if self.symbol_energy_estimates:
            avg_symbol_energy = numpy.mean(self.symbol_energy_estimates)
            print(f"Average symbol energy from data tones: {avg_symbol_energy:.6f}")
            print(f"Number of symbol energy estimates: {len(self.symbol_energy_estimates)}")
        else:
            avg_symbol_energy = None
            
        # Calculate SNR using the proper formula: SNR_dB = 10*log10(E_s/σ²)
        if avg_symbol_energy is not None and avg_noise_var is not None and avg_noise_var > 0:
            snr_from_formula = 10 * numpy.log10(avg_symbol_energy / avg_noise_var)
            print(f"SNR from formula (E_s/σ²): {snr_from_formula:.2f} dB")
            
            # Also calculate with noise variance (assuming signal power = 1 after normalization)
            if avg_noise_var > 0:
                snr_from_noise = 10 * numpy.log10(1.0 / avg_noise_var)
                print(f"SNR from noise variance (1/σ²): {snr_from_noise:.2f} dB")
        else:
            print("Insufficient data for SNR calculation from formula")
        
        if self.snr_estimates:
            avg_snr = numpy.mean(self.snr_estimates)
            print(f"Average SNR (frame estimates): {avg_snr:.2f} dB")
            print(f"Number of SNR estimates: {len(self.snr_estimates)}")
        
        if not self.noise_variance_estimates and not self.snr_estimates:
            print("No estimates calculated")
        print("================================\n")

    def forecast(self, noutput_items, ninputs):
        # We need one OFDM symbol (64 subcarriers) per output
        ninput_items_required = [noutput_items]
        return ninput_items_required

    def general_work(self, input_items, output_items):
        # Get full OFDM symbols (64 subcarriers each)
        ofdm_symbols = input_items[0]  # Array of 64-element vectors
        
        # Determine the number of symbols to process
        ninput_items = len(ofdm_symbols)
        
        # We'll produce outputs only when frames are complete
        noutput_produced = 0
        
        # Process each OFDM symbol
        for i in range(ninput_items):
            # Get current OFDM symbol (64 subcarriers)
            ofdm_symbol = ofdm_symbols[i]
            
            # Step 1: Calculate least-squares gain/phase correction α_m
            alpha_m = self.calculate_alpha_m(ofdm_symbol)
            
            if alpha_m is not None:
                # Track |α_m| for statistics
                self.alpha_magnitudes.append(numpy.abs(alpha_m))
                
                # Step 2: Normalize the whole symbol with α_m
                normalized_symbol = ofdm_symbol / alpha_m
                
                # Extract pilots and data from normalized symbol
                normalized_pilots = [normalized_symbol[idx] for idx in self.pilot_indices]
                normalized_data = [normalized_symbol[idx] for idx in self.data_indices]
                
                # Debug: Print pilot phases and verify normalization for first few symbols
                if len(self.alpha_magnitudes) <= 5:
                    pilot_phases = [numpy.angle(p) * 180 / numpy.pi for p in normalized_pilots]
                    pilot_magnitudes_sq = [numpy.abs(p)**2 for p in normalized_pilots]
                    mean_pilot_power = numpy.mean(pilot_magnitudes_sq)
                    phase_std = numpy.std(pilot_phases)
                    
                    print(f"Frame {self.frame_count}, Symbol {self.symbols_in_current_frame}: |α_m| = {numpy.abs(alpha_m):.4f}")
                    print(f"  Pilot phases = {[f'{p:.1f}°' for p in pilot_phases]} (std: {phase_std:.1f}°)")
                    print(f"  Mean |normalized pilots|² = {mean_pilot_power:.4f} (should be ≈ 1.0)")
            else:
                # If we can't calculate α_m, skip normalization (use raw data)
                normalized_pilots = [ofdm_symbol[idx] for idx in self.pilot_indices]
                normalized_data = [ofdm_symbol[idx] for idx in self.data_indices]
            
            # Add normalized data to current frame buffers
            self.frame_pilot_data_normalized.append(normalized_pilots)
            self.frame_payload_data_normalized.append(normalized_data)
            self.symbols_in_current_frame += 1
            
            # Check if we've completed a frame (20 symbols)
            if self.symbols_in_current_frame == self.frame_size:
                # We have processed a full frame, time to calculate and output SNR
                
                # Make sure we have space in output buffer
                if noutput_produced < len(output_items[0]):
                    # Step 3: Calculate noise variance from normalized pilots
                    noise_var = self.calculate_noise_variance(self.frame_pilot_data_normalized, self.frame_payload_data_normalized)
                    
                    # Step 4: Calculate symbol energy from normalized data
                    symbol_energy = self.calculate_symbol_energy(self.frame_payload_data_normalized)
                    
                    if noise_var is not None:
                        self.noise_variance_estimates.append(noise_var)
                        
                    if symbol_energy is not None:
                        self.symbol_energy_estimates.append(symbol_energy)
                    
                    # Step 5: Calculate SNR using the proper formula: SNR = 10*log10(E_s/σ²)
                    if noise_var is not None and symbol_energy is not None:
                        if noise_var > 1e-12:
                            snr_db = 10 * numpy.log10(symbol_energy / noise_var)
                        else:
                            snr_db = 100  # Very high SNR if noise is negligible
                        
                        # Limit SNR to reasonable range
                        snr_db = numpy.clip(snr_db, -50, 50)
                        
                        # Output the SNR for this completed frame
                        output_items[0][noutput_produced] = snr_db
                        self.snr_estimates.append(snr_db)
                        noutput_produced += 1
                        
                    else:
                        # Could not calculate SNR
                        output_items[0][noutput_produced] = 0.0
                        noutput_produced += 1
                
                # Reset frame buffers and counter for next frame
                self.frame_pilot_data_normalized = []
                self.frame_payload_data_normalized = []
                self.symbols_in_current_frame = 0
                self.frame_count += 1
        
        self.sample_count += ninput_items
        
        # Consume all input items and return number of output items produced
        self.consume_each(ninput_items)
        return noutput_produced

