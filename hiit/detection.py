import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.cluster import KMeans
import scipy.fft as fft
import multiprocessing as mp
from functools import partial

def compute_frequency_correlation_chunk(chunk_data):
    """Compute frequency correlation for a chunk of positions."""
    chunk_start, chunk_end, hr_filtered_chunk, window_size, dominant_freq, sampling_rate = chunk_data
    chunk_correlations = []
    
    # Process each position in this chunk
    for pos in range(chunk_end - chunk_start):
        # Check if we have enough data for this position
        if pos + window_size > len(hr_filtered_chunk):
            chunk_correlations.append(0.0)
            continue
            
        window_data = hr_filtered_chunk[pos:pos + window_size]
        
        # Perform FFT on this window
        window_fft = fft.fft(window_data)
        window_freqs = fft.fftfreq(len(window_data), 1/sampling_rate)
        
        # Find the power at the dominant frequency
        freq_mask = np.abs(window_freqs - dominant_freq) < (0.5 / window_size)  # Within 0.5 bins
        if np.any(freq_mask):
            power_at_dominant = np.abs(window_fft[freq_mask]).max()
        else:
            power_at_dominant = 0
        
        # Normalize by the total power in the window
        total_power = np.sum(np.abs(window_fft))
        if total_power > 0:
            correlation = power_at_dominant / total_power
        else:
            correlation = 0
        
        chunk_correlations.append(correlation)
    
    return chunk_correlations

def compute_correlation_chunk(chunk_data):
    """Compute correlation for a chunk of positions."""
    chunk_start, chunk_end, hr_filtered_chunk, speed_data_chunk, hr_templates, speed_templates, num_phases, template_length = chunk_data
    
    chunk_correlations = []
    
    # Process each position in this chunk
    for pos in range(chunk_end - chunk_start):
        # Calculate the actual position in the full signal
        actual_pos = chunk_start + pos
        
        # Check if we have enough data for this position
        if pos + template_length > len(hr_filtered_chunk):
            chunk_correlations.append(0.0)
            continue
            
        # Extract windows for both signals at this position
        hr_window = hr_filtered_chunk[pos:pos + template_length]
        speed_window = speed_data_chunk[pos:pos + template_length] if len(speed_data_chunk) > 0 else np.zeros(template_length)
        
        # Compute correlation with all phase-shifted templates for heart rate
        max_hr_correlation = 0
        for phase_idx in range(num_phases):
            hr_template = hr_templates[phase_idx]
            
            # Normalize both template and window for shape correlation
            hr_template_norm = (hr_template - np.mean(hr_template)) / np.std(hr_template)
            hr_window_norm = (hr_window - np.mean(hr_window)) / np.std(hr_window)
            
            # Compute shape correlation (normalized) - make it much stricter
            hr_shape_corr = np.abs(np.dot(hr_template_norm, hr_window_norm)) / (np.linalg.norm(hr_template_norm) * np.linalg.norm(hr_window_norm))
            
            # Compute amplitude similarity (how close the ranges are)
            hr_amp_similarity = 1.0 - min(1.0, abs(np.std(hr_template) - np.std(hr_window)) / max(np.std(hr_template), np.std(hr_window)))
            
            # Make correlation much stricter by requiring both high shape and amplitude similarity
            hr_correlation = hr_shape_corr * hr_amp_similarity  # Multiply instead of weighted sum
            max_hr_correlation = max(max_hr_correlation, hr_correlation)
        
        # Compute correlation with all phase-shifted templates for speed
        max_speed_correlation = 0
        for phase_idx in range(num_phases):
            speed_template = speed_templates[phase_idx]
            
            # Compute correlation for speed
            if np.linalg.norm(speed_template) > 0 and np.linalg.norm(speed_window) > 0:
                speed_template_norm = (speed_template - np.mean(speed_template)) / np.std(speed_template)
                speed_window_norm = (speed_window - np.mean(speed_window)) / np.std(speed_window)
                
                speed_shape_corr = np.abs(np.dot(speed_template_norm, speed_window_norm)) / (np.linalg.norm(speed_template_norm) * np.linalg.norm(speed_window_norm))
                speed_amp_similarity = 1.0 - min(1.0, abs(np.std(speed_template) - np.std(speed_window)) / max(np.std(speed_template), np.std(speed_window)))
                
                # Make correlation much stricter by requiring both high shape and amplitude similarity
                speed_correlation = speed_shape_corr * speed_amp_similarity  # Multiply instead of weighted sum
                max_speed_correlation = max(max_speed_correlation, speed_correlation)
            else:
                max_speed_correlation = 0
        
        # Average the correlations from both signals
        avg_correlation = (max_hr_correlation + max_speed_correlation) / 2
        chunk_correlations.append(avg_correlation)
    
    return chunk_correlations

def detect_hiit_period_frequency(df, manual_hint=None, manual_threshold=None):
    """
    Detect HIIT period using rolling window frequency correlation.
    
    Args:
        df: DataFrame with heart rate data
        manual_hint: Optional tuple (start_idx, end_idx) to guide the algorithm
        manual_threshold: Optional manual correlation threshold to override auto-detection
    
    Returns:
        start_idx, end_idx: Indices of HIIT period
        frequency_info: Dictionary with frequency analysis results
    """
    if 'heart_rate' not in df.columns:
        return None, None, {}
    
    # Get clean data
    heart_rate = df['heart_rate'].ffill().bfill()
    
    if len(heart_rate) < 100:  # Need sufficient data
        return None, None, {}
    
    # Calculate sampling rate (assuming 1-second intervals)
    sampling_rate = 1.0  # Hz
    
    # Apply low-pass filter to heart rate data to reduce noise
    # Use a conservative cutoff to focus on the periodicity we care about
    cutoff_freq = 0.01  # 100 second period
    nyquist = sampling_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    
    # Design Butterworth low-pass filter
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    hr_filtered = filtfilt(b, a, heart_rate.values)
    
    # Perform FFT on filtered heart rate data to find dominant period
    hr_fft = fft.fft(hr_filtered)
    freqs = fft.fftfreq(len(hr_filtered), 1/sampling_rate)
    
    # Focus on positive frequencies and periods of interest (30s to 300s)
    positive_mask = freqs > 0
    freqs_positive = freqs[positive_mask]
    hr_fft_positive = hr_fft[positive_mask]
    
    # Convert to periods (in seconds)
    periods = 1 / freqs_positive
    period_mask = (periods >= 30) & (periods <= 300)
    
    if not np.any(period_mask):
        return None, None, {}
    
    periods_filtered = periods[period_mask]
    hr_fft_filtered = hr_fft_positive[period_mask]
    
    # Find the period with maximum power
    max_power_idx = np.argmax(np.abs(hr_fft_filtered))
    dominant_period = periods_filtered[max_power_idx]
    dominant_freq = 1 / dominant_period
    dominant_power = np.abs(hr_fft_filtered[max_power_idx])
    
    # Calculate rolling window frequency correlation
    # Use smaller window for sharper edges - ~2x the dominant period instead of 4x
    window_size = int(dominant_period * 2)
    if window_size > len(hr_filtered) // 3:
        window_size = len(hr_filtered) // 3
    
    # Parallel frequency correlation computation
    
    # Split frequency correlation work into chunks
    total_freq_positions = len(hr_filtered) - window_size + 1
    num_freq_cores = min(mp.cpu_count(), 6)  # Use fewer cores for frequency correlation
    freq_chunk_size = max(1, total_freq_positions // num_freq_cores)
    
    print(f"DEBUG: Parallelizing frequency correlation with {num_freq_cores} cores, {freq_chunk_size} positions per chunk")
    
    # Prepare frequency correlation chunk data
    freq_chunk_data_list = []
    for chunk_idx in range(num_freq_cores):
        chunk_start = chunk_idx * freq_chunk_size
        chunk_end = min(chunk_start + freq_chunk_size, total_freq_positions)
        
        if chunk_start >= total_freq_positions:
            break
            
        # Extract data for this chunk (need extra window_size-1 samples for overlap)
        hr_chunk = hr_filtered[chunk_start:chunk_end + window_size - 1]
        
        chunk_data = (chunk_start, chunk_end, hr_chunk, window_size, dominant_freq, sampling_rate)
        freq_chunk_data_list.append(chunk_data)
    
    # Process frequency correlation chunks in parallel
    with mp.Pool(processes=num_freq_cores) as pool:
        freq_chunk_results = pool.map(compute_frequency_correlation_chunk, freq_chunk_data_list)
    
    # Combine frequency correlation results in the correct order
    frequency_correlation = [0] * total_freq_positions  # Initialize with zeros
    for chunk_idx, chunk_result in enumerate(freq_chunk_results):
        chunk_start = chunk_idx * freq_chunk_size
        chunk_end = min(chunk_start + freq_chunk_size, total_freq_positions)
        
        # Place chunk results in the correct positions
        for i, correlation_value in enumerate(chunk_result):
            if chunk_start + i < total_freq_positions:
                frequency_correlation[chunk_start + i] = correlation_value
    
    # Pad the correlation array to match the original signal length
    pad_size = len(hr_filtered) - len(frequency_correlation)
    frequency_correlation = [0] * (pad_size // 2) + frequency_correlation + [0] * (pad_size - pad_size // 2)
    
    # Normalize frequency correlation to [0, 1] range
    if frequency_correlation:
        freq_corr_array = np.array(frequency_correlation)
        if freq_corr_array.max() > freq_corr_array.min():
            frequency_correlation = (freq_corr_array - freq_corr_array.min()) / (freq_corr_array.max() - freq_corr_array.min())
        else:
            frequency_correlation = freq_corr_array
    
    # Apply gentle edge sharpening to get sharper detection boundaries
    # Use morphological operations to sharpen the correlation signal
    from scipy import ndimage
    freq_corr_array = np.array(frequency_correlation)
    
    # Use smaller kernel and higher threshold for less aggressive sharpening
    kernel_size = max(3, int(dominant_period // 20))  # Smaller kernel
    threshold = 0.3  # Lower threshold to preserve more signal
    freq_corr_sharpened = ndimage.binary_opening(freq_corr_array > threshold, structure=np.ones(kernel_size))
    freq_corr_sharpened = ndimage.binary_closing(freq_corr_sharpened, structure=np.ones(kernel_size))
    
    # Convert back to continuous values with sharp edges, but preserve original values
    frequency_correlation = freq_corr_array * (0.7 + 0.3 * freq_corr_sharpened.astype(float))
    
    # Optimize threshold for topological contiguity
    # Start with a high threshold and systematically lower it to find optimal contiguity
    correlation_array = np.array(frequency_correlation)
    
    # Use clustering to find natural "high" correlation regions
    
    # Reshape for clustering (1D array to 2D)
    correlation_2d = correlation_array.reshape(-1, 1)
    
    # Find 2 clusters: low and high correlation
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(correlation_2d)
    
    # Identify which cluster is "high" (higher mean)
    cluster_means = [correlation_array[cluster_labels == i].mean() for i in range(2)]
    high_cluster = 0 if cluster_means[0] > cluster_means[1] else 1
    
    # Start with the mean of the high cluster
    initial_threshold = cluster_means[high_cluster]
    
    # Systematically lower threshold to optimize for contiguity
    best_threshold = initial_threshold
    best_score = -np.inf
    best_regions = []
    
    # Try thresholds from 90th percentile down to 50th percentile
    threshold_candidates = np.percentile(correlation_array, np.arange(90, 49, -2))
    
    for threshold in threshold_candidates:
        # Find regions above this threshold
        above_threshold = correlation_array > threshold
        
        # Find continuous regions
        regions = []
        start_idx = None
        
        for i, is_above in enumerate(above_threshold):
            if is_above and start_idx is None:
                start_idx = i
            elif not is_above and start_idx is not None:
                regions.append((start_idx, i))
                start_idx = None
        
        if start_idx is not None:
            regions.append((start_idx, len(above_threshold)))
        
        if not regions:
            continue
        
        # Find the longest region
        longest_region = max(regions, key=lambda x: x[1] - x[0])
        region_start, region_end = longest_region
        region_length = region_end - region_start
        
        # Calculate contiguity score
        # Penalize for holes (gaps) and overrun
        total_above = np.sum(above_threshold)
        contiguity_ratio = region_length / total_above if total_above > 0 else 0
        
        # Penalize for overrun (region too close to start/end)
        signal_length = len(correlation_array)
        start_buffer = signal_length * 0.05  # 5% buffer from start
        end_buffer = signal_length * 0.05    # 5% buffer from end
        
        overrun_penalty = 0
        if region_start < start_buffer:
            overrun_penalty += (start_buffer - region_start) / start_buffer
        if region_end > (signal_length - end_buffer):
            overrun_penalty += (region_end - (signal_length - end_buffer)) / end_buffer
        
        # Ensure minimum duration (at least 2x dominant period)
        min_duration = int(dominant_period * 2)
        duration_penalty = 0
        if region_length < min_duration:
            duration_penalty = (min_duration - region_length) / min_duration
        
        # Combined score: favor high contiguity, penalize overrun and short duration
        score = contiguity_ratio - overrun_penalty - duration_penalty
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_regions = regions
    
    if not best_regions:
        return None, None, {}
    
    # Use the best threshold and regions
    correlation_threshold = manual_threshold if manual_threshold is not None else best_threshold
    longest_region = max(best_regions, key=lambda x: x[1] - x[0])
    best_start, best_end = longest_region
    
    # If manual hint is provided, bias the selection towards that region
    if manual_hint is not None:
        manual_start, manual_end = manual_hint
        
        # Score each region based on overlap with manual hint
        best_overlap_score = -1
        best_region = longest_region
        
        for region in best_regions:
            region_start, region_end = region
            
            # Calculate overlap with manual hint
            overlap_start = max(region_start, manual_start)
            overlap_end = min(region_end, manual_end)
            overlap = max(0, overlap_end - overlap_start)
            
            # Calculate overlap score (normalized by manual window size)
            manual_size = manual_end - manual_start
            if manual_size > 0:
                overlap_ratio = overlap / manual_size
                
                # Also consider how well the region contains the manual hint
                containment_ratio = overlap / (region_end - region_start) if (region_end - region_start) > 0 else 0
                
                # Combined score: favor high overlap and good containment
                overlap_score = overlap_ratio * 0.7 + containment_ratio * 0.3
                
                if overlap_score > best_overlap_score:
                    best_overlap_score = overlap_score
                    best_region = region
        
        # Use the region with best overlap, but ensure it meets minimum criteria
        best_start, best_end = best_region
        min_duration = int(dominant_period * 1.5)  # Slightly more lenient with manual hint
        
        if best_end - best_start < min_duration:
            # If the best overlap region is too short, try to extend it
            # Look for nearby high correlation regions to merge
            extended_start = best_start
            extended_end = best_end
            
            # Try to extend backwards
            for i in range(best_start - 1, max(0, best_start - int(dominant_period * 2)), -1):
                if i < len(correlation_array) and correlation_array[i] > correlation_threshold * 0.8:
                    extended_start = i
                else:
                    break
            
            # Try to extend forwards
            for i in range(best_end, min(len(correlation_array), best_end + int(dominant_period * 2))):
                if i < len(correlation_array) and correlation_array[i] > correlation_threshold * 0.8:
                    extended_end = i + 1
                else:
                    break
            
            if extended_end - extended_start >= min_duration:
                best_start, best_end = extended_start, extended_end
    
    # Final validation: ensure minimum duration
    min_duration = int(dominant_period * 2)
    if best_end - best_start < min_duration:
        return None, None, {}
    
    # Fine-tune boundaries using speed clustering
    print(f"DEBUG: About to call refine_hiit_boundaries with best_start={best_start}, best_end={best_end}")
    try:
        # Pass the frequency correlation initially, will be updated with combined correlation later
        refined_start, refined_end = refine_hiit_boundaries(df, best_start, best_end, dominant_period, frequency_correlation, correlation_threshold)
        print(f"DEBUG: Original boundaries: {best_start}-{best_end}")
        print(f"DEBUG: Refined boundaries: {refined_start}-{refined_end}")
    except Exception as e:
        print(f"DEBUG: Error in refine_hiit_boundaries: {e}")
        import traceback
        traceback.print_exc()
        refined_start, refined_end = best_start, best_end
    
    # Compute template correlation for visualization (but don't use for detection)
    template_correlation = None
    mean_template = None
    if refined_start is not None and refined_end is not None:
        # Get intervals from the detected window
        df_hiit = df.iloc[refined_start:refined_end]
        intervals = segment_intervals_speed_edges(df_hiit)
        
        # Adjust interval indices to match full df
        for interval in intervals:
            for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                if k in interval:
                    interval[k] += refined_start
        
        # If we have at least 2 intervals, create a template
        if len(intervals) >= 2:
            print(f"DEBUG: Creating template correlation with {len(intervals)} intervals")
            template_correlation, mean_hr_template, mean_speed_template = compute_template_correlation(df, intervals, hr_filtered, dominant_period)
            if template_correlation is not None:
                print(f"DEBUG: Template correlation created, length: {len(template_correlation)}, range: [{min(template_correlation):.3f}, {max(template_correlation):.3f}]")
            else:
                print("DEBUG: Template correlation is None")
        else:
            print(f"DEBUG: Not enough intervals for template ({len(intervals)} < 2)")
    
    # Create combined correlation signal (average of normalized frequency and template correlations)
    combined_correlation = None
    if frequency_correlation is not None and template_correlation is not None:
        # Both correlations are already normalized to [0, 1]
        combined_correlation = (np.array(frequency_correlation) + np.array(template_correlation)) / 2
        print(f"DEBUG: Combined correlation created, range: [{combined_correlation.min():.3f}, {combined_correlation.max():.3f}]")
        
        # Now refine boundaries using the combined correlation
        try:
            refined_start, refined_end = refine_hiit_boundaries(df, refined_start, refined_end, dominant_period, combined_correlation, correlation_threshold)
            print(f"DEBUG: Re-refined boundaries with combined correlation: {refined_start}-{refined_end}")
        except Exception as e:
            print(f"DEBUG: Error in re-refinement with combined correlation: {e}")
            # Keep the previous refined boundaries
    elif frequency_correlation is not None:
        # If no template correlation, use frequency correlation as combined
        combined_correlation = np.array(frequency_correlation)
        print(f"DEBUG: Using frequency correlation as combined correlation")
    else:
        print(f"DEBUG: No correlations available for refinement")
    
    frequency_info = {
        'dominant_period': dominant_period,
        'dominant_freq': dominant_freq,
        'dominant_power': dominant_power,
        'frequencies': freqs_positive,
        'periods': periods,
        'hr_fft_magnitude': np.abs(hr_fft_positive),
        'filtered_periods': periods_filtered,
        'filtered_power': np.abs(hr_fft_filtered),
        'hr_filtered': hr_filtered,
        'frequency_correlation': frequency_correlation,
        'correlation_threshold': correlation_threshold,
        'window_size': window_size,
        'regions_above_threshold': best_regions,
        'optimization_score': best_score,
        'cluster_means': cluster_means,
        'high_cluster': high_cluster,
        'original_start': best_start,
        'original_end': best_end,
        'refined_start': refined_start,
        'refined_end': refined_end,
        'template_correlation': template_correlation,
        'combined_correlation': combined_correlation,
        'mean_template': mean_hr_template,
        'mean_hr_template': mean_hr_template,
        'mean_speed_template': mean_speed_template
    }
    
    return refined_start, refined_end, frequency_info

def refine_hiit_boundaries(df, hiit_start, hiit_end, dominant_period, frequency_correlation, correlation_threshold):
    """
    Refine HIIT boundaries using data-driven approach:
    1. Find contiguous region where combined correlation > 0.65
    2. Refine start: Find largest speed derivative change near start
    3. Refine end: Find nearest heart rate baseline return near end
    
    Args:
        df: DataFrame with speed and heart rate data
        hiit_start, hiit_end: Initial HIIT boundaries
        dominant_period: Dominant period from frequency analysis
        frequency_correlation: Array of combined correlation values (frequency + template)
        correlation_threshold: Optimal correlation threshold
    
    Returns:
        refined_start, refined_end: Refined boundaries
    """
    print(f"DEBUG: refine_hiit_boundaries called with start={hiit_start}, end={hiit_end}, period={dominant_period}")
    
    if 'enhanced_speed' not in df.columns or 'heart_rate' not in df.columns:
        print("DEBUG: Missing required columns, returning original boundaries")
        return hiit_start, hiit_end
    
    # Step 1: Define HIIT window as contiguous region where combined correlation > 0.65
    # Use the combined correlation (frequency + template)
    if frequency_correlation is None:
        print("DEBUG: No combined correlation available, returning original boundaries")
        return hiit_start, hiit_end
    
    # Use the combined correlation directly
    combined_correlation = np.array(frequency_correlation)
    
    # Find contiguous regions above threshold
    threshold = 0.65
    above_threshold = combined_correlation > threshold
    
    # Find the largest contiguous region that overlaps with the original window
    regions = []
    start_idx = None
    
    for i, is_above in enumerate(above_threshold):
        if is_above and start_idx is None:
            start_idx = i
        elif not is_above and start_idx is not None:
            regions.append((start_idx, i))
            start_idx = None
    
    if start_idx is not None:
        regions.append((start_idx, len(above_threshold)))
    
    print(f"DEBUG: Found {len(regions)} regions above threshold {threshold}")
    
    # Find the region that best overlaps with the original window
    best_region = None
    best_overlap = 0
    
    for region_start, region_end in regions:
        overlap_start = max(hiit_start, region_start)
        overlap_end = min(hiit_end, region_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_region = (region_start, region_end)
    
    if best_region is None:
        print("DEBUG: No suitable region found, returning original boundaries")
        return hiit_start, hiit_end
    
    correlation_start, correlation_end = best_region
    print(f"DEBUG: Best correlation region: {correlation_start}-{correlation_end} (overlap: {best_overlap})")
    
    # Step 2: Refine start boundary - find largest speed derivative change near correlation start
    search_window = int(dominant_period * 0.5)  # Search within 0.5 periods
    start_search_begin = max(0, correlation_start - search_window)
    start_search_end = min(len(df), correlation_start + search_window)
    
    if start_search_end > start_search_begin:
        speed_data = df['enhanced_speed'].iloc[start_search_begin:start_search_end].ffill().bfill()
        
        if len(speed_data) > 5:
            # Compute speed derivative
            speed_derivative = np.gradient(speed_data.values)
            
            # Find top 5% of derivative values (largest changes)
            threshold_95 = np.percentile(np.abs(speed_derivative), 95)
            large_changes = np.where(np.abs(speed_derivative) >= threshold_95)[0]
            
            print(f"DEBUG: Found {len(large_changes)} large speed changes in start search window")
            
            if len(large_changes) > 0:
                # Find the change closest to the correlation start
                correlation_start_local = correlation_start - start_search_begin
                closest_change = large_changes[np.argmin(np.abs(large_changes - correlation_start_local))]
                refined_start = start_search_begin + closest_change
                
                print(f"DEBUG: Largest speed change at {closest_change}, refined start: {refined_start}")
            else:
                refined_start = correlation_start
                print(f"DEBUG: No large speed changes found, using correlation start: {refined_start}")
        else:
            refined_start = correlation_start
            print(f"DEBUG: Not enough speed data, using correlation start: {refined_start}")
    else:
        refined_start = correlation_start
        print(f"DEBUG: Invalid start search window, using correlation start: {refined_start}")
    
    # Step 3: Refine end boundary - find nearest heart rate baseline return near correlation end
    end_search_begin = max(0, correlation_end - search_window)
    end_search_end = min(len(df), correlation_end + search_window)
    
    if end_search_end > end_search_begin:
        hr_data = df['heart_rate'].iloc[end_search_begin:end_search_end].ffill().bfill()
        
        if len(hr_data) > 10:
            # Calculate baseline heart rate from the HIIT period
            hiit_hr_data = df['heart_rate'].iloc[hiit_start:hiit_end].ffill().bfill()
            hiit_baseline = hiit_hr_data.min()
            
            # Calculate pre-HIIT baseline
            pre_hiit_start = max(0, hiit_start - int(dominant_period * 2))
            pre_hiit_hr_data = df['heart_rate'].iloc[pre_hiit_start:hiit_start].ffill().bfill()
            pre_hiit_baseline = pre_hiit_hr_data.mean() if len(pre_hiit_hr_data) > 10 else hiit_baseline
            
            # Use the higher baseline as target (more conservative)
            target_baseline = max(hiit_baseline, pre_hiit_baseline)
            baseline_tolerance = 5
            
            print(f"DEBUG: Target baseline: {target_baseline:.1f} (HIIT: {hiit_baseline:.1f}, Pre-HIIT: {pre_hiit_baseline:.1f})")
            
            # Find points where heart rate is at baseline
            baseline_mask = hr_data <= (target_baseline + baseline_tolerance)
            baseline_points = np.where(baseline_mask)[0]
            
            print(f"DEBUG: Found {len(baseline_points)} baseline points in end search window")
            
            if len(baseline_points) > 0:
                # Find the baseline point closest to the correlation end
                correlation_end_local = correlation_end - end_search_begin
                closest_baseline = baseline_points[np.argmin(np.abs(baseline_points - correlation_end_local))]
                refined_end = end_search_begin + closest_baseline
                
                print(f"DEBUG: Closest baseline at {closest_baseline}, refined end: {refined_end}")
            else:
                refined_end = correlation_end
                print(f"DEBUG: No baseline points found, using correlation end: {refined_end}")
        else:
            refined_end = correlation_end
            print(f"DEBUG: Not enough HR data, using correlation end: {refined_end}")
    else:
        refined_end = correlation_end
        print(f"DEBUG: Invalid end search window, using correlation end: {refined_end}")
    
    # Ensure refined boundaries are reasonable
    min_duration = int(dominant_period * 1.5)
    if refined_end - refined_start < min_duration:
        print(f"DEBUG: Refined window too short ({refined_end - refined_start} < {min_duration}), using original")
        return hiit_start, hiit_end
    
    print(f"DEBUG: Final refined boundaries: {refined_start}-{refined_end}")
    return refined_start, refined_end

def segment_intervals_speed_edges(df_hiit):
    """
    Segment HIIT data into intervals based on speed rising edges from low to high states.
    
    Args:
        df_hiit: DataFrame containing HIIT period data
    
    Returns:
        List of interval dictionaries with precise boundaries
    """
    if df_hiit.empty or 'enhanced_speed' not in df_hiit.columns:
        return []
    
    # Get speed data and apply low-pass filtering to reduce noise
    speed_data = df_hiit['enhanced_speed'].ffill().bfill()
    
    # Apply low-pass filter to reduce noise and focus on the main speed patterns
    from scipy.signal import butter, filtfilt
    sampling_rate = 1.0  # Hz
    cutoff_freq = 0.1  # 10 second period
    nyquist = sampling_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    
    # Design Butterworth low-pass filter
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    speed_filtered = filtfilt(b, a, speed_data.values)
    
    # Classify speed states using K-means (high vs low)
    speed_2d = speed_filtered.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(speed_2d)
    
    # Identify which cluster is "low" (lower mean speed)
    cluster_means = [speed_filtered[cluster_labels == i].mean() for i in range(2)]
    low_cluster = np.argmin(cluster_means)
    
    # Create low/high state mask
    low_state_mask = (cluster_labels == low_cluster)
    
    # Find transitions from low to high (rising edges)
    transitions = np.diff(low_state_mask.astype(int))
    low_to_high = np.where(transitions == -1)[0]  # Low ends, high starts
    
    if len(low_to_high) < 2:
        return []
    
    # Post-process transitions using expected period
    # Estimate expected interval period from the data
    if len(low_to_high) > 2:
        intervals_between = np.diff(low_to_high)
        expected_period = np.median(intervals_between)
        period_tolerance = expected_period * 0.3  # 30% tolerance
    else:
        expected_period = 120  # Default 2 minutes if not enough data
        period_tolerance = 36  # 30% of 120 seconds
    
    # Filter transitions based on period consistency
    filtered_transitions = [low_to_high[0]]  # Always keep the first transition
    
    for i in range(1, len(low_to_high)):
        current_transition = low_to_high[i]
        last_transition = filtered_transitions[-1]
        time_since_last = current_transition - last_transition
        
        # Keep transition if it's within expected period range
        if time_since_last >= (expected_period - period_tolerance):
            filtered_transitions.append(current_transition)
        # If too close, keep the one that's closer to expected period
        elif time_since_last < (expected_period - period_tolerance):
            # Check if this transition is better than the last one we kept
            if len(filtered_transitions) > 1:
                prev_time_since_last = last_transition - filtered_transitions[-2]
                if abs(time_since_last - expected_period) < abs(prev_time_since_last - expected_period):
                    # Replace the last transition with this one
                    filtered_transitions[-1] = current_transition
    
    low_to_high = np.array(filtered_transitions)
    
    intervals = []
    
    # Create intervals from low-to-high transitions
    for i in range(len(low_to_high)):
        interval_start = low_to_high[i]
        
        # End of this interval is start of next, or end of data
        if i + 1 < len(low_to_high):
            interval_end = low_to_high[i + 1]
        else:
            interval_end = len(df_hiit)
        
        # Calculate metrics for this interval
        interval_section = df_hiit.iloc[interval_start:interval_end]
        
        # Find the peak heart rate within this interval
        if 'heart_rate' in interval_section.columns and not interval_section.empty:
            hr_data = interval_section['heart_rate'].ffill().bfill()
            hr_peaks, _ = find_peaks(hr_data.values, 
                                   height=np.mean(hr_data) + 0.3 * np.std(hr_data),
                                   distance=10)
            
            if len(hr_peaks) > 0:
                # Use the first peak as the high intensity period
                high_start = interval_start + hr_peaks[0]
                high_end = min(high_start + 60, interval_end)  # 60 second high period
            else:
                # No clear peak, use first half as high intensity
                high_start = interval_start
                high_end = interval_start + (interval_end - interval_start) // 2
        else:
            high_start = interval_start
            high_end = interval_start + (interval_end - interval_start) // 2
        
        # Recovery period is the rest
        recovery_start = high_end
        recovery_end = interval_end
        
        # Calculate metrics
        high_section = df_hiit.iloc[high_start:high_end]
        recovery_section = df_hiit.iloc[recovery_start:recovery_end]
        
        # Defensive checks for empty sections
        distance_start = high_section['distance'].iloc[0] if 'distance' in high_section.columns and not high_section.empty else 0
        if 'distance' in recovery_section.columns and not recovery_section.empty:
            distance_end = recovery_section['distance'].iloc[-1]
        else:
            distance_end = 0
        if 'altitude' in high_section.columns and not high_section.empty:
            dA_up = high_section['altitude'].diff().sum()
        else:
            dA_up = 0
        if 'altitude' in recovery_section.columns and not recovery_section.empty:
            dA_down = recovery_section['altitude'].diff().sum()
        else:
            dA_down = 0
        if 'temperature' in high_section.columns and not high_section.empty:
            temperature_mean = high_section['temperature'].mean()
        else:
            temperature_mean = 0
        
        interval_data = {
            'interval_num': i + 1,
            'interval_start': interval_start,
            'interval_end': interval_end,
            'high_start': high_start,
            'high_end': high_end,
            'recovery_start': recovery_start,
            'recovery_end': recovery_end,
            'high_duration': high_end - high_start,
            'recovery_duration': recovery_end - recovery_start,
            'high_speed_mean': high_section['enhanced_speed'].mean() if 'enhanced_speed' in high_section.columns and not high_section.empty else 0,
            'high_speed_std': high_section['enhanced_speed'].std() if 'enhanced_speed' in high_section.columns and not high_section.empty else 0,
            'recovery_speed_mean': recovery_section['enhanced_speed'].mean() if 'enhanced_speed' in recovery_section.columns and not recovery_section.empty else 0,
            'recovery_speed_std': recovery_section['enhanced_speed'].std() if 'enhanced_speed' in recovery_section.columns and not recovery_section.empty else 0,
            'high_hr_mean': high_section['heart_rate'].mean() if not high_section.empty else 0,
            'recovery_hr_mean': recovery_section['heart_rate'].mean() if not recovery_section.empty else 0,
            'temperature_mean': temperature_mean,
            'distance_start': distance_start,
            'distance_end': distance_end,
            'dA_up': dA_up,
            'dA_down': dA_down
        }
        
        intervals.append(interval_data)
    
    return intervals

def compute_template_correlation(df, intervals, hr_filtered, dominant_period):
    """
    Compute template correlation signal for visualization only.
    This is not used for detection, just for comparison.
    
    Args:
        df: DataFrame with heart rate data
        intervals: List of detected intervals
        hr_filtered: Filtered heart rate signal
        dominant_period: Dominant period from frequency analysis
    
    Returns:
        template_correlation: Array of template correlation values
        mean_template: The mean interval template for plotting
    """
    if len(intervals) < 2:
        print("DEBUG: Not enough intervals for template correlation")
        return None, None
    
    # Create template from all intervals (not just first few)
    template_length = int(dominant_period)
    print(f"DEBUG: Template length: {template_length}")
    
    # Extract heart rate and speed segments for each interval and average them
    hr_template_segments = []
    speed_template_segments = []
    
    for i, interval in enumerate(intervals):
        if 'interval_start' in interval and 'interval_end' in interval:
            interval_start = interval['interval_start']
            interval_end = interval['interval_end']
            
            print(f"DEBUG: Interval {i+1}: {interval_start} to {interval_end} (length: {interval_end - interval_start})")
            
            if interval_start < len(hr_filtered) and interval_end <= len(hr_filtered):
                # Extract heart rate segment (will be smoothed later)
                hr_segment = hr_filtered[interval_start:interval_end]
                
                # Extract speed segment (will be smoothed later)
                if 'enhanced_speed' in df.columns:
                    speed_segment = df['enhanced_speed'].iloc[interval_start:interval_end].values
                elif 'speed' in df.columns:
                    speed_segment = df['speed'].iloc[interval_start:interval_end].values
                else:
                    speed_segment = np.zeros(interval_end - interval_start)
                
                print(f"DEBUG: Segment length: {len(hr_segment)}, template_length: {template_length}")
                if len(hr_segment) >= template_length:
                    # Resize to template length
                    hr_segment = hr_segment[:template_length]
                    speed_segment = speed_segment[:template_length]
                    
                    hr_template_segments.append(hr_segment)
                    speed_template_segments.append(speed_segment)
                    print(f"DEBUG: Added segment {i+1}, length: {len(hr_segment)}")
                else:
                    print(f"DEBUG: Segment {i+1} too short: {len(hr_segment)} < {template_length}")
            else:
                print(f"DEBUG: Interval {i+1} indices out of range: {interval_start}, {interval_end} vs {len(hr_filtered)}")
    
    print(f"DEBUG: Total template segments: {len(hr_template_segments)}")
    
    if not hr_template_segments:
        print("DEBUG: No valid template segments found")
        return None, None
    
    # Design filters for template smoothing
    from scipy.signal import butter, filtfilt
    
    # Filter heart rate signal (already filtered, but apply additional smoothing)
    hr_smooth_cutoff = 0.02  # 50 second period
    hr_smooth_nyquist = 1.0 / 2
    hr_smooth_normal_cutoff = hr_smooth_cutoff / hr_smooth_nyquist
    hr_smooth_b, hr_smooth_a = butter(4, hr_smooth_normal_cutoff, btype='low', analog=False)
    
    # Use median filtering for speed data - better for square wave patterns
    from scipy.signal import medfilt
    speed_median_window = 15  # 15-second median window
    speed_b = None  # Not used for median filtering
    speed_a = None  # Not used for median filtering
    
    # Apply the same smoothing to template segments
    smoothed_hr_segments = []
    smoothed_speed_segments = []
    
    for hr_seg, speed_seg in zip(hr_template_segments, speed_template_segments):
        # Apply smoothing to HR segment
        hr_smoothed_seg = filtfilt(hr_smooth_b, hr_smooth_a, hr_seg)
        smoothed_hr_segments.append(hr_smoothed_seg)
        
        # Apply median filtering to speed segment (better for square waves)
        speed_smoothed_seg = medfilt(speed_seg, kernel_size=speed_median_window)
        smoothed_speed_segments.append(speed_smoothed_seg)
    
    # Average the smoothed segments to create templates
    mean_hr_template = np.mean(smoothed_hr_segments, axis=0)
    mean_speed_template = np.mean(smoothed_speed_segments, axis=0)
    print(f"DEBUG: Mean HR template length: {len(mean_hr_template)}, Mean speed template length: {len(mean_speed_template)}")
    
    # Create multiple phase-shifted versions of both templates
    num_phases = 8  # Number of phase shifts to try
    phase_shift = template_length // num_phases
    hr_phase_shifted_templates = []
    speed_phase_shifted_templates = []
    
    for phase in range(num_phases):
        # Create phase-shifted templates by rolling the arrays
        shifted_hr_template = np.roll(mean_hr_template, phase * phase_shift)
        shifted_speed_template = np.roll(mean_speed_template, phase * phase_shift)
        hr_phase_shifted_templates.append(shifted_hr_template)
        speed_phase_shifted_templates.append(shifted_speed_template)
    
    print(f"DEBUG: Created {num_phases} phase-shifted templates for both HR and speed with shift of {phase_shift} samples")
    
    # Parallel template correlation computation
    
    # Apply low-pass filtering to both signals for template matching
    # Use the same filters designed for template smoothing
    hr_smoothed = filtfilt(hr_smooth_b, hr_smooth_a, hr_filtered)
    
    # Filter speed data
    if 'enhanced_speed' in df.columns:
        speed_data = df['enhanced_speed'].values
    elif 'speed' in df.columns:
        speed_data = df['speed'].values
    else:
        speed_data = np.zeros(len(hr_filtered))
    
    # Apply median filtering to speed data (better for square waves)
    speed_smoothed = medfilt(speed_data, kernel_size=speed_median_window)
    
    # Debug: Check the effect of filtering
    speed_std_original = np.std(speed_data)
    speed_std_smoothed = np.std(speed_smoothed)
    speed_range_original = np.max(speed_data) - np.min(speed_data)
    speed_range_smoothed = np.max(speed_smoothed) - np.min(speed_smoothed)
    print(f"DEBUG: Speed filtering effect - Original std: {speed_std_original:.3f}, Smoothed std: {speed_std_smoothed:.3f}")
    print(f"DEBUG: Speed range - Original: {speed_range_original:.3f}, Smoothed: {speed_range_smoothed:.3f}")
    print(f"DEBUG: Speed median window: {speed_median_window}s")
    print(f"DEBUG: Applied median filtering to speed, low-pass to HR for template matching")
    
    # Split work into chunks for parallel processing
    total_positions = len(hr_filtered) - template_length + 1
    num_cores = min(mp.cpu_count(), 8)  # Limit to 8 cores to avoid overwhelming the system
    chunk_size = max(1, total_positions // num_cores)
    
    print(f"DEBUG: Parallelizing template correlation with {num_cores} cores, {chunk_size} positions per chunk")
    
    # Prepare chunk data
    chunk_data_list = []
    for chunk_idx in range(num_cores):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_positions)
        
        if chunk_start >= total_positions:
            break
            
        # Extract data for this chunk (need extra template_length-1 samples for overlap)
        hr_chunk = hr_smoothed[chunk_start:chunk_end + template_length - 1]
        speed_chunk = speed_smoothed[chunk_start:chunk_end + template_length - 1]
        
        chunk_data = (chunk_start, chunk_end, hr_chunk, speed_chunk, 
                     hr_phase_shifted_templates, speed_phase_shifted_templates, num_phases, template_length)
        chunk_data_list.append(chunk_data)
    
    # Process chunks in parallel
    with mp.Pool(processes=num_cores) as pool:
        chunk_results = pool.map(compute_correlation_chunk, chunk_data_list)
    
    # Combine results in the correct order
    template_correlation = [0] * total_positions  # Initialize with zeros
    for chunk_idx, chunk_result in enumerate(chunk_results):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_positions)
        
        # Place chunk results in the correct positions
        for i, correlation_value in enumerate(chunk_result):
            if chunk_start + i < total_positions:
                template_correlation[chunk_start + i] = correlation_value
    
    # Pad to match original signal length
    pad_size = len(hr_filtered) - len(template_correlation)
    template_correlation = [0] * (pad_size // 2) + template_correlation + [0] * (pad_size - pad_size // 2)
    
    print(f"DEBUG: Template correlation created, final length: {len(template_correlation)}")
    
    # Normalize template correlation to [0, 1] range
    if template_correlation:
        template_corr_array = np.array(template_correlation)
        if template_corr_array.max() > template_corr_array.min():
            template_correlation = (template_corr_array - template_corr_array.min()) / (template_corr_array.max() - template_corr_array.min())
        else:
            template_correlation = template_corr_array
    
    # Apply gentle edge sharpening to template correlation for sharper boundaries
    from scipy import ndimage
    template_corr_array = np.array(template_correlation)
    
    # Use smaller kernel and higher threshold for less aggressive sharpening
    kernel_size = max(3, int(dominant_period // 20))  # Smaller kernel
    threshold = 0.3  # Lower threshold to preserve more signal
    template_corr_sharpened = ndimage.binary_opening(template_corr_array > threshold, structure=np.ones(kernel_size))
    template_corr_sharpened = ndimage.binary_closing(template_corr_sharpened, structure=np.ones(kernel_size))
    
    # Convert back to continuous values with sharp edges, but preserve original values
    template_correlation = template_corr_array * (0.7 + 0.3 * template_corr_sharpened.astype(float))
    
    # Return both templates for plotting
    return template_correlation, mean_hr_template, mean_speed_template
