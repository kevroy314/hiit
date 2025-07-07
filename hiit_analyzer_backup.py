import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from fitparse import FitFile
import glob
import os
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Callable
import warnings
import json
from parse_and_plot_fit import segment_intervals
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="HIIT Analyzer",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2E3440;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
        margin: 0.5rem 0;
    }
    .interval-card {
        background-color: #3B4252;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #4C566A;
    }
    .nav-link {
        display: inline-block;
        margin-right: 1.5rem;
        font-size: 1.2rem;
        color: #FF6B6B;
        text-decoration: none;
        font-weight: bold;
    }
    .nav-link.selected {
        color: #FFFFFF;
        border-bottom: 2px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme for fields organized by groups
FIELD_COLORS = {
    # Speeds - Yellow
    'enhanced_speed': '#FFFF44',
    'vertical_speed': '#FFFF44', 
    'speed': '#FFFF44',
    # Positions - Green
    'distance': '#44FF44',
    'position_lat': '#44FF44',
    'position_lon': '#44FF44',
    # Altitudes - Blue
    'altitude': '#4444FF',
    'enhanced_altitude': '#4444FF',
    # Temperature - White
    'temperature': '#FFFFFF',
    # Heart Rate - Red
    'heart_rate': '#FF4444',
}

def get_field_color(field_name):
    field_lower = field_name.lower()
    for key, color in FIELD_COLORS.items():
        if key in field_lower:
            return color
    return '#888888'

def get_field_group(field_name):
    """Return the group name for a field."""
    field_lower = field_name.lower()
    if any(speed in field_lower for speed in ['enhanced_speed', 'vertical_speed', 'speed']):
        return 'Speeds'
    elif any(pos in field_lower for pos in ['distance', 'position_lat', 'position_lon']):
        return 'Positions'
    elif any(alt in field_lower for alt in ['altitude', 'enhanced_altitude']):
        return 'Altitudes'
    elif 'temperature' in field_lower:
        return 'Temperature'
    elif 'heart_rate' in field_lower:
        return 'Heart Rate'
    else:
        return 'Other'

def get_manual_window_settings_filename(filename):
    """Get the filename for storing manual window settings."""
    import os
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return f"./data/{base_name}_manual_window.json"

def load_manual_window_settings(filename):
    """Load manual window settings from JSON file."""
    settings_file = get_manual_window_settings_filename(filename)
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            return settings.get('manual_start'), settings.get('manual_end')
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None

def save_manual_window_settings(filename, start_idx, end_idx):
    """Save manual window settings to JSON file."""
    import os
    settings_file = get_manual_window_settings_filename(filename)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(settings_file), exist_ok=True)
    
    settings = {
        'manual_start': start_idx,
        'manual_end': end_idx,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)

def search_for_intervals_backward(df, start_idx, dominant_period, num_intervals, search_samples):
    """Search backward from start_idx to find additional intervals."""
    intervals = []
    current_idx = start_idx
    
    for i in range(num_intervals):
        # Calculate expected interval start (going backward)
        expected_start = current_idx - int(dominant_period)
        
        # Search within search range
        search_start = max(0, expected_start - search_samples)
        search_end = min(len(df), expected_start + search_samples)
        
        # Look for speed edges in this range
        df_search = df.iloc[search_start:search_end]
        if len(df_search) < 10:  # Need minimum data
            break
            
        # Use the same interval detection logic
        detected_intervals = segment_intervals_speed_edges(df_search)
        
        if detected_intervals:
            # Take the last interval found (most recent)
            interval = detected_intervals[-1]
            # Adjust indices to match full df
            for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                if k in interval:
                    interval[k] += search_start
            
            intervals.insert(0, interval)  # Insert at beginning (backward order)
            current_idx = interval['interval_start']  # Move to this interval's start
        else:
            break
    
    return intervals

def search_for_intervals_forward(df, end_idx, dominant_period, num_intervals, search_samples):
    """Search forward from end_idx to find additional intervals."""
    intervals = []
    current_idx = end_idx
    
    for i in range(num_intervals):
        # Calculate expected interval start (going forward)
        expected_start = current_idx + int(dominant_period)
        
        # Search within search range
        search_start = max(0, expected_start - search_samples)
        search_end = min(len(df), expected_start + search_samples)
        
        # Look for speed edges in this range
        df_search = df.iloc[search_start:search_end]
        if len(df_search) < 10:  # Need minimum data
            break
            
        # Use the same interval detection logic
        detected_intervals = segment_intervals_speed_edges(df_search)
        
        if detected_intervals:
            # Take the first interval found (earliest)
            interval = detected_intervals[0]
            # Adjust indices to match full df
            for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                if k in interval:
                    interval[k] += search_start
            
            intervals.append(interval)
            current_idx = interval['interval_end']  # Move to this interval's end
        else:
            break
    
    return intervals

def load_fit_data(fit_filename):
    """Load and parse FIT file data."""
    fitfile = FitFile(fit_filename)
    
    # Store time series data
    timestamps = []
    values = {}
    all_field_names = set()
    
    # First pass: collect all field names
    for record in fitfile.get_messages('record'):
        for field in record:
            if field.name != 'timestamp':
                all_field_names.add(field.name)
    
    # Initialize value arrays
    for field_name in all_field_names:
        values[field_name] = []
    
    # Second pass: collect data
    for record in fitfile.get_messages('record'):
        record_data = {}
        timestamp = None
        
        for field in record:
            record_data[field.name] = field.value
            if field.name == 'timestamp':
                timestamp = field.value
        
        if timestamp:
            timestamps.append(timestamp)
            for field_name in all_field_names:
                values[field_name].append(record_data.get(field_name, None))
    
    # Convert to DataFrame
    df = pd.DataFrame(values, index=timestamps)
    return df

import scipy.fft as fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks

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
    heart_rate = df['heart_rate'].fillna(method='ffill').fillna(method='bfill')
    
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
    # Window size should be ~4x the dominant period for good frequency resolution
    window_size = int(dominant_period * 4)
    if window_size > len(hr_filtered) // 2:
        window_size = len(hr_filtered) // 2
    
    # Calculate rolling correlation with the dominant frequency
    frequency_correlation = []
    
    for i in range(len(hr_filtered) - window_size + 1):
        window_data = hr_filtered[i:i + window_size]
        
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
        
        frequency_correlation.append(correlation)
    
    # Pad the correlation array to match the original signal length
    pad_size = len(hr_filtered) - len(frequency_correlation)
    frequency_correlation = [0] * (pad_size // 2) + frequency_correlation + [0] * (pad_size - pad_size // 2)
    
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
    refined_start, refined_end = refine_hiit_boundaries(df, best_start, best_end, dominant_period, correlation_array, correlation_threshold)
    
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
        'refined_end': refined_end
    }
    
    return refined_start, refined_end, frequency_info

def refine_hiit_boundaries(df, hiit_start, hiit_end, dominant_period, frequency_correlation, correlation_threshold):
    """
    Refine HIIT boundaries using speed clustering to find walking periods.
    
    Args:
        df: DataFrame with speed data
        hiit_start, hiit_end: Initial HIIT boundaries
        dominant_period: Dominant period from frequency analysis
        frequency_correlation: Array of frequency correlation values
        correlation_threshold: Optimal correlation threshold
    
    Returns:
        refined_start, refined_end: Refined boundaries
    """
    if 'enhanced_speed' not in df.columns:
        return hiit_start, hiit_end
    
    # Get speed data around the HIIT window
    # Look back 3x dominant period for start, forward 2x for end
    lookback = int(dominant_period * 3)
    lookforward = int(dominant_period * 2)
    
    start_search_begin = max(0, hiit_start - lookback)
    end_search_end = min(len(df), hiit_end + lookforward)
    
    # Extract speed data for clustering
    speed_data = df['enhanced_speed'].iloc[start_search_begin:end_search_end].fillna(method='ffill').fillna(method='bfill')
    
    if len(speed_data) < 10:
        return hiit_start, hiit_end
    
    # Cluster speed data to identify walking vs running
    speed_2d = speed_data.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # 3 clusters: walking, running, transition
    cluster_labels = kmeans.fit_predict(speed_2d)
    
    # Identify which cluster is walking (lowest mean speed)
    cluster_means = [speed_data[cluster_labels == i].mean() for i in range(3)]
    walking_cluster = np.argmin(cluster_means)
    
    # Create walking mask
    walking_mask = (cluster_labels == walking_cluster)
    
    # Refine start: find the last walking period before HIIT
    pre_hiit_speed = speed_data.iloc[:hiit_start - start_search_begin]
    pre_hiit_walking = walking_mask[:len(pre_hiit_speed)]
    
    refined_start = hiit_start
    if np.any(pre_hiit_walking):
        # Find the last continuous walking period
        walking_periods = []
        start_idx = None
        
        for i, is_walking in enumerate(pre_hiit_walking):
            if is_walking and start_idx is None:
                start_idx = i
            elif not is_walking and start_idx is not None:
                walking_periods.append((start_idx, i))
                start_idx = None
        
        if start_idx is not None:
            walking_periods.append((start_idx, len(pre_hiit_walking)))
        
        if walking_periods:
            # Use the last walking period that's close to the HIIT start
            last_walking = walking_periods[-1]
            walking_end = last_walking[1]
            
            # Only adjust if the walking period is reasonably close (within 1.5x dominant period)
            if (hiit_start - start_search_begin - walking_end) < (dominant_period * 1.5):
                refined_start = start_search_begin + walking_end
    
    # Refine end: use adaptive threshold widening for fatigue
    # Look further ahead for the end boundary (up to 3x dominant period)
    end_region_start = max(0, hiit_end - int(dominant_period * 1.5))
    end_region_end = min(len(frequency_correlation), hiit_end + int(dominant_period * 3))
    
    if end_region_end > end_region_start:
        end_correlations = frequency_correlation[end_region_start:end_region_end]
        
        # Use a much more lenient threshold for the end region (accounting for fatigue)
        fatigue_threshold = correlation_threshold * 0.6  # 40% more lenient
        
        # Find where correlation drops below fatigue threshold
        above_fatigue_threshold = end_correlations > fatigue_threshold
        
        if np.any(above_fatigue_threshold):
            # Find the last point above the fatigue threshold
            last_above = end_region_start + np.where(above_fatigue_threshold)[0][-1]
            
            # Add some buffer to ensure we don't cut off mid-interval
            # Look for a natural drop in correlation (not just noise)
            buffer_size = int(dominant_period * 0.5)  # Half a cycle buffer
            potential_end = min(last_above + buffer_size, len(df))
            
            # Ensure we don't go too far beyond the original end
            max_extension = hiit_end + int(dominant_period * 2)  # Max 2 cycles beyond original
            refined_end = min(potential_end, max_extension)
        else:
            refined_end = hiit_end
    else:
        refined_end = hiit_end
    
    # Fine-tune end boundary using heart rate trough detection
    if 'heart_rate' in df.columns and refined_end < len(df):
        # Look for the heart rate trough in the final region
        final_region_start = max(0, refined_end - int(dominant_period * 1.5))
        final_region_end = min(len(df), refined_end + int(dominant_period * 0.5))
        
        if final_region_end > final_region_start:
            final_hr_data = df['heart_rate'].iloc[final_region_start:final_region_end].fillna(method='ffill').fillna(method='bfill')
            
            if len(final_hr_data) > 10:  # Need sufficient data
                # Find valleys (troughs) in the heart rate
                from scipy.signal import find_peaks
                valleys, _ = find_peaks(-final_hr_data.values, 
                                      height=-np.mean(final_hr_data) + 0.5 * np.std(final_hr_data),
                                      distance=int(dominant_period * 0.3))  # Minimum distance between valleys
                
                if len(valleys) > 0:
                    # Find the valley closest to the current refined_end
                    current_end_idx = refined_end - final_region_start
                    closest_valley_idx = min(valleys, key=lambda x: abs(x - current_end_idx))
                    
                    # Only adjust if the valley is after the current end and within reasonable range
                    if closest_valley_idx > current_end_idx and closest_valley_idx < len(final_hr_data):
                        # Convert back to global index
                        trough_end = final_region_start + closest_valley_idx
                        
                        # Ensure we don't extend too far
                        max_trough_extension = refined_end + int(dominant_period * 0.5)
                        refined_end = min(trough_end, max_trough_extension)
    
    # Ensure refined boundaries are reasonable
    min_duration = int(dominant_period * 1.5)
    if refined_end - refined_start < min_duration:
        return hiit_start, hiit_end
    
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
    speed_data = df_hiit['enhanced_speed'].fillna(method='ffill').fillna(method='bfill')
    
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
            hr_data = interval_section['heart_rate'].fillna(method='ffill').fillna(method='bfill')
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

# Define curve fitting functions for heart rate analysis
def exp_rise(t, HR0, HRpeak, tau):
    return HR0 + (HRpeak - HR0) * (1 - np.exp(-t / tau))

def exp_fall(t, HRpeak, HRbaseline, tau):
    return HRbaseline + (HRpeak - HRbaseline) * np.exp(-t / tau)

def fit_hr_curve(t: np.ndarray, hr: np.ndarray, model: str) -> Tuple[np.ndarray, Callable]:
    """
    Fit heart rate data to a specified model.
    
    Parameters:
    - t: time values (numpy array)
    - hr: heart rate values (numpy array)
    - model: one of ["exp_rise", "exp_fall"]
    
    Returns:
    - popt: optimized parameters
    - model_func: function used for fitting
    """
    models = {
        "exp_rise": (exp_rise, [90, 180, 30]),
        "exp_fall": (exp_fall, [180, 90, 60]),
    }
    
    if model not in models:
        raise ValueError(f"Model '{model}' is not supported.")
    
    func, p0 = models[model]
    try:
        popt, _ = curve_fit(func, t, hr, p0=p0, maxfev=1000)
        return popt, func
    except:
        # Return default parameters if fitting fails
        return p0, func

def analyze_interval_performance(df, interval):
    """
    Analyze a single interval to extract performance metrics.
    
    Args:
        df: DataFrame with the full dataset
        interval: Dictionary with interval information
    
    Returns:
        Dictionary with performance metrics
    """
    if 'high_start' not in interval or 'high_end' not in interval:
        return None
    
    high_start = interval['high_start']
    high_end = interval['high_end']
    
    # Extract high-intensity period data
    high_period = df.iloc[high_start:high_end]
    
    if high_period.empty or 'heart_rate' not in high_period.columns:
        return None
    
    # Get heart rate data for the high period
    hr_data = high_period['heart_rate'].fillna(method='ffill').fillna(method='bfill')
    time_data = np.arange(len(hr_data))
    
    if len(hr_data) < 10:  # Need sufficient data for fitting
        return None
    
    # Find peak and baseline heart rates
    hr_peak = hr_data.max()
    hr_baseline = hr_data.min()
    
    # Split the interval into rise and fall phases
    mid_point = len(hr_data) // 2
    rise_data = hr_data.iloc[:mid_point]
    fall_data = hr_data.iloc[mid_point:]
    
    rise_time = np.arange(len(rise_data))
    fall_time = np.arange(len(fall_data))
    
    # Fit exponential curves
    try:
        rise_params, rise_func = fit_hr_curve(rise_time, rise_data.values, "exp_rise")
        fall_params, fall_func = fit_hr_curve(fall_time, fall_data.values, "exp_fall")
        
        # Extract tau values (time constants)
        raw_tau_rise = rise_params[2] if len(rise_params) > 2 else 30
        raw_tau_fall = fall_params[2] if len(fall_params) > 2 else 60
        
        # Debug: print raw tau values for first few intervals
        if interval.get('interval_num', 0) < 3:
            print(f"Interval {interval.get('interval_num', 0)}: raw_tau_rise={raw_tau_rise:.3f}, raw_tau_fall={raw_tau_fall:.3f}")
            print(f"  HR range: {hr_baseline:.1f} -> {hr_peak:.1f} (change: {hr_peak-hr_baseline:.1f})")
            print(f"  Rise data length: {len(rise_data)}, Fall data length: {len(fall_data)}")
            print(f"  Rise params: {rise_params}")
            print(f"  Fall params: {fall_params}")
        
        # Use raw tau values directly - these are the actual time constants
        # Smaller tau = faster response, larger tau = slower response
        tau_rise = raw_tau_rise
        tau_fall = raw_tau_fall
        
        # Don't clamp - let natural values show through
        # Only ensure we don't have negative values
        tau_rise = max(0, tau_rise)
        tau_fall = max(0, tau_fall)
        
        # Debug: print final tau values for first few intervals
        if interval.get('interval_num', 0) < 3:
            print(f"  Final tau_rise={tau_rise:.3f}, tau_fall={tau_fall:.3f}")
            print(f"  Rise time constant: {raw_tau_rise:.3f} seconds")
            print(f"  Fall time constant: {raw_tau_fall:.3f} seconds")
            print("---")
        

        
    except:
        tau_rise = 1.0  # Default reasonable rate
        tau_fall = 1.0  # Default reasonable rate
    
    # Calculate responsiveness as average rate of change
    responsiveness = (tau_rise + tau_fall) / 2
    
    # Analyze speed during high period
    if 'enhanced_speed' in high_period.columns:
        speed_data = high_period['enhanced_speed'].fillna(method='ffill').fillna(method='bfill')
        
        # Use clustering to separate high and low speed periods
        if len(speed_data) > 5:
            speed_2d = speed_data.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(speed_2d)
            
            # Identify high speed cluster
            cluster_means = [speed_data[cluster_labels == i].mean() for i in range(2)]
            high_speed_cluster = np.argmax(cluster_means)
            
            # Get high speed data
            high_speed_data = speed_data[cluster_labels == high_speed_cluster]
            
            speed_mean = high_speed_data.mean()
            speed_std = high_speed_data.std()
        else:
            speed_mean = speed_data.mean()
            speed_std = speed_data.std()
        
        # Calculate speed variability index
        speed_variability = speed_std / speed_mean if speed_mean > 0 else 0
    else:
        speed_variability = 0
    
    return {
        'interval_num': interval['interval_num'],
        'hr_peak': hr_peak,
        'hr_baseline': hr_baseline,
        'tau_rise': tau_rise,
        'tau_fall': tau_fall,
        'responsiveness': responsiveness,
        'speed_variability': speed_variability,
        'speed_mean': speed_mean if 'enhanced_speed' in high_period.columns else 0,
        'speed_std': speed_std if 'enhanced_speed' in high_period.columns else 0
    }

def plot_interval_analysis(df, intervals, filename):
    """Create interval analysis plots."""
    if not intervals:
        st.warning("No intervals detected. Try adjusting the parameters.")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Speed', 'Heart Rate', 'Altitude', 'Temperature'],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Plot speed
    if 'enhanced_speed' in df.columns:
        speed_data = df['enhanced_speed'].dropna()
        fig.add_trace(
            go.Scatter(
                x=speed_data.index,
                y=speed_data.values,
                mode='lines',
                name='Speed',
                line=dict(color='#FF8800', width=2)
            ),
            row=1, col=1
        )
    
    # Plot heart rate
    if 'heart_rate' in df.columns:
        hr_data = df['heart_rate'].dropna()
        fig.add_trace(
            go.Scatter(
                x=hr_data.index,
                y=hr_data.values,
                mode='lines',
                name='Heart Rate',
                line=dict(color='#FF4444', width=2)
            ),
            row=2, col=1
        )
    
    # Plot altitude
    if 'altitude' in df.columns:
        alt_data = df['altitude'].dropna()
        fig.add_trace(
            go.Scatter(
                x=alt_data.index,
                y=alt_data.values,
                mode='lines',
                name='Altitude',
                line=dict(color='#8844FF', width=2)
            ),
            row=3, col=1
        )
    
    # Plot temperature
    if 'temperature' in df.columns:
        temp_data = df['temperature'].dropna()
        fig.add_trace(
            go.Scatter(
                x=temp_data.index,
                y=temp_data.values,
                mode='lines',
                name='Temperature',
                line=dict(color='#FFFFFF', width=2)
            ),
            row=4, col=1
        )
    
    # Add interval overlays
    colors = px.colors.qualitative.Set3
    for i, interval in enumerate(intervals):
        color = colors[i % len(colors)]
        
        # High intensity period
        high_start = df.index[interval['high_start']]
        high_end = df.index[min(interval['high_end'], len(df)-1)]
        
        # Recovery period
        recovery_start = df.index[interval['recovery_start']]
        recovery_end = df.index[min(interval['recovery_end'], len(df)-1)]
        
        # Add rectangles for intervals
        for row in range(1, 5):
            fig.add_vrect(
                x0=high_start, x1=high_end,
                fillcolor=color, opacity=0.3,
                layer="below", line_width=0,
                row=row, col=1
            )
            fig.add_vrect(
                x0=recovery_start, x1=recovery_end,
                fillcolor=color, opacity=0.1,
                layer="below", line_width=0,
                row=row, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"HIIT Interval Analysis - {os.path.basename(filename)}",
        template="plotly_dark",
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)
    fig.update_yaxes(title_text="Altitude (m)", row=3, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=4, col=1)
    
    return fig

def create_metrics_visualization(intervals_df):
    """Create PCA visualization of interval metrics."""
    if len(intervals_df) < 2:
        st.warning("Need at least 2 intervals for PCA analysis.")
        return None
    
    # Select features for PCA
    feature_columns = [
        'high_duration', 'recovery_duration',
        'high_speed_mean', 'high_speed_std',
        'recovery_speed_mean', 'recovery_speed_std',
        'high_hr_mean', 'recovery_hr_mean',
        'temperature_mean', 'dA_up', 'dA_down'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in intervals_df.columns]
    
    if len(available_features) < 2:
        st.warning("Not enough features available for PCA analysis.")
        return None
    
    # Prepare data
    X = intervals_df[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers+text',
            marker=dict(
                size=15,
                color=intervals_df['interval_num'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Interval #")
            ),
            text=intervals_df['interval_num'],
            textposition="middle center",
            textfont=dict(color="white", size=12),
            hovertemplate='<b>Interval %{text}</b><br>' +
                         '<br>'.join([f'{col}: %{{customdata[{i}]:.2f}}' 
                                     for i, col in enumerate(available_features)]) +
                         '<extra></extra>',
            customdata=X.values
        )
    )
    
    # Add arrows showing progression
    for i in range(len(pca_result) - 1):
        fig.add_trace(
            go.Scatter(
                x=[pca_result[i, 0], pca_result[i+1, 0]],
                y=[pca_result[i, 1], pca_result[i+1, 1]],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.5)', width=2),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    fig.update_layout(
        title="HIIT Performance Metrics - PCA Visualization",
        template="plotly_dark",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        height=600
    )
    
    return fig

def plot_hiit_detection(df, hiit_start, hiit_end, intervals, frequency_info=None, filename=None):
    # Create subplots: main data, frequency analysis, interval overlay, speed stats, and histograms in grid
    if frequency_info:
        fig = make_subplots(
            rows=11, cols=2,
            subplot_titles=['Speed & Heart Rate with HIIT Detection', '', 'Frequency Correlation', '', 'Frequency Analysis', '', 'Period Power Spectrum', '', 'Interval Overlay', '', 'Speed Statistics', '', 'Tau Rise Distribution', 'Tau Fall Distribution', 'HR Baseline Distribution', 'HR Peak Distribution', 'Speed Mean Distribution', 'Speed Std Distribution'],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            row_heights=[0.15, 0.08, 0.08, 0.08, 0.12, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08],
            specs=[
                [{"secondary_y": True, "colspan": 2}, None],
                [{"secondary_y": False, "colspan": 2}, None],
                [{"secondary_y": False, "colspan": 2}, None],
                [{"secondary_y": False, "colspan": 2}, None],
                [{"secondary_y": True, "colspan": 2}, None],
                [{"secondary_y": False, "colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
    else:
        fig = make_subplots(
            rows=8, cols=2,
            subplot_titles=['Speed & Heart Rate with HIIT Detection', '', 'Interval Overlay', '', 'Speed Statistics', '', 'Tau Rise Distribution', 'Tau Fall Distribution', 'HR Baseline Distribution', 'HR Peak Distribution', 'Speed Mean Distribution', 'Speed Std Distribution'],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            row_heights=[0.25, 0.20, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08],
            specs=[
                [{"secondary_y": True, "colspan": 2}, None],
                [{"secondary_y": True, "colspan": 2}, None],
                [{"secondary_y": False, "colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
    
    # Main data plot - separate y-axes for speed and heart rate
    if 'enhanced_speed' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['enhanced_speed'],
            mode='lines',
            name='Speed (Raw)',
            line=dict(color=get_field_color('enhanced_speed'), width=1),
            opacity=0.7
        ), row=1, col=1, secondary_y=False)
        
        # Add smoothed speed if available (using simple moving average)
        if 'enhanced_speed' in df.columns:
            window_size = 5
            speed_smoothed = df['enhanced_speed'].rolling(window=window_size, center=True).mean().fillna(method='ffill').fillna(method='bfill')
            fig.add_trace(go.Scatter(
                x=df.index,
                y=speed_smoothed,
                mode='lines',
                name='Speed (Smoothed)',
                line=dict(color=get_field_color('enhanced_speed'), width=2)
            ), row=1, col=1, secondary_y=False)
    
    if 'heart_rate' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['heart_rate'],
            mode='lines',
            name='Heart Rate (Raw)',
            line=dict(color=get_field_color('heart_rate'), width=1),
            opacity=0.7
        ), row=1, col=1, secondary_y=True)
        
        # Add filtered heart rate if available
        if frequency_info and 'hr_filtered' in frequency_info:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=frequency_info['hr_filtered'],
                mode='lines',
                name='Heart Rate (Filtered)',
                line=dict(color=get_field_color('heart_rate'), width=2)
            ), row=1, col=1, secondary_y=True)
            
            # Add peak markers
            if 'peaks' in frequency_info:
                peak_times = [df.index[i] for i in frequency_info['peaks'] if 0 <= i < len(df)]
                peak_values = [frequency_info['hr_filtered'][i] for i in frequency_info['peaks'] if 0 <= i < len(df)]
                if peak_times:
                    fig.add_trace(go.Scatter(
                        x=peak_times,
                        y=peak_values,
                        mode='markers',
                        name='Peaks',
                        marker=dict(color='red', size=6, symbol='circle'),
                        showlegend=False
                    ), row=1, col=1, secondary_y=True)
    
    # HIIT window and transitions
    if hiit_start is not None and hiit_end is not None:
        hiit_x0 = df.index[hiit_start]
        hiit_x1 = df.index[hiit_end-1]
        
        window_color = '#44FF44'
        window_text = "HIIT Window (Auto)"
        
        fig.add_vrect(x0=hiit_x0, x1=hiit_x1, fillcolor=window_color, opacity=0.15, line_width=0, 
                     annotation_text=window_text, annotation_position="top left", row=1, col=1)
        
        # Show threshold-based window if different from detected window
        if frequency_info and 'regions_above_threshold' in frequency_info:
            regions = frequency_info['regions_above_threshold']
            if regions:
                threshold_start = regions[0][0]
                threshold_end = regions[-1][1]
                
                if threshold_start != hiit_start or threshold_end != hiit_end:
                    threshold_x0 = df.index[threshold_start]
                    threshold_x1 = df.index[threshold_end-1]
                    fig.add_vrect(x0=threshold_x0, x1=threshold_x1, fillcolor='#FFFF44', opacity=0.1, line_width=0, 
                                 annotation_text="Threshold Window", annotation_position="bottom left", row=1, col=1)
        
        # Show original window if different
        if frequency_info and 'original_start' in frequency_info and 'original_end' in frequency_info:
            orig_start = frequency_info['original_start']
            orig_end = frequency_info['original_end']
            if orig_start != hiit_start or orig_end != hiit_end:
                orig_x0 = df.index[orig_start]
                orig_x1 = df.index[orig_end-1]
                fig.add_vrect(x0=orig_x0, x1=orig_x1, fillcolor='#FF4444', opacity=0.1, line_width=0, 
                             annotation_text="Original Window", annotation_position="top right", row=1, col=1)
        
        # Add interval boundary lines and overlays
        for interval in intervals:
            # Add vertical lines at interval boundaries
            if 'interval_start' in interval and interval['interval_start'] < len(df):
                boundary_time = df.index[interval['interval_start']]
                fig.add_vline(x=boundary_time, line_dash="dash", line_color="yellow", 
                            line_width=2, row=1, col=1)
            
            # Add interval overlays
            high_x0 = df.index[interval['high_start']] if interval['high_start'] < len(df) else df.index[-1]
            high_x1 = df.index[interval['high_end']-1] if interval['high_end']-1 < len(df) else df.index[-1]
            rec_x0 = df.index[interval['recovery_start']] if interval['recovery_start'] < len(df) else df.index[-1]
            rec_x1 = df.index[interval['recovery_end']-1] if interval['recovery_end']-1 < len(df) else df.index[-1]
            fig.add_vrect(x0=high_x0, x1=high_x1, fillcolor='#FF8800', opacity=0.2, line_width=0, row=1, col=1)
            fig.add_vrect(x0=rec_x0, x1=rec_x1, fillcolor='#FF4444', opacity=0.1, line_width=0, row=1, col=1)
        
        # Add frequency correlation plot
        if frequency_info and 'frequency_correlation' in frequency_info:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=frequency_info['frequency_correlation'],
                mode='lines',
                name='Frequency Correlation',
                line=dict(color='#44FFFF', width=2)
            ), row=2, col=1)
            
            # Add threshold line
            if 'correlation_threshold' in frequency_info:
                fig.add_hline(
                    y=frequency_info['correlation_threshold'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Threshold",
                    row=2, col=1
                )
            
            # Highlight regions above threshold
            if 'regions_above_threshold' in frequency_info:
                for region_start, region_end in frequency_info['regions_above_threshold']:
                    if region_start < len(df) and region_end <= len(df):
                        fig.add_vrect(
                            x0=df.index[region_start],
                            x1=df.index[region_end-1],
                            fillcolor="rgba(255,255,0,0.2)",
                            line_width=0,
                            row=2, col=1
                        )
    
    # Frequency analysis plots
    if frequency_info:
        # Plot 1: Frequency spectrum
        if 'frequencies' in frequency_info and 'hr_fft_magnitude' in frequency_info:
            freqs = frequency_info['frequencies']
            magnitude = frequency_info['hr_fft_magnitude']
            # Focus on relevant frequency range (0.001 to 0.1 Hz = 10s to 1000s periods)
            freq_mask = (freqs >= 0.001) & (freqs <= 0.1)
            fig.add_trace(go.Scatter(
                x=freqs[freq_mask],
                y=magnitude[freq_mask],
                mode='lines',
                name='Frequency Spectrum',
                line=dict(color='#44FFFF', width=2)
            ), row=3, col=1)
            
            # Mark dominant frequency
            if 'dominant_freq' in frequency_info:
                fig.add_vline(x=frequency_info['dominant_freq'], line_dash="dash", line_color="red", 
                            annotation_text=f"Dominant: {frequency_info['dominant_period']:.1f}s", row=3, col=1)
        
        # Plot 2: Period power spectrum
        if 'filtered_periods' in frequency_info and 'filtered_power' in frequency_info:
            periods = frequency_info['filtered_periods']
            power = frequency_info['filtered_power']
            fig.add_trace(go.Scatter(
                x=periods,
                y=power,
                mode='lines+markers',
                name='Period Power',
                line=dict(color='#FF44FF', width=2),
                marker=dict(size=6)
            ), row=4, col=1)
            
            # Mark dominant period
            if 'dominant_period' in frequency_info:
                fig.add_vline(x=frequency_info['dominant_period'], line_dash="dash", line_color="red", 
                            annotation_text=f"Dominant Period", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=1800 if frequency_info else 1200,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    # Update axes with separate y-axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", color=get_field_color('enhanced_speed'), row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", color=get_field_color('heart_rate'), 
                    secondary_y=True, row=1, col=1)
    
    if frequency_info:
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Frequency Correlation", row=2, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
        fig.update_yaxes(title_text="Magnitude", row=3, col=1)
        fig.update_xaxes(title_text="Period (seconds)", row=4, col=1)
        fig.update_yaxes(title_text="Power", row=4, col=1)
        
        # Add interval overlay plot with exponential fits
        if intervals:
            overlay_row = 5 if frequency_info else 2
            
            # Create overlay plot with all intervals aligned
            for i, interval in enumerate(intervals):
                if 'interval_start' in interval and 'interval_end' in interval:
                    interval_start = interval['interval_start']
                    interval_end = interval['interval_end']
                    
                    if interval_start < len(df) and interval_end <= len(df):
                        # Extract interval data
                        interval_data = df.iloc[interval_start:interval_end]
                        
                        # Create time axis starting from 0
                        time_axis = np.arange(len(interval_data))
                        
                        # Calculate opacity (more transparent for earlier intervals)
                        opacity = 0.3 + (i / len(intervals)) * 0.7  # 0.3 to 1.0
                        
                        # Plot speed
                        if 'enhanced_speed' in interval_data.columns:
                            speed_data = interval_data['enhanced_speed'].fillna(method='ffill').fillna(method='bfill')
                            fig.add_trace(go.Scatter(
                                x=time_axis,
                                y=speed_data,
                                mode='lines',
                                name=f'Interval {interval["interval_num"]} Speed',
                                line=dict(color=get_field_color('enhanced_speed'), width=2),
                                opacity=opacity,
                                showlegend=False
                            ), row=overlay_row, col=1, secondary_y=False)
                        
                        # Plot heart rate with exponential fits
                        if 'heart_rate' in interval_data.columns:
                            hr_data = interval_data['heart_rate'].fillna(method='ffill').fillna(method='bfill')
                            fig.add_trace(go.Scatter(
                                x=time_axis,
                                y=hr_data,
                                mode='lines',
                                name=f'Interval {interval["interval_num"]} HR',
                                line=dict(color=get_field_color('heart_rate'), width=2),
                                opacity=opacity,
                                showlegend=False
                            ), row=overlay_row, col=1, secondary_y=True)
                            
                            # Add exponential fits if we have enough data
                            if len(hr_data) >= 10:
                                try:
                                    # Analyze performance for this interval
                                    metrics = analyze_interval_performance(df, interval)
                                    if metrics:
                                        # Split into rise and fall phases
                                        mid_point = len(hr_data) // 2
                                        rise_data = hr_data.iloc[:mid_point]
                                        fall_data = hr_data.iloc[mid_point:]
                                        
                                        rise_time = np.arange(len(rise_data))
                                        fall_time = np.arange(len(fall_data))
                                        
                                        # Fit exponential curves
                                        rise_params, rise_func = fit_hr_curve(rise_time, rise_data.values, "exp_rise")
                                        fall_params, fall_func = fit_hr_curve(fall_time, fall_data.values, "exp_fall")
                                        
                                        # Generate fitted curves
                                        rise_fit = rise_func(rise_time, *rise_params)
                                        fall_fit = fall_func(fall_time, *fall_params)
                                        
                                        # Plot rise fit
                                        fig.add_trace(go.Scatter(
                                            x=rise_time,
                                            y=rise_fit,
                                            mode='lines',
                                            name=f'Interval {interval["interval_num"]} Rise Fit',
                                            line=dict(color='#FF00FF', width=3, dash='dash'),
                                            opacity=opacity,
                                            showlegend=False
                                        ), row=overlay_row, col=1, secondary_y=True)
                                        
                                        # Plot fall fit
                                        fig.add_trace(go.Scatter(
                                            x=fall_time + mid_point,  # Offset by mid_point
                                            y=fall_fit,
                                            mode='lines',
                                            name=f'Interval {interval["interval_num"]} Fall Fit',
                                            line=dict(color='#00FFFF', width=3, dash='dash'),
                                            opacity=opacity,
                                            showlegend=False
                                        ), row=overlay_row, col=1, secondary_y=True)
                                
                                except Exception as e:
                                    # If fitting fails, just continue
                                    pass
            
            # Update overlay plot axes
            fig.update_xaxes(title_text="Time (seconds)", row=overlay_row, col=1)
            fig.update_yaxes(title_text="Speed (m/s)", color=get_field_color('enhanced_speed'), row=overlay_row, col=1)
            fig.update_yaxes(title_text="Heart Rate (bpm)", color=get_field_color('heart_rate'), 
                           secondary_y=True, row=overlay_row, col=1)
            
            # Add speed statistics plot
            stats_row = 6 if frequency_info else 3
            
            # Collect metrics for histograms
            all_metrics = []
            for interval in intervals:
                if 'high_start' in interval and 'high_end' in interval:
                    high_start = interval['high_start']
                    high_end = interval['high_end']
                    
                    if high_start < len(df) and high_end <= len(df):
                        # Extract high-intensity period data
                        high_period = df.iloc[high_start:high_end]
                        
                        if 'enhanced_speed' in high_period.columns and 'heart_rate' in high_period.columns:
                            try:
                                metrics = analyze_interval_performance(df, interval)
                                if metrics:
                                    all_metrics.append(metrics)
                            except:
                                pass
            
            # Calculate speed statistics for each interval
            for i, interval in enumerate(intervals):
                if 'high_start' in interval and 'high_end' in interval:
                    high_start = interval['high_start']
                    high_end = interval['high_end']
                    
                    if high_start < len(df) and high_end <= len(df):
                        # Extract high-intensity period data
                        high_period = df.iloc[high_start:high_end]
                        
                        if 'enhanced_speed' in high_period.columns:
                            speed_data = high_period['enhanced_speed'].fillna(method='ffill').fillna(method='bfill')
                            
                            if len(speed_data) > 5:
                                # Use clustering to separate high and low speed periods
                                speed_2d = speed_data.values.reshape(-1, 1)
                                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(speed_2d)
                                
                                # Identify high speed cluster
                                cluster_means = [speed_data[cluster_labels == i].mean() for i in range(2)]
                                high_speed_cluster = np.argmax(cluster_means)
                                
                                # Get high speed data
                                high_speed_data = speed_data[cluster_labels == high_speed_cluster]
                                
                                speed_mean = high_speed_data.mean()
                                speed_std = high_speed_data.std()
                                speed_variability = speed_std / speed_mean if speed_mean > 0 else 0
                                
                                # Calculate opacity
                                opacity = 0.3 + (i / len(intervals)) * 0.7
                                
                                # Plot speed statistics
                                fig.add_trace(go.Scatter(
                                    x=[interval['interval_num']],
                                    y=[speed_mean],
                                    mode='markers',
                                    name=f'Interval {interval["interval_num"]} Mean',
                                    marker=dict(
                                        color=get_field_color('enhanced_speed'),
                                        size=12,
                                        opacity=opacity
                                    ),
                                    error_y=dict(
                                        type='data',
                                        array=[speed_std],
                                        visible=True,
                                        color=get_field_color('enhanced_speed')
                                    ),
                                    showlegend=False
                                ), row=stats_row, col=1)
                                
                                # Add variability annotation
                                fig.add_annotation(
                                    x=interval['interval_num'],
                                    y=speed_mean + speed_std + 0.1,
                                    text=f"CV: {speed_variability:.3f}",
                                    showarrow=False,
                                    font=dict(size=8, color='white'),
                                    bgcolor='rgba(0,0,0,0.7)',
                                    bordercolor='white',
                                    borderwidth=1,
                                    row=stats_row, col=1
                                )
            
            # Update speed statistics axes
            fig.update_xaxes(title_text="Interval Number", row=stats_row, col=1)
            fig.update_yaxes(title_text="Speed (m/s)", row=stats_row, col=1)
            
            # Add histogram plots if we have metrics
            if all_metrics:
                # Tau Rise Distribution
                tau_rise_row = 7 if frequency_info else 4
                tau_rise_values = [m['tau_rise'] for m in all_metrics if m['tau_rise'] > 0]
                if tau_rise_values:
                    fig.add_trace(go.Histogram(
                        x=tau_rise_values,
                        name='Tau Rise',
                        nbinsx=10,
                        marker_color='#FF00FF',
                        opacity=0.7
                    ), row=tau_rise_row, col=1)
                
                # Tau Fall Distribution
                tau_fall_row = 7 if frequency_info else 4
                tau_fall_values = [m['tau_fall'] for m in all_metrics if m['tau_fall'] > 0]
                if tau_fall_values:
                    fig.add_trace(go.Histogram(
                        x=tau_fall_values,
                        name='Tau Fall',
                        nbinsx=10,
                        marker_color='#00FFFF',
                        opacity=0.7
                    ), row=tau_fall_row, col=2)
                
                # HR Baseline Distribution
                hr_baseline_row = 8 if frequency_info else 5
                hr_baseline_values = [m['hr_baseline'] for m in all_metrics if m['hr_baseline'] > 0]
                if hr_baseline_values:
                    fig.add_trace(go.Histogram(
                        x=hr_baseline_values,
                        name='HR Baseline',
                        nbinsx=10,
                        marker_color='#FF4444',
                        opacity=0.7
                    ), row=hr_baseline_row, col=1)
                
                # HR Peak Distribution
                hr_peak_row = 8 if frequency_info else 5
                hr_peak_values = [m['hr_peak'] for m in all_metrics if m['hr_peak'] > 0]
                if hr_peak_values:
                    fig.add_trace(go.Histogram(
                        x=hr_peak_values,
                        name='HR Peak',
                        nbinsx=10,
                        marker_color='#FF8888',
                        opacity=0.7
                    ), row=hr_peak_row, col=2)
                
                # Speed Mean Distribution
                speed_mean_row = 9 if frequency_info else 6
                speed_mean_values = [m['speed_mean'] for m in all_metrics if m['speed_mean'] > 0]
                if speed_mean_values:
                    fig.add_trace(go.Histogram(
                        x=speed_mean_values,
                        name='Speed Mean',
                        nbinsx=10,
                        marker_color='#FFFF44',
                        opacity=0.7
                    ), row=speed_mean_row, col=1)
                
                # Speed Std Distribution
                speed_std_row = 9 if frequency_info else 6
                speed_std_values = [m['speed_std'] for m in all_metrics if m['speed_std'] > 0]
                if speed_std_values:
                    fig.add_trace(go.Histogram(
                        x=speed_std_values,
                        name='Speed Std',
                        nbinsx=10,
                        marker_color='#FFFF88',
                        opacity=0.7
                    ), row=speed_std_row, col=2)
                
                # Update histogram axes
                fig.update_xaxes(title_text="Tau Rise (seconds)", row=tau_rise_row, col=1)
                fig.update_yaxes(title_text="Count", row=tau_rise_row, col=1)
                
                fig.update_xaxes(title_text="Tau Fall (seconds)", row=tau_fall_row, col=2)
                fig.update_yaxes(title_text="Count", row=tau_fall_row, col=2)
                
                fig.update_xaxes(title_text="HR Baseline (bpm)", row=hr_baseline_row, col=1)
                fig.update_yaxes(title_text="Count", row=hr_baseline_row, col=1)
                
                fig.update_xaxes(title_text="HR Peak (bpm)", row=hr_peak_row, col=2)
                fig.update_yaxes(title_text="Count", row=hr_peak_row, col=2)
                
                fig.update_xaxes(title_text="Speed Mean (m/s)", row=speed_mean_row, col=1)
                fig.update_yaxes(title_text="Count", row=speed_mean_row, col=1)
                
                fig.update_xaxes(title_text="Speed Std (m/s)", row=speed_std_row, col=2)
                fig.update_yaxes(title_text="Count", row=speed_std_row, col=2)
    
    return fig

# Main app
def main():
    # ... your main function code ...

def show_raw_data_page(filename):
    st.header("📊 Raw Data Analysis")
    # --- File selection as URL param ---
    import glob
    fit_files = glob.glob('data/*.fit')
    if not fit_files:
        st.error("No .fit files found in the data/ directory.")
        return
    url_file = st.query_params.get("file", None)
    if url_file and url_file in fit_files:
        selected_file = url_file
        st.session_state["file_selector_raw"] = selected_file
    else:
        selected_file = st.session_state.get("file_selector_raw", fit_files[0])
    # File selectbox (syncs with URL)
    new_file = st.selectbox("Select FIT file:", fit_files, index=fit_files.index(selected_file), key="file_selector_raw")
    if new_file != selected_file:
        st.query_params["file"] = new_file
        st.rerun()
    filename = new_file
    # --- End file selection as URL param ---
    
    # Load data
    with st.spinner("Loading FIT file..."):
        df = load_fit_data(filename)
    
    if df is None or df.empty:
        st.error("Failed to load data from the selected file.")
        return
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        if len(df) > 0:
            duration = df.index[-1] - df.index[0]
            st.metric("Duration", str(duration))
    
    with col3:
        if 'distance' in df.columns:
            distance_series = df['distance'].dropna()
            if len(distance_series) > 1:
                total_distance = distance_series.iloc[-1] - distance_series.iloc[0]
                st.metric("Total Distance", f"{total_distance:.2f} m")
    
    # Data summary
    st.subheader("Data Summary")
    
    summary_data = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) > 0:
            summary_data.append({
                'Field': col,
                'Records': len(series),
                'Missing': df[col].isna().sum(),
                'Mean': series.mean() if series.dtype in ['int64', 'float64'] else 'N/A',
                'Std': series.std() if series.dtype in ['int64', 'float64'] else 'N/A',
                'Min': series.min() if series.dtype in ['int64', 'float64'] else 'N/A',
                'Max': series.max() if series.dtype in ['int64', 'float64'] else 'N/A'
            })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Time series plots grouped by category
    st.subheader("Time Series Data")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Group fields by category
        field_groups = {}
        for col in numeric_cols:
            group = get_field_group(col)
            if group not in field_groups:
                field_groups[group] = []
            field_groups[group].append(col)
        
        # Create subplots for each group (max 3 per row)
        for group_name, group_fields in field_groups.items():
            if not group_fields:
                continue
                
            st.markdown(f"**{group_name}**")
            
            # Calculate layout for this group (max 3 columns)
            n_cols = min(3, len(group_fields))
            n_rows = (len(group_fields) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[col.replace('_', ' ').title() for col in group_fields],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            for i, col in enumerate(group_fields):
                row = (i // n_cols) + 1
                col_idx = (i % n_cols) + 1
                
                series = df[col].dropna()
                if len(series) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode='lines',
                            name=col.replace('_', ' ').title(),
                            line=dict(color=get_field_color(col), width=2)
                        ),
                        row=row, col=col_idx
                    )
            
            fig.update_layout(
                template="plotly_dark",
                height=250 * n_rows,
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")  # Separator between groups

def create_detection_figure(df, hiit_start, hiit_end, intervals, frequency_info=None, filename=None):
    """Create the main detection and frequency analysis figure with manual window selection."""
    if frequency_info:
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Speed & Heart Rate with HIIT Detection', 'Frequency Correlation', 'Frequency Analysis', 'Period Power Spectrum'],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Speed & Heart Rate with HIIT Detection'],
            specs=[[{"secondary_y": True}]]
        )
    
    # Main data plot - separate y-axes for speed and heart rate
    if 'enhanced_speed' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['enhanced_speed'],
            mode='lines',
            name='Speed (Raw)',
            line=dict(color=get_field_color('enhanced_speed'), width=1),
            opacity=0.7
        ), row=1, col=1, secondary_y=False)
        
        # Add smoothed speed if available
        window_size = 5
        speed_smoothed = df['enhanced_speed'].rolling(window=window_size, center=True).mean().fillna(method='ffill').fillna(method='bfill')
        fig.add_trace(go.Scatter(
            x=df.index,
            y=speed_smoothed,
            mode='lines',
            name='Speed (Smoothed)',
            line=dict(color=get_field_color('enhanced_speed'), width=2)
        ), row=1, col=1, secondary_y=False)
    
    if 'heart_rate' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['heart_rate'],
            mode='lines',
            name='Heart Rate (Raw)',
            line=dict(color=get_field_color('heart_rate'), width=1),
            opacity=0.7
        ), row=1, col=1, secondary_y=True)
        
        # Add filtered heart rate if available
        if frequency_info and 'hr_filtered' in frequency_info:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=frequency_info['hr_filtered'],
                mode='lines',
                name='Heart Rate (Filtered)',
                line=dict(color=get_field_color('heart_rate'), width=2)
            ), row=1, col=1, secondary_y=True)
    
    # HIIT window and transitions
    if hiit_start is not None and hiit_end is not None:
        hiit_x0 = df.index[hiit_start]
        hiit_x1 = df.index[hiit_end-1]
        
        # Check if using manual window
        manual_start, manual_end = load_manual_window_settings(filename) if filename else (None, None)
        using_manual = manual_start is not None and manual_end is not None
        
        window_color = '#FF4444' if using_manual else '#44FF44'
        window_text = "HIIT Window (Manual)" if using_manual else "HIIT Window (Auto)"
        
        fig.add_vrect(x0=hiit_x0, x1=hiit_x1, fillcolor=window_color, opacity=0.15, line_width=0, 
                     annotation_text=window_text, annotation_position="top left", row=1, col=1)
        
        # Show threshold-based window if different from detected window
        if frequency_info and 'regions_above_threshold' in frequency_info:
            regions = frequency_info['regions_above_threshold']
            if regions:
                threshold_start = regions[0][0]
                threshold_end = regions[-1][1]
                
                if threshold_start != hiit_start or threshold_end != hiit_end:
                    threshold_x0 = df.index[threshold_start]
                    threshold_x1 = df.index[threshold_end-1]
                    fig.add_vrect(x0=threshold_x0, x1=threshold_x1, fillcolor='#FFFF44', opacity=0.1, line_width=0, 
                                 annotation_text="Threshold Window", annotation_position="bottom left", row=1, col=1)
        
        # Show original window if different
        if frequency_info and 'original_start' in frequency_info and 'original_end' in frequency_info:
            orig_start = frequency_info['original_start']
            orig_end = frequency_info['original_end']
            if orig_start != hiit_start or orig_end != hiit_end:
                orig_x0 = df.index[orig_start]
                orig_x1 = df.index[orig_end-1]
                fig.add_vrect(x0=orig_x0, x1=orig_x1, fillcolor='#FF4444', opacity=0.1, line_width=0, 
                             annotation_text="Original Window", annotation_position="top right", row=1, col=1)
        
        # Add interval boundary lines and overlays
        for interval in intervals:
            # Add vertical lines at interval boundaries
            if 'interval_start' in interval and interval['interval_start'] < len(df):
                boundary_time = df.index[interval['interval_start']]
                fig.add_vline(x=boundary_time, line_dash="dash", line_color="yellow", 
                            line_width=2, row=1, col=1)
            
            # Add interval overlays
            high_x0 = df.index[interval['high_start']] if interval['high_start'] < len(df) else df.index[-1]
            high_x1 = df.index[interval['high_end']-1] if interval['high_end']-1 < len(df) else df.index[-1]
            rec_x0 = df.index[interval['recovery_start']] if interval['recovery_start'] < len(df) else df.index[-1]
            rec_x1 = df.index[interval['recovery_end']-1] if interval['recovery_end']-1 < len(df) else df.index[-1]
            fig.add_vrect(x0=high_x0, x1=high_x1, fillcolor='#FF8800', opacity=0.2, line_width=0, row=1, col=1)
            fig.add_vrect(x0=rec_x0, x1=rec_x1, fillcolor='#FF4444', opacity=0.1, line_width=0, row=1, col=1)
    
    # Add frequency correlation plot
    if frequency_info and 'frequency_correlation' in frequency_info:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=frequency_info['frequency_correlation'],
            mode='lines',
            name='Frequency Correlation',
            line=dict(color='#44FFFF', width=2)
        ), row=2, col=1)
        
        # Add threshold line
        if 'correlation_threshold' in frequency_info:
            fig.add_hline(
                y=frequency_info['correlation_threshold'],
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold",
                row=2, col=1
            )
        
        # Highlight regions above threshold
        if 'regions_above_threshold' in frequency_info:
            for region_start, region_end in frequency_info['regions_above_threshold']:
                if region_start < len(df) and region_end <= len(df):
                    fig.add_vrect(
                        x0=df.index[region_start],
                        x1=df.index[region_end-1],
                        fillcolor="rgba(255,255,0,0.2)",
                        line_width=0,
                        row=2, col=1
                    )
    
    # Frequency analysis plots
    if frequency_info:
        # Plot 1: Frequency spectrum
        if 'frequencies' in frequency_info and 'hr_fft_magnitude' in frequency_info:
            freqs = frequency_info['frequencies']
            magnitude = frequency_info['hr_fft_magnitude']
            freq_mask = (freqs >= 0.001) & (freqs <= 0.1)
            fig.add_trace(go.Scatter(
                x=freqs[freq_mask],
                y=magnitude[freq_mask],
                mode='lines',
                name='Frequency Spectrum',
                line=dict(color='#44FFFF', width=2)
            ), row=3, col=1)
        
        # Plot 2: Period power spectrum
        if 'filtered_periods' in frequency_info and 'filtered_power' in frequency_info:
            periods = frequency_info['filtered_periods']
            power = frequency_info['filtered_power']
            fig.add_trace(go.Scatter(
                x=periods,
                y=power,
                mode='lines+markers',
                name='Period Power',
                line=dict(color='#FF44FF', width=2),
                marker=dict(size=6)
            ), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=800 if frequency_info else 400,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", color=get_field_color('enhanced_speed'), row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", color=get_field_color('heart_rate'), 
                    secondary_y=True, row=1, col=1)
    
    if frequency_info:
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Frequency Correlation", row=2, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
        fig.update_yaxes(title_text="Magnitude", row=3, col=1)
        fig.update_xaxes(title_text="Period (seconds)", row=4, col=1)
        fig.update_yaxes(title_text="Power", row=4, col=1)
    
    return fig

def create_interval_overlay_figure(df, intervals):
    """Create the interval overlay figure with exponential fits."""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Interval Overlay with Exponential Fits'],
        specs=[[{"secondary_y": True}]]
    )
    
    # Create overlay plot with all intervals aligned
    for i, interval in enumerate(intervals):
        if 'interval_start' in interval and 'interval_end' in interval:
            interval_start = interval['interval_start']
            interval_end = interval['interval_end']
            
            if interval_start < len(df) and interval_end <= len(df):
                # Extract interval data
                interval_data = df.iloc[interval_start:interval_end]
                
                # Create time axis starting from 0
                time_axis = np.arange(len(interval_data))
                
                # Calculate opacity (more transparent for earlier intervals)
                opacity = 0.3 + (i / len(intervals)) * 0.7  # 0.3 to 1.0
                
                # Plot speed
                if 'enhanced_speed' in interval_data.columns:
                    speed_data = interval_data['enhanced_speed'].fillna(method='ffill').fillna(method='bfill')
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=speed_data,
                        mode='lines',
                        name=f'Interval {interval["interval_num"]} Speed',
                        line=dict(color=get_field_color('enhanced_speed'), width=2),
                        opacity=opacity,
                        showlegend=False
                    ), row=1, col=1, secondary_y=False)
                
                # Plot heart rate with exponential fits
                if 'heart_rate' in interval_data.columns:
                    hr_data = interval_data['heart_rate'].fillna(method='ffill').fillna(method='bfill')
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=hr_data,
                        mode='lines',
                        name=f'Interval {interval["interval_num"]} HR',
                        line=dict(color=get_field_color('heart_rate'), width=2),
                        opacity=opacity,
                        showlegend=False
                    ), row=1, col=1, secondary_y=True)
                    
                    # Add exponential fits if we have enough data
                    if len(hr_data) >= 10:
                        try:
                            # Analyze performance for this interval
                            metrics = analyze_interval_performance(df, interval)
                            if metrics:
                                # Split into rise and fall phases
                                mid_point = len(hr_data) // 2
                                rise_data = hr_data.iloc[:mid_point]
                                fall_data = hr_data.iloc[mid_point:]
                                
                                rise_time = np.arange(len(rise_data))
                                fall_time = np.arange(len(fall_data))
                                
                                # Fit exponential curves
                                rise_params, rise_func = fit_hr_curve(rise_time, rise_data.values, "exp_rise")
                                fall_params, fall_func = fit_hr_curve(fall_time, fall_data.values, "exp_fall")
                                
                                # Generate fitted curves
                                rise_fit = rise_func(rise_time, *rise_params)
                                fall_fit = fall_func(fall_time, *fall_params)
                                
                                # Plot rise fit
                                fig.add_trace(go.Scatter(
                                    x=rise_time,
                                    y=rise_fit,
                                    mode='lines',
                                    name=f'Interval {interval["interval_num"]} Rise Fit',
                                    line=dict(color='#FF00FF', width=3, dash='dash'),
                                    opacity=opacity,
                                    showlegend=False
                                ), row=1, col=1, secondary_y=True)
                                
                                # Plot fall fit
                                fig.add_trace(go.Scatter(
                                    x=fall_time + mid_point,  # Offset by mid_point
                                    y=fall_fit,
                                    mode='lines',
                                    name=f'Interval {interval["interval_num"]} Fall Fit',
                                    line=dict(color='#00FFFF', width=3, dash='dash'),
                                    opacity=opacity,
                                    showlegend=False
                                ), row=1, col=1, secondary_y=True)
                                
                                # Add interval boundary indicators
                                if 'high_start' in interval and 'high_end' in interval:
                                    # High intensity start
                                    high_start_rel = interval['high_start'] - interval_start
                                    if 0 <= high_start_rel < len(time_axis):
                                        fig.add_vline(
                                            x=high_start_rel, 
                                            line_dash="dot", 
                                            line_color="yellow",
                                            line_width=2,
                                            opacity=opacity,
                                            annotation_text=f"High {interval['interval_num']}",
                                            annotation_position="top right"
                                        )
                                    
                                    # High intensity end
                                    high_end_rel = interval['high_end'] - interval_start
                                    if 0 <= high_end_rel < len(time_axis):
                                        fig.add_vline(
                                            x=high_end_rel, 
                                            line_dash="dot", 
                                            line_color="orange",
                                            line_width=2,
                                            opacity=opacity,
                                            annotation_text=f"Recovery {interval['interval_num']}",
                                            annotation_position="top left"
                                        )
                        except Exception as e:
                            # If fitting fails, just continue
                            pass
    
    # Update overlay plot axes
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", color=get_field_color('enhanced_speed'), row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", color=get_field_color('heart_rate'), 
                   secondary_y=True, row=1, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig

def create_speed_stats_figure(df, intervals):
    """Create the speed statistics figure."""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Speed Statistics by Interval'],
        specs=[[{"secondary_y": False}]]
    )
    
    # Calculate speed statistics for each interval
    for i, interval in enumerate(intervals):
        if 'high_start' in interval and 'high_end' in interval:
            high_start = interval['high_start']
            high_end = interval['high_end']
            
            if high_start < len(df) and high_end <= len(df):
                # Extract high-intensity period data
                high_period = df.iloc[high_start:high_end]
                
                if 'enhanced_speed' in high_period.columns:
                    speed_data = high_period['enhanced_speed'].fillna(method='ffill').fillna(method='bfill')
                    
                    if len(speed_data) > 5:
                        # Use clustering to separate high and low speed periods
                        speed_2d = speed_data.values.reshape(-1, 1)
                        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(speed_2d)
                        
                        # Identify high speed cluster
                        cluster_means = [speed_data[cluster_labels == i].mean() for i in range(2)]
                        high_speed_cluster = np.argmax(cluster_means)
                        
                        # Get high speed data
                        high_speed_data = speed_data[cluster_labels == high_speed_cluster]
                        
                        speed_mean = high_speed_data.mean()
                        speed_std = high_speed_data.std()
                        speed_variability = speed_std / speed_mean if speed_mean > 0 else 0
                        
                        # Calculate opacity
                        opacity = 0.3 + (i / len(intervals)) * 0.7
                        
                        # Plot speed statistics
                        fig.add_trace(go.Scatter(
                            x=[interval['interval_num']],
                            y=[speed_mean],
                            mode='markers',
                            name=f'Interval {interval["interval_num"]} Mean',
                            marker=dict(
                                color=get_field_color('enhanced_speed'),
                                size=12,
                                opacity=opacity
                            ),
                            error_y=dict(
                                type='data',
                                array=[speed_std],
                                visible=True,
                                color=get_field_color('enhanced_speed')
                            ),
                            showlegend=False
                        ), row=1, col=1)
                        
                        # Add variability annotation
                        fig.add_annotation(
                            x=interval['interval_num'],
                            y=speed_mean + speed_std + 0.1,
                            text=f"CV: {speed_variability:.3f}",
                            showarrow=False,
                            font=dict(size=8, color='white'),
                            bgcolor='rgba(0,0,0,0.7)',
                            bordercolor='white',
                            borderwidth=1,
                            row=1, col=1
                        )
    
    # Update speed statistics axes
    fig.update_xaxes(title_text="Interval Number", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig

def create_metrics_distributions_figure(df, intervals):
    """Create the performance metrics distributions figure."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Tau Rise Distribution', 'Tau Fall Distribution', 'HR Baseline Distribution', 'HR Peak Distribution'],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Collect metrics for histograms
    all_metrics = []
    for interval in intervals:
        if 'high_start' in interval and 'high_end' in interval:
            high_start = interval['high_start']
            high_end = interval['high_end']
            
            if high_start < len(df) and high_end <= len(df):
                # Extract high-intensity period data
                high_period = df.iloc[high_start:high_end]
                
                if 'enhanced_speed' in high_period.columns and 'heart_rate' in high_period.columns:
                    try:
                        metrics = analyze_interval_performance(df, interval)
                        if metrics:
                            all_metrics.append(metrics)
                    except:
                        pass
    
    # Add histogram plots if we have metrics
    if all_metrics:
        # Tau Rise Distribution
        tau_rise_values = [m['tau_rise'] for m in all_metrics if m['tau_rise'] > 0]
        if tau_rise_values:
            fig.add_trace(go.Histogram(
                x=tau_rise_values,
                name='Tau Rise',
                nbinsx=10,
                marker_color='#FF00FF',
                opacity=0.7
            ), row=1, col=1)
        
        # Tau Fall Distribution
        tau_fall_values = [m['tau_fall'] for m in all_metrics if m['tau_fall'] > 0]
        if tau_fall_values:
            fig.add_trace(go.Histogram(
                x=tau_fall_values,
                name='Tau Fall',
                nbinsx=10,
                marker_color='#00FFFF',
                opacity=0.7
            ), row=1, col=2)
        
        # HR Baseline Distribution
        hr_baseline_values = [m['hr_baseline'] for m in all_metrics if m['hr_baseline'] > 0]
        if hr_baseline_values:
            fig.add_trace(go.Histogram(
                x=hr_baseline_values,
                name='HR Baseline',
                nbinsx=10,
                marker_color='#FF4444',
                opacity=0.7
            ), row=2, col=1)
        
        # HR Peak Distribution
        hr_peak_values = [m['hr_peak'] for m in all_metrics if m['hr_peak'] > 0]
        if hr_peak_values:
            fig.add_trace(go.Histogram(
                x=hr_peak_values,
                name='HR Peak',
                nbinsx=10,
                marker_color='#FF8888',
                opacity=0.7
            ), row=2, col=2)
        
        # Calculate shared x-axis range for tau histograms
        all_tau_values = tau_rise_values + tau_fall_values
        if all_tau_values:
            tau_min = min(all_tau_values)
            tau_max = max(all_tau_values)
            tau_range = tau_max - tau_min
            
            # Use percentiles to get better range if there are outliers
            if len(all_tau_values) > 5:
                tau_95 = np.percentile(all_tau_values, 95)
                tau_max = min(tau_max, tau_95 * 1.2)  # Cap at 95th percentile + 20%
            
            # Ensure reasonable range
            if tau_range < 0.01:  # Very small range
                tau_min = 0
                tau_max = max(tau_max, 0.5)
            else:
                # Add some padding
                tau_min = max(0, tau_min - tau_range * 0.1)
                tau_max = tau_max + tau_range * 0.1
        
        # Update histogram axes with consistent ranges
        fig.update_xaxes(title_text="Tau Rise (seconds)", row=1, col=1, range=[tau_min, tau_max] if all_tau_values else None)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Tau Fall (seconds)", row=1, col=2, range=[tau_min, tau_max] if all_tau_values else None)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="HR Baseline (bpm)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_xaxes(title_text="HR Peak (bpm)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=False
    )
    
    return fig

def create_top_level_metrics_figure(df, intervals):
    """Create the top-level metrics (responsiveness and speed variability) figure."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Responsiveness Distribution', 'Speed Variability Distribution'],
        horizontal_spacing=0.1
    )
    
    # Collect metrics for histograms
    all_metrics = []
    for interval in intervals:
        if 'high_start' in interval and 'high_end' in interval:
            high_start = interval['high_start']
            high_end = interval['high_end']
            
            if high_start < len(df) and high_end <= len(df):
                # Extract high-intensity period data
                high_period = df.iloc[high_start:high_end]
                
                if 'enhanced_speed' in high_period.columns and 'heart_rate' in high_period.columns:
                    try:
                        metrics = analyze_interval_performance(df, interval)
                        if metrics:
                            all_metrics.append(metrics)
                    except:
                        pass
    
    # Add histogram plots if we have metrics
    if all_metrics:
        # Responsiveness Distribution
        responsiveness_values = [m['responsiveness'] for m in all_metrics if m['responsiveness'] > 0]
        if responsiveness_values:
            fig.add_trace(go.Histogram(
                x=responsiveness_values,
                name='Responsiveness',
                nbinsx=10,
                marker_color='#44FF44',
                opacity=0.7
            ), row=1, col=1)
        
        # Speed Variability Distribution
        speed_variability_values = [m['speed_variability'] for m in all_metrics if m['speed_variability'] > 0]
        if speed_variability_values:
            fig.add_trace(go.Histogram(
                x=speed_variability_values,
                name='Speed Variability',
                nbinsx=10,
                marker_color='#FFFF44',
                opacity=0.7
            ), row=1, col=2)
        
        # Update histogram axes
        fig.update_xaxes(title_text="Responsiveness (bpm/s)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Speed Variability (CV)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig

def show_interval_analysis_page(filename):
    st.header("🔄 Interval Analysis")
    
    # --- File selection as URL param ---
    import glob
    fit_files = glob.glob('data/*.fit')
    if not fit_files:
        st.error("No .fit files found in the data/ directory.")
        return
    
    url_file = st.query_params.get("file", None)
    if url_file and url_file in fit_files:
        selected_file = url_file
        st.session_state["file_selector_interval"] = selected_file
    else:
        selected_file = st.session_state.get("file_selector_interval", fit_files[0])
    
    # File selectbox (syncs with URL)
    new_file = st.selectbox("Select FIT file:", fit_files, index=fit_files.index(selected_file), key="file_selector_interval")
    if new_file != selected_file:
        st.query_params["file"] = new_file
        st.rerun()
    filename = new_file
    # --- End file selection as URL param ---

    df = load_fit_data(filename)
    if df is None or df.empty:
        st.error("Failed to load data from the selected file.")
        return
    
    # Manual correlation threshold calibration
    st.subheader("Manual Threshold Calibration")
    
    # Load saved threshold for this file
    saved_threshold = load_correlation_threshold(filename)
    
    # Initialize session state for this file
    file_key = f"threshold_input_{filename}"
    if file_key not in st.session_state:
        st.session_state[file_key] = saved_threshold if saved_threshold is not None else 0.5
    
    # Initialize window selection for this file
    window_key = f"use_threshold_window_{filename}"
    if window_key not in st.session_state:
        st.session_state[window_key] = False
    
    if saved_threshold is not None:
        st.success(f"✅ Using saved correlation threshold: {saved_threshold:.3f}")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Saved Threshold", type="secondary", key="clear_threshold_btn"):
                clear_correlation_threshold(filename)
                st.session_state[file_key] = 0.5
                st.session_state[window_key] = False
                st.rerun()
    else:
        st.info("🔍 Using automatic threshold detection. Set a manual threshold below if needed.")
    
    # Manual threshold input
    col1, col2, col3 = st.columns(3)
    with col1:
        manual_threshold = st.number_input(
            "Correlation Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state[file_key], 
            step=0.01,
            format="%.3f",
            help="Higher values = more selective detection, Lower values = more inclusive detection",
        )
    
    with col2:
        if st.button("Save Threshold", type="primary", key="save_threshold_btn"):
            save_correlation_threshold(filename, manual_threshold)
            st.success(f"Saved threshold: {manual_threshold:.3f}")
            st.rerun()
    
    with col3:
        st.markdown("**Current Value:**")
        st.markdown(f"{manual_threshold:.3f}")
    
    st.info("💡 Adjust the correlation threshold to fine-tune HIIT detection sensitivity. Higher values detect fewer, more confident intervals.")
    
    # Run detection with manual threshold if available
    manual_threshold = load_correlation_threshold(filename)
    hiit_start, hiit_end, frequency_info = detect_hiit_period_frequency(df, manual_threshold=manual_threshold)
    
    # Show threshold-based window if we have frequency info and update window if needed
    if frequency_info and 'regions_above_threshold' in frequency_info and 'correlation_threshold' in frequency_info:
        regions = frequency_info['regions_above_threshold']
        threshold = frequency_info['correlation_threshold']
        
        if regions:
            # Find the first and last regions above threshold
            first_region_start = regions[0][0]
            last_region_end = regions[-1][1]
            
            # Check if user wants to use threshold-based window
            use_threshold_window = st.session_state[window_key]
            
            # Show the threshold-based window
            st.subheader("Threshold-Based Window")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Threshold", f"{threshold:.3f}")
            with col2:
                st.metric("First Region Start", f"{first_region_start}")
            with col3:
                st.metric("Last Region End", f"{last_region_end}")
            
            # Show current selection status
            if use_threshold_window:
                st.success("🟡 Currently using threshold-based window")
            else:
                st.info("ℹ️ Currently using detected window")
            
            # Show if the detected window matches the threshold window
            if hiit_start == first_region_start and hiit_end == last_region_end:
                st.success("✅ Detected window matches threshold-based window")
            else:
                st.info(f"ℹ️ Detected window ({hiit_start}-{hiit_end}) vs threshold window ({first_region_start}-{last_region_end})")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use Threshold-Based Window", key="use_threshold_window_btn"):
                        st.session_state[window_key] = True
                        st.rerun()
                with col2:
                    if st.button("Use Detected Window", key="use_detected_window_btn"):
                        st.session_state[window_key] = False
                        st.rerun()
            
            # Update window if user chose threshold-based window
            if use_threshold_window:
                hiit_start = first_region_start
                hiit_end = last_region_end
                st.success("🟡 Using threshold-based window")
    
    intervals = []
    if hiit_start is not None and hiit_end is not None:
        df_hiit = df.iloc[hiit_start:hiit_end]
        intervals = segment_intervals_speed_edges(df_hiit)
        for interval in intervals:
            for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                if k in interval:
                    interval[k] += hiit_start
    
    # Display frequency analysis results
    if frequency_info:
        st.subheader("Frequency Analysis Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'dominant_period' in frequency_info:
                st.metric("Dominant Period", f"{frequency_info['dominant_period']:.1f} seconds")
        
        with col2:
            if 'dominant_freq' in frequency_info:
                st.metric("Dominant Frequency", f"{frequency_info['dominant_freq']:.3f} Hz")
        
        with col3:
            if 'correlation_threshold' in frequency_info:
                st.metric("Optimal Threshold", f"{frequency_info['correlation_threshold']:.3f}")
        
        with col4:
            if 'optimization_score' in frequency_info:
                st.metric("Contiguity Score", f"{frequency_info['optimization_score']:.2f}")
    
    # Create multiple figures for better organization
    if intervals:
        # Figure 1: Main detection and frequency analysis
        st.subheader("📊 HIIT Detection & Frequency Analysis")
        fig_main = create_detection_figure(df, hiit_start, hiit_end, intervals, frequency_info, filename)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Figure 2: Interval overlay with exponential fits
        st.subheader("🔄 Interval Overlay with Exponential Fits")
        fig2 = create_interval_overlay_figure(df, intervals)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Figure 3: Speed statistics
        st.subheader("🏃 Speed Statistics by Interval")
        fig3 = create_speed_stats_figure(df, intervals)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Figure 4: Performance metrics distributions
        st.subheader("📈 Performance Metrics Distributions")
        fig4 = create_metrics_distributions_figure(df, intervals)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Figure 5: Top-level metrics (responsiveness and speed variability)
        st.subheader("🎯 Top-Level Performance Metrics")
        fig5 = create_top_level_metrics_figure(df, intervals)
        st.plotly_chart(fig5, use_container_width=True)
    
    # Interval table
    if intervals:
        st.subheader("Detected Intervals")
        interval_data = []
        for interval in intervals:
            interval_data.append({
                'Interval': interval['interval_num'],
                'High Start': str(df.index[interval['high_start']]),
                'High End': str(df.index[interval['high_end']-1]),
                'Recovery Start': str(df.index[interval['recovery_start']]),
                'Recovery End': str(df.index[interval['recovery_end']-1])
            })
        interval_df = pd.DataFrame(interval_data)
        st.dataframe(interval_df, use_container_width=True)
    else:
        st.warning("No intervals detected. Try adjusting the parameters or check if the data contains HIIT patterns.")

def show_metrics_page():
    st.header("📈 Performance Metrics")
    
    # Get all FIT files in the data directory
    import glob
    import os
    data_dir = "./data/"
    fit_files = glob.glob(os.path.join(data_dir, "*.fit"))
    if not fit_files:
        st.warning("No FIT files found in the data directory.")
        return
    
    # Sort files by modification time (oldest first)
    fit_files.sort(key=lambda x: os.path.getmtime(x))
    
    st.info(f"Found {len(fit_files)} FIT files. Analyzing all sessions...")
    
    all_performance_metrics = []
    file_colors = []
    
    # Process each file
    for file_idx, fit_file in enumerate(fit_files):
        try:
            # Load FIT data
            df = load_fit_data(fit_file)
            if df is None or df.empty:
                continue
                
            # Use frequency-based detection for metrics page
            hiit_start, hiit_end, frequency_info = detect_hiit_period_frequency(df)
            
            if hiit_start is None or hiit_end is None:
                continue
            
            df_hiit = df.iloc[hiit_start:hiit_end]
            intervals = segment_intervals_speed_edges(df_hiit)
            
            if not intervals:
                continue
            
            # Analyze interval performance for curve fitting
            for interval in intervals:
                metrics = analyze_interval_performance(df, interval)
                if metrics:
                    metrics['file_name'] = os.path.basename(fit_file)
                    metrics['file_idx'] = file_idx
                    metrics['high_duration'] = interval['high_duration']
                    metrics['recovery_duration'] = interval['recovery_duration']
                    all_performance_metrics.append(metrics)
                
        except Exception as e:
            st.warning(f"Error processing {fit_file}: {str(e)}")
            continue
    
    if not all_performance_metrics:
        st.warning("No intervals detected in any of the files.")
        return
    
    # Display overall metrics
    st.subheader("Overall Session Metrics")
    
    # Get start and end dates from file modification times
    file_times = [os.path.getmtime(fit_file) for fit_file in fit_files]
    start_date = datetime.fromtimestamp(min(file_times)).strftime('%Y-%m-%d')
    end_date = datetime.fromtimestamp(max(file_times)).strftime('%Y-%m-%d')
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Files", len(fit_files))
    
    with col2:
        st.metric("Total Intervals", len(all_performance_metrics))
    
    with col3:
        avg_high_duration = np.mean([m['high_duration'] for m in all_performance_metrics])
        st.metric("Avg High Duration", f"{avg_high_duration:.1f}s")
    
    with col4:
        avg_recovery_duration = np.mean([m['recovery_duration'] for m in all_performance_metrics])
        st.metric("Avg Recovery Duration", f"{avg_recovery_duration:.1f}s")
    
    with col5:
        st.metric("Start Date", start_date)
    
    with col6:
        st.metric("End Date", end_date)
    
    # Create scatter plot with multiple files
    st.subheader("Interval Performance Analysis")
    
    fig = go.Figure()
    
    # Group metrics by file
    file_groups = {}
    for metrics in all_performance_metrics:
        file_name = metrics['file_name']
        if file_name not in file_groups:
            file_groups[file_name] = []
        file_groups[file_name].append(metrics)
    
    # Create traces for each file
    for file_idx, (file_name, file_metrics) in enumerate(file_groups.items()):
        # Sort metrics by interval number
        file_metrics.sort(key=lambda x: x['interval_num'])
        
        x_values = [metrics['responsiveness'] for metrics in file_metrics]
        y_values = [metrics['speed_variability'] for metrics in file_metrics]
        interval_nums = [metrics['interval_num'] for metrics in file_metrics]
        
        # Use Viridis colors for session order (0 to 1)
        color_value = file_idx / (len(file_groups) - 1) if len(file_groups) > 1 else 0.5
        
        # Use opacity for interval progression within file (0.3 to 1.0)
        opacity_values = [0.3 + (i / len(file_metrics)) * 0.7 for i in range(len(file_metrics))]
        
        # Create custom colors by combining Viridis color with opacity
        import plotly.colors as pc
        viridis_colors = pc.sample_colorscale('Viridis', len(file_groups))
        base_color = viridis_colors[file_idx]
        
        # Parse RGB values and apply opacity
        custom_colors = []
        for opacity in opacity_values:
            # Extract RGB values from 'rgb(r, g, b)' format
            if base_color.startswith('rgb('):
                rgb_str = base_color[4:-1]  # Remove 'rgb(' and ')'
                rgb_values = [int(x.strip()) for x in rgb_str.split(',')]
                rgba_color = f'rgba({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]}, {opacity})'
            else:
                # Fallback for hex colors
                hex_color = base_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rgba_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
            custom_colors.append(rgba_color)
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            name=file_name,
            showlegend=False,  # Hide the legend
            marker=dict(
                size=15,
                color=custom_colors,
                showscale=False
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         f'Interval: %{{customdata}}<br>' +
                         f'Responsiveness: %{{x:.2f}}<br>' +
                         f'Speed Variability: %{{y:.3f}}<br>' +
                         '<extra></extra>',
            customdata=interval_nums
        ))
    
    # Add a colorbar for session order (Viridis)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=0,
            color=[0, 1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Session Order",
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Oldest', 'Newest'],
                outlinewidth=0,
                bgcolor='rgba(0,0,0,0)',
                bordercolor='white',
                borderwidth=1,
                x=1.02,  # Position closer to the plot
                len=0.8,  # Make it taller
                thickness=20  # Make it wider
            )
        ),
        showlegend=False
    ))
    
    # Add a colorbar for interval progression (opacity/grayscale)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=0,
            color=[0, 1],
            colorscale=[[0, 'rgba(255,255,255,0.3)'], [1, 'rgba(255,255,255,1.0)']],
            showscale=True,
            colorbar=dict(
                title="Interval Progression",
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Early', 'Late'],
                outlinewidth=0,
                bgcolor='rgba(0,0,0,0)',
                bordercolor='white',
                borderwidth=1,
                x=1.12,  # Position closer to the plot
                len=0.8,  # Make it taller
                thickness=20  # Make it wider
            )
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Performance Metrics: Responsiveness vs Speed Variability (All Sessions)",
        template="plotly_dark",
        xaxis_title="Responsiveness (bpm/s)",
        yaxis_title="Speed Variability (std/mean)",
        height=500,
        margin=dict(r=200)  # Add right margin for colorbars
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the metrics
    st.info("""
    **Performance Metrics Explanation:**
    - **Responsiveness**: How quickly your heart rate responds to exercise intensity
      - Higher values = faster cardiovascular response
      - Formula: (HR_peak - HR_baseline) / (tau_rise + tau_fall)
    - **Speed Variability**: Consistency of your running speed during high-intensity periods
      - Lower values = more consistent pace
      - Formula: Speed_std / Speed_mean
    - **Colors**: Different colors represent different training sessions (Viridis scale: oldest to newest)
    - **Opacity**: Within each session, earlier intervals are more transparent, later intervals are more opaque
    """)

    # Manual correlation threshold calibration
    st.subheader("Manual Threshold Calibration")
    
    # Load saved threshold for this file
    saved_threshold = load_correlation_threshold(filename)
    
    # Initialize session state for this file
    file_key = f"threshold_input_{filename}"
    if file_key not in st.session_state:
        st.session_state[file_key] = saved_threshold if saved_threshold is not None else 0.5
    
    # Initialize window selection for this file
    window_key = f"use_threshold_window_{filename}"
    if window_key not in st.session_state:
        st.session_state[window_key] = False
    
    if saved_threshold is not None:
        st.success(f"✅ Using saved correlation threshold: {saved_threshold:.3f}")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Saved Threshold", type="secondary", key="clear_threshold_btn"):
                clear_correlation_threshold(filename)
                st.session_state[file_key] = 0.5
                st.session_state[window_key] = False
                st.rerun()
    else:
        st.info("🔍 Using automatic threshold detection. Set a manual threshold below if needed.")
    
    # Manual threshold input
    col1, col2, col3 = st.columns(3)
    with col1:
        manual_threshold = st.number_input(
            "Correlation Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state[file_key], 
            step=0.01,
            format="%.3f",
            help="Higher values = more selective detection, Lower values = more inclusive detection",
        )
    
    with col2:
        if st.button("Save Threshold", type="primary", key="save_threshold_btn"):
            save_correlation_threshold(filename, manual_threshold)
            st.success(f"Saved threshold: {manual_threshold:.3f}")
            st.rerun()
    
    with col3:
        st.markdown("**Current Value:**")
        st.markdown(f"{manual_threshold:.3f}")
    
    st.info("💡 Adjust the correlation threshold to fine-tune HIIT detection sensitivity. Higher values detect fewer, more confident intervals.")
    
    # Run detection with manual threshold if available
    manual_threshold = load_correlation_threshold(filename)
    hiit_start, hiit_end, frequency_info = detect_hiit_period_frequency(df, manual_threshold=manual_threshold)
    
    # Show threshold-based window if we have frequency info and update window if needed
    if frequency_info and 'regions_above_threshold' in frequency_info and 'correlation_threshold' in frequency_info:
        regions = frequency_info['regions_above_threshold']
        threshold = frequency_info['correlation_threshold']
        
        if regions:
            # Find the first and last regions above threshold
            first_region_start = regions[0][0]
            last_region_end = regions[-1][1]
            
            # Check if user wants to use threshold-based window
            use_threshold_window = st.session_state[window_key]
            
            # Show the threshold-based window
            st.subheader("Threshold-Based Window")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Threshold", f"{threshold:.3f}")
            with col2:
                st.metric("First Region Start", f"{first_region_start}")
            with col3:
                st.metric("Last Region End", f"{last_region_end}")
            
            # Show current selection status
            if use_threshold_window:
                st.success("🟡 Currently using threshold-based window")
            else:
                st.info("ℹ️ Currently using detected window")
            
            # Show if the detected window matches the threshold window
            if hiit_start == first_region_start and hiit_end == last_region_end:
                st.success("✅ Detected window matches threshold-based window")
            else:
                st.info(f"ℹ️ Detected window ({hiit_start}-{hiit_end}) vs threshold window ({first_region_start}-{last_region_end})")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use Threshold-Based Window", key="use_threshold_window_btn"):
                        st.session_state[window_key] = True
                        st.rerun()
                with col2:
                    if st.button("Use Detected Window", key="use_detected_window_btn"):
                        st.session_state[window_key] = False
                        st.rerun()
            
            # Update window if user chose threshold-based window
            if use_threshold_window:
                hiit_start = first_region_start
                hiit_end = last_region_end
                st.success("🟡 Using threshold-based window")
    
    intervals = []
    if hiit_start is not None and hiit_end is not None:
        df_hiit = df.iloc[hiit_start:hiit_end]
        intervals = segment_intervals_speed_edges(df_hiit)
        for interval in intervals:
            for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                if k in interval:
                    interval[k] += hiit_start


    
    # Display frequency analysis results
    if frequency_info:
        st.subheader("Frequency Analysis Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'dominant_period' in frequency_info:
                st.metric("Dominant Period", f"{frequency_info['dominant_period']:.1f} seconds")
        
        with col2:
            if 'dominant_freq' in frequency_info:
                st.metric("Dominant Frequency", f"{frequency_info['dominant_freq']:.3f} Hz")
        
        with col3:
            if 'correlation_threshold' in frequency_info:
                st.metric("Optimal Threshold", f"{frequency_info['correlation_threshold']:.3f}")
        
        with col4:
            if 'optimization_score' in frequency_info:
                st.metric("Contiguity Score", f"{frequency_info['optimization_score']:.2f}")
    
    # Create multiple figures for better organization
    if intervals:
        # Figure 1: Main detection and frequency analysis
        st.subheader("📊 HIIT Detection & Frequency Analysis")
        fig_main = create_detection_figure(df, hiit_start, hiit_end, intervals, frequency_info, filename)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Figure 2: Interval overlay with exponential fits
        st.subheader("🔄 Interval Overlay with Exponential Fits")
        fig2 = create_interval_overlay_figure(df, intervals)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Figure 3: Speed statistics
        st.subheader("🏃 Speed Statistics by Interval")
        fig3 = create_speed_stats_figure(df, intervals)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Figure 4: Performance metrics distributions
        st.subheader("📈 Performance Metrics Distributions")
        fig4 = create_metrics_distributions_figure(df, intervals)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Figure 5: Top-level metrics (responsiveness and speed variability)
        st.subheader("🎯 Top-Level Performance Metrics")
        fig5 = create_top_level_metrics_figure(df, intervals)
        st.plotly_chart(fig5, use_container_width=True)
    
    # Interval table
    if intervals:
        st.subheader("Detected Intervals")
        interval_data = []
        for interval in intervals:
            interval_data.append({
                'Interval': interval['interval_num'],
                'High Start': str(df.index[interval['high_start']]),
                'High End': str(df.index[interval['high_end']-1]),
                'Recovery Start': str(df.index[interval['recovery_start']]),
                'Recovery End': str(df.index[interval['recovery_end']-1])
            })
        interval_df = pd.DataFrame(interval_data)
        st.dataframe(interval_df, use_container_width=True)
    else:
        st.warning("No intervals detected. Try adjusting the parameters or check if the data contains HIIT patterns.")

def get_correlation_threshold_filename(filename):
    """Get the filename for storing correlation threshold settings."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return f"settings/correlation_threshold_{base_name}.json"

def load_correlation_threshold(filename):
    """Load correlation threshold settings from JSON file."""
    settings_file = get_correlation_threshold_filename(filename)
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            return settings.get('correlation_threshold')
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_correlation_threshold(filename, threshold):
    """Save correlation threshold settings to JSON file."""
    settings_file = get_correlation_threshold_filename(filename)
    settings = {
        'correlation_threshold': threshold,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs(os.path.dirname(settings_file), exist_ok=True)
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)

def clear_correlation_threshold(filename):
    """Clear correlation threshold settings file."""
    settings_file = get_correlation_threshold_filename(filename)
    try:
        os.remove(settings_file)
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main() 