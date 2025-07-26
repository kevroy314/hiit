"""HIIT detection and analysis algorithm implementation."""

import numpy as np
import pandas as pd
from scipy import signal, optimize
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Any

from .config import Config


def preprocess_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess signals with lowpass filtering and clipping.
    
    Args:
        df: DataFrame containing raw signals
        
    Returns:
        DataFrame with preprocessed signals
    """
    df_processed = df.copy()
    
    # Calculate sampling rate
    time_diff = df.index.to_series().diff().mean()
    sampling_rate = 1 / time_diff.total_seconds()
    
    # Lowpass filter parameters
    # Convert period in minutes to frequency in Hz
    freq_high = 1 / (Config.LOWPASS_PERIOD_MIN * 60)
    freq_low = 1 / (Config.LOWPASS_PERIOD_MAX * 60)
    
    # Design butterworth filter
    nyquist = sampling_rate / 2
    b, a = signal.butter(3, [freq_low/nyquist, freq_high/nyquist], btype='band')
    
    # Apply filter to enhanced_speed and heart_rate
    if 'enhanced_speed' in df.columns:
        speed_values = df['enhanced_speed'].fillna(method='ffill').fillna(method='bfill')
        df_processed['enhanced_speed_filtered'] = signal.filtfilt(b, a, speed_values)
        # Clip speed values
        df_processed['enhanced_speed_filtered'] = df_processed['enhanced_speed_filtered'].clip(
            lower=Config.SPEED_CLIP_THRESHOLD
        )
    
    if 'heart_rate' in df.columns:
        hr_values = df['heart_rate'].fillna(method='ffill').fillna(method='bfill')
        df_processed['heart_rate_filtered'] = signal.filtfilt(b, a, hr_values)
    
    return df_processed


def calculate_frequency_spectrum(signal_data: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate frequency spectrum of a signal.
    
    Args:
        signal_data: Signal values
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (frequencies, power_spectrum)
    """
    # Compute FFT
    n = len(signal_data)
    fft_values = fft(signal_data)
    frequencies = fftfreq(n, 1/sampling_rate)
    
    # Get positive frequencies only
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    power_spectrum = np.abs(fft_values[positive_freq_idx]) ** 2
    
    return frequencies, power_spectrum


def find_dominant_frequency(signal_data: np.ndarray, sampling_rate: float) -> Tuple[float, float]:
    """Find dominant frequency in the signal.
    
    Args:
        signal_data: Signal values
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (dominant_frequency, dominant_period)
    """
    frequencies, power_spectrum = calculate_frequency_spectrum(signal_data, sampling_rate)
    
    # Find frequency range of interest (periods between MIN and MAX interval periods)
    freq_min = 1 / Config.MAX_INTERVAL_PERIOD
    freq_max = 1 / Config.MIN_INTERVAL_PERIOD
    
    mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    if not np.any(mask):
        return 0.0, 0.0
    
    # Find peak in the frequency range
    masked_power = power_spectrum[mask]
    masked_freq = frequencies[mask]
    
    peak_idx = np.argmax(masked_power)
    dominant_freq = masked_freq[peak_idx]
    dominant_period = 1 / dominant_freq if dominant_freq > 0 else 0.0
    
    return dominant_freq, dominant_period


def calculate_frequency_correlation(signal_data: np.ndarray, sampling_rate: float, 
                                  dominant_freq: float, window_size: int) -> np.ndarray:
    """Calculate correlation of frequency at various points in time.
    
    Args:
        signal_data: Signal values
        sampling_rate: Sampling rate in Hz
        dominant_freq: Dominant frequency to correlate with
        window_size: Window size for correlation calculation
        
    Returns:
        Array of correlation values
    """
    n = len(signal_data)
    correlations = np.zeros(n)
    
    for i in range(n):
        if i + window_size > n:
            break
            
        window = signal_data[i:i+window_size]
        
        # Calculate FFT for this window
        fft_values = fft(window)
        frequencies = fftfreq(window_size, 1/sampling_rate)
        
        # Find power at dominant frequency
        freq_mask = np.abs(frequencies - dominant_freq) < (0.5 * sampling_rate / window_size)
        if np.any(freq_mask):
            power_at_dominant = np.max(np.abs(fft_values[freq_mask]))
            total_power = np.sum(np.abs(fft_values))
            correlations[i] = power_at_dominant / total_power if total_power > 0 else 0
    
    return correlations


def detect_intervals_edge_based(speed_signal: np.ndarray, correlation: np.ndarray,
                               sampling_rate: float) -> List[Dict[str, int]]:
    """Detect intervals using edge detection and clustering.
    
    Args:
        speed_signal: Speed signal values
        correlation: Frequency correlation values
        sampling_rate: Sampling rate in Hz
        
    Returns:
        List of interval dictionaries with start/end indices
    """
    # Calculate gradient for edge detection
    speed_gradient = np.gradient(speed_signal)
    
    # Find rising edges (positive gradient above threshold)
    gradient_threshold = np.percentile(np.abs(speed_gradient), 90)
    rising_edges = np.where(speed_gradient > gradient_threshold)[0]
    
    if len(rising_edges) < 2:
        return []
    
    # Cluster rising edges to group nearby ones
    clustering = DBSCAN(eps=sampling_rate * 5, min_samples=1)  # 5 second epsilon
    rising_edges_2d = rising_edges.reshape(-1, 1)
    clusters = clustering.fit_predict(rising_edges_2d)
    
    # Get cluster centers (interval starts)
    interval_starts = []
    for cluster_id in np.unique(clusters):
        cluster_edges = rising_edges[clusters == cluster_id]
        interval_starts.append(cluster_edges[0])
    
    # Sort interval starts
    interval_starts = sorted(interval_starts)
    
    # Find interval ends (next start or falling edge)
    intervals = []
    for i in range(len(interval_starts) - 1):
        start = interval_starts[i]
        end = interval_starts[i + 1]
        
        # Find actual end by looking for speed drop
        speed_segment = speed_signal[start:end]
        if len(speed_segment) > 0:
            # Find where speed drops below median
            median_speed = np.median(speed_segment)
            below_median = np.where(speed_segment < median_speed)[0]
            if len(below_median) > 0:
                actual_end = start + below_median[0]
            else:
                actual_end = end
            
            intervals.append({
                'start': start,
                'end': actual_end,
                'work_start': start,
                'work_end': actual_end,
                'recovery_start': actual_end,
                'recovery_end': end
            })
    
    # Handle last interval
    if len(interval_starts) > 0:
        start = interval_starts[-1]
        end = len(speed_signal) - 1
        intervals.append({
            'start': start,
            'end': end,
            'work_start': start,
            'work_end': end,
            'recovery_start': end,
            'recovery_end': end
        })
    
    return intervals


def calculate_template_correlation(signal_data: np.ndarray, intervals: List[Dict[str, int]]) -> np.ndarray:
    """Calculate template correlation using average of detected intervals.
    
    Args:
        signal_data: Signal values
        intervals: List of detected intervals
        
    Returns:
        Template correlation array
    """
    if not intervals:
        return np.zeros_like(signal_data)
    
    # Find the median interval length
    interval_lengths = [interval['end'] - interval['start'] for interval in intervals]
    median_length = int(np.median(interval_lengths))
    
    # Create template by averaging intervals
    template = np.zeros(median_length)
    count = 0
    
    for interval in intervals:
        start = interval['start']
        end = interval['end']
        interval_signal = signal_data[start:end]
        
        # Resample to median length
        if len(interval_signal) > 0:
            resampled = np.interp(
                np.linspace(0, 1, median_length),
                np.linspace(0, 1, len(interval_signal)),
                interval_signal
            )
            template += resampled
            count += 1
    
    if count > 0:
        template /= count
    
    # Calculate correlation with template
    correlation = signal.correlate(signal_data, template, mode='same')
    correlation = correlation / np.max(np.abs(correlation))
    
    return correlation


def refine_intervals_with_hr(intervals: List[Dict[str, int]], heart_rate: np.ndarray) -> List[Dict[str, int]]:
    """Refine intervals using heart rate data, especially for the last interval.
    
    Args:
        intervals: List of detected intervals
        heart_rate: Heart rate signal
        
    Returns:
        Refined list of intervals
    """
    if not intervals:
        return intervals
    
    # Calculate median heart rate at low points
    low_hr_values = []
    for interval in intervals[:-1]:  # Exclude last interval
        recovery_start = interval['recovery_start']
        recovery_end = interval['recovery_end']
        if recovery_end > recovery_start:
            recovery_hr = heart_rate[recovery_start:recovery_end]
            if len(recovery_hr) > 0:
                low_hr_values.append(np.min(recovery_hr))
    
    if low_hr_values:
        hr_threshold = np.median(low_hr_values)
        
        # Refine last interval
        last_interval = intervals[-1]
        start = last_interval['start']
        
        # Find where heart rate drops back to threshold
        hr_segment = heart_rate[start:]
        below_threshold = np.where(hr_segment <= hr_threshold)[0]
        
        if len(below_threshold) > 0:
            end = start + below_threshold[0]
            intervals[-1]['end'] = end
            intervals[-1]['work_end'] = end
            intervals[-1]['recovery_start'] = end
            intervals[-1]['recovery_end'] = end
    
    return intervals


def check_contiguity(intervals: List[Dict[str, int]], max_gap_samples: int) -> List[Dict[str, int]]:
    """Check contiguity of intervals and remove outliers.
    
    Args:
        intervals: List of detected intervals
        max_gap_samples: Maximum allowed gap between intervals
        
    Returns:
        Filtered list of contiguous intervals
    """
    if len(intervals) <= 1:
        return intervals
    
    # Calculate gaps between intervals
    filtered_intervals = [intervals[0]]
    
    for i in range(1, len(intervals)):
        prev_end = filtered_intervals[-1]['end']
        curr_start = intervals[i]['start']
        gap = curr_start - prev_end
        
        if gap <= max_gap_samples:
            filtered_intervals.append(intervals[i])
    
    return filtered_intervals


def fit_exponential_rise(time: np.ndarray, hr: np.ndarray) -> Tuple[float, float, float]:
    """Fit exponential rise model to heart rate data.
    
    Args:
        time: Time values
        hr: Heart rate values
        
    Returns:
        Tuple of (HR0, HRpeak, tau)
    """
    def exp_rise(t, hr0, hr_peak, tau):
        return hr0 + (hr_peak - hr0) * (1 - np.exp(-t / tau))
    
    try:
        # Initial parameter guesses
        hr0_guess = hr[0]
        hr_peak_guess = np.max(hr)
        tau_guess = len(time) / 4
        
        popt, _ = optimize.curve_fit(
            exp_rise, time, hr,
            p0=[hr0_guess, hr_peak_guess, tau_guess],
            maxfev=Config.FIT_MAX_ITERATIONS
        )
        return tuple(popt)
    except:
        return hr[0], np.max(hr), len(time) / 4


def fit_exponential_fall(time: np.ndarray, hr: np.ndarray) -> Tuple[float, float, float]:
    """Fit exponential fall model to heart rate data.
    
    Args:
        time: Time values
        hr: Heart rate values
        
    Returns:
        Tuple of (HRpeak, HRbaseline, tau)
    """
    def exp_fall(t, hr_peak, hr_baseline, tau):
        return hr_baseline + (hr_peak - hr_baseline) * np.exp(-t / tau)
    
    try:
        # Initial parameter guesses
        hr_peak_guess = hr[0]
        hr_baseline_guess = np.min(hr)
        tau_guess = len(time) / 4
        
        popt, _ = optimize.curve_fit(
            exp_fall, time, hr,
            p0=[hr_peak_guess, hr_baseline_guess, tau_guess],
            maxfev=Config.FIT_MAX_ITERATIONS
        )
        return tuple(popt)
    except:
        return hr[0], np.min(hr), len(time) / 4


def calculate_interval_metrics(interval: Dict[str, int], df: pd.DataFrame,
                             sampling_rate: float) -> Dict[str, float]:
    """Calculate metrics for a single interval.
    
    Args:
        interval: Interval dictionary with start/end indices
        df: DataFrame with signal data
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of metrics
    """
    # Extract work and recovery phases
    work_start = interval['work_start']
    work_end = interval['work_end']
    recovery_start = interval['recovery_start']
    recovery_end = interval['recovery_end']
    
    # Calculate duty cycle
    work_duration = (work_end - work_start) / sampling_rate
    total_duration = (recovery_end - work_start) / sampling_rate
    duty_cycle = work_duration / total_duration if total_duration > 0 else 0
    
    # Extract heart rate data
    work_hr = df['heart_rate'].iloc[work_start:work_end].values
    recovery_hr = df['heart_rate'].iloc[recovery_start:recovery_end].values
    
    # Fit exponential models
    work_time = np.arange(len(work_hr)) / sampling_rate
    recovery_time = np.arange(len(recovery_hr)) / sampling_rate
    
    tau_up = 30.0  # Default
    tau_down = 60.0  # Default
    
    if len(work_hr) > 3:
        _, _, tau_up = fit_exponential_rise(work_time, work_hr)
    
    if len(recovery_hr) > 3:
        _, _, tau_down = fit_exponential_fall(recovery_time, recovery_hr)
    
    # Calculate speed metrics
    speed_data = df['enhanced_speed'].iloc[work_start:work_end].values
    if len(speed_data) > 0:
        # Top quartile median speed
        speed_threshold = np.percentile(speed_data, 75)
        top_speeds = speed_data[speed_data >= speed_threshold]
        median_top_speed = np.median(top_speeds) if len(top_speeds) > 0 else 0
    else:
        median_top_speed = 0
    
    return {
        'tau_up': tau_up,
        'tau_down': tau_down,
        'duty_cycle': duty_cycle,
        'median_top_speed': median_top_speed,
        'work_duration': work_duration,
        'recovery_duration': (recovery_end - recovery_start) / sampling_rate
    }


def detect_hiit_intervals(df: pd.DataFrame, manual_start: Optional[int] = None,
                         manual_end: Optional[int] = None) -> Dict[str, Any]:
    """Main HIIT detection algorithm.
    
    Args:
        df: DataFrame with preprocessed signals
        manual_start: Manual override for HIIT window start
        manual_end: Manual override for HIIT window end
        
    Returns:
        Dictionary containing detection results
    """
    # Calculate sampling rate
    time_diff = df.index.to_series().diff().mean()
    sampling_rate = 1 / time_diff.total_seconds()
    
    # Preprocess signals
    df_processed = preprocess_signals(df)
    
    # Use manual window if provided, otherwise use full signal
    if manual_start is not None and manual_end is not None:
        df_window = df_processed.iloc[manual_start:manual_end]
        window_start = manual_start
    else:
        df_window = df_processed
        window_start = 0
    
    # Extract filtered signals
    speed_signal = df_window['enhanced_speed_filtered'].values
    hr_signal = df_window['heart_rate_filtered'].values
    
    # Step 1: Calculate frequency spectrum
    dominant_freq, dominant_period = find_dominant_frequency(speed_signal, sampling_rate)
    
    # Step 2: Calculate frequency correlation
    window_size = int(dominant_period * sampling_rate) if dominant_period > 0 else int(30 * sampling_rate)
    freq_correlation = calculate_frequency_correlation(speed_signal, sampling_rate, dominant_freq, window_size)
    
    # Step 3: Detect intervals using edge detection
    intervals = detect_intervals_edge_based(speed_signal, freq_correlation, sampling_rate)
    
    # Step 4: Calculate template correlation
    template_correlation = calculate_template_correlation(speed_signal, intervals)
    
    # Step 5: Combine correlations
    combined_correlation = (freq_correlation + template_correlation) / 2
    
    # Re-detect intervals with combined correlation
    intervals = detect_intervals_edge_based(speed_signal, combined_correlation, sampling_rate)
    
    # Step 6: Check contiguity
    max_gap_samples = int(Config.INTERVAL_MAX_GAP * sampling_rate)
    intervals = check_contiguity(intervals, max_gap_samples)
    
    # Refine with heart rate
    intervals = refine_intervals_with_hr(intervals, hr_signal)
    
    # Step 7: Calculate metrics for each interval
    interval_metrics = []
    for interval in intervals:
        # Adjust indices to original dataframe
        adjusted_interval = {
            'start': interval['start'] + window_start,
            'end': interval['end'] + window_start,
            'work_start': interval['work_start'] + window_start,
            'work_end': interval['work_end'] + window_start,
            'recovery_start': interval['recovery_start'] + window_start,
            'recovery_end': interval['recovery_end'] + window_start
        }
        
        metrics = calculate_interval_metrics(adjusted_interval, df_processed, sampling_rate)
        interval_metrics.append({**adjusted_interval, **metrics})
    
    # Find overall HIIT window
    if intervals:
        hiit_start = intervals[0]['start'] + window_start
        hiit_end = intervals[-1]['end'] + window_start
    else:
        hiit_start = window_start
        hiit_end = window_start + len(speed_signal)
    
    return {
        'hiit_start': hiit_start,
        'hiit_end': hiit_end,
        'intervals': interval_metrics,
        'dominant_frequency': dominant_freq,
        'dominant_period': dominant_period,
        'frequency_correlation': freq_correlation,
        'template_correlation': template_correlation,
        'combined_correlation': combined_correlation,
        'sampling_rate': sampling_rate
    }