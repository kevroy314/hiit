import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from typing import Tuple, Callable

def exp_rise(t, HR0, HRpeak, tau):
    """Exponential rise function for heart rate modeling."""
    return HR0 + (HRpeak - HR0) * (1 - np.exp(-t / tau))

def exp_fall(t, HRpeak, HRbaseline, tau):
    """Exponential fall function for heart rate modeling."""
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
    hr_data = high_period['heart_rate'].ffill().bfill()
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
        speed_data = high_period['enhanced_speed'].ffill().bfill()
        
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
        speed_mean = 0
        speed_std = 0
    
    return {
        'interval_num': interval['interval_num'],
        'hr_peak': hr_peak,
        'hr_baseline': hr_baseline,
        'tau_rise': tau_rise,
        'tau_fall': tau_fall,
        'responsiveness': responsiveness,
        'speed_variability': speed_variability,
        'speed_mean': speed_mean,
        'speed_std': speed_std
    }

def calculate_interval_metrics(df, intervals):
    """
    Calculate metrics for all intervals.
    
    Args:
        df: DataFrame with the full dataset
        intervals: List of interval dictionaries
    
    Returns:
        List of dictionaries with metrics for each interval
    """
    metrics = []
    for interval in intervals:
        interval_metrics = analyze_interval_performance(df, interval)
        if interval_metrics:
            metrics.append(interval_metrics)
    return metrics

def calculate_performance_metrics(df, intervals):
    """
    Calculate overall performance metrics from intervals.
    
    Args:
        df: DataFrame with the full dataset
        intervals: List of interval dictionaries
    
    Returns:
        Dictionary with overall performance metrics
    """
    if not intervals:
        return {}
    
    # Calculate individual interval metrics
    interval_metrics = calculate_interval_metrics(df, intervals)
    
    if not interval_metrics:
        return {}
    
    # Aggregate metrics
    tau_rise_values = [m['tau_rise'] for m in interval_metrics]
    tau_fall_values = [m['tau_fall'] for m in interval_metrics]
    responsiveness_values = [m['responsiveness'] for m in interval_metrics]
    speed_variability_values = [m['speed_variability'] for m in interval_metrics]
    hr_peak_values = [m['hr_peak'] for m in interval_metrics]
    hr_baseline_values = [m['hr_baseline'] for m in interval_metrics]
    speed_mean_values = [m['speed_mean'] for m in interval_metrics]
    speed_std_values = [m['speed_std'] for m in interval_metrics]
    
    return {
        'num_intervals': len(intervals),
        'avg_tau_rise': np.mean(tau_rise_values),
        'avg_tau_fall': np.mean(tau_fall_values),
        'avg_responsiveness': np.mean(responsiveness_values),
        'avg_speed_variability': np.mean(speed_variability_values),
        'avg_hr_peak': np.mean(hr_peak_values),
        'avg_hr_baseline': np.mean(hr_baseline_values),
        'avg_speed_mean': np.mean(speed_mean_values),
        'avg_speed_std': np.mean(speed_std_values),
        'std_tau_rise': np.std(tau_rise_values),
        'std_tau_fall': np.std(tau_fall_values),
        'std_responsiveness': np.std(responsiveness_values),
        'std_speed_variability': np.std(speed_variability_values)
    }
