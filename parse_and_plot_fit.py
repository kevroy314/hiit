import os
import glob
from fitparse import FitFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from colorama import init, Fore, Back, Style

# Initialize colorama for colored output
init()

def get_field_color(field_name):
    """Return semantically appropriate colors for different field types."""
    field_lower = field_name.lower()
    
    # Heart rate and cardiovascular
    if 'heart' in field_lower or 'hr' in field_lower:
        return '#FF4444'  # Red for heart rate
    
    # Temperature
    if 'temp' in field_lower:
        return '#FFFFFF'  # White for temperature
    
    # Position and GPS
    if any(x in field_lower for x in ['position', 'lat', 'long', 'gps']):
        return '#44FF44'  # Green for position
    
    # Speed and velocity
    if any(x in field_lower for x in ['speed', 'velocity', 'pace']):
        return '#FF8800'  # Orange for speed
    
    # Altitude and elevation
    if any(x in field_lower for x in ['alt', 'elevation', 'height']):
        return '#8844FF'  # Purple for altitude
    
    # Distance
    if 'distance' in field_lower:
        return '#FF44FF'  # Magenta for distance
    
    # Power and energy
    if any(x in field_lower for x in ['power', 'energy', 'watts']):
        return '#FFFF44'  # Yellow for power
    
    # Cadence
    if 'cadence' in field_lower:
        return '#44FFFF'  # Cyan for cadence
    
    # Default
    return '#888888'  # Gray for others

def get_field_unit(field_name):
    """Return appropriate units for different field types."""
    field_lower = field_name.lower()
    
    if 'temp' in field_lower:
        return '°C'
    elif any(x in field_lower for x in ['speed', 'velocity']):
        return 'm/s'
    elif any(x in field_lower for x in ['alt', 'elevation', 'height']):
        return 'm'
    elif 'distance' in field_lower:
        return 'm'
    elif 'heart' in field_lower or 'hr' in field_lower:
        return 'bpm'
    elif 'power' in field_lower:
        return 'W'
    elif 'cadence' in field_lower:
        return 'rpm'
    elif 'time' in field_lower:
        return 's'
    else:
        return ''

def analyze_fit_data(fit_filename):
    """Analyze FIT file data and return comprehensive statistics."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}ANALYZING: {fit_filename}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    fitfile = FitFile(fit_filename)
    
    # Store time series data
    timestamps = []
    values = {}
    all_field_names = set()
    
    # First pass: collect all field names that exist in any record
    for record in fitfile.get_messages('record'):
        for field in record:
            if field.name != 'timestamp':
                all_field_names.add(field.name)
    
    # Initialize value arrays for all fields
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
            # Add values for all fields, using None for missing data
            for field_name in all_field_names:
                values[field_name].append(record_data.get(field_name, None))
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(values, index=timestamps)
    
    # Basic statistics
    print(f"\n{Fore.GREEN}📊 DATA SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Total Records: {len(timestamps):,}{Style.RESET_ALL}")
    
    if timestamps:
        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = end_time - start_time
        print(f"{Fore.YELLOW}Duration: {duration}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Start: {start_time}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}End: {end_time}{Style.RESET_ALL}")
        
        # Calculate total distance if available
        distance_fields = [field for field in all_field_names if 'distance' in field.lower()]
        if distance_fields:
            for field_name in distance_fields:
                if field_name in df.columns:
                    series = df[field_name].dropna()
                    if len(series) > 1:
                        total_distance = series.iloc[-1] - series.iloc[0]
                        unit = get_field_unit(field_name)
                        print(f"{Fore.YELLOW}Total Distance: {total_distance:.2f} {unit}{Style.RESET_ALL}")
                        break  # Use the first distance field found
    
    # Field-by-field analysis
    print(f"\n{Fore.GREEN}📈 FIELD STATISTICS{Style.RESET_ALL}")
    
    for field_name in all_field_names:
        if field_name in df.columns:
            series = df[field_name].dropna()
            if len(series) > 0:
                unit = get_field_unit(field_name)
                missing_count = df[field_name].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                print(f"\n{Fore.BLUE}🔹 {field_name.upper()}{Style.RESET_ALL}")
                print(f"   Records: {len(series):,} / {len(df):,} ({100-missing_pct:.1f}% complete)")
                print(f"   Missing: {missing_count:,} ({missing_pct:.1f}%)")
                
                if series.dtype in ['int64', 'float64']:
                    print(f"   Mean: {series.mean():.2f} {unit}")
                    print(f"   Median: {series.median():.2f} {unit}")
                    print(f"   Std Dev: {series.std():.2f} {unit}")
                    print(f"   Min: {series.min():.2f} {unit}")
                    print(f"   Max: {series.max():.2f} {unit}")
                    
                    # Special calculations for specific fields
                    if 'distance' in field_name.lower():
                        total_distance = series.iloc[-1] - series.iloc[0] if len(series) > 1 else 0
                        print(f"   Total Distance: {total_distance:.2f} {unit}")
                    
                    elif 'alt' in field_name.lower() or 'elevation' in field_name.lower():
                        if len(series) > 1:
                            elevation_gain = sum(np.diff(series)[np.diff(series) > 0])
                            elevation_loss = abs(sum(np.diff(series)[np.diff(series) < 0]))
                            print(f"   Total Climb: {elevation_gain:.2f} {unit}")
                            print(f"   Total Descent: {elevation_loss:.2f} {unit}")
    
    return df, timestamps, values

def segment_intervals(df_hiit, high_duration=60, hr_recovery_threshold=140):
    """
    Segment HIIT data into high/low intervals.
    Args:
        df_hiit: DataFrame containing HIIT period data
        high_duration: Target duration for high intensity periods (seconds)
        hr_recovery_threshold: Heart rate threshold for recovery periods (bpm)
    Returns:
        List of interval dictionaries
    """
    if df_hiit.empty:
        return []
    
    intervals = []
    current_interval = 1
    
    # Get heart rate data
    hr_data = df_hiit['heart_rate'].fillna(method='ffill').fillna(method='bfill')
    
    # Find peaks in heart rate (high intensity periods)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hr_data, 
                         height=np.mean(hr_data) + 0.5 * np.std(hr_data),
                         distance=high_duration)
    
    if len(peaks) < 2:
        return []
    
    # Segment into intervals
    for i in range(len(peaks) - 1):
        high_start = peaks[i]
        high_end = peaks[i + 1]
        
        # Find recovery period (lower heart rate)
        recovery_start = high_end
        recovery_end = len(df_hiit)
        
        # Look for next high intensity period
        if i + 1 < len(peaks):
            recovery_end = peaks[i + 1]
        
        # Calculate metrics for this interval
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
            'interval_num': current_interval,
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
        current_interval += 1
    
    return intervals 