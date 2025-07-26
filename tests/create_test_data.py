"""Create test FIT files for testing the HIIT Analyzer application."""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fitparse import FitFile, FitFileEncoder

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_test_fit_file(filename: str, n_intervals: int = 5, base_hr: int = 80, 
                        work_hr: int = 160, base_speed: float = 1.0, 
                        work_speed: float = 4.5) -> None:
    """Create a test FIT file with synthetic HIIT data.
    
    Args:
        filename: Output filename
        n_intervals: Number of HIIT intervals
        base_hr: Base heart rate
        work_hr: Work phase heart rate
        base_speed: Base speed (m/s)
        work_speed: Work phase speed (m/s)
    """
    # Create time series - 20 minutes total
    duration_seconds = 1200
    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(duration_seconds)]
    
    # Initialize arrays
    heart_rate = np.ones(duration_seconds) * base_hr
    speed = np.ones(duration_seconds) * base_speed
    
    # Warmup period (first 2 minutes)
    warmup_duration = 120
    for i in range(warmup_duration):
        heart_rate[i] = base_hr + (i / warmup_duration) * 20
        speed[i] = base_speed + (i / warmup_duration) * 0.5
    
    # HIIT intervals
    interval_duration = 60  # 60s work
    recovery_duration = 60  # 60s recovery
    
    for interval in range(n_intervals):
        start_time = warmup_duration + interval * (interval_duration + recovery_duration)
        
        # Work phase
        work_start = start_time
        work_end = start_time + interval_duration
        
        if work_end < duration_seconds:
            # Speed jumps immediately
            speed[work_start:work_end] = work_speed + np.random.normal(0, 0.2, min(interval_duration, duration_seconds - work_start))
            
            # Heart rate rises exponentially
            for i in range(min(interval_duration, duration_seconds - work_start)):
                t = i / 20.0  # Time constant ~20s
                heart_rate[work_start + i] = base_hr + (work_hr - base_hr) * (1 - np.exp(-t))
                heart_rate[work_start + i] += np.random.normal(0, 2)
        
        # Recovery phase
        recovery_start = work_end
        recovery_end = recovery_start + recovery_duration
        
        if recovery_start < duration_seconds and recovery_end <= duration_seconds:
            # Speed drops immediately
            speed[recovery_start:recovery_end] = base_speed + np.random.normal(0, 0.1, min(recovery_duration, duration_seconds - recovery_start))
            
            # Heart rate falls exponentially
            peak_hr = heart_rate[work_end - 1] if work_end < duration_seconds else work_hr
            for i in range(min(recovery_duration, duration_seconds - recovery_start)):
                t = i / 30.0  # Time constant ~30s
                heart_rate[recovery_start + i] = base_hr + (peak_hr - base_hr) * np.exp(-t)
                heart_rate[recovery_start + i] += np.random.normal(0, 2)
    
    # Cool down (last 2 minutes)
    cooldown_start = warmup_duration + n_intervals * (interval_duration + recovery_duration)
    if cooldown_start < duration_seconds:
        for i in range(cooldown_start, duration_seconds):
            progress = (i - cooldown_start) / (duration_seconds - cooldown_start)
            heart_rate[i] = heart_rate[cooldown_start - 1] * (1 - progress) + base_hr * progress
            speed[i] = base_speed
    
    # Calculate distance
    distance = np.cumsum(speed)
    
    # Add altitude variation
    altitude = 100 + np.sin(np.linspace(0, 4*np.pi, duration_seconds)) * 5 + np.random.normal(0, 0.5, duration_seconds)
    
    # Temperature
    temperature = 20 + np.random.normal(0, 1, duration_seconds)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'enhanced_speed': speed,
        'speed': speed,
        'distance': distance,
        'enhanced_altitude': altitude,
        'altitude': altitude,
        'temperature': temperature,
        'position_lat': 37.7749 + np.random.normal(0, 0.0001, duration_seconds),
        'position_long': -122.4194 + np.random.normal(0, 0.0001, duration_seconds)
    })
    
    # Save as CSV for now (since creating actual FIT files is complex)
    # In practice, we'd convert this to a proper FIT file
    df.to_csv(filename.replace('.fit', '.csv'), index=False)
    
    print(f"Created test file: {filename}")


def create_all_test_files():
    """Create multiple test FIT files with different characteristics."""
    os.makedirs('data', exist_ok=True)
    
    # Test file 1: Standard HIIT workout
    create_test_fit_file('data/test_workout_1.fit', 
                        n_intervals=5, base_hr=80, work_hr=160, 
                        base_speed=1.2, work_speed=4.5)
    
    # Test file 2: Shorter intervals
    create_test_fit_file('data/test_workout_2.fit', 
                        n_intervals=8, base_hr=75, work_hr=155, 
                        base_speed=1.0, work_speed=4.0)
    
    # Test file 3: Higher intensity
    create_test_fit_file('data/test_workout_3.fit', 
                        n_intervals=4, base_hr=85, work_hr=170, 
                        base_speed=1.3, work_speed=5.0)
    
    print("Created all test files in data/ directory")


if __name__ == '__main__':
    create_all_test_files()