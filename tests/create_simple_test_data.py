"""Create simple test CSV files for testing."""

import os
import random
from datetime import datetime, timedelta

def create_test_csv(filename, n_intervals=5):
    """Create a test CSV file with HIIT data."""
    
    # No need to create data directory as it's a symlink
    
    # Open file for writing
    with open(filename, 'w') as f:
        # Write header
        f.write('timestamp,heart_rate,enhanced_speed,speed,distance,enhanced_altitude,altitude,temperature\n')
        
        # Generate 20 minutes of data (1200 seconds)
        start_time = datetime.now()
        distance = 0.0
        
        for i in range(1200):
            timestamp = start_time + timedelta(seconds=i)
            
            # Determine if we're in work or rest phase
            interval_num = (i - 120) // 120  # After 2 min warmup, 60s work + 60s rest
            in_interval = interval_num >= 0 and interval_num < n_intervals
            phase = (i - 120) % 120  # Position within interval
            in_work = in_interval and phase < 60
            
            # Set heart rate and speed based on phase
            if i < 120:  # Warmup
                heart_rate = 80 + (i / 120) * 20
                speed = 1.2 + (i / 120) * 0.3
            elif in_work:  # Work phase
                heart_rate = 140 + random.gauss(0, 5) + (phase / 60) * 20
                speed = 4.5 + random.gauss(0, 0.2)
            elif in_interval:  # Recovery phase
                heart_rate = 160 - ((phase - 60) / 60) * 40 + random.gauss(0, 5)
                speed = 1.0 + random.gauss(0, 0.1)
            else:  # Cooldown
                heart_rate = 90 + random.gauss(0, 3)
                speed = 1.0
            
            # Update distance
            distance += speed
            
            # Altitude and temperature
            altitude = 100 + random.gauss(0, 1)
            temperature = 20 + random.gauss(0, 0.5)
            
            # Write row
            f.write(f'{timestamp},{heart_rate:.1f},{speed:.2f},{speed:.2f},'
                   f'{distance:.1f},{altitude:.1f},{altitude:.1f},{temperature:.1f}\n')
    
    print(f"Created {filename}")


# Create test files
create_test_csv('data/test_workout_1.csv', n_intervals=5)
create_test_csv('data/test_workout_2.csv', n_intervals=8)
create_test_csv('data/test_workout_3.csv', n_intervals=4)

print("Created all test CSV files")