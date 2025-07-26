"""Functional tests for HIIT Analyzer application."""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import (
    get_fit_files, load_fit_file, save_manual_window_settings,
    load_manual_window_settings, clear_manual_window_settings,
    save_cached_results, load_cached_results
)
from src.algorithm import (
    preprocess_signals, detect_hiit_intervals, find_dominant_frequency,
    calculate_interval_metrics
)
from src.config import Config


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_get_fit_files_empty(self, tmp_path):
        """Test getting FIT files from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily change to test directory
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('data', exist_ok=True)
            
            try:
                files = get_fit_files()
                assert files == []
            finally:
                os.chdir(original_cwd)
    
    def test_manual_window_settings(self, tmp_path):
        """Test saving and loading manual window settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('settings', exist_ok=True)
            
            try:
                # Test save
                save_manual_window_settings('test.fit', 100, 200)
                
                # Test load
                start, end = load_manual_window_settings('test.fit')
                assert start == 100
                assert end == 200
                
                # Test clear
                clear_manual_window_settings('test.fit')
                start, end = load_manual_window_settings('test.fit')
                assert start is None
                assert end is None
            finally:
                os.chdir(original_cwd)
    
    def test_cache_operations(self, tmp_path):
        """Test cache save and load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('settings', exist_ok=True)
            
            try:
                test_data = {'intervals': [1, 2, 3], 'metrics': {'tau': 30}}
                
                # Test save
                save_cached_results('test.fit', 'intervals', test_data)
                
                # Test load
                loaded = load_cached_results('test.fit', 'intervals')
                assert loaded == test_data
                
                # Test missing cache
                missing = load_cached_results('test.fit', 'nonexistent')
                assert missing is None
            finally:
                os.chdir(original_cwd)


class TestAlgorithm:
    """Test algorithm functionality."""
    
    def create_test_dataframe(self, n_samples=1000):
        """Create a test DataFrame with synthetic HIIT data."""
        # Create time index
        time_index = pd.date_range(
            start=datetime.now(),
            periods=n_samples,
            freq='1S'
        )
        
        # Create synthetic HIIT pattern
        # 5 intervals of 30s work / 30s rest
        interval_period = 60  # seconds
        n_intervals = 5
        
        speed = np.zeros(n_samples)
        heart_rate = np.zeros(n_samples)
        
        for i in range(n_intervals):
            start = i * interval_period
            work_end = start + 30
            rest_end = start + interval_period
            
            if rest_end < n_samples:
                # Work phase - high speed
                speed[start:work_end] = 5.0 + np.random.normal(0, 0.5, work_end - start)
                # Rest phase - low speed
                speed[work_end:rest_end] = 1.0 + np.random.normal(0, 0.2, rest_end - work_end)
                
                # Heart rate with exponential response
                hr_work = 120 + 40 * (1 - np.exp(-np.arange(work_end - start) / 10))
                heart_rate[start:work_end] = hr_work + np.random.normal(0, 2, work_end - start)
                
                hr_rest = 160 * np.exp(-np.arange(rest_end - work_end) / 20) + 100
                heart_rate[work_end:rest_end] = hr_rest + np.random.normal(0, 2, rest_end - work_end)
        
        # Add some baseline before and after
        heart_rate[:60] = 80 + np.random.normal(0, 2, 60)
        heart_rate[-60:] = 90 + np.random.normal(0, 2, 60)
        
        df = pd.DataFrame({
            'enhanced_speed': speed,
            'heart_rate': heart_rate,
            'distance': np.cumsum(speed),
            'altitude': 100 + np.random.normal(0, 1, n_samples)
        }, index=time_index)
        
        return df
    
    def test_preprocess_signals(self):
        """Test signal preprocessing."""
        df = self.create_test_dataframe()
        df_processed = preprocess_signals(df)
        
        # Check that filtered columns are created
        assert 'enhanced_speed_filtered' in df_processed.columns
        assert 'heart_rate_filtered' in df_processed.columns
        
        # Check that clipping is applied
        assert df_processed['enhanced_speed_filtered'].min() >= Config.SPEED_CLIP_THRESHOLD
    
    def test_find_dominant_frequency(self):
        """Test dominant frequency detection."""
        # Create a signal with known frequency
        sampling_rate = 1.0  # Hz
        duration = 300  # seconds
        t = np.arange(0, duration, 1/sampling_rate)
        
        # Create signal with 60-second period (0.0167 Hz)
        signal = np.sin(2 * np.pi * (1/60) * t)
        
        freq, period = find_dominant_frequency(signal, sampling_rate)
        
        # Check that detected period is close to 60 seconds
        assert abs(period - 60) < 5  # Within 5 seconds
    
    def test_detect_hiit_intervals(self):
        """Test HIIT interval detection."""
        df = self.create_test_dataframe()
        results = detect_hiit_intervals(df)
        
        # Check that results contain expected keys
        assert 'hiit_start' in results
        assert 'hiit_end' in results
        assert 'intervals' in results
        assert 'dominant_frequency' in results
        assert 'dominant_period' in results
        
        # Check that some intervals were detected
        assert len(results['intervals']) > 0
        
        # Check interval structure
        for interval in results['intervals']:
            assert 'start' in interval
            assert 'end' in interval
            assert 'tau_up' in interval
            assert 'tau_down' in interval
            assert 'duty_cycle' in interval
            assert 'median_top_speed' in interval
    
    def test_calculate_interval_metrics(self):
        """Test interval metrics calculation."""
        df = self.create_test_dataframe()
        
        # Create a simple interval
        interval = {
            'work_start': 60,
            'work_end': 90,
            'recovery_start': 90,
            'recovery_end': 120
        }
        
        metrics = calculate_interval_metrics(interval, df, 1.0)
        
        # Check metrics
        assert metrics['tau_up'] > 0
        assert metrics['tau_down'] > 0
        assert 0 <= metrics['duty_cycle'] <= 1
        assert metrics['median_top_speed'] > 0


class TestConfiguration:
    """Test configuration loading."""
    
    def test_config_values(self):
        """Test that configuration values are loaded correctly."""
        assert isinstance(Config.APP_HOST, str)
        assert isinstance(Config.APP_PORT, int)
        assert isinstance(Config.DEBUG, bool)
        
        # Test numeric conversions
        assert Config.LOWPASS_PERIOD_MIN > 0
        assert Config.LOWPASS_PERIOD_MAX > Config.LOWPASS_PERIOD_MIN
        assert Config.SPEED_CLIP_THRESHOLD > 0
    
    def test_color_mapping(self):
        """Test color configuration."""
        # Test known colors
        assert Config.get_color('heart_rate') == Config.COLORS['heart_rate']
        assert Config.get_color('enhanced_speed') == Config.COLORS['enhanced_speed']
        
        # Test default color for unknown field
        assert Config.get_color('unknown_field') == '#888888'


class TestURLRouting:
    """Test URL routing functionality."""
    
    def test_file_in_url(self):
        """Test that file parameter is included in URL."""
        # This would require running the Dash app, so we just test the logic
        from urllib.parse import urlparse, parse_qs
        
        test_url = "/interval?file=test.fit"
        parsed = urlparse(test_url)
        params = parse_qs(parsed.query)
        
        assert 'file' in params
        assert params['file'][0] == 'test.fit'


def test_directory_structure():
    """Test that required directory structure exists."""
    # Check that src directory exists
    assert os.path.exists('src')
    assert os.path.isdir('src')
    
    # Check required files
    assert os.path.exists('requirements.txt')
    assert os.path.exists('Dockerfile')
    assert os.path.exists('docker-compose.yml')
    assert os.path.exists('.env')
    assert os.path.exists('start_server.sh')
    assert os.path.exists('README.md')
    
    # Check that all code is in src
    python_files_outside_src = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and virtual environments
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__']]
        
        if 'src' not in root and 'tests' not in root:
            for file in files:
                if file.endswith('.py') and file != 'setup.py':
                    python_files_outside_src.append(os.path.join(root, file))
    
    # Should not find any Python files outside src (except tests)
    assert len(python_files_outside_src) == 0, f"Python files found outside src: {python_files_outside_src}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])