"""Data loader module for reading and processing FIT files."""

import os
import json
import glob
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
from fitparse import FitFile


def get_fit_files() -> List[str]:
    """Get list of available FIT files in the data directory.
    
    Returns:
        List of FIT file paths
    """
    return sorted(glob.glob('data/*.fit'))


def load_fit_file(filepath: str) -> Optional[pd.DataFrame]:
    """Load data from a FIT file.
    
    Args:
        filepath: Path to the FIT file
        
    Returns:
        DataFrame with the FIT file data or None if loading fails
    """
    try:
        fitfile = FitFile(filepath)
        
        # Extract all data records
        records = []
        for record in fitfile.get_messages('record'):
            data = {}
            for field in record:
                if field.value is not None:
                    data[field.name] = field.value
            if data:
                records.append(data)
        
        if not records:
            return None
            
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Convert timestamp to datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading FIT file {filepath}: {e}")
        return None


def get_manual_window_settings_path(filepath: str) -> str:
    """Get the path for manual window settings file.
    
    Args:
        filepath: Path to the FIT file
        
    Returns:
        Path to the settings JSON file
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join('settings', f'{basename}_manual_window.json')


def load_manual_window_settings(filepath: str) -> Tuple[Optional[int], Optional[int]]:
    """Load manual window settings for a FIT file.
    
    Args:
        filepath: Path to the FIT file
        
    Returns:
        Tuple of (start_idx, end_idx) or (None, None) if not found
    """
    settings_path = get_manual_window_settings_path(filepath)
    
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            return settings.get('manual_start'), settings.get('manual_end')
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None


def save_manual_window_settings(filepath: str, start_idx: int, end_idx: int) -> None:
    """Save manual window settings for a FIT file.
    
    Args:
        filepath: Path to the FIT file
        start_idx: Start index of the manual window
        end_idx: End index of the manual window
    """
    settings_path = get_manual_window_settings_path(filepath)
    os.makedirs('settings', exist_ok=True)
    
    settings = {
        'manual_start': start_idx,
        'manual_end': end_idx,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)


def clear_manual_window_settings(filepath: str) -> None:
    """Clear manual window settings for a FIT file.
    
    Args:
        filepath: Path to the FIT file
    """
    settings_path = get_manual_window_settings_path(filepath)
    try:
        os.remove(settings_path)
    except FileNotFoundError:
        pass


def get_cache_path(filepath: str, cache_type: str) -> str:
    """Get the path for cached algorithm results.
    
    Args:
        filepath: Path to the FIT file
        cache_type: Type of cache (e.g., 'intervals', 'metrics')
        
    Returns:
        Path to the cache file
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join('settings', f'{basename}_{cache_type}.cache')


def load_cached_results(filepath: str, cache_type: str) -> Optional[Any]:
    """Load cached algorithm results.
    
    Args:
        filepath: Path to the FIT file
        cache_type: Type of cache to load
        
    Returns:
        Cached data or None if not found
    """
    cache_path = get_cache_path(filepath, cache_type)
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_cached_results(filepath: str, cache_type: str, data: Any) -> None:
    """Save algorithm results to cache.
    
    Args:
        filepath: Path to the FIT file
        cache_type: Type of cache
        data: Data to cache
    """
    cache_path = get_cache_path(filepath, cache_type)
    os.makedirs('settings', exist_ok=True)
    
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)


def clear_cached_results(filepath: str) -> None:
    """Clear all cached results for a FIT file.
    
    Args:
        filepath: Path to the FIT file
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    cache_files = glob.glob(os.path.join('settings', f'{basename}_*.cache'))
    
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
        except FileNotFoundError:
            pass