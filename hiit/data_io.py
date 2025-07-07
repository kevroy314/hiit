import os
import json
from datetime import datetime
import pandas as pd
from fitparse import FitFile

def get_manual_window_settings_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return f"./data/{base_name}_manual_window.json"

def load_manual_window_settings(filename):
    settings_file = get_manual_window_settings_filename(filename)
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            return settings.get('manual_start'), settings.get('manual_end')
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None

def save_manual_window_settings(filename, start_idx, end_idx):
    settings_file = get_manual_window_settings_filename(filename)
    os.makedirs(os.path.dirname(settings_file), exist_ok=True)
    settings = {
        'manual_start': start_idx,
        'manual_end': end_idx,
        'timestamp': datetime.now().isoformat()
    }
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)

def load_fit_data(fit_filename):
    """Load and parse FIT file data."""
    fitfile = FitFile(fit_filename)
    timestamps = []
    values = {}
    all_field_names = set()
    for record in fitfile.get_messages('record'):
        for field in record:
            if field.name != 'timestamp':
                all_field_names.add(field.name)
    for field_name in all_field_names:
        values[field_name] = []
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
    df = pd.DataFrame(values, index=timestamps)
    return df

# Alias for backward compatibility
def load_fit_file(fit_filename):
    """Alias for load_fit_data for backward compatibility."""
    return load_fit_data(fit_filename)

def get_correlation_threshold_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return f"settings/correlation_threshold_{base_name}.json"

def load_correlation_threshold(filename):
    settings_file = get_correlation_threshold_filename(filename)
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            return settings.get('correlation_threshold')
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_correlation_threshold(filename, threshold):
    settings_file = get_correlation_threshold_filename(filename)
    settings = {
        'correlation_threshold': threshold,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs(os.path.dirname(settings_file), exist_ok=True)
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)

def clear_correlation_threshold(filename):
    settings_file = get_correlation_threshold_filename(filename)
    try:
        os.remove(settings_file)
    except FileNotFoundError:
        pass
