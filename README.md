# HIIT Analyzer

A comprehensive application for analyzing High-Intensity Interval Training (HIIT) data from FIT files, including heart rate analysis, interval detection, and performance metrics.

## Available Versions

This repository now includes two versions of the application:

1. **Original Streamlit Version** - `app.py`
2. **New Dash Version** - `app_dash.py` (recommended)

## Features

- 📊 **Raw Data Analysis**: View and analyze time series data from FIT files
- 🔄 **Interval Analysis**: Detect HIIT intervals with frequency analysis and correlation
- 📈 **Performance Metrics**: Comprehensive performance analysis across multiple files
- 🎯 **Interactive Visualizations**: Plotly-based charts with zoom, pan, and selection tools
- ⚙️ **Manual Calibration**: Fine-tune detection parameters with threshold adjustment
- 💾 **Settings Persistence**: Save and load custom thresholds and window settings

## Quick Start - Dash Version (Recommended)

### Prerequisites

- Python 3.7 or higher
- FIT files in the `data/` directory

### Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd hiit-analyzer
   ```

2. **Install Dash dependencies**:
   ```bash
   pip install -r requirements_dash.txt
   ```

3. **Ensure you have FIT files** in the `data/` directory:
   ```bash
   mkdir -p data
   # Copy your .fit files to the data/ directory
   ```

### Running the Dash Application

**Option 1: Direct Python execution**
```bash
python app_dash.py
```

**Option 2: Using convenience scripts**
```bash
# On Linux/macOS
./run_dash.sh

# On Windows
run_dash.bat
```

The application will start on `http://localhost:8050`

### Docker Deployment (Recommended)

For easier deployment, you can use Docker:

```bash
# Build and run with docker-compose (runs both Dash and Streamlit versions)
docker-compose up --build

# Or run just the Dash version
docker-compose up hiit-analyzer-dash

# Or build and run manually
docker build -t hiit-analyzer .
docker run -p 8050:8050 -v $(pwd)/data:/app/data hiit-analyzer
```

### Features of the Dash Version

- **Dark Theme**: Modern dark UI matching the original Streamlit theme
- **Responsive Design**: Works well on desktop and mobile devices
- **Interactive Navigation**: Tab-based navigation between different analysis modes
- **Real-time Updates**: Automatic updates when selecting different files
- **Preserved Functionality**: All original features and visualizations maintained

## Alternative - Streamlit Version

### Installation

```bash
pip install -r requirements.txt
```

### Running the Streamlit Application

```bash
streamlit run app.py
```

The application will start on `http://localhost:8501`

## Usage Guide

### 1. Raw Data Analysis
- **Purpose**: Explore your FIT file data structure and basic metrics
- **Features**:
  - Total records, duration, and distance metrics
  - Data field summary with statistics
  - Time series plots grouped by data type (speeds, positions, altitudes, etc.)

### 2. Interval Analysis
- **Purpose**: Detect and analyze HIIT intervals in your training data
- **Features**:
  - Automatic HIIT period detection using frequency analysis
  - Manual threshold calibration for fine-tuning detection sensitivity
  - Multiple visualization types:
    - Primary HIIT detection plot with heart rate and speed
    - Frequency correlation analysis
    - Period power spectrum
    - Interval overlay with exponential fits
    - Speed statistics by interval
    - Performance metrics distributions

### 3. Performance Metrics
- **Purpose**: Compare performance across multiple training sessions
- **Features**:
  - Aggregated analysis across all FIT files
  - Performance metrics visualization
  - Summary statistics table

### 4. Plotly JS Demo
- **Purpose**: Demonstrate advanced interactive features
- **Features**:
  - Interactive selection tools
  - Real-time bounds display
  - Advanced plot interactions

## Configuration

### File Structure
```
hiit-analyzer/
├── app.py                 # Original Streamlit app
├── app_dash.py           # New Dash app
├── requirements.txt      # Streamlit dependencies
├── requirements_dash.txt # Dash dependencies
├── data/                 # FIT files directory
│   └── *.fit            # Your training files
├── settings/             # App settings (auto-created)
├── hiit/                 # Core analysis modules
│   ├── __init__.py
│   ├── data_io.py       # File I/O operations
│   ├── detection.py     # HIIT detection algorithms
│   ├── metrics.py       # Performance calculations
│   ├── plotting.py      # Visualization functions
│   ├── ui.py           # UI components (Streamlit)
│   └── utils.py        # Utility functions
└── README.md
```

### Supported File Formats
- **FIT files** (`.fit`): Garmin and other fitness device files
- Currently optimized for running/cycling activities with heart rate data

### Manual Threshold Calibration
The application allows you to manually adjust the correlation threshold for HIIT detection:

1. Navigate to **Interval Analysis**
2. Adjust the **Correlation Threshold** (0.0 - 1.0)
   - Higher values = more selective detection
   - Lower values = more inclusive detection
3. Click **Save Threshold** to persist your settings
4. Use **Clear Saved Threshold** to reset to automatic detection

### Settings Persistence
- Correlation thresholds are saved per file in `settings/correlation_threshold_<filename>.json`
- Manual window selections are saved in `data/<filename>_manual_window.json`

## Technical Details

### Key Algorithms
- **Frequency Analysis**: Uses FFT and correlation to detect periodic patterns
- **HIIT Detection**: Template matching with configurable correlation thresholds
- **Interval Segmentation**: Speed-based edge detection for high/recovery periods
- **Performance Metrics**: Heart rate analysis, speed variability, and responsiveness

### Color Scheme
The application maintains a consistent color scheme across both versions:
- **Heart Rate**: Red (`#FF4444`)
- **Speed**: Yellow (`#FFFF44`)
- **Altitude**: Blue (`#4444FF`)
- **Distance/Position**: Green (`#44FF44`)
- **Temperature**: White (`#FFFFFF`)

### Dark Theme
Both versions use a dark theme optimized for data visualization:
- Background: `#0E1117`
- Cards/Panels: `#262730`
- Text: `#FAFAFA`
- Borders: `#464646`

## Troubleshooting

### Common Issues

1. **No FIT files found**
   - Ensure `.fit` files are in the `data/` directory
   - Check file permissions

2. **Import errors**
   - Install dependencies: `pip install -r requirements_dash.txt`
   - Ensure Python 3.7+ is being used

3. **Port conflicts**
   - Dash runs on port 8050 by default
   - Streamlit runs on port 8501 by default
   - Modify the port in the respective app files if needed

4. **Performance issues**
   - Large FIT files may take time to process
   - Consider reducing the analysis window for very long activities

### Development

To modify or extend the application:

1. **Core Logic**: Edit files in the `hiit/` directory
2. **Dash UI**: Modify `app_dash.py`
3. **Streamlit UI**: Modify `hiit/ui.py`
4. **Add Dependencies**: Update `requirements_dash.txt` or `requirements.txt`

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support/contact information here]