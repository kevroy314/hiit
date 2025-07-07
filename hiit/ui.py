import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import glob
from hiit.data_io import load_fit_file, load_manual_window_settings, save_manual_window_settings
from hiit.detection import detect_hiit_period_frequency, segment_intervals_speed_edges
from hiit.plotting import (
    plot_hiit_detection,
    plot_hiit_detection_altair,
    plot_frequency_correlation,
    plot_frequency_analysis,
    plot_period_power_spectrum,
    plot_interval_analysis,
    create_metrics_visualization,
    plot_raw_data_overview,
    create_interval_overlay_figure,
    create_speed_stats_figure,
    create_metrics_distributions_figure,
    create_top_level_metrics_figure,
)
from hiit.metrics import calculate_interval_metrics, calculate_performance_metrics
from hiit.utils import get_field_group, get_field_color

def main():
    st.set_page_config(layout="wide")
    st.markdown('<h1 class="main-header">🏃‍♂️ HIIT Analyzer</h1>', unsafe_allow_html=True)
    # Navigation with session state
    pages = [
        ("Raw Data Analysis", "raw"),
        ("Interval Analysis", "interval"),
        ("Performance Metrics", "metrics")
    ]
    
    # Create tab-like navigation with URL parameters
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        # Tab navigation using buttons with URL parameter handling
        # Get current page from URL parameters or session state
        url_page = st.query_params.get("page", None)
        
        if url_page and url_page in ["raw", "interval", "metrics"]:
            selected_page = url_page
            st.session_state["selected_page"] = selected_page
        else:
            selected_page = st.session_state.get("selected_page", "raw")
        
        # Create tab buttons in a row
        tab_cols = st.columns(3)
        for i, (name, slug) in enumerate(pages):
            with tab_cols[i]:
                button_type = "primary" if selected_page == slug else "secondary"
                if st.button(name, key=f"nav_{slug}", type=button_type):
                    st.session_state["selected_page"] = slug
                    # Update URL parameter
                    st.query_params["page"] = slug
                    st.rerun()
    
    # Get file from URL parameter for pages that need it
    url_file = st.query_params.get("file", None)
    fit_files = glob.glob('data/*.fit')
    
    # Page logic
    if selected_page == "raw":
        show_raw_data_page(url_file)
    elif selected_page == "interval":
        show_interval_analysis_page(url_file)
    elif selected_page == "metrics":
        show_metrics_page()

def show_raw_data_page(filename):
    st.header("📊 Raw Data Analysis")
    # File selection
    fit_files = glob.glob('data/*.fit')
    if not fit_files:
        st.error("No .fit files found in the data directory.")
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
    # Load data
    with st.spinner("Loading FIT file..."):
        df = load_fit_file(filename)
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
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    if len(numeric_cols) > 0:
        # Group fields by category
        field_groups = {}
        for col in numeric_cols:
            group = get_field_group(col)
            if group not in field_groups:
                field_groups[group] = []
            field_groups[group].append(col)
        # Create subplots for each group (max 3 per row)
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import numpy as np
        for group_name, group_fields in field_groups.items():
            if not group_fields:
                continue
            st.markdown(f"**{group_name}**")
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

    df = load_fit_file(filename)
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
    # Use refined boundaries if available, otherwise use original boundaries
    analysis_start = hiit_start
    analysis_end = hiit_end
    
    print(f"DEBUG: UI - Original boundaries: {hiit_start}-{hiit_end}")
    
    if frequency_info and 'refined_start' in frequency_info and 'refined_end' in frequency_info:
        if frequency_info['refined_start'] is not None and frequency_info['refined_end'] is not None:
            analysis_start = frequency_info['refined_start']
            analysis_end = frequency_info['refined_end']
            print(f"DEBUG: UI - Using refined boundaries: {analysis_start}-{analysis_end}")
        else:
            print(f"DEBUG: UI - Refined boundaries are None")
    else:
        print(f"DEBUG: UI - No refined boundaries in frequency_info")
    
    print(f"DEBUG: UI - Final analysis boundaries: {analysis_start}-{analysis_end}")
    
    if analysis_start is not None and analysis_end is not None:
        df_hiit = df.iloc[analysis_start:analysis_end]
        intervals = segment_intervals_speed_edges(df_hiit)
        for interval in intervals:
            for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                if k in interval:
                    interval[k] += analysis_start
    
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
    # Always show all figures, even if no intervals detected
    
    # Figure 1: Primary detection figure using Altair (interactive)
    st.subheader("🎯 Interactive HIIT Detection (Altair)")
    fig_altair = plot_hiit_detection_altair(df, hiit_start, hiit_end, intervals, frequency_info, filename)
    if fig_altair:
        st.altair_chart(fig_altair, use_container_width=True)
        st.info("💡 **Interactive Features**: Click and drag to select regions, hover for details. This will be enhanced with boundary selection soon!")
    
    # Figure 2: Primary detection figure (speed, heart rate, intervals, HIIT window) - Plotly version
    st.subheader("📊 Primary HIIT Detection (Plotly)")
    fig_main = plot_hiit_detection(df, hiit_start, hiit_end, intervals, frequency_info, filename)
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Figure 3: Frequency correlation analysis
    if frequency_info:
        st.subheader("🔍 Frequency & Template Correlation Analysis")
        fig_corr = plot_frequency_correlation(df, frequency_info)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Figure 4: Frequency spectrum analysis
    if frequency_info:
        st.subheader("📈 Frequency Spectrum Analysis")
        fig_freq = plot_frequency_analysis(frequency_info)
        if fig_freq:
            st.plotly_chart(fig_freq, use_container_width=True)
    
    # Figure 5: Period power spectrum
    if frequency_info:
        st.subheader("⚡ Period Power Spectrum")
        fig_period = plot_period_power_spectrum(frequency_info)
        if fig_period:
            st.plotly_chart(fig_period, use_container_width=True)
    
    # Figure 6: Interval overlay with exponential fits
    st.subheader("🔄 Interval Overlay with Exponential Fits")
    fig2 = create_interval_overlay_figure(df, intervals, frequency_info)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Figure 7: Speed statistics
    st.subheader("🏃 Speed Statistics by Interval")
    fig3 = create_speed_stats_figure(df, intervals)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Figure 8: Performance metrics distributions
    st.subheader("📈 Performance Metrics Distributions")
    fig4 = create_metrics_distributions_figure(df, intervals)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Figure 9: Top-level metrics (responsiveness and speed variability)
    st.subheader("🎯 Top-Level Performance Metrics")
    fig5 = create_top_level_metrics_figure(df, intervals)
    st.plotly_chart(fig5, use_container_width=True)
    
    # Interval table
    st.subheader("Detected Intervals")
    if intervals:
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
        # Show empty table with headers
        empty_df = pd.DataFrame(columns=['Interval', 'High Start', 'High End', 'Recovery Start', 'Recovery End'])
        st.dataframe(empty_df, use_container_width=True)

def show_metrics_page():
    st.header("📈 Performance Metrics")
    
    # Load all available files
    fit_files = glob.glob('data/*.fit')
    if not fit_files:
        st.error("No .fit files found in the data directory.")
        return
    
    # Process all files
    all_intervals = []
    all_metrics = []
    
    for file in fit_files:
        df = load_fit_file(file)
        if df is None:
            continue
        
        # Run detection
        hiit_start, hiit_end, frequency_info = detect_hiit_period_frequency(df)
        
        if hiit_start is not None and hiit_end is not None:
            df_hiit = df.iloc[hiit_start:hiit_end]
            intervals = segment_intervals_speed_edges(df_hiit)
            
            # Adjust indices
            for interval in intervals:
                for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                    if k in interval:
                        interval[k] += hiit_start
                interval['filename'] = os.path.basename(file)
            
            all_intervals.extend(intervals)
            
            # Calculate performance metrics
            if intervals:
                metrics = calculate_performance_metrics(df, intervals)
                metrics['filename'] = os.path.basename(file)
                all_metrics.append(metrics)
    
    if all_intervals:
        # Create metrics visualization
        intervals_df = pd.DataFrame(all_intervals)
        fig = create_metrics_visualization(intervals_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("No intervals found in any files.")

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
