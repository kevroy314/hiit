#!/usr/bin/env python3
"""
HIIT Analyzer - Dash Application

A Dash application for analyzing High-Intensity Interval Training (HIIT) data
from FIT files, including heart rate analysis, interval detection, and performance metrics.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import os
import json
from datetime import datetime
import glob
import numpy as np

# Import our existing modules
try:
    from hiit.data_io import load_fit_file, load_manual_window_settings, save_manual_window_settings, clear_manual_window_settings
    from hiit.detection import detect_hiit_period_frequency, segment_intervals_speed_edges
    from hiit.plotting import (
        plot_hiit_detection,
        plot_frequency_correlation,
        create_metrics_visualization,
        plot_frequency_analysis,
        plot_period_power_spectrum,
        plot_interval_analysis,
        create_interval_overlay_figure,
        create_speed_stats_figure,
        create_metrics_distributions_figure,
        create_top_level_metrics_figure,
    )
    from hiit.metrics import calculate_interval_metrics, calculate_performance_metrics
    from hiit.utils import get_field_group, get_field_color
except ImportError as e:
    print(f"Error importing HIIT modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements_dash.txt")
    exit(1)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Define custom CSS styles that match the Streamlit theme
custom_styles = {
    'dark_theme': {
        'backgroundColor': '#0E1117',
        'color': '#FAFAFA',
    },
    'card_style': {
        'backgroundColor': '#262730',
        'border': '1px solid #464646',
        'borderRadius': '8px',
        'margin': '10px 0px',
        'padding': '20px'
    },
    'metric_card': {
        'backgroundColor': '#1E1E1E',
        'border': '1px solid #464646',
        'borderRadius': '8px',
        'padding': '15px',
        'textAlign': 'center',
        'margin': '5px'
    },
    'nav_button_active': {
        'backgroundColor': '#FF4B4B',
        'color': 'white',
        'border': 'none',
        'borderRadius': '6px',
        'padding': '8px 16px',
        'margin': '0 5px',
        'fontWeight': 'bold'
    },
    'nav_button_inactive': {
        'backgroundColor': '#262730',
        'color': '#FAFAFA',
        'border': '1px solid #464646',
        'borderRadius': '6px',
        'padding': '8px 16px',
        'margin': '0 5px'
    }
}

# App layout
app.layout = dbc.Container([
    # Store components for maintaining state
    dcc.Store(id='selected-file-store'),
    dcc.Store(id='current-page-store', data='raw'),
    dcc.Store(id='correlation-threshold-store'),
    dcc.Store(id='bokeh-selection-store'),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🏃‍♂️ HIIT Analyzer", className="main-header text-center mb-4",
                   style={'color': '#FAFAFA', 'fontWeight': 'bold'})
        ])
    ]),
    
    # Navigation tabs
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("Raw Data Analysis", id="nav-raw", n_clicks=0, 
                          color="danger", outline=False),
                dbc.Button("Interval Analysis", id="nav-interval", n_clicks=0, 
                          color="secondary", outline=True),
                dbc.Button("Performance Metrics", id="nav-metrics", n_clicks=0, 
                          color="secondary", outline=True),
                dbc.Button("Plotly JS Demo", id="nav-plotly", n_clicks=0, 
                          color="secondary", outline=True),
            ], className="mb-4")
        ])
    ]),
    
    # Main content area
    html.Div(id="page-content", style=custom_styles['dark_theme']),
    
], fluid=True, style=custom_styles['dark_theme'])

# Callback for navigation
@app.callback(
    [Output('current-page-store', 'data'),
     Output('nav-raw', 'color'),
     Output('nav-raw', 'outline'),
     Output('nav-interval', 'color'),
     Output('nav-interval', 'outline'),
     Output('nav-metrics', 'color'),
     Output('nav-metrics', 'outline'),
     Output('nav-plotly', 'color'),
     Output('nav-plotly', 'outline')],
    [Input('nav-raw', 'n_clicks'),
     Input('nav-interval', 'n_clicks'),
     Input('nav-metrics', 'n_clicks'),
     Input('nav-plotly', 'n_clicks')],
    prevent_initial_call=False
)
def update_navigation(raw_clicks, interval_clicks, metrics_clicks, plotly_clicks):
    ctx = callback_context
    if not ctx.triggered:
        page = 'raw'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        page = button_id.split('-')[1]
    
    # Set button styles based on active page
    colors = ['secondary'] * 4
    outlines = [True] * 4
    
    page_index = {'raw': 0, 'interval': 1, 'metrics': 2, 'plotly': 3}
    if page in page_index:
        colors[page_index[page]] = 'danger'
        outlines[page_index[page]] = False
    
    return page, colors[0], outlines[0], colors[1], outlines[1], colors[2], outlines[2], colors[3], outlines[3]

# Main page content callback
@app.callback(
    Output('page-content', 'children'),
    [Input('current-page-store', 'data'),
     Input('selected-file-store', 'data')]
)
def display_page(page, selected_file):
    if page == 'raw':
        return create_raw_data_page(selected_file)
    elif page == 'interval':
        return create_interval_analysis_page(selected_file)
    elif page == 'metrics':
        return create_metrics_page()
    elif page == 'plotly':
        return create_plotly_js_page(selected_file)
    else:
        return create_raw_data_page(selected_file)

def get_fit_files():
    """Get list of available FIT files."""
    return glob.glob('data/*.fit')

def create_file_selector(selected_file=None, page_id=""):
    """Create file selector dropdown."""
    fit_files = get_fit_files()
    if not fit_files:
        return dbc.Alert("No .fit files found in the data directory.", color="danger")
    
    options = [{"label": os.path.basename(f), "value": f} for f in fit_files]
    value = selected_file if selected_file in fit_files else fit_files[0]
    
    return dcc.Dropdown(
        id=f'file-selector-{page_id}',
        options=options,
        value=value,
        placeholder="Select FIT file",
        style={'backgroundColor': '#262730', 'color': '#FAFAFA'}
    )

def create_raw_data_page(selected_file):
    """Create the raw data analysis page layout."""
    fit_files = get_fit_files()
    if not fit_files:
        return dbc.Alert("No .fit files found in the data directory.", color="danger")
    
    file_selector = create_file_selector(selected_file, "raw")
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📊 Raw Data Analysis", className="mb-3"),
                file_selector
            ])
        ]),
        
        # Metrics row
        dbc.Row(id="raw-metrics-row", className="mb-4"),
        
        # Data summary
        dbc.Row([
            dbc.Col([
                html.H4("Data Summary", className="mb-3"),
                html.Div(id="data-summary-table")
            ])
        ]),
        
        # Time series plots
        dbc.Row([
            dbc.Col([
                html.H4("Time Series Data", className="mb-3"),
                html.Div(id="time-series-plots")
            ])
        ])
    ], style=custom_styles['card_style'])

def create_interval_analysis_page(selected_file):
    """Create the interval analysis page layout."""
    fit_files = get_fit_files()
    if not fit_files:
        return dbc.Alert("No .fit files found in the data directory.", color="danger")
    
    file_selector = create_file_selector(selected_file, "interval")
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🔄 Interval Analysis", className="mb-3"),
                file_selector
            ])
        ]),
        
        # Manual threshold calibration
        dbc.Row([
            dbc.Col([
                html.H4("Manual Threshold Calibration", className="mb-3"),
                html.Div(id="threshold-calibration")
            ])
        ]),
        
        # Frequency analysis results
        dbc.Row(id="frequency-analysis-row", className="mb-4"),
        
        # Main plots
        html.Div(id="interval-analysis-plots"),
        
        # Interval table
        dbc.Row([
            dbc.Col([
                html.H4("Detected Intervals", className="mb-3"),
                html.Div(id="intervals-table")
            ])
        ])
    ], style=custom_styles['card_style'])

def create_metrics_page():
    """Create the performance metrics page layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📈 Performance Metrics", className="mb-3"),
                html.Div(id="metrics-content")
            ])
        ])
    ], style=custom_styles['card_style'])

def create_plotly_js_page(selected_file):
    """Create the Plotly JS demo page layout."""
    fit_files = get_fit_files()
    if not fit_files:
        return dbc.Alert("No .fit files found in the data directory.", color="danger")
    
    file_selector = create_file_selector(selected_file, "plotly")
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📊 Plotly JS Demo - Real-time Selection Bounds", className="mb-3"),
                file_selector
            ])
        ]),
        
        html.Div(id="plotly-js-content")
    ], style=custom_styles['card_style'])

# File selector callbacks for each page
@app.callback(
    Output('selected-file-store', 'data'),
    [Input('file-selector-raw', 'value'),
     Input('file-selector-interval', 'value'),
     Input('file-selector-plotly', 'value')],
    prevent_initial_call=True
)
def update_selected_file(raw_file, interval_file, plotly_file):
    ctx = callback_context
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'file-selector-raw':
        return raw_file
    elif trigger_id == 'file-selector-interval':
        return interval_file
    elif trigger_id == 'file-selector-plotly':
        return plotly_file
    return None

# Raw data page callbacks
@app.callback(
    [Output('raw-metrics-row', 'children'),
     Output('data-summary-table', 'children'),
     Output('time-series-plots', 'children')],
    [Input('file-selector-raw', 'value')],
    prevent_initial_call=True
)
def update_raw_data_content(filename):
    if not filename:
        return [], [], []
    
    try:
        df = load_fit_file(filename)
        if df is None or df.empty:
            return [dbc.Alert("Failed to load data from the selected file.", color="danger")], [], []
        
        # Create metrics
        metrics = []
        
        # Total records
        metrics.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(df):,}", className="text-center mb-0"),
                        html.P("Total Records", className="text-center text-muted mb-0")
                    ])
                ], style=custom_styles['metric_card'])
            ], width=4)
        )
        
        # Duration
        if len(df) > 0:
            duration = df.index[-1] - df.index[0]
            metrics.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(str(duration), className="text-center mb-0"),
                            html.P("Duration", className="text-center text-muted mb-0")
                        ])
                    ], style=custom_styles['metric_card'])
                ], width=4)
            )
        
        # Distance
        if 'distance' in df.columns:
            distance_series = df['distance'].dropna()
            if len(distance_series) > 1:
                total_distance = distance_series.iloc[-1] - distance_series.iloc[0]
                metrics.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{total_distance:.2f} m", className="text-center mb-0"),
                                html.P("Total Distance", className="text-center text-muted mb-0")
                            ])
                        ], style=custom_styles['metric_card'])
                    ], width=4)
                )
        
        # Create data summary table
        summary_data = []
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                summary_data.append({
                    'Field': col,
                    'Records': len(series),
                    'Missing': df[col].isna().sum(),
                    'Mean': f"{series.mean():.2f}" if series.dtype in ['int64', 'float64'] else 'N/A',
                    'Std': f"{series.std():.2f}" if series.dtype in ['int64', 'float64'] else 'N/A',
                    'Min': f"{series.min():.2f}" if series.dtype in ['int64', 'float64'] else 'N/A',
                    'Max': f"{series.max():.2f}" if series.dtype in ['int64', 'float64'] else 'N/A'
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create table
        table = dbc.Table.from_dataframe(
            summary_df, 
            striped=True, 
            bordered=True, 
            hover=True,
            dark=True,
            responsive=True
        )
        
        # Create time series plots
        plots = create_time_series_plots(df)
        
        return metrics, table, plots
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error loading file: {str(e)}", color="danger")
        return [error_msg], [], []

def create_time_series_plots(df):
    """Create time series plots grouped by category."""
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    if len(numeric_cols) == 0:
        return []
    
    # Group fields by category
    field_groups = {}
    for col in numeric_cols:
        group = get_field_group(col)
        if group not in field_groups:
            field_groups[group] = []
        field_groups[group].append(col)
    
    plots = []
    for group_name, group_fields in field_groups.items():
        if not group_fields:
            continue
        
        # Create subplot figure
        from plotly.subplots import make_subplots
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
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        plots.append(
            html.Div([
                html.H5(f"**{group_name}**", className="mb-3"),
                dcc.Graph(figure=fig, style={'height': f'{250 * n_rows}px'}),
                html.Hr()
            ])
        )
    
    return plots

# Interval analysis page callbacks
@app.callback(
    [Output('threshold-calibration', 'children'),
     Output('frequency-analysis-row', 'children'),
     Output('interval-analysis-plots', 'children'),
     Output('intervals-table', 'children')],
    [Input('file-selector-interval', 'value')],
    prevent_initial_call=True
)
def update_interval_analysis_content(filename):
    if not filename:
        return [], [], [], []
    
    try:
        df = load_fit_file(filename)
        if df is None or df.empty:
            return [dbc.Alert("Failed to load data from the selected file.", color="danger")], [], [], []
        
        # Get saved threshold
        saved_threshold = load_correlation_threshold(filename)
        
        # Create threshold calibration section
        threshold_section = create_threshold_calibration_section(filename, saved_threshold)
        
        # Run detection
        manual_threshold = load_correlation_threshold(filename)
        result = detect_hiit_period_frequency(df, manual_threshold=manual_threshold)
        if result[0] is not None:
            hiit_start, hiit_end, frequency_info = result
        else:
            hiit_start, hiit_end, frequency_info = None, None, {}
        
        # Create frequency analysis metrics
        frequency_metrics = create_frequency_analysis_metrics(frequency_info)
        
        # Create intervals
        intervals = []
        analysis_start = hiit_start
        analysis_end = hiit_end
        
        if frequency_info and 'refined_start' in frequency_info and 'refined_end' in frequency_info:
            if frequency_info['refined_start'] is not None and frequency_info['refined_end'] is not None:
                analysis_start = frequency_info['refined_start']
                analysis_end = frequency_info['refined_end']
        
        if analysis_start is not None and analysis_end is not None:
            df_hiit = df.iloc[analysis_start:analysis_end]
            intervals = segment_intervals_speed_edges(df_hiit)
            for interval in intervals:
                for k in ['interval_start', 'interval_end', 'high_start', 'high_end', 'recovery_start', 'recovery_end']:
                    if k in interval:
                        interval[k] += analysis_start
        
        # Create plots
        plots = create_interval_analysis_plots(df, hiit_start, hiit_end, intervals, frequency_info, filename)
        
        # Create intervals table
        intervals_table = create_intervals_table(df, intervals)
        
        return threshold_section, frequency_metrics, plots, intervals_table
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error processing file: {str(e)}", color="danger")
        return [error_msg], [], [], []

def create_threshold_calibration_section(filename, saved_threshold):
    """Create the threshold calibration section."""
    if saved_threshold is not None:
        alert = dbc.Alert(f"✅ Using saved correlation threshold: {saved_threshold:.3f}", color="success")
        clear_button = dbc.Button("Clear Saved Threshold", color="secondary", size="sm", id="clear-threshold-btn")
    else:
        alert = dbc.Alert("🔍 Using automatic threshold detection. Set a manual threshold below if needed.", color="info")
        clear_button = html.Div()
    
    return [
        alert,
        dbc.Row([
            dbc.Col([
                dbc.Label("Correlation Threshold"),
                dbc.Input(
                    id="threshold-input",
                    type="number",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=saved_threshold if saved_threshold is not None else 0.5,
                    placeholder="0.500"
                )
            ], width=4),
            dbc.Col([
                dbc.Button("Save Threshold", color="primary", id="save-threshold-btn", className="mt-4")
            ], width=2),
            dbc.Col([
                clear_button
            ], width=2),
        ]),
        dbc.Alert("💡 Adjust the correlation threshold to fine-tune HIIT detection sensitivity. Higher values detect fewer, more confident intervals.", color="info", className="mt-3")
    ]

def create_frequency_analysis_metrics(frequency_info):
    """Create frequency analysis metrics display."""
    if not frequency_info:
        return []
    
    metrics = []
    
    if 'dominant_period' in frequency_info:
        metrics.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{frequency_info['dominant_period']:.1f} sec", className="text-center mb-0"),
                        html.P("Dominant Period", className="text-center text-muted mb-0")
                    ])
                ], style=custom_styles['metric_card'])
            ], width=3)
        )
    
    if 'dominant_freq' in frequency_info:
        metrics.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{frequency_info['dominant_freq']:.3f} Hz", className="text-center mb-0"),
                        html.P("Dominant Frequency", className="text-center text-muted mb-0")
                    ])
                ], style=custom_styles['metric_card'])
            ], width=3)
        )
    
    if 'correlation_threshold' in frequency_info:
        metrics.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{frequency_info['correlation_threshold']:.3f}", className="text-center mb-0"),
                        html.P("Optimal Threshold", className="text-center text-muted mb-0")
                    ])
                ], style=custom_styles['metric_card'])
            ], width=3)
        )
    
    if 'optimization_score' in frequency_info:
        metrics.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{frequency_info['optimization_score']:.2f}", className="text-center mb-0"),
                        html.P("Contiguity Score", className="text-center text-muted mb-0")
                    ])
                ], style=custom_styles['metric_card'])
            ], width=3)
        )
    
    return metrics

def create_interval_analysis_plots(df, hiit_start, hiit_end, intervals, frequency_info, filename):
    """Create all the interval analysis plots."""
    plots = []
    
    # Primary HIIT Detection
    try:
        fig_main = plot_hiit_detection(df, hiit_start, hiit_end, intervals, frequency_info, filename)
        plots.append(
            html.Div([
                html.H4("📊 Primary HIIT Detection (Plotly)", className="mb-3"),
                dcc.Graph(figure=fig_main)
            ])
        )
    except Exception as e:
        plots.append(dbc.Alert(f"Error creating main plot: {str(e)}", color="warning"))
    
    # Frequency correlation analysis
    if frequency_info:
        try:
            fig_corr = plot_frequency_correlation(df, frequency_info)
            if fig_corr:
                plots.append(
                    html.Div([
                        html.H4("🔍 Frequency & Template Correlation Analysis", className="mb-3"),
                        dcc.Graph(figure=fig_corr)
                    ])
                )
        except Exception as e:
            plots.append(dbc.Alert(f"Error creating correlation plot: {str(e)}", color="warning"))
    
    # Frequency spectrum analysis
    if frequency_info:
        try:
            fig_freq = plot_frequency_analysis(frequency_info)
            if fig_freq:
                plots.append(
                    html.Div([
                        html.H4("📈 Frequency Spectrum Analysis", className="mb-3"),
                        dcc.Graph(figure=fig_freq)
                    ])
                )
        except Exception as e:
            plots.append(dbc.Alert(f"Error creating frequency plot: {str(e)}", color="warning"))
    
    # Additional plots
    plot_functions = [
        ("⚡ Period Power Spectrum", plot_period_power_spectrum, frequency_info),
        ("🔄 Interval Overlay with Exponential Fits", create_interval_overlay_figure, (df, intervals, frequency_info)),
        ("🏃 Speed Statistics by Interval", create_speed_stats_figure, (df, intervals)),
        ("📈 Performance Metrics Distributions", create_metrics_distributions_figure, (df, intervals)),
        ("🎯 Top-Level Performance Metrics", create_top_level_metrics_figure, (df, intervals))
    ]
    
    for title, func, args in plot_functions:
        try:
            if isinstance(args, tuple):
                fig = func(*args)
            else:
                fig = func(args)
            if fig:
                plots.append(
                    html.Div([
                        html.H4(title, className="mb-3"),
                        dcc.Graph(figure=fig)
                    ])
                )
        except Exception as e:
            plots.append(dbc.Alert(f"Error creating {title.lower()}: {str(e)}", color="warning"))
    
    return plots

def create_intervals_table(df, intervals):
    """Create the intervals table."""
    if not intervals:
        return dbc.Alert("No intervals detected. Try adjusting the parameters or check if the data contains HIIT patterns.", color="warning")
    
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
    return dbc.Table.from_dataframe(
        interval_df, 
        striped=True, 
        bordered=True, 
        hover=True,
        dark=True,
        responsive=True
    )

# Metrics page callback
@app.callback(
    Output('metrics-content', 'children'),
    [Input('current-page-store', 'data')],
    prevent_initial_call=True
)
def update_metrics_content(page):
    if page != 'metrics':
        return []
    
    try:
        # Load all available files
        fit_files = get_fit_files()
        if not fit_files:
            return dbc.Alert("No .fit files found in the data directory.", color="danger")
        
        # Process all files
        all_intervals = []
        all_metrics = []
        
        for file in fit_files:
            df = load_fit_file(file)
            if df is None:
                continue
            
            # Run detection
            result = detect_hiit_period_frequency(df)
            if result[0] is not None:
                hiit_start, hiit_end, frequency_info = result
            else:
                hiit_start, hiit_end, frequency_info = None, None, {}
            
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
            
            content = [
                dcc.Graph(figure=fig) if fig else dbc.Alert("No metrics visualization available", color="info")
            ]
            
            # Display summary statistics
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                table = dbc.Table.from_dataframe(
                    metrics_df, 
                    striped=True, 
                    bordered=True, 
                    hover=True,
                    dark=True,
                    responsive=True
                )
                content.append(
                    html.Div([
                        html.H4("Summary Statistics", className="mb-3 mt-4"),
                        table
                    ])
                )
            
            return content
        else:
            return dbc.Alert("No intervals found in any files.", color="warning")
            
    except Exception as e:
        return dbc.Alert(f"Error processing metrics: {str(e)}", color="danger")

# Plotly JS page callback
@app.callback(
    Output('plotly-js-content', 'children'),
    [Input('file-selector-plotly', 'value')],
    prevent_initial_call=True
)
def update_plotly_js_content(filename):
    if not filename:
        return dbc.Alert("Please select a FIT file to analyze.", color="info")
    
    try:
        # Load and process data
        df = load_fit_file(filename)
        
        if df is not None and not df.empty:
            # Detect HIIT periods
            result = detect_hiit_period_frequency(df)
            if result[0] is not None:
                hiit_start, hiit_end, frequency_info = result
            else:
                hiit_start, hiit_end, frequency_info = None, None, {}
            
            # Segment intervals
            intervals = segment_intervals_speed_edges(df.iloc[hiit_start:hiit_end]) if hiit_start is not None and hiit_end is not None else []
            
            # Create the main plot
            fig_main = plot_hiit_detection(df, hiit_start, hiit_end, intervals, frequency_info, os.path.basename(filename))
            
            return [
                dbc.Alert([
                    html.H4("🎯 JavaScript Injection Demo:", className="alert-heading"),
                    html.P("This example demonstrates interactive plot features similar to the Streamlit version."),
                    html.Hr(),
                    html.P([
                        "Instructions:",
                        html.Br(),
                        "1. Use the selection tool (box icon) in the plot toolbar",
                        html.Br(),
                        "2. Click and drag to select a region on the plot",
                        html.Br(),
                        "3. Use interactive zoom, pan, and selection tools"
                    ])
                ], color="info", className="mb-4"),
                
                dcc.Graph(figure=fig_main),
                
                dbc.Alert([
                    html.H5("Features:"),
                    html.Ul([
                        html.Li("Dual y-axes for heart rate (red) and speed (blue)"),
                        html.Li("Auto-detected HIIT boundaries (red dashed lines)"),
                        html.Li("Interval overlays (colored rectangles)"),
                        html.Li("Interactive zoom, pan, and selection tools")
                    ])
                ], color="secondary")
            ]
        else:
            return dbc.Alert("❌ Failed to load data from the selected file.", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"❌ Error processing file: {str(e)}", color="danger")

# Helper functions for correlation threshold management
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

# Threshold management callbacks
@app.callback(
    Output('threshold-input', 'value'),
    [Input('save-threshold-btn', 'n_clicks'),
     Input('clear-threshold-btn', 'n_clicks')],
    [State('threshold-input', 'value'),
     State('file-selector-interval', 'value')],
    prevent_initial_call=True
)
def handle_threshold_buttons(save_clicks, clear_clicks, threshold_value, filename):
    ctx = callback_context
    if not ctx.triggered or not filename:
        return threshold_value
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'save-threshold-btn' and threshold_value is not None:
        save_correlation_threshold(filename, threshold_value)
        return threshold_value
    elif button_id == 'clear-threshold-btn':
        clear_correlation_threshold(filename)
        return 0.5
    
    return threshold_value

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)