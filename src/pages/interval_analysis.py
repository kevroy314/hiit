"""HIIT Interval Analysis page with interactive selection."""

import os
import json
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc

from ..config import Config
from ..data_loader import (
    load_fit_file, get_fit_files, load_manual_window_settings,
    save_manual_window_settings, clear_manual_window_settings,
    load_cached_results, save_cached_results, clear_cached_results
)
from ..algorithm import detect_hiit_intervals, preprocess_signals, calculate_frequency_spectrum


def create_layout(selected_file: Optional[str] = None) -> html.Div:
    """Create the interval analysis page layout.
    
    Args:
        selected_file: Currently selected file path
        
    Returns:
        Dash HTML div containing the page layout
    """
    fit_files = get_fit_files()
    
    if not fit_files:
        return html.Div([
            dbc.Alert("No .fit files found in the data directory.", color="danger")
        ])
    
    # Default to first file if none selected
    if not selected_file or selected_file not in fit_files:
        selected_file = fit_files[0]
    
    return html.Div([
        dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H2("🏃 HIIT Interval Analysis", className="mb-3"),
                ])
            ]),
            
            # File selector
            dbc.Row([
                dbc.Col([
                    html.Label("Select FIT file:"),
                    dcc.Dropdown(
                        id='interval-file-selector',
                        options=[
                            {'label': os.path.basename(f), 'value': f}
                            for f in fit_files
                        ],
                        value=selected_file,
                        clearable=False
                    )
                ], width=6)
            ], className="mb-4"),
            
            # Hidden stores for state management
            dcc.Store(id='interval-selection-store'),
            dcc.Store(id='interval-cache-store'),
            
            # Main detection plot with interactive selection
            dbc.Row([
                dbc.Col([
                    html.H4("HIIT Detection", className="mb-3"),
                    html.Div([
                        dcc.Graph(id='interval-detection-plot'),
                        dbc.ButtonGroup([
                            dbc.Button("Save Selection", id='save-selection-btn', 
                                     color="primary", size="sm"),
                            dbc.Button("Reset Selection", id='reset-selection-btn', 
                                     color="secondary", size="sm")
                        ], className="mt-2")
                    ])
                ])
            ], className="mb-4"),
            
            # Frequency analysis
            dbc.Row([
                dbc.Col([
                    html.H4("Frequency Analysis", className="mb-3"),
                    dbc.Row(id='frequency-metrics-row', className="mb-3"),
                    dcc.Graph(id='frequency-spectrum-plot')
                ], width=6),
                dbc.Col([
                    html.Div(className="mt-5"),  # Spacer
                    dcc.Graph(id='frequency-correlation-plot')
                ], width=6)
            ], className="mb-4"),
            
            # Interval overlay visualization
            dbc.Row([
                dbc.Col([
                    html.H4("Interval Overlay", className="mb-3"),
                    dcc.Graph(id='interval-overlay-plot')
                ])
            ], className="mb-4"),
            
            # Metrics histograms
            dbc.Row([
                dbc.Col([
                    html.H4("Interval Metrics", className="mb-3"),
                    dcc.Graph(id='metrics-histograms')
                ])
            ], className="mb-4"),
            
            # Intervals table
            dbc.Row([
                dbc.Col([
                    html.H4("Detected Intervals", className="mb-3"),
                    html.Div(id='intervals-table')
                ])
            ])
        ], fluid=True)
    ])


@callback(
    Output('interval-cache-store', 'data'),
    [Input('interval-file-selector', 'value'),
     Input('interval-selection-store', 'data')],
    prevent_initial_call=False
)
def process_intervals(filepath: str, selection: Dict) -> Dict:
    """Process intervals when file or selection changes.
    
    Args:
        filepath: Path to selected FIT file
        selection: Manual selection data
        
    Returns:
        Processed interval data
    """
    if not filepath:
        return {}
    
    # Check if we should use cached results
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Load manual window settings if they exist
    manual_start, manual_end = load_manual_window_settings(filepath)
    
    # If selection was just updated, use it
    if trigger_id == 'interval-selection-store' and selection:
        manual_start = selection.get('start_idx')
        manual_end = selection.get('end_idx')
        # Clear cache when selection changes
        clear_cached_results(filepath)
    
    # Try to load cached results
    cached = load_cached_results(filepath, 'intervals')
    if cached and trigger_id != 'interval-selection-store':
        return cached
    
    # Load and process data
    df = load_fit_file(filepath)
    if df is None or df.empty:
        return {}
    
    # Run detection algorithm
    results = detect_hiit_intervals(df, manual_start, manual_end)
    
    # Prepare data for caching
    cache_data = {
        'hiit_start': results['hiit_start'],
        'hiit_end': results['hiit_end'],
        'intervals': results['intervals'],
        'dominant_frequency': results['dominant_frequency'],
        'dominant_period': results['dominant_period'],
        'manual_start': manual_start,
        'manual_end': manual_end,
        'df_shape': df.shape,
        'df_columns': list(df.columns)
    }
    
    # Save to cache
    save_cached_results(filepath, 'intervals', cache_data)
    
    return cache_data


@callback(
    [Output('interval-detection-plot', 'figure'),
     Output('frequency-metrics-row', 'children'),
     Output('frequency-spectrum-plot', 'figure'),
     Output('frequency-correlation-plot', 'figure'),
     Output('interval-overlay-plot', 'figure'),
     Output('metrics-histograms', 'figure'),
     Output('intervals-table', 'children')],
    [Input('interval-cache-store', 'data'),
     Input('interval-file-selector', 'value')]
)
def update_interval_content(cache_data: Dict, filepath: str) -> tuple:
    """Update all interval analysis content.
    
    Args:
        cache_data: Cached interval data
        filepath: Path to selected FIT file
        
    Returns:
        Tuple of all page components
    """
    if not cache_data or not filepath:
        empty_fig = go.Figure()
        return empty_fig, [], empty_fig, empty_fig, empty_fig, empty_fig, None
    
    # Load the full dataframe
    df = load_fit_file(filepath)
    if df is None:
        empty_fig = go.Figure()
        return empty_fig, [], empty_fig, empty_fig, empty_fig, empty_fig, None
    
    # Preprocess signals
    df_processed = preprocess_signals(df)
    
    # Create all visualizations
    detection_fig = create_detection_plot(df_processed, cache_data)
    freq_metrics = create_frequency_metrics(cache_data)
    spectrum_fig = create_frequency_spectrum_plot(df_processed, cache_data)
    correlation_fig = create_correlation_plot(df_processed, cache_data)
    overlay_fig = create_interval_overlay_plot(df_processed, cache_data['intervals'])
    histograms_fig = create_metrics_histograms(cache_data['intervals'])
    intervals_table = create_intervals_table(cache_data['intervals'], df.index)
    
    return (detection_fig, freq_metrics, spectrum_fig, correlation_fig,
            overlay_fig, histograms_fig, intervals_table)


def create_detection_plot(df: pd.DataFrame, cache_data: Dict) -> go.Figure:
    """Create the main detection plot with interactive selection.
    
    Args:
        df: Preprocessed DataFrame
        cache_data: Cached interval data
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Add enhanced speed (raw and filtered)
    if 'enhanced_speed' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['enhanced_speed'],
                name='Enhanced Speed (Raw)',
                line=dict(color=Config.get_color('enhanced_speed'), width=1, opacity=0.3),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['enhanced_speed_filtered'],
                name='Enhanced Speed (Filtered)',
                line=dict(color=Config.get_color('enhanced_speed'), width=2),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add heart rate (raw and filtered) on secondary y-axis
    if 'heart_rate' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['heart_rate'],
                name='Heart Rate (Raw)',
                line=dict(color=Config.get_color('heart_rate'), width=1, opacity=0.3),
                yaxis='y2',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['heart_rate_filtered'],
                name='Heart Rate (Filtered)',
                line=dict(color=Config.get_color('heart_rate'), width=2),
                yaxis='y2',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add HIIT window shading
    hiit_start = cache_data.get('hiit_start', 0)
    hiit_end = cache_data.get('hiit_end', len(df)-1)
    
    if hiit_start < len(df) and hiit_end < len(df):
        fig.add_vrect(
            x0=df.index[hiit_start], x1=df.index[hiit_end],
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            row=1, col=1
        )
    
    # Add interval markers
    for interval in cache_data.get('intervals', []):
        start_idx = interval['work_start']
        end_idx = interval['recovery_end']
        if start_idx < len(df) and end_idx < len(df):
            fig.add_vrect(
                x0=df.index[start_idx], x1=df.index[end_idx],
                fillcolor="orange", opacity=0.2,
                layer="below", line_width=0,
                row=1, col=1
            )
    
    # Create secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title="Heart Rate (bpm)",
            overlaying='y',
            side='right'
        )
    )
    
    # Update layout for interactive selection
    fig.update_layout(
        dragmode='select',
        selectdirection='horizontal',
        title="Click and drag to select HIIT interval",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    
    return fig


def create_frequency_metrics(cache_data: Dict) -> List[dbc.Col]:
    """Create frequency analysis metrics.
    
    Args:
        cache_data: Cached interval data
        
    Returns:
        List of metric columns
    """
    metrics = []
    
    # Dominant frequency
    freq = cache_data.get('dominant_frequency', 0)
    metrics.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{freq:.3f} Hz", className="text-center"),
                    html.P("Dominant Frequency", className="text-center text-muted")
                ])
            ])
        ], width=6)
    )
    
    # Dominant period
    period = cache_data.get('dominant_period', 0)
    metrics.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{period:.1f} s", className="text-center"),
                    html.P("Dominant Period", className="text-center text-muted")
                ])
            ])
        ], width=6)
    )
    
    return metrics


def create_frequency_spectrum_plot(df: pd.DataFrame, cache_data: Dict) -> go.Figure:
    """Create frequency spectrum plot.
    
    Args:
        df: Preprocessed DataFrame
        cache_data: Cached interval data
        
    Returns:
        Plotly figure
    """
    # Get HIIT window
    hiit_start = cache_data.get('hiit_start', 0)
    hiit_end = cache_data.get('hiit_end', len(df))
    
    # Extract signal
    speed_signal = df['enhanced_speed_filtered'].iloc[hiit_start:hiit_end].values
    
    # Calculate sampling rate
    time_diff = df.index.to_series().diff().mean()
    sampling_rate = 1 / time_diff.total_seconds()
    
    # Calculate frequency spectrum
    frequencies, power_spectrum = calculate_frequency_spectrum(speed_signal, sampling_rate)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=power_spectrum,
            mode='lines',
            name='Power Spectrum',
            line=dict(color=Config.get_color('frequency'))
        )
    )
    
    # Add dominant frequency marker
    dominant_freq = cache_data.get('dominant_frequency', 0)
    if dominant_freq > 0:
        fig.add_vline(
            x=dominant_freq,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Dominant: {dominant_freq:.3f} Hz"
        )
    
    fig.update_layout(
        title="Frequency Power Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        height=400,
        showlegend=False
    )
    
    return fig


def create_correlation_plot(df: pd.DataFrame, cache_data: Dict) -> go.Figure:
    """Create frequency correlation plot.
    
    Args:
        df: Preprocessed DataFrame
        cache_data: Cached interval data
        
    Returns:
        Plotly figure
    """
    # Recreate correlations (since they're arrays and hard to cache)
    from ..algorithm import (
        calculate_frequency_correlation, calculate_template_correlation,
        detect_intervals_edge_based
    )
    
    # Get HIIT window
    hiit_start = cache_data.get('hiit_start', 0)
    hiit_end = cache_data.get('hiit_end', len(df))
    df_window = df.iloc[hiit_start:hiit_end]
    
    # Extract signal
    speed_signal = df_window['enhanced_speed_filtered'].values
    
    # Calculate sampling rate
    time_diff = df.index.to_series().diff().mean()
    sampling_rate = 1 / time_diff.total_seconds()
    
    # Calculate correlations
    dominant_freq = cache_data.get('dominant_frequency', 0)
    dominant_period = cache_data.get('dominant_period', 30)
    window_size = int(dominant_period * sampling_rate)
    
    freq_correlation = calculate_frequency_correlation(
        speed_signal, sampling_rate, dominant_freq, window_size
    )
    
    # Recreate intervals for template correlation
    intervals = detect_intervals_edge_based(speed_signal, freq_correlation, sampling_rate)
    template_correlation = calculate_template_correlation(speed_signal, intervals)
    
    # Combined correlation
    combined_correlation = (freq_correlation + template_correlation) / 2
    
    # Create figure
    fig = go.Figure()
    
    # Frequency correlation
    fig.add_trace(
        go.Scatter(
            x=df_window.index,
            y=freq_correlation,
            mode='lines',
            name='Frequency Correlation',
            line=dict(color=Config.get_color('frequency'), width=2)
        )
    )
    
    # Template correlation
    fig.add_trace(
        go.Scatter(
            x=df_window.index,
            y=template_correlation,
            mode='lines',
            name='Template Correlation',
            line=dict(color=Config.get_color('template_correlation'), 
                     width=2, dash='dot')
        )
    )
    
    # Combined correlation
    fig.add_trace(
        go.Scatter(
            x=df_window.index,
            y=combined_correlation,
            mode='lines',
            name='Combined Correlation',
            line=dict(color=Config.get_color('combined_correlation'), width=3)
        )
    )
    
    fig.update_layout(
        title="Correlation Analysis",
        xaxis_title="Time",
        yaxis_title="Correlation",
        height=400,
        showlegend=True
    )
    
    return fig


def create_interval_overlay_plot(df: pd.DataFrame, intervals: List[Dict]) -> go.Figure:
    """Create interval overlay visualization.
    
    Args:
        df: Preprocessed DataFrame
        intervals: List of interval data
        
    Returns:
        Plotly figure
    """
    if not intervals:
        return go.Figure().add_annotation(text="No intervals detected")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Speed', 'Heart Rate')
    )
    
    # Calculate maximum interval length for normalization
    max_length = max(interval['recovery_end'] - interval['work_start'] 
                    for interval in intervals)
    
    # Prepare for mean calculation
    speed_traces = []
    hr_traces = []
    
    # Add each interval
    for i, interval in enumerate(intervals):
        start = interval['work_start']
        end = interval['recovery_end']
        
        # Extract interval data
        interval_speed = df['enhanced_speed'].iloc[start:end].values
        interval_hr = df['heart_rate'].iloc[start:end].values
        
        # Normalize time to [0, 1]
        norm_time = np.linspace(0, 1, len(interval_speed))
        
        # Calculate opacity (first intervals more transparent)
        opacity = 0.3 + (i / len(intervals)) * 0.7
        
        # Add speed trace
        fig.add_trace(
            go.Scatter(
                x=norm_time,
                y=interval_speed,
                mode='lines',
                name=f'Interval {i+1}',
                line=dict(color=Config.get_color('speed'), width=1),
                opacity=opacity,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add heart rate trace
        fig.add_trace(
            go.Scatter(
                x=norm_time,
                y=interval_hr,
                mode='lines',
                line=dict(color=Config.get_color('heart_rate'), width=1),
                opacity=opacity,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Store for mean calculation
        # Resample to common length for averaging
        resampled_speed = np.interp(np.linspace(0, 1, 100), norm_time, interval_speed)
        resampled_hr = np.interp(np.linspace(0, 1, 100), norm_time, interval_hr)
        speed_traces.append(resampled_speed)
        hr_traces.append(resampled_hr)
        
        # Add exponential fits if we have the models
        if 'tau_up' in interval and 'tau_down' in interval:
            # Work phase exponential (rising)
            work_end_norm = (interval['work_end'] - start) / (end - start)
            work_time = np.linspace(0, work_end_norm, 50)
            
            # Recovery phase exponential (falling)  
            recovery_start_norm = (interval['recovery_start'] - start) / (end - start)
            recovery_time = np.linspace(recovery_start_norm, 1, 50)
            
            # Simple exponential visualization (actual fits would need the model parameters)
            # For now, just show trend lines
            fig.add_trace(
                go.Scatter(
                    x=work_time,
                    y=np.linspace(interval_hr[0], np.max(interval_hr), len(work_time)),
                    mode='lines',
                    line=dict(color=Config.get_color('template_correlation'), 
                             width=1, dash='dot'),
                    opacity=opacity,
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Add mean traces
    if speed_traces:
        mean_speed = np.mean(speed_traces, axis=0)
        mean_hr = np.mean(hr_traces, axis=0)
        mean_time = np.linspace(0, 1, 100)
        
        fig.add_trace(
            go.Scatter(
                x=mean_time,
                y=mean_speed,
                mode='lines',
                name='Mean Speed',
                line=dict(color=Config.get_color('speed'), width=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=mean_time,
                y=mean_hr,
                mode='lines',
                name='Mean Heart Rate',
                line=dict(color=Config.get_color('heart_rate'), width=3)
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Normalized Time", row=2, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title="Interval Overlay Analysis"
    )
    
    return fig


def create_metrics_histograms(intervals: List[Dict]) -> go.Figure:
    """Create histograms of interval metrics.
    
    Args:
        intervals: List of interval data
        
    Returns:
        Plotly figure with subplots
    """
    if not intervals:
        return go.Figure().add_annotation(text="No intervals to analyze")
    
    # Extract metrics
    tau_up = [interval['tau_up'] for interval in intervals]
    tau_down = [interval['tau_down'] for interval in intervals]
    duty_cycle = [interval['duty_cycle'] for interval in intervals]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Tau Rise', 'Tau Fall', 'Duty Cycle')
    )
    
    # Tau Rise histogram
    fig.add_trace(
        go.Histogram(
            x=tau_up,
            name='Tau Rise',
            marker_color='purple',
            nbinsx=10
        ),
        row=1, col=1
    )
    
    # Tau Fall histogram
    fig.add_trace(
        go.Histogram(
            x=tau_down,
            name='Tau Fall',
            marker_color='lightblue',
            nbinsx=10
        ),
        row=1, col=2
    )
    
    # Duty Cycle histogram
    fig.add_trace(
        go.Histogram(
            x=duty_cycle,
            name='Duty Cycle',
            marker_color='green',
            nbinsx=10
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="Tau (s)", row=1, col=1)
    fig.update_xaxes(title_text="Tau (s)", row=1, col=2)
    fig.update_xaxes(title_text="Duty Cycle", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title="Interval Metrics Distribution"
    )
    
    return fig


def create_intervals_table(intervals: List[Dict], time_index: pd.DatetimeIndex) -> dash_table.DataTable:
    """Create table of detected intervals.
    
    Args:
        intervals: List of interval data
        time_index: DataFrame time index for timestamps
        
    Returns:
        Dash DataTable
    """
    if not intervals:
        return html.Div("No intervals detected")
    
    table_data = []
    for i, interval in enumerate(intervals):
        # Get timestamps
        work_start_time = time_index[interval['work_start']]
        work_end_time = time_index[interval['work_end']]
        recovery_start_time = time_index[interval['recovery_start']]
        recovery_end_time = time_index[interval['recovery_end']]
        
        table_data.append({
            'Interval': i + 1,
            'Work Start': work_start_time.strftime('%H:%M:%S'),
            'Work End': work_end_time.strftime('%H:%M:%S'),
            'Recovery Start': recovery_start_time.strftime('%H:%M:%S'),
            'Recovery End': recovery_end_time.strftime('%H:%M:%S'),
            'Tau Up (s)': f"{interval['tau_up']:.1f}",
            'Tau Down (s)': f"{interval['tau_down']:.1f}",
            'Duty Cycle': f"{interval['duty_cycle']:.2f}",
            'Max Speed (m/s)': f"{interval['median_top_speed']:.2f}"
        })
    
    return dash_table.DataTable(
        data=table_data,
        columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
        style_cell={'textAlign': 'center'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )


# Callbacks for interactive selection
@callback(
    Output('interval-selection-store', 'data'),
    [Input('save-selection-btn', 'n_clicks')],
    [State('interval-detection-plot', 'selectedData'),
     State('interval-file-selector', 'value'),
     State('interval-cache-store', 'data')],
    prevent_initial_call=True
)
def save_selection(n_clicks, selectedData, filepath, cache_data):
    """Save the user's selection of HIIT interval.
    
    Args:
        n_clicks: Number of button clicks
        selectedData: Selection data from plot
        filepath: Current file path
        cache_data: Current cache data
        
    Returns:
        Selection data to store
    """
    if not n_clicks or not selectedData or not filepath:
        return dash.no_update
    
    # Extract selection range
    if 'range' in selectedData and 'x' in selectedData['range']:
        x_range = selectedData['range']['x']
        
        # Load dataframe to get indices
        df = load_fit_file(filepath)
        if df is not None:
            # Convert timestamps to indices
            start_time = pd.to_datetime(x_range[0])
            end_time = pd.to_datetime(x_range[1])
            
            # Find closest indices
            start_idx = df.index.get_loc(start_time, method='nearest')
            end_idx = df.index.get_loc(end_time, method='nearest')
            
            # Save to file
            save_manual_window_settings(filepath, start_idx, end_idx)
            
            return {'start_idx': start_idx, 'end_idx': end_idx}
    
    return dash.no_update


@callback(
    Output('interval-selection-store', 'data', allow_duplicate=True),
    [Input('reset-selection-btn', 'n_clicks')],
    [State('interval-file-selector', 'value')],
    prevent_initial_call=True
)
def reset_selection(n_clicks, filepath):
    """Reset the manual selection.
    
    Args:
        n_clicks: Number of button clicks
        filepath: Current file path
        
    Returns:
        Empty selection data
    """
    if not n_clicks or not filepath:
        return dash.no_update
    
    # Clear saved settings
    clear_manual_window_settings(filepath)
    clear_cached_results(filepath)
    
    return {}