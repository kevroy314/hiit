"""Performance Analysis page analyzing all files."""

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, callback, Input, Output
import dash_bootstrap_components as dbc

from ..config import Config
from ..data_loader import get_fit_files, load_fit_file, load_cached_results
from ..algorithm import detect_hiit_intervals


def create_layout() -> html.Div:
    """Create the performance analysis page layout.
    
    Returns:
        Dash HTML div containing the page layout
    """
    return html.Div([
        dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H2("📈 Performance Analysis", className="mb-3"),
                    html.P("Analysis of all workouts in the data directory",
                          className="text-muted")
                ])
            ]),
            
            # Files summary table
            dbc.Row([
                dbc.Col([
                    html.H4("Files Summary", className="mb-3"),
                    html.Div(id='files-summary-table')
                ])
            ], className="mb-4"),
            
            # Metrics table
            dbc.Row([
                dbc.Col([
                    html.H4("Workout Metrics", className="mb-3"),
                    html.Div(id='workout-metrics-table')
                ])
            ], className="mb-4"),
            
            # Scatter plots
            dbc.Row([
                dbc.Col([
                    html.H4("Performance Scatter Plots", className="mb-3"),
                    dcc.Graph(id='performance-scatter-tau-up'),
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='performance-scatter-tau-down')
                ], width=6)
            ])
        ], fluid=True)
    ])


@callback(
    [Output('files-summary-table', 'children'),
     Output('workout-metrics-table', 'children'),
     Output('performance-scatter-tau-up', 'figure'),
     Output('performance-scatter-tau-down', 'figure')],
    [Input('files-summary-table', 'id')]  # Dummy input to trigger on load
)
def update_performance_content(dummy_id: str) -> tuple:
    """Update all performance analysis content.
    
    Args:
        dummy_id: Dummy input (not used)
        
    Returns:
        Tuple of all page components
    """
    # Get all FIT files
    fit_files = get_fit_files()
    
    if not fit_files:
        alert = dbc.Alert("No .fit files found in the data directory.", color="danger")
        empty_fig = go.Figure().add_annotation(text="No data available")
        return alert, None, empty_fig, empty_fig
    
    # Process all files
    all_file_data = []
    all_interval_data = []
    
    for i, filepath in enumerate(fit_files):
        # Load file
        df = load_fit_file(filepath)
        if df is None or df.empty:
            continue
        
        # Get basic file info
        file_info = {
            'File': os.path.basename(filepath),
            'Session': i + 1,
            'Records': len(df),
            'Duration': str(df.index[-1] - df.index[0]).split('.')[0] if len(df) > 0 else 'N/A',
            'Distance': f"{df['distance'].iloc[-1] - df['distance'].iloc[0]:.2f} m" 
                       if 'distance' in df.columns and len(df['distance'].dropna()) > 1 else 'N/A'
        }
        
        # Try to load cached interval data
        cached = load_cached_results(filepath, 'intervals')
        if cached:
            intervals = cached.get('intervals', [])
        else:
            # Run detection algorithm
            results = detect_hiit_intervals(df)
            intervals = results.get('intervals', [])
        
        file_info['Total HIIT Intervals'] = len(intervals)
        all_file_data.append(file_info)
        
        # Collect interval data for scatter plots
        for j, interval in enumerate(intervals):
            interval_data = {
                'session': i + 1,
                'interval_num': j + 1,
                'interval_letter': chr(65 + j),  # A, B, C, etc.
                'label': f"{i+1}.{chr(65+j)}",
                'tau_up': interval.get('tau_up', 0),
                'tau_down': interval.get('tau_down', 0),
                'duty_cycle': interval.get('duty_cycle', 0),
                'median_top_speed': interval.get('median_top_speed', 0),
                'filename': os.path.basename(filepath)
            }
            all_interval_data.append(interval_data)
    
    # Create tables
    files_table = create_files_summary_table(all_file_data)
    metrics_table = create_metrics_table(all_file_data)
    
    # Create scatter plots
    scatter_tau_up = create_scatter_plot(all_interval_data, 'tau_up')
    scatter_tau_down = create_scatter_plot(all_interval_data, 'tau_down')
    
    return files_table, metrics_table, scatter_tau_up, scatter_tau_down


def create_files_summary_table(file_data: List[Dict[str, Any]]) -> dash_table.DataTable:
    """Create summary table of all files.
    
    Args:
        file_data: List of file information dictionaries
        
    Returns:
        Dash DataTable
    """
    if not file_data:
        return html.Div("No files to display")
    
    return dash_table.DataTable(
        data=file_data,
        columns=[{'name': col, 'id': col} for col in file_data[0].keys()],
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


def create_metrics_table(file_data: List[Dict[str, Any]]) -> dash_table.DataTable:
    """Create metrics table.
    
    Args:
        file_data: List of file information dictionaries
        
    Returns:
        Dash DataTable
    """
    if not file_data:
        return html.Div("No metrics to display")
    
    # Extract metrics
    metrics_data = []
    for data in file_data:
        metrics_data.append({
            'File': data['File'],
            'Total Records': data['Records'],
            'Workout Time': data['Duration'],
            'Total Distance': data['Distance'],
            'Total HIIT Intervals': data['Total HIIT Intervals']
        })
    
    return dash_table.DataTable(
        data=metrics_data,
        columns=[{'name': col, 'id': col} for col in metrics_data[0].keys()],
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


def create_scatter_plot(interval_data: List[Dict[str, Any]], tau_type: str) -> go.Figure:
    """Create scatter plot for performance analysis.
    
    Args:
        interval_data: List of interval data dictionaries
        tau_type: Either 'tau_up' or 'tau_down'
        
    Returns:
        Plotly figure
    """
    if not interval_data:
        return go.Figure().add_annotation(text="No interval data available")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(interval_data)
    
    # Create color sequence based on session
    sessions = df['session'].unique()
    colors = px.colors.qualitative.Plotly[:len(sessions)]
    color_map = {session: colors[i] for i, session in enumerate(sessions)}
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each session
    for session in sessions:
        session_df = df[df['session'] == session]
        
        # Calculate opacity based on interval number (earlier = more transparent)
        max_interval = session_df['interval_num'].max()
        
        for _, row in session_df.iterrows():
            opacity = 0.3 + (row['interval_num'] / max_interval) * 0.7
            
            fig.add_trace(
                go.Scatter(
                    x=[row[tau_type]],
                    y=[row['median_top_speed']],
                    mode='markers+text',
                    name=f"Session {session}",
                    text=[row['label']],
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(
                        size=row['duty_cycle'] * 50,  # Scale marker size by duty cycle
                        color=color_map[session],
                        opacity=opacity,
                        line=dict(width=1, color='black')
                    ),
                    showlegend=row['interval_num'] == 1,  # Only show legend for first interval
                    hovertemplate=(
                        f"Session: {session}<br>"
                        f"Interval: {row['label']}<br>"
                        f"Tau: %{{x:.1f}}s<br>"
                        f"Speed: %{{y:.2f}}m/s<br>"
                        f"Duty Cycle: {row['duty_cycle']:.2f}<br>"
                        f"File: {row['filename']}<extra></extra>"
                    )
                )
            )
    
    # Update layout
    title = "Tau Up vs Speed" if tau_type == 'tau_up' else "Tau Down vs Speed"
    fig.update_layout(
        title=title,
        xaxis_title=f"{title.split(' ')[0]} {title.split(' ')[1]} (s)",
        yaxis_title="Median Top Quartile Speed (m/s)",
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title="Sessions",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add annotation about marker size
    fig.add_annotation(
        text="Marker size represents duty cycle",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    return fig