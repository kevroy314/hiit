"""Raw data analysis page."""

import os
from typing import Optional, List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc

from ..config import Config
from ..data_loader import load_fit_file, get_fit_files


def create_layout(selected_file: Optional[str] = None) -> html.Div:
    """Create the raw data page layout.
    
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
                    html.H2("📊 Raw Data Analysis", className="mb-3"),
                ])
            ]),
            
            # File selector
            dbc.Row([
                dbc.Col([
                    html.Label("Select FIT file:"),
                    dcc.Dropdown(
                        id='raw-file-selector',
                        options=[
                            {'label': os.path.basename(f), 'value': f}
                            for f in fit_files
                        ],
                        value=selected_file,
                        clearable=False
                    )
                ], width=6)
            ], className="mb-4"),
            
            # Metrics row
            dbc.Row(id='raw-metrics-row', className="mb-4"),
            
            # Data summary table
            dbc.Row([
                dbc.Col([
                    html.H4("Data Summary", className="mb-3"),
                    html.Div(id='raw-data-summary-table')
                ])
            ], className="mb-4"),
            
            # Time series plots
            dbc.Row([
                dbc.Col([
                    html.H4("Time Series Data", className="mb-3"),
                    html.Div(id='raw-time-series-plots')
                ])
            ])
        ], fluid=True)
    ])


@callback(
    [Output('raw-metrics-row', 'children'),
     Output('raw-data-summary-table', 'children'),
     Output('raw-time-series-plots', 'children')],
    [Input('raw-file-selector', 'value')]
)
def update_raw_data_content(filepath: str) -> tuple:
    """Update raw data page content when file is selected.
    
    Args:
        filepath: Path to selected FIT file
        
    Returns:
        Tuple of (metrics, summary_table, plots)
    """
    if not filepath:
        return [], None, None
    
    # Load data
    df = load_fit_file(filepath)
    if df is None or df.empty:
        alert = dbc.Alert("Failed to load data from the selected file.", color="danger")
        return [alert], None, None
    
    # Create metrics
    metrics = create_metrics(df)
    
    # Create summary table
    summary_table = create_summary_table(df)
    
    # Create time series plots
    plots = create_time_series_plots(df)
    
    return metrics, summary_table, plots


def create_metrics(df: pd.DataFrame) -> List[dbc.Col]:
    """Create metric cards for the data.
    
    Args:
        df: DataFrame with FIT file data
        
    Returns:
        List of column components containing metric cards
    """
    metrics = []
    
    # Total records
    metrics.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(df):,}", className="text-center"),
                    html.P("Total Records", className="text-center text-muted")
                ])
            ])
        ], width=4)
    )
    
    # Total workout time
    if len(df) > 0:
        duration = df.index[-1] - df.index[0]
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        metrics.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(duration_str, className="text-center"),
                        html.P("Total Workout Time", className="text-center text-muted")
                    ])
                ])
            ], width=4)
        )
    
    # Total distance
    if 'distance' in df.columns:
        distance_series = df['distance'].dropna()
        if len(distance_series) > 1:
            total_distance = distance_series.iloc[-1] - distance_series.iloc[0]
            metrics.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{total_distance:.2f} m", className="text-center"),
                            html.P("Total Distance", className="text-center text-muted")
                        ])
                    ])
                ], width=4)
            )
    
    return metrics


def create_summary_table(df: pd.DataFrame) -> dash_table.DataTable:
    """Create summary statistics table.
    
    Args:
        df: DataFrame with FIT file data
        
    Returns:
        Dash DataTable with summary statistics
    """
    summary_data = []
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            summary_data.append({
                'Variable': col,
                'Records': df[col].count(),
                'Missing': df[col].isna().sum(),
                'Mean': f"{df[col].mean():.2f}",
                'Std Dev': f"{df[col].std():.2f}",
                'Min': f"{df[col].min():.2f}",
                'Max': f"{df[col].max():.2f}"
            })
    
    return dash_table.DataTable(
        data=summary_data,
        columns=[{'name': col, 'id': col} for col in summary_data[0].keys()],
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


def create_time_series_plots(df: pd.DataFrame) -> dcc.Graph:
    """Create time series plots for all variables.
    
    Args:
        df: DataFrame with FIT file data
        
    Returns:
        Dash Graph component with subplots
    """
    # Define plot groups
    plot_groups = [
        {
            'title': 'Speed',
            'fields': ['enhanced_speed', 'speed', 'vertical_speed'],
            'ylabel': 'Speed (m/s)'
        },
        {
            'title': 'Heart Rate',
            'fields': ['heart_rate'],
            'ylabel': 'Heart Rate (bpm)'
        },
        {
            'title': 'Altitude',
            'fields': ['enhanced_altitude', 'altitude'],
            'ylabel': 'Altitude (m)'
        },
        {
            'title': 'Position',
            'fields': ['distance', 'position_long', 'position_lat'],
            'ylabel': 'Value'
        },
        {
            'title': 'Temperature',
            'fields': ['temperature'],
            'ylabel': 'Temperature (°C)'
        }
    ]
    
    # Count available plot groups
    available_groups = []
    for group in plot_groups:
        if any(field in df.columns for field in group['fields']):
            available_groups.append(group)
    
    if not available_groups:
        return html.Div("No plottable data found.")
    
    # Create subplots
    fig = make_subplots(
        rows=len(available_groups), 
        cols=1,
        subplot_titles=[g['title'] for g in available_groups],
        vertical_spacing=0.1
    )
    
    # Add traces for each group
    for i, group in enumerate(available_groups, 1):
        for field in group['fields']:
            if field in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[field],
                        name=field,
                        line=dict(color=Config.get_color(field)),
                        mode='lines'
                    ),
                    row=i, col=1
                )
        
        # Update y-axis label
        fig.update_yaxes(title_text=group['ylabel'], row=i, col=1)
    
    # Update layout
    fig.update_layout(
        height=250 * len(available_groups),
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Time", row=len(available_groups), col=1)
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})