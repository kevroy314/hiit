import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hiit.utils import get_field_color
from hiit.data_io import load_manual_window_settings
from hiit.metrics import calculate_interval_metrics, analyze_interval_performance, fit_hr_curve
import pandas as pd
import altair as alt
from bokeh.plotting import figure
from bokeh.models import RangeTool, LinearAxis, Range1d, ColumnDataSource, BoxAnnotation, CustomJS
from bokeh.layouts import column

def plot_interval_analysis(df, intervals, filename):
    """Create interval analysis plots."""
    if not intervals:
        st.warning("No intervals detected. Try adjusting the parameters.")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Speed', 'Heart Rate', 'Altitude', 'Temperature'],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Plot speed
    if 'enhanced_speed' in df.columns:
        speed_data = df['enhanced_speed'].dropna()
        fig.add_trace(
            go.Scatter(
                x=speed_data.index,
                y=speed_data.values,
                mode='lines',
                name='Speed',
                line=dict(color='#FF8800', width=2)
            ),
            row=1, col=1
        )
    
    # Plot heart rate
    if 'heart_rate' in df.columns:
        hr_data = df['heart_rate'].dropna()
        fig.add_trace(
            go.Scatter(
                x=hr_data.index,
                y=hr_data.values,
                mode='lines',
                name='Heart Rate',
                line=dict(color='#FF4444', width=2)
            ),
            row=2, col=1
        )
    
    # Plot altitude
    if 'altitude' in df.columns:
        alt_data = df['altitude'].dropna()
        fig.add_trace(
            go.Scatter(
                x=alt_data.index,
                y=alt_data.values,
                mode='lines',
                name='Altitude',
                line=dict(color='#8844FF', width=2)
            ),
            row=3, col=1
        )
    
    # Plot temperature
    if 'temperature' in df.columns:
        temp_data = df['temperature'].dropna()
        fig.add_trace(
            go.Scatter(
                x=temp_data.index,
                y=temp_data.values,
                mode='lines',
                name='Temperature',
                line=dict(color='#FFFFFF', width=2)
            ),
            row=4, col=1
        )
    
    # Add interval overlays
    colors = px.colors.qualitative.Set3
    for i, interval in enumerate(intervals):
        color = colors[i % len(colors)]
        
        # High intensity period
        high_start = df.index[interval['high_start']]
        high_end = df.index[min(interval['high_end'], len(df)-1)]
        
        # Recovery period
        recovery_start = df.index[interval['recovery_start']]
        recovery_end = df.index[min(interval['recovery_end'], len(df)-1)]
        
        # Add rectangles for intervals
        for row in range(1, 5):
            fig.add_vrect(
                x0=high_start, x1=high_end,
                fillcolor=color, opacity=0.3,
                layer="below", line_width=0,
                row=row, col=1
            )
            fig.add_vrect(
                x0=recovery_start, x1=recovery_end,
                fillcolor=color, opacity=0.1,
                layer="below", line_width=0,
                row=row, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"HIIT Interval Analysis - {os.path.basename(filename)}",
        template="plotly_dark",
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)
    fig.update_yaxes(title_text="Altitude (m)", row=3, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=4, col=1)
    
    return fig

def create_metrics_visualization(intervals_df):
    """Create PCA visualization of interval metrics."""
    if len(intervals_df) < 2:
        st.warning("Need at least 2 intervals for PCA analysis.")
        return None
    
    # Select features for PCA
    feature_columns = [
        'high_duration', 'recovery_duration',
        'high_speed_mean', 'high_speed_std',
        'recovery_speed_mean', 'recovery_speed_std',
        'high_hr_mean', 'recovery_hr_mean',
        'temperature_mean', 'dA_up', 'dA_down'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in intervals_df.columns]
    
    if len(available_features) < 2:
        st.warning("Not enough features available for PCA analysis.")
        return None
    
    # Prepare data
    X = intervals_df[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers+text',
            marker=dict(
                size=15,
                color=intervals_df['interval_num'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Interval #")
            ),
            text=intervals_df['interval_num'],
            textposition="middle center",
            textfont=dict(color="white", size=12),
            hovertemplate='<b>Interval %{text}</b><br>' +
                         '<br>'.join([f'{col}: %{{customdata[{i}]:.2f}}' 
                                     for i, col in enumerate(available_features)]) +
                         '<extra></extra>',
            customdata=X.values
        )
    )
    
    # Add arrows showing progression
    for i in range(len(pca_result) - 1):
        fig.add_trace(
            go.Scatter(
                x=[pca_result[i, 0], pca_result[i+1, 0]],
                y=[pca_result[i, 1], pca_result[i+1, 1]],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.5)', width=2),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    fig.update_layout(
        title="HIIT Performance Metrics - PCA Visualization",
        template="plotly_dark",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        height=600
    )
    
    return fig

def plot_hiit_detection(df, hiit_start, hiit_end, intervals, frequency_info=None, filename=None):
    """Primary detection figure showing speed, heart rate, intervals, and HIIT window."""
    fig = go.Figure()

    # Main data plot - separate y-axes for speed and heart rate
    if 'enhanced_speed' in df.columns:
        # Apply aggressive median filtering to raw speed data for better edge preservation in square wave data
        from scipy.signal import medfilt
        speed_median_filtered = medfilt(df['enhanced_speed'].values, kernel_size=45)  # Much more aggressive
        speed_median_series = pd.Series(speed_median_filtered, index=df.index)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=speed_median_series,
            mode='lines',
            name='Speed (Raw)',
            line=dict(color=get_field_color('enhanced_speed'), width=1),
            opacity=0.7,
            yaxis='y'
        ))
        window_size = 5
        speed_smoothed = speed_median_series.rolling(window=window_size, center=True).mean().ffill().bfill()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=speed_smoothed,
            mode='lines',
            name='Speed (Smoothed)',
            line=dict(color=get_field_color('enhanced_speed'), width=2),
            yaxis='y'
        ))
    
    if 'heart_rate' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['heart_rate'],
            mode='lines',
            name='Heart Rate (Raw)',
            line=dict(color=get_field_color('heart_rate'), width=1),
            opacity=0.7,
            yaxis='y2'
        ))
        if frequency_info and 'hr_filtered' in frequency_info:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=frequency_info['hr_filtered'],
                mode='lines',
                name='Heart Rate (Filtered)',
                line=dict(color=get_field_color('heart_rate'), width=2),
                yaxis='y2'
            ))
    
    # HIIT window and transitions - use refined boundaries if available
    display_start = hiit_start
    display_end = hiit_end
    
    print(f"DEBUG: Plotting - Original boundaries: {hiit_start}-{hiit_end}")
    
    # Use refined boundaries if available in frequency_info
    if frequency_info and 'refined_start' in frequency_info and 'refined_end' in frequency_info:
        if frequency_info['refined_start'] is not None and frequency_info['refined_end'] is not None:
            display_start = frequency_info['refined_start']
            display_end = frequency_info['refined_end']
            print(f"DEBUG: Plotting - Using refined boundaries: {display_start}-{display_end}")
        else:
            print(f"DEBUG: Plotting - Refined boundaries are None")
    else:
        print(f"DEBUG: Plotting - No refined boundaries in frequency_info")
    
    print(f"DEBUG: Plotting - Final display boundaries: {display_start}-{display_end}")
    
    if display_start is not None and display_end is not None:
        hiit_x0 = df.index[display_start]
        hiit_x1 = df.index[display_end-1]
        fig.add_vrect(x0=hiit_x0, x1=hiit_x1, fillcolor='#44FF44', opacity=0.15, line_width=0, 
                     annotation_text='HIIT Window (Refined)', annotation_position='top left')
        if frequency_info and 'regions_above_threshold' in frequency_info:
            regions = frequency_info['regions_above_threshold']
            if regions:
                threshold_start = regions[0][0]
                threshold_end = regions[-1][1]
                if threshold_start != display_start or threshold_end != display_end:
                    threshold_x0 = df.index[threshold_start]
                    threshold_x1 = df.index[threshold_end-1]
                    fig.add_vrect(x0=threshold_x0, x1=threshold_x1, fillcolor='#FFFF44', opacity=0.1, line_width=0, 
                                 annotation_text='Threshold Window', annotation_position='bottom left')
    
    # Add interval overlays
    for interval in intervals:
        if 'interval_start' in interval and interval['interval_start'] < len(df):
            boundary_time = df.index[interval['interval_start']]
            fig.add_vline(x=boundary_time, line_dash='dot', line_color='yellow', line_width=2)
        high_x0 = df.index[interval['high_start']] if interval['high_start'] < len(df) else df.index[-1]
        high_x1 = df.index[interval['high_end']-1] if interval['high_end']-1 < len(df) else df.index[-1]
        rec_x0 = df.index[interval['recovery_start']] if interval['recovery_start'] < len(df) else df.index[-1]
        rec_x1 = df.index[interval['recovery_end']-1] if interval['recovery_end']-1 < len(df) else df.index[-1]
        fig.add_vrect(x0=high_x0, x1=high_x1, fillcolor='#FF8800', opacity=0.2, line_width=0)
        fig.add_vrect(x0=rec_x0, x1=rec_x1, fillcolor='#FF4444', opacity=0.1, line_width=0)
    
    # Add dotted vertical lines for HIIT window start/end (without annotations to avoid timestamp issues)
    if display_start is not None and display_end is not None:
        hiit_start_time = df.index[display_start]
        hiit_end_time = df.index[display_end-1]
        fig.add_vline(x=hiit_start_time, line_dash='dot', line_color='#44FF44', line_width=2)
        fig.add_vline(x=hiit_end_time, line_dash='dot', line_color='#44FF44', line_width=2)
    
    # Update layout with dual y-axes
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        title=f"HIIT Detection - {os.path.basename(filename) if filename else 'Data'}"
    )
    
    # Configure dual y-axes
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text="Speed (m/s)",
                font=dict(color=get_field_color('enhanced_speed'))
            ),
            tickfont=dict(color=get_field_color('enhanced_speed')),
            side='left'
        ),
        yaxis2=dict(
            title=dict(
                text="Heart Rate (bpm)",
                font=dict(color=get_field_color('heart_rate'))
            ),
            tickfont=dict(color=get_field_color('heart_rate')),
            side='right',
            overlaying='y',
            anchor='x'
        ),
        xaxis=dict(title="Time")
    )
    
    return fig

def plot_frequency_correlation(df, frequency_info):
    """Frequency correlation analysis figure."""
    if not frequency_info or 'frequency_correlation' not in frequency_info:
        return None
    
    fig = go.Figure()
    
    # Frequency correlation and template correlation
    fig.add_trace(go.Scatter(
        x=df.index,
        y=frequency_info['frequency_correlation'],
        mode='lines',
        name='Frequency Correlation',
        line=dict(color='#44FFFF', width=2),
        yaxis='y'
    ))
    
    if 'template_correlation' in frequency_info and frequency_info['template_correlation'] is not None:
        template_corr = pd.Series(frequency_info['template_correlation']).rolling(window=11, center=True, min_periods=1).mean().values
        fig.add_trace(go.Scatter(
            x=df.index,
            y=template_corr,
            mode='lines',
            name='Template Correlation',
            line=dict(color='#FF00FF', width=2, dash='dot'),
            yaxis='y2'
        ))
    
    # Add combined correlation signal
    if 'combined_correlation' in frequency_info and frequency_info['combined_correlation'] is not None:
        combined_corr = pd.Series(frequency_info['combined_correlation']).rolling(window=11, center=True, min_periods=1).mean().values
        fig.add_trace(go.Scatter(
            x=df.index,
            y=combined_corr,
            mode='lines',
            name='Combined Correlation',
            line=dict(color='#00FF00', width=3),  # Bright green, thicker line
            yaxis='y'
        ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        title="Frequency & Template Correlation Analysis"
    )
    
    # Configure dual y-axes
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text="Frequency Correlation",
                font=dict(color='#44FFFF')
            ),
            tickfont=dict(color='#44FFFF'),
            side='left'
        ),
        yaxis2=dict(
            title=dict(
                text="Template Correlation",
                font=dict(color='#FF00FF')
            ),
            tickfont=dict(color='#FF00FF'),
            side='right',
            overlaying='y',
            anchor='x'
        ),
        xaxis=dict(title="Time")
    )
    
    return fig

def plot_frequency_analysis(frequency_info):
    """Frequency spectrum analysis figure."""
    if not frequency_info or 'frequencies' not in frequency_info or 'hr_fft_magnitude' not in frequency_info:
        return None
    
    fig = go.Figure()
    
    freqs = frequency_info['frequencies']
    magnitude = frequency_info['hr_fft_magnitude']
    
    # Focus on relevant frequency range (0.001 to 0.1 Hz = 10s to 1000s periods)
    freq_mask = (freqs >= 0.001) & (freqs <= 0.1)
    
    fig.add_trace(go.Scatter(
        x=freqs[freq_mask],
        y=magnitude[freq_mask],
        mode='lines',
        name='Frequency Spectrum',
        line=dict(color='#44FFFF', width=2)
    ))
    
    # Mark dominant frequency
    if 'dominant_freq' in frequency_info:
        fig.add_vline(x=frequency_info['dominant_freq'], line_dash="dash", line_color="red", 
                    annotation_text=f"Dominant: {frequency_info['dominant_period']:.1f}s")
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False,
        title="Frequency Spectrum Analysis",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude"
    )
    
    return fig

def plot_period_power_spectrum(frequency_info):
    """Period power spectrum analysis figure."""
    if not frequency_info or 'filtered_periods' not in frequency_info or 'filtered_power' not in frequency_info:
        return None
    
    fig = go.Figure()
    
    periods = frequency_info['filtered_periods']
    power = frequency_info['filtered_power']
    
    fig.add_trace(go.Scatter(
        x=periods,
        y=power,
        mode='lines+markers',
        name='Period Power',
        line=dict(color='#FF44FF', width=2),
        marker=dict(size=6)
    ))
    
    # Mark dominant period
    if 'dominant_period' in frequency_info:
        fig.add_vline(x=frequency_info['dominant_period'], line_dash="dash", line_color="red", 
                    annotation_text="Dominant Period")
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False,
        title="Period Power Spectrum",
        xaxis_title="Period (seconds)",
        yaxis_title="Power"
    )
    
    return fig

def create_interval_overlay_figure(df, intervals, frequency_info=None):
    """Create the interval overlay figure with exponential fits."""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Interval Overlay with Exponential Fits'],
        specs=[[{"secondary_y": True}]]
    )
    
    if not intervals:
        # Add a message when no intervals
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No intervals detected",
            showarrow=False,
            font=dict(size=16, color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1
        )
    else:
        # Create overlay plot with all intervals aligned
        for i, interval in enumerate(intervals):
            if 'interval_start' in interval and 'interval_end' in interval:
                interval_start = interval['interval_start']
                interval_end = interval['interval_end']
                if interval_start < len(df) and interval_end <= len(df):
                    # Extract interval data
                    interval_data = df.iloc[interval_start:interval_end]
                    time_axis = np.arange(len(interval_data))
                    opacity = 0.3 + (i / len(intervals)) * 0.7  # 0.3 to 1.0
                    if 'enhanced_speed' in interval_data.columns:
                        speed_data = interval_data['enhanced_speed'].ffill().bfill()
                        fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=speed_data,
                            mode='lines',
                            name=f'Interval {interval["interval_num"]} Speed',
                            line=dict(color=get_field_color('enhanced_speed'), width=2),
                            opacity=opacity,
                            showlegend=False
                        ), row=1, col=1, secondary_y=False)
                    if 'heart_rate' in interval_data.columns:
                        hr_data = interval_data['heart_rate'].ffill().bfill()
                        fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=hr_data,
                            mode='lines',
                            name=f'Interval {interval["interval_num"]} HR',
                            line=dict(color=get_field_color('heart_rate'), width=2),
                            opacity=opacity,
                            showlegend=False
                        ), row=1, col=1, secondary_y=True)
                        if len(hr_data) >= 10:
                            try:
                                metrics = analyze_interval_performance(df, interval)
                                if metrics:
                                    mid_point = len(hr_data) // 2
                                    rise_data = hr_data.iloc[:mid_point]
                                    fall_data = hr_data.iloc[mid_point:]
                                    rise_time = np.arange(len(rise_data))
                                    fall_time = np.arange(len(fall_data))
                                    rise_params, rise_func = fit_hr_curve(rise_time, rise_data.values, "exp_rise")
                                    fall_params, fall_func = fit_hr_curve(fall_time, fall_data.values, "exp_fall")
                                    rise_fit = rise_func(rise_time, *rise_params)
                                    fall_fit = fall_func(fall_time, *fall_params)
                                    fig.add_trace(go.Scatter(
                                        x=rise_time,
                                        y=rise_fit,
                                        mode='lines',
                                        name=f'Interval {interval["interval_num"]} Rise Fit',
                                        line=dict(color='#FF00FF', width=3, dash='dash'),
                                        opacity=opacity,
                                        showlegend=False
                                    ), row=1, col=1, secondary_y=True)
                                    fig.add_trace(go.Scatter(
                                        x=fall_time + mid_point,
                                        y=fall_fit,
                                        mode='lines',
                                        name=f'Interval {interval["interval_num"]} Fall Fit',
                                        line=dict(color='#00FFFF', width=3, dash='dash'),
                                        opacity=opacity,
                                        showlegend=False
                                    ), row=1, col=1, secondary_y=True)
                            except Exception as e:
                                pass
        
        # Add mean templates if available
        if frequency_info and 'mean_hr_template' in frequency_info and frequency_info['mean_hr_template'] is not None:
            mean_hr_template = frequency_info['mean_hr_template']
            template_time = np.arange(len(mean_hr_template))
            
            # Add mean HR template as a thick line
            fig.add_trace(go.Scatter(
                x=template_time,
                y=mean_hr_template,
                mode='lines',
                name='Mean HR Template',
                line=dict(color='#FFFFFF', width=6),  # Thick white line
                showlegend=True
            ), row=1, col=1, secondary_y=True)
        
        # Add mean speed template if available
        if frequency_info and 'mean_speed_template' in frequency_info and frequency_info['mean_speed_template'] is not None:
            mean_speed_template = frequency_info['mean_speed_template']
            template_time = np.arange(len(mean_speed_template))
            
            # Add mean speed template as a thick line
            fig.add_trace(go.Scatter(
                x=template_time,
                y=mean_speed_template,
                mode='lines',
                name='Mean Speed Template',
                line=dict(color='#FFFF00', width=6),  # Thick yellow line
                showlegend=True
            ), row=1, col=1, secondary_y=False)
    
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", color=get_field_color('enhanced_speed'), row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", color=get_field_color('heart_rate'), secondary_y=True, row=1, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    return fig

def create_speed_stats_figure(df, intervals):
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Speed Statistics by Interval'],
        specs=[[{"secondary_y": False}]]
    )
    if not intervals:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No intervals detected",
            showarrow=False,
            font=dict(size=16, color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1
        )
    else:
        for i, interval in enumerate(intervals):
            if 'high_start' in interval and 'high_end' in interval:
                high_start = interval['high_start']
                high_end = interval['high_end']
                if high_start < len(df) and high_end <= len(df):
                    high_period = df.iloc[high_start:high_end]
                    if 'enhanced_speed' in high_period.columns:
                        speed_data = high_period['enhanced_speed'].ffill().bfill()
                        if len(speed_data) > 5:
                            speed_2d = speed_data.values.reshape(-1, 1)
                            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                            cluster_labels = kmeans.fit_predict(speed_2d)
                            cluster_means = [speed_data[cluster_labels == i].mean() for i in range(2)]
                            high_speed_cluster = np.argmax(cluster_means)
                            high_speed_data = speed_data[cluster_labels == high_speed_cluster]
                            speed_mean = high_speed_data.mean()
                            speed_std = high_speed_data.std()
                            speed_variability = speed_std / speed_mean if speed_mean > 0 else 0
                            opacity = 0.3 + (i / len(intervals)) * 0.7
                            fig.add_trace(go.Scatter(
                                x=[interval['interval_num']],
                                y=[speed_mean],
                                mode='markers',
                                name=f'Interval {interval["interval_num"]} Mean',
                                marker=dict(
                                    color=get_field_color('enhanced_speed'),
                                    size=12,
                                    opacity=opacity
                                ),
                                error_y=dict(
                                    type='data',
                                    array=[speed_std],
                                    visible=True,
                                    color=get_field_color('enhanced_speed')
                                ),
                                showlegend=False
                            ), row=1, col=1)
                            fig.add_annotation(
                                x=interval['interval_num'],
                                y=speed_mean + speed_std + 0.1,
                                text=f"CV: {speed_variability:.3f}",
                                showarrow=False,
                                font=dict(size=8, color='white'),
                                bgcolor='rgba(0,0,0,0.7)',
                                bordercolor='white',
                                borderwidth=1,
                                row=1, col=1
                            )
    fig.update_xaxes(title_text="Interval Number", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    return fig

def create_metrics_distributions_figure(df, intervals):
    all_metrics = []
    for interval in intervals:
        if 'high_start' in interval and 'high_end' in interval:
            high_start = interval['high_start']
            high_end = interval['high_end']
            if high_start < len(df) and high_end <= len(df):
                high_period = df.iloc[high_start:high_end]
                if 'enhanced_speed' in high_period.columns and 'heart_rate' in high_period.columns:
                    try:
                        metrics = analyze_interval_performance(df, interval)
                        if metrics:
                            all_metrics.append(metrics)
                    except:
                        pass
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Tau Rise Distribution', 'Tau Fall Distribution', 
                       'HR Baseline Distribution', 'HR Peak Distribution',
                       'Speed Mean Distribution', 'Speed Std Distribution'],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    if not all_metrics:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No interval metrics to display",
            showarrow=False,
            font=dict(size=16, color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1
        )
    else:
        tau_rise_values = [m['tau_rise'] for m in all_metrics if m['tau_rise'] > 0]
        if tau_rise_values:
            fig.add_trace(go.Histogram(
                x=tau_rise_values,
                name='Tau Rise',
                nbinsx=10,
                marker_color='#FF00FF',
                opacity=0.7
            ), row=1, col=1)
        tau_fall_values = [m['tau_fall'] for m in all_metrics if m['tau_fall'] > 0]
        if tau_fall_values:
            fig.add_trace(go.Histogram(
                x=tau_fall_values,
                name='Tau Fall',
                nbinsx=10,
                marker_color='#00FFFF',
                opacity=0.7
            ), row=1, col=2)
        hr_baseline_values = [m['hr_baseline'] for m in all_metrics if m['hr_baseline'] > 0]
        if hr_baseline_values:
            fig.add_trace(go.Histogram(
                x=hr_baseline_values,
                name='HR Baseline',
                nbinsx=10,
                marker_color='#FF4444',
                opacity=0.7
            ), row=2, col=1)
        hr_peak_values = [m['hr_peak'] for m in all_metrics if m['hr_peak'] > 0]
        if hr_peak_values:
            fig.add_trace(go.Histogram(
                x=hr_peak_values,
                name='HR Peak',
                nbinsx=10,
                marker_color='#FF8888',
                opacity=0.7
            ), row=2, col=2)
        speed_mean_values = [m['speed_mean'] for m in all_metrics if m['speed_mean'] > 0]
        if speed_mean_values:
            fig.add_trace(go.Histogram(
                x=speed_mean_values,
                name='Speed Mean',
                nbinsx=10,
                marker_color='#FFFF44',
                opacity=0.7
            ), row=3, col=1)
        speed_std_values = [m['speed_std'] for m in all_metrics if m['speed_std'] > 0]
        if speed_std_values:
            fig.add_trace(go.Histogram(
                x=speed_std_values,
                name='Speed Std',
                nbinsx=10,
                marker_color='#FFFF88',
                opacity=0.7
            ), row=3, col=2)
    fig.update_xaxes(title_text="Tau Rise (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Tau Fall (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="HR Baseline (bpm)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="HR Peak (bpm)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_xaxes(title_text="Speed Mean (m/s)", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Speed Std (m/s)", row=3, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=2)
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=False
    )
    return fig

def create_top_level_metrics_figure(df, intervals):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Responsiveness Distribution', 'Speed Variability Distribution'],
        horizontal_spacing=0.1
    )
    all_metrics = []
    for interval in intervals:
        if 'high_start' in interval and 'high_end' in interval:
            high_start = interval['high_start']
            high_end = interval['high_end']
            if high_start < len(df) and high_end <= len(df):
                high_period = df.iloc[high_start:high_end]
                if 'enhanced_speed' in high_period.columns and 'heart_rate' in high_period.columns:
                    try:
                        metrics = analyze_interval_performance(df, interval)
                        if metrics:
                            all_metrics.append(metrics)
                    except:
                        pass
    if not all_metrics:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No top-level metrics to display",
            showarrow=False,
            font=dict(size=16, color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1
        )
    else:
        responsiveness_values = [m['responsiveness'] for m in all_metrics if m['responsiveness'] > 0]
        if responsiveness_values:
            fig.add_trace(go.Histogram(
                x=responsiveness_values,
                name='Responsiveness',
                nbinsx=10,
                marker_color='#44FF44',
                opacity=0.7
            ), row=1, col=1)
        speed_variability_values = [m['speed_variability'] for m in all_metrics if m['speed_variability'] > 0]
        if speed_variability_values:
            fig.add_trace(go.Histogram(
                x=speed_variability_values,
                name='Speed Variability',
                nbinsx=10,
                marker_color='#FFFF44',
                opacity=0.7
            ), row=1, col=2)
        fig.update_xaxes(title_text="Responsiveness (bpm/s)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Speed Variability (CV)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    return fig

def plot_raw_data_overview(df, filename=None):
    """
    Create four separate full-width figures for speed, heart rate, altitude, and temperature.
    Returns a dict of figures.
    """
    figs = {}
    # Speed
    if 'enhanced_speed' in df.columns:
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(
            x=df.index,
            y=df['enhanced_speed'],
            mode='lines',
            name='Speed',
            line=dict(color='#FF8800', width=2)
        ))
        fig_speed.update_layout(
            title=f"Speed vs. Time{' - ' + os.path.basename(filename) if filename else ''}",
            template="plotly_dark",
            height=300,
            xaxis_title="Time",
            yaxis_title="Speed (m/s)",
            showlegend=False
        )
        figs['speed'] = fig_speed
    # Heart Rate
    if 'heart_rate' in df.columns:
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            x=df.index,
            y=df['heart_rate'],
            mode='lines',
            name='Heart Rate',
            line=dict(color='#FF4444', width=2)
        ))
        fig_hr.update_layout(
            title=f"Heart Rate vs. Time{' - ' + os.path.basename(filename) if filename else ''}",
            template="plotly_dark",
            height=300,
            xaxis_title="Time",
            yaxis_title="Heart Rate (bpm)",
            showlegend=False
        )
        figs['heart_rate'] = fig_hr
    # Altitude
    if 'altitude' in df.columns:
        fig_alt = go.Figure()
        fig_alt.add_trace(go.Scatter(
            x=df.index,
            y=df['altitude'],
            mode='lines',
            name='Altitude',
            line=dict(color='#8844FF', width=2)
        ))
        fig_alt.update_layout(
            title=f"Altitude vs. Time{' - ' + os.path.basename(filename) if filename else ''}",
            template="plotly_dark",
            height=300,
            xaxis_title="Time",
            yaxis_title="Altitude (m)",
            showlegend=False
        )
        figs['altitude'] = fig_alt
    # Temperature
    if 'temperature' in df.columns:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=df.index,
            y=df['temperature'],
            mode='lines',
            name='Temperature',
            line=dict(color='#FFFFFF', width=2)
        ))
        fig_temp.update_layout(
            title=f"Temperature vs. Time{' - ' + os.path.basename(filename) if filename else ''}",
            template="plotly_dark",
            height=300,
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            showlegend=False
        )
        figs['temperature'] = fig_temp
    return figs

# Alias for compatibility with UI imports
create_detection_figure = plot_hiit_detection

def plot_hiit_detection_bokeh(df, hiit_start, hiit_end, intervals, frequency_info=None, filename=None, user_selection=None):
    """
    Create a Bokeh figure for HIIT detection with dual y-axes and interactive window selection.
    Returns (bokeh_layout, selection_range)
    """
    # Prepare data
    time = df.index
    speed = df['enhanced_speed'] if 'enhanced_speed' in df.columns else None
    hr = df['heart_rate'] if 'heart_rate' in df.columns else None
    
    source = ColumnDataSource(data={
        'time': time,
        'speed': speed,
        'hr': hr
    })

    # Main plot
    p = figure(
        width=900, height=350, x_axis_type='datetime',
        title="HIIT Detection (Bokeh)",
        tools="xpan,xwheel_zoom,reset",
        active_drag="xpan"
    )
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Speed (m/s)'
    p.y_range = Range1d(start=speed.min() if speed is not None else 0, end=speed.max() if speed is not None else 1)

    # Speed line (left y-axis)
    if speed is not None:
        p.line('time', 'speed', source=source, color='orange', legend_label='Speed (m/s)', line_width=2)

    # Heart rate line (right y-axis)
    if hr is not None:
        p.extra_y_ranges = {"hr": Range1d(start=hr.min(), end=hr.max())}
        p.add_layout(LinearAxis(y_range_name="hr", axis_label="Heart Rate (bpm)"), 'right')
        p.line('time', 'hr', source=source, color='red', legend_label='Heart Rate (bpm)', line_width=2, y_range_name="hr")

    # HIIT window overlay (auto-detected)
    if hiit_start is not None and hiit_end is not None:
        hiit_box = BoxAnnotation(
            left=time[hiit_start], right=time[hiit_end-1],
            fill_alpha=0.1, fill_color='green', line_color='green', line_alpha=0.5
        )
        p.add_layout(hiit_box)

    # User selection overlay
    selection_range = None
    if user_selection is not None:
        sel_start, sel_end = user_selection
        selection_range = (time[sel_start], time[sel_end-1])
        user_box = BoxAnnotation(
            left=selection_range[0], right=selection_range[1],
            fill_alpha=0.2, fill_color='orange', line_color='orange', line_alpha=0.8
        )
        p.add_layout(user_box)

    # RangeTool for selection
    select = figure(
        width=900, height=80, y_range=p.y_range, x_axis_type="datetime",
        tools="", toolbar_location=None
    )
    select.line('time', 'speed', source=source, color='orange')
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "orange"
    range_tool.overlay.fill_alpha = 0.2
    select.add_tools(range_tool)
    select.ygrid.grid_line_color = None
    select.xaxis.axis_label = 'Time (select HIIT window below)'

    # Add JS callback to emit selection
    range_tool.js_on_change('x_range', CustomJS(code="""
        if (cb_obj.x_range) {
            const start = cb_obj.x_range.start;
            const end = cb_obj.x_range.end;
            document.dispatchEvent(new CustomEvent("HIIT_RANGE", {detail: {start: start, end: end}}));
        }
    """))

    layout = column(p, select)
    return layout, range_tool

def plot_hiit_detection_plotly_with_js(df, hiit_start, hiit_end, intervals, frequency_info=None, filename=None):
    """
    Create Plotly HIIT detection plot with selection bounds displayed in Streamlit UI.
    Uses streamlit-plotly-events to capture selection events.
    """
    # Create the main figure
    fig = go.Figure()
    
    # Add heart rate trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['heart_rate'],
        name='Heart Rate',
        yaxis='y',
        line=dict(color='red', width=2),
        hovertemplate='Time: %{x}<br>HR: %{y:.1f} bpm<extra></extra>'
    ))
    
    # Add speed trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['speed'],
        name='Speed',
        yaxis='y2',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{x}<br>Speed: %{y:.2f} m/s<extra></extra>'
    ))
    
    # Add HIIT boundaries
    if hiit_start is not None and hiit_end is not None:
        fig.add_vline(x=df.index[hiit_start], line_dash="dash", line_color="red", 
                      annotation_text="Auto Start", annotation_position="top right")
        fig.add_vline(x=df.index[hiit_end], line_dash="dash", line_color="red", 
                      annotation_text="Auto End", annotation_position="top right")
    
    # Add interval overlays
    if intervals:
        colors = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)', 
                 'rgba(255,255,0,0.2)', 'rgba(255,0,255,0.2)']
        
        for i, interval in enumerate(intervals):
            color = colors[i % len(colors)]
            start_idx = interval['high_start']
            end_idx = interval['high_end']
            
            if start_idx < len(df) and end_idx < len(df):
                fig.add_vrect(
                    x0=df.index[start_idx],
                    x1=df.index[end_idx],
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    annotation_text=f"Interval {i+1}",
                    annotation_position="top left"
                )
    
    # Layout with dual y-axes
    fig.update_layout(
        title=f"Primary HIIT Detection (Plotly) - {filename}",
        xaxis_title="Time",
        yaxis=dict(
            title="Heart Rate (bpm)",
            side="left",
            titlefont=dict(color="red"),
            tickfont=dict(color="red")
        ),
        yaxis2=dict(
            title="Speed (m/s)",
            side="right",
            overlaying="y",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        dragmode='select',
        selectdirection='h',
        height=600,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_hiit_detection_plotly_js_injection(df, hiit_start, hiit_end, intervals, frequency_info=None, filename=None):
    """
    Create Plotly HIIT detection plot with custom JavaScript injection
    that displays real-time selection bounds in a floating div.
    """
    import streamlit.components.v1 as components
    
    # Create the main figure
    fig = go.Figure()
    
    # Add heart rate trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['heart_rate'],
        name='Heart Rate',
        yaxis='y',
        line=dict(color='red', width=2),
        hovertemplate='Time: %{x}<br>HR: %{y:.1f} bpm<extra></extra>'
    ))
    
    # Add speed trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['speed'],
        name='Speed',
        yaxis='y2',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{x}<br>Speed: %{y:.2f} m/s<extra></extra>'
    ))
    
    # Add HIIT boundaries
    if hiit_start is not None and hiit_end is not None:
        fig.add_vline(x=df.index[hiit_start], line_dash="dash", line_color="red", 
                      annotation_text="Auto Start", annotation_position="top right")
        fig.add_vline(x=df.index[hiit_end], line_dash="dash", line_color="red", 
                      annotation_text="Auto End", annotation_position="top right")
    
    # Add interval overlays
    if intervals:
        colors = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)', 
                 'rgba(255,255,0,0.2)', 'rgba(255,0,255,0.2)']
        
        for i, interval in enumerate(intervals):
            color = colors[i % len(colors)]
            start_idx = interval['high_start']
            end_idx = interval['high_end']
            
            if start_idx < len(df) and end_idx < len(df):
                fig.add_vrect(
                    x0=df.index[start_idx],
                    x1=df.index[end_idx],
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    annotation_text=f"Interval {i+1}",
                    annotation_position="top left"
                )
    
    # Layout with dual y-axes
    fig.update_layout(
        title=f"Primary HIIT Detection (Plotly JS) - {filename}",
        xaxis_title="Time",
        yaxis=dict(
            title="Heart Rate (bpm)",
            side="left",
            titlefont=dict(color="red"),
            tickfont=dict(color="red")
        ),
        yaxis2=dict(
            title="Speed (m/s)",
            side="right",
            overlaying="y",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        dragmode='select',
        selectdirection='h',
        height=600,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Custom JavaScript for real-time selection display
    js_code = """
    <script>
    // Create a div to display selection bounds
    const boundsDiv = document.createElement('div');
    boundsDiv.id = 'selection-bounds';
    boundsDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        z-index: 10000;
        border: 2px solid #666;
        min-width: 250px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    `;
    boundsDiv.innerHTML = '<strong>🎯 Selection Bounds</strong><br>No selection';
    document.body.appendChild(boundsDiv);
    
    // Function to update bounds display
    function updateBoundsDisplay(selection) {
        if (selection && selection.range && selection.range.x) {
            const xRange = selection.range.x;
            const yRange = selection.range.y;
            
            const startTime = new Date(xRange[0]).toLocaleTimeString();
            const endTime = new Date(xRange[1]).toLocaleTimeString();
            const duration = ((xRange[1] - xRange[0]) / 1000).toFixed(1);
            
            let boundsText = '<strong>🎯 Selection Bounds</strong><br>';
            boundsText += `⏰ X: ${startTime} → ${endTime}<br>`;
            boundsText += `⏱️ Duration: ${duration}s<br>`;
            
            if (yRange) {
                boundsText += `📊 Y: ${yRange[0].toFixed(1)} → ${yRange[1].toFixed(1)}<br>`;
                boundsText += `📈 Range: ${(yRange[1] - yRange[0]).toFixed(1)}`;
            }
            
            boundsDiv.innerHTML = boundsText;
            boundsDiv.style.background = 'rgba(0, 255, 0, 0.9)';
            boundsDiv.style.borderColor = '#00ff00';
        } else {
            boundsDiv.innerHTML = '<strong>🎯 Selection Bounds</strong><br>No selection';
            boundsDiv.style.background = 'rgba(0, 0, 0, 0.9)';
            boundsDiv.style.borderColor = '#666';
        }
    }
    
    // Wait for Plotly to load and attach event listeners
    function attachPlotlyListeners() {
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv) {
            // Listen for selection events
            plotDiv.on('plotly_selected', function(eventData) {
                console.log('Selection event:', eventData);
                updateBoundsDisplay(eventData);
            });
            
            // Listen for deselection events
            plotDiv.on('plotly_deselect', function(eventData) {
                console.log('Deselection event:', eventData);
                updateBoundsDisplay(null);
            });
            
            console.log('✅ Plotly event listeners attached successfully');
        } else {
            console.log('⏳ Plotly div not found, retrying...');
            setTimeout(attachPlotlyListeners, 500);
        }
    }
    
    // Start listening for Plotly
    setTimeout(attachPlotlyListeners, 1000);
    </script>
    """
    
    # Create the HTML component with Plotly and custom JavaScript
    plotly_html = f"""
    <div id="plotly-container">
        {js_code}
    </div>
    <script>
        // Render the Plotly figure
        const plotData = {fig.to_json()};
        Plotly.newPlot('plotly-container', plotData.data, plotData.layout, {{
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        }});
    </script>
    """
    
    # Display using Streamlit components
    components.html(plotly_html, height=650)
    
    return fig
