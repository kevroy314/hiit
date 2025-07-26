"""Configuration module for HIIT Analyzer application."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for application settings."""
    
    # Application settings
    APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
    APP_PORT = int(os.getenv('APP_PORT', '8050'))
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Algorithm parameters
    LOWPASS_PERIOD_MIN = float(os.getenv('LOWPASS_PERIOD_MIN', '1'))
    LOWPASS_PERIOD_MAX = float(os.getenv('LOWPASS_PERIOD_MAX', '4'))
    SPEED_CLIP_THRESHOLD = float(os.getenv('SPEED_CLIP_THRESHOLD', '1.34'))
    
    MIN_INTERVAL_PERIOD = float(os.getenv('MIN_INTERVAL_PERIOD', '10'))
    MAX_INTERVAL_PERIOD = float(os.getenv('MAX_INTERVAL_PERIOD', '300'))
    
    INTERVAL_EDGE_THRESHOLD = float(os.getenv('INTERVAL_EDGE_THRESHOLD', '0.5'))
    INTERVAL_MIN_DURATION = float(os.getenv('INTERVAL_MIN_DURATION', '20'))
    INTERVAL_MAX_GAP = float(os.getenv('INTERVAL_MAX_GAP', '60'))
    
    FIT_MAX_ITERATIONS = int(os.getenv('FIT_MAX_ITERATIONS', '1000'))
    FIT_TOLERANCE = float(os.getenv('FIT_TOLERANCE', '1e-6'))
    
    # Color mappings
    COLORS: Dict[str, str] = {
        'enhanced_speed': os.getenv('COLOR_ENHANCED_SPEED', '#FFFF00'),
        'speed': os.getenv('COLOR_SPEED', '#FFFF00'),
        'vertical_speed': os.getenv('COLOR_VERTICAL_SPEED', '#FFFF00'),
        'heart_rate': os.getenv('COLOR_HEART_RATE', '#FF0000'),
        'enhanced_altitude': os.getenv('COLOR_ENHANCED_ALTITUDE', '#0000FF'),
        'altitude': os.getenv('COLOR_ALTITUDE', '#0000FF'),
        'distance': os.getenv('COLOR_DISTANCE', '#00FF00'),
        'position_long': os.getenv('COLOR_POSITION_LONG', '#00FF00'),
        'position_lat': os.getenv('COLOR_POSITION_LAT', '#00FF00'),
        'temperature': os.getenv('COLOR_TEMPERATURE', '#FFFFFF'),
        'frequency': os.getenv('COLOR_FREQUENCY', '#00BFFF'),
        'template_correlation': os.getenv('COLOR_TEMPLATE_CORRELATION', '#800080'),
        'combined_correlation': os.getenv('COLOR_COMBINED_CORRELATION', '#00FF00'),
    }
    
    @classmethod
    def get_color(cls, field_name: str) -> str:
        """Get color for a given field name.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Hex color code
        """
        return cls.COLORS.get(field_name.lower(), '#888888')