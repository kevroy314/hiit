FIELD_COLORS = {
    # Speeds - Yellow
    'enhanced_speed': '#FFFF44',
    'vertical_speed': '#FFFF44', 
    'speed': '#FFFF44',
    # Positions - Green
    'distance': '#44FF44',
    'position_lat': '#44FF44',
    'position_lon': '#44FF44',
    # Altitudes - Blue
    'altitude': '#4444FF',
    'enhanced_altitude': '#4444FF',
    # Temperature - White
    'temperature': '#FFFFFF',
    # Heart Rate - Red
    'heart_rate': '#FF4444',
}

def get_field_color(field_name):
    field_lower = field_name.lower()
    for key, color in FIELD_COLORS.items():
        if key in field_lower:
            return color
    return '#888888'

def get_field_group(field_name):
    """Return the group name for a field."""
    field_lower = field_name.lower()
    if any(speed in field_lower for speed in ['enhanced_speed', 'vertical_speed', 'speed']):
        return 'Speeds'
    elif any(pos in field_lower for pos in ['distance', 'position_lat', 'position_lon']):
        return 'Positions'
    elif any(alt in field_lower for alt in ['altitude', 'enhanced_altitude']):
        return 'Altitudes'
    elif 'temperature' in field_lower:
        return 'Temperature'
    elif 'heart_rate' in field_lower:
        return 'Heart Rate'
    else:
        return 'Other'
