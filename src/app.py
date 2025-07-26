"""Main HIIT Analyzer Dash application."""

import os
from urllib.parse import urlparse, parse_qs
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

from .config import Config
from .pages import raw_data, interval_analysis, performance_analysis


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="HIIT Analyzer"
)

# Define navigation bar
def create_navbar() -> dbc.NavbarSimple:
    """Create the navigation bar.
    
    Returns:
        Dash Bootstrap NavbarSimple component
    """
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Raw Data", href="/raw", id="raw-link")),
            dbc.NavItem(dbc.NavLink("HIIT Interval Analysis", href="/interval", id="interval-link")),
            dbc.NavItem(dbc.NavLink("Performance Analysis", href="/performance", id="performance-link")),
        ],
        brand="HIIT Analyzer",
        brand_href="/",
        color="primary",
        dark=True,
        sticky="top"
    )


# Define app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_navbar(),
    html.Div(id='page-content', style={'padding': '20px'})
])


@callback(
    Output('page-content', 'children'),
    [Input('url', 'href')]
)
def display_page(href: str) -> html.Div:
    """Display the appropriate page based on URL.
    
    Args:
        href: Full URL including query parameters
        
    Returns:
        Page content
    """
    if not href:
        return raw_data.create_layout()
    
    # Parse URL
    parsed = urlparse(href)
    pathname = parsed.path
    query_params = parse_qs(parsed.query)
    
    # Get selected file from query params
    selected_file = query_params.get('file', [None])[0]
    
    # Route to appropriate page
    if pathname == '/interval':
        return interval_analysis.create_layout(selected_file)
    elif pathname == '/performance':
        return performance_analysis.create_layout()
    else:  # Default to raw data page
        return raw_data.create_layout(selected_file)


@callback(
    Output('url', 'href'),
    [Input('raw-file-selector', 'value'),
     Input('interval-file-selector', 'value')],
    [State('url', 'href')],
    prevent_initial_call=True
)
def update_url(raw_file: str, interval_file: str, current_href: str) -> str:
    """Update URL when file selection changes.
    
    Args:
        raw_file: Selected file in raw data page
        interval_file: Selected file in interval analysis page
        current_href: Current URL
        
    Returns:
        Updated URL
    """
    # Determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Parse current URL
    parsed = urlparse(current_href)
    pathname = parsed.path
    
    # Determine new file value
    if trigger_id == 'raw-file-selector' and raw_file:
        selected_file = os.path.basename(raw_file)
    elif trigger_id == 'interval-file-selector' and interval_file:
        selected_file = os.path.basename(interval_file)
    else:
        return dash.no_update
    
    # Build new URL
    new_href = f"{pathname}?file={selected_file}"
    return new_href


@callback(
    [Output('raw-link', 'active'),
     Output('interval-link', 'active'),
     Output('performance-link', 'active')],
    [Input('url', 'pathname')]
)
def update_nav_active(pathname: str) -> tuple:
    """Update active state of navigation links.
    
    Args:
        pathname: Current pathname
        
    Returns:
        Tuple of boolean values for each nav link
    """
    if pathname == '/interval':
        return False, True, False
    elif pathname == '/performance':
        return False, False, True
    else:
        return True, False, False


def main():
    """Run the application."""
    app.run_server(
        host=Config.APP_HOST,
        port=Config.APP_PORT,
        debug=Config.DEBUG
    )


if __name__ == '__main__':
    main()