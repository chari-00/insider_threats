"""
Live Cyber Threat Detection Dashboard with Real-time Simulation
Run: python app_live.py
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import threading
import time
from pathlib import Path
import logging

from src.config import DASHBOARD_PORT, COLOR_SCHEME
from src.live_simulator import LiveSimulator, create_simulation_data
from src.enhanced_adaptive_ml_detector import EnhancedAdaptiveMLDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# INITIALIZE SIMULATION SYSTEM
# ============================================

print("Initializing Live Cyber Threat Detection System...")

# Create simulation data if it doesn't exist
simulation_data_path = Path("data/synthetic")
if not simulation_data_path.exists() or not list(simulation_data_path.glob("*.csv")):
    print("Creating simulation data...")
    create_simulation_data(num_users=150)

# Initialize simulator and detector
simulator = LiveSimulator(num_users=150, simulation_speed=1.0)
detector = EnhancedAdaptiveMLDetector(enable_vae=True, enable_temporal=True)
detector.initialize_auto_encoders()

# Global state for dashboard
dashboard_state = {
    'is_simulating': False,
    'simulation_start_time': None,
    'total_events': 0,
    'blocked_users': 0,
    'high_risk_events': 0,
    'last_update': datetime.now()
}

# ============================================
# INITIALIZE DASH APP
# ============================================

app = dash.Dash(
    __name__,
    assets_folder='assets',
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

app.title = "Live Cyber Threat Detection Dashboard"

# ============================================
# LAYOUT
# ============================================

app.layout = dbc.Container([
    
    # Header Section
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🛡️ Live Cyber Threat Detection Dashboard", 
                       style={'color': 'white', 'fontSize': '2.2rem', 'fontWeight': '600', 'margin': '0'}),
                html.Div([
                    html.Div([
                        html.Span("Status: ", style={'fontWeight': '600', 'color': 'rgba(255,255,255,0.9)'}),
                        html.Span("Ready", id="status-indicator", 
                                style={'color': '#90EE90', 'fontWeight': '500'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                    html.Div([
                        html.Span("Last Update: ", style={'fontWeight': '600', 'color': 'rgba(255,255,255,0.9)'}),
                        html.Span(id="last-update", style={'color': 'white', 'fontWeight': '500'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                    html.Div([
                        html.Span("Total Users: ", style={'fontWeight': '600', 'color': 'rgba(255,255,255,0.9)'}),
                        html.Span("150", style={'color': 'white', 'fontWeight': '500'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'})
                ], style={'display': 'flex', 'gap': '30px', 'marginTop': '15px'})
            ], style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'padding': '20px 30px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.1)'
            })
        ], width=12)
    ], className="mb-4"),
    
    # Simulation Controls
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("🎮 Simulation Controls", style={'fontSize': '1.4rem', 'color': '#333', 'fontWeight': '600', 'margin': '0'})
                ], style={'padding': '20px 25px', 'background': '#f8f9fa', 'borderBottom': '1px solid #e9ecef'}),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="fas fa-play me-2"), "Start Simulation"],
                                id="start-simulation-btn",
                                color="success",
                                size="lg",
                                className="w-100"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="fas fa-stop me-2"), "Stop Simulation"],
                                id="stop-simulation-btn",
                                color="danger",
                                size="lg",
                                className="w-100"
                            )
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Label("Simulation Speed:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Slider(
                                    id="speed-slider",
                                    min=0.1,
                                    max=10.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: f"{i}x" for i in [0.1, 1, 2, 5, 10]},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label("Duration (minutes):", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Input(
                                    id="duration-input",
                                    type="number",
                                    value=60,
                                    min=1,
                                    max=480,
                                    style={'width': '100%'}
                                )
                            ])
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Label("Risk Block Threshold:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Input(
                                    id="block-threshold-input",
                                    type="number",
                                    value=80,
                                    min=50,
                                    max=100,
                                    style={'width': '100%'}
                                )
                            ])
                        ], width=3),
                        dbc.Col([
                            html.Div(id="simulation-stats", style={'padding': '10px', 'background': '#f8f9fa', 'borderRadius': '6px'})
                        ], width=6)
                    ])
                ], style={'padding': '25px'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 12px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], width=12)
    ]),
    
    # Real-time Metrics
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("📊 Real-time Metrics", style={'fontSize': '1.4rem', 'color': '#333', 'fontWeight': '600', 'margin': '0'})
                ], style={'padding': '20px 25px', 'background': '#f8f9fa', 'borderBottom': '1px solid #e9ecef'}),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H3(id="total-events", children="0", style={'color': '#007bff', 'fontSize': '2rem', 'margin': '0'}),
                                html.P("Total Events", style={'color': '#666', 'margin': '0'})
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H3(id="blocked-users", children="0", style={'color': '#dc3545', 'fontSize': '2rem', 'margin': '0'}),
                                html.P("Blocked Users", style={'color': '#666', 'margin': '0'})
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H3(id="high-risk-events", children="0", style={'color': '#ffc107', 'fontSize': '2rem', 'margin': '0'}),
                                html.P("High Risk Events", style={'color': '#666', 'margin': '0'})
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H3(id="current-risk-score", children="0.0", style={'color': '#28a745', 'fontSize': '2rem', 'margin': '0'}),
                                html.P("Avg Risk Score", style={'color': '#666', 'margin': '0'})
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], width=3)
                    ])
                ], style={'padding': '25px'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 12px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], width=12)
    ]),
    
    # Live Event Stream
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("🔴 Live Event Stream", style={'fontSize': '1.4rem', 'color': '#333', 'fontWeight': '600', 'margin': '0'})
                ], style={'padding': '20px 25px', 'background': '#f8f9fa', 'borderBottom': '1px solid #e9ecef'}),
                html.Div([
                    html.Div(id="live-events", children=[
                        html.P("No events yet. Start simulation to see live events.", 
                              style={'color': '#666', 'textAlign': 'center', 'padding': '20px'})
                    ], style={
                        'maxHeight': '400px',
                        'overflowY': 'auto',
                        'background': '#f8f9fa',
                        'borderRadius': '6px',
                        'padding': '15px'
                    })
                ], style={'padding': '25px'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 12px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], width=12)
    ]),
    
    # Risk Distribution and Trends
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("📈 Risk Distribution", style={'fontSize': '1.4rem', 'color': '#333', 'fontWeight': '600', 'margin': '0'})
                ], style={'padding': '20px 25px', 'background': '#f8f9fa', 'borderBottom': '1px solid #e9ecef'}),
                html.Div([
                    dcc.Graph(id="risk-distribution-chart", config={'displayModeBar': True, 'displaylogo': False})
                ], style={'padding': '25px'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 12px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], width=6),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("📊 Risk Trends", style={'fontSize': '1.4rem', 'color': '#333', 'fontWeight': '600', 'margin': '0'})
                ], style={'padding': '20px 25px', 'background': '#f8f9fa', 'borderBottom': '1px solid #e9ecef'}),
                html.Div([
                    dcc.Graph(id="risk-trends-chart", config={'displayModeBar': True, 'displaylogo': False})
                ], style={'padding': '25px'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 12px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], width=6)
    ]),
    
    # User Risk Table
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("👥 User Risk Assessment", style={'fontSize': '1.4rem', 'color': '#333', 'fontWeight': '600', 'margin': '0'}),
                    dcc.Dropdown(
                        id="user-risk-filter",
                        options=[
                            {'label': 'All Users', 'value': 'all'},
                            {'label': 'High Risk', 'value': 'high'},
                            {'label': 'Blocked Users', 'value': 'blocked'},
                        ],
                        value='all',
                        style={'minWidth': '180px'},
                        clearable=False
                    )
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '20px 25px', 'background': '#f8f9fa', 'borderBottom': '1px solid #e9ecef'}),
                html.Div([
                    dcc.Graph(id="user-risk-table", config={'displayModeBar': True, 'displaylogo': False})
                ], style={'padding': '25px'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 12px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], width=12)
    ]),
    
    # Detection Types Analysis
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("🔍 Enhanced Detection Analysis", style={'fontSize': '1.4rem', 'color': '#333', 'fontWeight': '600', 'margin': '0'})
                ], style={'padding': '20px 25px', 'background': '#f8f9fa', 'borderBottom': '1px solid #e9ecef'}),
                html.Div([
                    dcc.Graph(id="detection-types-chart", config={'displayModeBar': True, 'displaylogo': False})
                ], style={'padding': '25px'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 12px rgba(0, 0, 0, 0.1)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], width=12),
        
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    )
    
], fluid=True, style={'padding': '20px', 'background': '#f5f7fa', 'minHeight': '100vh'})


# ============================================
# CALLBACKS
# ============================================

@app.callback(
    [Output('start-simulation-btn', 'disabled'),
     Output('stop-simulation-btn', 'disabled'),
     Output('status-indicator', 'children'),
     Output('status-indicator', 'style')],
    [Input('start-simulation-btn', 'n_clicks'),
     Input('stop-simulation-btn', 'n_clicks')],
    [State('speed-slider', 'value'),
     State('duration-input', 'value'),
     State('block-threshold-input', 'value')]
)
def control_simulation(start_clicks, stop_clicks, speed, duration, block_threshold):
    """Control simulation start/stop"""
    
    ctx = callback_context
    if not ctx.triggered:
        return False, True, "Ready", {'color': '#90EE90', 'fontWeight': '500'}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-simulation-btn' and start_clicks:
        # Start simulation
        simulator.simulation_speed = speed
        simulator.BLOCK_THRESHOLD = block_threshold
        detector.BLOCK_THRESHOLD = block_threshold
        
        # Train auto-encoders if not already trained
        if (not detector.vae_detector or not detector.vae_detector.is_trained or 
            not detector.temporal_detector or not detector.temporal_detector.lstm_autoencoder.is_trained):
            print("Training auto-encoders with initial data...")
            detector.train_auto_encoders()
        
        simulator.start_simulation(duration_minutes=duration)
        dashboard_state['is_simulating'] = True
        dashboard_state['simulation_start_time'] = datetime.now()
        
        return True, False, "Running", {'color': '#ffc107', 'fontWeight': '500'}
    
    elif button_id == 'stop-simulation-btn' and stop_clicks:
        # Stop simulation
        simulator.stop_simulation()
        dashboard_state['is_simulating'] = False
        
        return False, True, "Stopped", {'color': '#dc3545', 'fontWeight': '500'}
    
    return False, True, "Ready", {'color': '#90EE90', 'fontWeight': '500'}


@app.callback(
    [Output('total-events', 'children'),
     Output('blocked-users', 'children'),
     Output('high-risk-events', 'children'),
     Output('current-risk-score', 'children'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_metrics(n):
    """Update real-time metrics"""
    
    stats = simulator.get_simulation_stats()
    
    total_events = stats.get('total_events', 0)
    blocked_users = stats.get('blocked_users', 0)
    high_risk_events = stats.get('high_risk_events', 0)
    
    # Calculate average risk score
    if simulator.events:
        avg_risk = np.mean([event.risk_score for event in simulator.events[-100:]])  # Last 100 events
    else:
        avg_risk = 0.0
    
    dashboard_state['total_events'] = total_events
    dashboard_state['blocked_users'] = blocked_users
    dashboard_state['high_risk_events'] = high_risk_events
    dashboard_state['last_update'] = datetime.now()
    
    return (
        f"{total_events:,}",
        f"{blocked_users}",
        f"{high_risk_events}",
        f"{avg_risk:.1f}",
        datetime.now().strftime("%H:%M:%S")
    )


@app.callback(
    Output('live-events', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_live_events(n):
    """Update live event stream"""
    
    if not simulator.events:
        return html.P("No events yet. Start simulation to see live events.", 
                     style={'color': '#666', 'textAlign': 'center', 'padding': '20px'})
    
    # Get last 20 events
    recent_events = simulator.events[-20:]
    recent_events.reverse()  # Show newest first
    
    event_cards = []
    for event in recent_events:
        # Determine risk color
        if event.risk_score >= 70:
            risk_color = '#dc3545'
            risk_icon = '🔴'
        elif event.risk_score >= 40:
            risk_color = '#ffc107'
            risk_icon = '🟡'
        else:
            risk_color = '#28a745'
            risk_icon = '🟢'
        
        # Create event card
        event_card = html.Div([
            html.Div([
                html.Span(f"{risk_icon} {event.user_id}", 
                         style={'fontWeight': '600', 'color': risk_color}),
                html.Span(f"Risk: {event.risk_score:.1f}", 
                         style={'float': 'right', 'fontWeight': '500', 'color': risk_color})
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span(f"📁 {event.resource_id}", style={'fontSize': '14px'}),
                html.Span(f"🔧 {event.action}", style={'fontSize': '14px', 'marginLeft': '10px'}),
                html.Span(f"🖥️ {event.destination_computer}", style={'fontSize': '14px', 'marginLeft': '10px'})
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span(f"⏰ {event.timestamp.strftime('%H:%M:%S')}", style={'fontSize': '12px', 'color': '#666'}),
                html.Span(f"🎯 {', '.join(event.detection_types[:2]) if hasattr(event, 'detection_types') and event.detection_types else event.detection_type}", style={'fontSize': '12px', 'color': '#666', 'marginLeft': '10px'})
            ])
        ], style={
            'background': 'white',
            'padding': '10px',
            'marginBottom': '8px',
            'borderRadius': '6px',
            'borderLeft': f'4px solid {risk_color}',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
        })
        
        event_cards.append(event_card)
    
    return event_cards


@app.callback(
    Output('risk-distribution-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_risk_distribution(n):
    """Update risk distribution chart"""
    
    if not simulator.events:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Start simulation to see risk distribution.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300, showlegend=False)
        return fig
    
    # Calculate risk distribution
    risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
    for event in simulator.events:
        if event.risk_score < 40:
            risk_levels['Low'] += 1
        elif event.risk_score < 70:
            risk_levels['Medium'] += 1
        else:
            risk_levels['High'] += 1
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_levels.keys()),
        values=list(risk_levels.values()),
        hole=0.4,
        marker_colors=['#28a745', '#ffc107', '#dc3545'],
        textinfo='label+percent+value',
        textfont_size=12
    )])
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True
    )
    
    return fig


@app.callback(
    Output('risk-trends-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_risk_trends(n):
    """Update risk trends chart"""
    
    if not simulator.events:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Start simulation to see risk trends.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300, showlegend=False)
        return fig
    
    # Get last 50 events for trend analysis
    recent_events = simulator.events[-50:]
    
    if len(recent_events) < 2:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for trends.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300, showlegend=False)
        return fig
    
    # Create trend data
    timestamps = [event.timestamp for event in recent_events]
    risk_scores = [event.risk_score for event in recent_events]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=risk_scores,
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#007bff', width=2),
        marker=dict(size=4)
    ))
    
    # Add risk level lines
    fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk")
    
    fig.update_layout(
        title="Risk Score Trends (Last 50 Events)",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


@app.callback(
    Output('user-risk-table', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('user-risk-filter', 'value')]
)
def update_user_risk_table(n, filter_value):
    """Update user risk table"""
    
    if not simulator.users:
        # Empty table
        fig = go.Figure()
        fig.add_annotation(
            text="No user data available. Start simulation to see user risk assessment.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    # Prepare user data
    user_data = []
    for user_id, user in simulator.users.items():
        user_data.append({
            'user_id': user_id,
            'department': user.department,
            'role': user.role,
            'risk_score': user.current_risk_score,
            'is_blocked': user.is_blocked,
            'access_count': user.access_count,
            'last_access': user.last_access.strftime('%H:%M:%S') if user.last_access else 'Never'
        })
    
    # Apply filter
    if filter_value == 'high':
        user_data = [u for u in user_data if u['risk_score'] >= 70]
    elif filter_value == 'blocked':
        user_data = [u for u in user_data if u['is_blocked']]
    
    # Sort by risk score
    user_data.sort(key=lambda x: x['risk_score'], reverse=True)
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['User ID', 'Department', 'Role', 'Risk Score', 'Status', 'Access Count', 'Last Access'],
            fill_color='#f8f9fa',
            align='left',
            font=dict(size=12, color='#495057'),
            height=40
        ),
        cells=dict(
            values=[
                [u['user_id'] for u in user_data],
                [u['department'] for u in user_data],
                [u['role'] for u in user_data],
                [f"{u['risk_score']:.1f}" for u in user_data],
                [("🔴 BLOCKED" if u['is_blocked'] else "🟢 ACTIVE") for u in user_data],
                [u['access_count'] for u in user_data],
                [u['last_access'] for u in user_data]
            ],
            fill_color=[['#ffffff' if i % 2 == 0 else '#f8f9fa' for i in range(len(user_data))]],
            align='left',
            font=dict(size=11),
            height=35
        )
    )])
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig


@app.callback(
    Output('detection-types-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_detection_types(n):
    """Update detection types analysis chart"""
    
    if not simulator.events:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Start simulation to see detection types analysis.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300, showlegend=False)
        return fig
    
    # Count detection types
    detection_counts = {}
    for event in simulator.events:
        if hasattr(event, 'detection_types') and event.detection_types:
            # Enhanced detection with multiple types
            for detection_type in event.detection_types:
                if detection_type and detection_type != 'normal':
                    detection_counts[detection_type] = detection_counts.get(detection_type, 0) + 1
        elif hasattr(event, 'detection_type') and event.detection_type and event.detection_type != 'normal':
            # Traditional detection with single type
            detection_counts[event.detection_type] = detection_counts.get(event.detection_type, 0) + 1
    
    if not detection_counts:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No anomalies detected yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300, showlegend=False)
        return fig
    
    # Create bar chart
    fig = go.Figure(data=[go.Bar(
        x=list(detection_counts.keys()),
        y=list(detection_counts.values()),
        marker_color='#007bff',
        text=list(detection_counts.values()),
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Detection Types Frequency",
        xaxis_title="Detection Type",
        yaxis_title="Count",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


@app.callback(
    Output('simulation-stats', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_simulation_stats(n):
    """Update simulation statistics"""
    
    if not dashboard_state['is_simulating']:
        return html.P("Simulation not running", style={'color': '#666', 'margin': '0'})
    
    stats = simulator.get_simulation_stats()
    
    return html.Div([
        html.P(f"⏱️ Runtime: {stats.get('simulation_time', '0:00:00')}", style={'margin': '2px 0', 'fontSize': '14px'}),
        html.P(f"📈 Events/min: {stats.get('events_per_minute', 0):.1f}", style={'margin': '2px 0', 'fontSize': '14px'}),
        html.P(f"🎯 Risk Distribution: {stats.get('risk_distribution', {})}", style={'margin': '2px 0', 'fontSize': '14px'})
    ])




# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Live Cyber Threat Detection Dashboard")
    print("="*60)
    print(f"\nDashboard ready!")
    print(f"Open browser to: http://localhost:{DASHBOARD_PORT}")
    print(f"Monitoring {simulator.num_users} users")
    print(f"Use simulation controls to start live detection")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, host='localhost', port=DASHBOARD_PORT)
