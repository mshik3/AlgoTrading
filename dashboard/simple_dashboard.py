"""
Simplified Professional Trading Dashboard
Works without complex dependencies - perfect for testing
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    ],
    title="AlgoTrading Dashboard",
)


# Mock data for demonstration
def get_mock_portfolio():
    return {
        "total_value": 105750.00,
        "cash": 45750.00,
        "total_pnl": 5750.00,
        "positions_value": 60000.00,
    }


def get_mock_positions():
    return [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "avg_price": 150.00,
            "current_price": 157.50,
            "market_value": 15750.00,
            "unrealized_pnl": 750.00,
            "pnl_percent": 0.05,
        },
        {
            "symbol": "MSFT",
            "quantity": 75,
            "avg_price": 280.00,
            "current_price": 295.00,
            "market_value": 22125.00,
            "unrealized_pnl": 1125.00,
            "pnl_percent": 0.0536,
        },
        {
            "symbol": "GOOGL",
            "quantity": 50,
            "avg_price": 130.00,
            "current_price": 142.50,
            "market_value": 7125.00,
            "unrealized_pnl": 625.00,
            "pnl_percent": 0.0962,
        },
    ]


# App layout with inline styles
app.layout = dbc.Container(
    [
        # Auto-refresh component
        dcc.Interval(
            id="dashboard-interval", interval=30 * 1000, n_intervals=0  # 30 seconds
        ),
        # Header Section
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "⚡ AlgoTrading Dashboard",
                            style={
                                "color": "#d1d4dc",
                                "textAlign": "center",
                                "margin": 0,
                            },
                        ),
                        html.P(
                            f"Last Updated: {datetime.now().strftime('%H:%M:%S')}",
                            id="last-update",
                            style={
                                "color": "#868b95",
                                "textAlign": "center",
                                "fontSize": "14px",
                            },
                        ),
                    ],
                    style={
                        "background": "linear-gradient(135deg, #161a25, #1e222d)",
                        "borderRadius": "12px",
                        "padding": "20px",
                        "marginBottom": "20px",
                        "border": "1px solid #363a45",
                    },
                )
            ]
        ),
        # KPI Cards Row
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Total Portfolio Value",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "$105,750.00",
                                            id="total-value",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "24px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "+$5,750.00",
                                            id="total-change",
                                            style={
                                                "color": "#26a69a",
                                                "fontSize": "12px",
                                                "fontWeight": "500",
                                            },
                                        ),
                                    ],
                                    style={
                                        "background": "#2a2e39",
                                        "borderRadius": "12px",
                                        "padding": "20px",
                                        "border": "1px solid #363a45",
                                        "height": "100%",
                                    },
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Available Cash",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "$45,750.00",
                                            id="available-cash",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "24px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "Ready to Trade",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "background": "#2a2e39",
                                        "borderRadius": "12px",
                                        "padding": "20px",
                                        "border": "1px solid #363a45",
                                        "height": "100%",
                                    },
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Today's P&L",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "+$1,725.00",
                                            id="daily-pnl",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "24px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "+1.73%",
                                            id="daily-change",
                                            style={
                                                "color": "#26a69a",
                                                "fontSize": "12px",
                                                "fontWeight": "500",
                                            },
                                        ),
                                    ],
                                    style={
                                        "background": "#2a2e39",
                                        "borderRadius": "12px",
                                        "padding": "20px",
                                        "border": "1px solid #363a45",
                                        "height": "100%",
                                    },
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Total Return",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "+5.75%",
                                            id="total-return",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "24px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "Since Inception",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "background": "#2a2e39",
                                        "borderRadius": "12px",
                                        "padding": "20px",
                                        "border": "1px solid #363a45",
                                        "height": "100%",
                                    },
                                )
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-4",
                )
            ]
        ),
        # Main Content Row
        dbc.Row(
            [
                # Left Column - Portfolio & Positions
                dbc.Col(
                    [
                        # Current Positions Table
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(
                                            className="fas fa-chart-line",
                                            style={"marginRight": "10px"},
                                        ),
                                        "Current Positions",
                                    ],
                                    style={
                                        "color": "#d1d4dc",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(id="positions-table"),
                            ],
                            style={
                                "background": "#2a2e39",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #363a45",
                                "marginBottom": "20px",
                            },
                        ),
                        # Strategy Monitor
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(
                                            className="fas fa-cogs",
                                            style={"marginRight": "10px"},
                                        ),
                                        "Strategy Monitor",
                                    ],
                                    style={
                                        "color": "#d1d4dc",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(id="strategy-monitor"),
                            ],
                            style={
                                "background": "#2a2e39",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #363a45",
                            },
                        ),
                    ],
                    md=6,
                ),
                # Right Column - Charts & Activity
                dbc.Col(
                    [
                        # Portfolio Performance Chart
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(
                                            className="fas fa-chart-area",
                                            style={"marginRight": "10px"},
                                        ),
                                        "Portfolio Performance",
                                    ],
                                    style={
                                        "color": "#d1d4dc",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginBottom": "15px",
                                    },
                                ),
                                dcc.Graph(id="portfolio-chart"),
                            ],
                            style={
                                "background": "#2a2e39",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #363a45",
                                "marginBottom": "20px",
                            },
                        ),
                        # Recent Activity Feed
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(
                                            className="fas fa-history",
                                            style={"marginRight": "10px"},
                                        ),
                                        "Recent Activity",
                                    ],
                                    style={
                                        "color": "#d1d4dc",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(id="activity-feed"),
                            ],
                            style={
                                "background": "#2a2e39",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #363a45",
                            },
                        ),
                    ],
                    md=6,
                ),
            ]
        ),
    ],
    fluid=True,
    style={
        "backgroundColor": "#0d1017",
        "minHeight": "100vh",
        "padding": "20px",
        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    },
)


# Callback for real-time updates
@app.callback(
    [
        Output("last-update", "children"),
        Output("positions-table", "children"),
        Output("strategy-monitor", "children"),
        Output("portfolio-chart", "figure"),
        Output("activity-feed", "children"),
    ],
    [Input("dashboard-interval", "n_intervals")],
)
def update_dashboard(n_intervals):
    """Update all dashboard components"""

    # Update timestamp
    current_time = f"Last Updated: {datetime.now().strftime('%H:%M:%S')}"

    # Create positions table
    positions = get_mock_positions()
    positions_df = pd.DataFrame(positions)

    positions_table = dash_table.DataTable(
        data=positions_df.to_dict("records"),
        columns=[
            {"name": "Symbol", "id": "symbol"},
            {"name": "Quantity", "id": "quantity", "type": "numeric"},
            {
                "name": "Avg Price",
                "id": "avg_price",
                "type": "numeric",
                "format": {"specifier": ",.2f"},
            },
            {
                "name": "Current",
                "id": "current_price",
                "type": "numeric",
                "format": {"specifier": ",.2f"},
            },
            {
                "name": "Value",
                "id": "market_value",
                "type": "numeric",
                "format": {"specifier": ",.2f"},
            },
            {
                "name": "P&L",
                "id": "unrealized_pnl",
                "type": "numeric",
                "format": {"specifier": "+,.2f"},
            },
            {
                "name": "P&L %",
                "id": "pnl_percent",
                "type": "numeric",
                "format": {"specifier": "+.2%"},
            },
        ],
        style_cell={
            "textAlign": "center",
            "padding": "10px",
            "backgroundColor": "#2a2e39",
            "color": "#d1d4dc",
            "border": "1px solid #363a45",
        },
        style_header={
            "backgroundColor": "#1e222d",
            "color": "#d1d4dc",
            "fontWeight": "bold",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": "{unrealized_pnl} > 0"},
                "color": "#26a69a",
                "fontWeight": "bold",
            },
            {
                "if": {"filter_query": "{unrealized_pnl} < 0"},
                "color": "#ef5350",
                "fontWeight": "bold",
            },
        ],
    )

    # Strategy monitor
    strategy_monitor = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "●",
                        style={
                            "color": "#26a69a",
                            "marginRight": "8px",
                            "fontSize": "12px",
                        },
                    ),
                    html.Span(
                        "Golden Cross Strategy",
                        style={
                            "color": "#d1d4dc",
                            "fontWeight": "600",
                            "fontSize": "16px",
                        },
                    ),
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Small(
                                                "Status: ", style={"color": "#868b95"}
                                            ),
                                            html.Span(
                                                "ACTIVE", className="badge bg-success"
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Small(
                                                "Last Signal: ",
                                                style={"color": "#868b95"},
                                            ),
                                            html.Span(
                                                "BUY", className="badge bg-primary"
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Small(
                                                "Win Rate: ", style={"color": "#868b95"}
                                            ),
                                            html.Span(
                                                "68%", style={"color": "#26a69a"}
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Small(
                                                "Total Trades: ",
                                                style={"color": "#868b95"},
                                            ),
                                            html.Span("24", style={"color": "#d1d4dc"}),
                                        ],
                                        md=6,
                                    ),
                                ],
                                style={"marginTop": "10px"},
                            ),
                        ]
                    ),
                ],
                style={"marginBottom": "15px"},
            )
        ]
    )

    # Portfolio chart
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D"
    )
    base_value = 100000
    daily_returns = [
        0.008,
        -0.012,
        0.015,
        -0.005,
        0.022,
        -0.008,
        0.011,
        0.006,
        -0.015,
        0.018,
        0.003,
        -0.009,
        0.025,
        -0.011,
        0.007,
        0.014,
        -0.006,
        0.019,
        -0.003,
        0.008,
        0.012,
        -0.007,
        0.009,
        0.016,
        -0.013,
        0.021,
        -0.004,
        0.010,
        0.005,
        -0.002,
        0.0575,
    ]

    values = [base_value]
    for ret in daily_returns[: len(dates) - 1]:
        values.append(values[-1] * (1 + ret))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates[: len(values)],
            y=values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#2962ff", width=2),
            hovertemplate="%{y:$,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#d1d4dc",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(gridcolor="#363a45", showgrid=True, color="#d1d4dc"),
        yaxis=dict(
            gridcolor="#363a45", showgrid=True, tickformat="$,.0f", color="#d1d4dc"
        ),
    )

    # Activity feed
    now = datetime.now()
    activities = [
        {
            "time": (now - timedelta(minutes=5)).strftime("%H:%M:%S"),
            "desc": "Portfolio updated: $105,750 (+$1,725)",
            "icon": "fas fa-chart-line",
        },
        {
            "time": (now - timedelta(minutes=15)).strftime("%H:%M:%S"),
            "desc": "Golden Cross BUY signal for AAPL",
            "icon": "fas fa-bullseye",
        },
        {
            "time": (now - timedelta(hours=2)).strftime("%H:%M:%S"),
            "desc": "BUY: 50 AAPL @ $155.25",
            "icon": "fas fa-exchange-alt",
        },
        {
            "time": (now - timedelta(hours=4)).strftime("%H:%M:%S"),
            "desc": "MSFT +5.36% ($1,125 gain)",
            "icon": "fas fa-arrow-up",
        },
        {
            "time": (now - timedelta(days=1)).strftime("%H:%M:%S"),
            "desc": "BUY: 25 MSFT @ $285.50",
            "icon": "fas fa-exchange-alt",
        },
    ]

    activity_feed = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.I(
                                className=activity["icon"],
                                style={"marginRight": "10px", "color": "#2962ff"},
                            ),
                            html.Span(
                                activity["time"],
                                style={"color": "#868b95", "fontSize": "11px"},
                            ),
                        ]
                    ),
                    html.Div(
                        activity["desc"],
                        style={
                            "color": "#d1d4dc",
                            "fontSize": "13px",
                            "marginTop": "4px",
                        },
                    ),
                ],
                style={
                    "padding": "15px",
                    "borderBottom": (
                        "1px solid #363a45" if i < len(activities) - 1 else "none"
                    ),
                },
            )
            for i, activity in enumerate(activities)
        ]
    )

    return current_time, positions_table, strategy_monitor, fig, activity_feed


if __name__ == "__main__":
    print("🚀 Starting Simplified AlgoTrading Dashboard...")
    print("📊 Visit: http://127.0.0.1:8050")
    app.run_server(debug=True, host="127.0.0.1", port=8050)
