"""
Professional Algorithmic Trading Dashboard
Built with Plotly Dash - Industry-grade financial dashboard
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mock classes for basic functionality - we'll use these for now
class PaperTradingSimulator:
    def get_portfolio_summary(self):
        return {
            "total_value": 105750.00,
            "cash": 45750.00,
            "total_pnl": 5750.00,
            "positions_value": 60000.00,
        }

    def get_positions(self):
        # Mock some positions for demonstration
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

    def get_trade_history(self, limit=10):
        from datetime import datetime, timedelta

        return [
            {
                "timestamp": datetime.now() - timedelta(hours=2),
                "action": "BUY",
                "symbol": "AAPL",
                "quantity": 50,
                "price": 155.25,
            },
            {
                "timestamp": datetime.now() - timedelta(days=1),
                "action": "BUY",
                "symbol": "MSFT",
                "quantity": 25,
                "price": 285.50,
            },
        ]


# Initialize paper trader
paper_trader = PaperTradingSimulator()


# Initialize the Dash app with professional theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,  # For grid system
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",  # Icons
    ],
    suppress_callback_exceptions=True,
    title="AlgoTrading Dashboard",
)

# App layout
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
                            "⚡ AlgoTrading Dashboard", className="dashboard-title"
                        ),
                        html.P(
                            f"Last Updated: {datetime.now().strftime('%H:%M:%S')}",
                            id="last-update",
                            className="dashboard-subtitle",
                        ),
                    ],
                    className="dashboard-header",
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
                                            className="kpi-title",
                                        ),
                                        html.Div(
                                            "$0.00",
                                            id="total-value",
                                            className="kpi-value",
                                        ),
                                        html.Div(
                                            "--",
                                            id="total-change",
                                            className="kpi-change neutral",
                                        ),
                                    ],
                                    className="kpi-card",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Available Cash", className="kpi-title"
                                        ),
                                        html.Div(
                                            "$0.00",
                                            id="available-cash",
                                            className="kpi-value",
                                        ),
                                        html.Div(
                                            "Ready to Trade",
                                            className="kpi-change neutral",
                                        ),
                                    ],
                                    className="kpi-card",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div("Today's P&L", className="kpi-title"),
                                        html.Div(
                                            "$0.00",
                                            id="daily-pnl",
                                            className="kpi-value",
                                        ),
                                        html.Div(
                                            "--",
                                            id="daily-change",
                                            className="kpi-change neutral",
                                        ),
                                    ],
                                    className="kpi-card",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div("Total Return", className="kpi-title"),
                                        html.Div(
                                            "0.00%",
                                            id="total-return",
                                            className="kpi-value",
                                        ),
                                        html.Div(
                                            "Since Inception",
                                            className="kpi-change neutral",
                                        ),
                                    ],
                                    className="kpi-card",
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
                                        html.I(className="fas fa-chart-line me-2"),
                                        "Current Positions",
                                    ],
                                    className="chart-title",
                                ),
                                html.Div(id="positions-table"),
                            ],
                            className="chart-container mb-4",
                        ),
                        # Strategy Monitor
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(className="fas fa-cogs me-2"),
                                        "Strategy Monitor",
                                    ],
                                    className="chart-title",
                                ),
                                html.Div(id="strategy-monitor"),
                            ],
                            className="chart-container",
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
                                        html.I(className="fas fa-chart-area me-2"),
                                        "Portfolio Performance",
                                    ],
                                    className="chart-title",
                                ),
                                dcc.Graph(id="portfolio-chart"),
                            ],
                            className="chart-container mb-4",
                        ),
                        # Recent Activity Feed
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(className="fas fa-history me-2"),
                                        "Recent Activity",
                                    ],
                                    className="chart-title",
                                ),
                                html.Div(id="activity-feed"),
                            ],
                            className="chart-container",
                        ),
                    ],
                    md=6,
                ),
            ]
        ),
    ],
    fluid=True,
    className="dashboard-container",
)


# Callback for real-time updates
@app.callback(
    [
        Output("last-update", "children"),
        Output("total-value", "children"),
        Output("available-cash", "children"),
        Output("daily-pnl", "children"),
        Output("total-return", "children"),
        Output("total-change", "children"),
        Output("total-change", "className"),
        Output("daily-change", "children"),
        Output("daily-change", "className"),
        Output("positions-table", "children"),
        Output("strategy-monitor", "children"),
        Output("portfolio-chart", "figure"),
        Output("activity-feed", "children"),
    ],
    [Input("dashboard-interval", "n_intervals")],
)
def update_dashboard(n_intervals):
    """Update all dashboard components with real-time data"""

    try:
        # Get current time
        current_time = f"Last Updated: {datetime.now().strftime('%H:%M:%S')}"

        # Get portfolio summary
        portfolio = paper_trader.get_portfolio_summary()
        positions = paper_trader.get_positions()

        # Calculate KPIs
        total_value = f"${portfolio.get('total_value', 0):,.2f}"
        cash = f"${portfolio.get('cash', 0):,.2f}"

        # Calculate daily P&L (mock for now)
        daily_pnl = (
            portfolio.get("total_pnl", 0) * 0.3
        )  # Assume 30% of total P&L is today
        daily_pnl_str = f"${daily_pnl:+,.2f}" if daily_pnl != 0 else "$0.00"
        daily_change_class = (
            "kpi-change positive"
            if daily_pnl > 0
            else "kpi-change negative" if daily_pnl < 0 else "kpi-change neutral"
        )
        daily_change_text = (
            f"{(daily_pnl/100000)*100:+.2f}%" if daily_pnl != 0 else "0.00%"
        )

        # Total return calculation
        initial_value = 100000  # Mock initial portfolio value
        current_value = portfolio.get("total_value", initial_value)
        total_return_pct = ((current_value - initial_value) / initial_value) * 100
        total_return_str = f"{total_return_pct:+.2f}%"

        # Total change
        total_change_str = f"${current_value - initial_value:+,.2f}"
        total_change_class = (
            "kpi-change positive"
            if total_return_pct > 0
            else "kpi-change negative" if total_return_pct < 0 else "kpi-change neutral"
        )

        # Build positions table
        positions_table = create_positions_table(positions)

        # Build strategy monitor
        strategy_monitor = create_strategy_monitor()

        # Build portfolio chart
        portfolio_chart = create_portfolio_chart()

        # Build activity feed
        activity_feed = create_activity_feed()

        return (
            current_time,
            total_value,
            cash,
            daily_pnl_str,
            total_return_str,
            total_change_str,
            total_change_class,
            daily_change_text,
            daily_change_class,
            positions_table,
            strategy_monitor,
            portfolio_chart,
            activity_feed,
        )

    except Exception as e:
        print(f"Dashboard update error: {e}")
        # Return default values on error
        current_time = (
            f"Last Updated: {datetime.now().strftime('%H:%M:%S')} (Error: {str(e)})"
        )
        return (
            current_time,
            "$105,750.00",
            "$45,750.00",
            "+$1,725.00",
            "+5.75%",
            "+$5,750.00",
            "kpi-change positive",
            "+1.73%",
            "kpi-change positive",
            create_positions_table([]),
            create_strategy_monitor(),
            create_empty_portfolio_chart(),
            create_empty_activity_feed(),
        )


def create_positions_table(positions):
    """Create the current positions table"""
    if not positions:
        return html.Div(
            [html.P("No active positions", className="text-center text-muted py-3")]
        )

    # Convert positions to DataFrame
    df = pd.DataFrame(positions)

    # Create table
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[
            {"name": "Symbol", "id": "symbol"},
            {
                "name": "Quantity",
                "id": "quantity",
                "type": "numeric",
                "format": {"specifier": ",.0f"},
            },
            {
                "name": "Avg Price",
                "id": "avg_price",
                "type": "numeric",
                "format": {"specifier": ",.2f"},
            },
            {
                "name": "Current Price",
                "id": "current_price",
                "type": "numeric",
                "format": {"specifier": ",.2f"},
            },
            {
                "name": "Market Value",
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
        style_cell={"textAlign": "center", "padding": "10px"},
        style_data_conditional=[
            {
                "if": {"filter_query": "{unrealized_pnl} > 0"},
                "color": "var(--profit-color)",
                "fontWeight": "bold",
            },
            {
                "if": {"filter_query": "{unrealized_pnl} < 0"},
                "color": "var(--loss-color)",
                "fontWeight": "bold",
            },
        ],
        className="trading-table",
    )


def create_empty_positions_table():
    """Create empty positions table for error state"""
    return html.Div(
        [html.P("No positions data available", className="text-center text-muted py-3")]
    )


def create_strategy_monitor():
    """Create the strategy monitoring section"""
    return html.Div(
        [
            # Golden Cross Strategy Status
            html.Div(
                [
                    html.Span(className="strategy-indicator active"),
                    html.Span("Golden Cross Strategy", className="strategy-name"),
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Small(
                                                "Status: ", className="text-muted"
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
                                                "Last Signal: ", className="text-muted"
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
                                                "Win Rate: ", className="text-muted"
                                            ),
                                            html.Span(
                                                "68%",
                                                style={"color": "var(--profit-color)"},
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Small(
                                                "Total Trades: ", className="text-muted"
                                            ),
                                            html.Span("24"),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="mt-2",
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            # Add more strategies here as they are implemented
            html.Div(
                [
                    html.Span(className="strategy-indicator inactive"),
                    html.Span("Mean Reversion Strategy", className="strategy-name"),
                    html.Small("Coming Soon...", className="text-muted"),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    html.Span(className="strategy-indicator inactive"),
                    html.Span("ETF Rotation Strategy", className="strategy-name"),
                    html.Small("Coming Soon...", className="text-muted"),
                ]
            ),
        ]
    )


def create_portfolio_chart():
    """Create the portfolio performance chart"""
    try:
        # Generate mock portfolio history
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D"
        )
        # Generate realistic portfolio progression
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

        # Portfolio line
        fig.add_trace(
            go.Scatter(
                x=dates[: len(values)],
                y=values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#2962ff", width=2),
                hovertemplate="%{y:$,.2f}<extra></extra>",
                fill="tonexty",
            )
        )

        # Styling
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="var(--text-primary)",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                gridcolor="var(--border-color)",
                showgrid=True,
                color="var(--text-primary)",
            ),
            yaxis=dict(
                gridcolor="var(--border-color)",
                showgrid=True,
                tickformat="$,.0f",
                color="var(--text-primary)",
            ),
        )

        return fig

    except Exception as e:
        print(f"Chart creation error: {e}")
        return create_empty_portfolio_chart()


def create_empty_portfolio_chart():
    """Create empty portfolio chart for error state"""
    fig = go.Figure()
    fig.add_annotation(
        text="Portfolio data loading...",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="var(--text-secondary)"),
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def create_activity_feed():
    """Create the recent activity feed"""
    # Mock activity data based on our paper trader
    now = datetime.now()
    activities = [
        {
            "time": (now - timedelta(minutes=5)).strftime("%H:%M:%S"),
            "description": "Portfolio value updated: $105,750 (+$1,725 today)",
            "type": "system",
        },
        {
            "time": (now - timedelta(minutes=15)).strftime("%H:%M:%S"),
            "description": "Golden Cross BUY signal detected for AAPL",
            "type": "signal",
        },
        {
            "time": (now - timedelta(hours=2)).strftime("%H:%M:%S"),
            "description": "Executed BUY: 50 shares of AAPL at $155.25",
            "type": "trade",
        },
        {
            "time": (now - timedelta(hours=4)).strftime("%H:%M:%S"),
            "description": "MSFT position up +5.36% ($1,125 unrealized gain)",
            "type": "performance",
        },
        {
            "time": (now - timedelta(days=1)).strftime("%H:%M:%S"),
            "description": "Executed BUY: 25 shares of MSFT at $285.50",
            "type": "trade",
        },
        {
            "time": (now - timedelta(days=1, hours=3)).strftime("%H:%M:%S"),
            "description": "Portfolio rebalanced - Risk level: Moderate",
            "type": "system",
        },
        {
            "time": (now - timedelta(days=2)).strftime("%H:%M:%S"),
            "description": "Golden Cross strategy performance: +8.2% this month",
            "type": "performance",
        },
    ]

    activity_items = []
    for activity in activities:
        icon = get_activity_icon(activity["type"])
        activity_items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.I(
                                className=icon,
                                style={
                                    "marginRight": "10px",
                                    "color": "var(--accent-primary)",
                                },
                            ),
                            html.Span(activity["time"], className="activity-time"),
                        ]
                    ),
                    html.Div(activity["description"], className="activity-description"),
                ],
                className="activity-item",
            )
        )

    return html.Div(activity_items, className="activity-feed")


def create_empty_activity_feed():
    """Create empty activity feed for error state"""
    return html.Div(
        [html.P("No recent activity", className="text-center text-muted py-3")]
    )


def get_activity_icon(activity_type):
    """Get appropriate icon for activity type"""
    icons = {
        "signal": "fas fa-bullseye",
        "trade": "fas fa-exchange-alt",
        "system": "fas fa-cog",
        "performance": "fas fa-chart-line",
    }
    return icons.get(activity_type, "fas fa-info-circle")


if __name__ == "__main__":
    print("🚀 Starting AlgoTrading Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    app.run_server(debug=True, host="127.0.0.1", port=8050)
