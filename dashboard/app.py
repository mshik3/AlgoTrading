"""
Professional Algorithmic Trading Dashboard
Built with Plotly Dash - Industry-grade financial dashboard
"""

import sys
import os

# Add project root to path for imports - ensure this happens before any other imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# Import analysis components
from dashboard.components.analysis import (
    create_analysis_layout,
    register_analysis_callbacks,
)


# Import real Alpaca account service
from dashboard.services.alpaca_account import AlpacaAccountService

# Initialize real Alpaca account service
try:
    alpaca_account = AlpacaAccountService()
    if not alpaca_account.is_connected():
        raise Exception("Alpaca connection failed")
except Exception as e:
    print(f"❌ Alpaca connection required: {e}")
    print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
    exit(1)


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
        # Analysis Section
        html.Div([create_analysis_layout()], className="mt-5"),
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

        # Get real portfolio summary from Alpaca
        portfolio = alpaca_account.get_account_summary()
        positions = alpaca_account.get_positions()

        # Validate positions data
        if not isinstance(positions, list):
            positions = []

        # Calculate KPIs
        # Ensure portfolio data is valid
        if not isinstance(portfolio, dict):
            portfolio = {"total_value": 0, "cash": 0, "total_pnl": 0}

        total_value = f"${portfolio.get('total_value', 0):,.2f}"
        cash = f"${portfolio.get('cash', 0):,.2f}"

        # Calculate daily P&L - return 0 if no positions
        positions_count = len(positions) if positions else 0
        if positions_count == 0:
            daily_pnl = 0.0
            daily_pnl_str = "$0.00"
            daily_change_class = "kpi-change neutral"
            daily_change_text = "0.00%"
        else:
            # Only calculate P&L if there are actual positions
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

        # Total return calculation - handle zero positions case
        if positions_count == 0:
            total_return_pct = 0.0
            total_return_str = "0.00%"
            total_change_str = "$0.00"
            total_change_class = "kpi-change neutral"
        else:
            initial_value = 100000  # Mock initial portfolio value
            current_value = portfolio.get("total_value", initial_value)
            total_return_pct = ((current_value - initial_value) / initial_value) * 100
            total_return_str = f"{total_return_pct:+.2f}%"

            # Total change
            total_change_str = f"${current_value - initial_value:+,.2f}"
            total_change_class = (
                "kpi-change positive"
                if total_return_pct > 0
                else (
                    "kpi-change negative"
                    if total_return_pct < 0
                    else "kpi-change neutral"
                )
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
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        # Return zero values on error when no positions exist
        current_time = (
            f"Last Updated: {datetime.now().strftime('%H:%M:%S')} (Error: {str(e)})"
        )
        return (
            current_time,
            "$0.00",
            "$0.00",
            "$0.00",
            "0.00%",
            "$0.00",
            "kpi-change neutral",
            "0.00%",
            "kpi-change neutral",
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
        css=[
            {
                "selector": ".dash-table-container",
                "rule": "background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border-color);",
            }
        ],
    )


def create_empty_positions_table():
    """Create empty positions table for error state"""
    return html.Div(
        [html.P("No positions data available", className="text-center text-muted py-3")]
    )


def get_real_strategy_metrics():
    """
    Get real strategy metrics from Alpaca account data.

    Returns:
        Dictionary with real strategy metrics
    """
    try:
        # Get recent orders from Alpaca
        recent_orders = alpaca_account.get_recent_orders(limit=100)

        # Get current positions to calculate P&L
        current_positions = alpaca_account.get_positions()

        # Calculate basic metrics
        total_orders = len(recent_orders)
        filled_orders = [o for o in recent_orders if o.get("status") == "filled"]
        total_filled = len(filled_orders)

        # Calculate win rate from actual P&L data
        win_rate = None
        if total_filled > 0:
            # Look at current positions for unrealized P&L
            profitable_positions = 0
            total_positions = len(current_positions)

            if total_positions > 0:
                for position in current_positions:
                    if position.get("unrealized_pnl", 0) > 0:
                        profitable_positions += 1

                # Calculate win rate based on current position performance
                win_rate = (
                    (profitable_positions / total_positions) * 100
                    if total_positions > 0
                    else None
                )
            else:
                # No current positions, can't calculate win rate without historical P&L
                win_rate = None

        # Get last signal
        last_signal = "NONE"
        if filled_orders:
            last_order = filled_orders[0]  # Most recent order
            last_signal = last_order.get("action", "NONE").upper()

        return {
            "total_trades": total_filled,
            "total_orders": total_orders,
            "win_rate": win_rate,
            "last_signal": last_signal,
            "account_connected": alpaca_account.is_connected(),
            "current_positions": len(current_positions),
        }

    except Exception as e:
        print(f"Error getting real strategy metrics: {e}")
        return {
            "total_trades": 0,
            "total_orders": 0,
            "win_rate": None,
            "last_signal": "N/A",
            "account_connected": False,
            "current_positions": 0,
        }


def create_strategy_monitor():
    """Create the strategy monitoring section with real Alpaca data"""
    try:
        # Get real strategy metrics from Alpaca
        metrics = get_real_strategy_metrics()

        # Determine status and styling based on real data
        if not metrics["account_connected"]:
            status = "DISCONNECTED"
            status_class = "bg-warning"
            last_signal = "N/A"
            last_signal_class = "bg-secondary"
            win_rate = "N/A"
            total_trades = "N/A"
        else:
            status = "ACTIVE"
            status_class = "bg-success"

            # Get last signal
            last_signal = metrics["last_signal"]
            if last_signal == "BUY":
                last_signal_class = "bg-primary"
            elif last_signal == "SELL":
                last_signal_class = "bg-danger"
            else:
                last_signal_class = "bg-secondary"

            # Get real trade metrics
            total_trades = metrics["total_trades"]
            if metrics["win_rate"] is not None:
                win_rate = f"{metrics['win_rate']:.0f}%"
            else:
                win_rate = "N/A"

        return html.Div(
            [
                # Golden Cross Strategy Status
                html.Div(
                    [
                        html.Span(
                            className=(
                                "strategy-indicator active"
                                if metrics["account_connected"]
                                else "strategy-indicator inactive"
                            )
                        ),
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
                                                    status,
                                                    className=f"badge {status_class}",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Small(
                                                    "Last Signal: ",
                                                    className="text-muted",
                                                ),
                                                html.Span(
                                                    last_signal,
                                                    className=f"badge {last_signal_class}",
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
                                                    win_rate,
                                                    style=(
                                                        {"color": "var(--profit-color)"}
                                                        if win_rate != "N/A"
                                                        else {}
                                                    ),
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Small(
                                                    "Total Trades: ",
                                                    className="text-muted",
                                                ),
                                                html.Span(str(total_trades)),
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

    except Exception as e:
        print(f"Error creating strategy monitor: {e}")
        # Fallback to basic display
        return html.Div(
            [
                html.Div(
                    [
                        html.Span(className="strategy-indicator inactive"),
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
                                                    "ERROR", className="badge bg-danger"
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Small(
                                                    "Last Signal: ",
                                                    className="text-muted",
                                                ),
                                                html.Span(
                                                    "N/A",
                                                    className="badge bg-secondary",
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
                                                html.Span("N/A"),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Small(
                                                    "Total Trades: ",
                                                    className="text-muted",
                                                ),
                                                html.Span("N/A"),
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
            ]
        )


def create_portfolio_chart():
    """Create the portfolio performance chart"""
    try:
        # Get real portfolio history from Alpaca
        from dashboard.data.live_data import LiveDataManager

        live_data = LiveDataManager()
        history_df = live_data.get_portfolio_history(days=30)

        if history_df.empty:
            # Fallback to simple chart if no data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D"
            )
            values = [100000] * len(dates)
        else:
            dates = history_df["date"]
            values = history_df["portfolio_value"]

        fig = go.Figure()

        # Portfolio line
        fig.add_trace(
            go.Scatter(
                x=dates,
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
    """Create the recent activity feed with real data"""
    # Get real account data and signals
    now = datetime.now()

    # Get real account status
    account_summary = alpaca_account.get_account_summary()
    positions = alpaca_account.get_positions()
    recent_orders = alpaca_account.get_recent_orders(limit=5)

    activities = []

    # Add account status update
    if account_summary.get("note"):
        activities.append(
            {
                "time": now.strftime("%H:%M:%S"),
                "description": f"Account Status: {account_summary.get('note', 'Connected')}",
                "type": "system",
            }
        )
    else:
        activities.append(
            {
                "time": now.strftime("%H:%M:%S"),
                "description": f"Account updated: ${account_summary.get('total_value', 0):,.2f} total value",
                "type": "system",
            }
        )

    # Add recent orders (real trades)
    if recent_orders:
        for order in recent_orders:
            if order.get("status") == "filled":
                activities.append(
                    {
                        "time": (
                            order["timestamp"].strftime("%H:%M:%S")
                            if hasattr(order["timestamp"], "strftime")
                            else str(order["timestamp"])
                        ),
                        "description": f"Executed {order['action']}: {order['quantity']} shares of {order['symbol']} at ${order['price']:.2f}",
                        "type": "trade",
                    }
                )
    else:
        # No recent orders - show appropriate message
        activities.append(
            {
                "time": (now - timedelta(minutes=5)).strftime("%H:%M:%S"),
                "description": "No recent trading activity - account is ready for new signals",
                "type": "system",
            }
        )

    # Add position updates (only if there are real positions)
    if positions:
        for position in positions:
            if position.get("unrealized_pnl", 0) != 0:
                pnl_sign = "+" if position["unrealized_pnl"] > 0 else ""
                activities.append(
                    {
                        "time": (now - timedelta(minutes=30)).strftime("%H:%M:%S"),
                        "description": f"{position['symbol']} position {pnl_sign}{position['pnl_percent']:.2f}% (${pnl_sign}{position['unrealized_pnl']:.0f} unrealized)",
                        "type": "performance",
                    }
                )
    else:
        # No positions - show appropriate message
        activities.append(
            {
                "time": (now - timedelta(minutes=15)).strftime("%H:%M:%S"),
                "description": "No open positions - ready for new trading opportunities",
                "type": "system",
            }
        )

    # Add strategy signal placeholder (will be populated by analysis)
    if not activities or len(activities) < 3:
        activities.append(
            {
                "time": (now - timedelta(minutes=10)).strftime("%H:%M:%S"),
                "description": "Ready for strategy analysis - run analysis to see signals",
                "type": "signal",
            }
        )

    # Add account connection status
    if not alpaca_account.is_connected():
        activities.append(
            {
                "time": (now - timedelta(minutes=1)).strftime("%H:%M:%S"),
                "description": "⚠️ Alpaca account not connected - check API credentials",
                "type": "system",
            }
        )

    # Sort activities by time (most recent first)
    activities.sort(key=lambda x: x["time"], reverse=True)

    # Create activity feed items
    activity_items = []
    for activity in activities[:10]:  # Show last 10 activities
        icon = get_activity_icon(activity["type"])
        activity_items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className=f"fas {icon} me-2"),
                            html.Span(activity["time"], className="text-muted small"),
                        ],
                        className="d-flex justify-content-between align-items-center mb-1",
                    ),
                    html.P(
                        activity["description"],
                        className="mb-0 small",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
                className="activity-item p-2 border-bottom",
            )
        )

    return html.Div(
        [
            html.H5(
                [
                    html.I(className="fas fa-history me-2"),
                    "Recent Activity",
                ],
                className="chart-title",
            ),
            html.Div(activity_items, className="activity-feed"),
        ],
        className="chart-container",
    )


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


# Register analysis callbacks
register_analysis_callbacks(app)


if __name__ == "__main__":
    print("🚀 Starting AlgoTrading Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    app.run_server(debug=True, host="127.0.0.1", port=8050)
