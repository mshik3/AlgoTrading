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

# Import strategy metrics service
from dashboard.services.strategy_metrics_service import StrategyMetricsService

# Service instances (will be initialized lazily)
_alpaca_account_service = None
_strategy_metrics_service = None


def get_alpaca_account_service():
    """Get or create Alpaca account service instance."""
    global _alpaca_account_service
    if _alpaca_account_service is None:
        try:
            _alpaca_account_service = AlpacaAccountService()
            if not _alpaca_account_service.is_connected():
                raise Exception("Alpaca connection failed")
        except Exception as e:
            print(f"❌ Alpaca connection required: {e}")
            print(
                "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables"
            )
            raise
    return _alpaca_account_service


def get_strategy_metrics_service():
    """Get or create strategy metrics service instance."""
    global _strategy_metrics_service
    if _strategy_metrics_service is None:
        try:
            _strategy_metrics_service = StrategyMetricsService()
        except Exception as e:
            print(f"⚠️ Strategy metrics service initialization failed: {e}")
            return None
    return _strategy_metrics_service


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
app.external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
]

# App layout
app.layout = dbc.Container(
    [
        # Auto-refresh component
        dcc.Interval(
            id="dashboard-interval", interval=30 * 1000, n_intervals=0  # 30 seconds
        ),
        # Paper Trading Banner - SAFETY INDICATOR
        html.Div(
            [
                html.Div(
                    [
                        html.I(className="fas fa-flask me-2"),
                        html.Strong("PAPER TRADING MODE"),
                        html.I(className="fas fa-flask ms-2"),
                        html.Span(" - No Real Money Involved", className="ms-2"),
                    ],
                    className="paper-trading-banner",
                )
            ],
            className="mb-3",
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
                        # Open Orders Table
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(className="fas fa-clock me-2"),
                                        "Open Orders",
                                    ],
                                    className="chart-title",
                                ),
                                html.Div(id="open-orders-table"),
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
        Output("open-orders-table", "children"),
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

        # Get services using lazy loading
        alpaca_account = get_alpaca_account_service()
        strategy_metrics_service = get_strategy_metrics_service()

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

        # Build open orders table
        open_orders_table = create_open_orders_table()

        # Build strategy monitor with multi-strategy support
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
            open_orders_table,
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
            create_empty_open_orders_table(),
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


def create_open_orders_table():
    """Create the open orders table with real data"""
    try:
        # Get services using lazy loading
        alpaca_account = get_alpaca_account_service()

        # Get recent orders and filter for open/pending ones
        all_orders = alpaca_account.get_recent_orders(limit=20)
        open_orders = [
            order
            for order in all_orders
            if order.get("status")
            in ["pending_new", "new", "accepted", "accepted_for_bidding"]
        ]

        if not open_orders:
            return html.Div(
                [html.P("No open orders", className="text-center text-muted py-3")]
            )

        # Convert to DataFrame for table display
        df = pd.DataFrame(open_orders)

        # Convert UUID to string for JSON serialization
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)

        # Format timestamp
        if "timestamp" in df.columns:
            df["time"] = df["timestamp"].apply(
                lambda x: x.strftime("%H:%M:%S") if hasattr(x, "strftime") else str(x)
            )
        else:
            df["time"] = "N/A"

        # Remove timestamp column to avoid serialization issues
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        # Create table
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[
                {"name": "Time", "id": "time"},
                {"name": "Symbol", "id": "symbol"},
                {"name": "Side", "id": "action"},
                {
                    "name": "Quantity",
                    "id": "quantity",
                    "type": "numeric",
                    "format": {"specifier": ",.0f"},
                },
                {
                    "name": "Price",
                    "id": "price",
                    "type": "numeric",
                    "format": {"specifier": ",.2f"},
                },
                {"name": "Status", "id": "status"},
                {"name": "Type", "id": "order_type"},
            ],
            style_cell={"textAlign": "center", "padding": "8px", "fontSize": "12px"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{action} = buy"},
                    "color": "var(--profit-color)",
                    "fontWeight": "bold",
                },
                {
                    "if": {"filter_query": "{action} = sell"},
                    "color": "var(--loss-color)",
                    "fontWeight": "bold",
                },
                {
                    "if": {"filter_query": "{status} = pending_new"},
                    "backgroundColor": "rgba(255, 193, 7, 0.1)",
                },
                {
                    "if": {"filter_query": "{status} = new"},
                    "backgroundColor": "rgba(40, 167, 69, 0.1)",
                },
            ],
            css=[
                {
                    "selector": ".dash-table-container",
                    "rule": "background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border-color);",
                }
            ],
        )

    except Exception as e:
        return html.Div(
            [
                html.P(
                    f"Error loading open orders: {str(e)}",
                    className="text-center text-muted py-3",
                )
            ]
        )


def create_empty_open_orders_table():
    """Create empty open orders table for error state"""
    return html.Div(
        [
            html.P(
                "No open orders data available", className="text-center text-muted py-3"
            )
        ]
    )


def get_multi_strategy_metrics():
    """
    Get real strategy metrics for all strategies from the strategy metrics service.

    Returns:
        Dictionary with metrics for all strategies
    """
    try:
        strategy_metrics_service = get_strategy_metrics_service()
        if strategy_metrics_service is None:
            # Fallback to basic metrics if service is not available
            return {
                "golden_cross": {
                    "name": "Golden Cross Strategy",
                    "status": "ERROR",
                    "account_connected": False,
                    "total_trades": 0,
                    "win_rate": None,
                    "last_signal": "N/A",
                    "is_active": False,
                },
                "mean_reversion": {
                    "name": "Mean Reversion Strategy",
                    "status": "ERROR",
                    "account_connected": False,
                    "total_trades": 0,
                    "win_rate": None,
                    "last_signal": "N/A",
                    "is_active": False,
                },
            }

        # Get metrics for all strategies
        all_metrics = strategy_metrics_service.get_all_strategy_metrics()
        return all_metrics

    except Exception as e:
        print(f"Error getting multi-strategy metrics: {e}")
        # Return fallback metrics
        return {
            "golden_cross": {
                "name": "Golden Cross Strategy",
                "status": "ERROR",
                "account_connected": False,
                "total_trades": 0,
                "win_rate": None,
                "last_signal": "N/A",
                "is_active": False,
            },
            "mean_reversion": {
                "name": "Mean Reversion Strategy",
                "status": "ERROR",
                "account_connected": False,
                "total_trades": 0,
                "win_rate": None,
                "last_signal": "N/A",
                "is_active": False,
            },
        }


def create_strategy_monitor():
    """Create the strategy monitoring section with real multi-strategy data"""
    try:
        # Get real strategy metrics for all strategies
        all_metrics = get_multi_strategy_metrics()

        strategy_components = []

        # Create components for each strategy
        for strategy_id, metrics in all_metrics.items():
            strategy_component = _create_strategy_component(strategy_id, metrics)
            strategy_components.append(strategy_component)

        # Add ETF Rotation Strategy as "Coming Soon"
        etf_rotation_component = html.Div(
            [
                html.Span(className="strategy-indicator inactive"),
                html.Span("ETF Rotation Strategy", className="strategy-name"),
                html.Small("Coming Soon...", className="text-muted"),
            ],
            className="mb-3",
        )
        strategy_components.append(etf_rotation_component)

        return html.Div(strategy_components)

    except Exception as e:
        print(f"Error creating strategy monitor: {e}")
        # Fallback to basic display
        return _create_fallback_strategy_monitor()


def _create_strategy_component(strategy_id: str, metrics: dict) -> html.Div:
    """Create a strategy component with metrics display."""
    try:
        # Determine status and styling based on metrics
        if not metrics.get("account_connected", False):
            status = "DISCONNECTED"
            status_class = "bg-warning"
            last_signal = "N/A"
            last_signal_class = "bg-secondary"
            win_rate = "N/A"
            total_trades = "N/A"
            indicator_class = "strategy-indicator inactive"
        else:
            status = "ACTIVE" if metrics.get("is_active", False) else "INACTIVE"
            status_class = "bg-success" if status == "ACTIVE" else "bg-secondary"

            # Get last signal
            last_signal = metrics.get("last_signal", "NONE")
            if last_signal == "BUY":
                last_signal_class = "bg-primary"
            elif last_signal == "SELL":
                last_signal_class = "bg-danger"
            else:
                last_signal_class = "bg-secondary"

            # Get real trade metrics
            total_trades = metrics.get("total_trades", 0)
            win_rate_raw = metrics.get("win_rate")
            if win_rate_raw is not None:
                win_rate = f"{win_rate_raw:.0f}%"
            else:
                win_rate = "N/A"

            indicator_class = (
                "strategy-indicator active"
                if status == "ACTIVE"
                else "strategy-indicator inactive"
            )

        # Create strategy-specific additional info
        additional_info = _create_strategy_additional_info(strategy_id, metrics)

        return html.Div(
            [
                html.Span(className=indicator_class),
                html.Span(
                    metrics.get("name", "Unknown Strategy"), className="strategy-name"
                ),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Small("Status: ", className="text-muted"),
                                        html.Span(
                                            status, className=f"badge {status_class}"
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
                                            "Total Trades: ", className="text-muted"
                                        ),
                                        html.Span(str(total_trades)),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="mt-2",
                        ),
                        additional_info,
                    ]
                ),
            ],
            className="mb-3",
        )

    except Exception as e:
        print(f"Error creating strategy component for {strategy_id}: {e}")
        return _create_error_strategy_component(strategy_id)


def _create_strategy_additional_info(strategy_id: str, metrics: dict) -> html.Div:
    """Create strategy-specific additional information display."""
    try:
        strategy_specific = metrics.get("strategy_specific", {})

        if strategy_id == "mean_reversion":
            # Show mean reversion specific metrics
            avg_holding = strategy_specific.get("avg_holding_period_days")
            enhancements = strategy_specific.get("enhancements", [])

            info_items = []
            if avg_holding is not None:
                info_items.append(f"Avg Hold: {avg_holding:.1f}d")

            if enhancements:
                info_items.append(f"Features: {len(enhancements)}")

            if info_items:
                return html.Div(
                    [
                        html.Small(
                            " | ".join(info_items),
                            className="text-muted",
                            style={"fontSize": "11px"},
                        )
                    ],
                    className="mt-1",
                )

        elif strategy_id == "golden_cross":
            # Show golden cross specific metrics
            crossover_signals = strategy_specific.get("crossover_signals", 0)
            ma_periods = strategy_specific.get("ma_periods", [50, 200])

            info_items = []
            if crossover_signals > 0:
                info_items.append(f"Crossovers: {crossover_signals}")

            if ma_periods:
                info_items.append(f"MA: {ma_periods[0]}/{ma_periods[1]}")

            if info_items:
                return html.Div(
                    [
                        html.Small(
                            " | ".join(info_items),
                            className="text-muted",
                            style={"fontSize": "11px"},
                        )
                    ],
                    className="mt-1",
                )

        elif strategy_id == "dual_momentum":
            # Show dual momentum specific metrics
            current_asset = strategy_specific.get("current_asset")
            defensive_mode = strategy_specific.get("defensive_mode", False)
            qualified_count = strategy_specific.get("qualified_assets_count", 0)

            info_items = []
            if current_asset:
                info_items.append(f"Asset: {current_asset}")
            elif defensive_mode:
                info_items.append("Defensive Mode")

            if qualified_count > 0:
                info_items.append(f"Qualified: {qualified_count}")

            if info_items:
                return html.Div(
                    [
                        html.Small(
                            " | ".join(info_items),
                            className="text-muted",
                            style={"fontSize": "11px"},
                        )
                    ],
                    className="mt-1",
                )

        elif strategy_id == "sector_rotation":
            # Show sector rotation specific metrics
            top_sectors = strategy_specific.get("top_sectors", [])
            benchmark = strategy_specific.get("benchmark_symbol", "SPY")

            info_items = []
            if top_sectors:
                info_items.append(f"Top: {', '.join(top_sectors[:2])}")

            info_items.append(f"Benchmark: {benchmark}")

            if info_items:
                return html.Div(
                    [
                        html.Small(
                            " | ".join(info_items),
                            className="text-muted",
                            style={"fontSize": "11px"},
                        )
                    ],
                    className="mt-1",
                )

        return html.Div()  # Empty div if no additional info

    except Exception as e:
        print(f"Error creating additional info for {strategy_id}: {e}")
        return html.Div()


def _create_error_strategy_component(strategy_id: str) -> html.Div:
    """Create an error state strategy component."""
    return html.Div(
        [
            html.Span(className="strategy-indicator inactive"),
            html.Span(f"Strategy {strategy_id.title()}", className="strategy-name"),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Small("Status: ", className="text-muted"),
                                    html.Span("ERROR", className="badge bg-danger"),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Small("Last Signal: ", className="text-muted"),
                                    html.Span("N/A", className="badge bg-secondary"),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Small("Win Rate: ", className="text-muted"),
                                    html.Span("N/A"),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Small(
                                        "Total Trades: ", className="text-muted"
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
    )


def _create_fallback_strategy_monitor() -> html.Div:
    """Create a fallback strategy monitor when metrics service fails."""
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
                                                "Last Signal: ", className="text-muted"
                                            ),
                                            html.Span(
                                                "N/A", className="badge bg-secondary"
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
                                                "Total Trades: ", className="text-muted"
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

    # Get services using lazy loading
    alpaca_account = get_alpaca_account_service()

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

    # Add recent orders (all orders including pending)
    if recent_orders:
        for order in recent_orders:
            # Format timestamp
            if hasattr(order["timestamp"], "strftime"):
                time_str = order["timestamp"].strftime("%H:%M:%S")
            else:
                time_str = str(order["timestamp"])

            # Create description based on order status
            status = order.get("status", "unknown")
            if status == "filled":
                description = f"✅ Executed {order['action']}: {order['quantity']} shares of {order['symbol']} at ${order['price']:.2f}"
                order_type = "trade"
            elif status in ["pending_new", "new", "accepted", "accepted_for_bidding"]:
                description = f"⏳ Pending {order['action']}: {order['quantity']} shares of {order['symbol']} at ${order['price']:.2f}"
                order_type = "pending"
            elif status == "canceled":
                description = f"❌ Cancelled {order['action']}: {order['quantity']} shares of {order['symbol']} at ${order['price']:.2f}"
                order_type = "cancelled"
            else:
                description = f"📋 {order['action'].title()} order: {order['quantity']} shares of {order['symbol']} at ${order['price']:.2f} ({status})"
                order_type = "order"

            activities.append(
                {
                    "time": time_str,
                    "description": description,
                    "type": order_type,
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
        "pending": "fas fa-clock",
        "cancelled": "fas fa-times-circle",
        "order": "fas fa-file-alt",
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
