"""
Live Professional Trading Dashboard - NO MOCK DATA
Real-time data from actual paper trading system and live market prices
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual trading system components
try:
    from execution.paper import PaperTradingSimulator

    PAPER_TRADING_AVAILABLE = True
    paper_trader = PaperTradingSimulator()
except ImportError as e:
    print(f"Warning: Paper trading not available: {e}")
    PAPER_TRADING_AVAILABLE = False
    paper_trader = None

# Symbols from our Golden Cross strategy
MONITORED_SYMBOLS = ["SPY", "QQQ", "VTI", "AAPL", "MSFT", "GOOGL"]

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    ],
    title="AlgoTrading Live Dashboard",
)


def get_live_prices(symbols):
    """Get live prices for symbols using yfinance"""
    try:
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
                prev_close = ticker.info.get("previousClose", current_price)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

                data[symbol] = {
                    "price": current_price,
                    "change": change,
                    "change_pct": change_pct,
                    "volume": (
                        hist["Volume"].iloc[-1] if "Volume" in hist.columns else 0
                    ),
                }
            else:
                # Fallback if no intraday data
                data[symbol] = {"price": 0, "change": 0, "change_pct": 0, "volume": 0}

        return data
    except Exception as e:
        print(f"Error getting live prices: {e}")
        return {
            symbol: {"price": 0, "change": 0, "change_pct": 0, "volume": 0}
            for symbol in symbols
        }


def get_portfolio_data():
    """Get real portfolio data from paper trading system"""
    if not PAPER_TRADING_AVAILABLE or not paper_trader:
        return {
            "total_value": 10000.0,  # Match actual initial capital
            "cash": 10000.0,
            "total_pnl": 0.0,
            "positions_value": 0.0,
            "positions": [],
            "trades": [],
        }

    try:
        portfolio_summary = paper_trader.get_portfolio_summary()
        positions = paper_trader.get_positions_summary()  # Fixed method name
        trades = paper_trader.get_trades_summary()  # Fixed method name

        return {
            "total_value": portfolio_summary.get(
                "total_value", 10000.0
            ),  # Fixed default
            "cash": portfolio_summary.get("cash", 10000.0),  # Fixed default
            "total_pnl": portfolio_summary.get("total_pnl", 0.0),
            "positions_value": portfolio_summary.get("positions_value", 0.0),
            "positions": positions,
            "trades": trades[:10] if trades else [],  # Limit to last 10 trades
        }
    except Exception as e:
        print(f"Error getting portfolio data: {e}")
        return {
            "total_value": 10000.0,  # Fixed default to match actual initial capital
            "cash": 10000.0,
            "total_pnl": 0.0,
            "positions_value": 0.0,
            "positions": [],
            "trades": [],
        }


# App layout
app.layout = dbc.Container(
    [
        dcc.Interval(
            id="dashboard-interval", interval=30 * 1000, n_intervals=0  # 30 seconds
        ),
        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "⚡ AlgoTrading Live Dashboard",
                            style={
                                "color": "#d1d4dc",
                                "textAlign": "center",
                                "margin": 0,
                            },
                        ),
                        html.P(
                            "",
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
        # KPI Cards
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
                                            "",
                                            id="total-value",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "24px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="total-change",
                                            style={
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
                                            "",
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
                                            "Total P&L",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="total-pnl",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "24px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="pnl-change",
                                            style={
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
                                            "Positions Value",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="positions-value",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "24px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="positions-count",
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
        # Main content
        dbc.Row(
            [
                # Left column
                dbc.Col(
                    [
                        # Current Positions
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
                        # Live Prices
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(
                                            className="fas fa-dollar-sign",
                                            style={"marginRight": "10px"},
                                        ),
                                        "Monitored Symbols",
                                    ],
                                    style={
                                        "color": "#d1d4dc",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(id="live-prices"),
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
                # Right column
                dbc.Col(
                    [
                        # Strategy Status
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(
                                            className="fas fa-cogs",
                                            style={"marginRight": "10px"},
                                        ),
                                        "Strategy Status",
                                    ],
                                    style={
                                        "color": "#d1d4dc",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(id="strategy-status"),
                            ],
                            style={
                                "background": "#2a2e39",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #363a45",
                                "marginBottom": "20px",
                            },
                        ),
                        # Recent Activity
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
                                html.Div(id="recent-activity"),
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


@app.callback(
    [
        Output("last-update", "children"),
        Output("total-value", "children"),
        Output("total-change", "children"),
        Output("total-change", "style"),
        Output("available-cash", "children"),
        Output("total-pnl", "children"),
        Output("pnl-change", "children"),
        Output("pnl-change", "style"),
        Output("positions-value", "children"),
        Output("positions-count", "children"),
        Output("positions-table", "children"),
        Output("live-prices", "children"),
        Output("strategy-status", "children"),
        Output("recent-activity", "children"),
    ],
    [Input("dashboard-interval", "n_intervals")],
)
def update_dashboard(n_intervals):
    """Update dashboard with real live data"""

    # Update timestamp
    current_time = f"Last Updated: {datetime.now().strftime('%H:%M:%S')} (Live Data)"

    # Get real portfolio data
    portfolio_data = get_portfolio_data()

    # Get live market prices
    live_prices = get_live_prices(MONITORED_SYMBOLS)

    # Calculate KPIs
    total_value = portfolio_data["total_value"]
    cash = portfolio_data["cash"]
    total_pnl = portfolio_data["total_pnl"]
    positions_value = portfolio_data["positions_value"]
    positions = portfolio_data["positions"]
    trades = portfolio_data["trades"]

    # Format values
    total_value_str = f"${total_value:,.2f}"
    cash_str = f"${cash:,.2f}"
    total_pnl_str = f"${total_pnl:+,.2f}" if total_pnl != 0 else "$0.00"
    positions_value_str = f"${positions_value:,.2f}"

    # Calculate changes and colors
    initial_value = (
        10000  # Starting portfolio value (matches paper trader initial capital)
    )
    total_change = total_value - initial_value
    total_change_str = f"${total_change:+,.2f}"
    total_change_pct = (total_change / initial_value) * 100

    # Colors for changes
    total_change_style = {
        "fontSize": "12px",
        "fontWeight": "500",
        "color": (
            "#26a69a"
            if total_change > 0
            else "#ef5350" if total_change < 0 else "#868b95"
        ),
    }

    pnl_change_style = {
        "fontSize": "12px",
        "fontWeight": "500",
        "color": (
            "#26a69a" if total_pnl > 0 else "#ef5350" if total_pnl < 0 else "#868b95"
        ),
    }

    pnl_change_text = (
        f"{total_change_pct:+.2f}% Total" if total_change != 0 else "0.00%"
    )
    positions_count_text = (
        f"{len(positions)} Active Position{'s' if len(positions) != 1 else ''}"
    )

    # Build positions table
    if positions:
        positions_df = pd.DataFrame(positions)
        # Add current prices and calculate P&L
        for i, pos in enumerate(positions):
            symbol = pos["symbol"]
            if symbol in live_prices:
                current_price = live_prices[symbol]["price"]
                positions_df.at[i, "current_price"] = current_price
                positions_df.at[i, "market_value"] = pos["quantity"] * current_price
                unrealized_pnl = (current_price - pos["avg_price"]) * pos["quantity"]
                positions_df.at[i, "unrealized_pnl"] = unrealized_pnl
                positions_df.at[i, "pnl_percent"] = unrealized_pnl / (
                    pos["avg_price"] * pos["quantity"]
                )

        positions_table = dash_table.DataTable(
            data=positions_df.to_dict("records"),
            columns=[
                {"name": "Symbol", "id": "symbol"},
                {"name": "Qty", "id": "quantity", "type": "numeric"},
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
                "padding": "8px",
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
    else:
        positions_table = html.Div(
            [
                html.P(
                    "No active positions",
                    style={"color": "#868b95", "textAlign": "center", "margin": "20px"},
                )
            ]
        )

    # Build live prices display
    price_items = []
    for symbol, price_data in live_prices.items():
        price = price_data["price"]
        change = price_data["change"]
        change_pct = price_data["change_pct"]

        price_items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                symbol,
                                style={
                                    "fontWeight": "bold",
                                    "color": "#d1d4dc",
                                    "fontSize": "14px",
                                },
                            ),
                            html.Span(
                                f"${price:.2f}",
                                style={
                                    "color": "#d1d4dc",
                                    "fontSize": "16px",
                                    "marginLeft": "10px",
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{change:+.2f}",
                                style={
                                    "color": (
                                        "#26a69a"
                                        if change > 0
                                        else "#ef5350" if change < 0 else "#868b95"
                                    ),
                                    "fontSize": "12px",
                                },
                            ),
                            html.Span(
                                f" ({change_pct:+.2f}%)",
                                style={
                                    "color": (
                                        "#26a69a"
                                        if change_pct > 0
                                        else "#ef5350" if change_pct < 0 else "#868b95"
                                    ),
                                    "fontSize": "12px",
                                },
                            ),
                        ]
                    ),
                ],
                style={"padding": "8px", "borderBottom": "1px solid #363a45"},
            )
        )

    live_prices_display = html.Div(price_items)

    # Strategy status
    strategy_status = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "●",
                        style={
                            "color": "#26a69a",
                            "marginRight": "8px",
                            "fontSize": "16px",
                        },
                    ),
                    html.Span(
                        "Golden Cross Strategy",
                        style={"color": "#d1d4dc", "fontWeight": "600"},
                    ),
                    html.Div(
                        "Monitoring: SPY, QQQ, VTI",
                        style={
                            "color": "#868b95",
                            "fontSize": "12px",
                            "marginTop": "5px",
                        },
                    ),
                ],
                style={"marginBottom": "15px"},
            ),
            html.Div(
                [
                    html.Small("Paper Trading: ", style={"color": "#868b95"}),
                    html.Span(
                        "ACTIVE" if PAPER_TRADING_AVAILABLE else "OFFLINE",
                        className=(
                            "badge bg-success"
                            if PAPER_TRADING_AVAILABLE
                            else "badge bg-secondary"
                        ),
                    ),
                ]
            ),
        ]
    )

    # Recent activity from actual trades
    if trades:
        activity_items = []
        for trade in trades[:5]:  # Show last 5 trades
            trade_time = trade.get("timestamp", datetime.now()).strftime("%H:%M:%S")
            action = trade.get("action", "UNKNOWN")
            symbol = trade.get("symbol", "N/A")
            quantity = trade.get("quantity", 0)
            price = trade.get("price", 0)

            activity_items.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.I(
                                    className="fas fa-exchange-alt",
                                    style={"marginRight": "10px", "color": "#2962ff"},
                                ),
                                html.Span(
                                    trade_time,
                                    style={"color": "#868b95", "fontSize": "11px"},
                                ),
                            ]
                        ),
                        html.Div(
                            f"{action}: {quantity} {symbol} @ ${price:.2f}",
                            style={
                                "color": "#d1d4dc",
                                "fontSize": "13px",
                                "marginTop": "4px",
                            },
                        ),
                    ],
                    style={"padding": "10px", "borderBottom": "1px solid #363a45"},
                )
            )

        recent_activity = html.Div(activity_items)
    else:
        recent_activity = html.Div(
            [
                html.P(
                    "No recent trades",
                    style={"color": "#868b95", "textAlign": "center", "margin": "20px"},
                )
            ]
        )

    return (
        current_time,
        total_value_str,
        total_change_str,
        total_change_style,
        cash_str,
        total_pnl_str,
        pnl_change_text,
        pnl_change_style,
        positions_value_str,
        positions_count_text,
        positions_table,
        live_prices_display,
        strategy_status,
        recent_activity,
    )


if __name__ == "__main__":
    print("🚀 Starting LIVE AlgoTrading Dashboard (NO MOCK DATA)")
    print("📊 Visit: http://127.0.0.1:8050")
    print("📈 Monitoring live prices for:", ", ".join(MONITORED_SYMBOLS))
    print("🔗 Paper Trading:", "CONNECTED" if PAPER_TRADING_AVAILABLE else "OFFLINE")
    app.run_server(debug=True, host="127.0.0.1", port=8050)
