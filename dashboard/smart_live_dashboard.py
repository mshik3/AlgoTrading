"""
Smart Live Trading Dashboard - Real Data with Caching
Starts with $0 (no investments) and uses caching to avoid rate limits
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
import time
from threading import Lock

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual trading system components
try:
    from execution.paper import PaperTradingSimulator

    PAPER_TRADING_AVAILABLE = True
    paper_trader = PaperTradingSimulator(
        initial_capital=0
    )  # START WITH $0 - NO INVESTMENTS YET
except ImportError as e:
    print(f"Warning: Paper trading not available: {e}")
    PAPER_TRADING_AVAILABLE = False
    paper_trader = None

# Symbols from our Golden Cross strategy
MONITORED_SYMBOLS = ["SPY", "QQQ", "VTI", "AAPL", "MSFT", "GOOGL"]

# Price cache to avoid rate limits
price_cache = {}
cache_lock = Lock()
CACHE_DURATION = 60  # Cache prices for 60 seconds

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    ],
    title="AlgoTrading Smart Dashboard",
)


def get_cached_prices(symbols):
    """Get prices with caching to avoid rate limits"""
    with cache_lock:
        now = time.time()

        # Check if we have valid cached data
        if (
            "timestamp" in price_cache
            and (now - price_cache["timestamp"]) < CACHE_DURATION
        ):
            print(
                f"Using cached prices (cached {int(now - price_cache['timestamp'])}s ago)"
            )
            return price_cache.get("data", {})

        print("Fetching fresh prices...")

        try:
            data = {}

            # Get all tickers at once to reduce API calls
            tickers_str = " ".join(symbols)
            tickers = yf.Tickers(tickers_str)

            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    # Try to get recent data - use 5d period to ensure we get data
                    hist = ticker.history(period="5d")

                    if not hist.empty:
                        current_price = hist["Close"].iloc[-1]

                        # Try to get previous close for change calculation
                        if len(hist) >= 2:
                            prev_close = hist["Close"].iloc[-2]
                        else:
                            prev_close = current_price

                        change = current_price - prev_close
                        change_pct = (
                            (change / prev_close) * 100 if prev_close > 0 else 0
                        )

                        data[symbol] = {
                            "price": round(current_price, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 2),
                            "volume": (
                                int(hist["Volume"].iloc[-1])
                                if "Volume" in hist.columns and len(hist) > 0
                                else 0
                            ),
                        }
                    else:
                        # No data available
                        data[symbol] = {
                            "price": 0.0,
                            "change": 0.0,
                            "change_pct": 0.0,
                            "volume": 0,
                        }
                        print(f"No historical data for {symbol}")

                except Exception as e:
                    print(f"Error getting data for {symbol}: {e}")
                    data[symbol] = {
                        "price": 0.0,
                        "change": 0.0,
                        "change_pct": 0.0,
                        "volume": 0,
                    }

            # Cache the results
            price_cache["data"] = data
            price_cache["timestamp"] = now

            print(
                f"Successfully fetched prices for {len([s for s in data if data[s]['price'] > 0])} symbols"
            )
            return data

        except Exception as e:
            print(f"Error fetching prices: {e}")
            # Return cached data if available, otherwise return zeros
            if "data" in price_cache:
                print("Using stale cached data due to error")
                return price_cache["data"]
            else:
                return {
                    symbol: {
                        "price": 0.0,
                        "change": 0.0,
                        "change_pct": 0.0,
                        "volume": 0,
                    }
                    for symbol in symbols
                }


def get_portfolio_data():
    """Get real portfolio data - starts with $0 since no investments made yet"""
    if not PAPER_TRADING_AVAILABLE or not paper_trader:
        return {
            "total_value": 0.0,  # NO INVESTMENTS YET
            "cash": 0.0,  # NO CASH DEPOSITED YET
            "total_pnl": 0.0,
            "positions_value": 0.0,
            "positions": [],
            "trades": [],
        }

    try:
        portfolio_summary = paper_trader.get_portfolio_summary()
        positions = paper_trader.get_positions_summary()
        trades = paper_trader.get_trades_summary()

        return {
            "total_value": portfolio_summary.get("total_value", 0.0),  # Real value
            "cash": portfolio_summary.get("cash", 0.0),  # Real cash
            "total_pnl": portfolio_summary.get("total_pnl", 0.0),
            "positions_value": portfolio_summary.get("positions_value", 0.0),
            "positions": positions,
            "trades": trades[:10] if trades else [],
        }
    except Exception as e:
        print(f"Error getting portfolio data: {e}")
        return {
            "total_value": 0.0,  # NO INVESTMENTS YET
            "cash": 0.0,  # NO CASH YET
            "total_pnl": 0.0,
            "positions_value": 0.0,
            "positions": [],
            "trades": [],
        }


# App layout
app.layout = dbc.Container(
    [
        dcc.Interval(
            id="dashboard-interval",
            interval=120 * 1000,  # 2 minutes to avoid rate limits
            n_intervals=0,
        ),
        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "⚡ AlgoTrading Smart Dashboard",
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
        # Status Alert for No Investments
        html.Div(
            [
                dbc.Alert(
                    [
                        html.I(
                            className="fas fa-info-circle",
                            style={"marginRight": "10px"},
                        ),
                        html.Strong("No Investments Yet"),
                        " - Your portfolio shows $0.00 because no funds have been deposited for trading.",
                    ],
                    color="info",
                    style={"marginBottom": "20px"},
                )
            ],
            id="no-investment-alert",
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
                                            "No funds deposited",
                                            id="cash-status",
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
                                            "Market Status",
                                            style={
                                                "color": "#868b95",
                                                "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="market-status",
                                            style={
                                                "color": "#d1d4dc",
                                                "fontSize": "18px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="price-cache-status",
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
                        # Strategy Status
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
                                html.Div(id="strategy-status"),
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
                                "marginBottom": "20px",
                            },
                        ),
                        # System Status
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.I(
                                            className="fas fa-server",
                                            style={"marginRight": "10px"},
                                        ),
                                        "System Status",
                                    ],
                                    style={
                                        "color": "#d1d4dc",
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(id="system-status"),
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
        Output("cash-status", "children"),
        Output("total-pnl", "children"),
        Output("pnl-change", "children"),
        Output("market-status", "children"),
        Output("price-cache-status", "children"),
        Output("positions-table", "children"),
        Output("live-prices", "children"),
        Output("strategy-status", "children"),
        Output("system-status", "children"),
        Output("no-investment-alert", "style"),
    ],
    [Input("dashboard-interval", "n_intervals")],
)
def update_dashboard(n_intervals):
    """Update dashboard with real cached data"""

    # Update timestamp
    current_time = f"Last Updated: {datetime.now().strftime('%H:%M:%S')} (Cached Data)"

    # Get real portfolio data
    portfolio_data = get_portfolio_data()

    # Get cached market prices
    live_prices = get_cached_prices(MONITORED_SYMBOLS)

    # Portfolio values
    total_value = portfolio_data["total_value"]
    cash = portfolio_data["cash"]
    total_pnl = portfolio_data["total_pnl"]
    positions = portfolio_data["positions"]

    # Format values
    total_value_str = f"${total_value:,.2f}"
    cash_str = f"${cash:,.2f}"
    total_pnl_str = f"${total_pnl:+,.2f}" if total_pnl != 0 else "$0.00"

    # Cash status
    cash_status = "No funds deposited" if cash == 0 else "Ready to Trade"

    # Changes and colors
    total_change_str = f"${total_pnl:+,.2f}"
    total_change_style = {
        "fontSize": "12px",
        "fontWeight": "500",
        "color": (
            "#26a69a" if total_pnl > 0 else "#ef5350" if total_pnl < 0 else "#868b95"
        ),
    }

    pnl_change_text = (
        f"{(total_pnl/100)*100:+.2f}%"
        if total_value > 0 and total_pnl != 0
        else "0.00%"
    )

    # Market status
    now = datetime.now()
    if 9 <= now.hour < 16:  # Rough market hours
        market_status = "🟢 OPEN"
    else:
        market_status = "🔴 CLOSED"

    # Cache status
    cache_age = time.time() - price_cache.get("timestamp", 0)
    if cache_age < CACHE_DURATION:
        cache_status = f"Fresh ({int(cache_age)}s ago)"
    else:
        cache_status = "Refreshing..."

    # Hide alert if we have investments
    alert_style = {"display": "none"} if total_value > 0 else {"marginBottom": "20px"}

    # Build positions table
    if positions:
        positions_df = pd.DataFrame(positions)
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
                    "name": "P&L",
                    "id": "unrealized_pnl",
                    "type": "numeric",
                    "format": {"specifier": "+,.2f"},
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

        if price > 0:
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
                                            else (
                                                "#ef5350"
                                                if change_pct < 0
                                                else "#868b95"
                                            )
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
        else:
            price_items.append(
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
                            " - Data unavailable",
                            style={
                                "color": "#ef5350",
                                "fontSize": "12px",
                                "marginLeft": "10px",
                            },
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
                            "color": "#ffa726",
                            "marginRight": "8px",
                            "fontSize": "16px",
                        },
                    ),
                    html.Span(
                        "Golden Cross Strategy",
                        style={"color": "#d1d4dc", "fontWeight": "600"},
                    ),
                    html.Div(
                        "Awaiting funding to start trading",
                        style={
                            "color": "#868b95",
                            "fontSize": "12px",
                            "marginTop": "5px",
                        },
                    ),
                ],
                style={"marginBottom": "15px"},
            ),
        ]
    )

    # System status
    working_prices = len([p for p in live_prices.values() if p["price"] > 0])
    system_status = html.Div(
        [
            html.Div(
                [
                    html.Small("Paper Trading: ", style={"color": "#868b95"}),
                    html.Span(
                        "CONNECTED" if PAPER_TRADING_AVAILABLE else "OFFLINE",
                        className=(
                            "badge bg-success"
                            if PAPER_TRADING_AVAILABLE
                            else "badge bg-secondary"
                        ),
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Small(
                        f"Price Data: {working_prices}/{len(MONITORED_SYMBOLS)} symbols",
                        style={"color": "#868b95"},
                    )
                ]
            ),
        ]
    )

    return (
        current_time,
        total_value_str,
        total_change_str,
        total_change_style,
        cash_str,
        cash_status,
        total_pnl_str,
        pnl_change_text,
        market_status,
        cache_status,
        positions_table,
        live_prices_display,
        strategy_status,
        system_status,
        alert_style,
    )


if __name__ == "__main__":
    print("🚀 Starting SMART AlgoTrading Dashboard")
    print("💰 Portfolio: $0.00 (No investments yet)")
    print("📊 Visit: http://127.0.0.1:8050")
    print("📈 Monitoring (cached):", ", ".join(MONITORED_SYMBOLS))
    print("🔄 Refresh: Every 2 minutes (cached 1 min)")
    print("🔗 Paper Trading:", "CONNECTED" if PAPER_TRADING_AVAILABLE else "OFFLINE")
    app.run_server(debug=True, host="127.0.0.1", port=8050)
