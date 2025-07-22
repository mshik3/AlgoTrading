"""
Modern Portfolio Optimization Dashboard Components.

This module provides Dash components for the modern portfolio optimization
features, including Black-Litterman optimization and risk metrics.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json

# Import the modern portfolio service
from dashboard.services.modern_portfolio_service import get_modern_portfolio_service


def create_optimization_card():
    """Create the main optimization card component."""
    return html.Div(
        [
            html.H4("🎯 Modern Portfolio Optimization", className="card-title"),
            html.Div(
                [
                    html.Label("Symbols (comma-separated):"),
                    dcc.Input(
                        id="optimization-symbols",
                        type="text",
                        value="AAPL,MSFT,GOOGL,AMZN,TSLA",
                        placeholder="Enter symbols...",
                        className="form-control mb-2",
                    ),
                    html.Label("Portfolio Value ($):"),
                    dcc.Input(
                        id="portfolio-value",
                        type="number",
                        value=100000,
                        className="form-control mb-2",
                    ),
                    html.Label("Optimization Method:"),
                    dcc.Dropdown(
                        id="optimization-method",
                        options=[
                            {"label": "Maximum Sharpe Ratio", "value": "max_sharpe"},
                            {"label": "Minimum Volatility", "value": "min_volatility"},
                            {"label": "Black-Litterman", "value": "black_litterman"},
                        ],
                        value="max_sharpe",
                        className="mb-2",
                    ),
                    html.Button(
                        "Optimize Portfolio",
                        id="optimize-button",
                        className="btn btn-primary btn-block",
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="optimization-results"),
        ],
        className="card",
    )


def create_black_litterman_card():
    """Create Black-Litterman optimization card."""
    return html.Div(
        [
            html.H4("🧠 Black-Litterman Optimization", className="card-title"),
            html.Div(
                [
                    html.Label("Investor Views (Expected Returns):"),
                    dcc.Textarea(
                        id="bl-views",
                        placeholder='{"AAPL": 0.15, "TSLA": -0.05}',
                        className="form-control mb-2",
                        rows=3,
                    ),
                    html.Button(
                        "Run Black-Litterman",
                        id="bl-optimize-button",
                        className="btn btn-success btn-block",
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="bl-results"),
        ],
        className="card",
    )


def create_risk_metrics_card():
    """Create risk metrics card."""
    return html.Div(
        [
            html.H4("📊 Risk Metrics", className="card-title"),
            html.Div(
                [
                    html.Button(
                        "Calculate Risk Metrics",
                        id="risk-metrics-button",
                        className="btn btn-info btn-block",
                    )
                ],
                className="mb-3",
            ),
            html.Div(id="risk-metrics-results"),
        ],
        className="card",
    )


def create_rebalancing_card():
    """Create rebalancing recommendations card."""
    return html.Div(
        [
            html.H4("⚖️ Rebalancing Recommendations", className="card-title"),
            html.Div(
                [
                    html.Label("Current Weights (JSON):"),
                    dcc.Textarea(
                        id="current-weights",
                        placeholder='{"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15}',
                        className="form-control mb-2",
                        rows=3,
                    ),
                    html.Button(
                        "Get Recommendations",
                        id="rebalancing-button",
                        className="btn btn-warning btn-block",
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="rebalancing-results"),
        ],
        className="card",
    )


# Callbacks for the components
@callback(
    Output("optimization-results", "children"),
    Input("optimize-button", "n_clicks"),
    State("optimization-symbols", "value"),
    State("portfolio-value", "value"),
    State("optimization-method", "value"),
    prevent_initial_call=True,
)
def run_optimization(n_clicks, symbols, portfolio_value, method):
    """Run portfolio optimization and display results."""
    if not n_clicks:
        return ""

    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

        # Get service
        service = get_modern_portfolio_service()

        if method == "black_litterman":
            # For Black-Litterman, we'll need views
            return html.Div(
                [
                    html.H5("Black-Litterman Optimization"),
                    html.P("Please use the Black-Litterman card to specify views."),
                ]
            )

        # Run optimization
        result = service.get_optimization_results(
            symbols=symbol_list, method=method, portfolio_value=portfolio_value
        )

        if not result.get("success"):
            return html.Div(
                [
                    html.H5("❌ Optimization Failed"),
                    html.P(f"Error: {result.get('error', 'Unknown error')}"),
                ]
            )

        # Create results display
        weights = result["weights"]
        weights_df = pd.DataFrame(list(weights.items()), columns=["Symbol", "Weight"])

        return html.Div(
            [
                html.H5("✅ Optimization Results"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H6("Performance Metrics"),
                                html.P(
                                    f"Expected Return: {result['expected_return']:.1%}"
                                ),
                                html.P(f"Volatility: {result['volatility']:.1%}"),
                                html.P(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}"),
                            ],
                            className="col-md-6",
                        ),
                        html.Div(
                            [
                                html.H6("Portfolio Allocation"),
                                dcc.Graph(
                                    figure=px.pie(
                                        weights_df,
                                        values="Weight",
                                        names="Symbol",
                                        title="Optimal Portfolio Weights",
                                    )
                                ),
                            ],
                            className="col-md-6",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.H6("Discrete Allocation"),
                        html.P(f"Portfolio Value: ${portfolio_value:,}"),
                        html.P(f"Leftover Cash: ${result.get('leftover_cash', 0):.2f}"),
                        html.Div(
                            [
                                html.P(f"{symbol}: {shares} shares")
                                for symbol, shares in result.get(
                                    "allocation", {}
                                ).items()
                            ]
                        ),
                    ]
                ),
            ]
        )

    except Exception as e:
        return html.Div([html.H5("❌ Error"), html.P(f"An error occurred: {str(e)}")])


@callback(
    Output("bl-results", "children"),
    Input("bl-optimize-button", "n_clicks"),
    State("optimization-symbols", "value"),
    State("portfolio-value", "value"),
    State("bl-views", "value"),
    prevent_initial_call=True,
)
def run_black_litterman(n_clicks, symbols, portfolio_value, views_text):
    """Run Black-Litterman optimization."""
    if not n_clicks:
        return ""

    try:
        # Parse symbols and views
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        views = json.loads(views_text) if views_text else {}

        # Get service
        service = get_modern_portfolio_service()

        # Run Black-Litterman optimization
        result = service.get_black_litterman_results(
            symbols=symbol_list, views=views, portfolio_value=portfolio_value
        )

        if not result.get("success"):
            return html.Div(
                [
                    html.H5("❌ Black-Litterman Failed"),
                    html.P(f"Error: {result.get('error', 'Unknown error')}"),
                ]
            )

        # Create results display
        weights = result["weights"]
        weights_df = pd.DataFrame(list(weights.items()), columns=["Symbol", "Weight"])

        return html.Div(
            [
                html.H5("✅ Black-Litterman Results"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H6("Performance Metrics"),
                                html.P(
                                    f"Expected Return: {result['expected_return']:.1%}"
                                ),
                                html.P(f"Volatility: {result['volatility']:.1%}"),
                                html.P(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}"),
                            ],
                            className="col-md-6",
                        ),
                        html.Div(
                            [
                                html.H6("Investor Views"),
                                html.Div(
                                    [
                                        html.P(f"{symbol}: {return_val:.1%}")
                                        for symbol, return_val in views.items()
                                    ]
                                ),
                            ],
                            className="col-md-6",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.H6("Optimized Portfolio Weights"),
                        dcc.Graph(
                            figure=px.pie(
                                weights_df,
                                values="Weight",
                                names="Symbol",
                                title="Black-Litterman Portfolio Weights",
                            )
                        ),
                    ]
                ),
            ]
        )

    except Exception as e:
        return html.Div([html.H5("❌ Error"), html.P(f"An error occurred: {str(e)}")])


@callback(
    Output("risk-metrics-results", "children"),
    Input("risk-metrics-button", "n_clicks"),
    State("optimization-symbols", "value"),
    prevent_initial_call=True,
)
def calculate_risk_metrics(n_clicks, symbols):
    """Calculate and display risk metrics."""
    if not n_clicks:
        return ""

    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

        # Get service
        service = get_modern_portfolio_service()

        # Get risk metrics
        result = service.get_risk_metrics(symbols=symbol_list)

        if not result.get("success"):
            return html.Div(
                [
                    html.H5("❌ Risk Metrics Failed"),
                    html.P(f"Error: {result.get('error', 'Unknown error')}"),
                ]
            )

        metrics = result["risk_metrics"]

        return html.Div(
            [
                html.H5("📊 Risk Metrics"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H6("Performance Risk"),
                                html.P(
                                    f"Expected Return: {metrics['expected_return']:.1%}"
                                ),
                                html.P(f"Volatility: {metrics['volatility']:.1%}"),
                                html.P(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"),
                            ],
                            className="col-md-6",
                        ),
                        html.Div(
                            [
                                html.H6("Portfolio Risk"),
                                html.P(
                                    f"Portfolio Value: ${metrics['portfolio_value']:,.0f}"
                                ),
                                html.P(f"Cash Ratio: {metrics['cash_ratio']:.1%}"),
                                html.P(f"Position Count: {metrics['position_count']}"),
                                html.P(
                                    f"Concentration Risk: {metrics['concentration_risk']:.3f}"
                                ),
                            ],
                            className="col-md-6",
                        ),
                    ],
                    className="row",
                ),
            ]
        )

    except Exception as e:
        return html.Div([html.H5("❌ Error"), html.P(f"An error occurred: {str(e)}")])


@callback(
    Output("rebalancing-results", "children"),
    Input("rebalancing-button", "n_clicks"),
    State("optimization-symbols", "value"),
    State("current-weights", "value"),
    prevent_initial_call=True,
)
def get_rebalancing_recommendations(n_clicks, symbols, current_weights_text):
    """Get rebalancing recommendations."""
    if not n_clicks:
        return ""

    try:
        # Parse symbols and current weights
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        current_weights = (
            json.loads(current_weights_text) if current_weights_text else {}
        )

        # Get service
        service = get_modern_portfolio_service()

        # Get rebalancing recommendations
        result = service.get_rebalancing_recommendations(
            symbols=symbol_list, current_weights=current_weights
        )

        if not result.get("success"):
            return html.Div(
                [
                    html.H5("❌ Rebalancing Failed"),
                    html.P(f"Error: {result.get('error', 'Unknown error')}"),
                ]
            )

        trades = result["rebalancing_trades"]

        return html.Div(
            [
                html.H5("⚖️ Rebalancing Recommendations"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H6("Recommended Trades"),
                                (
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong(f"{symbol}: "),
                                                    html.Span(
                                                        f"{trade['action'].upper()} {abs(trade['weight_diff']):.1%}"
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        f"Current: {trade['current_weight']:.1%} → Target: {trade['target_weight']:.1%}"
                                                    ),
                                                ],
                                                className="mb-2",
                                            )
                                            for symbol, trade in trades.items()
                                        ]
                                    )
                                    if trades
                                    else html.P("No rebalancing needed")
                                ),
                            ],
                            className="col-md-12",
                        )
                    ],
                    className="row",
                ),
            ]
        )

    except Exception as e:
        return html.Div([html.H5("❌ Error"), html.P(f"An error occurred: {str(e)}")])


def create_modern_optimization_layout():
    """Create the complete modern optimization layout."""
    return html.Div(
        [
            html.H3("🚀 Modern Portfolio Optimization", className="mb-4"),
            html.Div(
                [
                    html.Div(
                        [create_optimization_card(), create_black_litterman_card()],
                        className="col-md-6",
                    ),
                    html.Div(
                        [create_risk_metrics_card(), create_rebalancing_card()],
                        className="col-md-6",
                    ),
                ],
                className="row",
            ),
        ]
    )
