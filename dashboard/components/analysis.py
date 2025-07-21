"""
Analysis Component for Dashboard
Provides UI for running strategy analysis and viewing results.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash.dependencies
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Import the analysis service
from dashboard.services.analysis_service import DashboardAnalysisService


def create_analysis_layout():
    """
    Create the analysis layout with strategy buttons and results display.

    Returns:
        Dash layout component
    """
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H3(
                        [
                            html.I(className="fas fa-chart-line me-2"),
                            "Strategy Analysis",
                        ],
                        className="chart-title mb-3",
                    ),
                    html.P(
                        "Run individual strategies or combined analysis to identify trading opportunities",
                        className="text-muted",
                    ),
                ],
                className="mb-4",
            ),
            # Strategy Buttons Row 1 - Original Strategies
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-crosshairs me-2"),
                                    "Run Golden Cross",
                                ],
                                id="run-golden-cross-btn",
                                color="primary",
                                size="lg",
                                className="w-100 mb-2",
                            ),
                            html.Small(
                                "50-day MA crosses above 200-day MA",
                                className="text-muted",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-undo me-2"),
                                    "Run Mean Reversion",
                                ],
                                id="run-mean-reversion-btn",
                                color="success",
                                size="lg",
                                className="w-100 mb-2",
                            ),
                            html.Small(
                                "RSI oversold conditions", className="text-muted"
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-rocket me-2"),
                                    "Run Dual Momentum",
                                ],
                                id="run-dual-momentum-btn",
                                color="info",
                                size="lg",
                                className="w-100 mb-2",
                            ),
                            html.Small(
                                "Absolute & relative momentum rotation",
                                className="text-muted",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-chart-pie me-2"),
                                    "Run Sector Rotation",
                                ],
                                id="run-sector-rotation-btn",
                                color="secondary",
                                size="lg",
                                className="w-100 mb-2",
                            ),
                            html.Small(
                                "Sector strength & momentum analysis",
                                className="text-muted",
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            # Strategy Buttons Row 2 - Combined Analysis
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-layer-group me-2"),
                                    "Run Combined Analysis",
                                ],
                                id="run-combined-btn",
                                color="warning",
                                size="lg",
                                className="w-100 mb-2",
                            ),
                            html.Small(
                                "All strategies together", className="text-muted"
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-sync-alt me-2"),
                                    "Run ETF Rotation Analysis",
                                ],
                                id="run-etf-rotation-btn",
                                color="dark",
                                size="lg",
                                className="w-100 mb-2",
                            ),
                            html.Small(
                                "Dual Momentum + Sector Rotation",
                                className="text-muted",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-cogs me-2"),
                                    "Run All Strategies",
                                ],
                                id="run-all-strategies-btn",
                                color="danger",
                                size="lg",
                                className="w-100 mb-2",
                            ),
                            html.Small(
                                "Complete strategy comparison", className="text-muted"
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-4",
            ),
            # Loading Spinner
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Spinner(
                                html.Div(id="analysis-loading"),
                                color="primary",
                                size="lg",
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            # Results Section
            html.Div(
                [
                    # Summary Cards
                    html.Div(id="analysis-summary-cards", className="mb-4"),
                    # Charts Row
                    dbc.Row(
                        [
                            dbc.Col([html.Div(id="analysis-charts-left")], md=6),
                            dbc.Col([html.Div(id="analysis-charts-right")], md=6),
                        ],
                        className="mb-4",
                    ),
                    # Detailed Results Table
                    html.Div(id="analysis-results-table", className="mb-4"),
                    # Error Messages
                    html.Div(id="analysis-error", className="mb-4"),
                ],
                id="analysis-results",
                style={"display": "none"},
            ),
            # Hidden divs for storing data
            html.Div(id="analysis-data-store", style={"display": "none"}),
            html.Div(id="current-analysis-type", style={"display": "none"}),
        ],
        className="analysis-container",
    )


def create_summary_cards(summary_data, analysis_type):
    """
    Create summary cards for analysis results.

    Args:
        summary_data: Dictionary with summary statistics
        analysis_type: Type of analysis performed

    Returns:
        Dash layout component
    """
    if not summary_data:
        return html.Div("No summary data available", className="text-muted")

    cards = []

    # Total Signals Card
    cards.append(
        dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    summary_data.get("total_signals", 0),
                                    className="card-title",
                                ),
                                html.P(
                                    "Total Signals", className="card-text text-muted"
                                ),
                            ]
                        )
                    ],
                    className="text-center h-100",
                )
            ],
            md=3,
        )
    )

    # Buy Signals Card
    cards.append(
        dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    summary_data.get("buy_signals", 0),
                                    className="card-title text-success",
                                ),
                                html.P("Buy Signals", className="card-text text-muted"),
                            ]
                        )
                    ],
                    className="text-center h-100",
                )
            ],
            md=3,
        )
    )

    # High Confidence Card
    cards.append(
        dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    summary_data.get("high_confidence_signals", 0),
                                    className="card-title text-warning",
                                ),
                                html.P(
                                    "High Confidence", className="card-text text-muted"
                                ),
                            ]
                        )
                    ],
                    className="text-center h-100",
                )
            ],
            md=3,
        )
    )

    # Average Confidence Card
    avg_conf = summary_data.get("avg_confidence", 0)
    cards.append(
        dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    f"{avg_conf:.1%}", className="card-title text-info"
                                ),
                                html.P(
                                    "Avg Confidence", className="card-text text-muted"
                                ),
                            ]
                        )
                    ],
                    className="text-center h-100",
                )
            ],
            md=3,
        )
    )

    return dbc.Row(cards, className="mb-4")


def create_signals_table(signals, analysis_type):
    """
    Create a table showing detailed signal information with paper trading actions.

    Args:
        signals: List of strategy signals
        analysis_type: Type of analysis performed

    Returns:
        Dash layout component
    """
    if not signals:
        return html.Div("No signals generated", className="text-muted")

    # Store original signals for callbacks
    signal_cards = []

    # Convert signals to table data and create action buttons
    if analysis_type == "combined":
        # Handle combined analysis with nested signals
        for strategy, strategy_signals in signals.items():
            for i, signal in enumerate(strategy_signals):
                # Handle both object and dictionary formats
                symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
                signal_id = f"{strategy}_{symbol}_{i}"

                # Create signal card with paper trade buttons
                signal_card = create_signal_card(
                    signal, signal_id, strategy.replace("_", " ").title()
                )
                signal_cards.append(signal_card)
    else:
        # Handle single strategy analysis
        for i, signal in enumerate(signals):
            # Handle both object and dictionary formats
            symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
            signal_id = f"{analysis_type}_{symbol}_{i}"

            # Create signal card with paper trade buttons
            signal_card = create_signal_card(
                signal, signal_id, analysis_type.replace("_", " ").title()
            )
            signal_cards.append(signal_card)

    return html.Div(
        [
            html.H5(
                [
                    html.I(className="fas fa-flask me-2 text-warning"),
                    "Paper Trading Signals",
                    html.Span(
                        " (No Real Money)", className="badge bg-warning text-dark ms-2"
                    ),
                ],
                className="mb-3",
            ),
            html.Div(signal_cards, className="signals-list"),
            # Hidden div to store signal data for callbacks
            html.Div(id="signals-data-store", style={"display": "none"}),
            # Confirmation modal
            create_trade_confirmation_modal(),
            # Trade result alerts
            html.Div(id="trade-alerts", className="mt-3"),
        ]
    )


def create_signal_card(signal, signal_id, strategy_name):
    """
    Create a signal card with paper trading action buttons.

    Args:
        signal: StrategySignal object
        signal_id: Unique identifier for the signal
        strategy_name: Name of the strategy

    Returns:
        Dash card component
    """
    # Handle both StrategySignal objects and dictionary signals
    if hasattr(signal, "signal_type"):
        # StrategySignal object
        signal_type_value = signal.signal_type.value
        symbol = signal.symbol
        price = signal.price
        confidence = signal.confidence
        timestamp = signal.timestamp
    else:
        # Dictionary signal (after JSON serialization)
        signal_type_value = signal.get("signal_type", "UNKNOWN")
        symbol = signal.get("symbol", "UNKNOWN")
        price = signal.get("price", 0.0)
        confidence = signal.get("confidence", 0.0)
        timestamp = signal.get("timestamp", "")

    # Determine signal color and icon
    if signal_type_value == "BUY":
        signal_color = "success"
        signal_icon = "fas fa-arrow-up"
        paper_btn_color = "success"
        paper_btn_text = "🧪 PAPER BUY"
        paper_btn_icon = "fas fa-shopping-cart"
    elif signal_type_value in ["SELL", "CLOSE_LONG"]:
        signal_color = "danger"
        signal_icon = "fas fa-arrow-down"
        paper_btn_color = "danger"
        paper_btn_text = "🧪 PAPER SELL"
        paper_btn_icon = "fas fa-hand-holding-usd"
    else:
        signal_color = "secondary"
        signal_icon = "fas fa-minus"
        paper_btn_color = "secondary"
        paper_btn_text = "🧪 PAPER TRADE"
        paper_btn_icon = "fas fa-exchange-alt"

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6(
                                        [
                                            html.I(className=signal_icon + " me-2"),
                                            html.Strong(symbol),
                                            html.Span(
                                                signal_type_value,
                                                className=f"badge bg-{signal_color} ms-2",
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.P(
                                        [
                                            html.Strong("Strategy: "),
                                            strategy_name,
                                            html.Br(),
                                            html.Strong("Price: "),
                                            f"${price:.2f}",
                                            html.Br(),
                                            html.Strong("Confidence: "),
                                            f"{confidence:.1%}",
                                            html.Br(),
                                            html.Strong("Time: "),
                                            (
                                                timestamp[:16]
                                                if isinstance(timestamp, str)
                                                else timestamp.strftime(
                                                    "%Y-%m-%d %H:%M"
                                                )
                                            ),
                                        ],
                                        className="mb-0 small text-muted",
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className=paper_btn_icon + " me-2"),
                                            paper_btn_text,
                                        ],
                                        id={
                                            "type": "paper-trade-btn",
                                            "index": signal_id,
                                        },
                                        color=paper_btn_color,
                                        size="sm",
                                        className="w-100 mb-2",
                                        outline=True,
                                    ),
                                    html.Small(
                                        "⚠️ Paper Trading Only",
                                        className="text-warning d-block text-center",
                                    ),
                                ],
                                md=4,
                                className="d-flex flex-column justify-content-center",
                            ),
                        ]
                    )
                ]
            )
        ],
        className="mb-3 signal-card",
        style={"border": f"1px solid var(--{signal_color})"},
    )


def create_trade_confirmation_modal():
    """Create modal for confirming paper trades."""
    return dbc.Modal(
        [
            dbc.ModalHeader(
                [
                    html.I(className="fas fa-flask me-2 text-warning"),
                    html.H4(
                        "🧪 PAPER TRADE CONFIRMATION", className="mb-0 text-warning"
                    ),
                ],
                style={
                    "background-color": "#fff3cd",
                    "border-bottom": "2px solid #ffc107",
                },
            ),
            dbc.ModalBody(
                [
                    dbc.Alert(
                        [
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            html.Strong(
                                "This is a PAPER TRADE - No real money will be used!"
                            ),
                        ],
                        color="warning",
                        className="mb-3",
                    ),
                    html.Div(id="trade-confirmation-details"),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        [html.I(className="fas fa-times me-2"), "Cancel"],
                        id="cancel-trade-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-flask me-2"), "Execute Paper Trade"],
                        id="confirm-paper-trade-btn",
                        color="warning",
                        disabled=False,
                    ),
                ]
            ),
        ],
        id="trade-confirmation-modal",
        size="lg",
        is_open=False,
    )


def create_charts(charts_data, analysis_type):
    """
    Create charts for analysis results.

    Args:
        charts_data: Dictionary with chart figures
        analysis_type: Type of analysis performed

    Returns:
        Tuple of (left_charts, right_charts) components
    """
    if not charts_data:
        return html.Div("No charts available", className="text-muted"), html.Div(
            "", className="text-muted"
        )

    left_charts = []
    right_charts = []

    if analysis_type == "golden_cross":
        if "signal_distribution" in charts_data:
            left_charts.append(
                dcc.Graph(
                    figure=charts_data["signal_distribution"],
                    config={"displayModeBar": False},
                )
            )
        if "confidence_distribution" in charts_data:
            right_charts.append(
                dcc.Graph(
                    figure=charts_data["confidence_distribution"],
                    config={"displayModeBar": False},
                )
            )

    elif analysis_type == "mean_reversion":
        if "signal_distribution" in charts_data:
            left_charts.append(
                dcc.Graph(
                    figure=charts_data["signal_distribution"],
                    config={"displayModeBar": False},
                )
            )
        if "confidence_distribution" in charts_data:
            right_charts.append(
                dcc.Graph(
                    figure=charts_data["confidence_distribution"],
                    config={"displayModeBar": False},
                )
            )

    elif analysis_type == "combined":
        if "strategy_comparison" in charts_data:
            left_charts.append(
                dcc.Graph(
                    figure=charts_data["strategy_comparison"],
                    config={"displayModeBar": False},
                )
            )
        if "combined_signal_distribution" in charts_data:
            right_charts.append(
                dcc.Graph(
                    figure=charts_data["combined_signal_distribution"],
                    config={"displayModeBar": False},
                )
            )

    elif analysis_type == "dual_momentum":
        if "signal_distribution" in charts_data:
            left_charts.append(
                dcc.Graph(
                    figure=charts_data["signal_distribution"],
                    config={"displayModeBar": False},
                )
            )
        if "momentum_scores" in charts_data:
            right_charts.append(
                dcc.Graph(
                    figure=charts_data["momentum_scores"],
                    config={"displayModeBar": False},
                )
            )

    elif analysis_type == "sector_rotation":
        if "signal_distribution" in charts_data:
            left_charts.append(
                dcc.Graph(
                    figure=charts_data["signal_distribution"],
                    config={"displayModeBar": False},
                )
            )
        if "sector_rankings" in charts_data:
            right_charts.append(
                dcc.Graph(
                    figure=charts_data["sector_rankings"],
                    config={"displayModeBar": False},
                )
            )

    elif analysis_type == "etf_rotation":
        if "strategy_comparison" in charts_data:
            left_charts.append(
                dcc.Graph(
                    figure=charts_data["strategy_comparison"],
                    config={"displayModeBar": False},
                )
            )
        if "combined_signal_distribution" in charts_data:
            right_charts.append(
                dcc.Graph(
                    figure=charts_data["combined_signal_distribution"],
                    config={"displayModeBar": False},
                )
            )

    elif analysis_type == "all_strategies":
        if "all_strategies_comparison" in charts_data:
            left_charts.append(
                dcc.Graph(
                    figure=charts_data["all_strategies_comparison"],
                    config={"displayModeBar": False},
                )
            )
        if "all_strategies_signal_distribution" in charts_data:
            right_charts.append(
                dcc.Graph(
                    figure=charts_data["all_strategies_signal_distribution"],
                    config={"displayModeBar": False},
                )
            )

    # Wrap charts in containers
    left_container = (
        html.Div(left_charts)
        if left_charts
        else html.Div("No charts available", className="text-muted")
    )
    right_container = (
        html.Div(right_charts) if right_charts else html.Div("", className="text-muted")
    )

    return left_container, right_container


def create_error_message(error_msg):
    """
    Create an error message component.

    Args:
        error_msg: Error message string

    Returns:
        Dash layout component
    """
    return dbc.Alert(
        [
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Analysis Error: {error_msg}",
        ],
        color="danger",
        dismissable=True,
    )


# Initialize the analysis service
analysis_service = DashboardAnalysisService()


def register_analysis_callbacks(app):
    """
    Register all analysis-related callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        [
            Output("analysis-loading", "children"),
            Output("analysis-data-store", "children"),
            Output("current-analysis-type", "children"),
            Output("analysis-error", "children"),
        ],
        [
            Input("run-golden-cross-btn", "n_clicks"),
            Input("run-mean-reversion-btn", "n_clicks"),
            Input("run-dual-momentum-btn", "n_clicks"),
            Input("run-sector-rotation-btn", "n_clicks"),
            Input("run-combined-btn", "n_clicks"),
            Input("run-etf-rotation-btn", "n_clicks"),
            Input("run-all-strategies-btn", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def run_analysis(
        golden_clicks,
        mean_rev_clicks,
        dual_momentum_clicks,
        sector_rotation_clicks,
        combined_clicks,
        etf_rotation_clicks,
        all_strategies_clicks,
    ):
        """Run strategy analysis based on button clicks."""
        ctx = callback_context
        if not ctx.triggered:
            return "", "", "", ""

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        try:
            if button_id == "run-golden-cross-btn":
                results = analysis_service.run_golden_cross_analysis()
                analysis_type = "golden_cross"
            elif button_id == "run-mean-reversion-btn":
                results = analysis_service.run_mean_reversion_analysis()
                analysis_type = "mean_reversion"
            elif button_id == "run-dual-momentum-btn":
                results = analysis_service.run_dual_momentum_analysis()
                analysis_type = "dual_momentum"
            elif button_id == "run-sector-rotation-btn":
                results = analysis_service.run_sector_rotation_analysis()
                analysis_type = "sector_rotation"
            elif button_id == "run-combined-btn":
                results = analysis_service.run_combined_analysis()
                analysis_type = "combined"
            elif button_id == "run-etf-rotation-btn":
                results = analysis_service.run_etf_rotation_analysis()
                analysis_type = "etf_rotation"
            elif button_id == "run-all-strategies-btn":
                results = analysis_service.run_all_strategies_analysis()
                analysis_type = "all_strategies"
            else:
                return "", "", "", ""

            if results["success"]:
                # Store results as JSON string with proper signal serialization
                import json

                # Convert signals to serializable format
                serialized_results = results.copy()
                if "signals" in results:
                    serialized_results["signals"] = [
                        {
                            "symbol": signal.symbol,
                            "signal_type": signal.signal_type.value,
                            "confidence": signal.confidence,
                            "price": signal.price if signal.price else 0.0,
                            "quantity": signal.quantity,
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "timestamp": (
                                signal.timestamp.isoformat()
                                if signal.timestamp
                                else None
                            ),
                            "strategy_name": signal.strategy_name,
                            "metadata": signal.metadata or {},
                        }
                        for signal in results["signals"]
                    ]

                results_json = json.dumps(serialized_results, default=str)
                return "", results_json, analysis_type, ""
            else:
                return "", "", "", create_error_message(results["error"])

        except Exception as e:
            return "", "", "", create_error_message(str(e))

    @app.callback(
        [
            Output("analysis-results", "style"),
            Output("analysis-summary-cards", "children"),
            Output("analysis-charts-left", "children"),
            Output("analysis-charts-right", "children"),
            Output("analysis-results-table", "children"),
        ],
        [
            Input("analysis-data-store", "children"),
            Input("current-analysis-type", "children"),
        ],
    )
    def update_analysis_results(results_json, analysis_type):
        """Update analysis results display."""
        if not results_json or not analysis_type:
            return {"display": "none"}, "", "", "", ""

        try:
            import json

            results = json.loads(results_json)

            # Show results section
            style = {"display": "block"}

            # Create summary cards
            summary_cards = create_summary_cards(results["summary"], analysis_type)

            # Create charts
            left_charts, right_charts = create_charts(results["charts"], analysis_type)

            # Create results table
            results_table = create_signals_table(results["signals"], analysis_type)

            return style, summary_cards, left_charts, right_charts, results_table

        except Exception as e:
            return {"display": "none"}, "", "", "", create_error_message(str(e))

    # Paper Trading Callbacks
    @app.callback(
        [
            Output("trade-confirmation-modal", "is_open"),
            Output("trade-confirmation-details", "children"),
            Output("signals-data-store", "children"),
        ],
        [
            Input(
                {"type": "paper-trade-btn", "index": dash.dependencies.ALL}, "n_clicks"
            )
        ],
        [
            State("analysis-data-store", "children"),
            State("current-analysis-type", "children"),
        ],
        prevent_initial_call=True,
    )
    def handle_paper_trade_button_click(button_clicks, results_json, analysis_type):
        """Handle paper trade button clicks and show confirmation modal."""
        ctx = callback_context
        if not ctx.triggered or not any(button_clicks):
            return False, "", ""

        # Find which button was clicked
        button_id = ctx.triggered[0]["prop_id"]

        try:
            import json
            import re

            # Extract signal ID from button
            match = re.search(r'"index":"([^"]+)"', button_id)
            if not match:
                return False, "Error: Could not identify signal", ""

            signal_id = match.group(1)

            # Parse results to find the signal
            results = json.loads(results_json) if results_json else {"signals": []}

            # Find the signal data
            signal_data = find_signal_by_id(
                results["signals"], signal_id, analysis_type
            )
            if not signal_data:
                return False, "Error: Signal not found", ""

            # Create confirmation details
            confirmation_details = create_trade_confirmation_details(
                signal_data, signal_id
            )

            return (
                True,
                confirmation_details,
                json.dumps({"signal": signal_data, "signal_id": signal_id}),
            )

        except Exception as e:
            return False, f"Error: {str(e)}", ""

    @app.callback(
        Output("trade-confirmation-modal", "is_open", allow_duplicate=True),
        Input("cancel-trade-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_confirmation_modal(cancel_clicks):
        """Close the trade confirmation modal."""
        if cancel_clicks:
            return False
        return dash.no_update

    @app.callback(
        [
            Output("trade-confirmation-modal", "is_open", allow_duplicate=True),
            Output("trade-alerts", "children"),
            Output(
                "dashboard-interval", "n_intervals", allow_duplicate=True
            ),  # Trigger dashboard refresh
        ],
        Input("confirm-paper-trade-btn", "n_clicks"),
        State("signals-data-store", "children"),
        prevent_initial_call=True,
    )
    def execute_paper_trade(confirm_clicks, stored_data):
        """Execute the paper trade when confirmed."""
        if not confirm_clicks or not stored_data:
            return dash.no_update, "", dash.no_update

        try:
            import json
            from dashboard.services.trade_execution_service import (
                PaperTradingExecutionService,
            )
            from strategies.base import StrategySignal, SignalType
            from datetime import datetime

            # Parse stored signal data
            data = json.loads(stored_data)
            signal_data = data["signal"]
            signal_id = data["signal_id"]

            # Recreate StrategySignal object
            signal = StrategySignal(
                symbol=signal_data["symbol"],
                signal_type=SignalType(signal_data["signal_type"]),
                confidence=signal_data["confidence"],
                price=signal_data["price"],
                timestamp=datetime.fromisoformat(
                    signal_data["timestamp"].replace("Z", "+00:00")
                ),
            )

            # Initialize paper trading service and execute trade
            trading_service = PaperTradingExecutionService()
            result = trading_service.execute_paper_trade(signal)

            # Create result alert
            if result["success"]:
                alert = dbc.Alert(
                    [
                        html.I(className="fas fa-check-circle me-2"),
                        html.Strong("🧪 PAPER TRADE EXECUTED! "),
                        result["message"],
                        html.Br(),
                        html.Small(
                            f"⚠️ This was a paper trade - no real money was used",
                            className="text-muted",
                        ),
                    ],
                    color="success",
                    dismissable=True,
                    duration=8000,
                )

                # Trigger dashboard refresh by incrementing n_intervals
                refresh_trigger = 1

            else:
                alert = dbc.Alert(
                    [
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Strong("🧪 PAPER TRADE FAILED: "),
                        result.get("error", "Unknown error"),
                        html.Br(),
                        html.Small(
                            f"Trade type: {result.get('trade_type', 'UNKNOWN')}",
                            className="text-muted",
                        ),
                    ],
                    color="danger",
                    dismissable=True,
                    duration=8000,
                )
                refresh_trigger = dash.no_update

            return False, alert, refresh_trigger

        except Exception as e:
            error_alert = dbc.Alert(
                [
                    html.I(className="fas fa-times-circle me-2"),
                    html.Strong("EXECUTION ERROR: "),
                    str(e),
                ],
                color="danger",
                dismissable=True,
                duration=8000,
            )
            return False, error_alert, dash.no_update


def find_signal_by_id(signals, signal_id, analysis_type):
    """Find signal data by ID."""
    try:
        # Parse signal ID to extract components
        parts = signal_id.split("_")
        if len(parts) < 3:
            return None

        strategy_name = "_".join(parts[:-2])
        symbol = parts[-2]
        index = int(parts[-1])

        if analysis_type == "combined":
            # Combined analysis has nested structure
            if strategy_name in signals and index < len(signals[strategy_name]):
                signal = signals[strategy_name][index]
                # Signal is already in dictionary format after JSON serialization
                if isinstance(signal, dict):
                    signal["strategy_name"] = strategy_name.replace("_", " ").title()
                    return signal
                else:
                    # Fallback for old format
                    return signal
        else:
            # Single strategy analysis
            if index < len(signals):
                signal = signals[index]
                # Signal is already in dictionary format after JSON serialization
                if isinstance(signal, dict):
                    signal["strategy_name"] = analysis_type.replace("_", " ").title()
                    return signal
                else:
                    # Fallback for old format
                    return signal

        return None
    except Exception:
        return None


def create_trade_confirmation_details(signal_data, signal_id):
    """Create trade confirmation details component."""
    return html.Div(
        [
            html.H6(f"🧪 Paper Trade Details", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P([html.Strong("Symbol: "), signal_data["symbol"]]),
                            html.P(
                                [html.Strong("Action: "), signal_data["signal_type"]]
                            ),
                            html.P(
                                [
                                    html.Strong("Strategy: "),
                                    signal_data["strategy_name"],
                                ]
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.P(
                                [html.Strong("Price: "), f"${signal_data['price']:.2f}"]
                            ),
                            html.P(
                                [
                                    html.Strong("Confidence: "),
                                    f"{signal_data['confidence']:.1%}",
                                ]
                            ),
                            html.P(
                                [html.Strong("Time: "), signal_data["timestamp"][:16]]
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.Hr(),
            dbc.Alert(
                [
                    html.Strong("⚠️ REMINDER: "),
                    "This will execute a PAPER TRADE using your Alpaca paper trading account. ",
                    "No real money will be used, but the trade will appear in your paper portfolio.",
                ],
                color="warning",
                className="mt-3",
            ),
        ]
    )
