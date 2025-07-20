"""
Analysis Component for Dashboard
Provides UI for running strategy analysis and viewing results.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Import the analysis service
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.analysis_service import DashboardAnalysisService


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
            # Strategy Buttons Row
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
                        md=4,
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
                        md=4,
                    ),
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
                                "Both strategies together", className="text-muted"
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
    Create a table showing detailed signal information.

    Args:
        signals: List of strategy signals
        analysis_type: Type of analysis performed

    Returns:
        Dash layout component
    """
    if not signals:
        return html.Div("No signals generated", className="text-muted")

    # Convert signals to table data
    if analysis_type == "combined":
        # Handle combined analysis with nested signals
        all_signals = []
        for strategy, strategy_signals in signals.items():
            for signal in strategy_signals:
                all_signals.append(
                    {
                        "Symbol": signal.symbol,
                        "Strategy": strategy.replace("_", " ").title(),
                        "Signal": signal.signal_type.value,
                        "Confidence": f"{signal.confidence:.1%}",
                        "Price": f"${signal.price:.2f}",
                        "Timestamp": signal.timestamp.strftime("%Y-%m-%d %H:%M"),
                    }
                )
        signals = all_signals
    else:
        # Handle single strategy analysis
        signals = [
            {
                "Symbol": signal.symbol,
                "Signal": signal.signal_type.value,
                "Confidence": f"{signal.confidence:.1%}",
                "Price": f"${signal.price:.2f}",
                "Timestamp": signal.timestamp.strftime("%Y-%m-%d %H:%M"),
            }
            for signal in signals
        ]

    # Create table
    df = pd.DataFrame(signals)

    return html.Div(
        [
            html.H5("Signal Details", className="mb-3"),
            dash.dash_table.DataTable(
                id="signals-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "10px", "fontSize": "14px"},
                style_header={
                    "backgroundColor": "#f8f9fa",
                    "fontWeight": "bold",
                    "border": "1px solid #dee2e6",
                },
                style_data_conditional=[
                    {
                        "if": {"column_id": "Signal", "filter_query": "{Signal} = BUY"},
                        "color": "#28a745",
                        "fontWeight": "bold",
                    },
                    {
                        "if": {
                            "column_id": "Signal",
                            "filter_query": "{Signal} = SELL",
                        },
                        "color": "#dc3545",
                        "fontWeight": "bold",
                    },
                    {
                        "if": {
                            "column_id": "Signal",
                            "filter_query": "{Signal} = CLOSE_LONG",
                        },
                        "color": "#fd7e14",
                        "fontWeight": "bold",
                    },
                ],
            ),
        ]
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
            Input("run-combined-btn", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def run_analysis(golden_clicks, mean_rev_clicks, combined_clicks):
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
            elif button_id == "run-combined-btn":
                results = analysis_service.run_combined_analysis()
                analysis_type = "combined"
            else:
                return "", "", "", ""

            if results["success"]:
                # Store results as JSON string
                import json

                results_json = json.dumps(results, default=str)
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
