"""
TradingView Widget Integration
Professional financial charts using TradingView's free widgets
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_tradingview_widget(symbol="AAPL", theme="dark", width="100%", height="400"):
    """
    Create TradingView widget for professional financial charts

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        theme (str): 'dark' or 'light'
        width (str): Widget width
        height (str): Widget height

    Returns:
        html.Div: TradingView widget component
    """

    # TradingView widget HTML
    widget_html = f"""
    <div class="tradingview-widget-container" style="height:{height};width:{width}">
      <div class="tradingview-widget-container__widget" style="height:calc({height} - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "symbol": "NASDAQ:{symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "{theme}",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "backgroundColor": "rgba(13, 16, 23, 1)",
        "gridColor": "rgba(54, 58, 69, 1)",
        "hide_top_toolbar": false,
        "hide_legend": false,
        "save_image": false,
        "container_id": "tradingview_{symbol.lower()}"
      }}
      </script>
    </div>
    """

    return html.Div(
        [
            html.Div(
                id=f"tradingview-{symbol.lower()}",
                children=widget_html,
                style={
                    "height": height,
                    "width": width,
                    "borderRadius": "12px",
                    "overflow": "hidden",
                    "border": "1px solid var(--border-color)",
                },
            )
        ]
    )


def create_mini_chart_widget(symbol="SPX", width="100%", height="200"):
    """
    Create a mini TradingView chart widget for overview displays

    Args:
        symbol (str): Symbol to display
        width (str): Widget width
        height (str): Widget height

    Returns:
        html.Div: Mini TradingView widget
    """

    widget_html = f"""
    <div class="tradingview-widget-container" style="height:{height};width:{width}">
      <div class="tradingview-widget-container__widget" style="height:calc({height} - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
      {{
        "symbol": "{symbol}",
        "width": "100%",
        "height": "{height}",
        "locale": "en",
        "dateRange": "12M",
        "colorTheme": "dark",
        "trendLineColor": "#2962FF",
        "underLineColor": "rgba(41, 98, 255, 0.3)",
        "underLineBottomColor": "rgba(41, 98, 255, 0)",
        "isTransparent": true,
        "autosize": true,
        "largeChartUrl": ""
      }}
      </script>
    </div>
    """

    return html.Div(
        children=widget_html,
        style={
            "height": height,
            "width": width,
            "borderRadius": "8px",
            "overflow": "hidden",
        },
    )


def create_market_overview_widget():
    """
    Create market overview widget showing major indices

    Returns:
        html.Div: Market overview widget
    """

    widget_html = """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-quotes.js" async>
      {
        "width": "100%",
        "height": "300",
        "symbolsGroups": [
          {
            "name": "Indices",
            "originalName": "Indices",
            "symbols": [
              {"name": "OANDA:SPX500USD", "displayName": "S&P 500"},
              {"name": "OANDA:NAS100USD", "displayName": "Nasdaq 100"},
              {"name": "FOREXCOM:DJI", "displayName": "Dow 30"},
              {"name": "INDEX:RTY", "displayName": "Russell 2000"}
            ]
          },
          {
            "name": "Futures",
            "originalName": "Futures",
            "symbols": [
              {"name": "CME_MINI:ES1!", "displayName": "S&P 500"},
              {"name": "CME_MINI:NQ1!", "displayName": "Nasdaq 100"},
              {"name": "CBOT_MINI:YM1!", "displayName": "Dow 30"},
              {"name": "COMEX:GC1!", "displayName": "Gold"}
            ]
          }
        ],
        "showSymbolLogo": true,
        "isTransparent": true,
        "colorTheme": "dark",
        "locale": "en",
        "backgroundColor": "rgba(42, 46, 57, 0)"
      }
      </script>
    </div>
    """

    return html.Div(
        [
            html.H6("Market Overview", className="chart-title mb-3"),
            html.Div(
                children=widget_html,
                style={"minHeight": "300px"},
            ),
        ],
        className="chart-container",
    )


def create_economic_calendar_widget():
    """
    Create economic calendar widget for fundamental analysis

    Returns:
        html.Div: Economic calendar widget
    """

    widget_html = """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
      {
        "width": "100%",
        "height": "300",
        "colorTheme": "dark",
        "isTransparent": true,
        "locale": "en",
        "importanceFilter": "-1,0,1",
        "countryFilter": "us"
      }
      </script>
    </div>
    """

    return html.Div(
        [
            html.H6("Economic Calendar", className="chart-title mb-3"),
            html.Div(
                children=widget_html,
                style={"minHeight": "300px"},
            ),
        ],
        className="chart-container",
    )


def create_portfolio_composition_chart(positions_data):
    """
    Create a portfolio composition donut chart

    Args:
        positions_data (list): List of position dictionaries

    Returns:
        dcc.Graph: Portfolio composition chart
    """

    if not positions_data:
        # Empty portfolio chart
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_annotation(
            text="No positions to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="var(--text-secondary)"),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return dcc.Graph(figure=fig)

    # Create donut chart for portfolio composition
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame(positions_data)

    fig = px.pie(
        df,
        values="market_value",
        names="symbol",
        hole=0.6,
        title="Portfolio Allocation",
    )

    # Update styling for dark theme
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(
            colors=px.colors.qualitative.Set3, line=dict(color="#2a2e39", width=2)
        ),
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="var(--text-primary)",
        title_x=0.5,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )

    return dcc.Graph(figure=fig)
