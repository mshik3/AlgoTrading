"""
Analysis Service for Dashboard
Handles strategy execution and data processing for the dashboard interface.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import project modules
from utils.config import load_environment, get_env_var
from data.alpaca_collector import AlpacaDataCollector, AlpacaConfig
from strategies.equity.golden_cross import GoldenCrossStrategy
from strategies.equity.mean_reversion import MeanReversionStrategy
from strategies.base import StrategySignal, SignalType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardAnalysisService:
    """
    Service class for handling strategy analysis in the dashboard.
    Provides methods to run individual strategies and combined analysis.
    """

    def __init__(self):
        """Initialize the analysis service."""
        self.setup_environment()
        self.setup_data_collector()
        self.setup_strategies()

    def setup_environment(self):
        """Load environment variables and validate configuration."""
        try:
            load_environment()

            # Check for required environment variables
            alpaca_key = get_env_var("ALPACA_API_KEY", default=None)
            alpaca_secret = get_env_var("ALPACA_SECRET_KEY", default=None)

            if not alpaca_key or not alpaca_secret:
                logger.warning("Alpaca API credentials not found - using mock data")
                self.use_mock_data = True
            else:
                self.use_mock_data = False

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            self.use_mock_data = True

    def setup_data_collector(self):
        """Initialize the data collector."""
        try:
            if not self.use_mock_data:
                alpaca_key = get_env_var("ALPACA_API_KEY", required=True)
                alpaca_secret = get_env_var("ALPACA_SECRET_KEY", required=True)

                config = AlpacaConfig(
                    api_key=alpaca_key, secret_key=alpaca_secret, paper=True
                )
                self.data_collector = AlpacaDataCollector(config)
                logger.info("✓ Alpaca data collector initialized")
            else:
                self.data_collector = None
                logger.info("✓ Using mock data mode")

        except Exception as e:
            logger.error(f"Data collector setup failed: {e}")
            self.data_collector = None
            self.use_mock_data = True

    def setup_strategies(self):
        """Initialize trading strategies."""
        # Define the asset universe
        self.symbols = [
            # Major US ETFs (8)
            "SPY",
            "QQQ",
            "VTI",
            "IWM",
            "VEA",
            "VWO",
            "AGG",
            "TLT",
            # Sector ETFs (8)
            "XLF",
            "XLK",
            "XLV",
            "XLE",
            "XLI",
            "XLP",
            "XLU",
            "XLB",
            # Major Tech Stocks (8)
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "NVDA",
            "NFLX",
            # Financial & Industrial (6)
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "UNH",
            "JNJ",
            # International ETFs (6)
            "EFA",
            "EEM",
            "FXI",
            "EWJ",
            "EWG",
            "EWU",
            # Commodity ETFs (4)
            "GLD",
            "SLV",
            "USO",
            "DBA",
            # Crypto (10)
            "BTCUSD",
            "ETHUSD",
            "ADAUSD",
            "DOTUSD",
            "LINKUSD",
            "LTCUSD",
            "BCHUSD",
            "XRPUSD",
            "SOLUSD",
            "MATICUSD",
        ]

        # Initialize strategies
        self.golden_cross = GoldenCrossStrategy(symbols=self.symbols)
        self.mean_reversion = MeanReversionStrategy(symbols=self.symbols)

        logger.info(f"✓ Strategies initialized for {len(self.symbols)} symbols")

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for all symbols.

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        logger.info(f"Fetching market data for {len(self.symbols)} symbols...")

        market_data = {}
        successful_fetches = 0

        for symbol in self.symbols:
            try:
                if self.data_collector:
                    # Use real Alpaca data
                    data = self.data_collector.fetch_daily_data(symbol, period="2y")
                else:
                    raise Exception("Alpaca data collector not available")

                if data is not None and not data.empty and len(data) >= 250:
                    market_data[symbol] = data
                    successful_fetches += 1
                    logger.info(f"✓ {symbol}: {len(data)} days of data")
                else:
                    logger.warning(f"✗ Insufficient data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue

        logger.info(
            f"Successfully fetched data for {successful_fetches}/{len(self.symbols)} symbols"
        )
        return market_data

    def run_golden_cross_analysis(self) -> Dict:
        """
        Run Golden Cross strategy analysis.

        Returns:
            Dictionary with analysis results
        """
        logger.info("Running Golden Cross strategy analysis...")

        try:
            # Fetch market data
            market_data = self.fetch_market_data()

            if not market_data:
                return {
                    "success": False,
                    "error": "No market data available",
                    "signals": [],
                    "summary": {},
                    "charts": {},
                }

            # Generate signals
            signals = self.golden_cross.generate_signals(market_data)

            # Process results
            results = self._process_golden_cross_results(signals, market_data)

            logger.info(f"✓ Golden Cross analysis completed: {len(signals)} signals")
            return results

        except Exception as e:
            logger.error(f"Golden Cross analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "signals": [],
                "summary": {},
                "charts": {},
            }

    def run_mean_reversion_analysis(self) -> Dict:
        """
        Run Mean Reversion strategy analysis.

        Returns:
            Dictionary with analysis results
        """
        logger.info("Running Mean Reversion strategy analysis...")

        try:
            # Fetch market data
            market_data = self.fetch_market_data()

            if not market_data:
                return {
                    "success": False,
                    "error": "No market data available",
                    "signals": [],
                    "summary": {},
                    "charts": {},
                }

            # Generate signals
            signals = self.mean_reversion.generate_signals(market_data)

            # Process results
            results = self._process_mean_reversion_results(signals, market_data)

            logger.info(f"✓ Mean Reversion analysis completed: {len(signals)} signals")
            return results

        except Exception as e:
            logger.error(f"Mean Reversion analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "signals": [],
                "summary": {},
                "charts": {},
            }

    def run_combined_analysis(self) -> Dict:
        """
        Run combined analysis of both strategies.

        Returns:
            Dictionary with combined analysis results
        """
        logger.info("Running combined strategy analysis...")

        try:
            # Fetch market data
            market_data = self.fetch_market_data()

            if not market_data:
                return {
                    "success": False,
                    "error": "No market data available",
                    "signals": {},
                    "summary": {},
                    "charts": {},
                }

            # Generate signals from both strategies
            golden_signals = self.golden_cross.generate_signals(market_data)
            mean_rev_signals = self.mean_reversion.generate_signals(market_data)

            # Process combined results
            results = self._process_combined_results(
                golden_signals, mean_rev_signals, market_data
            )

            logger.info(
                f"✓ Combined analysis completed: {len(golden_signals)} + {len(mean_rev_signals)} signals"
            )
            return results

        except Exception as e:
            logger.error(f"Combined analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "signals": {},
                "summary": {},
                "charts": {},
            }

    def _process_golden_cross_results(
        self, signals: List[StrategySignal], market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Process Golden Cross strategy results."""
        # Categorize signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [
            s
            for s in signals
            if s.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]
        ]

        # Create summary
        summary = {
            "total_signals": len(signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "high_confidence_signals": len([s for s in signals if s.confidence >= 0.8]),
            "avg_confidence": (
                sum(s.confidence for s in signals) / len(signals) if signals else 0
            ),
            "symbols_with_signals": len(set(s.symbol for s in signals)),
            "timestamp": datetime.now(),
        }

        # Create charts
        charts = self._create_golden_cross_charts(signals, market_data)

        return {
            "success": True,
            "signals": signals,
            "summary": summary,
            "charts": charts,
        }

    def _process_mean_reversion_results(
        self, signals: List[StrategySignal], market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Process Mean Reversion strategy results."""
        # Categorize signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [
            s
            for s in signals
            if s.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]
        ]

        # Create summary
        summary = {
            "total_signals": len(signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "high_confidence_signals": len([s for s in signals if s.confidence >= 0.8]),
            "avg_confidence": (
                sum(s.confidence for s in signals) / len(signals) if signals else 0
            ),
            "symbols_with_signals": len(set(s.symbol for s in signals)),
            "timestamp": datetime.now(),
        }

        # Create charts
        charts = self._create_mean_reversion_charts(signals, market_data)

        return {
            "success": True,
            "signals": signals,
            "summary": summary,
            "charts": charts,
        }

    def _process_combined_results(
        self,
        golden_signals: List[StrategySignal],
        mean_rev_signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """Process combined strategy results."""
        # Create combined summary
        total_signals = len(golden_signals) + len(mean_rev_signals)
        all_signals = golden_signals + mean_rev_signals

        summary = {
            "total_signals": total_signals,
            "golden_cross_signals": len(golden_signals),
            "mean_reversion_signals": len(mean_rev_signals),
            "buy_signals": len(
                [s for s in all_signals if s.signal_type == SignalType.BUY]
            ),
            "sell_signals": len(
                [
                    s
                    for s in all_signals
                    if s.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]
                ]
            ),
            "high_confidence_signals": len(
                [s for s in all_signals if s.confidence >= 0.8]
            ),
            "avg_confidence": (
                sum(s.confidence for s in all_signals) / len(all_signals)
                if all_signals
                else 0
            ),
            "symbols_with_signals": len(set(s.symbol for s in all_signals)),
            "timestamp": datetime.now(),
        }

        # Create combined charts
        charts = self._create_combined_charts(
            golden_signals, mean_rev_signals, market_data
        )

        return {
            "success": True,
            "signals": {
                "golden_cross": golden_signals,
                "mean_reversion": mean_rev_signals,
            },
            "summary": summary,
            "charts": charts,
        }

    def _create_golden_cross_charts(
        self, signals: List[StrategySignal], market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Create charts for Golden Cross strategy results."""
        charts = {}

        if not signals:
            return charts

        # Signal distribution chart
        signal_types = [s.signal_type.value for s in signals]
        signal_counts = pd.Series(signal_types).value_counts()

        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Golden Cross Signal Distribution",
            color_discrete_map={
                "BUY": "#00ff00",
                "SELL": "#ff0000",
                "CLOSE_LONG": "#ff6600",
            },
        )
        charts["signal_distribution"] = fig

        # Confidence distribution chart
        confidences = [s.confidence for s in signals]
        fig = px.histogram(
            x=confidences,
            title="Golden Cross Signal Confidence Distribution",
            nbins=10,
            labels={"x": "Confidence", "y": "Count"},
        )
        charts["confidence_distribution"] = fig

        return charts

    def _create_mean_reversion_charts(
        self, signals: List[StrategySignal], market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Create charts for Mean Reversion strategy results."""
        charts = {}

        if not signals:
            return charts

        # Signal distribution chart
        signal_types = [s.signal_type.value for s in signals]
        signal_counts = pd.Series(signal_types).value_counts()

        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Mean Reversion Signal Distribution",
            color_discrete_map={
                "BUY": "#00ff00",
                "SELL": "#ff0000",
                "CLOSE_LONG": "#ff6600",
            },
        )
        charts["signal_distribution"] = fig

        # Confidence distribution chart
        confidences = [s.confidence for s in signals]
        fig = px.histogram(
            x=confidences,
            title="Mean Reversion Signal Confidence Distribution",
            nbins=10,
            labels={"x": "Confidence", "y": "Count"},
        )
        charts["confidence_distribution"] = fig

        return charts

    def _create_combined_charts(
        self,
        golden_signals: List[StrategySignal],
        mean_rev_signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """Create charts for combined strategy results."""
        charts = {}

        # Strategy comparison chart
        strategy_data = {
            "Strategy": ["Golden Cross", "Mean Reversion"],
            "Total Signals": [len(golden_signals), len(mean_rev_signals)],
            "Buy Signals": [
                len([s for s in golden_signals if s.signal_type == SignalType.BUY]),
                len([s for s in mean_rev_signals if s.signal_type == SignalType.BUY]),
            ],
            "High Confidence": [
                len([s for s in golden_signals if s.confidence >= 0.8]),
                len([s for s in mean_rev_signals if s.confidence >= 0.8]),
            ],
        }

        df = pd.DataFrame(strategy_data)

        fig = px.bar(
            df,
            x="Strategy",
            y=["Total Signals", "Buy Signals", "High Confidence"],
            title="Strategy Comparison",
            barmode="group",
        )
        charts["strategy_comparison"] = fig

        # Combined signal distribution
        all_signals = golden_signals + mean_rev_signals
        if all_signals:
            signal_types = [s.signal_type.value for s in all_signals]
            signal_counts = pd.Series(signal_types).value_counts()

            fig = px.pie(
                values=signal_counts.values,
                names=signal_counts.index,
                title="Combined Signal Distribution",
                color_discrete_map={
                    "BUY": "#00ff00",
                    "SELL": "#ff0000",
                    "CLOSE_LONG": "#ff6600",
                },
            )
            charts["combined_signal_distribution"] = fig

        return charts
