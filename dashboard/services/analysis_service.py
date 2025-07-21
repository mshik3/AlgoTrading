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
from strategies.etf.dual_momentum import DualMomentumStrategy
from strategies.etf.sector_rotation import SectorRotationStrategy
from strategies.base import StrategySignal, SignalType
from utils.asset_categorization import get_etf_universe_for_strategy

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
            # Crypto (10) - Only available in Alpaca API
            "BTCUSD",
            "ETHUSD",
            "DOTUSD",
            "LINKUSD",
            "LTCUSD",
            "BCHUSD",
            "XRPUSD",
            "SOLUSD",
            "AVAXUSD",
            "UNIUSD",
        ]

        # Initialize strategies
        self.golden_cross = GoldenCrossStrategy(symbols=self.symbols)
        self.mean_reversion = MeanReversionStrategy(symbols=self.symbols)

        logger.info(f"✓ Strategies initialized for {len(self.symbols)} symbols")

    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for all symbols with retry logic.

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        logger.info(f"Fetching market data for {len(self.symbols)} symbols...")

        market_data = {}
        successful_fetches = 0
        max_retries = 3

        for symbol in self.symbols:
            retry_count = 0
            data = None

            while retry_count < max_retries and data is None:
                try:
                    if self.data_collector:
                        # Validate symbol availability first
                        if not self.data_collector.validate_symbol_availability(symbol):
                            logger.warning(
                                f"Skipping {symbol} - not available in Alpaca API"
                            )
                            break

                        # Use real Alpaca data
                        data = self.data_collector.fetch_daily_data(symbol, period="2y")
                    else:
                        raise Exception("Alpaca data collector not available")

                    if data is not None and not data.empty and len(data) >= 200:
                        market_data[symbol] = data
                        successful_fetches += 1
                        logger.info(f"✓ {symbol}: {len(data)} days of data")
                        break
                    else:
                        if data is None:
                            logger.warning(
                                f"✗ No data returned for {symbol} (attempt {retry_count + 1})"
                            )
                        elif data.empty:
                            logger.warning(
                                f"✗ Empty data for {symbol} (attempt {retry_count + 1})"
                            )
                        else:
                            logger.warning(
                                f"✗ Insufficient data for {symbol}: {len(data)} days (need >= 200) (attempt {retry_count + 1})"
                            )
                        data = None  # Reset for retry

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(
                            f"Error fetching data for {symbol} (attempt {retry_count}): {e}. Retrying..."
                        )
                        import time

                        time.sleep(1)  # Brief delay before retry
                    else:
                        logger.error(
                            f"Error fetching data for {symbol} after {max_retries} attempts: {e}"
                        )

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

    def run_dual_momentum_analysis(self) -> Dict:
        """
        Run Dual Momentum ETF rotation analysis.

        Returns:
            Dictionary with dual momentum analysis results
        """
        logger.info("Running Dual Momentum ETF rotation analysis...")

        try:
            # Initialize dual momentum strategy
            etf_universe = get_etf_universe_for_strategy("dual_momentum")
            dual_momentum = DualMomentumStrategy(etf_universe=etf_universe)

            # Fetch market data for ETF universe
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
            signals = dual_momentum.generate_signals(market_data)

            # Process results
            results = self._process_dual_momentum_results(
                signals, market_data, dual_momentum
            )

            logger.info(f"✓ Dual Momentum analysis completed: {len(signals)} signals")
            return results

        except Exception as e:
            logger.error(f"Dual Momentum analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "signals": [],
                "summary": {},
                "charts": {},
            }

    def run_sector_rotation_analysis(self) -> Dict:
        """
        Run Sector Rotation ETF analysis.

        Returns:
            Dictionary with sector rotation analysis results
        """
        logger.info("Running Sector Rotation ETF analysis...")

        try:
            # Initialize sector rotation strategy
            etf_universe = get_etf_universe_for_strategy("sector_rotation")
            sector_rotation = SectorRotationStrategy(etf_universe=etf_universe)

            # Fetch market data for ETF universe
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
            signals = sector_rotation.generate_signals(market_data)

            # Process results
            results = self._process_sector_rotation_results(
                signals, market_data, sector_rotation
            )

            logger.info(f"✓ Sector Rotation analysis completed: {len(signals)} signals")
            return results

        except Exception as e:
            logger.error(f"Sector Rotation analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "signals": [],
                "summary": {},
                "charts": {},
            }

    def run_etf_rotation_analysis(self) -> Dict:
        """
        Run combined ETF rotation analysis (Dual Momentum + Sector Rotation).

        Returns:
            Dictionary with combined ETF rotation analysis results
        """
        logger.info("Running combined ETF rotation analysis...")

        try:
            # Initialize both ETF strategies
            dual_momentum_universe = get_etf_universe_for_strategy("dual_momentum")
            sector_rotation_universe = get_etf_universe_for_strategy("sector_rotation")

            dual_momentum = DualMomentumStrategy(etf_universe=dual_momentum_universe)
            sector_rotation = SectorRotationStrategy(
                etf_universe=sector_rotation_universe
            )

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

            # Generate signals from both strategies
            dual_momentum_signals = dual_momentum.generate_signals(market_data)
            sector_rotation_signals = sector_rotation.generate_signals(market_data)

            # Process combined ETF results
            results = self._process_etf_rotation_results(
                dual_momentum_signals,
                sector_rotation_signals,
                market_data,
                dual_momentum,
                sector_rotation,
            )

            logger.info(
                f"✓ Combined ETF rotation analysis completed: {len(dual_momentum_signals)} + {len(sector_rotation_signals)} signals"
            )
            return results

        except Exception as e:
            logger.error(f"Combined ETF rotation analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "signals": [],
                "summary": {},
                "charts": {},
            }

    def run_all_strategies_analysis(self) -> Dict:
        """
        Run analysis for all strategies (Golden Cross, Mean Reversion, Dual Momentum, Sector Rotation).

        Returns:
            Dictionary with all strategies analysis results
        """
        logger.info("Running all strategies analysis...")

        try:
            # Initialize all strategies
            etf_universe_dual = get_etf_universe_for_strategy("dual_momentum")
            etf_universe_sector = get_etf_universe_for_strategy("sector_rotation")

            dual_momentum = DualMomentumStrategy(etf_universe=etf_universe_dual)
            sector_rotation = SectorRotationStrategy(etf_universe=etf_universe_sector)

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

            # Generate signals from all strategies
            golden_signals = self.golden_cross.generate_signals(market_data)
            mean_rev_signals = self.mean_reversion.generate_signals(market_data)
            dual_momentum_signals = dual_momentum.generate_signals(market_data)
            sector_rotation_signals = sector_rotation.generate_signals(market_data)

            # Process all strategies results
            results = self._process_all_strategies_results(
                golden_signals,
                mean_rev_signals,
                dual_momentum_signals,
                sector_rotation_signals,
                market_data,
            )

            logger.info(
                f"✓ All strategies analysis completed: {len(golden_signals)} + {len(mean_rev_signals)} + {len(dual_momentum_signals)} + {len(sector_rotation_signals)} signals"
            )
            return results

        except Exception as e:
            logger.error(f"All strategies analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "signals": [],
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

    def _process_dual_momentum_results(
        self,
        signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
        strategy,
    ) -> Dict:
        """Process Dual Momentum strategy results."""
        # Get strategy summary
        strategy_summary = strategy.get_dual_momentum_summary()

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
            "current_asset": strategy_summary.get("current_asset"),
            "defensive_mode": strategy_summary.get("defensive_mode", False),
            "qualified_assets_count": len(
                strategy_summary.get("absolute_momentum_scores", {})
            ),
            "timestamp": datetime.now(),
        }

        # Create charts
        charts = self._create_dual_momentum_charts(
            signals, market_data, strategy_summary
        )

        return {
            "success": True,
            "signals": signals,
            "summary": summary,
            "charts": charts,
        }

    def _process_sector_rotation_results(
        self,
        signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
        strategy,
    ) -> Dict:
        """Process Sector Rotation strategy results."""
        # Get strategy summary
        strategy_summary = strategy.get_sector_rotation_summary()

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
            "top_sectors": list(strategy_summary.get("sector_rankings", {}).keys())[:4],
            "sector_count": len(strategy_summary.get("sector_rankings", {})),
            "benchmark_symbol": strategy_summary.get("sector_rotation_config", {}).get(
                "benchmark_symbol", "SPY"
            ),
            "timestamp": datetime.now(),
        }

        # Create charts
        charts = self._create_sector_rotation_charts(
            signals, market_data, strategy_summary
        )

        return {
            "success": True,
            "signals": signals,
            "summary": summary,
            "charts": charts,
        }

    def _process_etf_rotation_results(
        self,
        dual_momentum_signals: List[StrategySignal],
        sector_rotation_signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
        dual_momentum_strategy,
        sector_rotation_strategy,
    ) -> Dict:
        """Process combined ETF rotation results."""
        # Get strategy summaries
        dual_summary = dual_momentum_strategy.get_dual_momentum_summary()
        sector_summary = sector_rotation_strategy.get_sector_rotation_summary()

        # Create combined summary
        total_signals = len(dual_momentum_signals) + len(sector_rotation_signals)
        all_signals = dual_momentum_signals + sector_rotation_signals

        summary = {
            "total_signals": total_signals,
            "dual_momentum_signals": len(dual_momentum_signals),
            "sector_rotation_signals": len(sector_rotation_signals),
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
            "dual_momentum_current_asset": dual_summary.get("current_asset"),
            "sector_rotation_top_sectors": list(
                sector_summary.get("sector_rankings", {}).keys()
            )[:3],
            "timestamp": datetime.now(),
        }

        # Create combined charts
        charts = self._create_etf_rotation_charts(
            dual_momentum_signals,
            sector_rotation_signals,
            market_data,
            dual_summary,
            sector_summary,
        )

        return {
            "success": True,
            "signals": {
                "dual_momentum": dual_momentum_signals,
                "sector_rotation": sector_rotation_signals,
            },
            "summary": summary,
            "charts": charts,
        }

    def _process_all_strategies_results(
        self,
        golden_signals: List[StrategySignal],
        mean_rev_signals: List[StrategySignal],
        dual_momentum_signals: List[StrategySignal],
        sector_rotation_signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """Process all strategies results."""
        # Create comprehensive summary
        total_signals = (
            len(golden_signals)
            + len(mean_rev_signals)
            + len(dual_momentum_signals)
            + len(sector_rotation_signals)
        )
        all_signals = (
            golden_signals
            + mean_rev_signals
            + dual_momentum_signals
            + sector_rotation_signals
        )

        summary = {
            "total_signals": total_signals,
            "golden_cross_signals": len(golden_signals),
            "mean_reversion_signals": len(mean_rev_signals),
            "dual_momentum_signals": len(dual_momentum_signals),
            "sector_rotation_signals": len(sector_rotation_signals),
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

        # Create comprehensive charts
        charts = self._create_all_strategies_charts(
            golden_signals,
            mean_rev_signals,
            dual_momentum_signals,
            sector_rotation_signals,
            market_data,
        )

        return {
            "success": True,
            "signals": {
                "golden_cross": golden_signals,
                "mean_reversion": mean_rev_signals,
                "dual_momentum": dual_momentum_signals,
                "sector_rotation": sector_rotation_signals,
            },
            "summary": summary,
            "charts": charts,
        }

    def _create_dual_momentum_charts(
        self,
        signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
        strategy_summary: Dict,
    ) -> Dict:
        """Create charts for Dual Momentum strategy results."""
        charts = {}

        if not signals:
            return charts

        # Signal distribution chart
        signal_types = [s.signal_type.value for s in signals]
        signal_counts = pd.Series(signal_types).value_counts()

        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Dual Momentum Signal Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(height=300)
        charts["signal_distribution"] = fig

        # Momentum scores chart
        absolute_scores = strategy_summary.get("absolute_momentum_scores", {})
        relative_scores = strategy_summary.get("relative_momentum_scores", {})

        if absolute_scores and relative_scores:
            # Create momentum comparison chart
            symbols = list(absolute_scores.keys())
            abs_values = [absolute_scores[s] for s in symbols]
            rel_values = [relative_scores.get(s, 0) for s in symbols]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Absolute Momentum",
                    x=symbols,
                    y=abs_values,
                    marker_color="lightblue",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Relative Momentum",
                    x=symbols,
                    y=rel_values,
                    marker_color="lightcoral",
                )
            )

            fig.update_layout(
                title="Dual Momentum Scores",
                xaxis_title="ETF Symbols",
                yaxis_title="Momentum Score",
                barmode="group",
                height=400,
            )
            charts["momentum_scores"] = fig

        return charts

    def _create_sector_rotation_charts(
        self,
        signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
        strategy_summary: Dict,
    ) -> Dict:
        """Create charts for Sector Rotation strategy results."""
        charts = {}

        if not signals:
            return charts

        # Signal distribution chart
        signal_types = [s.signal_type.value for s in signals]
        signal_counts = pd.Series(signal_types).value_counts()

        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Sector Rotation Signal Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=300)
        charts["signal_distribution"] = fig

        # Sector rankings chart
        sector_rankings = strategy_summary.get("sector_rankings", {})
        sector_scores = strategy_summary.get("sector_scores", {})

        if sector_rankings and sector_scores:
            # Create sector performance chart
            sectors = list(sector_rankings.keys())[:10]  # Top 10 sectors
            scores = [sector_scores.get(s, 0) for s in sectors]

            fig = px.bar(
                x=sectors,
                y=scores,
                title="Sector Rotation Rankings",
                color=scores,
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(
                xaxis_title="Sectors",
                yaxis_title="Combined Score",
                height=400,
                xaxis_tickangle=-45,
            )
            charts["sector_rankings"] = fig

        return charts

    def _create_etf_rotation_charts(
        self,
        dual_momentum_signals: List[StrategySignal],
        sector_rotation_signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
        dual_summary: Dict,
        sector_summary: Dict,
    ) -> Dict:
        """Create charts for combined ETF rotation results."""
        charts = {}

        # Strategy comparison chart
        strategy_names = ["Dual Momentum", "Sector Rotation"]
        signal_counts = [len(dual_momentum_signals), len(sector_rotation_signals)]

        fig = px.bar(
            x=strategy_names,
            y=signal_counts,
            title="ETF Rotation Strategy Comparison",
            color=signal_counts,
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            xaxis_title="Strategy", yaxis_title="Number of Signals", height=300
        )
        charts["strategy_comparison"] = fig

        # Combined signal distribution
        all_signals = dual_momentum_signals + sector_rotation_signals
        if all_signals:
            signal_types = [s.signal_type.value for s in all_signals]
            signal_counts = pd.Series(signal_types).value_counts()

            fig = px.pie(
                values=signal_counts.values,
                names=signal_counts.index,
                title="Combined ETF Rotation Signal Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(height=300)
            charts["combined_signal_distribution"] = fig

        return charts

    def _create_all_strategies_charts(
        self,
        golden_signals: List[StrategySignal],
        mean_rev_signals: List[StrategySignal],
        dual_momentum_signals: List[StrategySignal],
        sector_rotation_signals: List[StrategySignal],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """Create charts for all strategies comparison."""
        charts = {}

        # Strategy comparison chart
        strategy_names = [
            "Golden Cross",
            "Mean Reversion",
            "Dual Momentum",
            "Sector Rotation",
        ]
        signal_counts = [
            len(golden_signals),
            len(mean_rev_signals),
            len(dual_momentum_signals),
            len(sector_rotation_signals),
        ]

        fig = px.bar(
            x=strategy_names,
            y=signal_counts,
            title="All Strategies Signal Comparison",
            color=signal_counts,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            xaxis_title="Strategy",
            yaxis_title="Number of Signals",
            height=400,
            xaxis_tickangle=-45,
        )
        charts["all_strategies_comparison"] = fig

        # Combined signal distribution
        all_signals = (
            golden_signals
            + mean_rev_signals
            + dual_momentum_signals
            + sector_rotation_signals
        )
        if all_signals:
            signal_types = [s.signal_type.value for s in all_signals]
            signal_counts = pd.Series(signal_types).value_counts()

            fig = px.pie(
                values=signal_counts.values,
                names=signal_counts.index,
                title="All Strategies Signal Distribution",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(height=300)
            charts["all_strategies_signal_distribution"] = fig

        return charts

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
