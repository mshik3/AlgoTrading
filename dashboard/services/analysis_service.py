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
from strategies.modern_strategies import (
    ModernGoldenCrossStrategy,
    ModernMeanReversionStrategy,
    ModernDualMomentumStrategy,
    ModernSectorRotationStrategy,
    create_strategy,
)
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
            # Only load environment if not already loaded
            from utils.config import _environment_loaded

            if not _environment_loaded:
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
        """Initialize the database-based data provider for instant strategy execution."""
        try:
            # Use new DatabaseMarketDataProvider for instant data access
            from dashboard.services.market_data_provider import get_market_data_provider

            self.data_provider = get_market_data_provider()

            # Also keep Alpaca collector as fallback for fresh data needs
            if not self.use_mock_data:
                alpaca_key = get_env_var("ALPACA_API_KEY", required=True)
                alpaca_secret = get_env_var("ALPACA_SECRET_KEY", required=True)

                config = AlpacaConfig(
                    api_key=alpaca_key, secret_key=alpaca_secret, paper=True
                )
                self.data_collector = AlpacaDataCollector(config)
                logger.info("✓ Database data provider + Alpaca fallback initialized")
            else:
                self.data_collector = None
                logger.info("✓ Database data provider initialized (mock mode)")

        except Exception as e:
            logger.error(f"Data provider setup failed: {e}")
            self.data_provider = None
            self.data_collector = None
            self.use_mock_data = True

    def setup_strategies(self):
        """Initialize trading strategies."""
        # Use new asset universe system
        from utils.asset_universe_config import (
            get_920_asset_universe,
            get_universe_summary,
        )

        # Get the complete 920-asset universe
        self.symbols = get_920_asset_universe()

        # Log universe summary
        summary = get_universe_summary()
        logger.info(f"Dashboard using {len(self.symbols)}-asset universe:")
        logger.info(f"  - Fortune 500: {summary['fortune500_count']}")
        logger.info(f"  - ETFs: {summary['etf_count']}")
        logger.info(f"  - Crypto: {summary['crypto_count']}")

        # Initialize strategies with modern PFund framework
        try:
            # Use modern strategy factory for consistent initialization
            self.golden_cross = create_strategy("golden_cross", symbols=self.symbols)
            self.mean_reversion = create_strategy(
                "mean_reversion", symbols=self.symbols
            )
            logger.info(
                f"✓ Modern PFund strategies initialized for {len(self.symbols)} symbols"
            )
        except Exception as e:
            logger.error(f"Modern strategy initialization failed: {e}")
            # Fallback to direct instantiation if factory fails
            try:
                self.golden_cross = ModernGoldenCrossStrategy(symbols=self.symbols)
                self.mean_reversion = ModernMeanReversionStrategy(symbols=self.symbols)
                logger.info(
                    f"✓ Modern strategies initialized with direct instantiation for {len(self.symbols)} symbols"
                )
            except Exception as e2:
                logger.error(f"Direct strategy initialization also failed: {e2}")
                self.golden_cross = None
                self.mean_reversion = None

    def fetch_market_data(self, strategy=None) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data from pre-loaded database for instant strategy execution.

        Args:
            strategy: Optional strategy instance to get minimum data requirements

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        import time

        # Get minimum data requirements from strategy if provided
        min_days_required = 200  # Default requirement
        if strategy and hasattr(strategy, "get_minimum_data_requirements"):
            min_days_required = strategy.get_minimum_data_requirements()
            logger.debug(
                f"Using strategy-specific data requirement: {min_days_required} days"
            )
        elif (
            strategy
            and hasattr(strategy, "config")
            and "min_data_days" in strategy.config
        ):
            min_days_required = strategy.config["min_data_days"]
            logger.debug(
                f"Using strategy config data requirement: {min_days_required} days"
            )

        logger.info(f"🚀 Fetching pre-loaded data for {len(self.symbols)} symbols...")

        start_time = time.time()

        try:
            if hasattr(self, "data_provider") and self.data_provider:
                # Use database provider for instant data access
                market_data = self.data_provider.get_market_data(
                    symbols=self.symbols,
                    period="2y",  # 2 years of data for comprehensive analysis
                    validate_completeness=True,
                )

                # Filter data based on minimum requirements
                valid_data = {}
                skipped_symbols = 0

                for symbol, data in market_data.items():
                    if len(data) >= min_days_required:
                        valid_data[symbol] = data
                    else:
                        logger.debug(
                            f"⚠️ {symbol}: Only {len(data)} days (need {min_days_required}+)"
                        )
                        skipped_symbols += 1

                fetch_time = time.time() - start_time

                logger.info(
                    f"✅ Database fetch complete: {len(valid_data)}/{len(self.symbols)} symbols "
                    f"in {fetch_time:.2f}s ({skipped_symbols} skipped)"
                )

                # Log performance improvement
                estimated_api_time = (
                    len(self.symbols) * 2
                )  # ~2 seconds per symbol via API
                time_saved = estimated_api_time - fetch_time
                if time_saved > 0:
                    logger.info(
                        f"⚡ Performance: {time_saved:.0f}s saved vs API calls "
                        f"({fetch_time:.2f}s vs ~{estimated_api_time:.0f}s)"
                    )

                return valid_data

            else:
                logger.warning(
                    "Database data provider not available, falling back to API"
                )
                return self._fallback_api_fetch(min_days_required)

        except Exception as e:
            logger.error(f"Database fetch failed: {str(e)}")
            logger.info("Attempting fallback to API data collection...")
            return self._fallback_api_fetch(min_days_required)

    def _fallback_api_fetch(self, min_days_required: int) -> Dict[str, pd.DataFrame]:
        """
        Fallback method to fetch data via API if database provider fails.

        Args:
            min_days_required: Minimum number of days required

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        logger.warning("Using fallback API data collection (this will be slower)")

        if not hasattr(self, "data_collector") or not self.data_collector:
            logger.error("No data collector available for fallback")
            return {}

        market_data = {}
        successful_fetches = 0

        # Use batch processing for better API efficiency
        try:
            from data.storage import get_session

            # Use incremental fetch batch method if available
            session = get_session()
            batch_results = self.data_collector.incremental_fetch_batch(
                session=session, symbols=self.symbols, period="2y", batch_size=10
            )
            session.close()

            # Filter and validate results
            for symbol, data in batch_results.items():
                if (
                    data is not None
                    and not data.empty
                    and len(data) >= min_days_required
                ):
                    market_data[symbol] = data
                    successful_fetches += 1

            logger.info(
                f"✓ Fallback API fetch complete: {successful_fetches}/{len(self.symbols)} symbols"
            )

        except Exception as e:
            logger.error(f"Fallback API fetch failed: {str(e)}")

        return market_data

    def run_golden_cross_analysis(self) -> Dict:
        """
        Run Golden Cross strategy analysis.

        Returns:
            Dictionary with analysis results
        """
        logger.info("Running Golden Cross strategy analysis...")

        try:
            # Fetch market data with strategy-specific requirements
            market_data = self.fetch_market_data(self.golden_cross)

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
            # Fetch market data with strategy-specific requirements
            market_data = self.fetch_market_data(self.mean_reversion)

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
            # Fetch market data with strategy-specific requirements (use lower requirement)
            market_data = self.fetch_market_data(self.mean_reversion)

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
            # Initialize dual momentum strategy with modern PFund framework
            etf_universe = get_etf_universe_for_strategy("dual_momentum")
            try:
                dual_momentum = create_strategy("dual_momentum", assets=etf_universe)
            except Exception as e:
                logger.error(f"Modern dual momentum strategy creation failed: {e}")
                dual_momentum = ModernDualMomentumStrategy(assets=etf_universe)

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
            # Initialize sector rotation strategy with modern PFund framework
            etf_universe = get_etf_universe_for_strategy("sector_rotation")
            try:
                sector_rotation = create_strategy(
                    "sector_rotation", sectors=etf_universe
                )
            except Exception as e:
                logger.error(f"Modern sector rotation strategy creation failed: {e}")
                sector_rotation = ModernSectorRotationStrategy(sectors=etf_universe)

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
            # Get Alpaca client for position synchronization
            alpaca_client = self._get_alpaca_client()

            # Initialize both ETF strategies with modern PFund framework
            dual_momentum_universe = get_etf_universe_for_strategy("dual_momentum")
            sector_rotation_universe = get_etf_universe_for_strategy("sector_rotation")

            try:
                dual_momentum = create_strategy(
                    "dual_momentum", assets=dual_momentum_universe
                )
                sector_rotation = create_strategy(
                    "sector_rotation", sectors=sector_rotation_universe
                )
            except Exception as e:
                logger.error(f"Modern ETF strategy creation failed: {e}")
                dual_momentum = ModernDualMomentumStrategy(
                    assets=dual_momentum_universe
                )
                sector_rotation = ModernSectorRotationStrategy(
                    sectors=sector_rotation_universe
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
            logger.error(f"Error in ETF rotation analysis: {str(e)}")
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
            # Initialize all strategies with modern PFund framework
            etf_universe_dual = get_etf_universe_for_strategy("dual_momentum")
            etf_universe_sector = get_etf_universe_for_strategy("sector_rotation")

            try:
                dual_momentum = create_strategy(
                    "dual_momentum", assets=etf_universe_dual
                )
                sector_rotation = create_strategy(
                    "sector_rotation", sectors=etf_universe_sector
                )
            except Exception as e:
                logger.error(f"Modern strategy creation failed: {e}")
                dual_momentum = ModernDualMomentumStrategy(assets=etf_universe_dual)
                sector_rotation = ModernSectorRotationStrategy(
                    sectors=etf_universe_sector
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
            "signals": {"mean_reversion": signals},
            "summary": summary,
            "charts": charts,
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
            "signals": {"golden_cross": signals},
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
        charts["signal_distribution"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

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
            charts["momentum_scores"] = (
                fig.to_dict()
            )  # Convert to dict for serialization

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
        charts["signal_distribution"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

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
            charts["sector_rankings"] = (
                fig.to_dict()
            )  # Convert to dict for serialization

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
        charts["strategy_comparison"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

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
            charts["combined_signal_distribution"] = (
                fig.to_dict()
            )  # Convert to dict for serialization

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
        charts["all_strategies_comparison"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

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
            charts["all_strategies_signal_distribution"] = (
                fig.to_dict()
            )  # Convert to dict for serialization

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
        charts["signal_distribution"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

        # Confidence distribution chart
        confidences = [s.confidence for s in signals]
        fig = px.histogram(
            x=confidences,
            title="Golden Cross Signal Confidence Distribution",
            nbins=10,
            labels={"x": "Confidence", "y": "Count"},
        )
        charts["confidence_distribution"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

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
        charts["signal_distribution"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

        # Confidence distribution chart
        confidences = [s.confidence for s in signals]
        fig = px.histogram(
            x=confidences,
            title="Mean Reversion Signal Confidence Distribution",
            nbins=10,
            labels={"x": "Confidence", "y": "Count"},
        )
        charts["confidence_distribution"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

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
        charts["strategy_comparison"] = (
            fig.to_dict()
        )  # Convert to dict for serialization

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
            charts["combined_signal_distribution"] = (
                fig.to_dict()
            )  # Convert to dict for serialization

        return charts

    def _get_alpaca_client(self):
        """Get Alpaca client for position synchronization."""
        try:
            from execution.alpaca import get_alpaca_client

            return get_alpaca_client()
        except Exception as e:
            logger.warning(f"Could not get Alpaca client: {e}")
            return None
