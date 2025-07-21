#!/usr/bin/env python3
"""
Today's Investment Analysis Script

This script runs both Golden Cross and Mean Reversion strategies
to identify current investment opportunities in the market.

Features:
- Real-time market data collection from Alpaca
- Dual strategy analysis (Golden Cross + Mean Reversion)
- Position sizing recommendations
- Risk analysis and market context
- Comprehensive reporting
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.config import load_environment, get_env_var
from data.alpaca_collector import AlpacaDataCollector, AlpacaConfig
from strategies.equity.golden_cross import GoldenCrossStrategy
from strategies.equity.mean_reversion import MeanReversionStrategy
from strategies.base import StrategySignal, SignalType

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TodayInvestmentAnalyzer:
    """
    Comprehensive investment analyzer for today's market opportunities.
    Uses only real market data from Alpaca.
    """

    def __init__(self):
        """Initialize the analyzer with environment and data collector."""
        self.setup_environment()
        self.setup_data_collector()
        self.setup_strategies()

    def setup_environment(self):
        """Load environment variables and validate configuration."""
        logger.info("Setting up environment...")

        # Load environment file
        load_environment()

        # Check for required environment variables
        try:
            # Database config (optional for this analysis)
            db_host = get_env_var("DB_HOST", default="localhost")
            db_name = get_env_var("DB_NAME", default="algotrading")

            # Alpaca config (required for real data)
            alpaca_key = get_env_var("ALPACA_API_KEY", default=None)
            alpaca_secret = get_env_var("ALPACA_SECRET_KEY", default=None)

            logger.info(f"Environment loaded - DB: {db_host}/{db_name}")
            if alpaca_key and alpaca_secret:
                logger.info("✓ Alpaca API credentials found")
            else:
                logger.error("✗ Alpaca API credentials required for real market data")
                raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            raise

    def setup_data_collector(self):
        """Initialize the data collector with Alpaca."""
        logger.info("Setting up Alpaca data collector...")

        try:
            # Get Alpaca credentials
            alpaca_key = get_env_var("ALPACA_API_KEY", required=True)
            alpaca_secret = get_env_var("ALPACA_SECRET_KEY", required=True)

            config = AlpacaConfig(
                api_key=alpaca_key, secret_key=alpaca_secret, paper=True
            )
            self.data_collector = AlpacaDataCollector(config)
            logger.info("✓ Alpaca data collector initialized")

        except Exception as e:
            logger.error(f"Data collector setup failed: {e}")
            raise

    def setup_strategies(self):
        """Initialize both trading strategies."""
        logger.info("Setting up trading strategies...")

        # Define the 50-asset universe
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
        Fetch real market data for all symbols from Alpaca.

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        logger.info(f"Fetching real market data for {len(self.symbols)} symbols...")

        market_data = {}
        successful_fetches = 0

        # Use only real market data from Alpaca
        for symbol in self.symbols:
            try:
                # Validate symbol availability first
                if not self.data_collector.validate_symbol_availability(symbol):
                    logger.warning(f"Skipping {symbol} - not available in Alpaca API")
                    continue
                
                # Use Alpaca collector to get real market data
                data = self.data_collector.fetch_daily_data(symbol, period="2y")

                if data is not None and not data.empty and len(data) >= 250:
                    market_data[symbol] = data
                    successful_fetches += 1
                    logger.info(f"✓ {symbol}: {len(data)} days of real data")
                else:
                    logger.warning(f"✗ Insufficient real data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching real data for {symbol}: {e}")
                continue

        logger.info(
            f"Successfully fetched real data for {successful_fetches}/{len(self.symbols)} symbols"
        )
        return market_data

    def generate_strategy_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[StrategySignal]]:
        """
        Generate signals from both strategies using real market data.

        Args:
            market_data: Dictionary of symbol -> OHLCV DataFrame

        Returns:
            Dictionary with strategy names as keys and signal lists as values
        """
        logger.info("Generating trading signals from real market data...")

        signals = {}

        # Golden Cross signals
        try:
            golden_signals = self.golden_cross.generate_signals(market_data)
            signals["golden_cross"] = golden_signals
            logger.info(f"✓ Golden Cross: {len(golden_signals)} signals")
        except Exception as e:
            logger.error(f"Golden Cross signal generation failed: {e}")
            signals["golden_cross"] = []

        # Mean Reversion signals
        try:
            mean_rev_signals = self.mean_reversion.generate_signals(market_data)
            signals["mean_reversion"] = mean_rev_signals
            logger.info(f"✓ Mean Reversion: {len(mean_rev_signals)} signals")
        except Exception as e:
            logger.error(f"Mean Reversion signal generation failed: {e}")
            signals["mean_reversion"] = []

        return signals

    def analyze_signals(
        self,
        signals: Dict[str, List[StrategySignal]],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """
        Analyze and categorize trading signals from real market data.

        Args:
            signals: Dictionary of strategy signals
            market_data: Current market data

        Returns:
            Analysis results with recommendations
        """
        logger.info("Analyzing trading signals from real market data...")

        analysis = {
            "timestamp": datetime.now(),
            "total_signals": sum(len(sig_list) for sig_list in signals.values()),
            "buy_signals": [],
            "sell_signals": [],
            "high_confidence_signals": [],
            "market_summary": {},
            "recommendations": [],
            "market_conditions": {},
        }

        # Process all signals
        for strategy_name, signal_list in signals.items():
            for signal in signal_list:
                signal_info = {
                    "symbol": signal.symbol,
                    "strategy": strategy_name,
                    "signal_type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "price": signal.price,
                    "timestamp": signal.timestamp,
                    "metadata": signal.metadata,
                }

                if signal.signal_type == SignalType.BUY:
                    analysis["buy_signals"].append(signal_info)
                    if signal.confidence >= 0.8:
                        analysis["high_confidence_signals"].append(signal_info)
                elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]:
                    analysis["sell_signals"].append(signal_info)

        # Generate market summary
        analysis["market_summary"] = self._generate_market_summary(market_data)

        # Analyze market conditions
        analysis["market_conditions"] = self._analyze_market_conditions(market_data)

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _generate_market_summary(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate market summary statistics from real data."""
        summary = {
            "total_symbols": len(market_data),
            "market_sectors": {},
            "data_quality": {},
        }

        # Categorize symbols by sector
        sectors = {
            "etfs": [
                "SPY",
                "QQQ",
                "VTI",
                "IWM",
                "VEA",
                "VWO",
                "AGG",
                "TLT",
                "XLF",
                "XLK",
                "XLV",
                "XLE",
                "XLI",
                "XLP",
                "XLU",
                "XLB",
            ],
            "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"],
            "financial": ["JPM", "BAC", "WFC", "GS"],
            "healthcare": ["UNH", "JNJ"],
            "international": ["EFA", "EEM", "FXI", "EWJ", "EWG", "EWU"],
            "commodities": ["GLD", "SLV", "USO", "DBA"],
            "crypto": [
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
            ],
        }

        for sector, symbols in sectors.items():
            sector_data = {s: market_data[s] for s in symbols if s in market_data}
            if sector_data:
                summary["market_sectors"][sector] = len(sector_data)

        # Data quality assessment
        for symbol, data in market_data.items():
            summary["data_quality"][symbol] = {
                "days_available": len(data),
                "latest_date": data.index[-1] if not data.empty else None,
                "has_sufficient_data": len(data) >= 250,
            }

        return summary

    def _analyze_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze current market conditions from real data."""
        conditions = {
            "overall_trend": "neutral",
            "volatility_level": "medium",
            "market_sentiment": "neutral",
            "key_observations": [],
        }

        # Analyze major indices for overall trend
        major_indices = ["SPY", "QQQ", "VTI"]
        available_indices = [s for s in major_indices if s in market_data]

        if available_indices:
            # Calculate recent performance
            recent_performance = {}
            for symbol in available_indices:
                data = market_data[symbol]
                if len(data) >= 30:
                    recent_return = (
                        data["Close"].iloc[-1] / data["Close"].iloc[-30] - 1
                    ) * 100
                    recent_performance[symbol] = recent_return

            # Determine overall trend
            avg_recent_return = sum(recent_performance.values()) / len(
                recent_performance
            )
            if avg_recent_return > 2:
                conditions["overall_trend"] = "bullish"
            elif avg_recent_return < -2:
                conditions["overall_trend"] = "bearish"
            else:
                conditions["overall_trend"] = "neutral"

            conditions["key_observations"].append(
                f"Recent 30-day performance: {avg_recent_return:.1f}%"
            )

        return conditions

    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate actionable investment recommendations from real signals."""
        recommendations = []

        # High confidence buy signals
        for signal in analysis["high_confidence_signals"]:
            rec = {
                "type": "STRONG_BUY",
                "symbol": signal["symbol"],
                "strategy": signal["strategy"],
                "confidence": signal["confidence"],
                "reasoning": f"High confidence {signal['strategy']} signal ({signal['confidence']:.1%})",
                "position_size": (
                    "30% of portfolio"
                    if signal["strategy"] == "golden_cross"
                    else "20% of portfolio"
                ),
                "risk_level": (
                    "Medium" if signal["strategy"] == "golden_cross" else "High"
                ),
            }
            recommendations.append(rec)

        # Regular buy signals
        for signal in analysis["buy_signals"]:
            if signal["confidence"] < 0.8:
                rec = {
                    "type": "BUY",
                    "symbol": signal["symbol"],
                    "strategy": signal["strategy"],
                    "confidence": signal["confidence"],
                    "reasoning": f"{signal['strategy']} signal with {signal['confidence']:.1%} confidence",
                    "position_size": "15% of portfolio",
                    "risk_level": "Medium",
                }
                recommendations.append(rec)

        # Sell signals
        for signal in analysis["sell_signals"]:
            rec = {
                "type": "SELL",
                "symbol": signal["symbol"],
                "strategy": signal["strategy"],
                "confidence": signal["confidence"],
                "reasoning": f"Exit signal from {signal['strategy']} strategy",
                "action": "Close position",
                "urgency": "High" if signal["confidence"] >= 0.8 else "Medium",
            }
            recommendations.append(rec)

        return recommendations

    def print_analysis_report(self, analysis: Dict):
        """Print a comprehensive analysis report based on real market data."""
        print("\n" + "=" * 80)
        print("🎯 TODAY'S INVESTMENT ANALYSIS REPORT")
        print("=" * 80)
        print(
            f"📅 Analysis Date: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"📊 Total Signals Generated: {analysis['total_signals']}")

        # Market Summary
        print(f"\n📈 MARKET SUMMARY:")
        summary = analysis["market_summary"]
        print(f"   • Total Symbols Analyzed: {summary['total_symbols']}")
        for sector, count in summary["market_sectors"].items():
            print(f"   • {sector.title()}: {count} symbols")

        # Market Conditions
        conditions = analysis["market_conditions"]
        print(f"\n🌍 MARKET CONDITIONS:")
        print(f"   • Overall Trend: {conditions['overall_trend'].title()}")
        print(f"   • Volatility Level: {conditions['volatility_level'].title()}")
        print(f"   • Market Sentiment: {conditions['market_sentiment'].title()}")
        for observation in conditions["key_observations"]:
            print(f"   • {observation}")

        # Signal Summary
        print(f"\n🔔 SIGNAL SUMMARY:")
        print(f"   • Buy Signals: {len(analysis['buy_signals'])}")
        print(f"   • Sell Signals: {len(analysis['sell_signals'])}")
        print(
            f"   • High Confidence Signals: {len(analysis['high_confidence_signals'])}"
        )

        if analysis["total_signals"] == 0:
            print(f"\n📋 NO TRADING SIGNALS GENERATED")
            print(f"   This indicates that current market conditions do not meet")
            print(
                f"   the criteria for either Golden Cross or Mean Reversion strategies."
            )
            print(f"   This is normal - strategies are designed to be selective.")
            print(
                f"   Market conditions: {conditions['overall_trend']} trend, {conditions['volatility_level']} volatility"
            )

        # High Confidence Signals
        if analysis["high_confidence_signals"]:
            print(f"\n⭐ HIGH CONFIDENCE OPPORTUNITIES:")
            for signal in analysis["high_confidence_signals"]:
                print(
                    f"   • {signal['symbol']} - {signal['strategy'].replace('_', ' ').title()}"
                )
                print(
                    f"     Confidence: {signal['confidence']:.1%} | Type: {signal['signal_type']}"
                )

        # Recommendations
        if analysis["recommendations"]:
            print(f"\n💡 INVESTMENT RECOMMENDATIONS:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"   {i}. {rec['type']} {rec['symbol']}")
                print(f"      Strategy: {rec['strategy'].replace('_', ' ').title()}")
                print(f"      Confidence: {rec['confidence']:.1%}")
                print(f"      Reasoning: {rec['reasoning']}")
                if "position_size" in rec:
                    print(f"      Position Size: {rec['position_size']}")
                print(f"      Risk Level: {rec['risk_level']}")
                print()

        # All Signals Detail
        if analysis["buy_signals"] or analysis["sell_signals"]:
            print(f"\n📋 DETAILED SIGNAL BREAKDOWN:")

            if analysis["buy_signals"]:
                print(f"   BUY SIGNALS:")
                for signal in analysis["buy_signals"]:
                    print(
                        f"     • {signal['symbol']} ({signal['strategy']}) - {signal['confidence']:.1%} confidence"
                    )

            if analysis["sell_signals"]:
                print(f"   SELL SIGNALS:")
                for signal in analysis["sell_signals"]:
                    print(
                        f"     • {signal['symbol']} ({signal['strategy']}) - {signal['confidence']:.1%} confidence"
                    )

        print("\n" + "=" * 80)
        print("⚠️  DISCLAIMER: This analysis is based on real market data from Alpaca.")
        print(
            "   Always conduct your own research and consider consulting a financial advisor."
        )
        print("   Past performance does not guarantee future results.")
        print("=" * 80)

    def run_analysis(self) -> Dict:
        """
        Run the complete investment analysis using only real market data.

        Returns:
            Complete analysis results
        """
        logger.info("🚀 Starting Today's Investment Analysis with Real Market Data...")

        try:
            # Step 1: Fetch real market data
            market_data = self.fetch_market_data()

            if not market_data:
                raise Exception("No real market data available for analysis")

            # Step 2: Generate strategy signals from real data
            signals = self.generate_strategy_signals(market_data)

            # Step 3: Analyze signals from real data
            analysis = self.analyze_signals(signals, market_data)

            # Step 4: Print report
            self.print_analysis_report(analysis)

            logger.info("✅ Analysis completed successfully with real market data")
            return analysis

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main execution function."""
    try:
        analyzer = TodayInvestmentAnalyzer()
        results = analyzer.run_analysis()

        # Return results for potential further processing
        return results

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        return None


if __name__ == "__main__":
    main()
