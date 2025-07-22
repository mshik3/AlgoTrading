#!/usr/bin/env python3
"""
Golden Cross Live Analysis and Trading Script

This script analyzes current market conditions for Golden Cross opportunities
and generates trading signals for execution on Alpaca paper trading.

Features:
- Real-time market data analysis
- Golden Cross signal generation
- Portfolio analysis and position sizing
- Trading recommendations
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.equity.golden_cross import GoldenCrossStrategy
from execution.alpaca import get_alpaca_client
from data.collectors import get_collector
from indicators.technical import TechnicalIndicators
from utils.asset_categorization import categorize_asset

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GoldenCrossAnalyzer:
    """
    Analyzer for Golden Cross strategy opportunities.
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.alpaca_client = None
        self.data_collector = None
        self.strategy = None

        # Use expanded 50-asset universe from strategy
        self.symbols = None  # Will be set by strategy initialization

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize Alpaca client
            self.alpaca_client = get_alpaca_client()
            logger.info("✓ Alpaca client initialized")

            # Initialize data collector (use Alpaca for reliable data)
            try:
                from alpaca_data_collector import AlpacaDataCollector

                self.data_collector = AlpacaDataCollector()
                logger.info("✓ Using Alpaca data collector")
            except Exception as e:
                logger.warning(f"Alpaca data collector not available: {e}")
                self.data_collector = get_collector("yahoo")
                logger.info("✓ Using Yahoo Finance data collector")
            logger.info("✓ Data collector initialized")

            # Initialize Golden Cross strategy with expanded 50-asset universe
            self.strategy = GoldenCrossStrategy()  # Uses default 50-asset universe
            self.symbols = self.strategy.symbols  # Get symbols from strategy
            logger.info(
                f"✓ Golden Cross strategy initialized with {len(self.symbols)} assets"
            )

        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """Test all connections."""
        try:
            # Test Alpaca connection
            if not self.alpaca_client.test_connection():
                logger.error("Alpaca connection failed")
                return False

            # Test data collector
            try:
                test_data = self.data_collector.fetch_daily_data("SPY", period="1mo")
                if test_data is None or test_data.empty:
                    logger.warning("Data collector test failed, but continuing...")
                    # Don't fail the entire test, just warn
                else:
                    logger.info("✓ Data collector test successful")
            except Exception as e:
                logger.warning(f"Data collector test failed: {e}, but continuing...")

            logger.info("✓ All connections successful")
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_current_market_data(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get current market data for analysis.

        Args:
            symbols: List of symbols to analyze (defaults to strategy symbols)

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        symbols = symbols or self.strategy.symbols
        market_data = {}

        logger.info(f"Fetching market data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                # Use incremental data loading for better performance
                try:
                    # Try to get database session for incremental loading
                    from data.storage import get_session

                    session = get_session()

                    # Use incremental fetch if session is available
                    data = self.data_collector.incremental_fetch_daily_data(
                        session=session, symbol=symbol, period="1y"
                    )
                    session.close()
                except Exception as session_error:
                    logger.warning(
                        f"Could not use incremental loading for {symbol}: {session_error}"
                    )
                    # Fallback to regular fetch
                    data = self.data_collector.fetch_daily_data(symbol, period="1y")

                if data is not None and not data.empty and len(data) >= 250:
                    market_data[symbol] = data
                    logger.info(f"✓ {symbol}: {len(data)} days of data")
                else:
                    logger.warning(
                        f"✗ No data for {symbol} (need at least 250 days for 200-day MA)"
                    )

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")

        logger.info(f"Retrieved data for {len(market_data)} symbols")
        return market_data

    def analyze_golden_cross_opportunities(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Analyze Golden Cross opportunities for each symbol.

        Args:
            market_data: Dictionary mapping symbol -> OHLCV DataFrame

        Returns:
            Dictionary with analysis results for each symbol
        """
        analysis_results = {}

        for symbol, data in market_data.items():
            try:
                # Calculate technical indicators
                indicators = TechnicalIndicators(data)
                indicators.add_sma(50, "Close")
                indicators.add_sma(200, "Close")
                indicators.add_volume_sma(20)

                enhanced_data = indicators.get_data()

                # Get latest values
                latest = enhanced_data.iloc[-1]
                prev = enhanced_data.iloc[-2]

                # Calculate Golden Cross conditions
                sma_50 = latest.get("SMA_50", 0)
                sma_200 = latest.get("SMA_200", 0)
                prev_sma_50 = prev.get("SMA_50", 0)
                prev_sma_200 = prev.get("SMA_200", 0)

                # Check for Golden Cross (50 MA crosses above 200 MA)
                golden_cross = (
                    sma_50 > sma_200
                    and prev_sma_50 <= prev_sma_200
                    and sma_50 > 0
                    and sma_200 > 0
                )

                # Check for Death Cross (50 MA crosses below 200 MA)
                death_cross = (
                    sma_50 < sma_200
                    and prev_sma_50 >= prev_sma_200
                    and sma_50 > 0
                    and sma_200 > 0
                )

                # Calculate trend strength
                trend_strength = (sma_50 - sma_200) / sma_200 if sma_200 > 0 else 0

                # Volume analysis
                current_volume = latest.get("Volume", 0)
                avg_volume = latest.get("Volume_SMA_20", 0)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

                # Current price analysis
                current_price = latest.get("Close", 0)
                price_vs_sma50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
                price_vs_sma200 = (
                    (current_price - sma_200) / sma_200 if sma_200 > 0 else 0
                )

                # Generate signal recommendation
                signal = "HOLD"
                confidence = 0.0

                if golden_cross and volume_ratio > 1.1:
                    signal = "BUY"
                    confidence = min(
                        0.9, 0.7 + (trend_strength * 10) + (volume_ratio - 1.0)
                    )
                elif death_cross:
                    signal = "SELL"
                    confidence = 0.8
                elif sma_50 > sma_200 and trend_strength > 0.02:
                    signal = "HOLD_LONG"
                    confidence = 0.6
                elif sma_50 < sma_200 and trend_strength < -0.02:
                    signal = "HOLD_CASH"
                    confidence = 0.6

                analysis_results[symbol] = {
                    "current_price": current_price,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "trend_strength": trend_strength,
                    "volume_ratio": volume_ratio,
                    "price_vs_sma50": price_vs_sma50,
                    "price_vs_sma200": price_vs_sma200,
                    "golden_cross": golden_cross,
                    "death_cross": death_cross,
                    "signal": signal,
                    "confidence": confidence,
                    "last_updated": datetime.now(),
                }

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                analysis_results[symbol] = {"error": str(e)}

        return analysis_results

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        try:
            portfolio = self.alpaca_client.get_portfolio_summary()
            positions = self.alpaca_client.get_positions()

            return {
                "portfolio": portfolio,
                "positions": positions,
                "position_count": len(positions),
                "total_positions_value": sum(p["market_value"] for p in positions),
                "total_unrealized_pnl": sum(p["unrealized_pnl"] for p in positions),
            }
        except Exception as e:
            logger.error(f"Error getting portfolio status: {str(e)}")
            return {}

    def generate_trading_recommendations(
        self, analysis_results: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Generate trading recommendations based on analysis.

        Args:
            analysis_results: Analysis results from analyze_golden_cross_opportunities

        Returns:
            List of trading recommendations
        """
        recommendations = []

        # Get portfolio status
        portfolio_status = self.get_portfolio_status()
        available_cash = portfolio_status.get("portfolio", {}).get("buying_power", 0)

        # Filter for strong buy signals
        buy_signals = [
            (symbol, data)
            for symbol, data in analysis_results.items()
            if data.get("signal") == "BUY" and data.get("confidence", 0) > 0.7
        ]

        # Sort by confidence
        buy_signals.sort(key=lambda x: x[1].get("confidence", 0), reverse=True)

        for symbol, data in buy_signals:
            current_price = data.get("current_price", 0)
            confidence = data.get("confidence", 0)

            if current_price > 0:
                # Calculate position size (5% of available cash per position)
                position_value = available_cash * 0.05
                shares = int(position_value / current_price)

                if shares > 0:
                    recommendations.append(
                        {
                            "symbol": symbol,
                            "action": "BUY",
                            "shares": shares,
                            "price": current_price,
                            "value": shares * current_price,
                            "confidence": confidence,
                            "reason": f"Golden Cross detected with {confidence:.1%} confidence",
                            "trend_strength": data.get("trend_strength", 0),
                            "volume_ratio": data.get("volume_ratio", 0),
                        }
                    )

        # Check for sell signals (existing positions)
        positions = portfolio_status.get("positions", [])
        for position in positions:
            symbol = position["symbol"]
            if symbol in analysis_results:
                data = analysis_results[symbol]
                if data.get("signal") == "SELL" and data.get("confidence", 0) > 0.7:
                    recommendations.append(
                        {
                            "symbol": symbol,
                            "action": "SELL",
                            "shares": position["quantity"],
                            "price": data.get("current_price", 0),
                            "value": position["quantity"]
                            * data.get("current_price", 0),
                            "confidence": data.get("confidence", 0),
                            "reason": "Death Cross detected",
                            "unrealized_pnl": position["unrealized_pnl"],
                        }
                    )

        return recommendations

    def execute_recommendations(
        self, recommendations: List[Dict], dry_run: bool = True
    ) -> List[Dict]:
        """
        Execute trading recommendations.

        Args:
            recommendations: List of trading recommendations
            dry_run: If True, only simulate execution

        Returns:
            List of execution results
        """
        results = []

        for rec in recommendations:
            try:
                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would {rec['action']} {rec['shares']} shares of {rec['symbol']} at ~${rec['price']:.2f}"
                    )
                    results.append(
                        {
                            "symbol": rec["symbol"],
                            "action": rec["action"],
                            "status": "DRY_RUN",
                            "message": f"Would execute {rec['action']} order",
                        }
                    )
                else:
                    # Create signal object
                    from strategies.base import StrategySignal, SignalType

                    signal = StrategySignal(
                        symbol=rec["symbol"],
                        signal_type=(
                            SignalType.BUY
                            if rec["action"] == "BUY"
                            else SignalType.SELL
                        ),
                        timestamp=datetime.now(),
                        confidence=rec["confidence"],
                        metadata={
                            "reason": rec["reason"],
                            "trend_strength": rec.get("trend_strength", 0),
                            "volume_ratio": rec.get("volume_ratio", 0),
                        },
                    )

                    # Execute signal
                    success = self.alpaca_client.execute_signal(signal)

                    results.append(
                        {
                            "symbol": rec["symbol"],
                            "action": rec["action"],
                            "status": "EXECUTED" if success else "FAILED",
                            "message": f"Order {'executed' if success else 'failed'}",
                        }
                    )

            except Exception as e:
                logger.error(
                    f"Error executing recommendation for {rec['symbol']}: {str(e)}"
                )
                results.append(
                    {
                        "symbol": rec["symbol"],
                        "action": rec["action"],
                        "status": "ERROR",
                        "message": str(e),
                    }
                )

        return results

    def run_analysis(self, dry_run: bool = True) -> Dict:
        """
        Run complete Golden Cross analysis.

        Args:
            dry_run: If True, only simulate trading

        Returns:
            Complete analysis results
        """
        logger.info("🚀 Starting Golden Cross Live Analysis...")

        # Test connections
        if not self.test_connection():
            raise Exception("Connection test failed")

        # Get market data
        market_data = self.get_current_market_data()

        if not market_data:
            raise Exception("No market data available")

        # Analyze opportunities
        analysis_results = self.analyze_golden_cross_opportunities(market_data)

        # Get portfolio status
        portfolio_status = self.get_portfolio_status()

        # Generate recommendations
        recommendations = self.generate_trading_recommendations(analysis_results)

        # Execute recommendations
        execution_results = self.execute_recommendations(
            recommendations, dry_run=dry_run
        )

        return {
            "timestamp": datetime.now(),
            "market_data_count": len(market_data),
            "analysis_results": analysis_results,
            "portfolio_status": portfolio_status,
            "recommendations": recommendations,
            "execution_results": execution_results,
            "dry_run": dry_run,
        }


def print_analysis_report(results: Dict):
    """Print a formatted analysis report."""
    print("\n" + "=" * 80)
    print("🎯 GOLDEN CROSS LIVE ANALYSIS REPORT")
    print("=" * 80)
    print(f"📅 Analysis Time: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Symbols Analyzed: {results['market_data_count']}")
    print(f"🔍 Analysis Mode: {'DRY RUN' if results['dry_run'] else 'LIVE TRADING'}")

    # Asset class breakdown
    analysis = results.get("analysis_results", {})
    asset_classes = {}
    for symbol in analysis.keys():
        asset_class = categorize_asset(symbol)
        asset_classes[asset_class] = asset_classes.get(asset_class, 0) + 1

    print(f"\n📈 ASSET CLASS BREAKDOWN:")
    for asset_class, count in sorted(asset_classes.items()):
        print(f"   {asset_class}: {count} assets")

    # Portfolio Summary
    portfolio = results.get("portfolio_status", {}).get("portfolio", {})
    if portfolio:
        print(f"\n💰 PORTFOLIO SUMMARY:")
        print(f"   Total Value: ${portfolio.get('total_value', 0):,.2f}")
        print(f"   Cash: ${portfolio.get('cash', 0):,.2f}")
        print(f"   Positions Value: ${portfolio.get('positions_value', 0):,.2f}")
        print(f"   Unrealized P&L: ${portfolio.get('unrealized_pnl', 0):,.2f}")
        print(f"   Buying Power: ${portfolio.get('buying_power', 0):,.2f}")

    # Golden Cross Opportunities
    analysis = results.get("analysis_results", {})
    golden_crosses = [
        (symbol, data)
        for symbol, data in analysis.items()
        if data.get("golden_cross", False)
    ]

    if golden_crosses:
        print(f"\n🟢 GOLDEN CROSS OPPORTUNITIES ({len(golden_crosses)}):")
        for symbol, data in golden_crosses:
            asset_class = categorize_asset(symbol)
            print(f"   {symbol} ({asset_class}): ${data.get('current_price', 0):.2f}")
            print(
                f"     50MA: ${data.get('sma_50', 0):.2f} | 200MA: ${data.get('sma_200', 0):.2f}"
            )
            print(f"     Trend Strength: {data.get('trend_strength', 0):.2%}")
            print(f"     Volume Ratio: {data.get('volume_ratio', 0):.1f}x")
            print(f"     Confidence: {data.get('confidence', 0):.1%}")

    # Death Cross Warnings
    death_crosses = [
        (symbol, data)
        for symbol, data in analysis.items()
        if data.get("death_cross", False)
    ]

    if death_crosses:
        print(f"\n🔴 DEATH CROSS WARNINGS ({len(death_crosses)}):")
        for symbol, data in death_crosses:
            print(f"   {symbol}: ${data.get('current_price', 0):.2f}")
            print(
                f"     50MA: ${data.get('sma_50', 0):.2f} | 200MA: ${data.get('sma_200', 0):.2f}"
            )
            print(f"     Trend Strength: {data.get('trend_strength', 0):.2%}")

    # Trading Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\n📈 TRADING RECOMMENDATIONS ({len(recommendations)}):")
        for rec in recommendations:
            action_emoji = "🟢" if rec["action"] == "BUY" else "🔴"
            print(
                f"   {action_emoji} {rec['action']} {rec['shares']} shares of {rec['symbol']}"
            )
            print(f"      Price: ${rec['price']:.2f} | Value: ${rec['value']:,.2f}")
            print(f"      Confidence: {rec['confidence']:.1%}")
            print(f"      Reason: {rec['reason']}")

    # Execution Results
    execution_results = results.get("execution_results", [])
    if execution_results:
        print(f"\n⚡ EXECUTION RESULTS:")
        for result in execution_results:
            status_emoji = "✅" if result["status"] == "EXECUTED" else "❌"
            print(
                f"   {status_emoji} {result['symbol']} {result['action']}: {result['status']}"
            )
            print(f"      {result['message']}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Golden Cross Live Analysis")
    parser.add_argument(
        "--live", action="store_true", help="Execute live trades (default: dry run)"
    )
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to analyze")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize analyzer
        analyzer = GoldenCrossAnalyzer()

        # Override symbols if specified
        if args.symbols:
            analyzer.symbols = args.symbols

        # Run analysis
        results = analyzer.run_analysis(dry_run=not args.live)

        # Print report
        print_analysis_report(results)

        if not args.live:
            print("\n💡 To execute live trades, run with --live flag")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
