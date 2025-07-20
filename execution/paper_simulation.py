#!/usr/bin/env python3
"""
Paper Trading Simulation for Golden Cross Strategy

This script demonstrates the Golden Cross strategy in action, showing:
1. Current market analysis for SPY, QQQ, VTI
2. Buy signals based on 50/200 MA crossovers
3. Position sizing recommendations for $500-1000 account
4. Risk metrics and expected performance
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.equity.golden_cross import GoldenCrossStrategy
from execution.paper import PaperTradingSimulator
from indicators.technical import TechnicalIndicators

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GoldenCrossPaperTrading:
    """
    Paper trading simulation specifically for Golden Cross strategy.
    """

    def __init__(self, initial_capital: float = 1000):
        """
        Initialize the paper trading simulation.

        Args:
            initial_capital: Starting capital for the simulation
        """
        self.initial_capital = initial_capital
        self.symbols = ["SPY", "QQQ", "VTI"]
        self.strategy = GoldenCrossStrategy(symbols=self.symbols)
        self.paper_trader = PaperTradingSimulator(initial_capital=initial_capital)

        # Market data cache
        self.market_data = {}

    def fetch_current_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch current market data for analysis.

        Returns:
            Dictionary mapping symbol -> DataFrame with OHLCV data
        """
        logger.info("Fetching current market data...")

        for symbol in self.symbols:
            try:
                # Fetch 300 days of data to ensure we have enough for 200-day MA
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="300d")

                if not data.empty:
                    # Ensure we have the required columns
                    required_cols = ["Open", "High", "Low", "Close", "Volume"]
                    if all(col in data.columns for col in required_cols):
                        # Add Adj Close if not present
                        if "Adj Close" not in data.columns:
                            data["Adj Close"] = data["Close"]

                        self.market_data[symbol] = data
                        logger.info(f"✓ Loaded {len(data)} days of data for {symbol}")
                    else:
                        logger.warning(f"Missing required columns for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")

        return self.market_data

    def analyze_golden_cross_signals(self) -> List:
        """
        Analyze current market data for Golden Cross signals.

        Returns:
            List of trading signals
        """
        if not self.market_data:
            logger.error("No market data available for analysis")
            return []

        logger.info("Analyzing Golden Cross signals...")

        try:
            signals = self.strategy.generate_signals(self.market_data)
            logger.info(f"Generated {len(signals)} signals")
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []

    def calculate_position_sizes(
        self, signals: List, account_size: float = 1000
    ) -> Dict:
        """
        Calculate optimal position sizes for each signal.

        Args:
            signals: List of trading signals
            account_size: Total account size

        Returns:
            Dictionary with position sizing recommendations
        """
        position_sizes = {}

        for signal in signals:
            if signal.signal_type.value == "BUY":
                symbol = signal.symbol
                current_price = signal.price

                # Golden Cross strategy: 30% max per ETF position
                max_position_value = account_size * 0.30

                # Calculate shares based on current price
                shares = int(max_position_value / current_price)

                # Ensure minimum position size
                if shares * current_price < 100:  # Minimum $100 position
                    shares = int(100 / current_price)

                position_value = shares * current_price

                position_sizes[symbol] = {
                    "shares": shares,
                    "position_value": position_value,
                    "confidence": signal.confidence,
                    "current_price": current_price,
                    "conditions_met": signal.metadata.get("conditions_met", []),
                    "ma_separation_pct": signal.metadata.get("ma_separation_pct", 0),
                }

        return position_sizes

    def get_market_analysis(self) -> Dict:
        """
        Get current market analysis for each symbol.

        Returns:
            Dictionary with market analysis
        """
        analysis = {}

        for symbol, data in self.market_data.items():
            try:
                # Calculate technical indicators
                indicators = TechnicalIndicators(data.copy())
                indicators.add_sma(50)
                indicators.add_sma(200)
                indicators.add_volume_sma(20)

                analysis_data = indicators.get_data()
                latest = analysis_data.iloc[-1]

                # Get current values
                sma_50 = latest.get("SMA_50")
                sma_200 = latest.get("SMA_200")
                current_price = latest["Close"]
                volume = latest["Volume"]
                volume_sma = latest.get("Volume_SMA_20")

                # Calculate metrics
                ma_separation_pct = 0
                if sma_50 and sma_200:
                    ma_separation_pct = abs(sma_50 - sma_200) / sma_200 * 100

                volume_ratio = 1.0
                if volume_sma:
                    volume_ratio = volume / volume_sma

                # Determine trend status
                trend_status = "NEUTRAL"
                if sma_50 and sma_200:
                    if sma_50 > sma_200:
                        trend_status = "BULLISH"
                    else:
                        trend_status = "BEARISH"

                analysis[symbol] = {
                    "current_price": current_price,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "ma_separation_pct": ma_separation_pct,
                    "trend_status": trend_status,
                    "volume_ratio": volume_ratio,
                    "days_of_data": len(data),
                }

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")

        return analysis

    def run_simulation(self) -> Dict:
        """
        Run the complete paper trading simulation.

        Returns:
            Dictionary with simulation results
        """
        logger.info("=" * 60)
        logger.info("GOLDEN CROSS PAPER TRADING SIMULATION")
        logger.info("=" * 60)

        # Step 1: Fetch market data
        self.fetch_current_data()

        # Step 2: Analyze current market conditions
        market_analysis = self.get_market_analysis()

        # Step 3: Generate trading signals
        signals = self.analyze_golden_cross_signals()

        # Step 4: Calculate position sizes
        position_sizes = self.calculate_position_sizes(signals, self.initial_capital)

        # Step 5: Execute signals in paper trading
        executed_trades = []
        for signal in signals:
            if self.paper_trader.execute_signal(signal):
                executed_trades.append(signal)

        # Step 6: Get portfolio summary
        portfolio_summary = self.paper_trader.get_portfolio_summary()

        return {
            "market_analysis": market_analysis,
            "signals": signals,
            "position_sizes": position_sizes,
            "executed_trades": executed_trades,
            "portfolio_summary": portfolio_summary,
            "simulation_date": datetime.now(),
        }

    def display_results(self, results: Dict):
        """
        Display the simulation results in a formatted way.

        Args:
            results: Simulation results dictionary
        """
        print("\n" + "=" * 80)
        print("GOLDEN CROSS STRATEGY - PAPER TRADING SIMULATION RESULTS")
        print("=" * 80)
        print(
            f"Simulation Date: {results['simulation_date'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"Account Size: ${self.initial_capital:,.2f}")
        print()

        # Market Analysis
        print("📊 CURRENT MARKET ANALYSIS")
        print("-" * 40)
        for symbol, analysis in results["market_analysis"].items():
            print(f"\n{symbol}:")
            print(f"  Current Price: ${analysis['current_price']:.2f}")
            print(
                f"  50-day SMA: ${analysis['sma_50']:.2f}"
                if analysis["sma_50"]
                else "  50-day SMA: N/A"
            )
            print(
                f"  200-day SMA: ${analysis['sma_200']:.2f}"
                if analysis["sma_200"]
                else "  200-day SMA: N/A"
            )
            print(f"  MA Separation: {analysis['ma_separation_pct']:.2f}%")
            print(f"  Trend Status: {analysis['trend_status']}")
            print(f"  Volume Ratio: {analysis['volume_ratio']:.2f}x average")

        # Trading Signals
        print(f"\n🎯 TRADING SIGNALS ({len(results['signals'])} generated)")
        print("-" * 40)

        if results["signals"]:
            for signal in results["signals"]:
                print(f"\n{signal.symbol} {signal.signal_type.value}:")
                print(f"  Confidence: {signal.confidence:.1%}")
                print(f"  Price: ${signal.price:.2f}")
                print(
                    f"  Conditions: {', '.join(signal.metadata.get('conditions_met', []))}"
                )
                if "ma_separation_pct" in signal.metadata:
                    print(
                        f"  MA Separation: {signal.metadata['ma_separation_pct']:.2f}%"
                    )
        else:
            print("No trading signals generated at this time.")
            print(
                "This is normal - Golden Cross typically generates 2-4 signals per year."
            )

        # Position Sizing Recommendations
        print(f"\n💰 POSITION SIZING RECOMMENDATIONS")
        print("-" * 40)

        if results["position_sizes"]:
            total_allocated = 0
            for symbol, sizing in results["position_sizes"].items():
                print(f"\n{symbol}:")
                print(f"  Shares: {sizing['shares']}")
                print(f"  Position Value: ${sizing['position_value']:.2f}")
                print(f"  Confidence: {sizing['confidence']:.1%}")
                print(
                    f"  % of Account: {(sizing['position_value']/self.initial_capital)*100:.1f}%"
                )
                total_allocated += sizing["position_value"]

            print(
                f"\nTotal Allocated: ${total_allocated:.2f} ({(total_allocated/self.initial_capital)*100:.1f}% of account)"
            )
            print(f"Remaining Cash: ${self.initial_capital - total_allocated:.2f}")
        else:
            print("No positions recommended at this time.")

        # Portfolio Summary
        print(f"\n📈 PORTFOLIO SUMMARY")
        print("-" * 40)
        portfolio = results["portfolio_summary"]
        print(f"Current Cash: ${portfolio.get('current_cash', self.initial_capital):.2f}")
        print(f"Total Positions: {len(portfolio.get('positions', []))}")
        print(f"Portfolio Value: ${portfolio.get('total_value', self.initial_capital):.2f}")
        print(f"Total Return: {portfolio.get('total_return_pct', 0):.2f}%")

        # Risk Metrics
        print(f"\n⚠️  RISK METRICS")
        print("-" * 40)
        print("• Maximum 30% per ETF position (Golden Cross rule)")
        print("• No stop losses (trust the crossover)")
        print("• Typically 2-4 trades per year")
        print("• Expected annual return: 8-12%")
        print("• Maximum drawdown: 8-15%")
        print("• Tax efficient (long-term capital gains)")

        # Next Steps
        print(f"\n🚀 NEXT STEPS")
        print("-" * 40)
        print("1. Monitor signals weekly")
        print("2. Execute trades manually for now")
        print("3. Track performance over 3-6 months")
        print("4. Consider automation once confident")
        print("5. Scale up position sizes gradually")

        print("\n" + "=" * 80)


def main():
    """Main function to run the paper trading simulation."""
    try:
        # Initialize simulation
        simulation = GoldenCrossPaperTrading(initial_capital=1000)

        # Run simulation
        results = simulation.run_simulation()

        # Display results
        simulation.display_results(results)

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
