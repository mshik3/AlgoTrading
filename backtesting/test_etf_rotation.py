"""
Backtesting scenarios for ETF rotation strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from strategies.etf.dual_momentum import DualMomentumStrategy
from strategies.etf.sector_rotation import SectorRotationStrategy
from backtesting.engine import BacktestingEngine
from utils.asset_categorization import get_etf_universe_for_strategy

logger = logging.getLogger(__name__)


def create_sample_market_data(
    symbols: list,
    start_date: datetime = None,
    end_date: datetime = None,
    days: int = 500,
) -> dict:
    """
    Create sample market data for backtesting.

    Args:
        symbols: List of symbols to create data for
        start_date: Start date for data
        end_date: End date for data
        days: Number of days if start/end dates not provided

    Returns:
        Dictionary mapping symbol -> OHLCV DataFrame
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    if end_date is None:
        end_date = datetime.now()

    dates = pd.date_range(start_date, end_date, freq="D")
    market_data = {}

    # Create different performance patterns for different asset types
    performance_patterns = {
        "SPY": 0.08,  # Moderate growth
        "QQQ": 0.12,  # High growth
        "TLT": 0.02,  # Low growth (bonds)
        "XLK": 0.15,  # Technology sector
        "XLF": 0.06,  # Financial sector
        "XLE": 0.04,  # Energy sector
        "XLV": 0.09,  # Healthcare sector
        "GLD": 0.03,  # Gold
        "EFA": 0.05,  # International
        "EEM": 0.07,  # Emerging markets
    }

    for symbol in symbols:
        # Get performance pattern or use default
        daily_return = performance_patterns.get(symbol, 0.06) / 252

        # Add some volatility
        volatility = 0.02

        # Generate price series
        prices = [100.0]  # Start at $100
        for i in range(1, len(dates)):
            # Random walk with drift
            change = np.random.normal(daily_return, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Don't go below $1

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "Open": [p * (1 + np.random.normal(0, 0.001)) for p in prices],
                "High": [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
                "Close": prices,
                "Volume": [np.random.randint(1000000, 10000000) for _ in prices],
            },
            index=dates,
        )

        market_data[symbol] = data

    return market_data


def test_dual_momentum_strategy():
    """Test dual momentum strategy backtesting."""
    logger.info("Testing Dual Momentum Strategy...")

    # Create strategy
    etf_universe = get_etf_universe_for_strategy("dual_momentum")
    strategy = DualMomentumStrategy(etf_universe=etf_universe)

    # Create market data
    symbols = ["SPY", "QQQ", "TLT", "GLD", "EFA", "EEM"]
    market_data = create_sample_market_data(symbols, days=600)

    # Create backtesting engine
    engine = BacktestingEngine(initial_capital=100000)

    # Run backtest
    result = engine.run_backtest(strategy, market_data)

    # Print results
    logger.info(f"Dual Momentum Strategy Results:")
    logger.info(f"Total Return: {result.total_return_pct:.2f}%")
    logger.info(f"Number of Trades: {len(result.trades)}")
    logger.info(f"Final Capital: ${result.final_capital:,.2f}")

    # Print strategy summary
    summary = strategy.get_dual_momentum_summary()
    logger.info(f"Current Asset: {summary.get('current_asset', 'None')}")
    logger.info(f"Defensive Mode: {summary.get('defensive_mode', False)}")

    return result


def test_sector_rotation_strategy():
    """Test sector rotation strategy backtesting."""
    logger.info("Testing Sector Rotation Strategy...")

    # Create strategy
    etf_universe = get_etf_universe_for_strategy("sector_rotation")
    strategy = SectorRotationStrategy(etf_universe=etf_universe)

    # Create market data including benchmark
    symbols = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU"]
    market_data = create_sample_market_data(symbols, days=600)

    # Create backtesting engine
    engine = BacktestingEngine(initial_capital=100000)

    # Run backtest
    result = engine.run_backtest(strategy, market_data)

    # Print results
    logger.info(f"Sector Rotation Strategy Results:")
    logger.info(f"Total Return: {result.total_return_pct:.2f}%")
    logger.info(f"Number of Trades: {len(result.trades)}")
    logger.info(f"Final Capital: ${result.final_capital:,.2f}")

    # Print strategy summary
    summary = strategy.get_sector_rotation_summary()
    logger.info(f"Top Sectors: {list(summary.get('sector_rankings', {}).keys())[:3]}")

    return result


def test_momentum_ranking_accuracy():
    """Test momentum ranking accuracy."""
    logger.info("Testing Momentum Ranking Accuracy...")

    # Create market data with known performance patterns
    symbols = ["SPY", "QQQ", "TLT"]
    market_data = create_sample_market_data(symbols, days=400)

    # Create strategy
    strategy = DualMomentumStrategy()

    # Test momentum calculations
    for symbol in symbols:
        data = market_data[symbol]
        momentum = strategy.calculate_momentum(data, lookback=252)
        logger.info(f"{symbol} Momentum: {momentum:.3f}")

    # Test relative momentum ranking
    qualified_assets = ["SPY", "QQQ"]
    relative_momentums = strategy.calculate_relative_momentum(
        market_data, qualified_assets
    )

    logger.info("Relative Momentum Rankings:")
    for symbol, momentum in relative_momentums:
        logger.info(f"  {symbol}: {momentum:.3f}")

    return relative_momentums


def test_rebalancing_frequency():
    """Test rebalancing frequency impact."""
    logger.info("Testing Rebalancing Frequency Impact...")

    # Create strategy with different rebalancing frequencies
    etf_universe = get_etf_universe_for_strategy("dual_momentum")

    frequencies = [7, 21, 63]  # Weekly, monthly, quarterly
    results = {}

    for freq in frequencies:
        strategy = DualMomentumStrategy(
            etf_universe=etf_universe, rebalance_frequency=freq
        )

        symbols = ["SPY", "QQQ", "TLT", "GLD"]
        market_data = create_sample_market_data(symbols, days=600)

        engine = BacktestingEngine(initial_capital=100000)
        result = engine.run_backtest(strategy, market_data)

        results[freq] = {
            "total_return": result.total_return_pct,
            "trades": len(result.trades),
            "final_capital": result.final_capital,
        }

        logger.info(f"Rebalancing every {freq} days:")
        logger.info(f"  Return: {result.total_return_pct:.2f}%")
        logger.info(f"  Trades: {len(result.trades)}")

    return results


def test_defensive_mode():
    """Test defensive mode functionality."""
    logger.info("Testing Defensive Mode...")

    # Create market data with declining trend
    symbols = ["SPY", "QQQ", "TLT", "SHY"]
    dates = pd.date_range(
        datetime.now() - timedelta(days=400), datetime.now(), freq="D"
    )

    market_data = {}
    for symbol in symbols:
        if symbol in ["SPY", "QQQ"]:
            # Declining trend for stocks
            prices = [100.0]
            for i in range(1, len(dates)):
                change = np.random.normal(-0.001, 0.02)  # Negative drift
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1.0))
        else:
            # Stable trend for bonds/cash
            prices = [100.0]
            for i in range(1, len(dates)):
                change = np.random.normal(0.0001, 0.005)  # Slight positive drift
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1.0))

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.001 for p in prices],
                "Low": [p * 0.999 for p in prices],
                "Close": prices,
                "Volume": [1000000] * len(prices),
            },
            index=dates,
        )

        market_data[symbol] = data

    # Create strategy
    etf_universe = get_etf_universe_for_strategy("dual_momentum")
    strategy = DualMomentumStrategy(etf_universe=etf_universe)

    # Run backtest
    engine = BacktestingEngine(initial_capital=100000)
    result = engine.run_backtest(strategy, market_data)

    # Check if strategy went defensive
    summary = strategy.get_dual_momentum_summary()
    logger.info(f"Defensive Mode: {summary.get('defensive_mode', False)}")
    logger.info(f"Current Asset: {summary.get('current_asset', 'None')}")
    logger.info(f"Total Return: {result.total_return_pct:.2f}%")

    return result


def run_all_etf_rotation_tests():
    """Run all ETF rotation backtesting scenarios."""
    logger.info("Running ETF Rotation Strategy Backtests...")

    results = {}

    try:
        # Test dual momentum strategy
        results["dual_momentum"] = test_dual_momentum_strategy()
    except Exception as e:
        logger.error(f"Dual momentum test failed: {str(e)}")

    try:
        # Test sector rotation strategy
        results["sector_rotation"] = test_sector_rotation_strategy()
    except Exception as e:
        logger.error(f"Sector rotation test failed: {str(e)}")

    try:
        # Test momentum ranking accuracy
        results["momentum_ranking"] = test_momentum_ranking_accuracy()
    except Exception as e:
        logger.error(f"Momentum ranking test failed: {str(e)}")

    try:
        # Test rebalancing frequency impact
        results["rebalancing_frequency"] = test_rebalancing_frequency()
    except Exception as e:
        logger.error(f"Rebalancing frequency test failed: {str(e)}")

    try:
        # Test defensive mode
        results["defensive_mode"] = test_defensive_mode()
    except Exception as e:
        logger.error(f"Defensive mode test failed: {str(e)}")

    logger.info("ETF Rotation Strategy Backtests Completed!")
    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run all tests
    results = run_all_etf_rotation_tests()

    # Print summary
    print("\n" + "=" * 50)
    print("ETF ROTATION STRATEGY BACKTESTING SUMMARY")
    print("=" * 50)

    for test_name, result in results.items():
        if hasattr(result, "total_return_pct"):
            print(f"{test_name.upper()}: {result.total_return_pct:.2f}% return")
        elif isinstance(result, dict):
            print(f"{test_name.upper()}: {len(result)} frequency tests completed")
        else:
            print(f"{test_name.upper()}: Test completed")
