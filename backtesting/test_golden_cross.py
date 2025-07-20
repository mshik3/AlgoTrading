"""
Test script for Golden Cross strategy backtesting.
Validates the strategy performance on historical SPY, QQQ, and VTI data.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from data import get_engine, get_session, MarketData
from strategies.equity.golden_cross import GoldenCrossStrategy
from backtesting import BacktestingEngine, PerformanceMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_historical_data(symbols, start_date=None, end_date=None):
    """
    Load historical market data from the database.

    Args:
        symbols: List of symbols to load
        start_date: Start date (optional)
        end_date: End date (optional)

    Returns:
        Dictionary mapping symbol -> DataFrame with OHLCV data
    """
    engine = get_engine()
    session = get_session(engine)

    market_data = {}

    for symbol in symbols:
        try:
            query = session.query(MarketData).filter(MarketData.symbol == symbol)

            if start_date:
                query = query.filter(MarketData.date >= start_date)
            if end_date:
                query = query.filter(MarketData.date <= end_date)

            query = query.order_by(MarketData.date)

            data = []
            for record in query.all():
                data.append(
                    {
                        "Date": record.date,
                        "Open": record.open,
                        "High": record.high,
                        "Low": record.low,
                        "Close": record.close,
                        "Volume": record.volume,
                        "Adj Close": record.adjusted_close,
                    }
                )

            if data:
                df = pd.DataFrame(data)
                df.set_index("Date", inplace=True)
                market_data[symbol] = df
                logger.info(f"Loaded {len(df)} records for {symbol}")
            else:
                logger.warning(f"No data found for {symbol}")

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")

    session.close()
    return market_data


def run_golden_cross_backtest():
    """Run a comprehensive backtest of the Golden Cross strategy."""
    logger.info("Starting Golden Cross strategy backtest...")

    # Test parameters
    symbols = ["SPY", "QQQ", "VTI"]
    initial_capital = 10000

    # Date range (last 5 years or available data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)  # ~5 years

    # Load historical data
    logger.info(
        f"Loading historical data for {symbols} from {start_date.date()} to {end_date.date()}"
    )
    market_data = load_historical_data(symbols, start_date, end_date)

    if not market_data:
        logger.error(
            "No market data loaded. Make sure you have run data collection first."
        )
        return None

    # Initialize strategy
    strategy = GoldenCrossStrategy(symbols=list(market_data.keys()))

    # Initialize backtesting engine
    backtest_engine = BacktestingEngine(
        initial_capital=initial_capital,
        commission_per_trade=0.0,  # Commission-free with Alpaca
        slippage_pct=0.001,  # 0.1% slippage
    )

    # Run backtest
    logger.info("Running backtest...")
    try:
        result = backtest_engine.run_backtest(
            strategy=strategy,
            market_data=market_data,
            start_date=start_date,
            end_date=end_date,
        )

        # Display results
        print_backtest_results(result)

        return result

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def print_backtest_results(result):
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print(f"GOLDEN CROSS STRATEGY BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nStrategy: {result.strategy_name}")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital: ${result.final_capital:,.2f}")
    print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")

    # Performance metrics
    metrics = result.performance_metrics
    metrics_calculator = PerformanceMetrics()
    print(f"\n{metrics_calculator.format_metrics_report(metrics)}")

    # Trade details
    if result.trades:
        print(f"\n=== TRADE DETAILS ===")
        print(f"Total Trades: {len(result.trades)}")

        winning_trades = [t for t in result.trades if t.profit_loss > 0]
        losing_trades = [t for t in result.trades if t.profit_loss <= 0]

        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")

        if result.trades:
            print(f"\nTrade History:")
            for i, trade in enumerate(result.trades[-10:], 1):  # Show last 10 trades
                action = (
                    "BUY" if trade.entry_signal.signal_type.value == "BUY" else "SELL"
                )
                print(
                    f"{i:2d}. {trade.symbol} {action} {trade.entry_date.date()} -> "
                    f"SELL {trade.exit_date.date()}: "
                    f"${trade.entry_price:.2f} -> ${trade.exit_price:.2f} = "
                    f"{trade.profit_loss_pct:+.2f}% (${trade.profit_loss:+.2f})"
                )

    # Signals generated
    if result.signals_generated:
        print(f"\n=== SIGNAL ANALYSIS ===")
        buy_signals = [
            s for s in result.signals_generated if s.signal_type.value == "BUY"
        ]
        sell_signals = [
            s
            for s in result.signals_generated
            if s.signal_type.value in ["SELL", "CLOSE_LONG"]
        ]

        print(f"Buy Signals Generated: {len(buy_signals)}")
        print(f"Sell Signals Generated: {len(sell_signals)}")

        # Show recent signals
        recent_signals = (
            result.signals_generated[-5:] if result.signals_generated else []
        )
        if recent_signals:
            print(f"\nRecent Signals:")
            for signal in recent_signals:
                print(
                    f"  {signal.symbol} {signal.signal_type.value} on {signal.timestamp.date()} "
                    f"(confidence: {signal.confidence:.2f})"
                )


def validate_strategy_logic():
    """Validate the Golden Cross strategy logic with sample data."""
    print("\n" + "=" * 60)
    print("VALIDATING GOLDEN CROSS STRATEGY LOGIC")
    print("=" * 60)

    # Create sample data to test Golden Cross detection
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")

    # Create synthetic data with a Golden Cross pattern
    prices = []
    base_price = 100

    for i, date in enumerate(dates):
        # Create uptrend after day 100 to trigger Golden Cross
        if i < 100:
            price = base_price + (i * 0.1)  # Slight uptrend
        else:
            price = base_price + (i * 0.5)  # Stronger uptrend

        # Add some noise
        price += (i % 7 - 3) * 0.5
        prices.append(price)

    # Create DataFrame
    sample_data = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [100000] * len(prices),
            "Adj Close": prices,
        },
        index=dates,
    )

    market_data = {"TEST": sample_data}

    # Test strategy
    strategy = GoldenCrossStrategy(symbols=["TEST"])

    # Test signal generation on the last 30 days
    recent_data = {"TEST": sample_data.tail(250)}  # Need enough data for 200-day MA

    try:
        signals = strategy.generate_signals(recent_data)

        print(f"Generated {len(signals)} signals on test data")
        for signal in signals:
            print(
                f"  {signal.symbol} {signal.signal_type.value} (confidence: {signal.confidence:.2f})"
            )
            if hasattr(signal, "metadata") and signal.metadata:
                conditions = signal.metadata.get("conditions_met", [])
                print(f"    Conditions: {conditions}")

        print("✓ Strategy logic validation completed successfully")

    except Exception as e:
        print(f"✗ Strategy validation failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Golden Cross Strategy Testing Suite")
    print("=" * 50)

    # First validate the strategy logic
    validate_strategy_logic()

    # Then run the actual backtest
    result = run_golden_cross_backtest()

    if result:
        print(f"\n✓ Backtest completed successfully!")
        print(f"Strategy shows {result.total_return_pct:.2f}% total return")

        # Basic validation
        if result.total_return_pct > 0:
            print("✓ Strategy is profitable")
        else:
            print("⚠ Strategy shows negative returns")

        if result.performance_metrics.get("sharpe_ratio", 0) > 1.0:
            print("✓ Good risk-adjusted returns (Sharpe > 1.0)")
        else:
            print("⚠ Low risk-adjusted returns")
    else:
        print("✗ Backtest failed - check your database setup and data collection")
