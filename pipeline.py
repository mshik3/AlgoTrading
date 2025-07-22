"""
Main orchestration module for the algorithmic trading system.
Coordinates the data collection, strategy execution, and trade generation.
"""

import os
import logging
import argparse
import time
from datetime import datetime, timedelta

from data import (
    get_collector,
    DataProcessor,
    get_engine,
    init_db,
    get_session,
    save_market_data,
    get_active_symbols,
    save_symbol,
    log_collection_start,
    log_collection_end,
    MarketData,
)

# Import strategy components
from strategies.equity.golden_cross import GoldenCrossStrategy
from backtesting import BacktestingEngine, PerformanceMetrics

# Import validation utilities
try:
    from utils.validators import validate_symbols, validate_period
    from utils.config import load_environment, validate_required_env_vars
except ImportError as e:
    logger.warning(f"Could not import utilities: {e}")

    # Provide fallback functions
    def validate_symbols(symbols):
        return symbols if symbols else []

    def validate_period(period):
        return period

    def load_environment():
        return True

    def validate_required_env_vars():
        return {"valid": True}


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_database():
    """Initialize the database and return an engine."""
    logger.info("Initializing database...")
    engine = get_engine()
    success = init_db(engine)
    if success:
        logger.info("Database initialized successfully")
    else:
        logger.error("Database initialization failed")
    return engine if success else None


def initialize_symbols(session, symbols=None):
    """
    Initialize the symbols table with default symbols if empty.

    Args:
        session: SQLAlchemy session
        symbols: List of symbols to initialize with
    """
    default_symbols = symbols or [
        # Major ETFs
        {
            "symbol": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "asset_type": "etf",
            "sector": "broad_market",
        },
        {
            "symbol": "QQQ",
            "name": "Invesco QQQ Trust",
            "asset_type": "etf",
            "sector": "technology",
        },
        {
            "symbol": "IWM",
            "name": "iShares Russell 2000 ETF",
            "asset_type": "etf",
            "sector": "small_cap",
        },
        {
            "symbol": "VTI",
            "name": "Vanguard Total Stock Market ETF",
            "asset_type": "etf",
            "sector": "broad_market",
        },
        {
            "symbol": "XLF",
            "name": "Financial Select Sector SPDR Fund",
            "asset_type": "etf",
            "sector": "financial",
        },
        {
            "symbol": "XLK",
            "name": "Technology Select Sector SPDR Fund",
            "asset_type": "etf",
            "sector": "technology",
        },
        {
            "symbol": "XLV",
            "name": "Health Care Select Sector SPDR Fund",
            "asset_type": "etf",
            "sector": "healthcare",
        },
        # Major stocks
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "asset_type": "stock",
            "sector": "technology",
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "asset_type": "stock",
            "sector": "technology",
        },
        {
            "symbol": "AMZN",
            "name": "Amazon.com Inc.",
            "asset_type": "stock",
            "sector": "consumer_cyclical",
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "asset_type": "stock",
            "sector": "communication_services",
        },
        {
            "symbol": "META",
            "name": "Meta Platforms Inc.",
            "asset_type": "stock",
            "sector": "communication_services",
        },
        {
            "symbol": "TSLA",
            "name": "Tesla Inc.",
            "asset_type": "stock",
            "sector": "consumer_cyclical",
        },
        {
            "symbol": "BRK-B",
            "name": "Berkshire Hathaway Inc.",
            "asset_type": "stock",
            "sector": "financial",
        },
        {
            "symbol": "JPM",
            "name": "JPMorgan Chase & Co.",
            "asset_type": "stock",
            "sector": "financial",
        },
        {
            "symbol": "JNJ",
            "name": "Johnson & Johnson",
            "asset_type": "stock",
            "sector": "healthcare",
        },
        {
            "symbol": "PG",
            "name": "Procter & Gamble Co.",
            "asset_type": "stock",
            "sector": "consumer_defensive",
        },
    ]

    # Get existing symbols to avoid duplicates
    existing = get_active_symbols(session)
    existing_symbols = [s.symbol for s in existing]

    # Add any missing symbols
    symbols_added = 0
    for symbol_data in default_symbols:
        if symbol_data["symbol"] not in existing_symbols:
            save_symbol(
                session,
                symbol_data["symbol"],
                symbol_data["name"],
                symbol_data["asset_type"],
                symbol_data["sector"],
            )
            symbols_added += 1

    logger.info(f"Added {symbols_added} symbols to database")
    return symbols_added


def collect_market_data(session, symbol, period="5y", force_update=False):
    """
    Collect market data for a symbol using Alpaca with incremental loading.

    Args:
        session: SQLAlchemy session
        symbol: Symbol to collect data for
        period: Period to collect (e.g., '5y' for 5 years)
        force_update: Whether to force update existing data

    Returns:
        tuple: (new_records, updated_records, error_occurred)
    """
    # Get Alpaca API credentials
    import os

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        logger.error(
            "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        )
        return 0, 0, True

    # Get Alpaca collector
    collector = get_collector(
        "alpaca", api_key=api_key, secret_key=secret_key, paper=True
    )

    try:
        # Use incremental loading instead of full data fetch
        logger.info(f"Starting incremental data collection for {symbol}")
        
        # Use the incremental fetch method
        data = collector.incremental_fetch_daily_data(
            session=session,
            symbol=symbol,
            period=period,
            force_update=force_update
        )

        if data is None or data.empty:
            logger.warning(f"No data collected for {symbol}")
            return 0, 0, False

        # Count new and updated records by checking what was actually saved
        from data.storage import get_symbol_data_range
        data_range = get_symbol_data_range(session, symbol)
        
        if data_range["has_data"]:
            logger.info(
                f"Incremental collection for {symbol}: {data_range['total_records']} total records "
                f"(from {data_range['earliest_date']} to {data_range['latest_date']})"
            )
            # For incremental loading, we can't easily track new vs updated
            # So we'll estimate based on the data range
            return data_range["total_records"], 0, False
        else:
            logger.warning(f"No data available for {symbol} after collection")
            return 0, 0, False

    except Exception as e:
        logger.error(f"Error collecting data for {symbol}: {str(e)}")
        return 0, 0, True


def run_data_collection(engine=None, symbols=None, period="5y", force_update=False):
    """
    Run the data collection process for all symbols with enhanced error handling.

    Args:
        engine: SQLAlchemy engine
        symbols: List of symbols to collect data for (if None, use active symbols)
        period: Period to collect (e.g., '5y' for 5 years)
        force_update: Whether to force update existing data

    Returns:
        bool: Success or failure
    """
    if engine is None:
        engine = initialize_database()
        if engine is None:
            return False

    session = get_session(engine)

    # Initialize default symbols if needed
    initialize_symbols(session)

    # Get symbols to collect
    if symbols is None:
        symbols = [s.symbol for s in get_active_symbols(session)]

    if not symbols:
        logger.warning("No symbols to collect data for")
        return False

    # Log collection start
    log_entry = log_collection_start(session)
    log_id = log_entry.id

    # Collect data for each symbol with progressive delays on errors
    total_new = 0
    total_updated = 0
    errors = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    base_delay = 2

    for i, symbol in enumerate(symbols):
        try:
            # Add progressive delay if we've had consecutive errors
            if consecutive_errors > 0:
                delay = base_delay * (2**consecutive_errors)
                logger.info(
                    f"Adding {delay}s delay due to {consecutive_errors} consecutive errors"
                )
                time.sleep(delay)

            new_records, updated_records, error_occurred = collect_market_data(
                session, symbol, period, force_update
            )

            if error_occurred:
                errors += 1
                consecutive_errors += 1

                # If we've had too many consecutive errors, pause longer
                if consecutive_errors >= max_consecutive_errors:
                    long_delay = 30 + (consecutive_errors * 10)
                    logger.warning(
                        f"Too many consecutive errors ({consecutive_errors}), pausing for {long_delay}s"
                    )
                    time.sleep(long_delay)
            else:
                total_new += new_records
                total_updated += updated_records
                consecutive_errors = 0  # Reset consecutive error count on success

        except Exception as e:
            logger.error(f"Unexpected error processing {symbol}: {str(e)}")
            errors += 1
            consecutive_errors += 1

        # Log progress every 5 symbols
        if (i + 1) % 5 == 0:
            logger.info(
                f"Progress: {i + 1}/{len(symbols)} symbols processed, {errors} errors so far"
            )

    # Log collection end
    log_collection_end(
        session,
        log_id,
        len(symbols),
        total_new,
        total_updated,
        errors=errors,
        error_message=None if errors == 0 else f"{errors} symbols failed",
    )

    logger.info(
        f"Data collection complete: {len(symbols)} symbols, "
        f"{total_new} new records, {total_updated} updated records, "
        f"{errors} errors"
    )

    # Consider it successful if we got some data and errors were less than 50%
    success_rate = (len(symbols) - errors) / len(symbols) if len(symbols) > 0 else 0
    return success_rate >= 0.5


def run_strategy_backtest(
    engine=None, strategy_name="golden_cross", symbols=None, period_years=3
):
    """
    Run backtesting for a specific strategy.

    Args:
        engine: SQLAlchemy engine
        strategy_name: Name of strategy to test ('golden_cross')
        symbols: List of symbols to test on (defaults to strategy defaults)
        period_years: Number of years of historical data to test

    Returns:
        BacktestResult object or None if failed
    """
    if engine is None:
        engine = initialize_database()
        if engine is None:
            return None

    session = get_session(engine)

    try:
        logger.info(f"Starting backtest for {strategy_name} strategy")

        # Initialize strategy
        if strategy_name == "golden_cross":
            strategy = GoldenCrossStrategy(symbols=symbols)
        else:
            logger.error(f"Unknown strategy: {strategy_name}")
            return None

        # Use strategy symbols if none provided
        test_symbols = symbols if symbols else strategy.symbols

        # Load historical data
        market_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_years * 365)

        for symbol in test_symbols:
            query = (
                session.query(MarketData)
                .filter(
                    MarketData.symbol == symbol,
                    MarketData.date >= start_date,
                    MarketData.date <= end_date,
                )
                .order_by(MarketData.date)
            )

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
                logger.warning(f"No historical data found for {symbol}")

        if not market_data:
            logger.error("No market data available for backtesting")
            return None

        # Run backtest
        backtest_engine = BacktestingEngine(
            initial_capital=10000,
            commission_per_trade=0.0,
            slippage_pct=0.001,
        )

        result = backtest_engine.run_backtest(
            strategy=strategy,
            market_data=market_data,
            start_date=start_date,
            end_date=end_date,
        )

        # Print summary
        logger.info(
            f"Backtest completed: {result.total_return_pct:.2f}% return over {period_years} years"
        )
        logger.info(f"Total trades: {len(result.trades)}")

        if result.performance_metrics.get("sharpe_ratio"):
            logger.info(
                f"Sharpe ratio: {result.performance_metrics['sharpe_ratio']:.2f}"
            )

        return result

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return None
    finally:
        session.close()


def run_strategy_signals(engine=None, strategy_name="golden_cross", symbols=None):
    """
    Generate current trading signals for a strategy.

    Args:
        engine: SQLAlchemy engine
        strategy_name: Name of strategy ('golden_cross')
        symbols: List of symbols to analyze

    Returns:
        List of StrategySignal objects
    """
    if engine is None:
        engine = initialize_database()
        if engine is None:
            return []

    session = get_session(engine)

    try:
        # Initialize strategy
        if strategy_name == "golden_cross":
            strategy = GoldenCrossStrategy(symbols=symbols)
        else:
            logger.error(f"Unknown strategy: {strategy_name}")
            return []

        # Use strategy symbols if none provided
        test_symbols = symbols if symbols else strategy.symbols

        # Load recent market data (need enough for 200-day MA)
        market_data = {}
        days_needed = 250  # Buffer for 200-day MA calculation

        for symbol in test_symbols:
            query = (
                session.query(MarketData)
                .filter(MarketData.symbol == symbol)
                .order_by(MarketData.date.desc())
                .limit(days_needed)
            )

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
                df.sort_index(inplace=True)  # Ensure chronological order
                market_data[symbol] = df
                logger.info(f"Loaded {len(df)} recent records for {symbol}")

        if not market_data:
            logger.warning("No market data available for signal generation")
            return []

        # Generate signals
        signals = strategy.generate_signals(market_data)

        # Log signals
        if signals:
            logger.info(f"Generated {len(signals)} signals:")
            for signal in signals:
                logger.info(
                    f"  {signal.symbol} {signal.signal_type.value} "
                    f"(confidence: {signal.confidence:.2f})"
                )
        else:
            logger.info("No trading signals generated")

        return signals

    except Exception as e:
        logger.error(f"Signal generation failed: {str(e)}")
        return []
    finally:
        session.close()


def main():
    """Main entry point for the pipeline with input validation."""
    # Load environment configuration first
    load_environment()

    # Validate environment setup
    env_validation = validate_required_env_vars()
    if not env_validation["valid"]:
        logger.error(
            f"Environment validation failed. Missing required variables: {env_validation['missing_required']}"
        )
        logger.error("Please check your .env file or system environment variables")
        return 1

    parser = argparse.ArgumentParser(description="Algorithmic Trading Pipeline")
    parser.add_argument(
        "--task",
        choices=["collect", "analyze", "backtest", "trade", "signals"],
        default="collect",
        help="Task to perform",
    )
    parser.add_argument(
        "--strategy",
        choices=["golden_cross"],
        default="golden_cross",
        help="Strategy to use for backtesting/signals",
    )
    parser.add_argument(
        "--years", type=int, default=3, help="Years of historical data for backtesting"
    )
    parser.add_argument(
        "--symbols", nargs="+", help="Symbols to process (e.g. AAPL MSFT)"
    )
    parser.add_argument(
        "--period", default="5y", help="Data period to collect (e.g. 5y, 1y, 6mo)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force update existing data"
    )

    try:
        args = parser.parse_args()

        # Validate inputs
        validated_symbols = None
        if args.symbols:
            try:
                validated_symbols = validate_symbols(args.symbols)
                logger.info(f"Validated symbols: {validated_symbols}")
            except ValueError as e:
                logger.error(f"Invalid symbols provided: {e}")
                return 1

        try:
            validated_period = validate_period(args.period)
            logger.info(f"Using period: {validated_period}")
        except ValueError as e:
            logger.error(f"Invalid period provided: {e}")
            return 1

        # Execute the requested task
        if args.task == "collect":
            success = run_data_collection(
                symbols=validated_symbols,
                period=validated_period,
                force_update=args.force,
            )
            return 0 if success else 1
        elif args.task == "backtest":
            result = run_strategy_backtest(
                strategy_name=args.strategy,
                symbols=validated_symbols,
                period_years=args.years,
            )
            return 0 if result else 1
        elif args.task == "signals":
            signals = run_strategy_signals(
                strategy_name=args.strategy, symbols=validated_symbols
            )
            return 0
        else:
            logger.error(f"Task '{args.task}' not implemented yet")
            return 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in main pipeline: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
