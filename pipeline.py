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
)

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
    Collect market data for a symbol with enhanced error handling.

    Args:
        session: SQLAlchemy session
        symbol: Symbol to collect data for
        period: Period to collect (e.g., '5y' for 5 years)
        force_update: Whether to force update existing data

    Returns:
        tuple: (new_records, updated_records, error_occurred)
    """
    # Get collector and processor with rate limiting protection
    collector = get_collector("yahoo")
    processor = DataProcessor()

    try:
        # Fetch data with built-in rate limiting protection
        market_data = collector.collect_and_transform(symbol, period=period)

        if not market_data:
            logger.warning(f"No data collected for {symbol}")
            return 0, 0, False

        # Save data
        new_records, updated_records = save_market_data(session, market_data)
        logger.info(
            f"Collected data for {symbol}: {new_records} new, {updated_records} updated"
        )

        return new_records, updated_records, False

    except Exception as e:
        error_msg = str(e)
        if (
            "rate limit" in error_msg.lower()
            or "too many requests" in error_msg.lower()
        ):
            logger.error(f"Rate limiting encountered for {symbol}: {error_msg}")
        else:
            logger.error(f"Unexpected error collecting data for {symbol}: {error_msg}")
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
        choices=["collect", "analyze", "backtest", "trade"],
        default="collect",
        help="Task to perform",
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
