"""
Main orchestration module for the algorithmic trading system.
Coordinates the data collection, strategy execution, and trade generation.
"""

import os
import logging
import argparse
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
    Collect market data for a symbol.

    Args:
        session: SQLAlchemy session
        symbol: Symbol to collect data for
        period: Period to collect (e.g., '5y' for 5 years)
        force_update: Whether to force update existing data

    Returns:
        tuple: (new_records, updated_records)
    """
    # Get collector and processor
    collector = get_collector("yahoo")
    processor = DataProcessor()

    # Fetch data
    market_data = collector.collect_and_transform(symbol, period=period)

    if not market_data:
        logger.warning(f"No data collected for {symbol}")
        return 0, 0

    # Save data
    new_records, updated_records = save_market_data(session, market_data)
    logger.info(
        f"Collected data for {symbol}: {new_records} new, {updated_records} updated"
    )

    return new_records, updated_records


def run_data_collection(engine=None, symbols=None, period="5y", force_update=False):
    """
    Run the data collection process for all symbols.

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

    # Collect data for each symbol
    total_new = 0
    total_updated = 0
    errors = 0

    for symbol in symbols:
        try:
            new_records, updated_records = collect_market_data(
                session, symbol, period, force_update
            )
            total_new += new_records
            total_updated += updated_records
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            errors += 1

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

    return errors == 0


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Algorithmic Trading Pipeline")
    parser.add_argument(
        "--task",
        choices=["collect", "analyze", "backtest", "trade"],
        default="collect",
        help="Task to perform",
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to process")
    parser.add_argument("--period", default="5y", help="Data period to collect")
    parser.add_argument(
        "--force", action="store_true", help="Force update existing data"
    )

    args = parser.parse_args()

    if args.task == "collect":
        success = run_data_collection(
            symbols=args.symbols, period=args.period, force_update=args.force
        )
        return 0 if success else 1
    else:
        logger.error(f"Task '{args.task}' not implemented yet")
        return 1


if __name__ == "__main__":
    exit(main())
