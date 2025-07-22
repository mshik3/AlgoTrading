"""
Data package for the algorithmic trading system.
Handles market data collection from Alpaca, processing, and storage.
"""

from .collectors import get_collector, AlpacaDataCollector
from .processors import DataProcessor, DataValidator, DataCleaner
from .storage import (
    MarketData,
    Symbol,
    DataCollectionLog,
    get_engine,
    init_db,
    get_session,
    save_market_data,
    get_market_data,
    save_symbol,
    get_active_symbols,
    log_collection_start,
    log_collection_end,
    # Database migration functions
    migrate_volume_column_to_bigint,
    # Incremental loading functions
    get_symbol_data_range,
    get_symbols_data_summary,
    get_missing_date_ranges,
    detect_data_gaps,
    # Caching functions
    get_cached_market_data,
    set_cached_market_data,
    invalidate_symbol_cache,
    clear_data_cache,
)

__all__ = [
    "get_collector",
    "AlpacaDataCollector",
    "DataProcessor",
    "DataValidator",
    "DataCleaner",
    "MarketData",
    "Symbol",
    "DataCollectionLog",
    "get_engine",
    "init_db",
    "get_session",
    "save_market_data",
    "get_market_data",
    "save_symbol",
    "get_active_symbols",
    "log_collection_start",
    "log_collection_end",
    # Database migration functions
    "migrate_volume_column_to_bigint",
    # Incremental loading functions
    "get_symbol_data_range",
    "get_symbols_data_summary",
    "get_missing_date_ranges",
    "detect_data_gaps",
    # Caching functions
    "get_cached_market_data",
    "set_cached_market_data",
    "invalidate_symbol_cache",
    "clear_data_cache",
]
