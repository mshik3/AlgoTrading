"""
Data package for the algorithmic trading system.
Handles data collection, processing, and storage.
"""

from .collectors import get_collector, YahooFinanceCollector, AlphaVantageCollector
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
)

__all__ = [
    "get_collector",
    "YahooFinanceCollector",
    "AlphaVantageCollector",
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
]
