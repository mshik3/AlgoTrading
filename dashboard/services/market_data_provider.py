"""
Database Market Data Provider for Strategy Execution

This service provides efficient database-based market data access for strategies,
replacing the need for real-time API calls during strategy execution.

Key Features:
- Efficient bulk data retrieval from database
- Smart caching for frequently accessed data
- Optimized queries for multiple symbols
- Data validation and completeness checking
- Strategy-specific data requirements handling
"""

import sys
import os
import logging
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import project modules
from data.storage import (
    get_session,
    get_engine,
    get_market_data,
    get_symbols_data_summary,
    get_cached_market_data,
    set_cached_market_data,
)
from utils.config import load_environment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataCache:
    """Smart caching system for frequently accessed market data."""

    def __init__(self, max_cache_size: int = 1000, cache_ttl: int = 300):
        """
        Initialize the cache system.

        Args:
            max_cache_size: Maximum number of cached entries
            cache_ttl: Cache time-to-live in seconds (5 minutes default)
        """
        self.cache = {}
        self.cache_timestamps = {}
        self.access_counts = {}
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl

    def _generate_cache_key(
        self, symbols: Union[str, List[str]], start_date: date, end_date: date
    ) -> str:
        """Generate a unique cache key for the data request."""
        if isinstance(symbols, str):
            symbols_str = symbols
        else:
            symbols_str = "_".join(sorted(symbols))
        return f"{symbols_str}_{start_date}_{end_date}"

    def get(
        self, symbols: Union[str, List[str]], start_date: date, end_date: date
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Retrieve data from cache if available and not expired."""
        cache_key = self._generate_cache_key(symbols, start_date, end_date)
        current_time = time.time()

        if cache_key in self.cache:
            # Check if cache entry is still valid
            if current_time - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl:
                # Update access count for LRU eviction
                self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
                logger.debug(f"Cache hit for key: {cache_key[:50]}...")
                return self.cache[cache_key]
            else:
                # Cache expired, remove it
                self._remove_cache_entry(cache_key)

        return None

    def set(
        self,
        symbols: Union[str, List[str]],
        start_date: date,
        end_date: date,
        data: Dict[str, pd.DataFrame],
    ):
        """Store data in cache with automatic size management."""
        cache_key = self._generate_cache_key(symbols, start_date, end_date)
        current_time = time.time()

        # Check if we need to evict old entries
        if len(self.cache) >= self.max_cache_size:
            self._evict_least_used()

        # Store the data
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = current_time
        self.access_counts[cache_key] = 1

        logger.debug(
            f"Cached data for key: {cache_key[:50]}... (cache size: {len(self.cache)})"
        )

    def _remove_cache_entry(self, cache_key: str):
        """Remove a specific cache entry."""
        self.cache.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)
        self.access_counts.pop(cache_key, None)

    def _evict_least_used(self):
        """Evict the least recently used cache entry."""
        if not self.cache:
            return

        # Find the least accessed key
        least_used_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
        self._remove_cache_entry(least_used_key)
        logger.debug(f"Evicted least used cache entry: {least_used_key[:50]}...")

    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.cache_timestamps.clear()
        self.access_counts.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = sum(
            1
            for timestamp in self.cache_timestamps.values()
            if current_time - timestamp < self.cache_ttl
        )

        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "cache_hit_potential": sum(self.access_counts.values()),
            "max_size": self.max_cache_size,
            "ttl_seconds": self.cache_ttl,
        }


class DatabaseMarketDataProvider:
    """
    Efficient database-based market data provider for strategy execution.

    This service replaces real-time API calls with fast database queries,
    dramatically improving strategy execution performance.
    """

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 300):
        """
        Initialize the database market data provider.

        Args:
            cache_size: Maximum number of cached data sets
            cache_ttl: Cache time-to-live in seconds
        """
        self.setup_database()
        self.cache = MarketDataCache(cache_size, cache_ttl)
        self.query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "database_queries": 0,
            "symbols_requested": 0,
            "records_retrieved": 0,
        }

    def setup_database(self):
        """Initialize database connection."""
        try:
            self.engine = get_engine()
            logger.info("✅ Database market data provider initialized")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

    def get_market_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[Union[datetime, str, date]] = None,
        end_date: Optional[Union[datetime, str, date]] = None,
        period: Optional[str] = None,
        validate_completeness: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get market data for symbols from database with caching.

        Args:
            symbols: Single symbol string or list of symbols
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            period: Period string (e.g., "1y", "2y") - alternative to dates
            validate_completeness: Whether to validate data completeness

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        # Track query statistics
        self.query_stats["total_queries"] += 1

        # Normalize inputs
        if isinstance(symbols, str):
            symbols = [symbols]

        self.query_stats["symbols_requested"] += len(symbols)

        # Handle period conversion
        if period:
            end_date = datetime.now().date()
            if period == "2y":
                start_date = end_date - timedelta(days=2 * 365)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year

        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().date()
        if not start_date:
            start_date = end_date - timedelta(days=365)

        # Convert dates to date objects
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)

        # Check cache first
        cached_data = self.cache.get(symbols, start_date, end_date)
        if cached_data:
            self.query_stats["cache_hits"] += 1
            logger.debug(f"Cache hit for {len(symbols)} symbols")
            return cached_data

        # Query database
        logger.debug(
            f"Querying database for {len(symbols)} symbols from {start_date} to {end_date}"
        )
        self.query_stats["database_queries"] += 1

        try:
            session = get_session(self.engine)

            market_data = {}
            total_records = 0

            # Batch query for better performance
            if len(symbols) > 10:
                market_data = self._bulk_query_symbols(
                    session, symbols, start_date, end_date
                )
            else:
                # Individual queries for smaller requests
                for symbol in symbols:
                    try:
                        data_records = get_market_data(
                            session, symbol, start_date, end_date
                        )

                        if data_records:
                            df = self._convert_to_dataframe(data_records)
                            if not df.empty:
                                market_data[symbol] = df
                                total_records += len(df)
                        else:
                            logger.warning(f"No data found for symbol: {symbol}")

                    except Exception as e:
                        logger.error(f"Error querying {symbol}: {str(e)}")
                        continue

            session.close()

            # Validate data completeness if requested
            if validate_completeness:
                market_data = self._validate_data_completeness(
                    market_data, symbols, start_date, end_date
                )

            # Cache the results
            if market_data:
                self.cache.set(symbols, start_date, end_date, market_data)
                self.query_stats["records_retrieved"] += total_records

            logger.info(
                f"✅ Retrieved data for {len(market_data)}/{len(symbols)} symbols "
                f"({total_records:,} total records)"
            )

            return market_data

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            return {}

    def _bulk_query_symbols(
        self, session, symbols: List[str], start_date: date, end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """
        Efficiently query multiple symbols using bulk database operations.

        Args:
            session: Database session
            symbols: List of symbols to query
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        from data.storage import MarketData
        from sqlalchemy import and_

        try:
            # Single query to get all data for all symbols
            query = (
                session.query(MarketData)
                .filter(
                    and_(
                        MarketData.symbol.in_(symbols),
                        MarketData.date >= start_date,
                        MarketData.date <= end_date,
                    )
                )
                .order_by(MarketData.symbol, MarketData.date)
            )

            # Execute query and group by symbol
            all_records = query.all()

            # Group records by symbol
            symbol_records = defaultdict(list)
            for record in all_records:
                symbol_records[record.symbol].append(record)

            # Convert to DataFrames
            market_data = {}
            for symbol, records in symbol_records.items():
                if records:
                    df = self._convert_to_dataframe(records)
                    if not df.empty:
                        market_data[symbol] = df

            logger.debug(
                f"Bulk query retrieved {len(all_records)} total records for {len(market_data)} symbols"
            )
            return market_data

        except Exception as e:
            logger.error(f"Bulk query failed: {str(e)}")
            # Fallback to individual queries
            return {}

    def _convert_to_dataframe(self, records) -> pd.DataFrame:
        """Convert database records to pandas DataFrame."""
        if not records:
            return pd.DataFrame()

        data = []
        for record in records:
            data.append(
                {
                    "Date": record.date,
                    "Open": float(record.open_price),
                    "High": float(record.high_price),
                    "Low": float(record.low_price),
                    "Close": float(record.close_price),
                    "Volume": int(record.volume),
                    "Adj Close": float(record.adj_close),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        return df

    def _normalize_date(self, date_input: Union[datetime, str, date]) -> date:
        """Normalize various date inputs to date object."""
        if isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, str):
            return datetime.fromisoformat(date_input.replace("Z", "+00:00")).date()
        elif isinstance(date_input, date):
            return date_input
        else:
            raise ValueError(f"Unsupported date type: {type(date_input)}")

    def _validate_data_completeness(
        self,
        market_data: Dict[str, pd.DataFrame],
        requested_symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate data completeness and log any issues.

        Args:
            market_data: Retrieved market data
            requested_symbols: Originally requested symbols
            start_date: Start date for validation
            end_date: End date for validation

        Returns:
            Validated market data (potentially filtered)
        """
        validated_data = {}
        issues = []

        expected_trading_days = self._calculate_expected_trading_days(
            start_date, end_date
        )
        min_required_days = int(expected_trading_days * 0.8)  # Allow 20% tolerance

        for symbol in requested_symbols:
            if symbol in market_data:
                df = market_data[symbol]

                if len(df) >= min_required_days:
                    validated_data[symbol] = df
                else:
                    issues.append(
                        f"{symbol}: Only {len(df)} days (expected ~{expected_trading_days})"
                    )
            else:
                issues.append(f"{symbol}: No data available")

        if issues:
            logger.warning(f"Data completeness issues for {len(issues)} symbols:")
            for issue in issues[:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
            if len(issues) > 5:
                logger.warning(f"  ... and {len(issues) - 5} more issues")

        return validated_data

    def _calculate_expected_trading_days(self, start_date: date, end_date: date) -> int:
        """Calculate expected number of trading days (approximate)."""
        total_days = (end_date - start_date).days
        # Rough approximation: ~252 trading days per year
        return int(total_days * (252 / 365))

    def get_data_summary(self, symbols: List[str]) -> Dict[str, Dict[str, any]]:
        """
        Get data availability summary for symbols.

        Args:
            symbols: List of symbols to check

        Returns:
            Dictionary mapping symbol -> data summary
        """
        try:
            session = get_session(self.engine)
            data_summary = get_symbols_data_summary(session, symbols)
            session.close()
            return data_summary

        except Exception as e:
            logger.error(f"Failed to get data summary: {str(e)}")
            return {}

    def get_available_symbols(self) -> List[str]:
        """Get list of all symbols with available data in the database."""
        try:
            session = get_session(self.engine)

            from data.storage import MarketData
            from sqlalchemy import distinct

            query = session.query(distinct(MarketData.symbol)).order_by(
                MarketData.symbol
            )
            symbols = [row[0] for row in query.all()]

            session.close()
            logger.info(f"Found {len(symbols)} symbols with available data")
            return symbols

        except Exception as e:
            logger.error(f"Failed to get available symbols: {str(e)}")
            return []

    def get_date_range_for_symbol(self, symbol: str) -> Optional[Tuple[date, date]]:
        """
        Get the available date range for a specific symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (earliest_date, latest_date) or None if no data
        """
        try:
            session = get_session(self.engine)

            from data.storage import get_symbol_data_range

            data_range = get_symbol_data_range(session, symbol)

            session.close()

            if data_range["has_data"]:
                return (data_range["earliest_date"], data_range["latest_date"])
            return None

        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {str(e)}")
            return None

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Market data cache cleared")

    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics for the data provider."""
        cache_stats = self.cache.get_stats()

        total_queries = self.query_stats["total_queries"]
        cache_hit_rate = (
            (self.query_stats["cache_hits"] / total_queries) * 100
            if total_queries > 0
            else 0
        )

        return {
            "query_stats": self.query_stats.copy(),
            "cache_hit_rate": cache_hit_rate,
            "cache_stats": cache_stats,
            "avg_records_per_query": (
                self.query_stats["records_retrieved"]
                / self.query_stats["database_queries"]
                if self.query_stats["database_queries"] > 0
                else 0
            ),
        }


def create_market_data_provider(
    cache_size: int = 1000, cache_ttl: int = 300
) -> DatabaseMarketDataProvider:
    """
    Factory function to create a configured DatabaseMarketDataProvider.

    Args:
        cache_size: Maximum number of cached data sets
        cache_ttl: Cache time-to-live in seconds

    Returns:
        Configured DatabaseMarketDataProvider instance
    """
    return DatabaseMarketDataProvider(cache_size=cache_size, cache_ttl=cache_ttl)


# Singleton instance for application-wide use
_provider_instance: Optional[DatabaseMarketDataProvider] = None


def get_market_data_provider() -> DatabaseMarketDataProvider:
    """
    Get or create the singleton market data provider instance.

    Returns:
        DatabaseMarketDataProvider instance
    """
    global _provider_instance

    if _provider_instance is None:
        _provider_instance = create_market_data_provider()
        logger.info("Created singleton DatabaseMarketDataProvider instance")

    return _provider_instance


# Example usage for testing
if __name__ == "__main__":
    try:
        # Test the market data provider
        provider = create_market_data_provider()

        # Test symbols
        test_symbols = ["SPY", "QQQ", "AAPL"]

        print("Testing DatabaseMarketDataProvider...")

        # Get data for test symbols
        market_data = provider.get_market_data(symbols=test_symbols, period="6mo")

        print(f"Retrieved data for {len(market_data)} symbols:")
        for symbol, df in market_data.items():
            print(f"  {symbol}: {len(df)} records, {df.index[0]} to {df.index[-1]}")

        # Test cache performance
        print("\nTesting cache performance...")
        start_time = time.time()

        # Second query should hit cache
        cached_data = provider.get_market_data(symbols=test_symbols, period="6mo")

        cache_time = time.time() - start_time
        print(f"Cached query completed in {cache_time:.3f} seconds")

        # Get performance stats
        stats = provider.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        print(f"  Total queries: {stats['query_stats']['total_queries']}")
        print(f"  Database queries: {stats['query_stats']['database_queries']}")
        print(f"  Records retrieved: {stats['query_stats']['records_retrieved']:,}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
