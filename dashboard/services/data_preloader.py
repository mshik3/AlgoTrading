"""
Market Data Pre-Loading Service for Dashboard Startup

This service handles bulk data collection during dashboard initialization,
ensuring that all strategies have access to pre-loaded database data
for instant execution without API delays.

Key Features:
- Bulk data collection for 920+ asset universe
- Progress tracking and status reporting
- Incremental loading with gap detection
- Database optimization and validation
- Error handling and retry logic
"""

import sys
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import project modules
from data.alpaca_collector import AlpacaDataCollector, get_alpaca_collector
from data.storage import get_session, get_engine, init_db, get_symbols_data_summary
from utils.asset_universe_config import get_920_asset_universe, get_universe_summary
from utils.config import load_environment, get_env_var

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreloadProgress:
    """Progress tracking for data pre-loading operations."""

    def __init__(self, total_symbols: int):
        self.total_symbols = total_symbols
        self.completed_symbols = 0
        self.failed_symbols = 0
        self.start_time = datetime.now()
        self.status_callback: Optional[Callable] = None

    def update(self, completed: int = 1, failed: int = 0):
        """Update progress counters."""
        self.completed_symbols += completed
        self.failed_symbols += failed

        if self.status_callback:
            self.status_callback(self.get_status())

    def get_status(self) -> Dict[str, any]:
        """Get current progress status."""
        elapsed_time = datetime.now() - self.start_time
        processed = self.completed_symbols + self.failed_symbols

        if processed > 0:
            estimated_total_time = elapsed_time * (self.total_symbols / processed)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = timedelta(0)

        return {
            "total_symbols": self.total_symbols,
            "completed": self.completed_symbols,
            "failed": self.failed_symbols,
            "remaining": self.total_symbols - processed,
            "progress_pct": (
                (processed / self.total_symbols) * 100 if self.total_symbols > 0 else 0
            ),
            "success_rate": (
                (self.completed_symbols / processed) * 100 if processed > 0 else 0
            ),
            "elapsed_time": elapsed_time,
            "remaining_time": remaining_time,
        }

    def set_status_callback(self, callback: Callable):
        """Set callback for status updates."""
        self.status_callback = callback


class MarketDataPreloader:
    """
    Service for pre-loading market data during dashboard startup.

    This service ensures that all strategies have access to fresh,
    comprehensive market data without needing to fetch during execution.
    """

    def __init__(self):
        """Initialize the market data pre-loader."""
        self.setup_environment()
        self.setup_data_collector()
        self.setup_database()
        self.symbols = get_920_asset_universe()
        self.progress = None

    def setup_environment(self):
        """Load and validate environment configuration."""
        try:
            load_environment()

            # Validate Alpaca credentials
            alpaca_key = get_env_var("ALPACA_API_KEY", default=None)
            alpaca_secret = get_env_var("ALPACA_SECRET_KEY", default=None)

            if not alpaca_key or not alpaca_secret:
                raise ValueError("Alpaca API credentials required for data pre-loading")

            self.use_mock_data = False
            logger.info("✅ Environment configuration loaded")

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            raise

    def setup_data_collector(self):
        """Initialize the Alpaca data collector."""
        try:
            self.data_collector = get_alpaca_collector()

            # Test connection
            if not self.data_collector.test_connection():
                raise ValueError("Alpaca API connection test failed")

            logger.info("✅ Alpaca data collector initialized and tested")

        except Exception as e:
            logger.error(f"Data collector setup failed: {e}")
            raise

    def setup_database(self):
        """Initialize database connection and ensure tables exist."""
        try:
            self.engine = get_engine()

            # Initialize database tables
            if not init_db(self.engine):
                raise ValueError("Database initialization failed")

            logger.info("✅ Database connection established and initialized")

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

    def preload_all_data(
        self,
        period: str = "2y",
        status_callback: Optional[Callable] = None,
        max_workers: int = 5,
    ) -> Dict[str, any]:
        """
        Pre-load market data for all symbols in the universe.

        Args:
            period: Time period for data collection (e.g., "2y", "1y", "6mo")
            status_callback: Optional callback for progress updates
            max_workers: Maximum number of concurrent workers

        Returns:
            Dictionary with pre-loading results
        """
        logger.info(
            f"🚀 Starting market data pre-loading for {len(self.symbols)} symbols"
        )
        logger.info(f"📊 Period: {period}, Max Workers: {max_workers}")

        # Initialize progress tracking
        self.progress = DataPreloadProgress(len(self.symbols))
        if status_callback:
            self.progress.set_status_callback(status_callback)

        # Get universe summary
        universe_summary = get_universe_summary()
        logger.info(f"Asset Universe Breakdown:")
        logger.info(f"  - Fortune 500: {universe_summary['fortune500_count']}")
        logger.info(f"  - ETFs: {universe_summary['etf_count']}")
        logger.info(f"  - Crypto: {universe_summary['crypto_count']}")

        start_time = datetime.now()
        results = {
            "success": True,
            "total_symbols": len(self.symbols),
            "completed_symbols": 0,
            "failed_symbols": 0,
            "total_records": 0,
            "new_records": 0,
            "updated_records": 0,
            "skipped_symbols": 0,
            "errors": [],
            "start_time": start_time,
            "end_time": None,
            "duration": None,
        }

        try:
            # Use session per thread for database operations
            session = get_session(self.engine)

            # Analyze current database state
            logger.info("📊 Analyzing current database state...")
            current_state = self._analyze_database_state(session)
            logger.info(
                f"Database Analysis: {current_state['symbols_with_data']} symbols have data, "
                f"{current_state['symbols_missing_data']} missing data"
            )

            # Prioritize symbols that need data most
            prioritized_symbols = self._prioritize_symbols(session, period)
            logger.info(
                f"🎯 Prioritized {len(prioritized_symbols)} symbols for data collection"
            )

            # Batch process symbols for optimal performance
            if max_workers > 1 and len(prioritized_symbols) > 20:
                results.update(
                    self._preload_parallel(
                        prioritized_symbols, period, max_workers, session
                    )
                )
            else:
                results.update(
                    self._preload_sequential(prioritized_symbols, period, session)
                )

            session.close()

        except Exception as e:
            logger.error(f"Pre-loading failed: {str(e)}")
            results["success"] = False
            results["errors"].append(str(e))

        # Finalize results
        end_time = datetime.now()
        results["end_time"] = end_time
        results["duration"] = end_time - start_time

        # Log summary
        self._log_preload_summary(results)

        return results

    def _analyze_database_state(self, session) -> Dict[str, any]:
        """Analyze current database state to optimize loading strategy."""
        try:
            # Get data summary for all symbols
            data_summary = get_symbols_data_summary(session, self.symbols)

            symbols_with_data = 0
            symbols_missing_data = 0
            total_records = 0

            for symbol, summary in data_summary.items():
                if summary["has_data"]:
                    symbols_with_data += 1
                    total_records += summary["total_records"]
                else:
                    symbols_missing_data += 1

            return {
                "symbols_with_data": symbols_with_data,
                "symbols_missing_data": symbols_missing_data,
                "total_records": total_records,
                "data_summary": data_summary,
            }

        except Exception as e:
            logger.error(f"Database state analysis failed: {e}")
            return {
                "symbols_with_data": 0,
                "symbols_missing_data": len(self.symbols),
                "total_records": 0,
                "data_summary": {},
            }

    def _prioritize_symbols(self, session, period: str) -> List[str]:
        """
        Prioritize symbols for data collection based on current state.

        Returns symbols in order of loading priority:
        1. Symbols with no data (highest priority)
        2. Symbols with outdated data
        3. Symbols with incomplete data ranges
        """
        try:
            from data.storage import get_missing_date_ranges
            from datetime import datetime, timedelta

            # Calculate target date range
            end_date = datetime.now()
            if period == "2y":
                start_date = end_date - timedelta(days=2 * 365)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            else:
                start_date = end_date - timedelta(days=365)

            priority_groups = {
                "no_data": [],
                "outdated_data": [],
                "incomplete_data": [],
                "up_to_date": [],
            }

            data_summary = get_symbols_data_summary(session, self.symbols)

            for symbol, summary in data_summary.items():
                if not summary["has_data"]:
                    priority_groups["no_data"].append(symbol)
                else:
                    # Check if data is recent enough
                    latest_date = summary["latest_date"]
                    days_old = (
                        (end_date.date() - latest_date).days if latest_date else 999
                    )

                    if days_old > 7:  # More than a week old
                        priority_groups["outdated_data"].append(symbol)
                    else:
                        # Check for gaps in data
                        missing_ranges = get_missing_date_ranges(
                            session, symbol, start_date.date(), end_date.date()
                        )

                        if missing_ranges:
                            priority_groups["incomplete_data"].append(symbol)
                        else:
                            priority_groups["up_to_date"].append(symbol)

            # Combine in priority order
            prioritized = (
                priority_groups["no_data"]
                + priority_groups["outdated_data"]
                + priority_groups["incomplete_data"]
                + priority_groups["up_to_date"]
            )

            logger.info(f"Symbol Prioritization:")
            logger.info(f"  - No data: {len(priority_groups['no_data'])}")
            logger.info(f"  - Outdated data: {len(priority_groups['outdated_data'])}")
            logger.info(
                f"  - Incomplete data: {len(priority_groups['incomplete_data'])}"
            )
            logger.info(f"  - Up to date: {len(priority_groups['up_to_date'])}")

            return prioritized

        except Exception as e:
            logger.error(f"Symbol prioritization failed: {e}")
            return self.symbols  # Fallback to original order

    def _preload_sequential(
        self, symbols: List[str], period: str, session
    ) -> Dict[str, any]:
        """Load data sequentially for smaller datasets or single-threaded mode."""
        results = {
            "completed_symbols": 0,
            "failed_symbols": 0,
            "total_records": 0,
            "new_records": 0,
            "updated_records": 0,
            "skipped_symbols": 0,
            "errors": [],
        }

        logger.info(f"📊 Sequential loading for {len(symbols)} symbols")

        for i, symbol in enumerate(symbols):
            try:
                symbol_start = time.time()

                # Use incremental fetch for optimal performance
                data_result = self.data_collector.incremental_fetch_daily_data(
                    session=session, symbol=symbol, period=period, force_update=False
                )

                symbol_time = time.time() - symbol_start

                if data_result is not None and not data_result.empty:
                    results["completed_symbols"] += 1
                    results["total_records"] += len(data_result)

                    logger.info(
                        f"✅ {symbol}: {len(data_result)} records ({symbol_time:.2f}s) "
                        f"[{i+1}/{len(symbols)}]"
                    )
                else:
                    results["skipped_symbols"] += 1
                    logger.warning(
                        f"⚠️ {symbol}: No data available [{i+1}/{len(symbols)}]"
                    )

                # Update progress
                self.progress.update(completed=1)

                # Rate limiting to avoid API overload
                if (i + 1) % 50 == 0:
                    logger.info(
                        f"💤 Rate limiting: processed {i+1} symbols, brief pause..."
                    )
                    time.sleep(1)

            except Exception as e:
                results["failed_symbols"] += 1
                results["errors"].append(f"{symbol}: {str(e)}")
                logger.error(f"❌ {symbol}: {str(e)} [{i+1}/{len(symbols)}]")

                # Update progress
                self.progress.update(failed=1)

        return results

    def _preload_parallel(
        self, symbols: List[str], period: str, max_workers: int, session
    ) -> Dict[str, any]:
        """Load data in parallel for faster processing of large datasets."""
        results = {
            "completed_symbols": 0,
            "failed_symbols": 0,
            "total_records": 0,
            "new_records": 0,
            "updated_records": 0,
            "skipped_symbols": 0,
            "errors": [],
        }

        logger.info(
            f"🚀 Parallel loading for {len(symbols)} symbols with {max_workers} workers"
        )

        # Create symbol batches for parallel processing
        batch_size = max(1, len(symbols) // max_workers)
        symbol_batches = [
            symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
        ]

        def process_batch(batch: List[str]) -> Dict[str, any]:
            """Process a batch of symbols in a single thread."""
            batch_results = {
                "completed": 0,
                "failed": 0,
                "total_records": 0,
                "new_records": 0,
                "updated_records": 0,
                "skipped": 0,
                "errors": [],
            }

            # Create new session for this thread
            thread_session = get_session(self.engine)

            try:
                for symbol in batch:
                    try:
                        data_result = self.data_collector.incremental_fetch_daily_data(
                            session=thread_session,
                            symbol=symbol,
                            period=period,
                            force_update=False,
                        )

                        if data_result is not None and not data_result.empty:
                            batch_results["completed"] += 1
                            batch_results["total_records"] += len(data_result)
                        else:
                            batch_results["skipped"] += 1

                        # Update global progress
                        self.progress.update(completed=1)

                    except Exception as e:
                        batch_results["failed"] += 1
                        batch_results["errors"].append(f"{symbol}: {str(e)}")
                        self.progress.update(failed=1)

            finally:
                thread_session.close()

            return batch_results

        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): i
                for i, batch in enumerate(symbol_batches)
            }

            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_result = future.result()

                    # Aggregate results
                    results["completed_symbols"] += batch_result["completed"]
                    results["failed_symbols"] += batch_result["failed"]
                    results["total_records"] += batch_result["total_records"]
                    results["new_records"] += batch_result["new_records"]
                    results["updated_records"] += batch_result["updated_records"]
                    results["skipped_symbols"] += batch_result["skipped"]
                    results["errors"].extend(batch_result["errors"])

                    logger.info(
                        f"✅ Batch {batch_idx + 1}/{len(symbol_batches)} completed: "
                        f"{batch_result['completed']} success, {batch_result['failed']} failed"
                    )

                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx + 1} failed: {str(e)}")
                    results["errors"].append(f"Batch {batch_idx + 1}: {str(e)}")

        return results

    def _log_preload_summary(self, results: Dict[str, any]):
        """Log comprehensive summary of pre-loading results."""
        logger.info("=" * 80)
        logger.info("📊 MARKET DATA PRE-LOADING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"🎯 Total Symbols: {results['total_symbols']}")
        logger.info(f"✅ Completed: {results['completed_symbols']}")
        logger.info(f"❌ Failed: {results['failed_symbols']}")
        logger.info(f"⏭️ Skipped: {results['skipped_symbols']}")
        logger.info(f"📈 Total Records: {results['total_records']:,}")
        logger.info(f"🆕 New Records: {results['new_records']:,}")
        logger.info(f"🔄 Updated Records: {results['updated_records']:,}")

        if results.get("duration"):
            duration = results["duration"]
            logger.info(f"⏱️ Duration: {duration}")

            if results["completed_symbols"] > 0:
                avg_time = duration.total_seconds() / results["completed_symbols"]
                logger.info(f"⚡ Average per symbol: {avg_time:.2f}s")

        success_rate = (
            (results["completed_symbols"] / results["total_symbols"]) * 100
            if results["total_symbols"] > 0
            else 0
        )
        logger.info(f"📊 Success Rate: {success_rate:.1f}%")

        if results["errors"]:
            logger.warning(f"⚠️ Errors encountered ({len(results['errors'])} total):")
            for error in results["errors"][:10]:  # Show first 10 errors
                logger.warning(f"   - {error}")
            if len(results["errors"]) > 10:
                logger.warning(f"   ... and {len(results['errors']) - 10} more")

        logger.info("=" * 80)

    def get_loading_status(self) -> Dict[str, any]:
        """Get current loading status if operation is in progress."""
        if self.progress:
            return self.progress.get_status()
        return {"status": "not_started"}

    def validate_preloaded_data(self, sample_size: int = 50) -> Dict[str, any]:
        """
        Validate that pre-loaded data is complete and accurate.

        Args:
            sample_size: Number of symbols to validate in detail

        Returns:
            Validation results dictionary
        """
        logger.info(f"🔍 Validating pre-loaded data (sample size: {sample_size})")

        try:
            session = get_session(self.engine)

            # Get data summary for validation sample
            validation_symbols = self.symbols[:sample_size]
            data_summary = get_symbols_data_summary(session, validation_symbols)

            validation_results = {
                "success": True,
                "total_validated": len(validation_symbols),
                "symbols_with_data": 0,
                "symbols_missing_data": 0,
                "avg_records_per_symbol": 0,
                "date_range_coverage": {},
                "issues": [],
            }

            total_records = 0
            symbols_with_data = 0

            for symbol, summary in data_summary.items():
                if summary["has_data"]:
                    symbols_with_data += 1
                    total_records += summary["total_records"]

                    # Validate date range coverage
                    if (
                        summary["total_records"] < 100
                    ):  # Expect at least 100 days for 2y period
                        validation_results["issues"].append(
                            f"{symbol}: Only {summary['total_records']} records (expected ~500)"
                        )
                else:
                    validation_results["issues"].append(f"{symbol}: No data found")

            validation_results["symbols_with_data"] = symbols_with_data
            validation_results["symbols_missing_data"] = (
                len(validation_symbols) - symbols_with_data
            )
            validation_results["avg_records_per_symbol"] = (
                total_records / symbols_with_data if symbols_with_data > 0 else 0
            )

            # Overall validation status
            if (
                len(validation_results["issues"]) > len(validation_symbols) * 0.1
            ):  # More than 10% issues
                validation_results["success"] = False

            logger.info(
                f"✅ Validation complete: {symbols_with_data}/{len(validation_symbols)} symbols valid"
            )
            if validation_results["issues"]:
                logger.warning(
                    f"⚠️ Found {len(validation_results['issues'])} validation issues"
                )

            session.close()
            return validation_results

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_validated": 0,
                "symbols_with_data": 0,
                "symbols_missing_data": sample_size,
            }


def create_preloader() -> MarketDataPreloader:
    """Factory function to create a configured MarketDataPreloader instance."""
    return MarketDataPreloader()


# Example usage for testing
if __name__ == "__main__":

    def progress_callback(status):
        print(
            f"Progress: {status['progress_pct']:.1f}% "
            f"({status['completed']}/{status['total_symbols']}) "
            f"ETA: {status['remaining_time']}"
        )

    try:
        preloader = create_preloader()

        # Test with small sample for development
        preloader.symbols = preloader.symbols[:10]  # Limit to 10 symbols for testing

        results = preloader.preload_all_data(
            period="6mo", status_callback=progress_callback, max_workers=2
        )

        print(f"Pre-loading completed: {results['success']}")
        print(
            f"Completed: {results['completed_symbols']}, Failed: {results['failed_symbols']}"
        )

        # Validate results
        validation = preloader.validate_preloaded_data(sample_size=5)
        print(f"Validation: {validation['success']}")

    except Exception as e:
        print(f"Test failed: {e}")
