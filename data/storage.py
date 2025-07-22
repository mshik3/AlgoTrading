"""
Database storage module for the algorithmic trading system.
Handles all database operations using SQLAlchemy ORM.
"""

import os
import time
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    String,
    Numeric,
    Date,
    DateTime,
    Boolean,
    Text,
    Index,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import QueuePool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom exceptions
try:
    from utils.exceptions import (
        DatabaseException,
        log_exception,
        safe_database_operation,
    )
except ImportError:
    # Fallback if utils module is not available
    logger.warning("Could not import custom exceptions, using basic error handling")
    DatabaseException = SQLAlchemyError

    def log_exception(logger, exception, context=None):
        logger.error(f"{context}: {exception}" if context else str(exception))

    def safe_database_operation(logger, operation_name):
        def decorator(func):
            return func

        return decorator


# Create declarative base
Base = declarative_base()


# Define models
class MarketData(Base):
    """Market data table for storing price history."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Numeric(10, 2), nullable=False)
    high_price = Column(Numeric(10, 2), nullable=False)
    low_price = Column(Numeric(10, 2), nullable=False)
    close_price = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)
    adj_close = Column(Numeric(10, 2), nullable=False)

    # Define unique constraint and additional indexes for performance
    __table_args__ = (
        Index("idx_market_data_symbol_date", "symbol", "date", unique=True),
        Index("idx_market_data_symbol", "symbol"),
        Index("idx_market_data_date", "date"),
        # Additional indexes for incremental loading performance
        Index(
            "idx_market_data_symbol_date_range", "symbol", "date"
        ),  # For date range queries
        Index(
            "idx_market_data_latest_date", "symbol", "date", unique=False
        ),  # For latest date queries
    )

    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', date='{self.date}', close={self.close_price})>"


class Symbol(Base):
    """Symbols table for tracking assets in the system."""

    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True)
    name = Column(String(100))
    asset_type = Column(String(20), nullable=False)  # stock, etf, etc.
    sector = Column(String(50))
    is_active = Column(Boolean, default=True)
    added_date = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Symbol(symbol='{self.symbol}', name='{self.name}', type='{self.asset_type}')>"


class DataCollectionLog(Base):
    """Log table for tracking data collection jobs."""

    __tablename__ = "data_collection_log"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    records_collected = Column(Integer)
    status = Column(String(20), default="running")
    error_message = Column(Text)

    def __repr__(self):
        return f"<DataCollectionLog(symbol='{self.symbol}', status='{self.status}', records_collected={self.records_collected})>"


# Database connection functions
def get_engine(db_uri=None, max_retries=3, retry_delay=5):
    """
    Create and return a database engine with connection pooling and retry logic.

    Args:
        db_uri: Database connection URI (if None, loaded from config)
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retry attempts in seconds

    Returns:
        SQLAlchemy engine

    Raises:
        DatabaseException: If connection fails after all retries
    """
    if db_uri is None:
        # Use the secure environment configuration system
        try:
            from utils.config import get_database_url

            db_uri = get_database_url()
        except ImportError:
            logger.error("Failed to import configuration module")
            raise
        except ValueError as e:
            logger.error(f"Database configuration error: {e}")
            raise

    # Attempt to create engine with retries
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            # Create engine with connection pooling
            engine = create_engine(
                db_uri,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                # Add connection test on connect
                pool_pre_ping=True,
            )

            # Test the connection
            from sqlalchemy import text

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            if attempt > 0:
                logger.info(f"Database connection successful on attempt {attempt + 1}")
            return engine

        except (SQLAlchemyError, OperationalError) as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {str(e)}"
                )
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Database connection failed after {max_retries + 1} attempts"
                )

    # If we get here, all attempts failed
    error_message = f"Failed to connect to database after {max_retries + 1} attempts. Last error: {str(last_exception)}"
    logger.error(error_message)
    logger.error("Please check:")
    logger.error("1. Database server is running")
    logger.error("2. Connection credentials are correct")
    logger.error("3. Network connectivity to database host")
    logger.error("4. Database exists and user has access")

    raise DatabaseException(
        error_message,
        operation="database_connection",
        details={"attempts": max_retries + 1, "last_error": str(last_exception)},
    )


def init_db(engine=None):
    """Initialize the database, creating tables if they don't exist."""
    if engine is None:
        try:
            engine = get_engine()
        except Exception as e:
            logger.error(f"Failed to get engine: {str(e)}")
            return False

    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")

        # Check if volume column migration is needed
        migration_result = migrate_volume_column_to_bigint(engine, check_only=True)
        if migration_result["success"] and migration_result["migration_needed"]:
            logger.info("Applying volume column migration for crypto compatibility...")
            migration_result = migrate_volume_column_to_bigint(engine, check_only=False)

            if migration_result["success"]:
                logger.info(migration_result["message"])
            else:
                logger.warning(
                    f"Volume column migration failed: {migration_result['error']}"
                )
                logger.warning("Crypto assets with large volumes may fail to save")

        logger.info("Database initialized successfully")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False


def get_session(engine=None, max_retries=3):
    """
    Create and return a new database session with proper error handling and retry logic.

    Args:
        engine: SQLAlchemy engine (if None, creates new engine)
        max_retries: Maximum number of session creation attempts

    Returns:
        SQLAlchemy session

    Raises:
        DatabaseException: If session creation fails after all retries
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if engine is None:
                engine = get_engine()

            Session = sessionmaker(bind=engine)
            session = Session()

            # Test the session with a simple query
            from sqlalchemy import text

            session.execute(text("SELECT 1"))

            if attempt > 0:
                logger.info(
                    f"Database session created successfully on attempt {attempt + 1}"
                )
            return session

        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Session creation attempt {attempt + 1} failed: {str(e)}"
                )
                logger.info("Retrying session creation...")
                # Reset engine to None to force reconnection
                engine = None
            else:
                logger.error(
                    f"Session creation failed after {max_retries + 1} attempts"
                )

    # If we get here, all attempts failed
    log_exception(
        logger, last_exception, "Failed to create database session after retries"
    )
    raise DatabaseException(
        f"Could not establish database session after {max_retries + 1} attempts",
        operation="create_session",
        details={
            "attempts": max_retries + 1,
            "last_error": str(last_exception),
            "engine": str(engine) if engine else None,
        },
    )


# --- PATCH: Add safe_date helper ---
def _safe_date(dt):
    from datetime import datetime, date

    if isinstance(dt, datetime):
        return dt.date()
    elif isinstance(dt, date):
        return dt
    else:
        raise TypeError(f"Expected datetime or date, got {type(dt)}")


def migrate_volume_column_to_bigint(engine=None, check_only=False):
    """
    Migrate volume column from Integer to BigInteger for crypto compatibility.

    This function handles upgrading existing databases to support large crypto volumes
    that exceed the Integer type limit (2.1 billion).

    Args:
        engine: SQLAlchemy engine (if None, creates new engine)
        check_only: If True, only check if migration is needed without applying it

    Returns:
        dict: Migration status with details
    """
    if engine is None:
        try:
            engine = get_engine()
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get database engine: {str(e)}",
                "migration_needed": None,
            }

    try:
        from sqlalchemy import inspect, text

        inspector = inspect(engine)

        # Check if market_data table exists
        if "market_data" not in inspector.get_table_names():
            return {
                "success": True,
                "message": "market_data table does not exist yet - no migration needed",
                "migration_needed": False,
            }

        # Check current column type
        columns = inspector.get_columns("market_data")
        volume_column = None

        for col in columns:
            if col["name"] == "volume":
                volume_column = col
                break

        if volume_column is None:
            return {
                "success": False,
                "error": "volume column not found in market_data table",
                "migration_needed": None,
            }

        # Check if volume column is already BigInteger
        col_type_str = str(volume_column["type"]).lower()
        is_bigint = "bigint" in col_type_str or "big_integer" in col_type_str

        if is_bigint:
            return {
                "success": True,
                "message": "volume column is already BigInteger - no migration needed",
                "migration_needed": False,
                "current_type": str(volume_column["type"]),
            }

        # Migration is needed
        if check_only:
            return {
                "success": True,
                "message": "Migration needed: volume column needs to be upgraded to BigInteger",
                "migration_needed": True,
                "current_type": str(volume_column["type"]),
            }

        # Apply the migration
        logger.info(
            "Starting migration: upgrading volume column from Integer to BigInteger"
        )

        with engine.begin() as conn:
            # PostgreSQL syntax
            try:
                conn.execute(
                    text("ALTER TABLE market_data ALTER COLUMN volume TYPE BIGINT")
                )
                logger.info(
                    "✅ Successfully migrated volume column to BigInteger (PostgreSQL)"
                )

            except Exception as pg_error:
                # Try SQLite syntax
                try:
                    logger.info(
                        "PostgreSQL migration failed, trying SQLite approach..."
                    )

                    # SQLite doesn't support ALTER COLUMN TYPE directly
                    # We need to recreate the table for SQLite
                    conn.execute(
                        text(
                            """
                        CREATE TABLE market_data_new AS 
                        SELECT id, symbol, date, open_price, high_price, low_price, 
                               close_price, CAST(volume AS INTEGER) as volume, adj_close 
                        FROM market_data
                    """
                        )
                    )

                    conn.execute(text("DROP TABLE market_data"))
                    conn.execute(
                        text("ALTER TABLE market_data_new RENAME TO market_data")
                    )

                    # Recreate indexes
                    conn.execute(
                        text(
                            """
                        CREATE UNIQUE INDEX idx_market_data_symbol_date 
                        ON market_data (symbol, date)
                    """
                        )
                    )
                    conn.execute(
                        text(
                            "CREATE INDEX idx_market_data_symbol ON market_data (symbol)"
                        )
                    )
                    conn.execute(
                        text("CREATE INDEX idx_market_data_date ON market_data (date)")
                    )

                    logger.info(
                        "✅ Successfully migrated volume column to BigInteger (SQLite)"
                    )

                except Exception as sqlite_error:
                    # Try generic SQL approach
                    try:
                        conn.execute(
                            text("ALTER TABLE market_data MODIFY COLUMN volume BIGINT")
                        )
                        logger.info(
                            "✅ Successfully migrated volume column to BigInteger (MySQL/Generic)"
                        )
                    except Exception as generic_error:
                        raise Exception(
                            f"Migration failed on all database types. "
                            f"PostgreSQL: {str(pg_error)}, SQLite: {str(sqlite_error)}, "
                            f"Generic: {str(generic_error)}"
                        )

        return {
            "success": True,
            "message": "✅ Successfully migrated volume column from Integer to BigInteger",
            "migration_needed": False,
            "previous_type": str(volume_column["type"]),
            "new_type": "BigInteger",
        }

    except Exception as e:
        logger.error(f"Volume column migration failed: {str(e)}")
        return {
            "success": False,
            "error": f"Migration failed: {str(e)}",
            "migration_needed": None,
        }


# Basic CRUD operations for market data
@safe_database_operation(logger, "save_market_data")
def save_market_data(session, data_list):
    """
    Save market data records to the database with improved error handling.
    Uses the 'upsert' pattern to insert or update records.
    Handles both single MarketData objects and lists of MarketData objects.

    Args:
        session: SQLAlchemy session
        data_list: MarketData object or list of MarketData objects

    Returns:
        tuple: (new_records, updated_records) or bool for single objects

    Raises:
        DatabaseException: If database operation fails
        ValueError: If data validation fails
    """
    if session is None:
        raise ValueError("Database session cannot be None")

    # Handle single object vs list
    if isinstance(data_list, MarketData):
        data_list = [data_list]
        return_single = True
    else:
        return_single = False

    if not data_list:
        logger.warning("Empty data_list provided to save_market_data")
        return (0, 0) if not return_single else False

    new_records = 0
    updated_records = 0
    processed_records = 0

    try:
        logger.debug(f"Starting to save {len(data_list)} market data records")

        # Optimize for bulk operations
        if len(data_list) > 1000:
            # Use bulk operations for large datasets
            logger.info(f"Using bulk operations for {len(data_list)} records")

            # Group by symbol for better performance
            from collections import defaultdict

            symbol_groups = defaultdict(list)
            for data in data_list:
                symbol_groups[data.symbol].append(data)

            for symbol, group_data in symbol_groups.items():
                # Get existing records for this symbol
                existing_dates = set()
                existing_records = (
                    session.query(MarketData.date)
                    .filter(MarketData.symbol == symbol)
                    .filter(
                        MarketData.date.in_([_safe_date(d.date) for d in group_data])
                    )
                    .all()
                )
                existing_dates = {record.date for record in existing_records}

                # Separate new and existing records
                new_records_group = []
                update_records_group = []

                for data in group_data:
                    if _safe_date(data.date) in existing_dates:
                        update_records_group.append(data)
                    else:
                        new_records_group.append(data)

                # Bulk insert new records
                if new_records_group:
                    session.bulk_save_objects(new_records_group)
                    new_records += len(new_records_group)

                # Update existing records
                for data in update_records_group:
                    existing = (
                        session.query(MarketData)
                        .filter(
                            MarketData.symbol == data.symbol,
                            MarketData.date == _safe_date(data.date),
                        )
                        .first()
                    )
                    if existing:
                        existing.open_price = data.open_price
                        existing.high_price = data.high_price
                        existing.low_price = data.low_price
                        existing.close_price = data.close_price
                        existing.volume = data.volume
                        existing.adj_close = data.adj_close
                        updated_records += 1

                # Commit each symbol group
                session.commit()
                logger.debug(
                    f"Processed {symbol}: {len(new_records_group)} new, {len(update_records_group)} updated"
                )

            logger.info(
                f"Bulk operation complete: {new_records} new, {updated_records} updated"
            )
            return new_records, updated_records

        # Standard processing for smaller datasets
        for data in data_list:
            # Validate data object
            if not isinstance(data, MarketData):
                raise ValueError(f"Expected MarketData object, got {type(data)}")

            if not data.symbol or not data.date:
                raise ValueError("MarketData must have symbol and date")

            # Check if record exists
            existing = (
                session.query(MarketData)
                .filter(
                    MarketData.symbol == data.symbol,
                    MarketData.date == _safe_date(data.date),
                )
                .first()
            )

            if existing:
                # Update existing record
                existing.open_price = data.open_price
                existing.high_price = data.high_price
                existing.low_price = data.low_price
                existing.close_price = data.close_price
                existing.volume = data.volume
                existing.adj_close = data.adj_close
                updated_records += 1
            else:
                # Add new record
                session.add(data)
                new_records += 1

            processed_records += 1

            # Periodic commit for large datasets
            if processed_records % 100 == 0:
                session.commit()
                logger.debug(
                    f"Committed batch: {processed_records}/{len(data_list)} records processed"
                )

        # Final commit
        session.commit()
        logger.info(
            f"Successfully saved market data: {new_records} new, {updated_records} updated"
        )

        if return_single:
            return new_records > 0 or updated_records > 0
        return new_records, updated_records

    except ValueError as e:
        # Data validation errors - don't rollback, just re-raise
        logger.error(f"Data validation error: {e}")
        raise

    except Exception as e:
        # All other exceptions are handled by the decorator
        raise


def get_market_data(session, symbol, start_date=None, end_date=None):
    """
    Retrieve market data for a specific symbol and date range.

    Args:
        session: SQLAlchemy session
        symbol: Stock symbol
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        list: List of MarketData objects
    """
    query = session.query(MarketData).filter(MarketData.symbol == symbol)

    if start_date:
        query = query.filter(MarketData.date >= start_date)

    if end_date:
        query = query.filter(MarketData.date <= end_date)

    # Order by date for time series consistency
    query = query.order_by(MarketData.date)

    return query.all()


def save_symbol(
    session, symbol, name=None, asset_type="stock", sector=None, active=True
):
    """
    Save a symbol to the database.

    Args:
        session: SQLAlchemy session
        symbol: Stock symbol (string) or Symbol object
        name: Company/asset name
        asset_type: Type of asset (stock, etf, etc.)
        sector: Market sector
        active: Whether to actively collect data for this symbol

    Returns:
        Symbol: The created or updated Symbol object
    """
    try:
        # Handle both string and Symbol object inputs
        if isinstance(symbol, Symbol):
            symbol_str = symbol.symbol
            name = name or symbol.name
            asset_type = asset_type or symbol.asset_type
            sector = sector or symbol.sector
            active = active if active is not True else symbol.is_active
        else:
            symbol_str = symbol

        # Check if symbol exists
        existing = session.query(Symbol).filter(Symbol.symbol == symbol_str).first()

        if existing:
            # Update existing symbol
            existing.name = name if name else existing.name
            existing.asset_type = asset_type
            existing.sector = sector if sector else existing.sector
            existing.is_active = active
            session.commit()
            return existing
        else:
            # Create new symbol
            new_symbol = Symbol(
                symbol=symbol_str,
                name=name,
                asset_type=asset_type,
                sector=sector,
                is_active=active,
            )
            session.add(new_symbol)
            session.commit()
            return new_symbol

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error saving symbol {symbol}: {str(e)}")
        raise


def get_active_symbols(session):
    """
    Get all active symbols from the database.

    Args:
        session: SQLAlchemy session

    Returns:
        list: List of Symbol objects with is_active=True
    """
    return session.query(Symbol).filter(Symbol.is_active == True).all()


def log_collection_start(session, symbol, start_date, end_date=None):
    """
    Log the start of a data collection job.

    Args:
        session: SQLAlchemy session
        symbol: Symbol being collected
        start_date: Start date for collection
        end_date: End date for collection (optional)

    Returns:
        int: The ID of the created log entry
    """
    log_entry = DataCollectionLog(
        symbol=symbol, start_date=start_date, end_date=end_date, status="running"
    )
    session.add(log_entry)
    session.commit()
    return log_entry.id


def log_collection_end(
    session,
    log_id,
    records_collected,
    status="completed",
    error_message=None,
):
    """
    Update a data collection log entry with completion information.

    Args:
        session: SQLAlchemy session
        log_id: ID of the log entry to update
        records_collected: Number of records collected
        status: Status of the collection (completed, failed, etc.)
        error_message: Error message if applicable

    Returns:
        bool: True if update was successful, False otherwise
    """
    log_entry = session.get(DataCollectionLog, log_id)
    if log_entry:
        from datetime import datetime

        log_entry.end_date = _safe_date(datetime.now())
        log_entry.records_collected = records_collected
        log_entry.status = status
        log_entry.error_message = error_message
        session.commit()
        return True
    return False


# Incremental data loading functions
def get_symbol_data_range(session, symbol):
    """
    Get the date range of available data for a symbol.

    Args:
        session: SQLAlchemy session
        symbol: Stock symbol

    Returns:
        dict: Dictionary with 'earliest_date', 'latest_date', 'total_records', and 'has_data'
    """
    from sqlalchemy import func

    try:
        # Get earliest and latest dates
        result = (
            session.query(
                func.min(MarketData.date).label("earliest_date"),
                func.max(MarketData.date).label("latest_date"),
                func.count(MarketData.id).label("total_records"),
            )
            .filter(MarketData.symbol == symbol)
            .first()
        )

        if result and result.earliest_date and result.latest_date:
            return {
                "earliest_date": result.earliest_date,
                "latest_date": result.latest_date,
                "total_records": result.total_records,
                "has_data": True,
            }
        else:
            return {
                "earliest_date": None,
                "latest_date": None,
                "total_records": 0,
                "has_data": False,
            }

    except SQLAlchemyError as e:
        logger.error(f"Error getting data range for {symbol}: {str(e)}")
        return {
            "earliest_date": None,
            "latest_date": None,
            "total_records": 0,
            "has_data": False,
        }


def get_symbols_data_summary(session, symbols):
    """
    Get data summary for multiple symbols at once.

    Args:
        session: SQLAlchemy session
        symbols: List of symbols

    Returns:
        dict: Dictionary mapping symbol -> data range info
    """
    from sqlalchemy import func

    try:
        # Get summary for all symbols in one query
        results = (
            session.query(
                MarketData.symbol,
                func.min(MarketData.date).label("earliest_date"),
                func.max(MarketData.date).label("latest_date"),
                func.count(MarketData.id).label("total_records"),
            )
            .filter(MarketData.symbol.in_(symbols))
            .group_by(MarketData.symbol)
            .all()
        )

        summary = {}
        for result in results:
            summary[result.symbol] = {
                "earliest_date": result.earliest_date,
                "latest_date": result.latest_date,
                "total_records": result.total_records,
                "has_data": True,
            }

        # Add symbols with no data
        for symbol in symbols:
            if symbol not in summary:
                summary[symbol] = {
                    "earliest_date": None,
                    "latest_date": None,
                    "total_records": 0,
                    "has_data": False,
                }

        return summary

    except SQLAlchemyError as e:
        logger.error(f"Error getting data summary for symbols: {str(e)}")
        # Fallback to individual queries
        summary = {}
        for symbol in symbols:
            summary[symbol] = get_symbol_data_range(session, symbol)
        return summary


def get_missing_date_ranges(session, symbol, target_start_date, target_end_date):
    """
    Calculate missing date ranges for incremental data loading.

    Args:
        session: SQLAlchemy session
        symbol: Stock symbol
        target_start_date: Target start date for data
        target_end_date: Target end date for data

    Returns:
        list: List of (start_date, end_date) tuples for missing ranges
    """
    from datetime import date, timedelta

    try:
        # Get existing data range
        data_range = get_symbol_data_range(session, symbol)

        if not data_range["has_data"]:
            # No existing data, need to fetch entire range
            return [(target_start_date, target_end_date)]

        existing_start = data_range["earliest_date"]
        existing_end = data_range["latest_date"]

        missing_ranges = []

        # Check if we need data before existing range
        if target_start_date < existing_start:
            missing_ranges.append(
                (target_start_date, existing_start - timedelta(days=1))
            )

        # Check if we need data after existing range
        if target_end_date > existing_end:
            missing_ranges.append((existing_end + timedelta(days=1), target_end_date))

        # Check for gaps in existing data
        if data_range["has_data"]:
            # Get all dates in existing range
            existing_dates = set()
            query = (
                session.query(MarketData.date)
                .filter(
                    MarketData.symbol == symbol,
                    MarketData.date >= existing_start,
                    MarketData.date <= existing_end,
                )
                .order_by(MarketData.date)
            )

            for record in query.all():
                existing_dates.add(record.date)

            # Find gaps in the data
            current_date = existing_start
            gap_start = None

            while current_date <= existing_end:
                if current_date not in existing_dates:
                    if gap_start is None:
                        gap_start = current_date
                else:
                    if gap_start is not None:
                        missing_ranges.append(
                            (gap_start, current_date - timedelta(days=1))
                        )
                        gap_start = None
                current_date += timedelta(days=1)

            # Handle gap at the end
            if gap_start is not None:
                missing_ranges.append((gap_start, existing_end))

        return missing_ranges

    except SQLAlchemyError as e:
        logger.error(f"Error calculating missing date ranges for {symbol}: {str(e)}")
        # Fallback: return entire range
        return [(target_start_date, target_end_date)]


def detect_data_gaps(session, symbol, start_date, end_date, max_gap_days=5):
    """
    Detect gaps in historical data that exceed a threshold.

    Args:
        session: SQLAlchemy session
        symbol: Stock symbol
        start_date: Start date to check
        end_date: End date to check
        max_gap_days: Maximum allowed gap in days

    Returns:
        list: List of gap ranges (start_date, end_date) that exceed threshold
    """
    from datetime import date, timedelta
    from sqlalchemy import func

    try:
        # Get all dates in range
        query = (
            session.query(MarketData.date)
            .filter(
                MarketData.symbol == symbol,
                MarketData.date >= start_date,
                MarketData.date <= end_date,
            )
            .order_by(MarketData.date)
        )

        existing_dates = [record.date for record in query.all()]

        if not existing_dates:
            return [(start_date, end_date)]  # Complete gap

        gaps = []
        current_date = start_date

        for existing_date in existing_dates:
            # Check if there's a gap before this date
            if current_date < existing_date:
                gap_days = (existing_date - current_date).days
                if gap_days > max_gap_days:
                    gaps.append((current_date, existing_date - timedelta(days=1)))
            current_date = existing_date + timedelta(days=1)

        # Check for gap at the end
        if current_date <= end_date:
            gap_days = (end_date - current_date).days
            if gap_days > max_gap_days:
                gaps.append((current_date, end_date))

        return gaps

    except SQLAlchemyError as e:
        logger.error(f"Error detecting data gaps for {symbol}: {str(e)}")
        return []


# Simple in-memory cache for frequently accessed data
_data_cache = {}
_cache_timestamps = {}


def get_cached_market_data(symbol, start_date, end_date, cache_timeout=300):
    """
    Get market data from cache if available and not expired.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        cache_timeout: Cache timeout in seconds

    Returns:
        list: Cached MarketData objects or None if not cached/expired
    """
    import time

    cache_key = f"{symbol}_{start_date}_{end_date}"
    current_time = time.time()

    if cache_key in _data_cache:
        if current_time - _cache_timestamps.get(cache_key, 0) < cache_timeout:
            logger.debug(f"Cache hit for {symbol} data")
            return _data_cache[cache_key]
        else:
            # Cache expired, remove it
            del _data_cache[cache_key]
            if cache_key in _cache_timestamps:
                del _cache_timestamps[cache_key]

    return None


def set_cached_market_data(symbol, start_date, end_date, data):
    """
    Cache market data for future use.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        data: MarketData objects to cache
    """
    import time

    cache_key = f"{symbol}_{start_date}_{end_date}"
    _data_cache[cache_key] = data
    _cache_timestamps[cache_key] = time.time()
    logger.debug(f"Cached data for {symbol}")


def invalidate_symbol_cache(symbol):
    """
    Invalidate all cached data for a symbol.

    Args:
        symbol: Stock symbol to invalidate cache for
    """
    keys_to_remove = [key for key in _data_cache.keys() if key.startswith(f"{symbol}_")]
    for key in keys_to_remove:
        del _data_cache[key]
        if key in _cache_timestamps:
            del _cache_timestamps[key]
    logger.debug(f"Invalidated cache for {symbol}")


def clear_data_cache():
    """Clear all cached data."""
    global _data_cache, _cache_timestamps
    _data_cache.clear()
    _cache_timestamps.clear()
    logger.debug("Cleared all data cache")
