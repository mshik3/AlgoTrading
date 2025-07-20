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
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Numeric(10, 2), nullable=False)

    # Define unique constraint
    __table_args__ = (
        Index("idx_market_data_symbol_date", "symbol", "date", unique=True),
    )

    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', date='{self.date}', close={self.close})>"


class Symbol(Base):
    """Symbols table for tracking assets in the system."""

    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True)
    name = Column(String(100))
    asset_type = Column(String(20), nullable=False)  # stock, etf, etc.
    sector = Column(String(50))
    active = Column(Boolean, default=True)
    added_date = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Symbol(symbol='{self.symbol}', name='{self.name}', type='{self.asset_type}')>"


class DataCollectionLog(Base):
    """Log table for tracking data collection jobs."""

    __tablename__ = "data_collection_log"

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime)
    symbols_processed = Column(Integer)
    new_records_added = Column(Integer)
    records_updated = Column(Integer)
    errors = Column(Integer, default=0)
    status = Column(String(20), default="running")
    error_message = Column(Text)

    def __repr__(self):
        return f"<DataCollectionLog(id={self.id}, status='{self.status}', symbols_processed={self.symbols_processed})>"


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
        engine = get_engine()

    try:
        # Create all tables
        Base.metadata.create_all(engine)
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


# Basic CRUD operations for market data
@safe_database_operation(logger, "save_market_data")
def save_market_data(session, data_list):
    """
    Save a list of market data records to the database with improved error handling.
    Uses the 'upsert' pattern to insert or update records.

    Args:
        session: SQLAlchemy session
        data_list: List of MarketData objects

    Returns:
        tuple: (new_records, updated_records)

    Raises:
        DatabaseException: If database operation fails
        ValueError: If data validation fails
    """
    if not data_list:
        logger.warning("Empty data_list provided to save_market_data")
        return 0, 0

    if session is None:
        raise ValueError("Database session cannot be None")

    new_records = 0
    updated_records = 0
    processed_records = 0

    try:
        logger.debug(f"Starting to save {len(data_list)} market data records")

        for data in data_list:
            # Validate data object
            if not isinstance(data, MarketData):
                raise ValueError(f"Expected MarketData object, got {type(data)}")

            if not data.symbol or not data.date:
                raise ValueError("MarketData must have symbol and date")

            # Check if record exists
            existing = (
                session.query(MarketData)
                .filter(MarketData.symbol == data.symbol, MarketData.date == data.date)
                .first()
            )

            if existing:
                # Update existing record
                existing.open = data.open
                existing.high = data.high
                existing.low = data.low
                existing.close = data.close
                existing.volume = data.volume
                existing.adjusted_close = data.adjusted_close
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
        symbol: Stock symbol
        name: Company/asset name
        asset_type: Type of asset (stock, etf, etc.)
        sector: Market sector
        active: Whether to actively collect data for this symbol

    Returns:
        Symbol: The created or updated Symbol object
    """
    try:
        # Check if symbol exists
        existing = session.query(Symbol).filter(Symbol.symbol == symbol).first()

        if existing:
            # Update existing symbol
            existing.name = name if name else existing.name
            existing.asset_type = asset_type
            existing.sector = sector if sector else existing.sector
            existing.active = active
            session.commit()
            return existing
        else:
            # Create new symbol
            new_symbol = Symbol(
                symbol=symbol,
                name=name,
                asset_type=asset_type,
                sector=sector,
                active=active,
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
        list: List of Symbol objects with active=True
    """
    return session.query(Symbol).filter(Symbol.active == True).all()


def log_collection_start(session):
    """
    Log the start of a data collection job.

    Args:
        session: SQLAlchemy session

    Returns:
        DataCollectionLog: The created log entry
    """
    log_entry = DataCollectionLog(start_time=datetime.utcnow(), status="running")
    session.add(log_entry)
    session.commit()
    return log_entry


def log_collection_end(
    session,
    log_id,
    symbols_processed,
    new_records,
    updated_records,
    errors=0,
    error_message=None,
):
    """
    Update a data collection log entry with completion information.

    Args:
        session: SQLAlchemy session
        log_id: ID of the log entry to update
        symbols_processed: Number of symbols processed
        new_records: Number of new records added
        updated_records: Number of records updated
        errors: Number of errors encountered
        error_message: Error message if applicable

    Returns:
        DataCollectionLog: The updated log entry
    """
    log_entry = session.query(DataCollectionLog).get(log_id)
    if log_entry:
        log_entry.end_time = datetime.utcnow()
        log_entry.symbols_processed = symbols_processed
        log_entry.new_records_added = new_records
        log_entry.records_updated = updated_records
        log_entry.errors = errors
        log_entry.status = "error" if errors > 0 else "completed"
        log_entry.error_message = error_message
        session.commit()
        return log_entry
    return None
