"""
Unit tests for storage layer classes.
Tests database operations, data persistence, and query functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.storage import (
    MarketData,
    Symbol,
    DataCollectionLog,
    get_engine,
    init_db,
    get_session,
)
from utils.exceptions import DatabaseException


class TestStorageModels:
    """Test storage model classes."""

    def test_market_data_model_creation(self):
        """Test MarketData model creation."""
        market_data = MarketData(
            symbol="AAPL",
            date=datetime.now().date(),
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=1000000,
            adj_close=101.0,
        )

        assert market_data.symbol == "AAPL"
        assert market_data.open_price == 100.0
        assert market_data.high_price == 102.0
        assert market_data.low_price == 99.0
        assert market_data.close_price == 101.0
        assert market_data.volume == 1000000
        assert market_data.adj_close == 101.0

    def test_symbol_model_creation(self):
        """Test Symbol model creation."""
        symbol = Symbol(
            symbol="AAPL", name="Apple Inc.", asset_type="stock", is_active=True
        )

        assert symbol.symbol == "AAPL"
        assert symbol.name == "Apple Inc."
        assert symbol.asset_type == "stock"
        assert symbol.is_active is True

    def test_data_collection_log_model_creation(self):
        """Test DataCollectionLog model creation."""
        log = DataCollectionLog(
            symbol="AAPL",
            start_date=datetime.now().date(),
            end_date=datetime.now().date(),
            records_collected=100,
            status="completed",
            error_message=None,
        )

        assert log.symbol == "AAPL"
        assert log.records_collected == 100
        assert log.status == "completed"
        assert log.error_message is None

    def test_market_data_relationships(self):
        """Test MarketData relationships."""
        symbol = Symbol(symbol="AAPL", name="Apple Inc.", asset_type="stock")
        market_data = MarketData(
            symbol="AAPL",
            date=datetime.now().date(),
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=1000000,
            adj_close=101.0,
        )

        # Test that relationships can be established
        assert market_data.symbol == "AAPL"

    def test_model_validation(self):
        """Test model validation."""
        # Test with invalid data types - SQLAlchemy will handle type conversion
        # but we should test that the model can be created with proper types
        try:
            market_data = MarketData(
                symbol="AAPL",
                date=datetime.now().date(),
                open_price=100.0,
                high_price=102.0,
                low_price=99.0,
                close_price=101.0,
                volume=1000000,
                adj_close=101.0,
            )
            # If we get here, the model was created successfully
            assert market_data.symbol == "AAPL"
            assert market_data.open_price == 100.0
        except Exception as e:
            pytest.fail(f"Model creation failed unexpectedly: {e}")


class TestDatabaseOperations:
    """Test database operations."""

    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        # Use SQLite in-memory database for faster tests
        return create_engine("sqlite:///:memory:", echo=False)

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        init_db(test_engine)
        Session = sessionmaker(bind=test_engine)
        return Session()

    def test_get_engine(self):
        """Test get_engine function."""
        # Use SQLite for testing
        engine = get_engine("sqlite:///:memory:")
        assert engine is not None

    def test_init_db_success(self, test_engine):
        """Test successful database initialization."""
        success = init_db(test_engine)
        assert success is True

    def test_init_db_failure(self):
        """Test database initialization failure."""
        # Test with invalid engine
        with patch(
            "sqlalchemy.schema.MetaData.create_all", side_effect=Exception("DB Error")
        ):
            success = init_db(None)
            assert success is False

    def test_get_session(self, test_engine):
        """Test get_session function."""
        init_db(test_engine)
        session = get_session(test_engine)
        assert session is not None

    def test_save_market_data(self, test_session):
        """Test saving market data."""
        from data.storage import save_market_data

        market_data = MarketData(
            symbol="AAPL",
            date=datetime.now().date(),
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=1000000,
            adj_close=101.0,
        )

        success = save_market_data(test_session, market_data)
        assert success is True

    def test_save_market_data_duplicate(self, test_session):
        """Test saving duplicate market data."""
        from data.storage import save_market_data

        date = datetime.now().date()
        market_data1 = MarketData(
            symbol="AAPL",
            date=date,
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=1000000,
            adj_close=101.0,
        )

        market_data2 = MarketData(
            symbol="AAPL",
            date=date,  # Same date
            open_price=101.0,  # Different price
            high_price=103.0,
            low_price=100.0,
            close_price=102.0,
            volume=1100000,
            adj_close=102.0,
        )

        # Save first record
        success1 = save_market_data(test_session, market_data1)
        assert success1 is True

        # Save duplicate record (should update)
        success2 = save_market_data(test_session, market_data2)
        assert success2 is True

    def test_get_market_data(self, test_session):
        """Test retrieving market data."""
        from data.storage import get_market_data, save_market_data

        # Save test data
        market_data = MarketData(
            symbol="AAPL",
            date=datetime.now().date(),
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=1000000,
            adj_close=101.0,
        )
        save_market_data(test_session, market_data)

        # Retrieve data
        retrieved_data = get_market_data(test_session, "AAPL")
        assert retrieved_data is not None
        assert len(retrieved_data) > 0

    def test_save_symbol(self, test_session):
        """Test saving symbol."""
        from data.storage import save_symbol

        symbol = Symbol(
            symbol="AAPL", name="Apple Inc.", asset_type="stock", is_active=True
        )

        result = save_symbol(test_session, symbol)
        assert result is not None
        assert result.symbol == "AAPL"

    def test_get_active_symbols(self, test_session):
        """Test retrieving active symbols."""
        from data.storage import get_active_symbols, save_symbol

        # Save test symbols
        symbol1 = Symbol(
            symbol="AAPL", name="Apple Inc.", asset_type="stock", is_active=True
        )
        symbol2 = Symbol(
            symbol="MSFT", name="Microsoft Corp.", asset_type="stock", is_active=True
        )
        symbol3 = Symbol(
            symbol="INACTIVE",
            name="Inactive Corp.",
            asset_type="stock",
            is_active=False,
        )

        save_symbol(test_session, symbol1)
        save_symbol(test_session, symbol2)
        save_symbol(test_session, symbol3)

        # Get active symbols
        active_symbols = get_active_symbols(test_session)
        assert len(active_symbols) == 2
        assert "AAPL" in [s.symbol for s in active_symbols]
        assert "MSFT" in [s.symbol for s in active_symbols]
        assert "INACTIVE" not in [s.symbol for s in active_symbols]

    def test_log_collection_start(self, test_session):
        """Test logging collection start."""
        from data.storage import log_collection_start

        log_id = log_collection_start(
            test_session, "AAPL", datetime.now().date(), datetime.now().date()
        )
        assert log_id is not None

    def test_log_collection_end(self, test_session):
        """Test logging collection end."""
        from data.storage import log_collection_start, log_collection_end

        # Start collection
        log_id = log_collection_start(
            test_session, "AAPL", datetime.now().date(), datetime.now().date()
        )

        # End collection
        success = log_collection_end(test_session, log_id, 100, "completed")
        assert success is True

    def test_log_collection_error(self, test_session):
        """Test logging collection error."""
        from data.storage import log_collection_start, log_collection_end

        # Start collection
        log_id = log_collection_start(
            test_session, "AAPL", datetime.now().date(), datetime.now().date()
        )

        # End with error
        success = log_collection_end(test_session, log_id, 0, "failed", "API Error")
        assert success is True


class TestDataPersistence:
    """Test data persistence operations."""

    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        # Use SQLite in-memory database for faster tests
        return create_engine("sqlite:///:memory:", echo=False)

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        init_db(test_engine)
        Session = sessionmaker(bind=test_engine)
        return Session()

    @pytest.fixture
    def sample_market_data_df(self):
        """Create sample market data DataFrame."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            "Adj Close": [101.0, 102.0, 103.0, 104.0, 105.0],
        }
        return pd.DataFrame(data, index=dates)

    def test_save_dataframe_to_db(self, test_session, sample_market_data_df):
        """Test saving DataFrame to database."""
        from data.storage import save_market_data

        symbol = "AAPL"
        success_count = sum(
            bool(
                save_market_data(
                    test_session,
                    MarketData(
                        symbol=symbol,
                        date=date.date(),
                        open_price=row["Open"],
                        high_price=row["High"],
                        low_price=row["Low"],
                        close_price=row["Close"],
                        volume=row["Volume"],
                        adj_close=row["Adj Close"],
                    ),
                )
            )
            for date, row in sample_market_data_df.iterrows()
        )
        assert success_count == len(sample_market_data_df)

    def test_bulk_data_operations(self, test_session):
        """Test bulk data operations."""
        from data.storage import save_market_data

        # Create multiple records
        records = [
            MarketData(
                symbol=f"SYMBOL_{i}",
                date=datetime.now().date(),
                open_price=100.0 + i,
                high_price=102.0 + i,
                low_price=99.0 + i,
                close_price=101.0 + i,
                volume=1000000 + i * 1000,
                adj_close=101.0 + i,
            )
            for i in range(100)
        ]

        # Save all records
        success_count = sum(
            bool(save_market_data(test_session, record)) for record in records
        )

        assert success_count == len(records)

    def test_data_retrieval_performance(self, test_session):
        """Test data retrieval performance."""
        from data.storage import get_market_data, save_market_data
        import time

        # Create test data
        symbol = "PERF_TEST"
        for i in range(1000):
            market_data = MarketData(
                symbol=symbol,
                date=datetime.now().date() - timedelta(days=i),
                open_price=100.0,
                high_price=102.0,
                low_price=99.0,
                close_price=101.0,
                volume=1000000,
                adj_close=101.0,
            )
            save_market_data(test_session, market_data)

        # Test retrieval performance
        start_time = time.time()
        retrieved_data = get_market_data(test_session, symbol)
        end_time = time.time()

        assert len(retrieved_data) == 1000
        assert end_time - start_time < 1.0  # Should complete within 1 second


class TestErrorHandling:
    """Test error handling in storage operations."""

    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        # Use SQLite in-memory database for faster tests
        return create_engine("sqlite:///:memory:", echo=False)

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        init_db(test_engine)
        Session = sessionmaker(bind=test_engine)
        return Session()

    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        with patch(
            "data.storage.create_engine", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(Exception):
                get_engine("sqlite:///:memory:")

    def test_session_creation_error(self):
        """Test handling of session creation errors."""
        with patch(
            "data.storage.sessionmaker",
            side_effect=Exception("Session creation failed"),
        ):
            with pytest.raises(DatabaseException):
                get_session(None)

    def test_save_operation_error(self, test_session):
        """Test handling of save operation errors."""
        from data.storage import save_market_data

        # Test with invalid data
        with patch.object(test_session, "add", side_effect=Exception("Save failed")):
            market_data = MarketData(
                symbol="AAPL",
                date=datetime.now().date(),
                open_price=100.0,
                high_price=102.0,
                low_price=99.0,
                close_price=101.0,
                volume=1000000,
                adj_close=101.0,
            )

            with pytest.raises(DatabaseException):
                save_market_data(test_session, market_data)

    def test_query_operation_error(self, test_session):
        """Test handling of query operation errors."""
        from data.storage import get_market_data

        with patch.object(test_session, "query", side_effect=Exception("Query failed")):
            with pytest.raises(Exception):
                get_market_data(test_session, "AAPL")
