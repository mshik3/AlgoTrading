"""
Pytest fixtures and configuration for algorithmic trading system tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    np.random.seed(42)  # For reproducible tests

    data = {
        "Date": dates,
        "Open": 100 + np.random.randn(len(dates)) * 2,
        "High": 102 + np.random.randn(len(dates)) * 2,
        "Low": 98 + np.random.randn(len(dates)) * 2,
        "Close": 100 + np.random.randn(len(dates)) * 2,
        "Volume": np.random.randint(1000000, 10000000, len(dates)),
        "Adj Close": 100 + np.random.randn(len(dates)) * 2,
    }

    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)

    # Ensure High >= Low and other realistic constraints
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)

    return df


@pytest.fixture
def sample_symbols():
    """Provide sample stock symbols for testing."""
    return ["AAPL", "MSFT", "GOOGL", "TSLA"]


@pytest.fixture
def mock_database_session():
    """Mock database session for testing."""
    session = Mock()
    session.query.return_value = Mock()
    session.commit.return_value = None
    session.rollback.return_value = None
    session.close.return_value = None
    return session


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    env_vars = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_algotrading",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "LOG_LEVEL": "INFO",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def mock_alpaca_data(sample_market_data):
    """Mock Alpaca data download."""
    mock_client = Mock()
    mock_client.get_stock_bars.return_value = sample_market_data

    with patch("alpaca.data.StockHistoricalDataClient", return_value=mock_client):
        yield mock_client


class TestConfig:
    """Test configuration constants."""

    TEST_SYMBOLS = ["AAPL", "MSFT", "TEST"]
    TEST_PERIOD = "1mo"
    TEST_DB_URI = "sqlite:///:memory:"
