"""
Basic performance tests for critical components.
Tests system performance under normal load.
"""

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from data.processors import DataProcessor
from strategies.equity.golden_cross import GoldenCrossStrategy
from dashboard.data.live_data import PaperTradingSimulator


class TestDataProcessingPerformance:
    """Test data processing performance."""

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        data = {
            "Open": np.random.uniform(100, 200, 1000),
            "High": np.random.uniform(200, 300, 1000),
            "Low": np.random.uniform(50, 100, 1000),
            "Close": np.random.uniform(100, 200, 1000),
            "Volume": np.random.randint(1000000, 10000000, 1000),
            "Adj Close": np.random.uniform(100, 200, 1000),
        }
        return pd.DataFrame(data, index=dates)

    def test_data_processing_speed(self, large_dataset):
        """Test data processing speed."""
        processor = DataProcessor()

        start_time = time.time()
        processed_data = processor.process_data(large_dataset, "AAPL")
        processing_time = time.time() - start_time

        # Should process 1000 rows within 2 seconds
        assert processing_time < 2.0
        assert len(processed_data) > 0

    def test_data_processing_memory(self, large_dataset):
        """Test data processing memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        processor = DataProcessor()
        processed_data = processor.process_data(large_dataset, "AAPL")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestStrategyPerformance:
    """Test strategy performance."""

    @pytest.fixture
    def market_data(self):
        """Create market data for strategy testing."""
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        data = {
            "Open": [100.0] * 200,
            "High": [102.0] * 200,
            "Low": [99.0] * 200,
            "Close": [101.0] * 200,
            "Volume": [1000000] * 200,
            "Adj Close": [101.0] * 200,
        }
        return pd.DataFrame(data, index=dates)

    def test_strategy_signal_generation_speed(self, market_data):
        """Test strategy signal generation speed."""
        strategy = GoldenCrossStrategy(symbols=["AAPL"])
        market_data_dict = {"AAPL": market_data}

        start_time = time.time()
        signals = strategy.generate_signals(market_data_dict)
        generation_time = time.time() - start_time

        # Should generate signals within 1 second
        assert generation_time < 1.0
        assert isinstance(signals, list)

    def test_multiple_symbols_performance(self):
        """Test performance with multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
        market_data_dict = {}

        for symbol in symbols:
            dates = pd.date_range("2023-01-01", periods=100, freq="D")
            data = {
                "Open": [100.0] * 100,
                "High": [102.0] * 100,
                "Low": [99.0] * 100,
                "Close": [101.0] * 100,
                "Volume": [1000000] * 100,
                "Adj Close": [101.0] * 100,
            }
            market_data_dict[symbol] = pd.DataFrame(data, index=dates)

        strategy = GoldenCrossStrategy(symbols=symbols)

        start_time = time.time()
        signals = strategy.generate_signals(market_data_dict)
        generation_time = time.time() - start_time

        # Should handle multiple symbols within 2 seconds
        assert generation_time < 2.0
        assert isinstance(signals, list)


class TestExecutionPerformance:
    """Test execution performance."""

    def test_paper_trading_speed(self):
        """Test paper trading execution speed."""
        simulator = PaperTradingSimulator(initial_capital=10000)

        # Create multiple signals
        from strategies.base import StrategySignal, SignalType

        signals = []
        for i in range(100):
            signal = StrategySignal(
                symbol="AAPL",
                signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                confidence=0.8,
                price=150.0,
                quantity=10,
            )
            signals.append(signal)

        start_time = time.time()
        for signal in signals:
            simulator.execute_signal(signal)
        execution_time = time.time() - start_time

        # Should execute 100 signals within 1 second
        assert execution_time < 1.0

    def test_portfolio_calculation_speed(self):
        """Test portfolio calculation speed."""
        simulator = PaperTradingSimulator(initial_capital=10000)

        # Create some positions
        from strategies.base import StrategySignal, SignalType

        for i in range(10):
            signal = StrategySignal(
                symbol=f"SYMBOL_{i}",
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=100.0,
                quantity=10,
            )
            simulator.execute_signal(signal)

        # Test portfolio calculation speed
        start_time = time.time()
        for _ in range(100):
            portfolio_value = simulator.get_portfolio_value()
            portfolio_summary = simulator.get_portfolio_summary()
        calculation_time = time.time() - start_time

        # Should calculate portfolio 100 times within 1 second
        assert calculation_time < 1.0
        assert portfolio_value > 0


class TestSystemLoad:
    """Test system performance under load."""

    def test_concurrent_data_processing(self):
        """Test concurrent data processing."""
        import threading

        def process_data():
            processor = DataProcessor()
            dates = pd.date_range("2023-01-01", periods=100, freq="D")
            data = {
                "Open": [100.0] * 100,
                "High": [102.0] * 100,
                "Low": [99.0] * 100,
                "Close": [101.0] * 100,
                "Volume": [1000000] * 100,
                "Adj Close": [101.0] * 100,
            }
            df = pd.DataFrame(data, index=dates)
            return processor.process_data(df, "AAPL")

        # Start multiple threads
        threads = []
        results = []

        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(process_data()))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert result is not None

    def test_memory_under_load(self):
        """Test memory usage under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run multiple operations
        for _ in range(10):
            # Data processing
            processor = DataProcessor()
            dates = pd.date_range("2023-01-01", periods=50, freq="D")
            data = {
                "Open": [100.0] * 50,
                "High": [102.0] * 50,
                "Low": [99.0] * 50,
                "Close": [101.0] * 50,
                "Volume": [1000000] * 50,
                "Adj Close": [101.0] * 50,
            }
            df = pd.DataFrame(data, index=dates)
            processed_data = processor.process_data(df, "AAPL")

            # Strategy execution
            strategy = GoldenCrossStrategy(symbols=["AAPL"])
            market_data = {"AAPL": processed_data}
            signals = strategy.generate_signals(market_data)

            # Paper trading
            simulator = PaperTradingSimulator(initial_capital=10000)
            for signal in signals[:5]:  # Execute first 5 signals
                simulator.execute_signal(signal)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 200MB)
        assert memory_increase < 200 * 1024 * 1024
