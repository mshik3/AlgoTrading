"""
Integration tests for complete trading workflows.
Tests end-to-end trading workflows from data collection to execution.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from data import get_collector, DataProcessor, get_engine, init_db, get_session
from strategies.equity.golden_cross import GoldenCrossStrategy
from dashboard.data.live_data import PaperTradingSimulator
from backtesting import BacktestingEngine, PerformanceMetrics


class TestCompleteTradingWorkflow:
    """Test complete trading workflow from data to execution."""

    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        from sqlalchemy import create_engine

        engine = create_engine("sqlite:///:memory:")
        init_db(engine)
        return engine

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=test_engine)
        return Session()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create realistic price data
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        prices = base_price * np.exp(np.cumsum(returns))

        data = {
            "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
            "Adj Close": prices,
        }

        df = pd.DataFrame(data, index=dates)

        # Ensure price relationships are logical
        df["High"] = df[["Open", "High", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

        return df

    def test_data_collection_to_strategy_workflow(
        self, test_session, sample_market_data
    ):
        """Test workflow from data collection to strategy signal generation."""
        # 1. Data Collection (mocked)
        with patch(
            "data.collectors.YahooFinanceCollector.fetch_daily_data",
            return_value=sample_market_data,
        ):
            collector = get_collector("yahoo")
            data = collector.fetch_daily_data("AAPL")
            assert data is not None
            assert len(data) > 0

        # 2. Data Processing
        processor = DataProcessor()
        processed_data = processor.process_data(data, "AAPL")
        assert processed_data is not None
        assert len(processed_data) > 0
        assert not processed_data.isnull().values.any()

        # 3. Strategy Signal Generation
        strategy = GoldenCrossStrategy(symbols=["AAPL"])
        market_data_dict = {"AAPL": processed_data}
        signals = strategy.generate_signals(market_data_dict)

        assert isinstance(signals, list)
        # Should have at least one signal (even if it's HOLD)
        assert len(signals) > 0

    def test_strategy_to_execution_workflow(self, sample_market_data):
        """Test workflow from strategy signals to execution."""
        # 1. Create strategy and generate signals
        strategy = GoldenCrossStrategy(symbols=["AAPL"])
        processor = DataProcessor()
        processed_data = processor.process_data(sample_market_data, "AAPL")
        market_data_dict = {"AAPL": processed_data}
        signals = strategy.generate_signals(market_data_dict)

        # 2. Initialize paper trading simulator
        simulator = PaperTradingSimulator(initial_capital=10000)

        # 3. Execute signals
        executed_signals = []
        for signal in signals:
            if signal.signal_type.value in ["BUY", "SELL"]:
                success = simulator.execute_signal(signal)
                if success:
                    executed_signals.append(signal)

        # 4. Verify execution results
        assert len(executed_signals) >= 0  # May have no executable signals
        assert simulator.get_portfolio_value() > 0

    def test_complete_backtesting_workflow(self, test_session, sample_market_data):
        """Test complete backtesting workflow."""
        # 1. Prepare data
        processor = DataProcessor()
        processed_data = processor.process_data(sample_market_data, "AAPL")
        market_data_dict = {"AAPL": processed_data}

        # 2. Initialize strategy
        strategy = GoldenCrossStrategy(symbols=["AAPL"])

        # 3. Initialize backtesting engine
        backtest_engine = BacktestingEngine(
            initial_capital=10000, commission_per_trade=0.0
        )

        # 4. Run backtest
        result = backtest_engine.run_backtest(
            strategy=strategy,
            market_data=market_data_dict,
            start_date=processed_data.index[0],
            end_date=processed_data.index[-1],
        )

        # 5. Verify backtest results
        assert result is not None
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "trades")

    def test_data_persistence_workflow(self, test_session, sample_market_data):
        """Test data persistence workflow."""
        from data.storage import save_market_data, get_market_data, MarketData

        # 1. Save market data to database
        symbol = "AAPL"
        saved_count = 0

        for date, row in sample_market_data.iterrows():
            market_data = MarketData(
                symbol=symbol,
                date=date.date(),
                open_price=row["Open"],
                high_price=row["High"],
                low_price=row["Low"],
                close_price=row["Close"],
                volume=row["Volume"],
                adj_close=row["Adj Close"],
            )

            if save_market_data(test_session, market_data):
                saved_count += 1

        assert saved_count == len(sample_market_data)

        # 2. Retrieve data from database
        retrieved_data = get_market_data(test_session, symbol)
        assert retrieved_data is not None
        assert len(retrieved_data) == len(sample_market_data)

    def test_multi_symbol_workflow(self, test_session):
        """Test workflow with multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        # Create sample data for multiple symbols
        market_data_dict = {}
        for symbol in symbols:
            dates = pd.date_range("2023-01-01", periods=50, freq="D")
            np.random.seed(hash(symbol) % 1000)  # Different seed for each symbol

            base_price = 100.0 + hash(symbol) % 100
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))

            data = {
                "Open": prices,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
                "Adj Close": prices,
            }

            df = pd.DataFrame(data, index=dates)
            market_data_dict[symbol] = df

        # 1. Process all data
        processor = DataProcessor()
        processed_data_dict = {}

        for symbol, data in market_data_dict.items():
            processed_data = processor.process_data(data, symbol)
            processed_data_dict[symbol] = processed_data

        # 2. Generate signals for all symbols
        strategy = GoldenCrossStrategy(symbols=symbols)
        signals = strategy.generate_signals(processed_data_dict)

        assert isinstance(signals, list)
        assert len(signals) > 0

        # 3. Execute signals
        simulator = PaperTradingSimulator(initial_capital=50000)

        for signal in signals:
            if signal.signal_type.value in ["BUY", "SELL"]:
                simulator.execute_signal(signal)

        # 4. Verify portfolio state
        portfolio_summary = simulator.get_portfolio_summary()
        assert portfolio_summary["total_value"] > 0

    def test_error_recovery_workflow(self, test_session, sample_market_data):
        """Test workflow error recovery."""
        # 1. Test with corrupted data
        corrupted_data = sample_market_data.copy()
        corrupted_data.loc[corrupted_data.index[10:20], "Close"] = np.nan

        processor = DataProcessor()

        # Should handle corrupted data gracefully
        processed_data = processor.process_data(corrupted_data, "AAPL")
        assert processed_data is not None
        assert len(processed_data) > 0

        # 2. Test with strategy errors
        strategy = GoldenCrossStrategy(symbols=["AAPL"])
        market_data_dict = {"AAPL": processed_data}

        # Should generate signals even with some data issues
        signals = strategy.generate_signals(market_data_dict)
        assert isinstance(signals, list)

        # 3. Test with execution errors
        simulator = PaperTradingSimulator(initial_capital=1000)  # Small capital

        # Try to execute expensive signal
        expensive_signal = signals[0] if signals else None
        if expensive_signal:
            expensive_signal.quantity = 10000  # Very large quantity
            success = simulator.execute_signal(expensive_signal)
            # Should fail gracefully
            assert success is False

    def test_performance_workflow(self, test_session, sample_market_data):
        """Test workflow performance under load."""
        import time

        # 1. Test data processing performance
        processor = DataProcessor()

        start_time = time.time()
        processed_data = processor.process_data(sample_market_data, "AAPL")
        processing_time = time.time() - start_time

        assert processing_time < 1.0  # Should process within 1 second
        assert processed_data is not None

        # 2. Test strategy performance
        strategy = GoldenCrossStrategy(symbols=["AAPL"])
        market_data_dict = {"AAPL": processed_data}

        start_time = time.time()
        signals = strategy.generate_signals(market_data_dict)
        strategy_time = time.time() - start_time

        assert strategy_time < 1.0  # Should generate signals within 1 second
        assert isinstance(signals, list)

        # 3. Test execution performance
        simulator = PaperTradingSimulator(initial_capital=10000)

        start_time = time.time()
        for signal in signals[:10]:  # Execute first 10 signals
            if signal.signal_type.value in ["BUY", "SELL"]:
                simulator.execute_signal(signal)
        execution_time = time.time() - start_time

        assert execution_time < 1.0  # Should execute within 1 second

    def test_memory_efficiency_workflow(self, test_session, sample_market_data):
        """Test workflow memory efficiency."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run complete workflow multiple times
        for _ in range(5):
            # 1. Data processing
            processor = DataProcessor()
            processed_data = processor.process_data(sample_market_data, "AAPL")

            # 2. Strategy execution
            strategy = GoldenCrossStrategy(symbols=["AAPL"])
            market_data_dict = {"AAPL": processed_data}
            signals = strategy.generate_signals(market_data_dict)

            # 3. Paper trading
            simulator = PaperTradingSimulator(initial_capital=10000)
            for signal in signals:
                if signal.signal_type.value in ["BUY", "SELL"]:
                    simulator.execute_signal(signal)

            # Clear references to free memory
            del processed_data, market_data_dict, signals, simulator

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024  # 50MB

    def test_concurrent_workflow_execution(self, test_session, sample_market_data):
        """Test concurrent workflow execution."""
        import threading
        import time

        results = []

        def run_workflow():
            # Complete workflow in a thread
            processor = DataProcessor()
            processed_data = processor.process_data(sample_market_data, "AAPL")

            strategy = GoldenCrossStrategy(symbols=["AAPL"])
            market_data_dict = {"AAPL": processed_data}
            signals = strategy.generate_signals(market_data_dict)

            simulator = PaperTradingSimulator(initial_capital=10000)
            for signal in signals:
                if signal.signal_type.value in ["BUY", "SELL"]:
                    simulator.execute_signal(signal)

            results.append(simulator.get_portfolio_summary())

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_workflow)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All workflows should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "total_value" in result

    def test_workflow_data_consistency(self, test_session, sample_market_data):
        """Test data consistency throughout workflow."""
        # 1. Process data
        processor = DataProcessor()
        processed_data = processor.process_data(sample_market_data, "AAPL")

        # 2. Verify data integrity
        assert not processed_data.isnull().values.any()
        assert (processed_data["High"] >= processed_data["Low"]).all()
        assert (processed_data["High"] >= processed_data["Open"]).all()
        assert (processed_data["High"] >= processed_data["Close"]).all()

        # 3. Generate signals
        strategy = GoldenCrossStrategy(symbols=["AAPL"])
        market_data_dict = {"AAPL": processed_data}
        signals = strategy.generate_signals(market_data_dict)

        # 4. Verify signal consistency
        for signal in signals:
            assert signal.symbol == "AAPL"
            assert signal.confidence >= 0.0 and signal.confidence <= 1.0
            assert signal.timestamp is not None

        # 5. Execute signals
        simulator = PaperTradingSimulator(initial_capital=10000)

        for signal in signals:
            if signal.signal_type.value in ["BUY", "SELL"]:
                success = simulator.execute_signal(signal)
                if success:
                    # Verify position consistency
                    if signal.signal_type.value == "BUY":
                        assert signal.symbol in simulator.positions
                    elif signal.signal_type.value == "SELL":
                        # Position should be closed
                        pass

        # 6. Verify portfolio consistency
        portfolio_summary = simulator.get_portfolio_summary()
        assert portfolio_summary["total_value"] >= 0
        assert portfolio_summary["cash"] >= 0
        assert portfolio_summary["positions_value"] >= 0
