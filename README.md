# AlgoTrading System

A Python-based algorithmic trading system designed for small accounts ($500-$1000) with a focus on tax efficiency, minimal operating costs, and maintainable infrastructure.

## What This Is

This is a complete algorithmic trading system that:

- Runs multiple diversified strategies across various assets
- Uses industry-standard libraries (PFund, Backtrader, Cvxportfolio, PyPortfolioOpt)
- Provides a professional dashboard with real-time Alpaca integration
- Includes comprehensive backtesting and performance analysis
- Supports both paper trading and live trading

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <your-repo>
cd AlgoTrading

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env.example .env
```

Edit `.env` with your configuration:

```
DB_HOST=localhost
DB_NAME=algotrading
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432

# Required for Alpaca integration
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

### 2. Collect Market Data

```bash
# Collect 5 years of data for SPY, QQQ, VTI
python pipeline.py --task collect --symbols SPY QQQ VTI --period 5y
```

### 3. Run Backtesting

```bash
# Test Golden Cross strategy
python pipeline.py --task backtest --strategy golden_cross --years 3

# Or use the dedicated test script
python backtesting/test_golden_cross.py
```

### 4. Generate Trading Signals

```bash
# Check current signals
python pipeline.py --task signals --strategy golden_cross
```

### 5. Launch Dashboard

```bash
# Start the web dashboard
python run_dashboard.py
```

## Trading Strategies

The system implements four diversified strategies designed for small accounts:

### 1. Golden Cross Strategy

- **Concept**: 50/200-day moving average crossover
- **Allocation**: 50% broad market ETFs (SPY, QQQ), 50% large-cap stocks
- **Frequency**: Few trades per year, trend-following
- **Risk**: Medium

### 2. Mean Reversion Strategy

- **Concept**: Identifies assets 2+ standard deviations from 50-day moving average
- **Allocation**: 60% individual stocks, 40% sector ETFs
- **Frequency**: Weekly review and adjustments
- **Risk**: Medium-High

### 3. ETF Rotation Strategy

- **Concept**: Rotates to strongest sector ETFs based on 3-month performance
- **Allocation**: 100% sector ETFs
- **Frequency**: Monthly rotation
- **Risk**: Medium

### 4. Dual Momentum Strategy

- **Concept**: Gary Antonacci's absolute + relative momentum approach
- **Allocation**: Diversified across asset classes
- **Frequency**: Monthly rebalancing
- **Risk**: Medium

## System Architecture

### Core Components

1. **Data Pipeline** (`pipeline.py`)

   - Collects market data from Alpaca API
   - Processes and stores data in PostgreSQL
   - Handles symbol normalization and data validation

2. **Strategy Engine** (`strategies/`)

   - Implements all four trading strategies
   - Generates buy/sell signals
   - Includes risk management and position sizing

3. **Backtesting Engine** (`backtesting/`)

   - Historical performance validation
   - Comprehensive metrics (Sharpe ratio, drawdown, win rate)
   - Uses Backtrader framework (industry standard)

4. **Execution Engine** (`execution/`)

   - Paper trading and live trading capabilities
   - Alpaca API integration
   - Order management and position tracking

5. **Dashboard** (`dashboard/`)
   - Web-based interface for monitoring
   - Real-time portfolio tracking
   - Performance analytics and reporting

### Key Libraries Used

- **PFund**: Modern ML-ready algo-trading framework
- **Backtrader**: Professional backtesting (used by banks)
- **Cvxportfolio**: Academic-grade portfolio optimization
- **PyPortfolioOpt**: Community-tested optimization
- **Alpaca**: Commission-free trading API
- **PostgreSQL**: Trade tracking and performance metrics

## Portfolio Management

### Small Account Safety Controls

For accounts under $1K, the system implements specific safety measures:

- **Position Limits**: Maximum $300 per position (30% of $1K)
- **Minimum Position**: $50 per position (5% of $1K)
- **Cash Buffer**: Maintain $100 minimum for opportunities
- **Maximum Positions**: 4 positions for $1K account
- **Position Sizing**: $200-300 based on signal confidence

### Risk Management

- **Stop Losses**: Automatic stop-loss placement
- **Position Correlation**: Avoids highly correlated positions
- **Sector Limits**: Maximum 40% in any single sector
- **Cash Management**: Maintains liquidity for opportunities

## Development and Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Adding New Strategies

1. Create strategy class in `strategies/`
2. Implement required methods (generate_signals, etc.)
3. Add to strategy factory in `strategies/modern_strategies.py`
4. Create backtesting script
5. Add to dashboard components

### Code Structure

```
AlgoTrading/
├── strategies/          # Trading strategy implementations
├── backtesting/         # Backtesting engine and tests
├── data/               # Data collection and processing
├── execution/          # Order execution and Alpaca integration
├── dashboard/          # Web dashboard and UI
├── utils/              # Utilities and helpers
├── tests/              # Test suite
└── scripts/            # Utility scripts
```

## Production Deployment

### Daily Automation

The system can be automated for daily execution:

```bash
# Setup cron job for 4:30 PM ET daily execution
0 16 30 * * * /path/to/python /path/to/scripts/daily_after_market.py
```

### Email Notifications

Configure email notifications for daily signals and portfolio updates in `.env`:

```
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

### Monitoring

- **Logs**: All operations logged to `logs/` directory
- **Alerts**: Email notifications for critical events
- **Dashboard**: Real-time monitoring via web interface
- **Performance**: Automated performance tracking and reporting

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade**: Mean profit/loss per trade
- **Volatility**: Portfolio price variability

## Incremental Data Loading

The system now includes intelligent incremental data loading to significantly improve performance:

### Key Features

- **Smart Delta Detection**: Only downloads missing data by checking local database first
- **Caching System**: In-memory caching for frequently accessed data
- **Bulk Operations**: Optimized database operations for large datasets
- **Gap Detection**: Automatically detects and fills data gaps
- **Batch Processing**: Efficient processing of multiple symbols

### Performance Benefits

- **Faster Strategy Runs**: Subsequent runs use cached/existing data
- **Reduced API Calls**: Only fetches missing dates, not entire datasets
- **Lower Bandwidth**: Minimal data transfer for daily updates
- **Better Reliability**: Handles network issues gracefully with fallbacks

### Usage

The incremental loading is automatically used by all data collection methods:

```python
# Automatic incremental loading
data = collector.incremental_fetch_daily_data(session, "SPY", period="1y")

# Batch processing for multiple symbols
batch_data = collector.incremental_fetch_batch(session, ["SPY", "QQQ", "VTI"])
```

### Testing

Test the incremental loading system:

```bash
python test_incremental_loading.py
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**

   - Check PostgreSQL is running
   - Verify credentials in `.env`
   - Ensure database exists

2. **Alpaca API Errors**

   - Verify API keys in `.env`
   - Check account status in Alpaca dashboard
   - Ensure paper trading is enabled for testing

3. **Data Collection Issues**
   - Check internet connection
   - Verify symbol names are valid
   - Check Alpaca API rate limits

### Getting Help

- Check logs in `logs/` directory
- Review test output for specific errors
- Verify environment configuration
- Check Alpaca account status

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

This project is for educational and personal use. Please ensure compliance with your local trading regulations and broker terms of service.

## Disclaimer

This software is for educational purposes only. Past performance does not guarantee future results. Trading involves risk of loss. Always test thoroughly with paper trading before using real money.
