# Algorithmic Trading System

This is a comprehensive algorithmic trading system designed for small accounts ($500-$1000) focusing on equity markets. The system follows a modular pipeline architecture to collect market data, implement diversified equity trading strategies, and execute trades with sophisticated risk management.

## System Architecture

```
algotrading/
├── data/          # Data collection and processing
│   ├── collectors.py     # Market data fetching from Yahoo Finance, Alpha Vantage
│   ├── processors.py     # Data cleaning and validation
│   └── storage.py        # Database operations with PostgreSQL
├── strategies/    # Trading strategy implementations
│   ├── equity/           # 4 equity strategies
│   │   ├── mean_reversion.py
│   │   ├── deep_value.py
│   │   ├── etf_rotation.py
│   │   └── golden_cross.py
│   └── base.py           # Common strategy interface
├── indicators/    # Technical indicator calculations
│   └── technical.py      # RSI, MACD, Bollinger Bands, Moving Averages
├── backtesting/   # Backtesting framework
├── risk/          # Risk management
│   └── portfolio_risk.py # Position sizing and circuit breakers
├── execution/     # Trade execution components
│   ├── alpaca.py         # Alpaca API integration
│   └── paper.py          # Paper trading simulation
├── utils/         # Utility functions and configs
└── pipeline.py    # Main orchestration
```

## Portfolio Structure

```
Total Portfolio ($500-$1000)
├── Mean Reversion Strategy (25%)
├── Deep Value Strategy (25%)
├── ETF Rotation Strategy (25%)
└── Golden Cross Strategy (25%)
```

## Features

### Equity Trading Capabilities

- Data collection from Yahoo Finance with incremental updates
- Four diversified equity strategies for different market conditions
- Technical indicator analysis (RSI, MACD, moving averages, Bollinger Bands)
- Risk management with position sizing and circuit breakers
- Tax-efficient trading approach with long-term capital gains optimization

### System-Wide Features

- PostgreSQL database for comprehensive trade tracking
- Performance tracking and analytics with key metrics
- Backtesting framework for strategy validation
- Automated data collection and processing pipeline
- Configurable risk management and position sizing

## Requirements

- Python 3.8+
- PostgreSQL database
- Alpaca brokerage account for live trading
- Required Python packages (see requirements.txt)

## Setup Instructions

### 1. Set up a virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Set up the PostgreSQL database

- Install PostgreSQL if not already installed
- Create a database named 'algotrading'
- Set environment variables for database connection:

```bash
# Copy the example file and customize it
cp env.example .env

# Edit .env with your database credentials
# Required: DB_HOST, DB_NAME, DB_USER
# Optional: DB_PASSWORD, ALPACA_API_KEY, etc.
```

### 4. Initialize database and collect initial data

```bash
# Run the pipeline with the 'collect' task
python pipeline.py --task collect
```

## Usage

### Data Collection

```bash
# Collect data for all active symbols
python pipeline.py --task collect

# Collect data for specific symbols
python pipeline.py --task collect --symbols AAPL MSFT SPY

# Force update of existing data
python pipeline.py --task collect --force
```

### Strategy Execution (Coming Soon)

```bash
# Run all strategies in paper trading mode
python pipeline.py --task trade --paper

# Run specific strategy
python pipeline.py --task trade --strategy mean_reversion --paper

# View strategy signals without trading
python pipeline.py --task signals
```

### Backtesting (Coming Soon)

```bash
# Backtest all strategies
python pipeline.py --task backtest

# Backtest specific strategy
python pipeline.py --task backtest --strategy mean_reversion

# Generate performance report
python pipeline.py --task backtest --report
```

## Trading Strategies

The system implements four diversified equity strategies to reduce risk and maximize opportunities across different market conditions:

### 1. Mean Reversion Strategy (25% of Portfolio)

**Concept**: Identifies assets that have deviated significantly from their historical averages and bets on them returning to normal levels.

**Implementation**:

- Track stocks and ETFs that are 2+ standard deviations from their 50-day moving average
- Use RSI and Bollinger Bands for confirmation
- Place limit orders when assets are significantly oversold
- Set profit targets at 15-20% to offset tax impacts

**Asset Allocation**:

- 60% individual stocks (higher volatility creates more opportunities)
- 40% sector ETFs

### 2. Deep Value Strategy (25% of Portfolio)

**Concept**: Places limit orders significantly below market price and waits for volatility to fill them, similar to patient value investing.

**Implementation**:

- Identify quality assets with good fundamentals
- Place limit buy orders 20-25% below current market prices
- Set take-profit targets at 20%+ (preferably holding for >1 year)
- Accept that many orders will never fill

**Asset Allocation**:

- 60% individual quality stocks with higher volatility
- 40% sector ETFs for market-wide correction opportunities

### 3. ETF Rotation Strategy (25% of Portfolio)

**Concept**: Identifies the strongest sector ETFs and rotates capital to follow market trends.

**Implementation**:

- Track 5-10 sector ETFs (technology, finance, healthcare, etc.)
- Weekly: Rank them by 3-month performance metrics
- Buy top 1-2 performers, hold until they drop below a certain rank
- Review and potentially rotate positions monthly

**Asset Allocation**:

- 100% sector ETFs (this strategy is specifically designed for ETFs)

### 4. Golden Cross Strategy (25% of Portfolio)

**Concept**: A classic trend-following approach that buys when short-term momentum crosses above long-term trends.

**Implementation**:

- Buy when the 50-day moving average crosses above the 200-day moving average
- Sell when it crosses below
- Apply to both broad market ETFs and select large-cap stocks
- Typically generates just a few trades per year

**Asset Allocation**:

- 50% broad market ETFs (SPY, QQQ)
- 50% large-cap individual stocks

## Risk Management

### Position Sizing & Risk Controls

- Maximum 20% of portfolio in any single stock position
- Maximum 30% of portfolio in any single ETF position
- Stop-loss orders set at maximum 10% below entry for individual stocks
- Portfolio-level circuit breakers (pause trading if overall portfolio drops >5% in a week)
- Fractional shares used to properly size positions within budget

### Risk Management Protocols

- Profit-taking targets defined for each strategy
- Weekly review of all positions and strategy performance
- Tax-loss harvesting protocols
- Position sizes scaled according to volatility (more volatile assets get smaller positions)

## Performance Tracking

The system tracks comprehensive metrics to evaluate strategy performance:

### Key Performance Metrics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits divided by gross losses
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Tax Efficiency**: Long-term vs short-term capital gains ratio

### Database Tracking

- **Trades Table**: Complete transaction history with entry/exit prices, dates, P&L
- **Trade Signals Table**: Technical indicators at time of signal generation
- **Strategy Performance**: Win/loss ratios, average holding periods, profit factors by strategy
- **Portfolio Tracking**: Asset allocation over time, cumulative returns, sector exposure

## Development Status & Progress Tracker

**Current Status**: ~40% Complete (Core Infrastructure Phase)

### 📊 **Phase 1: Data Infrastructure** ✅ COMPLETE

- [x] **Database Setup**: PostgreSQL connection and schema
- [x] **Data Models**: SQLAlchemy ORM models (MarketData, Symbol, Logs)
- [x] **Data Processing**: Validation, cleaning, and transformation pipeline
- [x] **Rate Limiting**: Enhanced Yahoo Finance collector with retry logic
- [x] **Pipeline Orchestration**: Basic data collection workflow

### 🔧 **Phase 2: Technical Analysis Engine** ✅ COMPLETE

- [x] **Technical Indicators**: 21 comprehensive indicators (RSI, MACD, Bollinger Bands, etc.)
- [x] **Indicator Framework**: TechnicalIndicators class with method chaining
- [x] **Signal Detection**: Built-in oversold/overbought/crossover detection
- [x] **Performance Optimization**: Pandas-based calculations for speed

### 🎯 **Phase 3: Strategy Framework** ✅ COMPLETE

- [x] **Base Strategy Class**: Abstract interface for all trading strategies
- [x] **Signal Management**: StrategySignal class with confidence scoring
- [x] **Position Tracking**: Entry/exit tracking with P&L calculation
- [x] **Risk Management**: Stop-loss, take-profit, position sizing logic
- [x] **Mean Reversion Strategy**: Full implementation with multi-factor signals

### 🚀 **Phase 4: Equity Strategies** 🔄 IN PROGRESS

- [x] **Mean Reversion**: ✅ Complete - RSI + Bollinger Bands + MA deviation
- [ ] **Deep Value**: Limit order strategy 20-25% below market
- [ ] **ETF Rotation**: Top-performing sector ETF rotation based on 3-month performance
- [ ] **Golden Cross**: 50/200 day MA crossover strategy

### 📈 **Phase 5: Backtesting & Analysis** ❌ NOT STARTED

- [ ] **Backtesting Engine**: Historical data replay with realistic fills
- [ ] **Performance Metrics**: Sharpe ratio, max drawdown, win rate analysis
- [ ] **Strategy Comparison**: Side-by-side performance evaluation
- [ ] **Risk Analysis**: Portfolio-level risk assessment and stress testing

### 🔗 **Phase 6: Broker Integration** ❌ NOT STARTED

- [ ] **Alpaca API**: Paper trading integration
- [ ] **Order Management**: Limit orders, stop losses, bracket orders
- [ ] **Real-time Data**: Live market data feeds for signal generation
- [ ] **Trade Execution**: Automated order placement with safety checks

### 🎛️ **Phase 7: Portfolio Management** ❌ NOT STARTED

- [ ] **Risk Controls**: Portfolio-level circuit breakers and position limits
- [ ] **Tax Optimization**: Long-term vs short-term gains management
- [ ] **Performance Reporting**: Comprehensive strategy and portfolio analytics
- [ ] **Rebalancing**: Automated portfolio rebalancing between strategies

### 🔄 **Current Priorities**

**NEXT UP**:

1. **Complete Remaining Equity Strategies** - Deep Value, ETF Rotation, Golden Cross
2. **Build Backtesting Framework** - Historical strategy validation and performance metrics
3. **Implement Alpha Vantage** - Alternative data source for enhanced reliability
4. **Portfolio Risk Management** - Complete risk management system implementation

### 🔧 **Known Issues**

- **Rate Limiting**: Yahoo Finance API has rate limits that may cause delays during initial data collection
- **Data Quality**: Some data cleaning rules may need refinement based on actual market data patterns
- **Strategy Testing**: Need comprehensive backtesting before live deployment

## Operational Schedule

- **Daily**: Automated data collection and storage
- **Weekly (Weekend)**: Strategy calculations and signal generation
- **Weekly (Monday)**: Order placement for the coming week
- **Monthly**: Performance review and strategy adjustment
- **Quarterly**: Tax optimization review

## Success Criteria

### Year 1 Success Metrics

- **Profitability**: Outperform high-yield savings accounts (>4% annual return)
- **Risk Management**: Maximum drawdown less than 15% at any point
- **Strategy Performance**: At least 2 of the 4 strategies showing positive returns
- **Automation**: System operating with minimal manual intervention

### Future Expansion

As the account grows and strategies prove successful:

- Increase capital allocation
- Develop more sophisticated machine learning models
- Expand data sources to include alternative data
- Explore additional asset classes (REITs, international ETFs)

## Contributing

This is a personal algorithmic trading system designed for educational and investment purposes. The codebase follows Python best practices and includes comprehensive testing.

## License

This project is for personal use only. Please ensure compliance with all applicable financial regulations and broker terms of service.
