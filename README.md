# Algorithmic Trading System

This is a comprehensive algorithmic trading system designed for small accounts ($500-$1000) with dual-asset capabilities across both equity and options markets. The system follows a modular pipeline architecture to collect market data, implement trading strategies across both asset types, and execute trades with sophisticated risk management.

## System Architecture

```
algotrading/
├── data/          # Enhanced data collection and processing
│   ├── collectors.py     # Equity + Options data fetching
│   ├── processors.py     # Data cleaning for both asset types
│   └── storage.py        # Enhanced schema for options data
├── strategies/    # Trading strategy implementations
│   ├── equity/           # 4 equity strategies
│   │   ├── mean_reversion.py
│   │   ├── deep_value.py
│   │   ├── etf_rotation.py
│   │   └── golden_cross.py
│   ├── options/          # 5 options strategies
│   │   ├── covered_calls.py
│   │   ├── cash_secured_puts.py
│   │   ├── iron_condors.py
│   │   ├── credit_spreads.py
│   │   └── protective_puts.py
│   └── base.py           # Common strategy interface
├── indicators/    # Technical indicator calculations
│   ├── technical.py      # Equity indicators
│   └── options.py        # Options indicators (IV, Greeks)
├── backtesting/   # Backtesting framework for both asset types
├── risk/          # Enhanced risk management
│   ├── equity_risk.py    # Equity-specific risk rules
│   ├── options_risk.py   # Options-specific risk rules
│   └── portfolio_risk.py # Portfolio-level risk management
├── execution/     # Trade execution components
│   ├── alpaca.py         # Enhanced for options trading
│   ├── paper.py          # Paper trading for both assets
│   └── allocation.py     # Asset allocation management
├── utils/         # Utility functions and configs
└── pipeline.py    # Main orchestration with dual-asset support
```

## Dual-Asset Portfolio Structure

```
Total Portfolio ($500-$1000)
├── Equity Strategies (60-80% allocation)
│   ├── Mean Reversion (15-20%)
│   ├── Deep Value (15-20%)
│   ├── ETF Rotation (15-20%)
│   └── Golden Cross (15-20%)
└── Options Strategies (20-40% allocation)
    ├── Covered Calls (5-10%)
    ├── Cash Secured Puts (5-10%)
    ├── Iron Condors (3-8%)
    ├── Credit Spreads (3-8%)
    └── Protective Puts (2-5%)
```

## Features

### Equity Trading Capabilities

- Data collection from Yahoo Finance with incremental updates
- Four diversified equity strategies (mean reversion, deep value, ETF rotation, golden cross)
- Technical indicator analysis (RSI, MACD, moving averages, Bollinger Bands)
- Risk management with position sizing and circuit breakers

### Options Trading Capabilities

- Real-time options chain data collection via Alpaca API
- Five complementary options strategies for income and portfolio insurance
- Greeks-based risk management (Delta, Gamma, Theta, Vega monitoring)
- Implied volatility analysis and volatility-based strategy selection
- Assignment risk management and time decay optimization

### System-Wide Features

- PostgreSQL database for comprehensive trade tracking across both asset types
- Dynamic asset allocation between equity and options strategies
- Performance tracking and analytics for both asset classes
- Tax-efficient trading approach with long-term capital gains optimization
- Comprehensive backtesting framework for both equities and options

## Requirements

- Python 3.8+
- PostgreSQL database
- Alpaca brokerage account with options trading approval
- Required Python packages (see requirements.txt)

## Setup Instructions

### 1. Set up a virtual environment

```bash
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On macOS/Linux:
source env/bin/activate
# On Windows:
# env\Scripts\activate
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
# Collect equity data for all active symbols
python pipeline.py --task collect --asset-type equity

# Collect options data for specific underlyings
python pipeline.py --task collect --asset-type options --symbols AAPL MSFT SPY

# Collect data for both asset types
python pipeline.py --task collect --asset-type both

# Force update of existing data
python pipeline.py --task collect --force
```

### Strategy Execution (Coming Soon)

```bash
# Run equity strategies only
python pipeline.py --task trade --asset-type equity --paper

# Run options strategies only
python pipeline.py --task trade --asset-type options --paper

# Run all strategies with allocation management
python pipeline.py --task trade --asset-type both --paper
```

### Backtesting (Coming Soon)

```bash
# Backtest equity strategies
python pipeline.py --task backtest --asset-type equity --strategy mean_reversion

# Backtest options strategies
python pipeline.py --task backtest --asset-type options --strategy covered_calls

# Backtest complete dual-asset portfolio
python pipeline.py --task backtest --asset-type both
```

### Asset Allocation Management

```bash
# View current allocation
python pipeline.py --task allocation --action view

# Rebalance between equity and options
python pipeline.py --task allocation --action rebalance

# Set custom allocation percentages
python pipeline.py --task allocation --action set --equity-pct 70 --options-pct 30
```

## Trading Strategies

### Equity Strategies (60-80% of Portfolio)

1. **Mean Reversion**: Identifies assets 2+ standard deviations from 50-day moving average
2. **Deep Value**: Places limit orders 20-25% below market price on quality stocks
3. **ETF Rotation**: Rotates capital to top-performing sector ETFs based on 3-month performance
4. **Golden Cross**: Trades 50-day/200-day moving average crossovers

### Options Strategies (20-40% of Portfolio)

1. **Covered Calls**: Generate income on existing equity positions
2. **Cash Secured Puts**: Get paid to potentially buy stocks at lower prices
3. **Iron Condors**: Profit from low volatility, range-bound markets
4. **Credit Spreads**: Generate income with directional bias
5. **Protective Puts**: Portfolio insurance for equity positions

## Risk Management

### Equity Risk Controls

- Maximum 20% of portfolio in any single stock
- Maximum 30% of portfolio in any single ETF
- 10% stop-loss on individual positions
- Portfolio circuit breakers on 5% weekly losses

### Options Risk Controls

- Maximum 10% of portfolio per options trade
- Maximum 25% total options exposure
- Greeks-based limits (Delta: -0.3 to +0.5, controlled Gamma)
- Time-based exits (50% profit or 21 DTE)
- Assignment risk management

### Cross-Asset Risk Management

- Correlation monitoring between equity and options positions
- Dynamic allocation adjustments based on performance
- Stress testing across various market scenarios

## Performance Tracking

The system tracks comprehensive metrics across both asset types:

- **Equity Metrics**: Win rate, profit factor, Sharpe ratio, tax efficiency
- **Options Metrics**: Premium collection efficiency, assignment rates, Greeks performance
- **Portfolio Metrics**: Combined risk-adjusted returns, allocation efficiency, maximum drawdown

## Development Status & Progress Tracker

**Current Status**: ~60% Complete (Core Trading Engine Phase)

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

### 💰 **Phase 5: Options Trading** ❌ NOT STARTED

- [ ] **Options Data Collection**: Real-time options chains from Alpaca
- [ ] **Greeks Calculations**: Delta, Gamma, Theta, Vega analysis
- [ ] **Covered Calls Strategy**: Income generation on equity positions
- [ ] **Cash Secured Puts**: Quality stock acquisition at discounts
- [ ] **Iron Condors**: Low volatility range-bound profit strategy
- [ ] **Credit Spreads**: Directional income with technical bias
- [ ] **Protective Puts**: Portfolio insurance during uncertainty

### 📈 **Phase 6: Backtesting & Analysis** ❌ NOT STARTED

- [ ] **Backtesting Engine**: Historical data replay with realistic fills
- [ ] **Performance Metrics**: Sharpe ratio, max drawdown, win rate analysis
- [ ] **Strategy Comparison**: Side-by-side performance evaluation
- [ ] **Risk Analysis**: Portfolio-level risk assessment and stress testing

### 🔗 **Phase 7: Broker Integration** ❌ NOT STARTED

- [ ] **Alpaca API**: Paper trading integration for both equity and options
- [ ] **Order Management**: Limit orders, stop losses, bracket orders
- [ ] **Real-time Data**: Live market data feeds for signal generation
- [ ] **Trade Execution**: Automated order placement with safety checks

### 🎛️ **Phase 8: Portfolio Management** ❌ NOT STARTED

- [ ] **Asset Allocation**: Dynamic rebalancing between equity/options (60-80% / 20-40%)
- [ ] **Risk Controls**: Portfolio-level circuit breakers and position limits
- [ ] **Tax Optimization**: Long-term vs short-term gains management
- [ ] **Performance Reporting**: Comprehensive strategy and portfolio analytics

### 🔄 **Current Priorities**

**NEXT UP**:

1. **Test Complete Pipeline** - Validate strategy signals with existing database data
2. **Implement Alpha Vantage** - Alternative data source to address rate limiting
3. **Build Backtesting Framework** - Historical strategy validation and performance metrics
4. **Complete Remaining Equity Strategies** - Deep Value, ETF Rotation, Golden Cross

### 🔧 **Known Issues**

- **Yahoo Finance Rate Limiting**: Enhanced protection implemented but still encounters limits
- **Limited Historical Data**: Only ~20 records in database, need more for strategy testing
- **No Real Broker Connection**: Currently theoretical - need Alpaca integration for live trading

## Contributing

This is a personal project, but suggestions and improvements are welcome.

## License

MIT License
