# Algo Trading Setup

## 1. Introduction

This document serves as a comprehensive guide for implementing a personal algorithmic trading system designed for a small account ($500-$1000). The system is built with tax efficiency, minimal operating costs, and maintainable infrastructure in mind.

The system is designed for a single trader with Python expertise, using home equipment for execution and potentially lightweight cloud services. The primary goal is to develop profitable strategies that require minimal maintenance while allowing the portfolio to grow organically.

## 2. System Overview

### 2.1 Core Architecture

The algorithmic trading system consists of five primary components:

1. **Data Feed** - Collects market data from free/low-cost sources
2. **Strategy Engine** - Processes data and generates trading signals
3. **Execution Engine** - Places orders with the broker
4. **Risk Manager** - Monitors positions and prevents excessive losses
5. **Analytics** - Tracks performance and provides feedback

The system will execute trades on a weekly basis to minimize transaction costs and tax implications, while monitoring the market daily for significant events.

### 2.2 Technical Stack

- **Programming Language**: Python
- **Database**: PostgreSQL for trade tracking and performance metrics
- **Broker API**: Alpaca (commission-free API trading)
- **Data Sources**: Yahoo Finance API, Alpha Vantage
- **Execution Environment**: Local system with scheduled tasks
- **Optional Cloud Infrastructure**: Minimal cloud resources if needed for reliability

## 3. Trading Strategies

The system will implement multiple diversified strategies across various assets to reduce risk and create an experimental framework for discovering what works best. The initial investment will be split into four strategy buckets ($125 each for a $500 starting account), with further diversification within each strategy.

### 3.1 Mean Reversion Strategy

**Concept**: This strategy identifies assets that have deviated significantly from their historical averages and bets on them returning to normal levels.

**Implementation**:

- Track stocks and ETFs that are 2+ standard deviations from their 50-day moving average
- Place limit orders when assets are significantly oversold
- Set profit targets at 15-20% to offset tax impacts
- Perform weekly review and position adjustments

**Asset Allocation**:

- 60% individual stocks (higher volatility creates more opportunities)
- 40% sector ETFs

### 3.2 Deep Value Limit Orders

**Concept**: Similar to the eBay strategy described (placing low bids and waiting patiently), this approach sets limit orders significantly below market price and waits for volatility to fill them.

**Implementation**:

- Identify quality assets with good fundamentals
- Place limit buy orders 20-25% below current market prices
- Set take-profit targets at 20%+ (preferably holding for >1 year if trending well)
- Accept that many orders will never fill

**Asset Allocation**:

- 60% individual quality stocks with higher volatility
- 40% sector ETFs for market-wide correction opportunities

### 3.3 ETF Rotation Strategy

**Concept**: This strategy identifies the strongest sector ETFs and rotates capital to follow market trends.

**Implementation**:

- Track 5-10 sector ETFs (technology, finance, healthcare, etc.)
- Weekly: Rank them by 3-month performance metrics
- Buy top 1-2 performers, hold until they drop below a certain rank
- Review and potentially rotate positions monthly

**Asset Allocation**:

- 100% sector ETFs (this strategy is specifically designed for ETFs)

### 3.4 Golden Cross Strategy

**Concept**: A classic trend-following approach that buys when short-term momentum crosses above long-term trends.

**Implementation**:

- Buy when the 50-day moving average crosses above the 200-day moving average
- Sell when it crosses below
- Apply to both broad market ETFs and select large-cap stocks
- Typically generates just a few trades per year

**Asset Allocation**:

- 50% broad market ETFs (SPY, QQQ)
- 50% large-cap individual stocks

### Position-Aware Strategy Logic

All strategies in this system are now position-aware, which is standard in professional trading systems:

- The system checks your current broker/account positions before making any trade recommendation.
- If you already hold a position, the strategy will only recommend scaling up, scaling down, holding, or closing, as appropriate.
- This prevents redundant trades, reduces unnecessary transaction costs, and ensures your portfolio stays in sync with strategy targets.
- Position-aware logic is a best practice for risk management and portfolio rebalancing in institutional trading.

## 4. Position Sizing & Risk Management

### 4.1 Position Sizing

- Maximum 20% of portfolio in any single stock position
- Maximum 30% of portfolio in any single ETF position
- Fractional shares will be used to properly size positions within budget
- Initial position sizes will be scaled according to volatility (more volatile assets get smaller positions)

### 4.2 Risk Management Protocols

- Stop-loss orders set at maximum 10% below entry for individual stocks
- Portfolio-level circuit breakers (pause trading if overall portfolio drops >5% in a week)
- Profit-taking targets defined for each strategy
- Weekly review of all positions and strategy performance

## 5. Data Processing & Analysis

### 5.1 Data Sources

- Free tier of financial APIs (Yahoo Finance, Alpha Vantage)
- Weekly candlestick data for analysis
- Fundamental data for initial stock selection

### 5.2 Technical Indicators

The system will leverage several key technical indicators:

- Moving Averages (50-day, 200-day)
- Relative Strength Index (RSI)
- Bollinger Bands
- Volume analysis
- Candlestick patterns (for additional confirmation)

### 5.3 Candlestick Pattern Analysis

For more sophisticated pattern recognition, the system will identify:

- Engulfing patterns (bullish/bearish)
- Doji formations (signal potential reversals)
- Inside bar patterns (consolidation before breakouts)

These patterns will serve as confirmatory signals rather than primary triggers.

## 6. Performance Tracking

### 6.1 Key Performance Metrics

- Win Rate (percentage of profitable trades)
- Profit Factor (gross profits divided by gross losses)
- Maximum Drawdown (largest peak-to-trough decline)
- Sharpe Ratio (risk-adjusted returns)
- Tax Efficiency (long-term vs short-term gains)

## 6.2 Database Schema

A PostgreSQL database will track:

- **Trades Table**
  - Transaction ID, ticker, entry/exit prices, dates, quantity
  - Strategy that triggered the trade
  - Position size and type (long/short)
  - Profit/loss and tax classification
- **Trade Signals Table**
  - Technical indicators at time of entry/exit:
    - RSI value
    - Bollinger Band position (% from middle band)
    - Moving average values and relationships
    - Volume indicators
    - Candlestick patterns identified
  - Fundamental metrics if applicable
  - Market conditions (VIX, sector performance)
- **Strategy Performance**
  - Win/loss ratio by strategy
  - Average holding period
  - Profit factor and Sharpe ratio
  - Maximum drawdown
  - Performance by asset class
- **Portfolio Tracking**
  - Asset allocation over time
  - Cumulative returns
  - Exposure by sector
  - Tax implications of realized/unrealized gains
- **System Logs**
  - Execution timestamps
  - Errors and warnings
  - API call monitoring
  - Performance metrics

## 7. System Execution

### 7.1 Operational Schedule

- **Daily**: Automated data collection and storage
- **Weekly (Weekend)**: Strategy calculations and signal generation
- **Weekly (Monday)**: Order placement for the coming week
- **Monthly**: Performance review and strategy adjustment
- **Quarterly**: Tax optimization review

### 7.2 Infrastructure Requirements

- Home computer with reliable internet connection
- Scheduled tasks (cron jobs or Windows Task Scheduler)
- Database backup system
- Optional: Basic cloud VPS for reliability if home system is insufficient

## 8. Development Roadmap

### 8.1 Phase 1: Setup & Backtesting

- Implement data collection infrastructure
- Develop strategy logic for all four strategies
- Backtest strategies against historical data
- Set up PostgreSQL database and tracking system

### 8.2 Phase 2: Paper Trading

- Connect to Alpaca API in paper trading mode
- Run all strategies simultaneously with virtual money
- Track performance and refine strategies
- Implement all risk management protocols

### 8.3 Phase 3: Live Trading

- Start with minimal capital allocation ($500)
- Gradually increase position sizes as strategies prove effective
- Implement tax-loss harvesting protocols
- Begin developing more sophisticated strategies

## 9. Success Criteria & Expansion

### 9.1 Success Metrics

- Profitable after 1 year (benchmark: outperforming high-yield savings accounts)
- Maximum drawdown less than 15% at any point
- At least 2 of the 4 strategies showing positive returns
- System operating with minimal manual intervention

### 9.2 Future Expansion

As the account grows and strategies prove successful:

- Increase capital allocation
- Develop more sophisticated machine learning models
- Expand data sources to include alternative data
- Potentially explore option-based strategies for income generation

## 10. Conclusion

This algorithmic trading system is designed as an experimental framework to discover profitable trading strategies while minimizing risk and operating costs. The diversified approach across both strategies and assets allows for ongoing learning and adaptation.

By implementing this system with careful risk management and consistent evaluation, we can build a robust trading approach that grows alongside improving expertise and increasing capital.

# Algorithmic Trading System: Technical Implementation Decisions

## 1. System Architecture

- **Approach**: Pipeline-based architecture with sequential processing
- **Components**: Distinct modules for data collection, strategy calculation, signal generation, and execution
- **Rationale**: Aligns with the sequential nature of trading operations and provides flexibility to modify individual components
- **Structure**:
  ```
  algotrading/
  ├── data/          # Data collection and processing
  │   ├── collectors.py  # Fetches data from sources
  │   ├── processors.py  # Cleans and prepares data
  │   └── storage.py     # Handles database operations
  ├── strategies/    # Trading strategy implementations
  │   ├── mean_reversion.py
  │   ├── deep_value.py
  │   ├── etf_rotation.py
  │   ├── golden_cross.py
  │   └── base.py        # Common strategy interface
  ├── indicators/    # Technical indicator calculations
  │   └── technical.py   # Wrapper for technical indicators
  ├── backtesting/   # Backtesting framework
  │   └── engine.py      # Interface with backtrader
  ├── risk/          # Risk management rules
  │   ├── strategy_risk.py  # Strategy-specific risk rules
  │   └── portfolio_risk.py # Portfolio-level risk management
  ├── execution/     # Trade execution components
  │   ├── alpaca.py      # Alpaca API integration
  │   ├── paper.py       # Paper trading simulation
  │   └── manual.py      # Manual execution helpers
  ├── utils/         # Utility functions and configs
  │   ├── config.py      # Configuration management
  │   └── logging.py     # Logging utilities
  └── pipeline.py    # Main orchestration
  ```

## 2. Database & Storage

- **Database**: PostgreSQL (local instance)
- **Schema Design**: Single table approach for market data

  ```sql
  CREATE TABLE market_data (
      id SERIAL PRIMARY KEY,
      symbol VARCHAR(10) NOT NULL,
      date DATE NOT NULL,
      open NUMERIC(10,2) NOT NULL,
      high NUMERIC(10,2) NOT NULL,
      low NUMERIC(10,2) NOT NULL,
      close NUMERIC(10,2) NOT NULL,
      volume BIGINT NOT NULL,
      adjusted_close NUMERIC(10,2) NOT NULL,
      UNIQUE(symbol, date)
  );

  ```

- **Storage Requirements**: ~10MB for initial dataset (5 years, 50 assets)
- **Scalability**: Start simple, with option to partition if needed in future

## 3. Data Collection

- **Approach**: Incremental data collection with tracking of last successful update
- **Data Sources**: Begin with Yahoo Finance API, with flexibility to add additional sources
- **Initial Dataset**:
  - 30-40 liquid stocks across major sectors
  - 10-15 key ETFs
  - 5 years of historical daily data
- **Required Fields**: Date, Open, High, Low, Close, Volume, Adjusted Close
- **Benchmarks**: VIX daily values, S&P 500 daily values

## 4. Technical Indicators

- **Implementation**: Leverage existing libraries (ta-lib, pandas-ta) for efficiency
- **Core Indicators**:
  - Moving Averages (50-day, 200-day)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - Volume analysis
  - Candlestick patterns

## 5. Strategy Implementation

- **Approach**: Implement all four strategies simultaneously for research purposes
- **Strategy Framework**: Common base class with strategy-specific implementations
- **Implementation Order**:
  1. Data collection first
  2. Backtesting framework
  3. Individual strategy implementation
  4. Risk management integration

## 6. Backtesting Framework

- **Library**: Backtrader for both vectorized and event-driven capabilities
- **Approach**: Hybrid approach - start with vectorized for speed, refine with event-driven
- **Historical Analysis**: Perform thorough backtesting before live implementation
- **Performance Metrics**: Implement all metrics specified in whitepaper (win rate, profit factor, etc.)

## 7. Risk Management

- **Approach**: Hybrid risk management
  - Basic risk parameters in each strategy
  - Portfolio-level risk manager for final approval
- **Implementation**: Rules-based system following whitepaper specifications
  - Maximum 20% of portfolio in any single stock position
  - Maximum 30% of portfolio in any single ETF position
  - Stop-loss orders at maximum 10% below entry for individual stocks
  - Portfolio-level circuit breakers (pause trading if overall portfolio drops >5% in a week)

## 8. Execution System

- **Flexibility**: Modular design supporting three execution modes:
  1. Manual review workflow (default for initial implementation)
  2. Semi-automated with approval
  3. Fully automated
- **Broker**: Alpaca API integration
- **Transition**: Designed to transition from manual to automated as confidence builds

## 9. Performance Tracking

- **Approach**: Database-driven metrics calculation with caching for historical data
- **Metrics Storage**: Dedicated tables for trade history and performance metrics
- **Dashboard**: Simple visualization planned for later phases
- **Tax Efficiency**: Tracking of tax implications for each trade

## 10. Scheduling & Automation

- **Scheduling Method**: Cron jobs for reliability and simplicity
- **Operational Schedule**:
  - Daily data collection
  - Weekly strategy calculations
  - Weekly trade execution (default manual review)
  - Monthly performance review

## 11. Development Practices

- **Testing**: Hybrid approach with unit tests for critical components
- **Code Organization**: Traditional Python project structure
- **Development Principle**: 80/20 approach - progress over perfection
- **Configuration**: Hybrid approach
  - Environment variables for sensitive data (.env file, excluded from Git)
  - Config files for non-sensitive settings
  - Template .env.example for documentation

## 12. Deployment & Operations

- **Environment**: Local execution with potential for lightweight cloud later
- **Monitoring**: Simple logging with error reporting
- **Backup**: Regular database backups
- **Security**: API keys and credentials stored securely in environment variables

## 13. Implementation Roadmap

1. **Phase 1**: Data Collection Pipeline
   - Database schema setup
   - Incremental data collector implementation
   - Data validation and storage
2. **Phase 2**: Backtesting Framework
   - Backtrader integration
   - Strategy implementation
   - Performance metrics
3. **Phase 3**: Risk Management & Execution
   - Risk rules implementation
   - Alpaca API integration
   - Execution modes
4. **Phase 4**: Monitoring & Refinement
   - Performance dashboard
   - Strategy refinement
   - System optimization
