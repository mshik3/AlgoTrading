# Algo Trading Setup

## 1. Introduction

This document serves as a comprehensive guide for implementing a personal algorithmic trading system designed for a small account ($500-$1000). The system is built with tax efficiency, minimal operating costs, and maintainable infrastructure in mind.

The system is designed for a single trader with Python expertise, using home equipment for execution and potentially lightweight cloud services. The primary goal is to develop profitable strategies across both equity and options markets that require minimal maintenance while allowing the portfolio to grow organically.

## 2. System Overview

### 2.1 Core Architecture

The algorithmic trading system consists of six primary components:

1. **Data Feed** - Collects market data from free/low-cost sources (equity + options)
2. **Strategy Engine** - Processes data and generates trading signals for both asset types
3. **Execution Engine** - Places orders with the broker for equities and options
4. **Risk Manager** - Monitors positions and prevents excessive losses across asset types
5. **Asset Allocation Manager** - Manages capital distribution between equity and options strategies
6. **Analytics** - Tracks performance and provides feedback for both asset classes

The system will execute trades on a weekly basis to minimize transaction costs and tax implications, while monitoring the market daily for significant events across both equity and options markets.

### 2.2 Enhanced Technical Stack

- **Programming Language**: Python
- **Database**: PostgreSQL for trade tracking, performance metrics, and options data
- **Broker API**: Alpaca (commission-free API trading for stocks and options)
- **Data Sources**: Yahoo Finance API, Alpha Vantage, Alpaca Options Data
- **Execution Environment**: Local system with scheduled tasks
- **Optional Cloud Infrastructure**: Minimal cloud resources if needed for reliability

### 2.3 Dual-Asset Portfolio Architecture

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

## 3. Equity Trading Strategies

The system will implement four diversified equity strategies across various assets to reduce risk and create an experimental framework for discovering what works best. The equity portion (60-80% of portfolio) will be split into four strategy buckets with further diversification within each strategy.

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

## 4. Options Trading Strategies

The options component (20-40% of portfolio) implements five complementary strategies designed to generate income, provide portfolio insurance, and capitalize on volatility opportunities.

### 4.1 Covered Calls Strategy

**Concept**: Generate additional income on existing equity positions by selling call options against owned stocks.

**Implementation**:

- Write covered calls on equity positions when implied volatility is above historical average
- Target 30-45 days to expiration with strikes 5-10% out of the money
- Close positions at 50% profit or 21 days to expiration
- Roll positions if assignment risk becomes high on profitable equity positions

**Risk Management**:

- Only write calls on stocks willing to sell at strike price
- Maximum 50% of any equity position can be covered
- Monitor dividend dates to avoid early assignment

**Asset Allocation**: 5-10% of total portfolio

### 4.2 Cash Secured Puts Strategy

**Concept**: Generate income while potentially acquiring quality stocks at discounted prices.

**Implementation**:

- Sell cash-secured puts on stocks from the equity strategy watchlist
- Target strikes 10-20% below current market price
- Use 30-45 days to expiration for optimal time decay
- Keep cash reserves to handle assignment

**Risk Management**:

- Only sell puts on stocks willing to own long-term
- Ensure sufficient cash to handle assignment
- Avoid earnings announcements and major events

**Asset Allocation**: 5-10% of total portfolio

### 4.3 Iron Condors Strategy

**Concept**: Profit from low volatility and range-bound market conditions.

**Implementation**:

- Sell OTM put spread and OTM call spread simultaneously
- Target high implied volatility periods (VIX > 20)
- Use broad market ETFs (SPY, QQQ) for better liquidity
- Target 30-45 days to expiration

**Risk Management**:

- Close at 50% profit or when underlying approaches either strike
- Maximum loss limited to spread width minus premium collected
- Monitor Delta exposure and adjust if needed

**Asset Allocation**: 3-8% of total portfolio

### 4.4 Credit Spreads Strategy

**Concept**: Generate income with directional bias using vertical spreads.

**Implementation**:

- Bull put spreads in uptrending markets
- Bear call spreads in downtrending markets
- Use technical analysis from equity strategies to determine direction
- Target 30-45 days to expiration

**Risk Management**:

- Limit spread width to manage maximum loss
- Close at 50% profit or when technical signals reverse
- Avoid spreads through earnings announcements

**Asset Allocation**: 3-8% of total portfolio

### 4.5 Protective Puts Strategy

**Concept**: Provide portfolio insurance for equity positions during uncertain market conditions.

**Implementation**:

- Buy puts on major equity positions when VIX spikes above 25
- Target puts 5-10% out of the money with 60-90 days to expiration
- Use during earnings seasons or major economic events
- Focus on largest equity positions for cost efficiency

**Risk Management**:

- Limit premium spent to 2-3% of protected position value
- Close puts when VIX returns to normal levels (below 20)
- Consider rolling puts if protection period needs extension

**Asset Allocation**: 2-5% of total portfolio

## 5. Enhanced Position Sizing & Risk Management

### 5.1 Equity Position Sizing

- Maximum 20% of portfolio in any single stock position
- Maximum 30% of portfolio in any single ETF position
- Fractional shares will be used to properly size positions within budget
- Initial position sizes will be scaled according to volatility

### 5.2 Options Position Sizing

- Maximum 10% of portfolio in any single options trade
- Maximum 25% of portfolio in total options exposure
- No more than 5 options contracts per trade initially
- Position sizes scaled based on strategy risk profile

### 5.3 Portfolio-Level Risk Management

**Equity Risk Controls**:

- Stop-loss orders set at maximum 10% below entry for individual stocks
- Portfolio-level circuit breakers (pause trading if overall portfolio drops >5% in a week)
- Profit-taking targets defined for each strategy

**Options Risk Controls**:

- Greeks-based position limits:
  - Portfolio Delta: -0.3 to +0.5 (slight bullish bias)
  - Maximum Gamma exposure: 0.1 per $1000 portfolio
  - Target positive Theta (net premium collection)
  - Monitor Vega during high volatility periods
- Time-based exits: Close positions at 50% profit or 21 days to expiration
- Assignment risk management for short options positions

**Cross-Asset Risk Management**:

- Correlation monitoring between equity and options positions
- Dynamic allocation adjustments based on performance
- Stress testing for various market scenarios

## 6. Enhanced Data Processing & Analysis

### 6.1 Data Sources

**Equity Data**:

- Free tier of financial APIs (Yahoo Finance, Alpha Vantage)
- Weekly candlestick data for analysis
- Fundamental data for initial stock selection

**Options Data**:

- Real-time options chains from Alpaca API
- Implied volatility data and historical volatility
- Options volume and open interest data
- Greeks calculations (Delta, Gamma, Theta, Vega)

### 6.2 Technical Indicators

**Equity Indicators**:

- Moving Averages, RSI, Bollinger Bands, Volume analysis

**Options Indicators**:

- Greeks calculations (Delta, Gamma, Theta, Vega)
- Implied Volatility Rank and Percentile
- Put/Call Ratio analysis
- Volatility skew analysis

### 6.3 Enhanced Pattern Analysis

**Equity Patterns**:

- Engulfing patterns (bullish/bearish)
- Doji formations (signal potential reversals)
- Inside bar patterns (consolidation before breakouts)

**Options Patterns**:

- Volatility expansion/contraction cycles
- Earnings volatility crush patterns
- Expiration pinning effects
- Delta hedging flow analysis

## 7. Enhanced Performance Tracking

### 7.1 Equity Performance Metrics

- Win Rate (percentage of profitable trades)
- Profit Factor (gross profits divided by gross losses)
- Maximum Drawdown (largest peak-to-trough decline)
- Sharpe Ratio (risk-adjusted returns)
- Tax Efficiency (long-term vs short-term gains)

### 7.2 Options Performance Metrics

- Premium collection efficiency
- Assignment rate and profitability
- Greeks performance over time
- Volatility timing effectiveness
- Time decay capture rate

### 7.3 Cross-Asset Performance Analysis

- Correlation between equity and options performance
- Risk-adjusted returns by asset type
- Allocation efficiency metrics
- Combined portfolio Sharpe ratio and maximum drawdown

## 7.4 Enhanced Database Schema

A PostgreSQL database will track:

**Equity Tables**:

- **Trades Table**: Transaction ID, ticker, entry/exit prices, dates, quantity, strategy, position_type, profit_loss, tax_classification
- **Trade Signals Table**: Technical indicators, fundamental metrics, market conditions

**Options Tables**:

- **Options Data Table**: Symbol, expiration, strike, option_type, bid, ask, last, volume, open_interest, implied_volatility, Greeks
- **Options Trades Table**: Contract details, entry/exit, strategy_type, underlying_symbol, profit_loss, assignment_status
- **Options Positions Table**: Multi-leg position tracking for spreads and complex strategies

**Cross-Asset Tables**:

- **Asset Allocation Table**: Equity vs options allocation over time, rebalancing history
- **Portfolio Performance Table**: Combined metrics, risk-adjusted returns, drawdown analysis
- **Strategy Performance Table**: Win/loss ratio by strategy type, performance by asset class
- **System Logs Table**: Execution timestamps, errors, API monitoring, performance metrics

## 8. Enhanced System Execution

### 8.1 Operational Schedule

- **Daily**: Automated data collection for both equities and options
- **Weekly (Weekend)**: Strategy calculations and signal generation for both asset types
- **Weekly (Monday)**: Order placement for equity and options strategies
- **Monthly**: Performance review, allocation rebalancing, and strategy adjustment
- **Quarterly**: Tax optimization review and Greeks analysis

### 8.2 Infrastructure Requirements

- Home computer with reliable internet connection and options trading approval
- Scheduled tasks (cron jobs or Windows Task Scheduler)
- Enhanced database backup system for options data
- Optional: Basic cloud VPS for reliability if home system is insufficient

## 9. Enhanced Development Roadmap

### 9.1 Phase 1: Enhanced Setup & Backtesting

- Implement data collection infrastructure for equities and options
- Develop strategy logic for all four equity strategies and five options strategies
- Backtest strategies against historical data (including options historical data)
- Set up enhanced PostgreSQL database with options tracking

### 9.2 Phase 2: Dual-Asset Paper Trading

- Connect to Alpaca API in paper trading mode for both equities and options
- Run all strategies simultaneously with virtual money
- Track performance across both asset classes
- Implement all risk management protocols including Greeks monitoring

### 9.3 Phase 3: Live Dual-Asset Trading

- Start with minimal capital allocation ($500)
- Begin with conservative options strategies (covered calls, cash-secured puts)
- Gradually introduce more complex options strategies
- Implement tax-loss harvesting across both asset types

### 9.4 Phase 4: Advanced Options Integration

- Implement dynamic hedging strategies
- Add volatility forecasting models
- Develop cross-asset correlation strategies
- Implement automated assignment handling

## 10. Enhanced Success Criteria & Expansion

### 10.1 Success Metrics

**Overall Portfolio**:

- Profitable after 1 year (benchmark: outperforming high-yield savings accounts)
- Maximum drawdown less than 15% at any point
- Combined Sharpe ratio > 1.0

**Equity Strategies**:

- At least 2 of the 4 equity strategies showing positive returns
- Long-term capital gains rate > 60% of equity profits

**Options Strategies**:

- At least 3 of the 5 options strategies showing positive returns
- Options premium collection covering at least 50% of equity strategy drawdowns
- Assignment rate < 20% for short options strategies

**System Performance**:

- System operating with minimal manual intervention
- Less than 5% of trades requiring manual override

### 10.2 Future Expansion

As the account grows and strategies prove successful:

- Increase capital allocation across both asset types
- Develop more sophisticated options strategies (butterflies, calendars)
- Expand to futures and commodities options
- Implement machine learning models for volatility prediction
- Explore cryptocurrency options markets

## 11. Conclusion

This enhanced algorithmic trading system creates a comprehensive framework that leverages both equity and options markets to generate returns while managing risk. The diversified approach across strategies and asset types provides multiple income streams and risk mitigation.

By implementing this dual-asset system with careful risk management and consistent evaluation, we can build a robust trading approach that maximizes opportunities across both traditional equity investments and options income strategies.

The options component adds significant complexity but also provides powerful tools for income generation, portfolio insurance, and volatility trading that can enhance overall portfolio performance while maintaining controlled risk exposure.

# Algorithmic Trading System: Enhanced Technical Implementation

## 1. Enhanced System Architecture

- **Approach**: Dual-asset pipeline architecture with parallel processing for equities and options
- **Components**: Distinct modules for data collection, strategy calculation, signal generation, and execution across both asset types
- **Structure**:

```
algotrading/
├── data/          # Enhanced data collection and processing
│   ├── collectors.py     # Equity + Options data fetching
│   ├── processors.py     # Data cleaning for both asset types
│   └── storage.py        # Enhanced schema for options data
├── strategies/    # Trading strategy implementations
│   ├── equity/           # Original 4 equity strategies
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
├── backtesting/   # Enhanced backtesting framework
│   ├── equity_engine.py  # Equity backtesting
│   └── options_engine.py # Options backtesting
├── risk/          # Enhanced risk management
│   ├── equity_risk.py    # Equity-specific risk rules
│   ├── options_risk.py   # Options-specific risk rules
│   └── portfolio_risk.py # Portfolio-level risk management
├── execution/     # Enhanced trade execution
│   ├── alpaca.py         # Enhanced for options trading
│   ├── paper.py          # Paper trading for both assets
│   └── allocation.py     # Asset allocation management
├── utils/         # Utility functions and configs
│   ├── config.py         # Enhanced configuration
│   ├── logging.py        # Logging utilities
│   └── greeks.py         # Options Greeks calculations
└── pipeline.py    # Enhanced orchestration
```

## 2. Enhanced Database & Storage

- **Database**: PostgreSQL (local instance) with enhanced schema
- **Schema Design**: Multi-table approach for comprehensive tracking

```sql
-- Enhanced market data for equities
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

-- New options data table
CREATE TABLE options_data (
    id SERIAL PRIMARY KEY,
    underlying_symbol VARCHAR(10) NOT NULL,
    option_symbol VARCHAR(20) NOT NULL,
    expiration_date DATE NOT NULL,
    strike_price NUMERIC(10,2) NOT NULL,
    option_type VARCHAR(4) NOT NULL, -- 'CALL' or 'PUT'
    bid NUMERIC(6,2),
    ask NUMERIC(6,2),
    last NUMERIC(6,2),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility NUMERIC(6,4),
    delta NUMERIC(6,4),
    gamma NUMERIC(6,4),
    theta NUMERIC(6,4),
    vega NUMERIC(6,4),
    date_recorded DATE NOT NULL,
    UNIQUE(option_symbol, date_recorded)
);

-- Enhanced trades table for both asset types
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    asset_type VARCHAR(10) NOT NULL, -- 'EQUITY' or 'OPTION'
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    action VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'BTO', 'STC', etc.
    quantity INTEGER NOT NULL,
    price NUMERIC(10,2) NOT NULL,
    trade_date TIMESTAMP NOT NULL,
    position_type VARCHAR(20), -- 'LONG', 'SHORT', 'COVERED_CALL', etc.
    underlying_symbol VARCHAR(10), -- For options
    expiration_date DATE, -- For options
    strike_price NUMERIC(10,2), -- For options
    option_type VARCHAR(4), -- For options
    profit_loss NUMERIC(10,2),
    tax_classification VARCHAR(20),
    assignment_status VARCHAR(20) -- For options
);

-- Asset allocation tracking
CREATE TABLE asset_allocation (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    equity_percentage NUMERIC(5,2) NOT NULL,
    options_percentage NUMERIC(5,2) NOT NULL,
    equity_value NUMERIC(12,2) NOT NULL,
    options_value NUMERIC(12,2) NOT NULL,
    total_portfolio_value NUMERIC(12,2) NOT NULL,
    rebalance_trigger VARCHAR(100)
);
```

## 3. Enhanced Data Collection

- **Equity Data**: Yahoo Finance API with 5 years historical data
- **Options Data**: Alpaca API for real-time options chains and Greeks
- **Storage Requirements**: ~50MB for comprehensive dataset (equities + options)
- **Update Frequency**: Daily for equities, real-time for options during trading hours

## 4. Enhanced Technical Indicators

**Equity Indicators**:

- Moving Averages, RSI, Bollinger Bands, Volume analysis

**Options Indicators**:

- Greeks calculations (Delta, Gamma, Theta, Vega)
- Implied Volatility Rank and Percentile
- Put/Call Ratio analysis
- Volatility skew analysis

## 5. Enhanced Strategy Implementation

- **Dual-Asset Framework**: Parallel execution of equity and options strategies
- **Risk Coordination**: Cross-asset risk management and correlation monitoring
- **Performance Tracking**: Separate and combined performance metrics

## 6. Enhanced Risk Management

**Equity Risk Management**:

- Position sizing: Max 20% stocks, 30% ETFs
- Stop-losses: 10% maximum loss
- Portfolio circuit breakers: 5% weekly loss pause

**Options Risk Management**:

- Position sizing: Max 10% per trade, 25% total exposure
- Greeks limits: Delta -0.3 to +0.5, controlled Gamma exposure
- Time management: Close at 50% profit or 21 DTE
- Assignment risk protocols

**Cross-Asset Risk Management**:

- Correlation monitoring
- Dynamic allocation adjustments
- Stress testing across scenarios

## 7. Enhanced Execution System

- **Broker**: Alpaca API with options trading capabilities
- **Execution Modes**: Manual review, semi-automated, fully automated
- **Asset Allocation**: Dynamic rebalancing between equity and options strategies

This enhanced technical implementation provides a comprehensive framework for trading both equities and options with sophisticated risk management and performance tracking across both asset types.
