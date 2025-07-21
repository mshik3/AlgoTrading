# ETF Rotation Strategies

This document provides comprehensive documentation for the ETF rotation strategies implemented in the AlgoTrading system.

## Overview

ETF rotation strategies are systematic approaches to tactical asset allocation that aim to outperform buy-and-hold strategies by dynamically allocating capital to the best-performing asset classes or sectors at any given time. These strategies leverage momentum and relative strength analysis to identify and invest in leading assets while avoiding or reducing exposure to underperforming ones.

## Strategy Types

### 1. Dual Momentum Strategy

**Implementation**: `strategies.etf.dual_momentum.DualMomentumStrategy`

**Concept**: Based on Gary Antonacci's proven dual momentum approach, this strategy combines:

- **Absolute Momentum**: Only invest in assets with positive momentum vs risk-free rate
- **Relative Momentum**: Among qualified assets, choose the one with highest momentum

**Key Features**:

- Monthly rebalancing (21-day frequency)
- Defensive positioning when no assets meet criteria
- Single asset allocation (concentrated approach)
- Risk-free rate comparison for absolute momentum

**Default ETF Universe**:

```python
{
    "US_Equities": ["SPY", "QQQ", "VTI", "IWM"],
    "International": ["EFA", "EEM", "VEA", "VWO"],
    "Bonds": ["TLT", "AGG", "BND", "LQD"],
    "Real_Estate": ["VNQ", "IYR", "SCHH"],
    "Commodities": ["GLD", "SLV", "USO", "DBA"],
    "Cash_Equivalents": ["SHY", "BIL", "SHV"]
}
```

**Configuration Parameters**:

- `absolute_momentum_lookback`: 252 days (1 year)
- `relative_momentum_lookback`: 252 days (1 year)
- `risk_free_rate`: 2% annual
- `defensive_asset`: "SHY" (short-term Treasuries)
- `max_positions`: 1 (single asset allocation)

### 2. Sector Rotation Strategy

**Implementation**: `strategies.etf.sector_rotation.SectorRotationStrategy`

**Concept**: Rotates among sector ETFs based on relative strength and momentum analysis to capture sector leadership changes.

**Key Features**:

- Multi-sector allocation (top 4 sectors by default)
- Equal-weight allocation within selected sectors
- Monthly rebalancing
- Volatility-adjusted momentum scoring
- Benchmark-relative performance analysis

**Default ETF Universe**:

```python
{
    "Technology": ["XLK", "VGT", "SMH"],
    "Financials": ["XLF", "VFH", "KBE"],
    "Healthcare": ["XLV", "VHT", "IHI"],
    "Consumer_Discretionary": ["XLY", "VCR", "XRT"],
    "Consumer_Staples": ["XLP", "VDC", "XLP"],
    "Industrials": ["XLI", "VIS", "XAR"],
    "Energy": ["XLE", "VDE", "XOP"],
    "Materials": ["XLB", "VAW", "XME"],
    "Real_Estate": ["XLRE", "VNQ", "IYR"],
    "Utilities": ["XLU", "VPU", "XLU"],
    "Communications": ["XLC", "VOX", "XLC"]
}
```

**Configuration Parameters**:

- `momentum_lookback`: 63 days (3 months)
- `relative_strength_lookback`: 252 days (1 year)
- `max_positions`: 4 sectors
- `benchmark_symbol`: "SPY"
- `sector_momentum_weight`: 0.6
- `relative_strength_weight`: 0.4

## Base Architecture

### BaseETFRotationStrategy

All ETF rotation strategies inherit from `BaseETFRotationStrategy`, which provides:

**Common Functionality**:

- Momentum calculation (returns, Sharpe, Sortino ratios)
- Relative strength analysis
- ETF ranking and selection
- Rebalancing frequency management
- Position sizing and allocation
- Risk management (stop-loss, take-profit)

**Key Methods**:

- `calculate_momentum()`: Calculate momentum scores
- `calculate_relative_strength()`: Compare vs benchmark
- `rank_etfs_by_momentum()`: Rank ETFs by performance
- `should_rebalance()`: Check rebalancing frequency
- `generate_rebalancing_signals()`: Create rebalancing orders

## Momentum Calculation Methods

### 1. Returns Method

Simple cumulative return over lookback period:

```
momentum = (current_price / price_lookback_periods_ago - 1)
```

### 2. Sharpe Ratio

Risk-adjusted return using excess return over volatility:

```
sharpe = (excess_return_mean / return_std) * sqrt(252)
```

### 3. Sortino Ratio

Risk-adjusted return using downside deviation:

```
sortino = (excess_return_mean / downside_deviation) * sqrt(252)
```

### 4. Trend Method

Linear trend strength using regression slope:

```
trend = slope / initial_price
```

## Relative Strength Analysis

Relative strength measures how an asset performs compared to a benchmark:

```
relative_strength = asset_return - benchmark_return
```

This helps identify assets that are outperforming the broader market.

## Rebalancing Logic

### Rebalancing Frequency

- **Dual Momentum**: Monthly (21 days)
- **Sector Rotation**: Monthly (21 days)
- Configurable via `rebalancing_frequency` parameter

### Rebalancing Triggers

1. **Time-based**: Regular rebalancing at specified intervals
2. **Performance-based**: Exit positions when momentum deteriorates
3. **Ranking changes**: Adjust allocations when rankings change

### Position Sizing

- **Dual Momentum**: 100% allocation to single best asset
- **Sector Rotation**: Equal weight among top N sectors
- **Risk Management**: Stop-loss and take-profit levels

## Risk Management

### Stop-Loss

Default 15% stop-loss to limit downside:

```python
stop_loss = entry_price * (1 - 0.15)
```

### Take-Profit

Default 30% take-profit to lock in gains:

```python
take_profit = entry_price * (1 + 0.30)
```

### Defensive Mode

When no assets meet momentum criteria:

- Exit all positions
- Optionally enter defensive asset (bonds/cash)
- Wait for momentum to improve

## Performance Metrics

### Strategy-Specific Metrics

- **Dual Momentum**:

  - Current asset allocation
  - Defensive mode status
  - Absolute/relative momentum scores
  - Qualified assets count

- **Sector Rotation**:
  - Sector rankings
  - Combined momentum scores
  - Volatility adjustments
  - Top sector allocations

### Common Metrics

- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Average trade duration
- Rebalancing frequency

## Usage Examples

### Basic Dual Momentum Strategy

```python
from strategies.etf.dual_momentum import DualMomentumStrategy
from utils.asset_categorization import get_etf_universe_for_strategy

# Get ETF universe
etf_universe = get_etf_universe_for_strategy("dual_momentum")

# Create strategy
strategy = DualMomentumStrategy(etf_universe=etf_universe)

# Generate signals
signals = strategy.generate_signals(market_data)
```

### Custom Sector Rotation Strategy

```python
from strategies.etf.sector_rotation import SectorRotationStrategy

# Custom configuration
config = {
    "max_positions": 3,  # Hold top 3 sectors
    "momentum_lookback": 126,  # 6 months
    "rebalance_frequency": 14,  # Bi-weekly
}

strategy = SectorRotationStrategy(**config)
```

### Backtesting

```python
from backtesting.engine import BacktestingEngine

# Create backtesting engine
engine = BacktestingEngine(initial_capital=100000)

# Run backtest
result = engine.run_backtest(strategy, market_data)

# Get results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
```

## Testing

### Unit Tests

Run unit tests for ETF rotation strategies:

```bash
pytest tests/unit/test_etf_rotation.py -v
```

### Backtesting Tests

Run backtesting scenarios:

```bash
python backtesting/test_etf_rotation.py
```

### Test Coverage

- Strategy initialization
- Momentum calculations
- Signal generation
- Rebalancing logic
- Risk management
- Performance tracking

## Configuration

### Strategy Parameters

All strategies support extensive configuration:

```python
# Dual Momentum Configuration
dual_momentum_config = {
    "absolute_momentum_lookback": 252,
    "relative_momentum_lookback": 252,
    "risk_free_rate": 0.02,
    "defensive_asset": "SHY",
    "max_positions": 1,
    "rebalance_frequency": 21,
}

# Sector Rotation Configuration
sector_config = {
    "momentum_lookback": 63,
    "relative_strength_lookback": 252,
    "max_positions": 4,
    "benchmark_symbol": "SPY",
    "sector_momentum_weight": 0.6,
    "relative_strength_weight": 0.4,
}
```

### ETF Universe Configuration

ETF universes are configurable via `utils.asset_categorization`:

```python
from utils.asset_categorization import get_etf_universe_for_strategy

# Get predefined universes
dual_momentum_universe = get_etf_universe_for_strategy("dual_momentum")
sector_universe = get_etf_universe_for_strategy("sector_rotation")

# Custom universe
custom_universe = {
    "US_Large_Cap": ["SPY", "VOO", "IVV"],
    "US_Small_Cap": ["IWM", "VTWO", "SCHA"],
    "International": ["EFA", "VEA", "VWO"],
}
```

## Best Practices

### 1. Data Quality

- Ensure sufficient historical data (at least 252 days)
- Handle missing data appropriately
- Validate price data integrity

### 2. Rebalancing

- Consider transaction costs in rebalancing decisions
- Use appropriate rebalancing thresholds
- Monitor rebalancing frequency impact

### 3. Risk Management

- Set appropriate stop-loss levels
- Monitor position concentration
- Consider correlation between assets

### 4. Performance Monitoring

- Track strategy performance regularly
- Monitor momentum regime changes
- Adjust parameters based on market conditions

## Limitations and Considerations

### 1. Momentum Regime Dependence

- Strategies perform best in trending markets
- May underperform in choppy/sideways markets
- Require sufficient momentum persistence

### 2. Transaction Costs

- Frequent rebalancing increases costs
- Consider impact on net returns
- Optimize rebalancing frequency

### 3. Market Timing

- No guarantee of outperformance
- Past performance doesn't predict future results
- Requires disciplined execution

### 4. Data Requirements

- Need reliable historical data
- Real-time data for live trading
- Proper data synchronization

## Future Enhancements

### Planned Features

1. **Multi-Asset Rotation**: Extend to include more asset classes
2. **Machine Learning Integration**: Use ML for momentum prediction
3. **Dynamic Parameter Optimization**: Adaptive parameter selection
4. **Risk Parity Integration**: Risk-weighted allocations
5. **Alternative Data**: Incorporate sentiment and macro indicators

### Research Areas

1. **Momentum Persistence**: Study momentum duration patterns
2. **Regime Detection**: Identify market regime changes
3. **Correlation Analysis**: Dynamic correlation modeling
4. **Volatility Forecasting**: Improve volatility predictions

## Conclusion

ETF rotation strategies provide a systematic approach to tactical asset allocation that can potentially enhance returns while managing risk. The implemented strategies offer flexibility and extensibility while maintaining robust risk management and performance tracking capabilities.

For questions or contributions, please refer to the main project documentation or create an issue in the repository.

## Position-Aware Logic (Best Practice)

Modern professional trading systems are always position-aware. This means:

- Strategies check current broker/account positions before making any recommendation.
- Signal generation logic considers existing holdings and their sizes.
- Instead of always recommending a BUY, the strategy will:
  - Recommend scaling up if the current position is below the target.
  - Recommend scaling down if the current position is above the target.
  - Recommend holding if the current position matches the target.
  - Recommend closing if the asset should no longer be held.
- This approach prevents redundant trades, reduces transaction costs, and ensures portfolio allocations are accurate.

**Implementation Note:**

The ETF rotation strategies in this repository now synchronize with your broker (e.g., Alpaca) and generate position-aware signals. This is standard practice in institutional and professional algorithmic trading systems.
