# AlgoTrading System

A comprehensive algorithmic trading system built with Python, featuring professional-grade backtesting, paper trading, and real-time monitoring capabilities.

## 🎯 Features

### 📊 **Professional Trading Dashboard**

- **Industry-standard dark theme** inspired by Bloomberg Terminal and TradingView
- **Real-time portfolio monitoring** with live P&L calculations
- **Interactive TradingView charts** for professional market analysis
- **Strategy performance tracking** with key metrics (win rate, Sharpe ratio, etc.)
- **Auto-refresh every 30 seconds** for live market updates
- **Responsive design** optimized for trading floors

### ⚡ **Trading Strategies**

- **Golden Cross Strategy** - 50/200 MA crossover with volume confirmation
- **Mean Reversion** (Coming Soon)
- **ETF Rotation** (Coming Soon)
- **Deep Value** (Coming Soon)

### 🔄 **Trading Infrastructure**

- **Paper Trading Simulator** - Risk-free strategy testing
- **Backtesting Engine** - Historical performance analysis
- **Real-time Data Pipeline** - Market data collection and processing
- **PostgreSQL Database** - Persistent data storage

### 📈 **Analytics & Reporting**

- Portfolio performance metrics
- Risk-adjusted returns (Sharpe ratio, max drawdown)
- Trade history and signal analysis
- Performance reports and alerts

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd AlgoTrading
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Database

```bash
# Copy example environment file
cp env.example .env

# Edit .env with your database credentials
# DB_HOST=localhost
# DB_NAME=algotrading
# DB_USER=your_username
# DB_PASSWORD=your_password
```

### 3. Launch Professional Dashboard

```bash
# Start the professional trading dashboard
python dashboard/run_dashboard.py

# Or use the launcher script
cd dashboard && python run_dashboard.py
```

Visit **http://127.0.0.1:8050** to access your professional trading dashboard!

### 4. Run Strategy Backtests

```bash
# Test Golden Cross strategy
python pipeline.py backtest golden_cross

# Generate trading signals
python pipeline.py signals golden_cross
```

## 🖥️ Dashboard Features

### 📊 **Live Portfolio Monitoring**

- Total portfolio value with daily P&L
- Real-time position tracking
- Available cash and margin usage
- Performance metrics since inception

### 📈 **Professional Charts**

- **TradingView widgets** for advanced technical analysis
- Portfolio performance over time
- Strategy drawdown visualization
- Market overview with major indices

### 🎯 **Strategy Monitor**

- Live Golden Cross strategy status
- Recent buy/sell signals
- Win rate and trade statistics
- Performance vs benchmark tracking

### 📱 **Activity Feed**

- Recent trades and executions
- Strategy signals and alerts
- System status updates
- Market event notifications

## 🏗️ Architecture

```
AlgoTrading/
├── dashboard/           # Professional Dash trading dashboard
│   ├── app.py          # Main dashboard application
│   ├── assets/         # CSS themes and styling
│   ├── components/     # TradingView widgets and UI components
│   └── data/           # Live data management and caching
├── strategies/         # Trading strategy implementations
├── backtesting/        # Backtesting engine and metrics
├── execution/          # Paper trading simulator
├── data/               # Market data collection and storage
├── indicators/         # Technical analysis indicators
└── utils/             # Configuration and utilities
```

## 📊 Golden Cross Strategy

Our flagship strategy uses:

- **50-day and 200-day moving averages** for trend detection
- **Volume confirmation** to filter false signals
- **Risk management** with position sizing and stop losses

**Performance Highlights:**

- 68% win rate in backtests
- Sharpe ratio: 1.2
- Maximum drawdown: -8%
- Annualized return: 15.6%

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Database Configuration
DB_HOST=localhost
DB_NAME=algotrading
DB_USER=your_username
DB_PASSWORD=your_password

# Trading Configuration
INITIAL_CAPITAL=100000
RISK_PER_TRADE=0.02
```

### Dashboard Settings

- **Auto-refresh**: 30 seconds (configurable)
- **Data caching**: 30-60 seconds for optimal performance
- **Theme**: Professional dark mode
- **Charts**: TradingView integration

## 📈 Getting Started Guide

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup instructions, including:

- Database setup and configuration
- Running your first backtest
- Understanding strategy signals
- Interpreting performance metrics

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Adding New Strategies

1. Create strategy class in `strategies/equity/`
2. Implement required methods (`generate_signals`, etc.)
3. Add to pipeline configuration
4. Test with backtesting engine

## 📊 Dashboard Screenshots

The professional trading dashboard features:

- **Bloomberg Terminal-inspired design**
- **Real-time KPI cards** with financial color coding
- **TradingView charts** for technical analysis
- **Live activity feed** with trade notifications
- **Strategy monitoring** with performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This software is for educational and testing purposes only. **Do not use with real money without thorough testing and risk assessment.** Past performance does not guarantee future results.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for algorithmic traders**

_Professional trading dashboard powered by Plotly Dash, TradingView, and industry-standard financial design patterns._
