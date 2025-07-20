# Algorithmic Trading System

This is a personal algorithmic trading system designed for small accounts ($500-$1000). The system follows a modular pipeline architecture to collect market data, implement trading strategies, and execute trades.

## System Architecture

```
algotrading/
├── data/          # Data collection and processing
├── strategies/    # Trading strategy implementations
├── indicators/    # Technical indicator calculations
├── backtesting/   # Backtesting framework
├── risk/          # Risk management rules
├── execution/     # Trade execution components
├── utils/         # Utility functions and configs
└── pipeline.py    # Main orchestration
```

## Features

- Data collection from Yahoo Finance with incremental updates
- PostgreSQL database for storing market data
- Multiple trading strategies including mean reversion, trend following, and more
- Risk management with position sizing and circuit breakers
- Performance tracking and analytics
- Tax-efficient trading approach

## Requirements

- Python 3.8+
- PostgreSQL database
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
# Example .env file
DB_URI=postgresql://username:password@localhost:5432/algotrading
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
python pipeline.py --task collect --symbols AAPL MSFT GOOG

# Collect data for a specific time period
python pipeline.py --task collect --period 1y

# Force update of existing data
python pipeline.py --task collect --force
```

### Backtesting (Coming Soon)

```bash
# Backtest a strategy
python pipeline.py --task backtest --strategy mean_reversion
```

### Trading (Coming Soon)

```bash
# Run trading simulation
python pipeline.py --task trade --paper
```

## Contributing

This is a personal project, but suggestions and improvements are welcome.

## License

MIT License
