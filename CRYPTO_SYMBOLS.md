# Crypto Symbols in Alpaca API

## Overview

The Alpaca API provides access to a limited set of cryptocurrency symbols. This document explains which symbols are available and how to handle unavailable symbols.

## Available Crypto Symbols

Alpaca currently supports **63 tradable crypto assets**. The available symbols include:

### Major Cryptocurrencies

- **BTC/USD** - Bitcoin
- **ETH/USD** - Ethereum
- **SOL/USD** - Solana
- **LINK/USD** - Chainlink
- **DOT/USD** - Polkadot
- **AVAX/USD** - Avalanche
- **UNI/USD** - Uniswap
- **LTC/USD** - Litecoin
- **BCH/USD** - Bitcoin Cash
- **XRP/USD** - Ripple

### Other Popular Cryptocurrencies

- **DOGE/USD** - Dogecoin
- **SHIB/USD** - Shiba Inu
- **AAVE/USD** - Aave
- **BAT/USD** - Basic Attention Token
- **CRV/USD** - Curve
- **GRT/USD** - The Graph
- **MKR/USD** - Maker
- **PEPE/USD** - Pepe
- **SUSHI/USD** - SushiSwap
- **XTZ/USD** - Tezos
- **YFI/USD** - Yearn Finance

### Multiple Quote Currencies

Many cryptocurrencies are available with different quote currencies:

- **BTC/USDC**, **BTC/USDT**
- **ETH/USDC**, **ETH/USDT**
- **SOL/USDC**, **SOL/USDT**
- And many more...

## Unavailable Symbols

**Important**: Some popular cryptocurrencies are **NOT available** in Alpaca's API, including:

- **ADA (Cardano)** - Not available
- **MATIC (Polygon)** - Not available
- **ATOM (Cosmos)** - Not available
- **XLM (Stellar)** - Not available
- **ALGO (Algorand)** - Not available
- **VET (VeChain)** - Not available
- **ICP (Internet Computer)** - Not available
- **FIL (Filecoin)** - Not available
- **TRX (TRON)** - Not available
- **ETC (Ethereum Classic)** - Not available
- **XMR (Monero)** - Not available

## Recent Fix

**Update (2025)**: The system has been updated to automatically filter out unavailable symbols before attempting data collection. This prevents the annoying warnings about empty data for symbols like ADAUSD that are not available in Alpaca's API.

The symbol lists in all analysis services have been updated to include only available crypto symbols:
- Removed: ADAUSD, MATICUSD
- Added: AVAXUSD, UNIUSD (which are available in Alpaca)

## How the System Handles Unavailable Symbols

### 1. Symbol Classification

The system automatically classifies symbols as crypto or stock based on the Alpaca Assets API:

```python
# Available crypto symbols are automatically detected
collector.is_crypto_symbol_available('BTCUSD')  # True
collector.is_crypto_symbol_available('ADAUSD')  # False
```

### 2. Error Handling

When an unavailable crypto symbol is requested:

```python
# This will fail gracefully with a clear error message
data = collector.fetch_daily_data('ADAUSD', period='1mo')
# Returns None with detailed error logging
```

### 3. Alternative Suggestions

The system provides suggestions for available alternatives:

```python
suggestions = collector.suggest_alternative_crypto_symbols('ADAUSD')
# Returns: ['BTCUSD', 'ETHUSD', 'SOLUSD', 'LINKUSD', 'DOTUSD']
```

## Best Practices

### 1. Check Symbol Availability

Always verify symbol availability before making requests:

```python
if collector.is_crypto_symbol_available(symbol):
    data = collector.fetch_daily_data(symbol, period='1mo')
else:
    alternatives = collector.suggest_alternative_crypto_symbols(symbol)
    print(f"Symbol {symbol} not available. Try: {alternatives}")
```

### 2. Use Available Symbols

Focus on the 63 available crypto symbols for reliable data collection.

### 3. Handle Errors Gracefully

The system now provides clear error messages when symbols are unavailable, so your application can handle these cases appropriately.

## Getting the Full List

To get the complete list of available crypto symbols:

```python
from data.alpaca_collector import AlpacaDataCollector

collector = AlpacaDataCollector()
available_symbols = collector.get_available_crypto_symbols()
print(f"Available crypto symbols: {available_symbols}")
```

## Conclusion

The Alpaca API provides access to a curated selection of 63 cryptocurrency pairs. While this doesn't include every cryptocurrency, it covers the major ones and provides reliable, high-quality data for algorithmic trading strategies.

For symbols not available in Alpaca, consider:

1. Using alternative data providers
2. Focusing on the available symbols for your strategies
3. Using the suggested alternatives provided by the system
