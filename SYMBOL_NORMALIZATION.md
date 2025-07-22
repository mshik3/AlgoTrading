# Symbol Normalization System

## Overview

The Symbol Normalization System handles special characters in stock symbols to ensure compatibility with the Alpaca API. This system automatically converts symbols with hyphens, dots, and other special characters to their Alpaca-compatible equivalents.

## Problem Solved

Alpaca's API has strict symbol validation and rejects symbols with special characters like:

- `BRK-B` (Berkshire Hathaway Class B)
- `BF.B` (Brown-Forman Class B)
- `BRK.A` (Berkshire Hathaway Class A)

These symbols need to be normalized to work with Alpaca's API:

- `BRK-B` → `BRKB`
- `BF.B` → `BFB`
- `BRK.A` → `BRKA`

## How It Works

### 1. Symbol Mapping

The system maintains a mapping of problematic symbols to their Alpaca-compatible equivalents:

```python
STOCK_SYMBOL_MAPPING = {
    "BRK-B": "BRKB",
    "BRK.B": "BRKB",
    "BRK-A": "BRKA",
    "BF.B": "BFB",
    "BF-B": "BFB",
}
```

### 2. Automatic Normalization

For symbols not in the explicit mapping, the system automatically:

- Removes hyphens (`-`)
- Removes dots (`.`)

Example: `TEST-A.B` → `TESTAB`

### 3. Integration Points

The normalization is automatically applied in:

- **Data Collection**: `data/alpaca_collector.py`
- **Trading Execution**: `execution/alpaca.py`
- **Account Services**: `dashboard/services/alpaca_account.py`
- **Portfolio Management**: All trading-related functions

## Usage

### Basic Usage

```python
from utils.symbol_normalization import normalize_symbol_for_alpaca

# Normalize a symbol
normalized = normalize_symbol_for_alpaca("BRK-B")  # Returns "BRKB"
```

### Check if Normalization is Needed

```python
from utils.symbol_normalization import is_symbol_normalized

# Check if symbol needs normalization
needs_norm = is_symbol_normalized("BRK-B")  # Returns True
```

### Get Mapping Information

```python
from utils.symbol_normalization import get_symbol_mapping_info

# Get complete mapping info
info = get_symbol_mapping_info("BRK-B")
# Returns: {
#     "original": "BRK-B",
#     "normalized": "BRKB",
#     "display": "BRK-B",
#     "needs_normalization": True,
#     "is_mapped": True
# }
```

### Add New Mappings

```python
from utils.symbol_normalization import add_symbol_mapping

# Add a new symbol mapping
add_symbol_mapping("NEW-SYMBOL", "NEWSYMBOL")
```

## Implementation Details

### Files Modified

1. **`utils/symbol_normalization.py`** - New utility module
2. **`data/alpaca_collector.py`** - Updated `_get_alpaca_symbol()` method
3. **`execution/alpaca.py`** - Updated `_get_alpaca_symbol()` method
4. **`dashboard/services/alpaca_account.py`** - Updated position fetching
5. **`utils/validators.py`** - Updated symbol validation regex

### Key Functions

- `normalize_symbol_for_alpaca(symbol)` - Convert to Alpaca format
- `denormalize_symbol_for_display(symbol)` - Convert back to display format
- `is_symbol_normalized(symbol)` - Check if normalization needed
- `get_symbol_mapping_info(symbol)` - Get detailed mapping info
- `add_symbol_mapping(original, normalized)` - Add new mappings

## Testing

The system includes comprehensive tests in `tests/unit/test_symbol_normalization.py`:

```bash
python -m pytest tests/unit/test_symbol_normalization.py -v
```

## Benefits

1. **Seamless Integration**: Works automatically without code changes
2. **Backward Compatibility**: Existing code continues to work
3. **Extensible**: Easy to add new symbol mappings
4. **Transparent**: Logs normalization for debugging
5. **Robust**: Handles edge cases and error conditions

## Future Enhancements

- Dynamic symbol mapping from Alpaca Assets API
- Support for more special characters
- Symbol validation against actual Alpaca availability
- Caching of symbol mappings for performance

## Troubleshooting

### Common Issues

1. **Symbol still not working**: Check if the symbol is actually available in Alpaca's API
2. **Wrong normalization**: Verify the mapping in `STOCK_SYMBOL_MAPPING`
3. **Performance issues**: Consider caching for high-frequency operations

### Debugging

Enable debug logging to see symbol normalization in action:

```python
import logging
logging.getLogger('utils.symbol_normalization').setLevel(logging.DEBUG)
```

## Migration Notes

This change is **backward compatible**. Existing code using symbols like `BRK-B` will continue to work, but now they'll be automatically normalized for Alpaca API calls.

No code changes are required for existing implementations.
