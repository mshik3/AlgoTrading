"""
Momentum Ranking Utilities.

Provides reusable momentum calculation and ranking utilities for ETF selection
and rotation strategies. These utilities can be used across different strategy types.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def calculate_momentum_score(
    data: pd.DataFrame,
    lookback: int = 252,
    method: str = "returns",
    risk_free_rate: float = 0.02,
) -> float:
    """
    Calculate momentum score for a given dataset.

    Args:
        data: OHLCV data
        lookback: Lookback period in days
        method: Momentum calculation method ('returns', 'sharpe', 'sortino', 'trend')
        risk_free_rate: Annual risk-free rate for risk-adjusted calculations

    Returns:
        Momentum score
    """
    if len(data) < lookback:
        return np.nan

    try:
        if method == "returns":
            # Simple cumulative return
            return data["Close"].iloc[-1] / data["Close"].iloc[-lookback] - 1

        elif method == "sharpe":
            # Sharpe ratio (excess return / volatility)
            returns = data["Close"].pct_change().dropna()
            if len(returns) < 2:
                return np.nan
            excess_returns = returns - risk_free_rate / 252
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        elif method == "sortino":
            # Sortino ratio (excess return / downside deviation)
            returns = data["Close"].pct_change().dropna()
            if len(returns) < 2:
                return np.nan
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return np.nan
            downside_deviation = downside_returns.std()
            return excess_returns.mean() / downside_deviation * np.sqrt(252)

        elif method == "trend":
            # Linear trend strength
            prices = data["Close"].tail(lookback)
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            return slope / prices.iloc[0]  # Normalized slope

        else:
            raise ValueError(f"Unknown momentum method: {method}")

    except Exception as e:
        logger.warning(f"Error calculating momentum score: {str(e)}")
        return np.nan


def calculate_relative_strength(
    symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame, lookback: int = 252
) -> float:
    """
    Calculate relative strength vs benchmark.

    Args:
        symbol_data: Symbol OHLCV data
        benchmark_data: Benchmark OHLCV data
        lookback: Lookback period

    Returns:
        Relative strength score
    """
    if len(symbol_data) < lookback or len(benchmark_data) < lookback:
        return np.nan

    try:
        # Calculate cumulative returns
        symbol_return = (
            symbol_data["Close"].iloc[-1] / symbol_data["Close"].iloc[-lookback] - 1
        )
        benchmark_return = (
            benchmark_data["Close"].iloc[-1] / benchmark_data["Close"].iloc[-lookback]
            - 1
        )

        # Relative strength = symbol return - benchmark return
        return symbol_return - benchmark_return

    except Exception as e:
        logger.warning(f"Error calculating relative strength: {str(e)}")
        return np.nan


def rank_assets_by_momentum(
    market_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    lookback: int = 252,
    method: str = "returns",
    min_threshold: float = -np.inf,
) -> List[Tuple[str, float]]:
    """
    Rank assets by momentum score.

    Args:
        market_data: Dictionary of symbol -> OHLCV data
        symbols: List of symbols to rank
        lookback: Lookback period for momentum calculation
        method: Momentum calculation method
        min_threshold: Minimum momentum threshold

    Returns:
        List of (symbol, momentum_score) tuples, sorted by momentum
    """
    rankings = []

    for symbol in symbols:
        if symbol not in market_data or market_data[symbol].empty:
            continue

        momentum = calculate_momentum_score(
            market_data[symbol], lookback=lookback, method=method
        )

        if not np.isnan(momentum) and momentum >= min_threshold:
            rankings.append((symbol, momentum))

    # Sort by momentum (highest first)
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def rank_assets_by_relative_strength(
    market_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    benchmark_symbol: str,
    lookback: int = 252,
    min_threshold: float = -np.inf,
) -> List[Tuple[str, float]]:
    """
    Rank assets by relative strength vs benchmark.

    Args:
        market_data: Dictionary of symbol -> OHLCV data
        symbols: List of symbols to rank
        benchmark_symbol: Benchmark symbol
        lookback: Lookback period
        min_threshold: Minimum relative strength threshold

    Returns:
        List of (symbol, relative_strength) tuples, sorted by relative strength
    """
    rankings = []

    if benchmark_symbol not in market_data:
        logger.warning(f"Benchmark {benchmark_symbol} not found in market data")
        return rankings

    benchmark_data = market_data[benchmark_symbol]

    for symbol in symbols:
        if symbol not in market_data or market_data[symbol].empty:
            continue

        relative_strength = calculate_relative_strength(
            market_data[symbol], benchmark_data, lookback=lookback
        )

        if not np.isnan(relative_strength) and relative_strength >= min_threshold:
            rankings.append((symbol, relative_strength))

    # Sort by relative strength (highest first)
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def calculate_combined_score(
    momentum_rankings: List[Tuple[str, float]],
    relative_strength_rankings: List[Tuple[str, float]],
    momentum_weight: float = 0.6,
    rs_weight: float = 0.4,
) -> List[Tuple[str, float]]:
    """
    Calculate combined score from momentum and relative strength rankings.

    Args:
        momentum_rankings: List of (symbol, momentum_score) tuples
        relative_strength_rankings: List of (symbol, relative_strength) tuples
        momentum_weight: Weight for momentum score
        rs_weight: Weight for relative strength score

    Returns:
        List of (symbol, combined_score) tuples, sorted by combined score
    """
    # Create dictionaries for easy lookup
    momentum_dict = dict(momentum_rankings)
    rs_dict = dict(relative_strength_rankings)

    # Get all unique symbols
    all_symbols = set(momentum_dict.keys()) | set(rs_dict.keys())

    combined_scores = []

    for symbol in all_symbols:
        momentum_score = momentum_dict.get(symbol, 0.0)
        rs_score = rs_dict.get(symbol, 0.0)

        # Calculate combined score
        combined_score = momentum_score * momentum_weight + rs_score * rs_weight
        combined_scores.append((symbol, combined_score))

    # Sort by combined score (highest first)
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    return combined_scores


def calculate_volatility_adjustment(
    market_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    benchmark_symbol: str,
    lookback: int = 252,
) -> Dict[str, float]:
    """
    Calculate volatility adjustment factors for risk-adjusted momentum.

    Args:
        market_data: Dictionary of symbol -> OHLCV data
        symbols: List of symbols to calculate for
        benchmark_symbol: Benchmark symbol
        lookback: Lookback period

    Returns:
        Dictionary mapping symbol -> volatility adjustment factor
    """
    adjustments = {}

    if benchmark_symbol not in market_data:
        logger.warning(f"Benchmark {benchmark_symbol} not found in market data")
        return {symbol: 1.0 for symbol in symbols}

    benchmark_vol = market_data[benchmark_symbol]["Close"].pct_change().std() * np.sqrt(
        252
    )

    for symbol in symbols:
        if symbol not in market_data or market_data[symbol].empty:
            adjustments[symbol] = 1.0
            continue

        try:
            symbol_vol = market_data[symbol]["Close"].pct_change().std() * np.sqrt(252)
            vol_ratio = benchmark_vol / symbol_vol if symbol_vol > 0 else 1.0
            adjustments[symbol] = vol_ratio
        except Exception as e:
            logger.warning(
                f"Error calculating volatility adjustment for {symbol}: {str(e)}"
            )
            adjustments[symbol] = 1.0

    return adjustments


def get_top_ranked_assets(
    rankings: List[Tuple[str, float]], max_positions: int, min_score: float = -np.inf
) -> List[str]:
    """
    Get top-ranked assets from rankings list.

    Args:
        rankings: List of (symbol, score) tuples, sorted by score
        max_positions: Maximum number of positions to select
        min_score: Minimum score threshold

    Returns:
        List of top-ranked symbols
    """
    top_assets = []

    for symbol, score in rankings:
        if len(top_assets) >= max_positions:
            break

        if score >= min_score:
            top_assets.append(symbol)

    return top_assets


def calculate_momentum_percentile(
    symbol_score: float, all_scores: List[float]
) -> float:
    """
    Calculate momentum percentile rank.

    Args:
        symbol_score: Score for the symbol
        all_scores: List of all scores

    Returns:
        Percentile rank (0-100)
    """
    if not all_scores:
        return 50.0

    valid_scores = [s for s in all_scores if not np.isnan(s)]
    if not valid_scores:
        return 50.0

    percentile = (
        sum(1 for s in valid_scores if s < symbol_score) / len(valid_scores)
    ) * 100
    return percentile


def calculate_momentum_consistency(
    data: pd.DataFrame, lookback: int = 252, periods: int = 4
) -> float:
    """
    Calculate momentum consistency over multiple periods.

    Args:
        data: OHLCV data
        lookback: Total lookback period
        periods: Number of sub-periods to check

    Returns:
        Consistency score (0-1, higher is more consistent)
    """
    if len(data) < lookback:
        return 0.0

    try:
        period_length = lookback // periods
        consistency_scores = []

        for i in range(periods):
            start_idx = -(lookback - i * period_length)
            end_idx = -(lookback - (i + 1) * period_length) if i < periods - 1 else None

            period_data = data.iloc[start_idx:end_idx]
            if len(period_data) < 2:
                continue

            period_return = (
                period_data["Close"].iloc[-1] / period_data["Close"].iloc[0] - 1
            )
            consistency_scores.append(1.0 if period_return > 0 else 0.0)

        if not consistency_scores:
            return 0.0

        return sum(consistency_scores) / len(consistency_scores)

    except Exception as e:
        logger.warning(f"Error calculating momentum consistency: {str(e)}")
        return 0.0
