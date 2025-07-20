"""
Statistical tests for mean reversion analysis.

This module provides comprehensive statistical testing for mean reversion properties
in financial time series. Includes ADF tests, Hurst exponent calculation, and
various other statistical measures used in academic research.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.stats import norm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def calculate_hurst_exponent(data: pd.Series, max_lag: int = 20) -> Dict[str, Any]:
    """
    Calculate the Hurst exponent using R/S analysis.

    The Hurst exponent H indicates the type of time series:
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Trending (persistent)

    Args:
        data: Time series data
        max_lag: Maximum lag for R/S analysis

    Returns:
        Dictionary with Hurst exponent and analysis details
    """
    try:
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) < 50:
            return {
                "hurst_exponent": 0.5,
                "confidence": 0.0,
                "r_squared": 0.0,
                "error": "Insufficient data for Hurst calculation",
            }

        # Calculate returns
        returns = clean_data.pct_change().dropna()

        # R/S analysis
        lags = range(10, min(max_lag, len(returns) // 4))
        rs_values = []

        for lag in lags:
            # Split data into chunks
            chunks = len(returns) // lag
            if chunks < 2:
                continue

            rs_chunk = []
            for i in range(chunks):
                chunk = returns[i * lag : (i + 1) * lag]
                if len(chunk) < lag:
                    continue

                # Calculate R (range)
                cumulative = chunk.cumsum()
                R = cumulative.max() - cumulative.min()

                # Calculate S (standard deviation)
                S = chunk.std()

                if S > 0:
                    rs_chunk.append(R / S)

            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))

        if len(rs_values) < 3:
            return {
                "hurst_exponent": 0.5,
                "confidence": 0.0,
                "r_squared": 0.0,
                "error": "Insufficient data points for regression",
            }

        # Linear regression of log(R/S) vs log(lag)
        log_lags = np.log([lags[i] for i in range(len(rs_values))])
        log_rs = np.log(rs_values)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)
        hurst_exponent = slope

        return {
            "hurst_exponent": hurst_exponent,
            "confidence": 1 - p_value,
            "r_squared": r_value**2,
            "std_error": std_err,
            "p_value": p_value,
            "interpretation": (
                "mean_reverting"
                if hurst_exponent < 0.5
                else "trending" if hurst_exponent > 0.5 else "random_walk"
            ),
        }

    except Exception as e:
        logger.error(f"Error calculating Hurst exponent: {e}")
        return {
            "hurst_exponent": 0.5,
            "confidence": 0.0,
            "r_squared": 0.0,
            "error": str(e),
        }


def adf_test(data: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Args:
        data: Time series data
        significance_level: Significance level for the test

    Returns:
        Dictionary with ADF test results
    """
    try:
        from statsmodels.tsa.stattools import adfuller

        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) < 50:
            return {
                "adf_statistic": 0.0,
                "p_value": 1.0,
                "critical_values": {},
                "is_stationary": False,
                "error": "Insufficient data for ADF test",
            }

        # Perform ADF test
        adf_result = adfuller(clean_data, autolag="AIC")

        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]

        # Determine if series is stationary
        is_stationary = p_value < significance_level

        return {
            "adf_statistic": adf_statistic,
            "p_value": p_value,
            "critical_values": critical_values,
            "is_stationary": is_stationary,
            "significance_level": significance_level,
        }

    except ImportError:
        logger.warning("statsmodels not available, skipping ADF test")
        return {
            "adf_statistic": 0.0,
            "p_value": 1.0,
            "critical_values": {},
            "is_stationary": False,
            "error": "statsmodels not available",
        }
    except Exception as e:
        logger.error(f"Error in ADF test: {e}")
        return {
            "adf_statistic": 0.0,
            "p_value": 1.0,
            "critical_values": {},
            "is_stationary": False,
            "error": str(e),
        }


def calculate_variance_ratio(
    data: pd.Series, periods: list = [2, 4, 8, 16]
) -> Dict[str, Any]:
    """
    Calculate variance ratio test for mean reversion.

    Variance ratio < 1 indicates mean reversion.

    Args:
        data: Time series data
        periods: List of periods to test

    Returns:
        Dictionary with variance ratio results
    """
    try:
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) < 50:
            return {
                "variance_ratios": {},
                "mean_reversion_score": 0,
                "error": "Insufficient data for variance ratio test",
            }

        # Calculate returns
        returns = clean_data.pct_change().dropna()

        variance_ratios = {}
        mean_reversion_count = 0

        for period in periods:
            if period >= len(returns):
                continue

            # Calculate variance of k-period returns
            k_period_returns = returns.rolling(period).sum().dropna()
            var_k = k_period_returns.var()

            # Calculate variance of 1-period returns
            var_1 = returns.var()

            # Variance ratio
            if var_1 > 0:
                vr = var_k / (period * var_1)
                variance_ratios[period] = vr

                if vr < 1.0:
                    mean_reversion_count += 1

        # Calculate mean reversion score (0-4, higher is better)
        mean_reversion_score = mean_reversion_count

        return {
            "variance_ratios": variance_ratios,
            "mean_reversion_score": mean_reversion_score,
            "max_score": len(periods),
        }

    except Exception as e:
        logger.error(f"Error calculating variance ratio: {e}")
        return {"variance_ratios": {}, "mean_reversion_score": 0, "error": str(e)}


def calculate_autocorrelation(data: pd.Series, max_lag: int = 10) -> Dict[str, Any]:
    """
    Calculate autocorrelation coefficients for mean reversion analysis.

    Negative autocorrelation at lag 1 indicates mean reversion.

    Args:
        data: Time series data
        max_lag: Maximum lag to calculate

    Returns:
        Dictionary with autocorrelation results
    """
    try:
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) < 50:
            return {
                "autocorrelations": {},
                "mean_reversion_score": 0,
                "error": "Insufficient data for autocorrelation analysis",
            }

        # Calculate returns
        returns = clean_data.pct_change().dropna()

        autocorrelations = {}
        mean_reversion_count = 0

        for lag in range(1, min(max_lag + 1, len(returns) // 4)):
            autocorr = returns.autocorr(lag=lag)
            autocorrelations[lag] = autocorr

            # Negative autocorrelation indicates mean reversion
            if autocorr < -0.1:
                mean_reversion_count += 1

        return {
            "autocorrelations": autocorrelations,
            "mean_reversion_score": mean_reversion_count,
            "max_score": max_lag,
            "lag1_autocorr": autocorrelations.get(1, 0),
        }

    except Exception as e:
        logger.error(f"Error calculating autocorrelation: {e}")
        return {"autocorrelations": {}, "mean_reversion_score": 0, "error": str(e)}


def calculate_half_life(data: pd.Series) -> Dict[str, Any]:
    """
    Calculate half-life of mean reversion using linear regression.

    Args:
        data: Time series data

    Returns:
        Dictionary with half-life results
    """
    try:
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) < 50:
            return {
                "half_life": np.nan,
                "confidence": 0.0,
                "error": "Insufficient data for half-life calculation",
            }

        # Calculate price changes and lagged prices
        price_changes = clean_data.diff().dropna()
        lagged_prices = clean_data.shift(1).dropna()

        # Align data
        price_changes = price_changes[1:]
        lagged_prices = lagged_prices[1:]

        if len(price_changes) < 20:
            return {
                "half_life": np.nan,
                "confidence": 0.0,
                "error": "Insufficient aligned data",
            }

        # Linear regression: ΔP_t = α + β * P_{t-1} + ε_t
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            lagged_prices, price_changes
        )

        # Half-life = -log(2) / β
        if slope < 0:
            half_life = -np.log(2) / slope
        else:
            half_life = np.nan

        return {
            "half_life": half_life,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "confidence": 1 - p_value,
            "is_mean_reverting": slope < 0 and p_value < 0.05,
        }

    except Exception as e:
        logger.error(f"Error calculating half-life: {e}")
        return {"half_life": np.nan, "confidence": 0.0, "error": str(e)}


def comprehensive_mean_reversion_test(data: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive mean reversion testing combining multiple statistical measures.

    This function runs all available statistical tests and provides an overall
    assessment of mean reversion properties.

    Args:
        data: Time series data (typically price series)

    Returns:
        Dictionary with comprehensive test results and overall assessment
    """
    try:
        logger.info("Running comprehensive mean reversion tests...")

        # Run all statistical tests
        hurst_results = calculate_hurst_exponent(data)
        adf_results = adf_test(data)
        variance_ratio_results = calculate_variance_ratio(data)
        autocorr_results = calculate_autocorrelation(data)
        half_life_results = calculate_half_life(data)

        # Calculate overall mean reversion score (0-9)
        score = 0

        # Hurst exponent (0-2 points)
        hurst_value = hurst_results.get("hurst_exponent", 0.5)
        if hurst_value < 0.4:
            score += 2
        elif hurst_value < 0.5:
            score += 1

        # ADF test (0-2 points)
        if adf_results.get("is_stationary", False):
            score += 2

        # Variance ratio (0-2 points)
        vr_score = variance_ratio_results.get("mean_reversion_score", 0)
        score += min(vr_score, 2)

        # Autocorrelation (0-2 points)
        ac_score = autocorr_results.get("mean_reversion_score", 0)
        score += min(ac_score, 2)

        # Half-life (0-1 point)
        if half_life_results.get("is_mean_reverting", False):
            score += 1

        # Overall assessment
        overall_assessment = {
            "mean_reversion_score": score,
            "max_possible_score": 9,
            "strength": (
                "strong" if score >= 7 else "moderate" if score >= 4 else "weak"
            ),
            "recommendation": "suitable" if score >= 5 else "not_suitable",
        }

        return {
            "hurst_exponent": hurst_results,
            "adf_test": adf_results,
            "variance_ratio": variance_ratio_results,
            "autocorrelation": autocorr_results,
            "half_life": half_life_results,
            "overall_assessment": overall_assessment,
            "test_summary": {
                "total_tests": 5,
                "passed_tests": sum(
                    [
                        1 if hurst_value < 0.5 else 0,
                        1 if adf_results.get("is_stationary", False) else 0,
                        (
                            1
                            if variance_ratio_results.get("mean_reversion_score", 0) > 0
                            else 0
                        ),
                        1 if autocorr_results.get("mean_reversion_score", 0) > 0 else 0,
                        1 if half_life_results.get("is_mean_reverting", False) else 0,
                    ]
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error in comprehensive mean reversion test: {e}")
        return {
            "error": str(e),
            "overall_assessment": {
                "mean_reversion_score": 0,
                "max_possible_score": 9,
                "strength": "unknown",
                "recommendation": "not_suitable",
            },
        }
