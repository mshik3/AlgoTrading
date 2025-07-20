"""
Ornstein-Uhlenbeck (O-U) Process for Mean Reversion Analysis.

This module implements the Ornstein-Uhlenbeck process fitting and analysis
for mean reversion trading strategies. The O-U process is a continuous-time
stochastic process that models mean-reverting behavior.

The process is defined by:
dX_t = θ(μ - X_t)dt + σdW_t

Where:
- θ: Speed of mean reversion
- μ: Long-term mean
- σ: Volatility
- W_t: Wiener process
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class OUProcess:
    """
    Ornstein-Uhlenbeck process implementation for mean reversion analysis.

    This class provides methods to fit O-U parameters to price data and
    calculate optimal trading thresholds based on academic research.
    """

    def __init__(self):
        """Initialize the O-U process."""
        self.theta = None  # Speed of mean reversion
        self.mu = None  # Long-term mean
        self.sigma = None  # Volatility
        self.fitted = False
        self.fitting_error = None

    def fit(self, data: pd.Series) -> bool:
        """
        Fit O-U process parameters to the given data.

        Args:
            data: Price series data

        Returns:
            True if fitting was successful, False otherwise
        """
        try:
            # Remove NaN values
            clean_data = data.dropna()
            if len(clean_data) < 50:
                logger.warning("Insufficient data for O-U process fitting")
                self.fitted = False
                return False

            # Calculate log returns
            log_returns = np.log(clean_data / clean_data.shift(1)).dropna()

            if len(log_returns) < 20:
                logger.warning("Insufficient log returns for O-U fitting")
                self.fitted = False
                return False

            # Fit O-U parameters using maximum likelihood estimation
            success = self._fit_mle(log_returns)

            if success:
                logger.info(
                    f"O-U process fitted successfully: θ={self.theta:.4f}, μ={self.mu:.4f}, σ={self.sigma:.4f}"
                )
                self.fitted = True
            else:
                logger.warning("O-U process fitting failed")
                self.fitted = False

            return self.fitted

        except Exception as e:
            logger.error(f"Error fitting O-U process: {e}")
            self.fitted = False
            self.fitting_error = str(e)
            return False

    def _fit_mle(self, returns: pd.Series) -> bool:
        """
        Fit O-U parameters using Maximum Likelihood Estimation.

        Args:
            returns: Log returns series

        Returns:
            True if fitting was successful
        """
        try:
            # Initial parameter estimates
            mu_0 = returns.mean()
            sigma_0 = returns.std()
            theta_0 = 0.1  # Initial guess for mean reversion speed

            # Define negative log-likelihood function
            def neg_log_likelihood(params):
                theta, mu, sigma = params

                if theta <= 0 or sigma <= 0:
                    return np.inf

                # Calculate log-likelihood
                dt = 1  # Assuming daily data
                n = len(returns)

                # Expected value at t+1 given t
                expected_next = returns[:-1] * np.exp(-theta * dt) + mu * (
                    1 - np.exp(-theta * dt)
                )

                # Variance of the process
                var = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

                if var <= 0:
                    return np.inf

                # Log-likelihood
                residuals = returns[1:] - expected_next
                log_likelihood = (
                    -0.5 * n * np.log(2 * np.pi * var)
                    - 0.5 * np.sum(residuals**2) / var
                )

                return -log_likelihood

            # Optimize parameters
            initial_params = [theta_0, mu_0, sigma_0]
            bounds = [(0.001, 10), (-1, 1), (0.001, 1)]  # Reasonable bounds

            result = minimize(
                neg_log_likelihood,
                initial_params,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 1000},
            )

            if result.success:
                self.theta, self.mu, self.sigma = result.x
                return True
            else:
                logger.warning(f"O-U fitting optimization failed: {result.message}")
                return False

        except Exception as e:
            logger.error(f"Error in MLE fitting: {e}")
            return False

    def get_half_life(self) -> float:
        """
        Calculate the half-life of mean reversion.

        Returns:
            Half-life in time units (days for daily data)
        """
        if not self.fitted or self.theta is None or self.theta <= 0:
            return np.nan

        return np.log(2) / self.theta

    def calculate_optimal_thresholds(
        self, transaction_cost: float = 0.002, discount_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate optimal entry and exit thresholds based on O-U process.

        This implements the optimal trading strategy for O-U processes
        as described in academic literature.

        Args:
            transaction_cost: Transaction cost as a fraction of price
            discount_rate: Discount rate for future cash flows

        Returns:
            Dictionary with optimal thresholds
        """
        if not self.fitted:
            return {
                "entry_threshold": np.nan,
                "exit_threshold": np.nan,
                "take_profit_threshold": np.nan,
                "optimal": False,
            }

        try:
            # Calculate optimal thresholds using analytical solution
            # Based on the work of Bertram (2010) and others

            # Optimal entry threshold (when to buy)
            entry_threshold = self.mu - np.sqrt(
                (2 * self.sigma**2 * transaction_cost) / (self.theta * discount_rate)
            )

            # Optimal exit threshold (when to sell)
            exit_threshold = self.mu + np.sqrt(
                (2 * self.sigma**2 * transaction_cost) / (self.theta * discount_rate)
            )

            # Take profit threshold (conservative)
            take_profit_threshold = self.mu + 2 * np.sqrt(
                (2 * self.sigma**2 * transaction_cost) / (self.theta * discount_rate)
            )

            return {
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "take_profit_threshold": take_profit_threshold,
                "optimal": True,
                "transaction_cost": transaction_cost,
                "discount_rate": discount_rate,
            }

        except Exception as e:
            logger.error(f"Error calculating optimal thresholds: {e}")
            return {
                "entry_threshold": np.nan,
                "exit_threshold": np.nan,
                "take_profit_threshold": np.nan,
                "optimal": False,
                "error": str(e),
            }

    def expected_return_per_unit_time(
        self, entry_threshold: float, exit_threshold: float
    ) -> Dict[str, float]:
        """
        Calculate expected return per unit time for given thresholds.

        Args:
            entry_threshold: Entry threshold
            exit_threshold: Exit threshold

        Returns:
            Dictionary with expected return metrics
        """
        if not self.fitted:
            return {
                "expected_return": np.nan,
                "annualized_return": np.nan,
                "sharpe_ratio": np.nan,
            }

        try:
            # Calculate expected return using O-U process properties
            # This is an approximation based on the process parameters

            # Expected profit per trade
            expected_profit = exit_threshold - entry_threshold

            # Expected time to exit (approximation)
            expected_time = self.get_half_life()

            # Expected return per unit time
            if expected_time > 0:
                expected_return = expected_profit / expected_time
            else:
                expected_return = 0

            # Annualized return (assuming daily data)
            annualized_return = expected_return * 252

            # Sharpe ratio approximation
            volatility = self.sigma / np.sqrt(2 * self.theta)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            return {
                "expected_return": expected_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "expected_profit_per_trade": expected_profit,
                "expected_time_to_exit": expected_time,
            }

        except Exception as e:
            logger.error(f"Error calculating expected return: {e}")
            return {
                "expected_return": np.nan,
                "annualized_return": np.nan,
                "sharpe_ratio": np.nan,
                "error": str(e),
            }

    def simulate_path(self, initial_value: float, n_steps: int = 100) -> np.ndarray:
        """
        Simulate an O-U process path.

        Args:
            initial_value: Starting value
            n_steps: Number of simulation steps

        Returns:
            Simulated path
        """
        if not self.fitted:
            return np.array([initial_value] * n_steps)

        try:
            dt = 1  # Daily time step
            path = np.zeros(n_steps)
            path[0] = initial_value

            for i in range(1, n_steps):
                # O-U process discretization
                drift = self.theta * (self.mu - path[i - 1]) * dt
                diffusion = self.sigma * np.sqrt(dt) * np.random.normal(0, 1)
                path[i] = path[i - 1] + drift + diffusion

            return path

        except Exception as e:
            logger.error(f"Error simulating O-U path: {e}")
            return np.array([initial_value] * n_steps)

    def get_parameters(self) -> Dict[str, float]:
        """
        Get fitted O-U process parameters.

        Returns:
            Dictionary with process parameters
        """
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "fitted": self.fitted,
            "half_life": self.get_half_life(),
            "fitting_error": self.fitting_error,
        }


def fit_ou_process_to_data(data: pd.Series) -> OUProcess:
    """
    Fit O-U process to price data.

    Args:
        data: Price series data

    Returns:
        Fitted OUProcess object
    """
    ou_process = OUProcess()
    ou_process.fit(data)
    return ou_process


def calculate_ou_optimal_strategy(
    data: pd.Series, transaction_cost: float = 0.002, discount_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate optimal trading strategy using O-U process.

    Args:
        data: Price series data
        transaction_cost: Transaction cost as fraction of price
        discount_rate: Discount rate for future cash flows

    Returns:
        Dictionary with optimal strategy parameters
    """
    try:
        # Fit O-U process
        ou_process = fit_ou_process_to_data(data)

        if not ou_process.fitted:
            return {
                "success": False,
                "error": "O-U process fitting failed",
                "recommendation": "use_traditional_methods",
            }

        # Calculate optimal thresholds
        thresholds = ou_process.calculate_optimal_thresholds(
            transaction_cost=transaction_cost, discount_rate=discount_rate
        )

        # Calculate expected returns
        expected_returns = ou_process.expected_return_per_unit_time(
            thresholds["entry_threshold"], thresholds["exit_threshold"]
        )

        # Get process parameters
        parameters = ou_process.get_parameters()

        return {
            "success": True,
            "ou_process": ou_process,
            "thresholds": thresholds,
            "expected_returns": expected_returns,
            "parameters": parameters,
            "recommendation": (
                "use_ou_strategy"
                if thresholds["optimal"]
                else "use_traditional_methods"
            ),
        }

    except Exception as e:
        logger.error(f"Error calculating O-U optimal strategy: {e}")
        return {
            "success": False,
            "error": str(e),
            "recommendation": "use_traditional_methods",
        }
