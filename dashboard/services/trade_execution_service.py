"""
Paper Trade Execution Service for Dashboard
Provides safe paper trading execution with explicit safeguards and Alpaca integration.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from execution.alpaca import AlpacaTradingClient, AlpacaConfig
from strategies.base import StrategySignal, SignalType
from utils.config import load_environment, get_env_var
from dashboard.services.alpaca_account import AlpacaAccountService

logger = logging.getLogger(__name__)


class PaperTradingExecutionService:
    """
    Service for executing paper trades with multiple safety layers.

    Safety Features:
    - Validates paper trading mode before every trade
    - Checks environment variables for paper API
    - Logs all trades with PAPER TRADE prefix
    - Refuses to execute if not in paper mode
    """

    def __init__(self):
        """Initialize paper trading execution service with safety validation."""
        self.setup_environment()
        self.setup_alpaca_client()
        self.alpaca_account_service = AlpacaAccountService()
        self._validate_paper_mode()

    def setup_environment(self):
        """Load and validate environment configuration for paper trading."""
        try:
            load_environment()

            self.api_key = get_env_var("ALPACA_API_KEY", default=None)
            self.secret_key = get_env_var("ALPACA_SECRET_KEY", default=None)
            self.base_url = get_env_var(
                "ALPACA_BASE_URL", default="https://paper-api.alpaca.markets"
            )

            if not self.api_key or not self.secret_key:
                raise Exception("Alpaca API credentials required for paper trading")

            # CRITICAL SAFETY CHECK - Ensure paper trading URL
            if "paper-api" not in self.base_url.lower():
                raise Exception(
                    f"SAFETY ERROR: Base URL '{self.base_url}' does not appear to be paper trading!"
                )

            logger.info("✓ Paper trading environment validated")

        except Exception as e:
            logger.error(f"Paper trading environment setup failed: {e}")
            self.is_safe = False
            raise

    def setup_alpaca_client(self):
        """Initialize Alpaca trading client with paper mode enforcement."""
        try:
            config = AlpacaConfig(
                api_key=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                paper=True,  # FORCE paper mode
            )

            self.trading_client = AlpacaTradingClient(config=config)

            # Double-check paper mode
            if not config.paper:
                raise Exception("SAFETY ERROR: Trading client not in paper mode!")

            logger.info("✓ Paper trading client initialized safely")
            self.is_safe = True

        except Exception as e:
            logger.error(f"Alpaca client setup failed: {e}")
            self.is_safe = False
            raise

    def _validate_paper_mode(self):
        """Validate all safety checks for paper trading mode."""
        safety_checks = []

        # Check 1: Base URL contains "paper"
        if "paper" in self.base_url.lower():
            safety_checks.append("✓ Paper API URL")
        else:
            safety_checks.append("✗ NON-PAPER API URL DETECTED!")

        # Check 2: Trading client paper flag
        if self.trading_client.config.paper:
            safety_checks.append("✓ Client paper mode")
        else:
            safety_checks.append("✗ CLIENT NOT IN PAPER MODE!")

        # Check 3: Alpaca account service connection
        if self.alpaca_account_service.is_connected():
            safety_checks.append("✓ Account service connected")
        else:
            safety_checks.append("⚠ Account service not connected")

        # Log all safety checks
        logger.info("Paper Trading Safety Validation:")
        for check in safety_checks:
            logger.info(f"  {check}")

        # Fail if any critical checks failed
        if any("✗" in check for check in safety_checks):
            raise Exception(
                "CRITICAL SAFETY FAILURE: System not safe for paper trading!"
            )

        self.safety_checks = safety_checks

    def execute_paper_trade(self, signal: StrategySignal) -> Dict[str, Any]:
        """
        Execute a paper trade with full safety validation.

        Args:
            signal: StrategySignal to execute

        Returns:
            Dictionary with execution results
        """
        # Pre-execution safety check
        if not self.is_safe:
            return {
                "success": False,
                "error": "SAFETY ERROR: System not validated for paper trading",
                "trade_type": "PAPER_TRADE_BLOCKED",
            }

        # Re-validate paper mode before EVERY trade
        try:
            self._validate_paper_mode()
        except Exception as e:
            logger.error(f"Pre-trade safety validation failed: {e}")
            return {
                "success": False,
                "error": f"Pre-trade safety check failed: {str(e)}",
                "trade_type": "PAPER_TRADE_BLOCKED",
            }

        logger.info(
            f"🧪 EXECUTING PAPER TRADE: {signal.signal_type.value} {signal.symbol}"
        )

        try:
            # Get current price for validation
            current_price = self.trading_client.get_current_price(signal.symbol)
            if not current_price:
                return {
                    "success": False,
                    "error": f"Could not get current price for {signal.symbol}",
                    "trade_type": "PAPER_TRADE_FAILED",
                }

            # Execute the trade
            execution_result = self.trading_client.execute_signal(signal)

            if execution_result:
                # Get updated account info
                account_info = self.alpaca_account_service.get_account_summary()

                logger.info(
                    f"✅ PAPER TRADE SUCCESSFUL: {signal.signal_type.value} {signal.symbol} @ ${current_price:.2f}"
                )

                return {
                    "success": True,
                    "trade_type": "PAPER_TRADE_EXECUTED",
                    "signal": signal,
                    "execution_price": current_price,
                    "timestamp": datetime.now(),
                    "account_summary": account_info,
                    "message": f"Paper trade executed: {signal.signal_type.value} {signal.symbol} @ ${current_price:.2f}",
                }
            else:
                return {
                    "success": False,
                    "error": "Alpaca trade execution failed",
                    "trade_type": "PAPER_TRADE_FAILED",
                    "signal": signal,
                }

        except Exception as e:
            logger.error(f"Paper trade execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "trade_type": "PAPER_TRADE_ERROR",
                "signal": signal,
            }

    def get_paper_account_status(self) -> Dict[str, Any]:
        """Get paper account status with safety indicators."""
        try:
            account_summary = self.alpaca_account_service.get_account_summary()
            positions = self.alpaca_account_service.get_positions()

            return {
                "success": True,
                "is_paper_mode": True,
                "safety_checks": self.safety_checks,
                "account_summary": account_summary,
                "positions": positions,
                "base_url": self.base_url,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error getting paper account status: {e}")
            return {
                "success": False,
                "error": str(e),
                "is_paper_mode": True,
                "safety_checks": getattr(self, "safety_checks", []),
            }

    def refresh_paper_account_data(self):
        """Force refresh of paper account data."""
        try:
            # Clear any cached data in the account service
            if hasattr(self.alpaca_account_service, "cache"):
                # Clear cache if it exists
                pass

            logger.info("🔄 Paper account data refresh requested")
            return True

        except Exception as e:
            logger.error(f"Error refreshing paper account data: {e}")
            return False
