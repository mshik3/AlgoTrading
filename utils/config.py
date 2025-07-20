"""
Configuration management module for the algorithmic trading system.
Handles environment variables, security, and configuration validation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import quote_plus

# Set up logging
logger = logging.getLogger(__name__)

# Track if environment has been loaded
_environment_loaded = False


def load_environment(env_file: str = ".env") -> bool:
    """
    Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file

    Returns:
        True if environment was loaded successfully
    """
    global _environment_loaded

    env_path = Path(env_file)
    if not env_path.exists():
        logger.warning(
            f"Environment file {env_file} not found, using system environment variables only"
        )
        _environment_loaded = True
        return False

    try:
        with open(env_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse key=value pairs
                if "=" not in line:
                    logger.warning(f"Invalid line {line_num} in {env_file}: {line}")
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Only set if not already in environment (system env takes precedence)
                if key not in os.environ:
                    os.environ[key] = value

        _environment_loaded = True
        logger.info(f"Environment variables loaded from {env_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to load environment file {env_file}: {e}")
        _environment_loaded = True  # Continue with system env vars
        return False


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get an environment variable with optional default and validation.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether this variable is required

    Returns:
        Environment variable value

    Raises:
        ValueError: If required variable is missing
    """
    # Ensure environment is loaded
    if not _environment_loaded:
        load_environment()

    value = os.environ.get(key, default)

    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")

    return value


def get_database_url() -> str:
    """
    Construct database URL from environment variables with proper escaping.

    Returns:
        Database connection URL (PostgreSQL or SQLite)

    Raises:
        ValueError: If required database variables are missing
    """
    # First try to get the full DB_URI if it exists
    db_uri = get_env_var("DB_URI")
    if db_uri:
        logger.debug("Using full DB_URI connection string")
        return db_uri

    # Check if we're using SQLite
    db_type = get_env_var("DB_TYPE", "postgresql").lower()
    if db_type == "sqlite":
        db_path = get_env_var("DB_PATH", "/path/to/database.db")
        connection_url = f"sqlite:///{db_path}"
        logger.debug(f"Constructed SQLite database URL: {db_path}")
        return connection_url

    # Otherwise construct PostgreSQL URL from individual components
    try:
        host = get_env_var("DB_HOST", "localhost", required=True)
        port = get_env_var("DB_PORT", "5432")
        database = get_env_var("DB_NAME", "algotrading", required=True)
        username = get_env_var("DB_USER", required=True)
        password = get_env_var("DB_PASSWORD", "")

        # URL encode password to handle special characters
        if password:
            password = quote_plus(password)
            user_part = f"{username}:{password}"
        else:
            user_part = username

        connection_url = f"postgresql://{user_part}@{host}:{port}/{database}"
        logger.debug(f"Constructed database URL for host: {host}, database: {database}")
        return connection_url

    except ValueError as e:
        logger.error(
            "Database configuration error. Please check your environment variables."
        )
        logger.error("Required variables: DB_HOST, DB_NAME, DB_USER")
        logger.error("Optional variables: DB_PASSWORD, DB_PORT")
        logger.error("Alternative: Set DB_URI with full connection string")
        logger.error("Example: copy env.example to .env and update with your values")
        raise ValueError(f"Database configuration incomplete: {str(e)}")


def validate_required_env_vars() -> Dict[str, Any]:
    """
    Validate that all required environment variables are present.

    Returns:
        Dictionary with validation results
    """
    # Check if DB_URI is provided as alternative to individual DB vars
    db_uri_provided = get_env_var("DB_URI") is not None

    if db_uri_provided:
        required_vars = []  # DB_URI covers all database requirements
        logger.debug("Using DB_URI for database configuration")
    else:
        required_vars = ["DB_HOST", "DB_NAME", "DB_USER"]

    optional_vars = [
        "DB_PASSWORD",
        "DB_PORT",
        "DB_URI",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "LOG_LEVEL",
        "ENVIRONMENT",
    ]

    results = {
        "valid": True,
        "missing_required": [],
        "missing_optional": [],
        "present": [],
        "setup_instructions": [],
    }

    # Check required variables
    for var in required_vars:
        value = get_env_var(var)
        if value is None or value.strip() == "":
            results["missing_required"].append(var)
            results["valid"] = False
        else:
            results["present"].append(var)

    # Check optional variables
    for var in optional_vars:
        value = get_env_var(var)
        if value is None or value.strip() == "":
            results["missing_optional"].append(var)
        else:
            results["present"].append(var)

    # Add helpful setup instructions if there are missing required vars
    if results["missing_required"]:
        results["setup_instructions"] = [
            "1. Copy 'env.example' to '.env' in the project root",
            "2. Edit .env file with your database credentials",
            "3. Required: DB_HOST, DB_NAME, DB_USER",
            "4. Optional: DB_PASSWORD, ALPACA_API_KEY, etc.",
            "5. Alternative: Set DB_URI with full connection string",
        ]

        logger.error("Missing required environment variables!")
        logger.error(f"Missing: {', '.join(results['missing_required'])}")
        for instruction in results["setup_instructions"]:
            logger.error(instruction)

    return results


def get_log_config() -> Dict[str, Any]:
    """
    Get logging configuration from environment variables.

    Returns:
        Dictionary with logging configuration
    """
    return {
        "level": get_env_var("LOG_LEVEL", "INFO").upper(),
        "format": get_env_var(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": get_env_var("LOG_LEVEL", "INFO").upper(),
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "algotrading.log",
                "level": get_env_var("LOG_LEVEL", "INFO").upper(),
                "formatter": "default",
            },
        },
        "formatters": {
            "default": {
                "format": get_env_var(
                    "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            }
        },
    }


def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration from environment variables.

    Returns:
        Dictionary with API configuration
    """
    return {
        "alpaca": {
            "api_key": get_env_var("ALPACA_API_KEY"),
            "secret_key": get_env_var("ALPACA_SECRET_KEY"),
            "base_url": get_env_var(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
        },
        "alpha_vantage_key": get_env_var("ALPHA_VANTAGE_API_KEY"),
        "yahoo_min_delay": int(get_env_var("YAHOO_MIN_DELAY", "1")),
        "yahoo_max_delay": int(get_env_var("YAHOO_MAX_DELAY", "3")),
        "yahoo_max_retries": int(get_env_var("YAHOO_MAX_RETRIES", "3")),
    }
