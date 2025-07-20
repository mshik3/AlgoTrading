"""
Unit tests for config module.
Tests configuration management, environment variable handling, and validation.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from utils.config import (
    load_environment,
    get_env_var,
    get_database_url,
    validate_required_env_vars,
    get_log_config,
    get_api_config,
)


class TestLoadEnvironment:
    """Test environment loading functionality."""

    def test_load_environment_success(self):
        """Test successful environment loading."""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DB_HOST=localhost\n")
            f.write("DB_NAME=test_db\n")
            f.write("DB_USER=test_user\n")
            f.write("DB_PASSWORD=test_password\n")
            env_file = f.name

        try:
            # Clear existing environment variables to test loading
            original_env = {}
            for key in ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]:
                if key in os.environ:
                    original_env[key] = os.environ[key]
                    del os.environ[key]

            success = load_environment(env_file)
            assert success is True

            # Check that variables were loaded
            assert os.environ.get("DB_HOST") == "localhost"
            assert os.environ.get("DB_NAME") == "test_db"
            assert os.environ.get("DB_USER") == "test_user"
            assert os.environ.get("DB_PASSWORD") == "test_password"
        finally:
            # Restore original environment
            for key, value in original_env.items():
                os.environ[key] = value
            os.unlink(env_file)

    def test_load_environment_file_not_found(self):
        """Test loading environment from non-existent file."""
        success = load_environment("nonexistent.env")
        assert success is False

    def test_load_environment_with_comments(self):
        """Test loading environment with comments."""
        # Store and clear existing environment variables
        original_env = {}
        for key in ["DB_HOST", "DB_NAME"]:
            if key in os.environ:
                original_env[key] = os.environ[key]
                del os.environ[key]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# This is a comment\n")
            f.write("DB_HOST=localhost\n")
            f.write("  # Another comment\n")
            f.write("DB_NAME=test_db\n")
            env_file = f.name

        try:
            success = load_environment(env_file)
            assert success is True

            # Check that variables were loaded (comments ignored)
            assert os.environ.get("DB_HOST") == "localhost"
            assert os.environ.get("DB_NAME") == "test_db"
        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                os.environ[key] = value
            os.unlink(env_file)

    def test_load_environment_with_quotes(self):
        """Test loading environment with quoted values."""
        # Store and clear existing environment variables
        original_env = {}
        for key in ["DB_HOST", "DB_NAME", "DB_USER"]:
            if key in os.environ:
                original_env[key] = os.environ[key]
                del os.environ[key]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write('DB_HOST="localhost"\n')
            f.write("DB_NAME='test_db'\n")
            f.write("DB_USER=test_user\n")
            env_file = f.name

        try:
            success = load_environment(env_file)
            assert success is True

            # Check that quotes were removed
            assert os.environ.get("DB_HOST") == "localhost"
            assert os.environ.get("DB_NAME") == "test_db"
            assert os.environ.get("DB_USER") == "test_user"
        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                os.environ[key] = value
            os.unlink(env_file)

    def test_load_environment_invalid_line(self):
        """Test loading environment with invalid lines."""
        # Store and clear existing environment variables
        original_env = {}
        for key in ["DB_HOST", "DB_NAME"]:
            if key in os.environ:
                original_env[key] = os.environ[key]
                del os.environ[key]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DB_HOST=localhost\n")
            f.write("INVALID_LINE\n")  # No equals sign
            f.write("DB_NAME=test_db\n")
            env_file = f.name

        try:
            success = load_environment(env_file)
            assert success is True

            # Should still load valid lines
            assert os.environ.get("DB_HOST") == "localhost"
            assert os.environ.get("DB_NAME") == "test_db"
        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                os.environ[key] = value
            os.unlink(env_file)

    def test_load_environment_preserve_existing(self):
        """Test that existing environment variables are preserved."""
        # Set existing environment variable
        os.environ["EXISTING_VAR"] = "existing_value"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("NEW_VAR=new_value\n")
            env_file = f.name

        try:
            success = load_environment(env_file)
            assert success is True

            # Existing variable should be preserved
            assert os.environ.get("EXISTING_VAR") == "existing_value"
            # New variable should be loaded
            assert os.environ.get("NEW_VAR") == "new_value"
        finally:
            os.unlink(env_file)

    def test_load_environment_error_handling(self):
        """Test error handling during environment loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DB_HOST=localhost\n")
            env_file = f.name

        try:
            # Mock open to raise an exception
            with patch("builtins.open", side_effect=Exception("File error")):
                success = load_environment(env_file)
                assert success is False
        finally:
            os.unlink(env_file)


class TestGetEnvVar:
    """Test environment variable retrieval."""

    def test_get_env_var_existing(self):
        """Test getting existing environment variable."""
        os.environ["TEST_VAR"] = "test_value"

        value = get_env_var("TEST_VAR")
        assert value == "test_value"

    def test_get_env_var_with_default(self):
        """Test getting environment variable with default."""
        # Remove variable if it exists
        os.environ.pop("NONEXISTENT_VAR", None)

        value = get_env_var("NONEXISTENT_VAR", "default_value")
        assert value == "default_value"

    def test_get_env_var_required_missing(self):
        """Test getting required environment variable that is missing."""
        os.environ.pop("REQUIRED_VAR", None)

        with pytest.raises(ValueError, match="Required environment variable"):
            get_env_var("REQUIRED_VAR", required=True)

    def test_get_env_var_required_present(self):
        """Test getting required environment variable that is present."""
        os.environ["REQUIRED_VAR"] = "required_value"

        value = get_env_var("REQUIRED_VAR", required=True)
        assert value == "required_value"

    def test_get_env_var_empty_string(self):
        """Test getting environment variable that is empty string."""
        os.environ["EMPTY_VAR"] = ""

        value = get_env_var("EMPTY_VAR", "default_value")
        assert value == ""

    def test_get_env_var_none_value(self):
        """Test getting environment variable that is None."""
        os.environ["NONE_VAR"] = "None"

        value = get_env_var("NONE_VAR", "default_value")
        assert value == "None"  # String "None", not None


class TestGetDatabaseUrl:
    """Test database URL generation."""

    def test_get_database_url_postgres(self):
        """Test PostgreSQL database URL generation."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "test_db",
                "DB_USER": "test_user",
                "DB_PASSWORD": "test_password",
            },
        ):
            url = get_database_url()
            assert "postgresql://test_user:test_password@localhost:5432/test_db" in url

    def test_get_database_url_sqlite(self):
        """Test SQLite database URL generation."""
        with patch.dict(
            os.environ, {"DB_TYPE": "sqlite", "DB_PATH": "/path/to/database.db"}
        ):
            url = get_database_url()
            assert "sqlite:////path/to/database.db" in url

    def test_get_database_url_missing_credentials(self):
        """Test database URL generation with missing credentials."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_NAME": "test_db",
                # Missing user and password
            },
        ):
            url = get_database_url()
            # Should still generate a URL, possibly with empty credentials
            assert "postgresql://" in url

    def test_get_database_url_special_characters(self):
        """Test database URL generation with special characters in password."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "test_db",
                "DB_USER": "test_user",
                "DB_PASSWORD": "test@password#123",
            },
        ):
            url = get_database_url()
            # Password should be URL-encoded
            assert "test%40password%23123" in url

    def test_get_database_url_default_port(self):
        """Test database URL generation with default port."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_NAME": "test_db",
                "DB_USER": "test_user",
                "DB_PASSWORD": "test_password",
                # No port specified
            },
        ):
            url = get_database_url()
            assert ":5432" in url  # Default PostgreSQL port


class TestValidateRequiredEnvVars:
    """Test required environment variables validation."""

    def test_validate_required_env_vars_all_present(self):
        """Test validation when all required variables are present."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_NAME": "test_db",
                "DB_USER": "test_user",
                "DB_PASSWORD": "test_password",
            },
        ):
            result = validate_required_env_vars()
            assert result["valid"] is True
            assert result["missing_required"] == []

    def test_validate_required_env_vars_missing(self):
        """Test validation when required variables are missing."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost"
                # Missing DB_NAME, DB_USER, DB_PASSWORD
            },
            clear=True,
        ), patch("utils.config.load_environment") as mock_load:
            mock_load.return_value = False
            result = validate_required_env_vars()
            assert result["valid"] is False
            assert len(result["missing_required"]) > 0
            assert "DB_NAME" in result["missing_required"]

    def test_validate_required_env_vars_empty_values(self):
        """Test validation with empty values."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_NAME": "",  # Empty value
                "DB_USER": "test_user",
                "DB_PASSWORD": "test_password",
            },
            clear=True,
        ), patch("utils.config.load_environment") as mock_load:
            mock_load.return_value = False
            result = validate_required_env_vars()
            assert result["valid"] is False
            assert "DB_NAME" in result["missing_required"]

    def test_validate_required_env_vars_optional_vars(self):
        """Test validation with optional variables."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_NAME": "test_db",
                "DB_USER": "test_user",
                "DB_PASSWORD": "test_password",
                "OPTIONAL_VAR": "optional_value",  # Optional variable
            },
        ):
            result = validate_required_env_vars()
            assert result["valid"] is True


class TestGetLogConfig:
    """Test logging configuration."""

    def test_get_log_config_default(self):
        """Test default logging configuration."""
        config = get_log_config()

        assert "level" in config
        assert "format" in config
        assert "handlers" in config

    def test_get_log_config_custom_level(self):
        """Test logging configuration with custom level."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            config = get_log_config()
            assert config["level"] == "DEBUG"

    def test_get_log_config_invalid_level(self):
        """Test logging configuration with invalid level."""
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID_LEVEL"}):
            config = get_log_config()
            # Should fall back to default level
            assert "level" in config


class TestGetApiConfig:
    """Test API configuration."""

    def test_get_api_config_alpaca(self):
        """Test Alpaca API configuration."""
        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "test_key",
                "ALPACA_SECRET_KEY": "test_secret",
                "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
            },
        ):
            config = get_api_config()

            assert "alpaca" in config
            assert config["alpaca"]["api_key"] == "test_key"
            assert config["alpaca"]["secret_key"] == "test_secret"
            assert config["alpaca"]["base_url"] == "https://paper-api.alpaca.markets"

    def test_get_api_config_missing_keys(self):
        """Test API configuration with missing keys."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.config.load_environment", return_value=False):
                config = get_api_config()

                # Should still return config structure
                assert "alpaca" in config
                assert config["alpaca"]["api_key"] is None
                assert config["alpaca"]["secret_key"] is None

    def test_get_api_config_partial_keys(self):
        """Test API configuration with partial keys."""
        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "test_key"
                # Missing secret key
            },
            clear=True,
        ):
            with patch("utils.config.load_environment", return_value=False):
                config = get_api_config()

                assert config["alpaca"]["api_key"] == "test_key"
                assert config["alpaca"]["secret_key"] is None


class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Store original environment variables
        original_env = {}
        for key in [
            "DB_HOST",
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
            "LOG_LEVEL",
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
        ]:
            if key in os.environ:
                original_env[key] = os.environ[key]
                del os.environ[key]

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DB_HOST=localhost\n")
            f.write("DB_NAME=test_db\n")
            f.write("DB_USER=test_user\n")
            f.write("DB_PASSWORD=test_password\n")
            f.write("LOG_LEVEL=INFO\n")
            f.write("ALPACA_API_KEY=test_key\n")
            f.write("ALPACA_SECRET_KEY=test_secret\n")
            env_file = f.name

        try:
            # Load environment
            success = load_environment(env_file)
            assert success is True

            # Validate required variables
            validation = validate_required_env_vars()
            assert validation["valid"] is True

            # Get database URL
            db_url = get_database_url()
            assert "postgresql://" in db_url

            # Get log config
            log_config = get_log_config()
            assert log_config["level"] == "INFO"

            # Get API config
            api_config = get_api_config()
            assert api_config["alpaca"]["api_key"] == "test_key"

        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                os.environ[key] = value
            os.unlink(env_file)

    def test_config_error_recovery(self):
        """Test configuration error recovery."""
        # Test with missing environment file
        success = load_environment("nonexistent.env")
        assert success is False

        # System should still work with system environment variables
        validation = validate_required_env_vars()
        # Result depends on what's in system environment
        assert "valid" in validation

    def test_config_performance(self):
        """Test configuration performance."""
        import time

        start_time = time.time()

        # Load environment multiple times
        for _ in range(100):
            get_env_var("TEST_VAR", "default")

        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
