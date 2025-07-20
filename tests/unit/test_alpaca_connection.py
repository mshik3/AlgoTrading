#!/usr/bin/env python3
"""
Alpaca Connection Test Script

This script helps debug Alpaca API connection issues.
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_environment, get_api_config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_environment_loading():
    """Test if environment variables are being loaded correctly."""
    print("🔍 Testing Environment Loading...")
    print("=" * 50)

    # Load environment
    load_environment()

    # Get API config
    config = get_api_config()

    print(
        f"Alpaca API Key: {config['alpaca']['api_key'][:10]}..."
        if config["alpaca"]["api_key"]
        else "NOT_FOUND"
    )
    print(
        f"Alpaca Secret Key: {config['alpaca']['secret_key'][:10]}..."
        if config["alpaca"]["secret_key"]
        else "NOT_FOUND"
    )
    print(f"Alpaca Base URL: {config['alpaca']['base_url']}")

    # Check if keys look valid
    api_key = config["alpaca"]["api_key"]
    secret_key = config["alpaca"]["secret_key"]

    if api_key and secret_key:
        print(f"API Key Length: {len(api_key)}")
        print(f"Secret Key Length: {len(secret_key)}")

        # Alpaca API keys typically start with 'PK' for paper trading
        if api_key.startswith("PK"):
            print("✅ API Key format looks correct (Paper Trading)")
        elif api_key.startswith("AK"):
            print("⚠️  API Key format looks like Live Trading key")
        else:
            print("❌ API Key format doesn't match expected pattern")

        if len(secret_key) >= 20:
            print("✅ Secret Key length looks reasonable")
        else:
            print("❌ Secret Key seems too short")
    else:
        print("❌ Missing API keys")

    print()


def test_alpaca_connection():
    """Test Alpaca connection with detailed error handling."""
    print("🔌 Testing Alpaca Connection...")
    print("=" * 50)

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data import StockHistoricalDataClient

        # Load environment
        load_environment()
        config = get_api_config()

        api_key = config["alpaca"]["api_key"]
        secret_key = config["alpaca"]["secret_key"]
        base_url = config["alpaca"]["base_url"]

        if not api_key or not secret_key:
            print("❌ Missing API credentials")
            return False

        print(f"Using Base URL: {base_url}")
        print(f"API Key: {api_key[:10]}...")
        print(f"Secret Key: {secret_key[:10]}...")

        # Test trading client
        print("\n📊 Testing Trading Client...")
        try:
            trading_client = TradingClient(
                api_key=api_key, secret_key=secret_key, paper=True
            )

            # Try to get account info
            account = trading_client.get_account()
            print("✅ Trading Client Connection Successful!")
            print(f"   Account ID: {account.id}")
            print(f"   Status: {account.status}")
            print(f"   Cash: ${float(account.cash):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")

        except Exception as e:
            print(f"❌ Trading Client Error: {str(e)}")
            if "403" in str(e):
                print("   This suggests an authentication issue.")
                print("   Possible causes:")
                print("   - API keys are incorrect")
                print("   - API keys are for live trading instead of paper")
                print("   - API keys haven't been activated yet")
                print("   - Account is suspended")
            return False

        # Test data client
        print("\n📈 Testing Data Client...")
        try:
            data_client = StockHistoricalDataClient(
                api_key=api_key, secret_key=secret_key
            )

            # Try to get a small amount of data
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data import TimeFrame

            request = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame.Day,
                start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                end=datetime.now(),
            )

            bars = data_client.get_stock_bars(request)
            print("✅ Data Client Connection Successful!")
            print(f"   Retrieved {len(bars) if bars else 0} bars for SPY")

        except Exception as e:
            print(f"❌ Data Client Error: {str(e)}")
            return False

        return True

    except ImportError as e:
        print(f"❌ Alpaca SDK not available: {str(e)}")
        print("   Install with: pip install alpaca-py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False


def test_direct_api_call():
    """Test Alpaca API directly with requests."""
    print("🌐 Testing Direct API Call...")
    print("=" * 50)

    try:
        import requests

        # Load environment
        load_environment()
        config = get_api_config()

        api_key = config["alpaca"]["api_key"]
        secret_key = config["alpaca"]["secret_key"]
        base_url = config["alpaca"]["base_url"]

        if not api_key or not secret_key:
            print("❌ Missing API credentials")
            return False

        # Test account endpoint
        url = f"{base_url}/v2/account"
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

        print(f"Making request to: {url}")
        print(f"Headers: {headers}")

        response = requests.get(url, headers=headers)

        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")

        if response.status_code == 200:
            print("✅ Direct API call successful!")
            return True
        elif response.status_code == 403:
            print("❌ 403 Forbidden - Authentication failed")
            print("   This usually means:")
            print("   - API keys are incorrect")
            print("   - API keys are for wrong environment (live vs paper)")
            print("   - API keys haven't been activated")
            return False
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Direct API call failed: {str(e)}")
        return False


def main():
    """Main test function."""
    print("🚀 Alpaca Connection Troubleshooting")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test 1: Environment loading
    test_environment_loading()

    # Test 2: Direct API call
    direct_success = test_direct_api_call()

    # Test 3: Alpaca SDK connection
    if direct_success:
        sdk_success = test_alpaca_connection()
    else:
        print("\n⏭️  Skipping SDK test due to direct API failure")
        sdk_success = False

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    if direct_success and sdk_success:
        print("✅ All tests passed! Alpaca connection is working.")
        print("   You can now run the Golden Cross analysis.")
    elif direct_success and not sdk_success:
        print("⚠️  Direct API works but SDK has issues.")
        print("   This might be a SDK version or configuration issue.")
    else:
        print("❌ Connection failed. Please check:")
        print("   1. API keys are correct")
        print("   2. API keys are for paper trading (start with 'PK')")
        print("   3. API keys have been activated in Alpaca dashboard")
        print("   4. Account is not suspended")
        print("   5. You're using the correct base URL")

    print("\n🔧 Next Steps:")
    if not direct_success:
        print("   - Verify API keys in Alpaca dashboard")
        print("   - Check if keys are for paper trading")
        print("   - Ensure account is active")
    else:
        print("   - Run: python golden_cross_live_analysis.py")


if __name__ == "__main__":
    main()
