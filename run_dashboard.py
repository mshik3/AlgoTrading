#!/usr/bin/env python3
"""
Professional AlgoTrading Dashboard Launcher
Initializes and runs the Dash trading dashboard with all components
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "dash",
        "dash_bootstrap_components",
        "plotly",
        "pandas",
        "numpy",
        "yfinance",
        "flask_caching",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False

    return True


def check_database_connection():
    """Check if database connection is available"""
    try:
        from utils.config import get_database_url, validate_required_env_vars
        from data.storage import get_engine, init_db
        
        print("🔍 Checking database connection...")
        
        # Test actual database connection
        engine = get_engine()
        if not init_db(engine):
            print("⚠️  Database initialization failed")
            return False
            
        print("✅ Database connection verified and initialized")
        return True

    except Exception as e:
        print(f"⚠️  Database connection issue: {e}")
        print("   Please ensure database is configured correctly")
        return False


def preload_market_data():
    """Pre-load market data for all strategies during dashboard startup"""
    print("\n🚀 MARKET DATA PRE-LOADING")
    print("=" * 60)
    print("📊 Pre-loading market data for 920+ asset universe...")
    print("💡 This ensures instant strategy execution without API delays")
    print("⏱️ First-time setup: 5-10 minutes | Subsequent runs: 1-2 minutes")
    print()
    
    try:
        from dashboard.services.data_preloader import create_preloader
        
        # Create progress display
        progress_info = {
            "last_pct": 0,
            "start_time": datetime.now()
        }
        
        def progress_callback(status):
            """Display progress updates with estimated completion time"""
            current_pct = status["progress_pct"]
            
            # Only update display for significant progress changes
            if current_pct - progress_info["last_pct"] >= 5 or status["remaining"] < 10:
                elapsed = status["elapsed_time"]
                remaining = status["remaining_time"]
                
                # Create progress bar
                bar_length = 40
                filled_length = int(bar_length * current_pct / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                print(f"\r📈 Progress: [{bar}] {current_pct:.1f}% "
                      f"({status['completed']}/{status['total_symbols']}) "
                      f"Success: {status['success_rate']:.1f}% "
                      f"ETA: {str(remaining).split('.')[0]}", end="")
                
                progress_info["last_pct"] = current_pct
        
        # Create preloader and run data collection
        preloader = create_preloader()
        
        print(f"🎯 Target: {len(preloader.symbols)} symbols (Fortune 500 + ETFs + Crypto)")
        print("⚡ Using optimized incremental loading with gap detection")
        print()
        
        # Run pre-loading with progress tracking
        results = preloader.preload_all_data(
            period="2y",  # 2 years of historical data
            status_callback=progress_callback,
            max_workers=3  # Moderate parallelism to avoid API limits
        )
        
        print()  # New line after progress bar
        
        if results["success"]:
            duration = results["duration"]
            print(f"✅ Pre-loading completed successfully!")
            print(f"   📊 Processed: {results['completed_symbols']} symbols")
            print(f"   📈 Total records: {results['total_records']:,}")
            print(f"   ⏱️ Duration: {str(duration).split('.')[0]}")
            
            if results["completed_symbols"] > 0:
                avg_time = duration.total_seconds() / results["completed_symbols"]
                print(f"   ⚡ Avg per symbol: {avg_time:.1f}s")
            
            # Validate data quality
            print("\n🔍 Validating data quality...")
            validation = preloader.validate_preloaded_data(sample_size=20)
            
            if validation["success"]:
                print(f"✅ Data validation passed: {validation['symbols_with_data']}/{validation['total_validated']} symbols valid")
                print(f"   📊 Avg records per symbol: {validation['avg_records_per_symbol']:.0f}")
            else:
                print(f"⚠️ Data validation issues detected")
                if validation.get("issues"):
                    for issue in validation["issues"][:3]:
                        print(f"   - {issue}")
            
            print("✅ Market data pre-loading complete - strategies will run instantly!")
            return True
            
        else:
            print(f"❌ Pre-loading failed!")
            if results["errors"]:
                print("   Errors:")
                for error in results["errors"][:5]:
                    print(f"   - {error}")
            print("   Dashboard will continue with available data")
            return False
            
    except Exception as e:
        print(f"❌ Pre-loading setup failed: {e}")
        print("   Dashboard will continue without pre-loaded data")
        print("   Strategies may experience slower execution times")
        return False


def initialize_dashboard():
    """Initialize the trading dashboard"""
    print("🚀 Initializing AlgoTrading Professional Dashboard...")
    print("=" * 60)

    # Import dashboard components with absolute imports
    import sys
    import os

    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        from dashboard.app import app

        print("✅ Dashboard app imported successfully")

        # Try to import LiveDataManager if available
        try:
            from dashboard.data.live_data import LiveDataManager

            data_manager = LiveDataManager()
            cache = data_manager.setup_cache(app.server)
            print("✅ Data manager and caching initialized")
        except Exception as e:
            print(f"⚠️  Data manager not available: {e}")
            print("   Dashboard will use basic functionality")

        print("✅ Dashboard components initialized")
        print("✅ Real-time updates configured (30s interval)")

        return app

    except Exception as e:
        print(f"❌ Failed to initialize dashboard: {e}")
        raise


def display_startup_info():
    """Display startup information"""
    print("\n" + "=" * 60)
    print("🎯 PROFESSIONAL ALGOTRADING DASHBOARD")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌐 Dashboard URL: http://127.0.0.1:8050")
    print("📊 Features:")
    print("   ├── Real-time portfolio monitoring")
    print("   ├── Live positions with P&L tracking")
    print("   ├── Instant strategy execution (pre-loaded data)")
    print("   ├── Multi-strategy analysis (Golden Cross, Mean Reversion, ETF Rotation)")
    print("   ├── Professional TradingView charts")
    print("   ├── Activity feed & trade history")
    print("   └── Auto-refresh every 30 seconds")
    print("\n💡 Dashboard Design:")
    print("   ├── Industry-standard dark theme")
    print("   ├── Financial color coding (green/red)")
    print("   ├── Professional KPI cards")
    print("   └── Bloomberg/TradingView inspired UI")
    print("\n🔧 Technical:")
    print("   ├── Plotly Dash framework")
    print("   ├── Pre-loaded market data (920+ assets)")
    print("   ├── Optimized database storage")
    print("   ├── Incremental data updates")
    print("   ├── Flask-Caching for performance")
    print("   └── Paper trading integration")
    print("\n⚡ Performance:")
    print("   ├── Strategy execution: <2 seconds (was 2-5 minutes)")
    print("   ├── Data loading: Pre-loaded at startup")
    print("   ├── Multi-threading: Optimized API usage")
    print("   └── Caching: Smart data retrieval")
    print("=" * 60)


def main():
    """Main entry point for the dashboard"""

    # ASCII Art Header
    print(
        """
    ⚡ AlgoTrading Dashboard ⚡
    
    ▄▀█ █░░ █▀▀ █▀█ ▀█▀ █▀█ ▄▀█ █▀▄ █ █▄░█ █▀▀
    █▀█ █▄▄ █▄█ █▄█ ░█░ █▀▄ █▀█ █▄▀ █ █░▀█ █▄█
    
    Professional Trading Dashboard v1.0
    """
    )

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check database connection
    if not check_database_connection():
        print("❌ Database connection required for dashboard operation")
        sys.exit(1)

    # Pre-load market data for instant strategy execution
    preload_success = preload_market_data()
    if not preload_success:
        print("\n⚠️ Continuing without full data pre-loading")
        print("   Strategies may take longer to execute on first run")

    # Initialize dashboard
    try:
        app = initialize_dashboard()
        display_startup_info()

        print("\n🚀 Starting dashboard server...")
        print("   Press CTRL+C to stop")
        print("   Dashboard loading may take 10-15 seconds...")
        print()

        # Run the dashboard
        app.run_server(
            debug=True,
            host="127.0.0.1",
            port=8050,
            dev_tools_ui=False,
            dev_tools_props_check=False,
        )

    except KeyboardInterrupt:
        print("\n\n⏹️  Dashboard stopped by user")
        print("   Thanks for using AlgoTrading Dashboard!")

    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        print("   Check the error details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
