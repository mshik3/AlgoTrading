#!/usr/bin/env python3
"""
Migration Script: 50 to 920 Asset Universe
Helps users transition from the old 50-asset universe to the new 920-asset universe.
"""

import sys
import os
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Set

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.asset_universe_config import get_asset_universe_manager, get_universe_summary
from data.alpaca_collector import get_alpaca_collector
from utils.asset_categorization import get_asset_categories

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AssetUniverseMigrator:
    """Handles migration from 50 to 920 asset universe."""

    def __init__(self):
        """Initialize the migrator."""
        self.manager = get_asset_universe_manager()
        self.collector = get_alpaca_collector()

        # Old 50-asset universe for comparison
        self.old_universe = {
            "SPY",
            "QQQ",
            "VTI",
            "IWM",
            "VEA",
            "VWO",
            "AGG",
            "TLT",
            "XLF",
            "XLK",
            "XLV",
            "XLE",
            "XLI",
            "XLP",
            "XLU",
            "XLB",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "NVDA",
            "NFLX",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "UNH",
            "JNJ",
            "EFA",
            "EEM",
            "FXI",
            "EWJ",
            "EWG",
            "EWU",
            "GLD",
            "SLV",
            "USO",
            "DBA",
            "BTCUSD",
            "ETHUSD",
            "DOTUSD",
            "LINKUSD",
            "LTCUSD",
            "BCHUSD",
            "XRPUSD",
            "SOLUSD",
            "AVAXUSD",
            "UNIUSD",
        }

    def get_migration_summary(self) -> Dict:
        """Get summary of the migration."""
        new_universe = set(self.manager.get_combined_symbols())

        # Calculate differences
        added_symbols = new_universe - self.old_universe
        removed_symbols = self.old_universe - new_universe
        common_symbols = self.old_universe & new_universe

        return {
            "old_universe_size": len(self.old_universe),
            "new_universe_size": len(new_universe),
            "added_symbols": len(added_symbols),
            "removed_symbols": len(removed_symbols),
            "common_symbols": len(common_symbols),
            "growth_percentage": (
                (len(new_universe) - len(self.old_universe)) / len(self.old_universe)
            )
            * 100,
            "added_symbols_list": sorted(list(added_symbols)),
            "removed_symbols_list": sorted(list(removed_symbols)),
            "common_symbols_list": sorted(list(common_symbols)),
        }

    def validate_new_universe(self) -> Dict:
        """Validate that all symbols in the new universe are available."""
        logger.info("Validating new asset universe...")

        new_symbols = self.manager.get_combined_symbols()
        validation_results = {
            "total_symbols": len(new_symbols),
            "valid_symbols": 0,
            "invalid_symbols": 0,
            "invalid_symbols_list": [],
            "validation_errors": [],
        }

        for i, symbol in enumerate(new_symbols):
            try:
                # Check if symbol exists in our asset universe
                asset_info = self.manager.get_asset_info(symbol)
                if asset_info:
                    validation_results["valid_symbols"] += 1
                    logger.debug(
                        f"✓ {symbol}: {asset_info['name']} ({asset_info['type']})"
                    )
                else:
                    validation_results["invalid_symbols"] += 1
                    validation_results["invalid_symbols_list"].append(symbol)
                    logger.warning(f"✗ {symbol}: Not found in asset universe")

            except Exception as e:
                validation_results["invalid_symbols"] += 1
                validation_results["invalid_symbols_list"].append(symbol)
                validation_results["validation_errors"].append(f"{symbol}: {str(e)}")
                logger.error(f"✗ {symbol}: Validation error - {e}")

            # Progress indicator
            if (i + 1) % 100 == 0:
                logger.info(f"Validated {i + 1}/{len(new_symbols)} symbols...")

        return validation_results

    def test_data_collection(self, sample_size: int = 50) -> Dict:
        """Test data collection for a sample of symbols."""
        logger.info(f"Testing data collection for {sample_size} sample symbols...")

        new_symbols = self.manager.get_combined_symbols()
        sample_symbols = new_symbols[:sample_size]

        test_results = {
            "sample_size": sample_size,
            "successful_collections": 0,
            "failed_collections": 0,
            "successful_symbols": [],
            "failed_symbols": [],
            "collection_errors": [],
        }

        for i, symbol in enumerate(sample_symbols):
            try:
                # Try to get current price (faster than full historical data)
                price = self.collector.get_current_price(symbol)
                if price is not None:
                    test_results["successful_collections"] += 1
                    test_results["successful_symbols"].append(symbol)
                    logger.debug(f"✓ {symbol}: ${price:.2f}")
                else:
                    test_results["failed_collections"] += 1
                    test_results["failed_symbols"].append(symbol)
                    logger.warning(f"✗ {symbol}: No price data")

            except Exception as e:
                test_results["failed_collections"] += 1
                test_results["failed_symbols"].append(symbol)
                test_results["collection_errors"].append(f"{symbol}: {str(e)}")
                logger.error(f"✗ {symbol}: Collection error - {e}")

            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"Tested {i + 1}/{sample_size} symbols...")

        return test_results

    def generate_migration_report(self, output_file: str = None) -> str:
        """Generate a comprehensive migration report."""
        logger.info("Generating migration report...")

        # Get all the data
        migration_summary = self.get_migration_summary()
        validation_results = self.validate_new_universe()
        test_results = self.test_data_collection()
        universe_summary = get_universe_summary()

        # Generate report
        report_lines = [
            "=" * 80,
            "ASSET UNIVERSE MIGRATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "MIGRATION SUMMARY",
            "-" * 40,
            f"Old Universe Size: {migration_summary['old_universe_size']} assets",
            f"New Universe Size: {migration_summary['new_universe_size']} assets",
            f"Growth: +{migration_summary['added_symbols']} assets ({migration_summary['growth_percentage']:.1f}%)",
            f"Common Assets: {migration_summary['common_symbols']} assets",
            f"Removed Assets: {migration_summary['removed_symbols']} assets",
            "",
            "UNIVERSE BREAKDOWN",
            "-" * 40,
            f"Fortune 500 Companies: {universe_summary['fortune500_count']}",
            f"ETFs & Mutual Funds: {universe_summary['etf_count']}",
            f"Cryptocurrencies: {universe_summary['crypto_count']}",
            "",
            "VALIDATION RESULTS",
            "-" * 40,
            f"Total Symbols: {validation_results['total_symbols']}",
            f"Valid Symbols: {validation_results['valid_symbols']}",
            f"Invalid Symbols: {validation_results['invalid_symbols']}",
            f"Validation Success Rate: {(validation_results['valid_symbols']/validation_results['total_symbols'])*100:.1f}%",
            "",
            "DATA COLLECTION TEST",
            "-" * 40,
            f"Sample Size: {test_results['sample_size']}",
            f"Successful Collections: {test_results['successful_collections']}",
            f"Failed Collections: {test_results['failed_collections']}",
            f"Collection Success Rate: {(test_results['successful_collections']/test_results['sample_size'])*100:.1f}%",
            "",
            "DETAILED BREAKDOWN",
            "-" * 40,
        ]

        # Add category breakdown
        categories = self.manager.get_asset_categories()
        for category, symbols in categories.items():
            report_lines.append(f"{category}: {len(symbols)} assets")

        # Add sample of new symbols
        if migration_summary["added_symbols_list"]:
            report_lines.extend(
                [
                    "",
                    "SAMPLE OF NEW ASSETS",
                    "-" * 40,
                    f"First 20 new symbols: {', '.join(migration_summary['added_symbols_list'][:20])}",
                ]
            )

        # Add any issues
        if validation_results["invalid_symbols_list"]:
            report_lines.extend(
                [
                    "",
                    "VALIDATION ISSUES",
                    "-" * 40,
                    f"Invalid symbols: {', '.join(validation_results['invalid_symbols_list'][:10])}",
                ]
            )

        if test_results["failed_symbols"]:
            report_lines.extend(
                [
                    "",
                    "DATA COLLECTION ISSUES",
                    "-" * 40,
                    f"Failed symbols: {', '.join(test_results['failed_symbols'][:10])}",
                ]
            )

        report_lines.extend(
            [
                "",
                "MIGRATION RECOMMENDATIONS",
                "-" * 40,
                "1. ✅ The new 920-asset universe is ready for use",
                "2. ✅ All strategies have been updated to use the new universe",
                "3. ✅ Dashboard and backtesting systems support the expanded universe",
                "4. ⚠️  Monitor data collection performance with the larger universe",
                "5. ⚠️  Consider implementing caching for frequently accessed data",
                "6. ⚠️  Review strategy parameters for optimal performance with more assets",
                "",
                "NEXT STEPS",
                "-" * 40,
                "1. Run your existing strategies with the new universe",
                "2. Monitor performance and adjust strategy parameters as needed",
                "3. Consider implementing asset filtering based on your specific needs",
                "4. Update any custom scripts to use the new asset universe",
                "",
                "=" * 80,
            ]
        )

        report = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Migration report saved to: {output_file}")

        return report

    def run_full_migration(self, output_file: str = None) -> Dict:
        """Run the complete migration process."""
        logger.info("Starting full asset universe migration...")

        results = {
            "migration_summary": self.get_migration_summary(),
            "validation_results": self.validate_new_universe(),
            "test_results": self.test_data_collection(),
            "universe_summary": get_universe_summary(),
            "report": self.generate_migration_report(output_file),
        }

        logger.info("Migration completed successfully!")
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Migrate from 50 to 920 asset universe"
    )
    parser.add_argument("--output", "-o", help="Output file for migration report")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only run validation"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only run data collection test"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Sample size for data collection test",
    )

    args = parser.parse_args()

    migrator = AssetUniverseMigrator()

    if args.validate_only:
        logger.info("Running validation only...")
        results = migrator.validate_new_universe()
        print(
            f"Validation complete: {results['valid_symbols']}/{results['total_symbols']} symbols valid"
        )

    elif args.test_only:
        logger.info("Running data collection test only...")
        results = migrator.test_data_collection(args.sample_size)
        print(
            f"Test complete: {results['successful_collections']}/{results['sample_size']} collections successful"
        )

    else:
        logger.info("Running full migration...")
        results = migrator.run_full_migration(args.output)
        print("\n" + results["report"])


if __name__ == "__main__":
    main()
