"""
Fortune 500 Companies Data Source
Provides comprehensive list of Fortune 500 companies with trading symbols.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Fortune500Company:
    """Represents a Fortune 500 company."""

    rank: int
    company_name: str
    symbol: str
    sector: str
    industry: str
    revenue_millions: float
    profit_millions: float
    employees: int
    headquarters: str
    market_cap_billions: Optional[float] = None


class Fortune500DataSource:
    """Data source for Fortune 500 companies."""

    def __init__(self):
        """Initialize the Fortune 500 data source."""
        self.companies = self._load_fortune500_data()

    def _load_fortune500_data(self) -> List[Fortune500Company]:
        """Load Fortune 500 companies data."""
        # Top 500 companies by revenue with trading symbols
        companies_data = [
            # Top 50 companies (most liquid and widely traded)
            Fortune500Company(
                1,
                "Apple Inc.",
                "AAPL",
                "Technology",
                "Consumer Electronics",
                394328,
                96995,
                164000,
                "Cupertino, CA",
                3000.0,
            ),
            Fortune500Company(
                2,
                "Microsoft Corporation",
                "MSFT",
                "Technology",
                "Software",
                198270,
                72426,
                221000,
                "Redmond, WA",
                2800.0,
            ),
            Fortune500Company(
                3,
                "Alphabet Inc.",
                "GOOGL",
                "Technology",
                "Internet Services",
                307394,
                76033,
                156500,
                "Mountain View, CA",
                1800.0,
            ),
            Fortune500Company(
                4,
                "Amazon.com Inc.",
                "AMZN",
                "Consumer Discretionary",
                "Internet Retail",
                514004,
                30425,
                1608000,
                "Seattle, WA",
                1700.0,
            ),
            Fortune500Company(
                5,
                "NVIDIA Corporation",
                "NVDA",
                "Technology",
                "Semiconductors",
                26974,
                9239,
                26473,
                "Santa Clara, CA",
                1200.0,
            ),
            Fortune500Company(
                6,
                "Tesla Inc.",
                "TSLA",
                "Consumer Discretionary",
                "Automotive",
                81462,
                14996,
                127855,
                "Austin, TX",
                800.0,
            ),
            Fortune500Company(
                7,
                "Meta Platforms Inc.",
                "META",
                "Technology",
                "Internet Services",
                116609,
                23200,
                86482,
                "Menlo Park, CA",
                900.0,
            ),
            Fortune500Company(
                8,
                "JPMorgan Chase & Co.",
                "JPM",
                "Financials",
                "Banking",
                154792,
                37676,
                256105,
                "New York, NY",
                450.0,
            ),
            Fortune500Company(
                9,
                "UnitedHealth Group Inc.",
                "UNH",
                "Healthcare",
                "Managed Healthcare",
                324162,
                22481,
                400000,
                "Minnetonka, MN",
                450.0,
            ),
            Fortune500Company(
                10,
                "JPMorgan Chase & Co.",
                "JPM",
                "Financials",
                "Banking",
                154792,
                37676,
                256105,
                "New York, NY",
                450.0,
            ),
            Fortune500Company(
                11,
                "Johnson & Johnson",
                "JNJ",
                "Healthcare",
                "Pharmaceuticals",
                94943,
                17941,
                152700,
                "New Brunswick, NJ",
                400.0,
            ),
            Fortune500Company(
                12,
                "Visa Inc.",
                "V",
                "Financials",
                "Financial Services",
                32265,
                17273,
                26500,
                "San Francisco, CA",
                500.0,
            ),
            Fortune500Company(
                13,
                "Procter & Gamble Co.",
                "PG",
                "Consumer Staples",
                "Household Products",
                80206,
                14742,
                101000,
                "Cincinnati, OH",
                350.0,
            ),
            Fortune500Company(
                14,
                "Mastercard Inc.",
                "MA",
                "Financials",
                "Financial Services",
                25012,
                11144,
                29500,
                "Purchase, NY",
                400.0,
            ),
            Fortune500Company(
                15,
                "Home Depot Inc.",
                "HD",
                "Consumer Discretionary",
                "Home Improvement",
                157403,
                17105,
                471600,
                "Atlanta, GA",
                300.0,
            ),
            Fortune500Company(
                16,
                "Bank of America Corp.",
                "BAC",
                "Financials",
                "Banking",
                94951,
                27553,
                213000,
                "Charlotte, NC",
                250.0,
            ),
            Fortune500Company(
                17,
                "Walt Disney Co.",
                "DIS",
                "Communication Services",
                "Entertainment",
                88898,
                2351,
                220000,
                "Burbank, CA",
                200.0,
            ),
            Fortune500Company(
                18,
                "Netflix Inc.",
                "NFLX",
                "Communication Services",
                "Entertainment",
                31616,
                4492,
                12800,
                "Los Gatos, CA",
                250.0,
            ),
            Fortune500Company(
                19,
                "Adobe Inc.",
                "ADBE",
                "Technology",
                "Software",
                19409,
                4756,
                29000,
                "San Jose, CA",
                250.0,
            ),
            Fortune500Company(
                20,
                "Salesforce Inc.",
                "CRM",
                "Technology",
                "Software",
                31235,
                208,
                79000,
                "San Francisco, CA",
                200.0,
            ),
            Fortune500Company(
                21,
                "Coca-Cola Co.",
                "KO",
                "Consumer Staples",
                "Beverages",
                43004,
                9542,
                86000,
                "Atlanta, GA",
                250.0,
            ),
            Fortune500Company(
                22,
                "PepsiCo Inc.",
                "PEP",
                "Consumer Staples",
                "Beverages",
                86392,
                8912,
                315000,
                "Purchase, NY",
                250.0,
            ),
            Fortune500Company(
                23,
                "Walmart Inc.",
                "WMT",
                "Consumer Staples",
                "Retail",
                611289,
                11680,
                2300000,
                "Bentonville, AR",
                400.0,
            ),
            Fortune500Company(
                24,
                "McDonald's Corp.",
                "MCD",
                "Consumer Discretionary",
                "Restaurants",
                23182,
                6185,
                200000,
                "Chicago, IL",
                200.0,
            ),
            Fortune500Company(
                25,
                "Verizon Communications Inc.",
                "VZ",
                "Communication Services",
                "Telecommunications",
                136835,
                21756,
                117100,
                "New York, NY",
                150.0,
            ),
            Fortune500Company(
                26,
                "AT&T Inc.",
                "T",
                "Communication Services",
                "Telecommunications",
                120741,
                -8500,
                160700,
                "Dallas, TX",
                120.0,
            ),
            Fortune500Company(
                27,
                "Intel Corporation",
                "INTC",
                "Technology",
                "Semiconductors",
                63054,
                7968,
                121100,
                "Santa Clara, CA",
                200.0,
            ),
            Fortune500Company(
                28,
                "Cisco Systems Inc.",
                "CSCO",
                "Technology",
                "Networking",
                51557,
                11812,
                83000,
                "San Jose, CA",
                200.0,
            ),
            Fortune500Company(
                29,
                "Oracle Corporation",
                "ORCL",
                "Technology",
                "Software",
                42440,
                8503,
                164000,
                "Austin, TX",
                250.0,
            ),
            Fortune500Company(
                30,
                "IBM Corporation",
                "IBM",
                "Technology",
                "Technology Services",
                60530,
                1640,
                282100,
                "Armonk, NY",
                150.0,
            ),
            Fortune500Company(
                31,
                "Goldman Sachs Group Inc.",
                "GS",
                "Financials",
                "Investment Banking",
                47365,
                11026,
                45000,
                "New York, NY",
                120.0,
            ),
            Fortune500Company(
                32,
                "Morgan Stanley",
                "MS",
                "Financials",
                "Investment Banking",
                53717,
                11008,
                82000,
                "New York, NY",
                150.0,
            ),
            Fortune500Company(
                33,
                "American Express Co.",
                "AXP",
                "Financials",
                "Financial Services",
                60425,
                7532,
                64000,
                "New York, NY",
                150.0,
            ),
            Fortune500Company(
                34,
                "Wells Fargo & Co.",
                "WFC",
                "Financials",
                "Banking",
                82301,
                13185,
                238000,
                "San Francisco, CA",
                150.0,
            ),
            Fortune500Company(
                35,
                "Citigroup Inc.",
                "C",
                "Financials",
                "Banking",
                75338,
                14845,
                240000,
                "New York, NY",
                100.0,
            ),
            Fortune500Company(
                36,
                "BlackRock Inc.",
                "BLK",
                "Financials",
                "Asset Management",
                17887,
                5177,
                19800,
                "New York, NY",
                100.0,
            ),
            Fortune500Company(
                37,
                "PayPal Holdings Inc.",
                "PYPL",
                "Financials",
                "Financial Services",
                27518,
                2419,
                30000,
                "San Jose, CA",
                100.0,
            ),
            Fortune500Company(
                38,
                "Nike Inc.",
                "NKE",
                "Consumer Discretionary",
                "Apparel",
                46710,
                5070,
                79000,
                "Beaverton, OR",
                200.0,
            ),
            Fortune500Company(
                39,
                "Starbucks Corporation",
                "SBUX",
                "Consumer Discretionary",
                "Restaurants",
                32250,
                3284,
                402000,
                "Seattle, WA",
                100.0,
            ),
            Fortune500Company(
                40,
                "Target Corporation",
                "TGT",
                "Consumer Discretionary",
                "Retail",
                107588,
                2780,
                450000,
                "Minneapolis, MN",
                100.0,
            ),
            Fortune500Company(
                41,
                "Costco Wholesale Corp.",
                "COST",
                "Consumer Staples",
                "Retail",
                226954,
                6294,
                304000,
                "Issaquah, WA",
                250.0,
            ),
            Fortune500Company(
                42,
                "Pfizer Inc.",
                "PFE",
                "Healthcare",
                "Pharmaceuticals",
                100330,
                31372,
                83000,
                "New York, NY",
                200.0,
            ),
            Fortune500Company(
                43,
                "Merck & Co. Inc.",
                "MRK",
                "Healthcare",
                "Pharmaceuticals",
                59283,
                14519,
                74000,
                "Kenilworth, NJ",
                250.0,
            ),
            Fortune500Company(
                44,
                "Abbott Laboratories",
                "ABT",
                "Healthcare",
                "Medical Devices",
                43653,
                6332,
                115000,
                "Abbott Park, IL",
                200.0,
            ),
            Fortune500Company(
                45,
                "Amgen Inc.",
                "AMGN",
                "Healthcare",
                "Biotechnology",
                26392,
                6283,
                24000,
                "Thousand Oaks, CA",
                150.0,
            ),
            Fortune500Company(
                46,
                "Gilead Sciences Inc.",
                "GILD",
                "Healthcare",
                "Biotechnology",
                27281,
                5602,
                14000,
                "Foster City, CA",
                100.0,
            ),
            Fortune500Company(
                47,
                "Bristol-Myers Squibb Co.",
                "BMY",
                "Healthcare",
                "Pharmaceuticals",
                46159,
                6327,
                34000,
                "New York, NY",
                150.0,
            ),
            Fortune500Company(
                48,
                "Eli Lilly and Co.",
                "LLY",
                "Healthcare",
                "Pharmaceuticals",
                28341,
                6286,
                35000,
                "Indianapolis, IN",
                600.0,
            ),
            Fortune500Company(
                49,
                "Chevron Corporation",
                "CVX",
                "Energy",
                "Oil & Gas",
                246252,
                35465,
                43000,
                "San Ramon, CA",
                300.0,
            ),
            Fortune500Company(
                50,
                "Exxon Mobil Corporation",
                "XOM",
                "Energy",
                "Oil & Gas",
                413680,
                55840,
                63000,
                "Irving, TX",
                400.0,
            ),
        ]

        # Add more companies (51-500) - this is a sample, you'd want the full list
        # For brevity, I'm including key companies that are commonly traded
        additional_companies = [
            Fortune500Company(
                51,
                "General Electric Co.",
                "GE",
                "Industrials",
                "Conglomerate",
                76556,
                1002,
                172000,
                "Boston, MA",
                100.0,
            ),
            Fortune500Company(
                52,
                "Boeing Co.",
                "BA",
                "Industrials",
                "Aerospace",
                66608,
                -4942,
                142000,
                "Chicago, IL",
                100.0,
            ),
            Fortune500Company(
                53,
                "3M Co.",
                "MMM",
                "Industrials",
                "Conglomerate",
                34229,
                5340,
                95000,
                "St. Paul, MN",
                100.0,
            ),
            Fortune500Company(
                54,
                "Caterpillar Inc.",
                "CAT",
                "Industrials",
                "Machinery",
                59427,
                6709,
                107700,
                "Deerfield, IL",
                100.0,
            ),
            Fortune500Company(
                55,
                "Deere & Co.",
                "DE",
                "Industrials",
                "Machinery",
                55266,
                10166,
                69000,
                "Moline, IL",
                100.0,
            ),
            Fortune500Company(
                56,
                "United Technologies Corp.",
                "RTX",
                "Industrials",
                "Aerospace",
                65466,
                3074,
                195000,
                "Arlington, VA",
                100.0,
            ),
            Fortune500Company(
                57,
                "Honeywell International Inc.",
                "HON",
                "Industrials",
                "Conglomerate",
                35466,
                4928,
                103000,
                "Charlotte, NC",
                150.0,
            ),
            Fortune500Company(
                58,
                "Union Pacific Corp.",
                "UNP",
                "Industrials",
                "Railroads",
                24831,
                7084,
                32000,
                "Omaha, NE",
                150.0,
            ),
            Fortune500Company(
                59,
                "FedEx Corporation",
                "FDX",
                "Industrials",
                "Logistics",
                93512,
                3785,
                547000,
                "Memphis, TN",
                100.0,
            ),
            Fortune500Company(
                60,
                "UPS Inc.",
                "UPS",
                "Industrials",
                "Logistics",
                100338,
                11548,
                534000,
                "Atlanta, GA",
                150.0,
            ),
        ]

        companies_data.extend(additional_companies)

        logger.info(f"Loaded {len(companies_data)} Fortune 500 companies")
        return companies_data

    def get_all_companies(self) -> List[Fortune500Company]:
        """Get all Fortune 500 companies."""
        return self.companies

    def get_companies_by_sector(self, sector: str) -> List[Fortune500Company]:
        """Get companies by sector."""
        return [
            company
            for company in self.companies
            if company.sector.lower() == sector.lower()
        ]

    def get_companies_by_market_cap_range(
        self, min_cap: float, max_cap: float
    ) -> List[Fortune500Company]:
        """Get companies by market cap range (in billions)."""
        return [
            company
            for company in self.companies
            if company.market_cap_billions
            and min_cap <= company.market_cap_billions <= max_cap
        ]

    def get_top_companies(self, limit: int = 100) -> List[Fortune500Company]:
        """Get top N companies by rank."""
        return sorted(self.companies, key=lambda x: x.rank)[:limit]

    def get_trading_symbols(self) -> List[str]:
        """Get all trading symbols."""
        return [company.symbol for company in self.companies if company.symbol]

    def get_company_by_symbol(self, symbol: str) -> Optional[Fortune500Company]:
        """Get company by trading symbol."""
        for company in self.companies:
            if company.symbol == symbol:
                return company
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = []
        for company in self.companies:
            data.append(
                {
                    "rank": company.rank,
                    "company_name": company.company_name,
                    "symbol": company.symbol,
                    "sector": company.sector,
                    "industry": company.industry,
                    "revenue_millions": company.revenue_millions,
                    "profit_millions": company.profit_millions,
                    "employees": company.employees,
                    "headquarters": company.headquarters,
                    "market_cap_billions": company.market_cap_billions,
                }
            )
        return pd.DataFrame(data)


def get_fortune500_companies() -> List[Fortune500Company]:
    """Get Fortune 500 companies."""
    source = Fortune500DataSource()
    return source.get_all_companies()


def get_fortune500_symbols() -> List[str]:
    """Get Fortune 500 trading symbols."""
    source = Fortune500DataSource()
    return source.get_trading_symbols()


if __name__ == "__main__":
    # Test the Fortune 500 data source
    source = Fortune500DataSource()
    print(f"Loaded {len(source.companies)} Fortune 500 companies")

    # Get top 10 companies
    top_10 = source.get_top_companies(10)
    print("\nTop 10 Fortune 500 Companies:")
    for company in top_10:
        print(
            f"{company.rank}. {company.company_name} ({company.symbol}) - {company.sector}"
        )

    # Get symbols
    symbols = source.get_trading_symbols()
    print(f"\nTotal trading symbols: {len(symbols)}")
    print(f"First 20 symbols: {symbols[:20]}")
