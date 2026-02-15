"""Cross-market arbitrage strategy.

Monitors identical events across Polymarket, Kalshi, and other venues
to identify arbitrage opportunities where YES+NO prices sum to <100c.

Hypothesis: Cross-platform prediction market arbitrage yields 1-3%
per trade with ~68% APY potential, while delta-neutral volume qualifies
for airdrop farming across multiple platforms.

Entry trigger: YES+NO prices sum to <100c across two venues (after fees)
Exit: Close spread converges OR hold to resolution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import ArbitrageOpportunity
from .calculator import SpreadCalculator, quick_spread_check
from .kalshi_client import KalshiClient
from .matcher import EventMatcher
from .polymarket_client import PolymarketClient
from .tracker import PaperTradeTracker

logger = logging.getLogger(__name__)

# Strategy configuration
DEFAULT_MIN_GROSS_SPREAD = 0.01  # 1% minimum gross spread
DEFAULT_MIN_NET_SPREAD = 0.005  # 0.5% minimum net spread after fees
DEFAULT_MAX_POSITIONS = 10  # Maximum open positions
DEFAULT_POSITION_SIZE = 1.0  # Contracts per side


@dataclass(frozen=True)
class CrossMarketScanResult:
    """Result of a cross-market arbitrage scan.

    Attributes:
        timestamp: When scan was performed
        polymarket_markets: Number of Polymarket markets scanned
        kalshi_markets: Number of Kalshi markets scanned
        matched_events: Number of matched cross-venue events
        opportunities: Number of arbitrage opportunities found
        opportunities_list: List of ArbitrageOpportunity
        trades_entered: Number of new paper trades entered
        dry_run: Whether this was a dry run
    """

    timestamp: datetime
    polymarket_markets: int
    kalshi_markets: int
    matched_events: int
    opportunities: int
    opportunities_list: list[ArbitrageOpportunity]
    trades_entered: int
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "polymarket_markets": self.polymarket_markets,
            "kalshi_markets": self.kalshi_markets,
            "matched_events": self.matched_events,
            "opportunities": self.opportunities,
            "opportunities_list": [o.to_dict() for o in self.opportunities_list],
            "trades_entered": self.trades_entered,
            "dry_run": self.dry_run,
        }


class CrossMarketArbitrage:
    """Cross-market arbitrage strategy implementation."""

    def __init__(
        self,
        min_gross_spread: float = DEFAULT_MIN_GROSS_SPREAD,
        min_net_spread: float = DEFAULT_MIN_NET_SPREAD,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        position_size: float = DEFAULT_POSITION_SIZE,
        data_dir: Path | None = None,
        kalshi_api_key: str | None = None,
        kalshi_api_secret: str | None = None,
    ) -> None:
        """Initialize strategy.

        Args:
            min_gross_spread: Minimum gross spread (before fees)
            min_net_spread: Minimum net spread (after fees)
            max_positions: Maximum concurrent open positions
            position_size: Contracts per side for each trade
            data_dir: Directory for data storage
            kalshi_api_key: Optional Kalshi API key
            kalshi_api_secret: Optional Kalshi API secret
        """
        self.min_gross_spread = min_gross_spread
        self.min_net_spread = min_net_spread
        self.max_positions = max_positions
        self.position_size = position_size

        # Initialize clients
        self.polymarket = PolymarketClient()
        self.kalshi = KalshiClient(
            api_key=kalshi_api_key,
            api_secret=kalshi_api_secret,
        )

        # Initialize matcher and calculator
        self.matcher = EventMatcher()
        self.calculator = SpreadCalculator(
            min_gross_spread=min_gross_spread,
            min_net_spread=min_net_spread,
        )

        # Initialize tracker
        self.tracker = PaperTradeTracker(data_dir=data_dir)

    def scan(
        self,
        categories: list[str] | None = None,
        dry_run: bool = True,
    ) -> CrossMarketScanResult:
        """Run a cross-market arbitrage scan.

        Args:
            categories: Categories to scan (politics, crypto, sports, etc.)
            dry_run: If True, don't actually enter trades

        Returns:
            CrossMarketScanResult with scan results
        """
        now = datetime.now(UTC)
        logger.info("Starting cross-market arbitrage scan at %s", now.isoformat())

        # Step 1: Fetch markets from all venues
        logger.info("Fetching markets from Polymarket...")
        pm_markets = self.polymarket.fetch_active_markets(
            limit=100,
            categories=categories,
        )

        logger.info("Fetching markets from Kalshi...")
        kalshi_markets = self.kalshi.fetch_active_markets(
            limit=100,
            categories=categories,
        )

        # Step 2: Match equivalent events
        logger.info("Matching events across venues...")
        venues_data = {
            "polymarket": pm_markets,
            "kalshi": kalshi_markets,
        }
        matched_events = self.matcher.match_events(venues_data)

        # Step 3: Calculate spreads
        logger.info("Calculating arbitrage spreads...")
        opportunities = self.calculator.calculate_all_opportunities(matched_events)

        # Step 4: Enter paper trades (respecting limits)
        trades_entered = 0
        open_positions = len(self.tracker.get_open_positions())

        for opp in opportunities:
            if open_positions >= self.max_positions:
                logger.info(
                    "Max positions reached (%d), skipping remaining opportunities",
                    self.max_positions,
                )
                break

            if not dry_run:
                # In live mode, would check balance and execute trades
                logger.warning("Live trading not yet implemented, using paper trade")

            # Enter paper trade
            self.tracker.enter_position(opp, self.position_size)
            trades_entered += 1
            open_positions += 1

            logger.info(
                "Entered trade: %s YES @ %.3f + %s NO @ %.3f (net spread: %.3f%%)",
                opp.venue_yes,
                opp.yes_price,
                opp.venue_no,
                opp.no_price,
                opp.net_spread * 100,
            )

        # Update tracker
        self.tracker.update_open_positions()

        result = CrossMarketScanResult(
            timestamp=now,
            polymarket_markets=len(pm_markets),
            kalshi_markets=len(kalshi_markets),
            matched_events=len(matched_events),
            opportunities=len(opportunities),
            opportunities_list=opportunities,
            trades_entered=trades_entered,
            dry_run=dry_run,
        )

        logger.info(
            "Scan complete: %d markets, %d matched events, %d opportunities, %d trades entered",
            len(pm_markets) + len(kalshi_markets),
            len(matched_events),
            len(opportunities),
            trades_entered,
        )

        return result

    def get_performance_report(self) -> dict[str, Any]:
        """Get current performance report.

        Returns:
            Performance summary dictionary
        """
        return self.tracker.get_performance_summary()

    def export_trades(self, out_path: Path | None = None) -> Path:
        """Export all trades to file.

        Args:
            out_path: Output file path

        Returns:
            Path to exported file
        """
        return self.tracker.export_trades(out_path)

    def close(self) -> None:
        """Close all clients."""
        self.polymarket.close()
        self.kalshi.close()

    def __enter__(self) -> CrossMarketArbitrage:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def run_cross_market_scan(
    data_dir: Path | None = None,
    categories: list[str] | None = None,
    min_spread: float = 0.005,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Convenience function to run a cross-market scan.

    Args:
        data_dir: Data directory
        categories: Categories to scan
        min_spread: Minimum net spread
        dry_run: If True, don't enter live trades

    Returns:
        Scan results dictionary
    """
    with CrossMarketArbitrage(
        min_net_spread=min_spread,
        data_dir=data_dir,
    ) as strategy:
        result = strategy.scan(categories=categories, dry_run=dry_run)
        return result.to_dict()


def check_price_pair(
    yes_price: float,
    no_price: float,
) -> dict[str, Any]:
    """Quick check if a price pair represents an arbitrage opportunity.

    Args:
        yes_price: Price to buy YES
        no_price: Price to buy NO

    Returns:
        Spread analysis dictionary
    """
    return quick_spread_check(yes_price, no_price)
