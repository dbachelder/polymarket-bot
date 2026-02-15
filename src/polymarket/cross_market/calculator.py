"""Spread calculator for cross-market arbitrage.

Calculates arbitrage opportunities including fees and provides
PnL estimates for paper trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from . import (
    ArbitrageOpportunity,
    CrossMarketEvent,
    FeeSchedule,
    KALSHI_FEE_SCHEDULE,
    POLYMARKET_FEE_SCHEDULE,
    VenueMarket,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpreadCalculation:
    """Detailed spread calculation for an arbitrage opportunity.

    Attributes:
        buy_yes_venue: Venue to buy YES
        buy_no_venue: Venue to buy NO
        buy_yes_price: Price to buy YES (ask)
        buy_no_price: Price to buy NO (ask)
        sum_cost: Total cost to buy both sides
        gross_profit: Profit before fees (1 - sum_cost)
        yes_fees: Fees for YES position
        no_fees: Fees for NO position
        withdrawal_fees: Fees to withdraw from venues
        total_fees: Combined all fees
        net_profit: Profit after all fees
        roi: Return on investment
        min_liquidity: Minimum liquidity across both venues
        days_to_resolution: Estimated days until resolution
        annualized_apr: Estimated annualized return
    """

    buy_yes_venue: str
    buy_no_venue: str
    buy_yes_price: float
    buy_no_price: float
    sum_cost: float
    gross_profit: float
    yes_fees: float
    no_fees: float
    withdrawal_fees: float
    total_fees: float
    net_profit: float
    roi: float
    min_liquidity: float
    days_to_resolution: int | None
    annualized_apr: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "buy_yes_venue": self.buy_yes_venue,
            "buy_no_venue": self.buy_no_venue,
            "buy_yes_price": self.buy_yes_price,
            "buy_no_price": self.buy_no_venue,
            "sum_cost": self.sum_cost,
            "gross_profit": self.gross_profit,
            "yes_fees": self.yes_fees,
            "no_fees": self.no_fees,
            "withdrawal_fees": self.withdrawal_fees,
            "total_fees": self.total_fees,
            "net_profit": self.net_profit,
            "roi": self.roi,
            "min_liquidity": self.min_liquidity,
            "days_to_resolution": self.days_to_resolution,
            "annualized_apr": self.annualized_apr,
        }


class SpreadCalculator:
    """Calculates arbitrage spreads including all fees."""

    def __init__(
        self,
        min_gross_spread: float = 0.01,
        min_net_spread: float = 0.005,
        include_withdrawal_fees: bool = True,
    ) -> None:
        """Initialize calculator.

        Args:
            min_gross_spread: Minimum gross spread to consider (default 1%)
            min_net_spread: Minimum net spread after fees (default 0.5%)
            include_withdrawal_fees: Whether to include withdrawal fees
        """
        self.min_gross_spread = min_gross_spread
        self.min_net_spread = min_net_spread
        self.include_withdrawal_fees = include_withdrawal_fees

    def calculate_position_fees(
        self,
        position_size: float,
        entry_price: float,
        exit_price: float,
        fee_schedule: FeeSchedule,
    ) -> float:
        """Calculate total fees for a position.

        Args:
            position_size: Number of contracts
            entry_price: Entry price per contract
            exit_price: Exit price per contract
            fee_schedule: Fee schedule for the venue

        Returns:
            Total fees in dollars
        """
        entry_value = position_size * entry_price
        exit_value = position_size * exit_price

        # Trading fees (applied to both entry and exit)
        taker_fee = fee_schedule.taker_fee
        entry_fee = entry_value * taker_fee
        exit_fee = exit_value * taker_fee

        total_fee = entry_fee + exit_fee

        # Apply max fee cap if present
        if fee_schedule.max_fee is not None:
            total_fee = min(total_fee, fee_schedule.max_fee * position_size)

        # Withdrawal fee (applied once per withdrawal)
        if self.include_withdrawal_fees:
            total_fee += fee_schedule.withdrawal_fee

        return total_fee

    def calculate_spread(
        self,
        event: CrossMarketEvent,
    ) -> ArbitrageOpportunity | None:
        """Calculate arbitrage spread for a cross-market event.

        Args:
            event: CrossMarketEvent with venues

        Returns:
            ArbitrageOpportunity if spread exists, None otherwise
        """
        venues = list(event.venues.items())
        if len(venues) < 2:
            return None

        best_opportunity: ArbitrageOpportunity | None = None
        best_net_spread = 0.0

        # Try all venue pairs
        for i, (venue_a, market_a) in enumerate(venues):
            for venue_b, market_b in venues[i + 1:]:
                # Calculate both directions:
                # 1. Buy YES at A, Buy NO at B
                # 2. Buy YES at B, Buy NO at A

                for buy_yes_venue, buy_yes_market, buy_no_venue, buy_no_market in [
                    (venue_a, market_a, venue_b, market_b),
                    (venue_b, market_b, venue_a, market_a),
                ]:
                    opp = self._calculate_opportunity(
                        event=event,
                        buy_yes_venue=buy_yes_venue,
                        buy_yes_market=buy_yes_market,
                        buy_no_venue=buy_no_venue,
                        buy_no_market=buy_no_market,
                    )

                    if opp and opp.net_spread > best_net_spread:
                        best_opportunity = opp
                        best_net_spread = opp.net_spread

        return best_opportunity

    def _calculate_opportunity(
        self,
        event: CrossMarketEvent,
        buy_yes_venue: str,
        buy_yes_market: VenueMarket,
        buy_no_venue: str,
        buy_no_market: VenueMarket,
    ) -> ArbitrageOpportunity | None:
        """Calculate a specific arbitrage opportunity.

        Args:
            event: The cross-market event
            buy_yes_venue: Venue to buy YES
            buy_yes_market: YES market data
            buy_no_venue: Venue to buy NO
            buy_no_market: NO market data

        Returns:
            ArbitrageOpportunity or None
        """
        # Get ask prices (what we pay to buy)
        yes_price = buy_yes_market.yes_ask or buy_yes_market.yes_price
        no_price = buy_no_market.no_ask or buy_no_market.no_price

        if yes_price is None or no_price is None:
            return None

        # Calculate gross spread
        sum_cost = yes_price + no_price
        gross_spread = 1.0 - sum_cost

        # Check minimum gross spread
        if gross_spread < self.min_gross_spread:
            return None

        # Calculate fees
        yes_fees = self._estimate_fees(yes_price, buy_yes_market.fees)
        no_fees = self._estimate_fees(no_price, buy_no_market.fees)

        # Withdrawal fees (count once per venue)
        withdrawal_fees = 0.0
        venues_seen: set[str] = set()
        for venue, market in [(buy_yes_venue, buy_yes_market), (buy_no_venue, buy_no_market)]:
            if venue not in venues_seen and market.fees:
                withdrawal_fees += market.fees.withdrawal_fee
                venues_seen.add(venue)

        total_fees = yes_fees + no_fees + withdrawal_fees
        net_spread = gross_spread - total_fees

        # Check minimum net spread
        if net_spread < self.min_net_spread:
            return None

        # Calculate confidence based on liquidity and price recency
        min_liquidity = min(buy_yes_market.liquidity, buy_no_market.liquidity)
        liquidity_score = min(1.0, min_liquidity / 1000)  # Normalize to 1000

        # Recency score (assume fresh if no timestamp)
        recency_score = 1.0
        now = datetime.now(UTC)
        for market in [buy_yes_market, buy_no_market]:
            if market.last_updated:
                age_seconds = (now - market.last_updated).total_seconds()
                if age_seconds > 300:  # Older than 5 minutes
                    recency_score *= 0.8

        confidence = liquidity_score * recency_score

        return ArbitrageOpportunity(
            event=event,
            venue_yes=buy_yes_venue,
            venue_no=buy_no_venue,
            yes_price=yes_price,
            no_price=no_price,
            gross_spread=gross_spread,
            net_spread=net_spread,
            fees_yes=yes_fees,
            fees_no=no_fees,
            total_fees=total_fees,
            confidence=confidence,
            timestamp=now,
        )

    def _estimate_fees(self, price: float, fee_schedule: FeeSchedule | None) -> float:
        """Estimate fees for a single contract position.

        Args:
            price: Entry price
            fee_schedule: Fee schedule

        Returns:
            Estimated fees
        """
        if fee_schedule is None:
            return 0.0

        # Assume 1 contract position for estimation
        position_size = 1.0

        # Entry fee
        entry_fee = price * position_size * fee_schedule.taker_fee

        # Exit fee (assume exit at $1 for winning side, $0 for losing)
        # This is a simplification - actual exit depends on resolution
        exit_fee = 1.0 * position_size * fee_schedule.taker_fee

        total = entry_fee + exit_fee

        if fee_schedule.max_fee is not None:
            total = min(total, fee_schedule.max_fee * position_size)

        return total

    def calculate_all_opportunities(
        self,
        events: list[CrossMarketEvent],
    ) -> list[ArbitrageOpportunity]:
        """Calculate all arbitrage opportunities from a list of events.

        Args:
            events: List of cross-market events

        Returns:
            List of profitable ArbitrageOpportunity
        """
        opportunities: list[ArbitrageOpportunity] = []

        for event in events:
            try:
                opp = self.calculate_spread(event)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug("Error calculating spread for %s: %s", event.event_id, e)

        # Sort by net spread descending
        opportunities.sort(key=lambda o: o.net_spread, reverse=True)

        logger.info(
            "Found %d arbitrage opportunities from %d events",
            len(opportunities),
            len(events),
        )

        return opportunities


def quick_spread_check(
    yes_price: float,
    no_price: float,
    yes_fee_schedule: FeeSchedule | None = None,
    no_fee_schedule: FeeSchedule | None = None,
) -> dict[str, Any]:
    """Quick check if a price pair represents an arbitrage opportunity.

    Args:
        yes_price: Price to buy YES
        no_price: Price to buy NO
        yes_fee_schedule: Fee schedule for YES venue
        no_fee_schedule: Fee schedule for NO venue

    Returns:
        Dictionary with spread analysis
    """
    yes_fees = yes_fee_schedule or POLYMARKET_FEE_SCHEDULE
    no_fees = no_fee_schedule or KALSHI_FEE_SCHEDULE

    sum_cost = yes_price + no_price
    gross_spread = 1.0 - sum_cost

    # Estimate fees (1 contract each)
    yes_fee = yes_price * yes_fees.taker_fee + 1.0 * yes_fees.taker_fee
    no_fee = no_price * no_fees.taker_fee + 1.0 * no_fees.taker_fee

    if yes_fees.max_fee:
        yes_fee = min(yes_fee, yes_fees.max_fee)
    if no_fees.max_fee:
        no_fee = min(no_fee, no_fees.max_fee)

    total_fees = yes_fee + no_fee + yes_fees.withdrawal_fee + no_fees.withdrawal_fee
    net_spread = gross_spread - total_fees

    roi = net_spread / sum_cost if sum_cost > 0 else 0.0

    return {
        "sum_cost": sum_cost,
        "gross_spread": gross_spread,
        "gross_spread_pct": gross_spread * 100,
        "yes_fees": yes_fee,
        "no_fees": no_fee,
        "withdrawal_fees": yes_fees.withdrawal_fee + no_fees.withdrawal_fee,
        "total_fees": total_fees,
        "net_spread": net_spread,
        "net_spread_pct": net_spread * 100,
        "roi": roi,
        "is_profitable": net_spread > 0.005,  # > 0.5% net
    }
