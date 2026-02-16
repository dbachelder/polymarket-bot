"""Combinatorial arbitrage (Dutch book) strategy for prediction markets.

This module implements cross-market combinatorial arbitrage where mutually
exclusive outcomes across different markets sum to less than $1.00 minus fees.

Example: In a winner-take-all election with candidates A, B, C:
- Market A: "Will A win?" YES = $0.30
- Market B: "Will B win?" YES = $0.25
- Market C: "Will C win?" YES = $0.20
Sum = $0.75 → Buy all three for guaranteed $0.25 profit (before fees)

Key insight: Only ONE candidate can win, so exactly one position pays $1.
If sum < $1, guaranteed profit exists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .gamma import get_events
from .clob import get_book

logger = logging.getLogger(__name__)

# Strategy parameters
DEFAULT_FEE_RATE = 0.0315  # Polymarket settlement fee
DEFAULT_MIN_EDGE_AFTER_FEES = 0.015  # 1.5% minimum profit
DEFAULT_MAX_BASKET_SIZE = 4  # Complexity limit
DEFAULT_MIN_LIQUIDITY = 100.0  # $100 minimum per outcome


@dataclass(frozen=True)
class BasketOutcome:
    """A single outcome in a Dutch book basket.

    Represents one leg of the arbitrage - buying YES on a specific market.
    """

    market_id: str
    market_slug: str
    market_question: str
    token_id_yes: str
    best_ask_yes: float  # Price to buy YES (what we pay)
    best_bid_yes: float  # Price to sell YES
    liquidity: float  # Available liquidity
    outcome_index: int  # Position in basket

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "token_id_yes": self.token_id_yes,
            "best_ask_yes": self.best_ask_yes,
            "best_bid_yes": self.best_bid_yes,
            "liquidity": self.liquidity,
            "outcome_index": self.outcome_index,
        }


@dataclass(frozen=True)
class DutchBookBasket:
    """A basket of mutually exclusive outcomes forming a Dutch book.

    Attributes:
        basket_id: Unique identifier for this basket
        event_id: Parent event ID
        event_title: Human-readable event title
        relationship_type: Type of relationship (e.g., "winner_take_all")
        outcomes: List of BasketOutcome (the legs)
        sum_best_ask: Sum of best ask prices (total cost)
        fee_rate: Settlement fee rate
        min_edge_after_fees: Minimum profit margin required
        timestamp: When basket was analyzed
        notes: Additional context about the basket
    """

    basket_id: str
    event_id: str
    event_title: str
    relationship_type: str
    outcomes: list[BasketOutcome]
    sum_best_ask: float
    fee_rate: float
    min_edge_after_fees: float
    timestamp: datetime
    notes: str = ""

    @property
    def gross_profit(self) -> float:
        """Profit before fees: $1 - sum(costs)."""
        return 1.0 - self.sum_best_ask

    @property
    def settlement_fees(self) -> float:
        """Total settlement fees (only winner pays)."""
        # Only the winning position pays settlement fee
        # We assume 1 contract per position, winner gets $1
        return self.fee_rate * 1.0

    @property
    def net_profit(self) -> float:
        """Profit after settlement fees."""
        return self.gross_profit - self.settlement_fees

    @property
    def net_edge_percent(self) -> float:
        """Net profit as percentage of capital deployed."""
        if self.sum_best_ask <= 0:
            return 0.0
        return (self.net_profit / self.sum_best_ask) * 100

    @property
    def is_profitable(self) -> bool:
        """True if net profit meets minimum edge threshold."""
        return self.net_profit >= self.min_edge_after_fees

    @property
    def min_liquidity(self) -> float:
        """Minimum liquidity across all outcomes."""
        if not self.outcomes:
            return 0.0
        return min(o.liquidity for o in self.outcomes)

    @property
    def outcome_count(self) -> int:
        """Number of outcomes in basket."""
        return len(self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "basket_id": self.basket_id,
            "event_id": self.event_id,
            "event_title": self.event_title,
            "relationship_type": self.relationship_type,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "sum_best_ask": self.sum_best_ask,
            "gross_profit": self.gross_profit,
            "settlement_fees": self.settlement_fees,
            "net_profit": self.net_profit,
            "net_edge_percent": self.net_edge_percent,
            "is_profitable": self.is_profitable,
            "min_liquidity": self.min_liquidity,
            "outcome_count": self.outcome_count,
            "fee_rate": self.fee_rate,
            "min_edge_after_fees": self.min_edge_after_fees,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CombinatorialScanResult:
    """Result of a combinatorial arbitrage scan."""

    timestamp: datetime
    events_scanned: int
    baskets_constructed: int
    opportunities_found: int
    baskets: list[DutchBookBasket]
    profitable_baskets: list[DutchBookBasket]
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "events_scanned": self.events_scanned,
            "baskets_constructed": self.baskets_constructed,
            "opportunities_found": self.opportunities_found,
            "baskets": [b.to_dict() for b in self.baskets],
            "profitable_baskets": [b.to_dict() for b in self.profitable_baskets],
            "parameters": self.parameters,
        }


# Manual basket definitions for Phase 1
# These are curated event relationships known to be mutually exclusive
MANUAL_BASKET_DEFINITIONS: list[dict[str, Any]] = [
    {
        "event_slug_starts_with": "2026-fifa-world-cup-winner",
        "relationship_type": "winner_take_all",
        "description": "2026 FIFA World Cup - exactly one team wins",
        "max_outcomes": 10,  # Top 10 by liquidity for initial scan
    },
    {
        "event_slug_starts_with": "democratic-presidential-nominee-2028",
        "relationship_type": "nomination_winner",
        "description": "2028 Democratic Nominee - exactly one candidate wins",
        "max_outcomes": 10,
    },
    {
        "event_slug_starts_with": "republican-presidential-nominee-2028",
        "relationship_type": "nomination_winner",
        "description": "2028 Republican Nominee - exactly one candidate wins",
        "max_outcomes": 10,
    },
    {
        "event_slug_starts_with": "presidential-election-winner-2028",
        "relationship_type": "election_winner",
        "description": "2028 US Presidential Election - exactly one candidate wins",
        "max_outcomes": 10,
    },
    {
        "event_slug_starts_with": "texas-republican-senate-primary-winner",
        "relationship_type": "primary_winner",
        "description": "Texas Republican Senate Primary - exactly one candidate wins",
        "max_outcomes": 8,
    },
    {
        "event_slug_starts_with": "uefa-champions-league-winner",
        "relationship_type": "winner_take_all",
        "description": "2025-26 Champions League - exactly one team wins",
        "max_outcomes": 10,
    },
    {
        "event_slug_starts_with": "english-premier-league-winner",
        "relationship_type": "winner_take_all",
        "description": "2025-26 EPL - exactly one team wins",
        "max_outcomes": 8,
    },
    {
        "event_slug_starts_with": "la-liga-winner",
        "relationship_type": "winner_take_all",
        "description": "2025-26 La Liga - exactly one team wins",
        "max_outcomes": 8,
    },
    {
        "event_slug_starts_with": "bundesliga-winner",
        "relationship_type": "winner_take_all",
        "description": "2025-26 Bundesliga - exactly one team wins",
        "max_outcomes": 8,
    },
    {
        "event_slug_starts_with": "the-masters-winner",
        "relationship_type": "winner_take_all",
        "description": "2026 Masters - exactly one golfer wins",
        "max_outcomes": 10,
    },
]


def _get_best_asks(token_ids: list[str]) -> dict[str, dict[str, float]]:
    """Fetch best ask prices for a list of token IDs.

    Returns dict mapping token_id -> {"ask": float, "bid": float, "liquidity": float}
    """
    from polymarket.clob import get_best_prices

    results = {}
    for token_id in token_ids:
        try:
            book = get_book(token_id)
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            best_bid, best_ask = get_best_prices(book)

            # Estimate liquidity at best prices (sum top 3 levels)
            # Sort appropriately: bids desc, asks asc
            sorted_bids = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
            sorted_asks = sorted(asks, key=lambda x: float(x["price"]))
            ask_liquidity = sum(float(a["size"]) for a in sorted_asks[:3]) if asks else 0
            bid_liquidity = sum(float(b["size"]) for b in sorted_bids[:3]) if bids else 0
            liquidity = min(ask_liquidity, bid_liquidity)

            results[token_id] = {
                "ask": best_ask,
                "bid": best_bid,
                "liquidity": liquidity,
            }
        except Exception as e:
            logger.debug(f"Failed to fetch book for {token_id}: {e}")
            results[token_id] = {"ask": None, "bid": None, "liquidity": 0}

    return results


def _parse_outcome_prices(outcome_prices: str) -> list[float]:
    """Parse outcome prices from JSON string."""
    try:
        prices = json.loads(outcome_prices)
        return [float(p) for p in prices]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def build_winner_take_all_basket(
    event: dict[str, Any],
    relationship_type: str = "winner_take_all",
    max_outcomes: int = 10,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_edge: float = DEFAULT_MIN_EDGE_AFTER_FEES,
    min_liquidity: float = DEFAULT_MIN_LIQUIDITY,
) -> DutchBookBasket | None:
    """Build a Dutch book basket from a winner-take-all event.

    Args:
        event: Event dict from Gamma API
        relationship_type: Type of relationship
        max_outcomes: Maximum number of outcomes to include
        fee_rate: Settlement fee rate
        min_edge: Minimum edge required
        min_liquidity: Minimum liquidity per outcome

    Returns:
        DutchBookBasket if valid basket can be constructed, None otherwise
    """
    event_id = str(event.get("id", ""))
    event_title = event.get("title", "Unknown")
    markets = event.get("markets", [])

    if len(markets) < 2:
        return None

    # Sort markets by liquidity (descending) and take top N
    sorted_markets = sorted(
        markets,
        key=lambda m: float(m.get("liquidityNum", 0) or 0),
        reverse=True,
    )[:max_outcomes]

    # Collect token IDs for batch price fetch
    token_ids = []
    market_data = []

    for m in sorted_markets:
        clob_token_ids = m.get("clobTokenIds", "")
        if not clob_token_ids:
            continue
        try:
            tokens = json.loads(clob_token_ids)
            if len(tokens) >= 1:
                # tokens[0] is YES, tokens[1] is NO
                yes_token = str(tokens[0])
                token_ids.append(yes_token)
                market_data.append(
                    {
                        "market_id": str(m.get("id", "")),
                        "slug": m.get("slug", ""),
                        "question": m.get("question", ""),
                        "yes_token": yes_token,
                        "liquidity": float(m.get("liquidityNum", 0) or 0),
                    }
                )
        except (json.JSONDecodeError, IndexError):
            continue

    if len(market_data) < 2:
        return None

    # Fetch prices
    prices = _get_best_asks(token_ids)

    # Build outcomes
    outcomes = []
    for i, md in enumerate(market_data):
        price_info = prices.get(md["yes_token"], {})
        best_ask = price_info.get("ask")
        best_bid = price_info.get("bid")
        liquidity = price_info.get("liquidity", 0)

        if best_ask is None:
            continue

        # Use max of estimated liquidity and reported liquidity
        effective_liquidity = max(liquidity, md["liquidity"])

        if effective_liquidity < min_liquidity:
            continue

        outcomes.append(
            BasketOutcome(
                market_id=md["market_id"],
                market_slug=md["slug"],
                market_question=md["question"],
                token_id_yes=md["yes_token"],
                best_ask_yes=best_ask,
                best_bid_yes=best_bid or 0.0,
                liquidity=effective_liquidity,
                outcome_index=i,
            )
        )

    if len(outcomes) < 2:
        return None

    # Calculate basket metrics
    sum_best_ask = sum(o.best_ask_yes for o in outcomes)
    basket_id = f"{event_id}_{relationship_type}"

    return DutchBookBasket(
        basket_id=basket_id,
        event_id=event_id,
        event_title=event_title,
        relationship_type=relationship_type,
        outcomes=outcomes,
        sum_best_ask=sum_best_ask,
        fee_rate=fee_rate,
        min_edge_after_fees=min_edge,
        timestamp=datetime.now(UTC),
        notes=f"Manual basket: {len(outcomes)} outcomes",
    )


def scan_combinatorial_opportunities(
    event_limit: int = 100,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_edge_after_fees: float = DEFAULT_MIN_EDGE_AFTER_FEES,
    max_basket_size: int = DEFAULT_MAX_BASKET_SIZE,
    min_liquidity: float = DEFAULT_MIN_LIQUIDITY,
    use_manual_definitions: bool = True,
) -> CombinatorialScanResult:
    """Scan for combinatorial arbitrage opportunities.

    Phase 1 implementation: Uses manual basket definitions to identify
    winner-take-all events and construct Dutch book baskets.

    Args:
        event_limit: Maximum events to fetch from API
        fee_rate: Settlement fee rate
        min_edge_after_fees: Minimum profit margin after fees
        max_basket_size: Maximum outcomes per basket
        min_liquidity: Minimum liquidity per outcome
        use_manual_definitions: If True, use curated basket definitions

    Returns:
        CombinatorialScanResult with all baskets and opportunities
    """
    timestamp = datetime.now(UTC)
    logger.info("Starting combinatorial arbitrage scan...")

    # Fetch events
    events = get_events(active=True, limit=event_limit)
    logger.info(f"Fetched {len(events)} active events")

    baskets: list[DutchBookBasket] = []

    if use_manual_definitions:
        # Use manual definitions to find candidate events
        for definition in MANUAL_BASKET_DEFINITIONS:
            slug_prefix = definition["event_slug_starts_with"]
            relationship_type = definition["relationship_type"]
            max_outcomes = min(definition.get("max_outcomes", 10), max_basket_size)

            # Find matching events
            for event in events:
                event_slug = event.get("slug", "")
                if event_slug.startswith(slug_prefix):
                    logger.info(f"Found event: {event.get('title')} ({relationship_type})")

                    basket = build_winner_take_all_basket(
                        event=event,
                        relationship_type=relationship_type,
                        max_outcomes=max_outcomes,
                        fee_rate=fee_rate,
                        min_edge=min_edge_after_fees,
                        min_liquidity=min_liquidity,
                    )

                    if basket:
                        baskets.append(basket)
                        logger.info(
                            f"  Basket: {basket.outcome_count} outcomes, "
                            f"sum={basket.sum_best_ask:.4f}, "
                            f"net_profit={basket.net_profit:.4f}"
                        )

    # Filter profitable baskets
    profitable_baskets = [b for b in baskets if b.is_profitable]
    profitable_baskets.sort(key=lambda b: b.net_profit, reverse=True)

    logger.info(f"Scan complete: {len(baskets)} baskets, {len(profitable_baskets)} profitable")

    return CombinatorialScanResult(
        timestamp=timestamp,
        events_scanned=len(events),
        baskets_constructed=len(baskets),
        opportunities_found=len(profitable_baskets),
        baskets=baskets,
        profitable_baskets=profitable_baskets,
        parameters={
            "fee_rate": fee_rate,
            "min_edge_after_fees": min_edge_after_fees,
            "max_basket_size": max_basket_size,
            "min_liquidity": min_liquidity,
        },
    )


def format_scan_report(result: CombinatorialScanResult, detailed: bool = False) -> str:
    """Format scan results as human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMBINATORIAL ARBITRAGE SCAN REPORT")
    lines.append("=" * 80)
    lines.append(f"Scan time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"Events scanned: {result.events_scanned}")
    lines.append(f"Baskets constructed: {result.baskets_constructed}")
    lines.append(f"Profitable opportunities: {result.opportunities_found}")
    lines.append("")

    # Parameters
    lines.append("--- Parameters ---")
    for key, value in result.parameters.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    # All baskets summary
    lines.append("--- All Baskets ---")
    for basket in result.baskets:
        status = "✓ PROFITABLE" if basket.is_profitable else "✗ No edge"
        lines.append(
            f"  {basket.event_title[:50]:<50} | "
            f"sum={basket.sum_best_ask:.4f} | "
            f"net={basket.net_profit:+.4f} | "
            f"edge={basket.net_edge_percent:+.2f}% | "
            f"{status}"
        )

    if result.profitable_baskets:
        lines.append("")
        lines.append("=" * 80)
        lines.append("PROFITABLE OPPORTUNITIES")
        lines.append("=" * 80)

        for i, basket in enumerate(result.profitable_baskets, 1):
            lines.append(f"\n{i}. {basket.event_title}")
            lines.append(f"   Type: {basket.relationship_type}")
            lines.append(f"   Outcomes: {basket.outcome_count}")
            lines.append(f"   Sum of asks: ${basket.sum_best_ask:.4f}")
            lines.append(f"   Gross profit: ${basket.gross_profit:.4f}")
            lines.append(f"   Settlement fees: ${basket.settlement_fees:.4f}")
            lines.append(f"   Net profit: ${basket.net_profit:.4f}")
            lines.append(f"   Net edge: {basket.net_edge_percent:+.2f}%")
            lines.append(f"   Min liquidity: ${basket.min_liquidity:.2f}")

            if detailed:
                lines.append("   Outcomes:")
                for outcome in basket.outcomes:
                    lines.append(
                        f"     - {outcome.market_question[:60]:<60} "
                        f"ask={outcome.best_ask_yes:.4f} "
                        f"liq=${outcome.liquidity:.0f}"
                    )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def run_combinatorial_scan(
    event_limit: int = 100,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_edge: float = DEFAULT_MIN_EDGE_AFTER_FEES,
    max_basket_size: int = DEFAULT_MAX_BASKET_SIZE,
    min_liquidity: float = DEFAULT_MIN_LIQUIDITY,
    detailed: bool = False,
) -> dict[str, Any]:
    """Convenience function to run scan and return results.

    Args:
        event_limit: Maximum events to fetch
        fee_rate: Settlement fee rate
        min_edge: Minimum edge after fees
        max_basket_size: Maximum outcomes per basket
        min_liquidity: Minimum liquidity per outcome
        detailed: Include detailed output in report

    Returns:
        Dictionary with full results and formatted report
    """
    result = scan_combinatorial_opportunities(
        event_limit=event_limit,
        fee_rate=fee_rate,
        min_edge_after_fees=min_edge,
        max_basket_size=max_basket_size,
        min_liquidity=min_liquidity,
    )

    return {
        "result": result.to_dict(),
        "report": format_scan_report(result, detailed=detailed),
        "opportunity_count": len(result.profitable_baskets),
    }
