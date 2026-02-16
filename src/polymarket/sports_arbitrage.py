"""Cross-platform sports arbitrage strategy.

Detects arbitrage opportunities between Polymarket sports markets
and sharp sportsbook lines (Pinnacle, Betfair, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .odds_api import SportsMarket
    from .sports_markets import PolymarketSportMarket


# Strategy constants
MIN_EDGE_PERCENT = Decimal("0.02")  # 2% minimum edge
PM_WITHDRAWAL_FEE = Decimal("0.02")  # 2% Polymarket withdrawal fee
MIN_MARKET_VOLUME = Decimal("10000")  # $10k minimum volume
MAX_HOLD_DAYS = 7  # Maximum hold time
MIN_HOURS_TO_EVENT = 2  # No trades within 2 hours
MAX_BANKROLL_PCT = Decimal("0.05")  # Max 5% bankroll per trade
KELLY_DIVISOR = 4  # Kelly/4 for conservative sizing


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """A detected arbitrage opportunity.

    Attributes:
        pm_market: Polymarket sport market
        sharp_market: Corresponding sharp book market
        side: "yes" or "no" - which side to buy on Polymarket
        pm_implied: Polymarket implied probability
        sharp_implied: Sharp book implied probability
        edge: Edge percentage (sharp - pm for yes, pm - sharp for no)
        edge_after_fees: Edge after accounting for withdrawal fees
        confidence: Confidence score (0-1)
        timestamp: Detection timestamp
    """

    pm_market: dict
    sharp_market: dict
    side: str
    pm_implied: Decimal
    sharp_implied: Decimal
    edge: Decimal
    edge_after_fees: Decimal
    confidence: Decimal
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pm_market": self.pm_market,
            "sharp_market": self.sharp_market,
            "side": self.side,
            "pm_implied": str(self.pm_implied),
            "sharp_implied": str(self.sharp_implied),
            "edge": str(self.edge),
            "edge_after_fees": str(self.edge_after_fees),
            "confidence": str(self.confidence),
            "timestamp": self.timestamp,
        }

    @property
    def is_valid(self) -> bool:
        """Check if opportunity meets all criteria."""
        return (
            self.edge_after_fees >= MIN_EDGE_PERCENT
            and self.confidence >= Decimal("0.5")
        )


@dataclass(frozen=True)
class PaperTrade:
    """A paper trade record.

    Attributes:
        trade_id: Unique trade ID
        timestamp: Entry timestamp
        pm_market_id: Polymarket market ID
        pm_token_id: Token ID traded
        side: "yes" or "no"
        size: Position size in contracts
        entry_price: Entry price
        sharp_implied_at_entry: Sharp book implied prob at entry
        edge_at_entry: Edge percentage at entry
        status: "open", "closed", or "expired"
        exit_price: Exit price (if closed)
        exit_timestamp: Exit timestamp
        pnl: Profit/loss
        close_reason: "resolution", "edge_compression", "stop_loss"
    """

    trade_id: str
    timestamp: str
    pm_market_id: str
    pm_token_id: str
    side: str
    size: Decimal
    entry_price: Decimal
    sharp_implied_at_entry: Decimal
    edge_at_entry: Decimal
    status: str = "open"
    exit_price: Decimal | None = None
    exit_timestamp: str | None = None
    pnl: Decimal | None = None
    close_reason: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "pm_market_id": self.pm_market_id,
            "pm_token_id": self.pm_token_id,
            "side": self.side,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "sharp_implied_at_entry": str(self.sharp_implied_at_entry),
            "edge_at_entry": str(self.edge_at_entry),
            "status": self.status,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_timestamp": self.exit_timestamp,
            "pnl": str(self.pnl) if self.pnl else None,
            "close_reason": self.close_reason,
        }


def calculate_edge(
    pm_implied: Decimal,
    sharp_implied: Decimal,
    side: str,
) -> Decimal:
    """Calculate edge percentage for a trade side.

    Args:
        pm_implied: Polymarket implied probability
        sharp_implied: Sharp book implied probability
        side: "yes" or "no"

    Returns:
        Edge as decimal (e.g., 0.03 for 3%).
    """
    if side == "yes":
        # Buy YES when PM is cheaper than sharp book
        return sharp_implied - pm_implied
    else:
        # Buy NO when PM NO is cheaper than sharp NO
        # NO implied = 1 - YES implied
        pm_no = Decimal("1") - pm_implied
        sharp_no = Decimal("1") - sharp_implied
        return pm_no - sharp_no


def calculate_edge_after_fees(edge: Decimal) -> Decimal:
    """Calculate edge after Polymarket withdrawal fees.

    Args:
        edge: Raw edge percentage

    Returns:
        Edge after fees.
    """
    return edge - PM_WITHDRAWAL_FEE


def calculate_kelly_size(
    bankroll: Decimal,
    edge: Decimal,
    implied_prob: Decimal,
) -> Decimal:
    """Calculate Kelly-optimal position size.

    Uses Kelly/4 for conservative sizing.

    Args:
        bankroll: Total bankroll
        edge: Edge percentage
        implied_prob: Implied probability of winning

    Returns:
        Recommended position size.
    """
    if edge <= 0 or implied_prob <= 0:
        return Decimal("0")

    # Kelly fraction = edge / odds
    # odds = (1 - implied) / implied
    odds = (Decimal("1") - implied_prob) / implied_prob
    kelly_fraction = edge / odds

    # Kelly/4 for conservatism
    conservative_fraction = kelly_fraction / KELLY_DIVISOR

    # Cap at max bankroll percentage
    max_size = bankroll * MAX_BANKROLL_PCT
    kelly_size = bankroll * conservative_fraction

    return min(kelly_size, max_size)


def match_markets(
    pm_markets: list[PolymarketSportMarket],
    sharp_markets: list[SportsMarket],
) -> list[tuple[PolymarketSportMarket, SportsMarket]]:
    """Match Polymarket markets to sharp book markets.

    Uses fuzzy matching on event names and outcomes.

    Args:
        pm_markets: Polymarket sports markets
        sharp_markets: Sharp book markets

    Returns:
        List of matched market pairs.
    """
    matches: list[tuple[PolymarketSportMarket, SportsMarket]] = []

    for pm in pm_markets:
        if pm.sport_key is None:
            continue

        for sharp in sharp_markets:
            # Sport must match
            if not _sport_matches(pm, sharp):
                continue

            # Event must match
            if _event_matches(pm, sharp):
                matches.append((pm, sharp))
                break

    return matches


def _sport_matches(pm: PolymarketSportMarket, sharp: SportsMarket) -> bool:
    """Check if sports match between markets."""
    pm_sport = (pm.sport_key or "").lower()
    sharp_sport = sharp.sport_key.lower()

    # Direct match
    if pm_sport == sharp_sport:
        return True

    # Partial match (e.g., "americanfootball_nfl" vs "americanfootball_nfl")
    if pm_sport in sharp_sport or sharp_sport in pm_sport:
        return True

    return False


def _event_matches(pm: PolymarketSportMarket, sharp: SportsMarket) -> bool:
    """Check if events match between markets."""
    pm_event = (pm.event_name or pm.question).lower()
    sharp_event = f"{sharp.home_team} vs {sharp.away_team}".lower()

    # Check team names appear in both
    pm_parts = set(pm_event.replace("vs", " ").replace("@", " ").split())
    sharp_parts = set(sharp_event.replace("vs", " ").replace("@", " ").split())

    # Require at least 2 matching words (team names)
    common = pm_parts & sharp_parts
    return len(common) >= 2


def find_arbitrage_opportunities(
    pm_markets: list[PolymarketSportMarket],
    sharp_markets: list[SportsMarket],
    bankroll: Decimal = Decimal("10000"),
) -> list[ArbitrageOpportunity]:
    """Find all arbitrage opportunities between markets.

    Args:
        pm_markets: Polymarket sports markets
        sharp_markets: Sharp book markets
        bankroll: Available bankroll for sizing

    Returns:
        List of valid arbitrage opportunities.
    """
    opportunities: list[ArbitrageOpportunity] = []
    matches = match_markets(pm_markets, sharp_markets)

    for pm, sharp in matches:
        # Check YES side
        yes_opp = _check_side(pm, sharp, "yes", bankroll)
        if yes_opp and yes_opp.is_valid:
            opportunities.append(yes_opp)

        # Check NO side
        no_opp = _check_side(pm, sharp, "no", bankroll)
        if no_opp and no_opp.is_valid:
            opportunities.append(no_opp)

    # Sort by edge (highest first)
    opportunities.sort(key=lambda o: o.edge_after_fees, reverse=True)
    return opportunities


def _check_side(
    pm: PolymarketSportMarket,
    sharp: SportsMarket,
    side: str,
    bankroll: Decimal,
) -> ArbitrageOpportunity | None:
    """Check for arbitrage on a specific side.

    Args:
        pm: Polymarket market
        sharp: Sharp book market
        side: "yes" or "no"
        bankroll: Available bankroll

    Returns:
        Opportunity if found, None otherwise.
    """
    pm_implied = pm.implied_yes_prob if side == "yes" else pm.implied_no_prob

    # Get sharp implied prob - try to match outcome
    sharp_implied = None
    if side == "yes":
        # Try to find matching outcome in sharp market
        for outcome in sharp.outcomes:
            # Check if outcome name appears in PM question
            if outcome.name.lower() in pm.question.lower():
                sharp_implied = outcome.best_sharp_prob
                break
        # Fallback to first outcome if no match
        if sharp_implied is None and sharp.outcomes:
            sharp_implied = sharp.outcomes[0].best_sharp_prob
    else:
        # For NO, we need the complement
        for outcome in sharp.outcomes:
            if outcome.name.lower() in pm.question.lower():
                prob = outcome.best_sharp_prob
                sharp_implied = Decimal("1") - prob if prob else None
                break
        if sharp_implied is None and sharp.outcomes:
            prob = sharp.outcomes[0].best_sharp_prob
            sharp_implied = Decimal("1") - prob if prob else None

    if sharp_implied is None:
        return None

    # Calculate edge
    edge = calculate_edge(pm_implied, sharp_implied, side)
    edge_after_fees = calculate_edge_after_fees(edge)

    # Calculate confidence based on liquidity and data quality
    confidence = _calculate_confidence(pm, sharp)

    return ArbitrageOpportunity(
        pm_market={
            "market_id": pm.market_id,
            "question": pm.question,
            "volume": str(pm.volume),
            "yes_price": str(pm.yes_price),
            "yes_token_id": pm.yes_token_id,
        },
        sharp_market={
            "id": sharp.id,
            "sport": sharp.sport_title,
            "home": sharp.home_team,
            "away": sharp.away_team,
        },
        side=side,
        pm_implied=pm_implied,
        sharp_implied=sharp_implied,
        edge=edge,
        edge_after_fees=edge_after_fees,
        confidence=confidence,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _calculate_confidence(pm: PolymarketSportMarket, sharp: SportsMarket) -> Decimal:
    """Calculate confidence score for a match.

    Factors:
    - Volume on Polymarket
    - Recency of sharp book data
    - Match quality
    """
    confidence = Decimal("0.5")

    # Volume factor (more volume = more confidence)
    if pm.volume >= Decimal("100000"):
        confidence += Decimal("0.2")
    elif pm.volume >= Decimal("50000"):
        confidence += Decimal("0.1")
    elif pm.volume >= Decimal("10000"):
        confidence += Decimal("0.05")

    # Event match quality
    if pm.event_name and _event_matches(pm, sharp):
        confidence += Decimal("0.2")

    # Cap at 1.0
    return min(confidence, Decimal("1.0"))


def execute_paper_trade(
    opportunity: ArbitrageOpportunity,
    bankroll: Decimal,
    trade_id: str | None = None,
) -> PaperTrade:
    """Execute a paper trade for an arbitrage opportunity.

    Args:
        opportunity: Arbitrage opportunity
        bankroll: Available bankroll
        trade_id: Optional trade ID (generated if not provided)

    Returns:
        Paper trade record.
    """
    if trade_id is None:
        trade_id = f"arb_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{opportunity.pm_market['market_id'][:8]}"

    # Calculate size
    implied_prob = opportunity.pm_implied if opportunity.side == "yes" else Decimal("1") - opportunity.pm_implied
    size = calculate_kelly_size(bankroll, opportunity.edge_after_fees, implied_prob)

    # Token ID
    token_id = opportunity.pm_market.get("yes_token_id", "")

    return PaperTrade(
        trade_id=trade_id,
        timestamp=opportunity.timestamp,
        pm_market_id=opportunity.pm_market["market_id"],
        pm_token_id=token_id,
        side=opportunity.side,
        size=size,
        entry_price=opportunity.pm_implied,
        sharp_implied_at_entry=opportunity.sharp_implied,
        edge_at_entry=opportunity.edge_after_fees,
    )


class SportsArbitrageStrategy:
    """Main strategy class for cross-platform sports arbitrage."""

    def __init__(
        self,
        bankroll: Decimal = Decimal("10000"),
        data_dir: Path | None = None,
    ) -> None:
        """Initialize strategy.

        Args:
            bankroll: Starting bankroll for paper trading
            data_dir: Directory for trade logs
        """
        self.bankroll = bankroll
        self.data_dir = data_dir or Path("data/sports_arb")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.opportunities_file = self.data_dir / "opportunities.jsonl"
        self.trades_file = self.data_dir / "trades.jsonl"

    def scan(self) -> list[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities.

        Returns:
            List of opportunities found.
        """
        from .odds_api import OddsApiClient
        from .sports_markets import get_sports_markets_from_gamma

        # Fetch Polymarket sports markets
        pm_markets = get_sports_markets_from_gamma(active=True, limit=100)

        # Fetch sharp book odds for relevant sports
        client = OddsApiClient()
        sharp_markets: list[SportsMarket] = []

        # Get unique sports from PM markets
        sports = {m.sport_key for m in pm_markets if m.sport_key}

        for sport in sports:
            try:
                # Map our sport keys to odds API format
                odds_api_sport = _map_sport_key(sport)
                if odds_api_sport:
                    markets = client.get_pinnacle_odds(odds_api_sport)
                    sharp_markets.extend(markets)
            except Exception:
                # Skip sports that fail to fetch
                continue

        # Find opportunities
        opportunities = find_arbitrage_opportunities(pm_markets, sharp_markets, self.bankroll)

        # Log opportunities
        self._log_opportunities(opportunities)

        return opportunities

    def paper_trade(self, opportunity: ArbitrageOpportunity) -> PaperTrade:
        """Execute a paper trade.

        Args:
            opportunity: Arbitrage opportunity

        Returns:
            Paper trade record.
        """
        trade = execute_paper_trade(opportunity, self.bankroll)
        self._log_trade(trade)
        return trade

    def _log_opportunities(self, opportunities: list[ArbitrageOpportunity]) -> None:
        """Append opportunities to log file."""
        with open(self.opportunities_file, "a") as f:
            for opp in opportunities:
                f.write(json.dumps(opp.to_dict()) + "\n")

    def _log_trade(self, trade: PaperTrade) -> None:
        """Append trade to log file."""
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        stats = {
            "total_opportunities": 0,
            "total_trades": 0,
            "open_trades": 0,
            "closed_trades": 0,
            "total_pnl": Decimal("0"),
            "avg_edge": Decimal("0"),
        }

        # Count opportunities
        if self.opportunities_file.exists():
            with open(self.opportunities_file) as f:
                stats["total_opportunities"] = sum(1 for _ in f)

        # Calculate trade stats
        if self.trades_file.exists():
            trades: list[PaperTrade] = []
            with open(self.trades_file) as f:
                for line in f:
                    data = json.loads(line)
                    # Parse back to PaperTrade
                    trade = PaperTrade(
                        trade_id=data["trade_id"],
                        timestamp=data["timestamp"],
                        pm_market_id=data["pm_market_id"],
                        pm_token_id=data["pm_token_id"],
                        side=data["side"],
                        size=Decimal(data["size"]),
                        entry_price=Decimal(data["entry_price"]),
                        sharp_implied_at_entry=Decimal(data["sharp_implied_at_entry"]),
                        edge_at_entry=Decimal(data["edge_at_entry"]),
                        status=data["status"],
                        exit_price=Decimal(data["exit_price"]) if data.get("exit_price") else None,
                        exit_timestamp=data.get("exit_timestamp"),
                        pnl=Decimal(data["pnl"]) if data.get("pnl") else None,
                        close_reason=data.get("close_reason"),
                    )
                    trades.append(trade)

            stats["total_trades"] = len(trades)
            stats["open_trades"] = sum(1 for t in trades if t.status == "open")
            stats["closed_trades"] = sum(1 for t in trades if t.status == "closed")

            closed_pnls = [t.pnl for t in trades if t.pnl is not None]
            if closed_pnls:
                stats["total_pnl"] = sum(closed_pnls, Decimal("0"))

            edges = [t.edge_at_entry for t in trades]
            if edges:
                stats["avg_edge"] = sum(edges, Decimal("0")) / len(edges)

        return stats


def _map_sport_key(sport_key: str) -> str | None:
    """Map internal sport key to odds API format."""
    mapping = {
        "americanfootball_nfl": "americanfootball_nfl",
        "basketball_nba": "basketball_nba",
        "basketball_ncaa": "basketball_ncaa",
        "baseball_mlb": "baseball_mlb",
        "soccer_usa_mls": "soccer_usa_mls",
        "soccer_epl": "soccer_epl",
        "soccer_uefa": "soccer_uefa_champs_league",
        "icehockey_nhl": "icehockey_nhl",
        "mma": "mma_mixed_martial_arts",
    }
    return mapping.get(sport_key)
