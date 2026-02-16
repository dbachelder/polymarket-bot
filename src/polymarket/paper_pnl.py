"""Paper PnL evaluation for pricefeed-aligned Polymarket snapshots.

Provides simple paper trading simulation with configurable exit rules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Paper trading position."""

    market_slug: str
    token_id: str
    side: str  # 'yes' or 'no'
    entry_price: float
    entry_time: str
    size: float = 1.0
    exit_price: float | None = None
    exit_time: str | None = None
    exit_reason: str | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def close(self, exit_price: float, exit_time: str, reason: str) -> None:
        """Close the position."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.size
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_slug": self.market_slug,
            "token_id": self.token_id,
            "side": self.side,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "size": self.size,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class TradeResult:
    """Result of a single paper trade."""

    market_slug: str
    decision: str  # 'up' or 'down'
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    confidence: float
    pnl: float
    pnl_pct: float
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_slug": self.market_slug,
            "decision": self.decision,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "confidence": self.confidence,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
        }


@dataclass
class PaperPnLResult:
    """Complete paper PnL evaluation result."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    avg_pnl_per_trade: float
    win_rate: float
    sharpe_ratio: float | None
    max_drawdown: float
    trades: list[TradeResult]
    parameters: dict[str, Any]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trades": [t.to_dict() for t in self.trades],
            "parameters": self.parameters,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _calculate_mid_price(best_bid: float | None, best_ask: float | None) -> float | None:
    """Calculate mid price from bid/ask."""
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2


def _extract_btc_market(
    pm_data: dict[str, Any],
    target_substring: str = "bitcoin",
) -> dict[str, Any] | None:
    """Extract BTC market from Polymarket snapshot data."""
    markets = pm_data.get("markets", [])

    for market in markets:
        slug = market.get("market_slug", "")
        title = market.get("title", "")

        if target_substring.lower() in slug.lower() or target_substring.lower() in title.lower():
            return market

    return None


def _get_market_price(market: dict[str, Any], side: str) -> float | None:
    """Get price for a side of the market."""
    books = market.get("books", {})
    token_book = books.get(side, {})

    # Get best bid or ask depending on what we want
    bids = token_book.get("bids", [])
    asks = token_book.get("asks", [])

    if bids and asks:
        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 0
        return _calculate_mid_price(best_bid, best_ask)

    return None


def evaluate_simple_exit_rule(
    aligned_data: list[dict[str, Any]],
    exit_rule: str = "mark_at_end",
    timebox_minutes: float = 15.0,
    confidence_threshold: float = 0.6,
    target_substring: str = "bitcoin",
) -> PaperPnLResult:
    """Evaluate paper PnL with simple exit rules.

    Args:
        aligned_data: List of aligned snapshot records
        exit_rule: Exit rule ('mark_at_end', 'timebox', 'stop_loss', 'take_profit')
        timebox_minutes: Time limit for timebox exit
        confidence_threshold: Minimum confidence to take a trade
        target_substring: Market filter substring

    Returns:
        PaperPnLResult with trade statistics
    """
    trades: list[TradeResult] = []
    positions: dict[str, Position] = {}

    # Sort by timestamp
    sorted_data = sorted(aligned_data, key=lambda x: x.get("polymarket_timestamp", ""))

    for record in sorted_data:
        pm_data = record.get("polymarket_data", {})
        pf_features = record.get("pricefeed_features", {})

        # Extract BTC market
        market = _extract_btc_market(pm_data, target_substring)
        if not market:
            continue

        market_slug = market.get("market_slug", "")
        pm_time_str = record.get("polymarket_timestamp", "")

        # Get current prices
        yes_price = _get_market_price(market, "yes")
        no_price = _get_market_price(market, "no")

        if yes_price is None or no_price is None:
            continue

        # Get pricefeed returns for signal generation
        returns = pf_features.get("returns", [])
        if not returns:
            continue

        # Simple signal: positive short-term return = UP, negative = DOWN
        short_term_return = None
        for r in returns:
            if r.get("horizon_seconds", 0) <= 60:  # Use 60s or shorter horizon
                short_term_return = r.get("simple_return", 0)
                break

        if short_term_return is None:
            continue

        # Generate confidence based on return magnitude
        confidence = min(1.0, abs(short_term_return) * 10 + 0.5)

        # Check if we have an open position
        position_key = market_slug

        if position_key in positions:
            # Check exit conditions
            position = positions[position_key]
            should_exit = False
            exit_price = None
            exit_reason = None

            if exit_rule == "mark_at_end":
                # Exit at the last available price
                if position.side == "yes":
                    exit_price = yes_price
                else:
                    exit_price = no_price
                should_exit = True
                exit_reason = "mark_at_end"

            elif exit_rule == "timebox":
                # Check if timebox exceeded
                entry_time = datetime.fromisoformat(position.entry_time.replace("Z", "+00:00"))
                current_time = datetime.fromisoformat(pm_time_str.replace("Z", "+00:00"))
                elapsed = (current_time - entry_time).total_seconds() / 60.0

                if elapsed >= timebox_minutes:
                    if position.side == "yes":
                        exit_price = yes_price
                    else:
                        exit_price = no_price
                    should_exit = True
                    exit_reason = f"timebox_{timebox_minutes}m"

            if should_exit and exit_price is not None:
                position.close(exit_price, pm_time_str, exit_reason)

                trade = TradeResult(
                    market_slug=market_slug,
                    decision="up" if position.side == "yes" else "down",
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    entry_time=position.entry_time,
                    exit_time=pm_time_str,
                    confidence=confidence,
                    pnl=position.pnl,
                    pnl_pct=position.pnl_pct,
                    exit_reason=exit_reason,
                )
                trades.append(trade)
                del positions[position_key]

        elif confidence >= confidence_threshold:
            # Open new position
            side = "yes" if short_term_return > 0 else "no"
            entry_price = yes_price if side == "yes" else no_price

            positions[position_key] = Position(
                market_slug=market_slug,
                token_id=market.get("clob_token_ids", ["", ""])[0] if side == "yes" else market.get("clob_token_ids", ["", ""])[1],
                side=side,
                entry_price=entry_price,
                entry_time=pm_time_str,
                size=1.0,
            )

    # Close any remaining positions at the last available price
    if sorted_data and positions:
        last_record = sorted_data[-1]
        last_pm_data = last_record.get("polymarket_data", {})
        last_time = last_record.get("polymarket_timestamp", "")

        last_market = _extract_btc_market(last_pm_data, target_substring)
        if last_market:
            yes_price = _get_market_price(last_market, "yes")
            no_price = _get_market_price(last_market, "no")

            for position_key, position in list(positions.items()):
                exit_price = yes_price if position.side == "yes" else no_price
                if exit_price is not None:
                    position.close(exit_price, last_time, "mark_at_end")

                    trade = TradeResult(
                        market_slug=position.market_slug,
                        decision="up" if position.side == "yes" else "down",
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        entry_time=position.entry_time,
                        exit_time=last_time,
                        confidence=0.5,  # Unknown at close
                        pnl=position.pnl,
                        pnl_pct=position.pnl_pct,
                        exit_reason="mark_at_end",
                    )
                    trades.append(trade)

    # Calculate statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl < 0)
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # Calculate Sharpe ratio (simplified)
    if total_trades > 1:
        returns_array = np.array([t.pnl for t in trades])
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe_ratio = mean_return / std_return if std_return > 0 else None
    else:
        sharpe_ratio = None

    # Calculate max drawdown
    max_drawdown = 0.0
    running_pnl = 0.0
    peak_pnl = 0.0
    for trade in trades:
        running_pnl += trade.pnl
        peak_pnl = max(peak_pnl, running_pnl)
        drawdown = peak_pnl - running_pnl
        max_drawdown = max(max_drawdown, drawdown)

    return PaperPnLResult(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        total_pnl=total_pnl,
        avg_pnl_per_trade=avg_pnl,
        win_rate=win_rate,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        trades=trades,
        parameters={
            "exit_rule": exit_rule,
            "timebox_minutes": timebox_minutes,
            "confidence_threshold": confidence_threshold,
            "target_substring": target_substring,
        },
    )


def generate_report(
    result: PaperPnLResult,
    format: str = "human",
) -> str:
    """Generate a human-readable or JSON report."""
    if format == "json":
        return result.to_json()

    lines = [
        "=" * 70,
        "PAPER PnL EVALUATION REPORT",
        "=" * 70,
        f"Generated:      {result.generated_at}",
        "",
        "Parameters:",
        f"  Exit Rule:    {result.parameters.get('exit_rule', 'N/A')}",
        f"  Timebox:      {result.parameters.get('timebox_minutes', 'N/A')} minutes",
        f"  Confidence:   >= {result.parameters.get('confidence_threshold', 'N/A')}",
        "",
        "--- Summary ---",
        f"Total Trades:   {result.total_trades}",
        f"Winning:        {result.winning_trades}",
        f"Losing:         {result.losing_trades}",
        f"Win Rate:       {result.win_rate:.1%}",
        "",
        "--- PnL ---",
        f"Total PnL:      {result.total_pnl:+.4f}",
        f"Avg per Trade:  {result.avg_pnl_per_trade:+.4f}",
        f"Max Drawdown:   {result.max_drawdown:.4f}",
    ]

    if result.sharpe_ratio is not None:
        lines.append(f"Sharpe Ratio:   {result.sharpe_ratio:.2f}")

    if result.trades:
        lines.extend([
            "",
            "--- Recent Trades (last 10) ---",
        ])
        for t in result.trades[-10:]:
            entry_dt = datetime.fromisoformat(t.entry_time.replace("Z", "+00:00"))
            lines.append(
                f"  {entry_dt.strftime('%Y-%m-%d %H:%M')} | "
                f"{t.decision.upper():<4} | "
                f"entry={t.entry_price:.3f} | "
                f"exit={t.exit_price:.3f} | "
                f"PnL={t.pnl:+.4f} ({t.pnl_pct:+.1%}) | "
                f"{t.exit_reason}"
            )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def run_paper_evaluation(
    aligned_file: Path,
    out_file: Path | None = None,
    exit_rule: str = "mark_at_end",
    timebox_minutes: float = 15.0,
    confidence_threshold: float = 0.6,
    format: str = "human",
) -> PaperPnLResult:
    """Run paper PnL evaluation from aligned data file.

    Args:
        aligned_file: Path to aligned features JSON file
        out_file: Optional path to write report
        exit_rule: Exit rule to use
        timebox_minutes: Time limit for timebox exit
        confidence_threshold: Minimum confidence to trade
        format: Output format ('human' or 'json')

    Returns:
        PaperPnLResult
    """
    # Load aligned data
    data = json.loads(aligned_file.read_text())

    # Evaluate
    result = evaluate_simple_exit_rule(
        aligned_data=data,
        exit_rule=exit_rule,
        timebox_minutes=timebox_minutes,
        confidence_threshold=confidence_threshold,
    )

    # Generate report
    report = generate_report(result, format=format)

    # Write to file if specified
    if out_file:
        out_file.write_text(report)
        logger.info("Report saved to %s", out_file)

    # Print report
    print(report)

    return result
