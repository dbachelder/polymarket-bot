"""Paper trade tracker for cross-market arbitrage.

Tracks paper trades and calculates theoretical/realized PnL.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import ArbitrageOpportunity, PaperTrade

logger = logging.getLogger(__name__)


class PaperTradeTracker:
    """Tracks paper trades for cross-market arbitrage strategy.

    Stores trade history and calculates PnL for backtesting
    arbitrage opportunities without real capital.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize tracker.

        Args:
            data_dir: Directory to store trade data (default: data/cross_market/)
        """
        self.data_dir = data_dir or Path("data/cross_market")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.trades: dict[str, PaperTrade] = {}
        self._load_trades()

    def _get_trades_file(self) -> Path:
        """Get path to trades storage file."""
        return self.data_dir / "paper_trades.json"

    def _load_trades(self) -> None:
        """Load existing trades from storage."""
        trades_file = self._get_trades_file()
        if not trades_file.exists():
            return

        try:
            data = json.loads(trades_file.read_text())
            for trade_data in data.get("trades", []):
                try:
                    trade = self._deserialize_trade(trade_data)
                    if trade:
                        self.trades[trade.trade_id] = trade
                except Exception as e:
                    logger.debug("Error deserializing trade: %s", e)

            logger.info("Loaded %d paper trades", len(self.trades))
        except Exception as e:
            logger.exception("Error loading trades: %s", e)

    def _deserialize_trade(self, data: dict[str, Any]) -> PaperTrade | None:
        """Deserialize a trade from stored data."""
        # Simplified deserialization - would need full ArbitrageOpportunity reconstruction
        # For now, just log that we found it
        return None

    def _save_trades(self) -> None:
        """Save trades to storage."""
        trades_file = self._get_trades_file()

        try:
            data = {
                "saved_at": datetime.now(UTC).isoformat(),
                "trade_count": len(self.trades),
                "trades": [trade.to_dict() for trade in self.trades.values()],
            }

            trades_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.exception("Error saving trades: %s", e)

    def enter_position(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: float = 1.0,
    ) -> PaperTrade:
        """Enter a new paper trade position.

        Args:
            opportunity: The arbitrage opportunity
            position_size: Number of contracts per side

        Returns:
            The created PaperTrade
        """
        trade_id = str(uuid.uuid4())[:8]

        trade = PaperTrade(
            trade_id=trade_id,
            opportunity=opportunity,
            position_size=position_size,
            entry_yes_price=opportunity.yes_price,
            entry_no_price=opportunity.no_price,
            entry_time=datetime.now(UTC),
            status="open",
            theoretical_pnl=self._calculate_theoretical_pnl(
                opportunity, position_size, opportunity.yes_price, opportunity.no_price
            ),
        )

        self.trades[trade_id] = trade
        self._save_trades()

        logger.info(
            "Entered paper trade %s: %s YES @ %.3f + %s NO @ %.3f (spread: %.3f)",
            trade_id,
            opportunity.venue_yes,
            opportunity.yes_price,
            opportunity.venue_no,
            opportunity.no_price,
            opportunity.net_spread,
        )

        return trade

    def close_position(
        self,
        trade_id: str,
        exit_yes_price: float | None = None,
        exit_no_price: float | None = None,
    ) -> PaperTrade | None:
        """Close an open paper trade position.

        Args:
            trade_id: Trade ID to close
            exit_yes_price: Exit price for YES (if closing early)
            exit_no_price: Exit price for NO (if closing early)

        Returns:
            Updated PaperTrade or None if not found
        """
        trade = self.trades.get(trade_id)
        if not trade:
            logger.warning("Trade %s not found", trade_id)
            return None

        if trade.status != "open":
            logger.warning("Trade %s is not open (status: %s)", trade_id, trade.status)
            return trade

        now = datetime.now(UTC)

        # If no exit prices provided, assume held to resolution
        # YES pays $1, NO pays $0 (or vice versa depending on outcome)
        # For paper trading, we assume the arbitrage resolves profitably
        if exit_yes_price is None and exit_no_price is None:
            # Held to resolution - assume we capture the full $1
            exit_yes_price = 1.0  # YES won
            exit_no_price = 0.0  # NO lost
            status = "held_to_resolution"
        else:
            # Closed early
            exit_yes_price = exit_yes_price or trade.entry_yes_price
            exit_no_price = exit_no_price or trade.entry_no_price
            status = "closed"

        # Calculate realized PnL
        realized_pnl = self._calculate_realized_pnl(trade, exit_yes_price, exit_no_price)

        updated_trade = PaperTrade(
            trade_id=trade.trade_id,
            opportunity=trade.opportunity,
            position_size=trade.position_size,
            entry_yes_price=trade.entry_yes_price,
            entry_no_price=trade.entry_no_price,
            entry_time=trade.entry_time,
            exit_yes_price=exit_yes_price,
            exit_no_price=exit_no_price,
            exit_time=now,
            status=status,
            realized_pnl=realized_pnl,
        )

        self.trades[trade_id] = updated_trade
        self._save_trades()

        logger.info(
            "Closed paper trade %s: realized_pnl=%.3f, status=%s",
            trade_id,
            realized_pnl,
            status,
        )

        return updated_trade

    def update_open_positions(self) -> None:
        """Update theoretical PnL for all open positions."""
        for trade_id, trade in self.trades.items():
            if trade.status != "open":
                continue

            # Update theoretical PnL
            theoretical_pnl = self._calculate_theoretical_pnl(
                trade.opportunity,
                trade.position_size,
                trade.entry_yes_price,
                trade.entry_no_price,
            )

            updated = PaperTrade(
                trade_id=trade.trade_id,
                opportunity=trade.opportunity,
                position_size=trade.position_size,
                entry_yes_price=trade.entry_yes_price,
                entry_no_price=trade.entry_no_price,
                entry_time=trade.entry_time,
                exit_yes_price=trade.exit_yes_price,
                exit_no_price=trade.exit_no_price,
                exit_time=trade.exit_time,
                status=trade.status,
                realized_pnl=trade.realized_pnl,
                theoretical_pnl=theoretical_pnl,
            )

            self.trades[trade_id] = updated

        self._save_trades()

    def _calculate_theoretical_pnl(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: float,
        current_yes_price: float,
        current_no_price: float,
    ) -> float:
        """Calculate theoretical PnL for a position.

        Args:
            opportunity: The arbitrage opportunity
            position_size: Position size
            current_yes_price: Current YES price
            current_no_price: Current NO price

        Returns:
            Theoretical PnL
        """
        # Cost basis
        cost = (opportunity.yes_price + opportunity.no_price) * position_size

        # At resolution, one side pays $1, the other $0
        # So resolved value = $1 * position_size
        resolved_value = 1.0 * position_size

        # Theoretical PnL if held to resolution
        gross_pnl = resolved_value - cost

        # Subtract fees
        net_pnl = gross_pnl - (opportunity.total_fees * position_size)

        return net_pnl

    def _calculate_realized_pnl(
        self,
        trade: PaperTrade,
        exit_yes_price: float,
        exit_no_price: float,
    ) -> float:
        """Calculate realized PnL when closing a position.

        Args:
            trade: The trade
            exit_yes_price: Exit price for YES
            exit_no_price: Exit price for NO

        Returns:
            Realized PnL
        """
        # Cost basis
        cost = (trade.entry_yes_price + trade.entry_no_price) * trade.position_size

        # Exit value
        exit_value = (exit_yes_price + exit_no_price) * trade.position_size

        # PnL
        gross_pnl = exit_value - cost

        # Subtract fees
        net_pnl = gross_pnl - (trade.opportunity.total_fees * trade.position_size)

        return net_pnl

    def get_open_positions(self) -> list[PaperTrade]:
        """Get all open positions.

        Returns:
            List of open PaperTrade
        """
        return [t for t in self.trades.values() if t.status == "open"]

    def get_closed_positions(self) -> list[PaperTrade]:
        """Get all closed positions.

        Returns:
            List of closed PaperTrade
        """
        return [t for t in self.trades.values() if t.status != "open"]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all trades.

        Returns:
            Dictionary with performance metrics
        """
        open_trades = self.get_open_positions()
        closed_trades = self.get_closed_positions()

        total_trades = len(self.trades)
        open_count = len(open_trades)
        closed_count = len(closed_trades)

        # Calculate PnL
        total_realized_pnl = sum((t.realized_pnl or 0) for t in closed_trades)
        total_theoretical_pnl = sum((t.theoretical_pnl or 0) for t in open_trades)

        # Win rate
        winning_trades = [t for t in closed_trades if (t.realized_pnl or 0) > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0

        # Average metrics
        avg_realized_pnl = total_realized_pnl / len(closed_trades) if closed_trades else 0.0
        avg_spread = (
            sum(t.opportunity.net_spread for t in self.trades.values()) / total_trades
            if total_trades
            else 0.0
        )

        return {
            "total_trades": total_trades,
            "open_positions": open_count,
            "closed_positions": closed_count,
            "win_rate": win_rate,
            "total_realized_pnl": total_realized_pnl,
            "total_theoretical_pnl": total_theoretical_pnl,
            "total_pnl": total_realized_pnl + total_theoretical_pnl,
            "avg_realized_pnl": avg_realized_pnl,
            "avg_spread_captured": avg_spread,
            "trades_by_status": {
                "open": open_count,
                "closed": len([t for t in closed_trades if t.status == "closed"]),
                "held_to_resolution": len(
                    [t for t in closed_trades if t.status == "held_to_resolution"]
                ),
            },
        }

    def export_trades(self, out_path: Path | None = None) -> Path:
        """Export all trades to a JSON file.

        Args:
            out_path: Output file path (default: data_dir/trades_export_YYYYMMDDT.json)

        Returns:
            Path to exported file
        """
        if out_path is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            out_path = self.data_dir / f"trades_export_{timestamp}.json"

        summary = self.get_performance_summary()
        data = {
            "export_time": datetime.now(UTC).isoformat(),
            "summary": summary,
            "trades": [trade.to_dict() for trade in self.trades.values()],
        }

        out_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Exported %d trades to %s", len(self.trades), out_path)

        return out_path

    def generate_daily_report(self, date: datetime | None = None) -> dict[str, Any]:
        """Generate a daily performance report.

        Args:
            date: Report date (default: today)

        Returns:
            Daily report dictionary
        """
        if date is None:
            date = datetime.now(UTC)

        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)

        # Filter trades for this date
        day_trades = [t for t in self.trades.values() if date_start <= t.entry_time < date_end]

        day_opportunities = len(day_trades)
        day_pnl = sum((t.realized_pnl or t.theoretical_pnl or 0) for t in day_trades)

        return {
            "date": date_start.strftime("%Y-%m-%d"),
            "opportunities_traded": day_opportunities,
            "day_pnl": day_pnl,
            "trades": [t.trade_id for t in day_trades],
            "cumulative_trades": len(self.trades),
        }
