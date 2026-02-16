"""CLI commands for trader profiling and copy trading.

Commands:
- trader discover: Discover top traders from leaderboard
- trader list: List tracked traders with scores
- trader sync: Sync fills for tracked traders
- trader nav: Show NAV for a specific trader
- copy sync: Sync and copy trades from top traders
- copy status: Show copy trading performance
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal


def cmd_trader_discover(args: argparse.Namespace) -> None:
    """Discover top traders from leaderboard."""
    from polymarket.trader_profiler import TraderProfiler

    profiler = TraderProfiler(data_dir=args.data_dir)

    print("Fetching traders from leaderboard...")
    discovered = profiler.discover_traders(
        top_n=args.top_n,
        include_leaderboard=True,
        manual_addresses=args.address,
    )

    # Compute scores
    scores = profiler.compute_scores(
        min_volume=Decimal(str(args.min_volume)),
        min_markets=args.min_markets,
    )

    if args.format == "json":
        output = {
            "discovered_count": len(discovered),
            "traders": [t.to_dict() for t in discovered],
            "scores": [s.to_dict() for s in scores[:20]],
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 70)
        print("TRADER DISCOVERY")
        print("=" * 70)
        print(f"Discovered: {len(discovered)} traders")
        print()

        if scores:
            print(f"--- Top Scored Traders ---")
            print(f"{'Rank':<6}{'Address':<44}{'Score':<10}{'30d PnL':<12}{'Volume':<15}")
            print("-" * 70)
            for i, score in enumerate(scores[:20], 1):
                trader = profiler.get_trader(score.address)
                if trader:
                    addr_short = score.address[:40] + "..."
                    pnl = f"${float(trader.pnl_30d):,.0f}"
                    vol = f"${float(trader.volume_lifetime)/1e6:.1f}M"
                    print(f"{i:<6}{addr_short:<44}{score.total_score:<10.1f}{pnl:<12}{vol:<15}")

        print("=" * 70)


def cmd_trader_list(args: argparse.Namespace) -> None:
    """List tracked traders with scores."""
    from polymarket.trader_profiler import TraderProfiler

    profiler = TraderProfiler(data_dir=args.data_dir)

    traders = profiler.get_all_traders()
    scores = list(profiler.scores.values())

    # Filter by tag if specified
    if args.tag:
        traders = [t for t in traders if args.tag in t.tags]

    # Sort by score if available
    from polymarket.trader_profiler import TraderScore

    score_map = {s.address: s for s in scores}
    traders.sort(key=lambda t: score_map.get(t.address.lower(), TraderScore(t.address, 0)).total_score, reverse=True)

    if args.format == "json":
        output = {
            "traders": [t.to_dict() for t in traders],
            "scores": [s.to_dict() for s in scores],
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 90)
        print("TRACKED TRADERS")
        print("=" * 90)
        print(f"{'Address':<44}{'Score':<10}{'30d PnL':<12}{'7d PnL':<12}{'Markets':<10}{'Source':<10}")
        print("-" * 90)

        for trader in traders[:args.limit]:
            score = score_map.get(trader.address.lower())
            score_str = f"{score.total_score:.1f}" if score else "N/A"
            pnl_30d = f"${float(trader.pnl_30d):,.0f}" if trader.pnl_30d else "$0"
            pnl_7d = f"${float(trader.pnl_7d):,.0f}" if trader.pnl_7d else "$0"
            addr_short = trader.address[:40] + "..."
            print(f"{addr_short:<44}{score_str:<10}{pnl_30d:<12}{pnl_7d:<12}{trader.markets_traded:<10}{trader.source:<10}")

        print("=" * 90)
        print(f"Showing {min(len(traders), args.limit)} of {len(traders)} traders")


def cmd_trader_sync(args: argparse.Namespace) -> None:
    """Sync fills for tracked traders."""
    from polymarket.trader_profiler import TraderProfiler
    from polymarket.trader_fills import TraderFillTracker

    tracker = TraderFillTracker(data_dir=args.data_dir)
    profiler = TraderProfiler(data_dir=args.data_dir)

    # Get traders to sync
    if args.address:
        addresses = args.address
    else:
        # Sync all tracked traders
        addresses = [t.address for t in profiler.get_all_traders()]

    print(f"Syncing fills for {len(addresses)} traders...")
    results = tracker.sync_all_traders(addresses, fetch_limit=args.limit)

    if args.format == "json":
        output = {
            "synced": len(results),
            "results": {addr: {"new": new, "total": total} for addr, (new, total) in results.items()},
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 70)
        print("TRADER FILL SYNC")
        print("=" * 70)
        total_new = 0
        for addr, (new, total) in results.items():
            addr_short = addr[:40] + "..."
            print(f"{addr_short}: +{new} new fills (total: {total})")
            total_new += new
        print("-" * 70)
        print(f"Total: {total_new} new fills synced")
        print("=" * 70)


def cmd_trader_nav(args: argparse.Namespace) -> None:
    """Show NAV for a specific trader."""
    from polymarket.trader_fills import TraderFillTracker

    tracker = TraderFillTracker(data_dir=args.data_dir)

    # Compute NAV
    nav = tracker.compute_trader_nav(args.address)
    summary = tracker.get_trader_summary(args.address)
    history = tracker.get_nav_history(args.address)

    if args.format == "json":
        output = {
            "address": args.address,
            "current_nav": nav.to_dict(),
            "summary": summary,
            "history": [h.to_dict() for h in history[-args.history:]],
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 70)
        print(f"TRADER NAV: {args.address[:40]}...")
        print("=" * 70)
        print()
        print("--- Current State ---")
        print(f"Cash Balance:       ${float(nav.cash_balance):>12,.2f}")
        print(f"Positions Value:    ${float(nav.positions_value):>12,.2f}")
        print(f"NAV:                ${float(nav.nav):>12,.2f}")
        print()
        print("--- PnL Breakdown ---")
        print(f"Realized PnL:       ${float(nav.realized_pnl):>12,.2f}")
        print(f"Unrealized PnL:     ${float(nav.unrealized_pnl):>12,.2f}")
        print(f"Total Fees:         ${float(nav.total_fees):>12,.2f}")
        print()
        print("--- Statistics ---")
        print(f"Total Fills:        {summary['total_fills']}")
        print(f"Total Positions:    {summary['total_positions']}")
        print(f"Open Positions:     {summary['open_positions']}")
        print(f"Total Return:       ${summary['total_return']:,.2f} ({summary['total_return_pct']:.2f}%)")
        print(f"Max Drawdown:       ${summary['max_drawdown']:,.2f}")

        if history and args.history > 0:
            print()
            print(f"--- NAV History (last {min(len(history), args.history)} points) ---")
            for snap in history[-args.history:]:
                ts = snap.timestamp[:19]  # Trim to seconds
                print(f"  {ts} | NAV: ${float(snap.nav):>12,.2f} | Pos: {snap.open_position_count}")

        print("=" * 70)


def cmd_copy_sync(args: argparse.Namespace) -> None:
    """Sync and copy trades from top traders."""
    from polymarket.copy_trading import PaperCopyEngine
    from polymarket.trader_fills import TraderFillTracker
    from polymarket.trader_profiler import TraderProfiler

    tracker = TraderFillTracker(data_dir=args.trader_data_dir)
    profiler = TraderProfiler(data_dir=args.trader_data_dir)
    engine = PaperCopyEngine(
        data_dir=args.data_dir,
        config=None,  # Load existing or create default
    )

    # Get top traders to copy
    top_traders = profiler.get_top_traders(k=args.top_k, min_score=args.min_score)

    if not top_traders:
        print("No top traders found. Run 'trader discover' first.")
        return

    print(f"Copying trades from top {len(top_traders)} traders...")

    all_copies = []
    for profile, score in top_traders:
        copies = engine.sync_from_trader(
            profile.address,
            tracker,
            max_copies=args.max_copies_per_trader,
        )
        all_copies.extend(copies)
        if copies:
            print(f"  {profile.address[:40]}...: {len(copies)} trades copied")

    # Record equity snapshot
    equity = engine.record_equity()

    if args.format == "json":
        output = {
            "copied_trades": len(all_copies),
            "top_traders": [
                {"address": p.address, "score": s.total_score}
                for p, s in top_traders
            ],
            "current_equity": equity.to_dict(),
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 70)
        print("COPY TRADING SYNC")
        print("=" * 70)
        print(f"Traders copied: {len(top_traders)}")
        print(f"New trades: {len(all_copies)}")
        print(f"Current equity: ${float(equity.net_equity):,.2f}")
        print(f"Open positions: {equity.open_position_count}")
        print("=" * 70)


def cmd_copy_status(args: argparse.Namespace) -> None:
    """Show copy trading performance status."""
    from polymarket.copy_trading import PaperCopyEngine

    engine = PaperCopyEngine(data_dir=args.data_dir)
    summary = engine.get_performance_summary()
    equity_curve = engine.get_equity_curve()

    if args.format == "json":
        output = {
            "summary": summary,
            "equity_curve": [e.to_dict() for e in equity_curve[-args.history:]],
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 70)
        print("COPY TRADING STATUS")
        print("=" * 70)
        print()
        print("--- Performance Summary ---")
        print(f"Starting Cash:      ${summary['starting_cash']:>12,.2f}")
        print(f"Current Cash:       ${summary['current_cash']:>12,.2f}")
        print(f"Current Equity:     ${summary['current_equity']:>12,.2f}")
        print(f"Total Return:       ${summary['total_return']:>12,.2f} ({summary['total_return_pct']:.2f}%)")
        print(f"Max Drawdown:       ${summary['max_drawdown']:>12,.2f}")
        print()
        print("--- PnL Breakdown ---")
        print(f"Realized PnL:       ${summary['realized_pnl']:>12,.2f}")
        print(f"Unrealized PnL:     ${summary['unrealized_pnl']:>12,.2f}")
        print(f"Total Fees:         ${summary['total_fees']:>12,.2f}")
        print()
        print("--- Trading Activity ---")
        print(f"Open Positions:     {summary['open_positions']}")
        print(f"Total Positions:    {summary['total_positions']}")
        print(f"Copy Trades:        {summary['total_copy_trades']}")
        print(f"Copied Traders:     {summary['copied_traders']}")

        if equity_curve and args.history > 0:
            print()
            print(f"--- Equity Curve (last {min(len(equity_curve), args.history)} points) ---")
            for snap in equity_curve[-args.history:]:
                ts = snap.timestamp[:19]
                print(f"  {ts} | Equity: ${float(snap.net_equity):>12,.2f} | Pos: {snap.open_position_count}")

        print("=" * 70)


def cmd_copy_config(args: argparse.Namespace) -> None:
    """Show or update copy trading configuration."""
    from polymarket.copy_trading import PaperCopyEngine

    engine = PaperCopyEngine(data_dir=args.data_dir)
    config = engine.config

    if args.set:
        # Update config values
        for key, value in args.set.items():
            if hasattr(config, key):
                setattr(config, key, value)
        engine._save_config(config)
        print(f"Configuration updated: {args.set}")

    if args.format == "json":
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print("=" * 70)
        print("COPY TRADING CONFIGURATION")
        print("=" * 70)
        print(f"Top K traders:           {config.top_k}")
        print(f"Min trader score:        {config.min_trader_score}")
        print(f"Position size:           ${float(config.position_size_usd):,.2f}")
        print(f"Max position/market:     ${float(config.max_position_per_market):,.2f}")
        print(f"Max total exposure:      ${float(config.max_total_exposure):,.2f}")
        print(f"Max open positions:      {config.max_open_positions}")
        print(f"Max positions/trader:    {config.max_positions_per_trader}")
        print(f"Starting cash:           ${float(config.starting_cash):,.2f}")
        print()
        print("--- Slippage Model ---")
        print(f"  Base slippage:         {config.slippage.base_slippage_bps} bps")
        print(f"  Spread impact:         {config.slippage.spread_impact_bps} bps")
        print(f"  Size impact factor:    {float(config.slippage.size_impact_factor):.6f}")
        print(f"  Max slippage:          {config.slippage.max_slippage_bps} bps")
        print("=" * 70)


# Need to import this for type checking
def add_trader_commands(subparsers) -> None:
    """Add trader profiling commands to CLI."""
    # Main trader command
    trader = subparsers.add_parser("trader", help="Trader profiling and discovery")
    trader_sub = trader.add_subparsers(dest="trader_cmd", required=True)

    # trader discover
    t_disc = trader_sub.add_parser("discover", help="Discover top traders from leaderboard")
    t_disc.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top traders to fetch (default: 50)",
    )
    t_disc.add_argument(
        "--min-volume",
        type=float,
        default=10000.0,
        help="Minimum lifetime volume to track (default: 10000)",
    )
    t_disc.add_argument(
        "--min-markets",
        type=int,
        default=3,
        help="Minimum markets traded (default: 3)",
    )
    t_disc.add_argument(
        "--address",
        action="append",
        default=[],
        help="Additional trader addresses to add manually",
    )
    t_disc.add_argument(
        "--data-dir",
        default="data/trader_profiles",
        help="Data directory (default: data/trader_profiles)",
    )
    t_disc.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    t_disc.set_defaults(func=cmd_trader_discover)

    # trader list
    t_list = trader_sub.add_parser("list", help="List tracked traders with scores")
    t_list.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum traders to show (default: 50)",
    )
    t_list.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filter by tag",
    )
    t_list.add_argument(
        "--data-dir",
        default="data/trader_profiles",
        help="Data directory (default: data/trader_profiles)",
    )
    t_list.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    t_list.set_defaults(func=cmd_trader_list)

    # trader sync
    t_sync = trader_sub.add_parser("sync", help="Sync fills for tracked traders")
    t_sync.add_argument(
        "--address",
        action="append",
        default=None,
        help="Specific addresses to sync (default: all tracked)",
    )
    t_sync.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum fills per trader (default: 1000)",
    )
    t_sync.add_argument(
        "--data-dir",
        default="data/trader_profiles",
        help="Data directory (default: data/trader_profiles)",
    )
    t_sync.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    t_sync.set_defaults(func=cmd_trader_sync)

    # trader nav
    t_nav = trader_sub.add_parser("nav", help="Show NAV for a specific trader")
    t_nav.add_argument("address", help="Trader wallet address")
    t_nav.add_argument(
        "--history",
        type=int,
        default=10,
        help="Number of NAV history points to show (default: 10)",
    )
    t_nav.add_argument(
        "--data-dir",
        default="data/trader_profiles",
        help="Data directory (default: data/trader_profiles)",
    )
    t_nav.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    t_nav.set_defaults(func=cmd_trader_nav)

    # Copy trading command
    copy = subparsers.add_parser("copy", help="Paper-copy trading from top traders")
    copy_sub = copy.add_subparsers(dest="copy_cmd", required=True)

    # copy sync
    c_sync = copy_sub.add_parser("sync", help="Sync and copy trades from top traders")
    c_sync.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top traders to copy (default: 5)",
    )
    c_sync.add_argument(
        "--min-score",
        type=float,
        default=30.0,
        help="Minimum trader score to copy (default: 30.0)",
    )
    c_sync.add_argument(
        "--max-copies-per-trader",
        type=int,
        default=10,
        help="Max new trades per trader per sync (default: 10)",
    )
    c_sync.add_argument(
        "--data-dir",
        default="data/copy_trading",
        help="Copy trading data directory (default: data/copy_trading)",
    )
    c_sync.add_argument(
        "--trader-data-dir",
        default="data/trader_profiles",
        help="Trader profiles data directory (default: data/trader_profiles)",
    )
    c_sync.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    c_sync.set_defaults(func=cmd_copy_sync)

    # copy status
    c_status = copy_sub.add_parser("status", help="Show copy trading performance")
    c_status.add_argument(
        "--data-dir",
        default="data/copy_trading",
        help="Data directory (default: data/copy_trading)",
    )
    c_status.add_argument(
        "--history",
        type=int,
        default=10,
        help="Number of equity points to show (default: 10)",
    )
    c_status.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    c_status.set_defaults(func=cmd_copy_status)

    # copy config
    c_config = copy_sub.add_parser("config", help="Show or update copy trading config")
    c_config.add_argument(
        "--data-dir",
        default="data/copy_trading",
        help="Data directory (default: data/copy_trading)",
    )
    c_config.add_argument(
        "--set",
        type=json.loads,
        default=None,
        help="JSON object with config updates (e.g., '{\"top_k\": 10}')",
    )
    c_config.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    c_config.set_defaults(func=cmd_copy_config)
