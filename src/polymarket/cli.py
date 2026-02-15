from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import clob
from .microstructure import (
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_EXTREME_PIN_THRESHOLD,
    DEFAULT_SPREAD_ALERT_THRESHOLD,
    analyze_snapshot_microstructure,
    generate_microstructure_summary,
    log_microstructure_alerts,
)


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_markets_5m(args: argparse.Namespace) -> None:
    from .site import extract_5m_markets, fetch_predictions_page, parse_next_data

    html = fetch_predictions_page("5M")
    data = parse_next_data(html)
    ms = extract_5m_markets(data)
    out = [m.__dict__ for m in ms[: args.limit]]
    _print({"count": len(out), "markets": out})


def cmd_markets_15m(args: argparse.Namespace) -> None:
    from .site import (
        extract_crypto_interval_events,
        fetch_crypto_interval_page,
        parse_next_data,
    )

    html = fetch_crypto_interval_page("15M")
    data = parse_next_data(html)
    ms = extract_crypto_interval_events(data, interval_slug="15M")
    out = [m.__dict__ for m in ms[: args.limit]]
    _print({"count": len(out), "markets": out})


def cmd_book(args: argparse.Namespace) -> None:
    book = clob.get_book(args.token_id)
    _print(book)


def cmd_price(args: argparse.Namespace) -> None:
    price = clob.get_price(args.token_id, side=args.side)
    _print(price)


def cmd_collect_5m(args: argparse.Namespace) -> None:
    from .collector import collect_5m_snapshot
    from .collector_loop import collect_5m_loop

    out_dir = Path(args.out)

    if args.every_seconds is not None:
        collect_5m_loop(
            out_dir=out_dir,
            interval_seconds=float(args.every_seconds),
            max_backoff_seconds=float(args.max_backoff_seconds),
            retention_hours=args.retention_hours,
        )
    else:
        out = collect_5m_snapshot(out_dir)
        print(str(out))


def cmd_collect_15m(args: argparse.Namespace) -> None:
    from .collector import collect_15m_snapshot

    out = collect_15m_snapshot(Path(args.out))
    print(str(out))


def cmd_universe_5m(args: argparse.Namespace) -> None:
    from .universe import build_universe, save_universe

    data = build_universe(cross_check=args.cross_check)
    out_path = save_universe(data, Path(args.out))
    print(str(out_path))


def cmd_collect_15m_loop(args: argparse.Namespace) -> None:
    from .collector_loop import collect_15m_loop

    collect_15m_loop(
        out_dir=Path(args.out),
        interval_seconds=float(args.interval_seconds),
        max_backoff_seconds=float(args.max_backoff_seconds),
        retention_hours=args.retention_hours,
        max_snapshots=args.max_snapshots,
        microstructure_interval_seconds=float(args.microstructure_interval_seconds),
        microstructure_target=args.microstructure_target,
        spread_alert_threshold=float(args.spread_alert_threshold),
        extreme_pin_threshold=float(args.extreme_pin_threshold),
        depth_levels=int(args.depth_levels),
    )


def cmd_pnl_verify(args: argparse.Namespace) -> None:
    """Verify PnL from fills data with cash tracking and sanity checks."""
    from decimal import Decimal

    from .pnl import (
        PnLVerifier,
        load_fills_from_file,
        load_orderbooks_from_file,
        load_orderbooks_from_snapshot,
        save_daily_summary,
    )

    # Determine fills source
    fills: list = []

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(
                json.dumps({"error": f"Input file not found: {args.input}"}),
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)
        fills = load_fills_from_file(input_path)
    elif args.data_dir:
        # Load from data directory - look for fills.json or fills.jsonl
        data_dir = Path(args.data_dir)
        fills_path = data_dir / "fills.json"
        fills_jsonl_path = data_dir / "fills.jsonl"

        if fills_path.exists():
            fills = load_fills_from_file(fills_path)
        elif fills_jsonl_path.exists():
            fills = load_fills_from_file(fills_jsonl_path)
        else:
            print(
                json.dumps(
                    {"error": f"No fills.json or fills.jsonl found in {args.data_dir}"}
                ),
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)
    else:
        print(
            json.dumps(
                {"error": "Must specify --input or --data-dir"}
            ),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    # Load optional orderbooks for liquidation value
    orderbooks = None
    if args.books:
        books_path = Path(args.books)
        if books_path.exists():
            orderbooks = load_orderbooks_from_file(books_path)
    elif args.snapshot:
        snapshot_path = Path(args.snapshot)
        if snapshot_path.exists():
            orderbooks = load_orderbooks_from_snapshot(snapshot_path)
    elif args.data_dir:
        # Try to load latest snapshot from data directory
        data_dir = Path(args.data_dir)
        snapshot_files = sorted(data_dir.glob("snapshot_*.json"))
        if snapshot_files:
            orderbooks = load_orderbooks_from_snapshot(snapshot_files[-1])

    # Build verifier
    starting_cash = Decimal(str(args.starting_cash)) if args.starting_cash else Decimal("0")
    verifier = PnLVerifier(starting_cash=starting_cash)
    verifier.add_fills(fills)

    # Compute report
    report = verifier.compute_pnl(
        orderbooks=orderbooks,
        since=args.since,
        market_filter=args.market,
    )

    # Save daily summary if requested
    if args.save_daily:
        summary_path = save_daily_summary(
            report,
            out_dir=Path(args.daily_dir) if args.daily_dir else None,
        )
        if args.format == "human":
            print(f"\nDaily summary saved: {summary_path}")

    # Output format
    if args.format == "json":
        print(report.to_json())
    else:
        # Human-readable format
        output = report.to_dict()

        print("=" * 70)
        print("PnL VERIFICATION REPORT")
        print("=" * 70)

        # Metadata
        if output["metadata"]["since"]:
            print(f"Since:            {output['metadata']['since']}")
        if output["metadata"]["market_filter"]:
            print(f"Market Filter:    {output['metadata']['market_filter']}")
        print(f"Generated:        {output['metadata']['generated_at']}")

        print("\n--- Summary ---")
        print(f"Total Fills:      {output['summary']['total_fills']}")
        print(f"Unique Tokens:    {output['summary']['unique_tokens']}")
        print(f"Unique Markets:   {output['summary']['unique_markets']}")

        print("\n--- Cash Tracking ---")
        print(f"Starting Cash:    ${output['cash']['starting_cash']:,.2f}")
        print(f"Cash Flow:        ${output['cash']['cash_flow_from_fills']:,.2f}")
        print(f"Ending Cash:      ${output['cash']['ending_cash']:,.2f}")

        print("\n--- PnL Breakdown ---")
        print(f"Realized PnL:     ${output['pnl']['realized_pnl']:,.2f}")
        print(f"Unrealized PnL:   ${output['pnl']['unrealized_pnl']:,.2f}")
        print(f"Total Fees:       ${output['pnl']['total_fees']:,.2f}")
        print(f"Net PnL:          ${output['pnl']['net_pnl']:,.2f}")

        print("\n--- Liquidation Analysis ---")
        print(f"Mark to Mid:      ${output['liquidation']['mark_to_mid']:,.2f}")
        print(f"Liquidation Val:  ${output['liquidation']['liquidation_value']:,.2f}")
        print(
            f"Discount:         ${output['liquidation']['liquidation_discount']:,.2f} "
            f"({output['liquidation']['discount_pct']:.1f}%)"
        )

        print("\n--- Verification ---")
        cash_ok = "✓" if output["verification"]["cashflow_conserved"] else "✗"
        pos_ok = "✓" if output["verification"]["position_verified"] else "✗"
        print(f"Cashflow:         {cash_ok} Conserved")
        print(f"Positions:        {pos_ok} Verified")

        if output["positions"]:
            print(f"\n--- Open Positions ({len(output['positions'])}) ---")
            for pos in output["positions"][:10]:  # Limit to 10
                slug = pos.get("market_slug", "")
                token_display = slug if slug else pos["token_id"][:20] + "..."
                print(f"\n  {token_display}")
                print(f"    Size: {pos['net_size']:,.2f} @ ${pos['avg_cost_basis']:.3f}")
                print(
                    f"    Mid:  ${pos['current_price']:.3f} | "
                    f"Unrealized: ${pos['unrealized_pnl']:,.2f}"
                )
                if pos.get("liquidation_value") != pos.get("mark_to_mid"):
                    print(
                        f"    Liquid: ${pos['liquidation_value']:,.2f} "
                        f"(discount: ${pos['mark_to_mid'] - pos['liquidation_value']:,.2f})"
                    )

        if output["verification"]["warnings"]:
            print(f"\n--- Warnings ({len(output['verification']['warnings'])}) ---")
            for warning in output["verification"]["warnings"]:
                print(f"  ! {warning}")

        print("\n" + "=" * 70)


def cmd_microstructure(args: argparse.Namespace) -> None:
    """Analyze microstructure for markets in a snapshot."""
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(
            json.dumps({"error": f"Snapshot file not found: {args.snapshot}"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    if args.summary:
        # Generate summary with alerts
        summary = generate_microstructure_summary(
            snapshot_path=snapshot_path,
            target_market_substring=args.target,
            spread_threshold=float(args.spread_threshold),
            extreme_pin_threshold=float(args.extreme_pin_threshold),
            depth_levels=int(args.depth_levels),
        )

        if args.format == "json":
            print(json.dumps(summary, indent=2))
        else:
            # Human-readable format
            print("=" * 70)
            print("MARKET MICROSTRUCTURE SUMMARY")
            print("=" * 70)
            print(f"Generated:      {summary['generated_at']}")
            print(f"Snapshot:       {summary['snapshot_path']}")
            print(f"Markets:        {summary['markets_analyzed']}")
            print("\nThresholds:")
            print(f"  Spread alert:     > {summary['spread_threshold']:.2f}")
            print(
                f"  Extreme pin:      <= {summary['extreme_pin_threshold']:.2f} "
                f"or >= {1.0 - summary['extreme_pin_threshold']:.2f}"
            )
            print(f"  Depth levels:     {summary['depth_levels']}")

            if summary["alerts"]:
                print(f"\n--- ALERTS ({summary['alert_count']}) ---")
                for alert in summary["alerts"]:
                    print(f"  ⚠ {alert}")
            else:
                print("\n--- ALERTS ---")
                print("  ✓ No alerts. Markets appear healthy.")

            if summary["market_summaries"]:
                print(f"\n--- MARKET DETAILS ({len(summary['market_summaries'])}) ---")
                for ms in summary["market_summaries"]:
                    print(f"\n  {ms['market_title']}")
                    print(
                        f"    YES: bid={ms.get('yes_best_bid'):.2f} ask={ms.get('yes_best_ask'):.2f} "
                        f"spread={ms.get('yes_spread'):.2f} imbalance={ms.get('yes_imbalance', 0):+.3f}"
                    )
                    print(
                        f"    NO:  bid={ms.get('no_best_bid'):.2f} ask={ms.get('no_best_ask'):.2f} "
                        f"spread={ms.get('no_spread'):.2f} imbalance={ms.get('no_imbalance', 0):+.3f}"
                    )
                    if ms.get("implied_sum"):
                        print(
                            f"    Implied: YES_mid={ms.get('implied_yes_mid', 0):.2f} "
                            f"NO_mid={ms.get('implied_no_mid', 0):.2f} "
                            f"sum={ms.get('implied_sum', 0):.2f} "
                            f"consistency={ms.get('consistency_diff', 0):.3f}"
                        )

            print("\n" + "=" * 70)

        # Log alerts to stderr as well
        log_microstructure_alerts(summary)
    else:
        # Raw analysis output
        analyses = analyze_snapshot_microstructure(
            snapshot_path=snapshot_path,
            target_market_substring=args.target,
            depth_levels=int(args.depth_levels),
        )
        print(json.dumps(analyses, indent=2))


def cmd_binance_collect(args: argparse.Namespace) -> None:
    """Collect Binance BTC market data (REST API single snapshot)."""
    from pathlib import Path

    from .binance_collector import collect_snapshot_rest

    out_dir = Path(args.out)
    out_path = collect_snapshot_rest(
        out_dir=out_dir,
        symbol=args.symbol,
        kline_intervals=args.intervals,
    )
    print(str(out_path))


def cmd_binance_loop(args: argparse.Namespace) -> None:
    """Run Binance WebSocket collector loop."""
    from pathlib import Path

    from .binance_collector import run_collector_loop

    run_collector_loop(
        out_dir=Path(args.out),
        symbol=args.symbol,
        kline_intervals=args.intervals,
        snapshot_interval_seconds=float(args.snapshot_interval_seconds),
        max_reconnect_delay=float(args.max_reconnect_delay),
        retention_hours=args.retention_hours,
    )


def cmd_binance_features(args: argparse.Namespace) -> None:
    """Build features from Binance data and align to Polymarket snapshots."""
    from pathlib import Path

    from .binance_features import (
        align_to_polymarket_snapshots,
        save_aligned_features,
    )

    binance_dir = Path(args.binance_dir)
    polymarket_dir = Path(args.polymarket_dir)
    out_path = Path(args.out)

    aligned = align_to_polymarket_snapshots(
        binance_data_dir=binance_dir,
        polymarket_data_dir=polymarket_dir,
        tolerance_seconds=float(args.tolerance),
    )

    save_aligned_features(aligned, out_path)
    print(f"Aligned {len(aligned)} records to {out_path}")


def cmd_health_check(args: argparse.Namespace) -> None:
    """Check collector health and staleness SLA."""
    from pathlib import Path

    from .collector_loop import check_staleness_sla

    out_dir = Path(args.data_dir)
    result = check_staleness_sla(
        out_dir=out_dir,
        max_age_seconds=float(args.max_age_seconds),
        prefix=args.prefix,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        status = "✓ HEALTHY" if result["healthy"] else "✗ UNHEALTHY"
        print(f"{status}: {result['message']}")
        if result["age_seconds"] is not None:
            print(f"  Age: {result['age_seconds']:.1f}s (max: {result['max_age_seconds']:.1f}s)")

    # Exit with error code if unhealthy and --fail is set
    if args.fail and not result["healthy"]:
        raise SystemExit(1)


def cmd_sports_scan(args: argparse.Namespace) -> None:
    """Scan for sports arbitrage opportunities."""
    from decimal import Decimal

    from .sports_arbitrage import SportsArbitrageStrategy

    strategy = SportsArbitrageStrategy(
        bankroll=Decimal(str(args.bankroll)),
    )

    opportunities = strategy.scan()

    if args.format == "json":
        print(json.dumps([opp.to_dict() for opp in opportunities], indent=2))
    else:
        print("=" * 80)
        print("SPORTS ARBITRAGE SCAN")
        print("=" * 80)
        print(f"Opportunities found: {len(opportunities)}")
        print(f"Min edge: {args.min_edge}%")
        print()

        for i, opp in enumerate(opportunities[: args.limit], 1):
            print(f"--- Opportunity {i} ---")
            print(f"Market: {opp.pm_market.get('question', 'N/A')}")
            print(f"Side: {opp.side.upper()}")
            print(f"Polymarket implied: {float(opp.pm_implied):.2%}")
            print(f"Sharp book implied: {float(opp.sharp_implied):.2%}")
            print(f"Edge: {float(opp.edge):.2%}")
            print(f"Edge after fees: {float(opp.edge_after_fees):.2%}")
            print(f"Confidence: {float(opp.confidence):.2%}")
            print()

        print("=" * 80)


def cmd_sports_trade(args: argparse.Namespace) -> None:
    """Execute paper trades for arbitrage opportunities."""
    from decimal import Decimal

    from .sports_arbitrage import SportsArbitrageStrategy

    strategy = SportsArbitrageStrategy(
        bankroll=Decimal(str(args.bankroll)),
    )

    if args.scan:
        # Scan and trade all opportunities
        opportunities = strategy.scan()
        trades = []
        for opp in opportunities:
            if opp.is_valid:
                trade = strategy.paper_trade(opp)
                trades.append(trade)

        if args.format == "json":
            print(json.dumps([t.to_dict() for t in trades], indent=2))
        else:
            print("=" * 80)
            print("SPORTS ARBITRAGE PAPER TRADES")
            print("=" * 80)
            print(f"Opportunities found: {len(opportunities)}")
            print(f"Trades executed: {len(trades)}")
            print()

            for t in trades:
                print(f"Trade: {t.trade_id}")
                print(f"  Market: {t.pm_market_id}")
                print(f"  Side: {t.side.upper()}")
                print(f"  Size: ${float(t.size):.2f}")
                print(f"  Entry: {float(t.entry_price):.4f}")
                print(f"  Edge: {float(t.edge_at_entry):.2%}")
                print()

            print("=" * 80)


def cmd_sports_stats(args: argparse.Namespace) -> None:
    """Show sports arbitrage strategy statistics."""
    from decimal import Decimal

    from .sports_arbitrage import SportsArbitrageStrategy

    strategy = SportsArbitrageStrategy(
        bankroll=Decimal(str(args.bankroll)),
    )

    stats = strategy.get_stats()

    if args.format == "json":
        # Convert Decimals to strings for JSON serialization
        json_stats = {k: str(v) if isinstance(v, Decimal) else v for k, v in stats.items()}
        print(json.dumps(json_stats, indent=2))
    else:
        print("=" * 80)
        print("SPORTS ARBITRAGE STATISTICS")
        print("=" * 80)
        print(f"Total opportunities detected: {stats['total_opportunities']}")
        print(f"Total trades: {stats['total_trades']}")
        print(f"Open trades: {stats['open_trades']}")
        print(f"Closed trades: {stats['closed_trades']}")
        print(f"Total PnL: ${float(stats['total_pnl']):,.2f}")
        print(f"Average edge: {float(stats['avg_edge']):.2%}")
        print("=" * 80)


def cmd_imbalance_backtest(args: argparse.Namespace) -> None:
    """Run orderbook imbalance strategy backtest."""
    from pathlib import Path

    from .strategy_imbalance import (
        load_snapshots_for_backtest,
        parameter_sweep,
        run_backtest,
    )

    data_dir = Path(args.data_dir)

    # Load snapshots
    snapshots = load_snapshots_for_backtest(
        data_dir=data_dir,
        interval=args.interval,
    )

    if not snapshots:
        print(
            json.dumps({"error": f"No snapshots found in {data_dir} for interval {args.interval}"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    if args.sweep:
        # Parameter sweep
        results = parameter_sweep(
            snapshots=snapshots,
            target_market_substring=args.target,
        )

        if args.format == "json":
            print(json.dumps({"results": results}, indent=2))
        else:
            print("=" * 80)
            print("ORDERBOOK IMBALANCE STRATEGY - PARAMETER SWEEP")
            print("=" * 80)
            print(f"Snapshots analyzed: {len(snapshots)}")
            print(f"Target market: {args.target}")
            print(f"\n{'Rank':<6}{'k':<4}{'theta':<8}{'p_max':<8}{'Trades':<8}{'UP':<6}{'DOWN':<8}{'Avg Conf':<10}")
            print("-" * 80)

            for i, result in enumerate(results[:10], 1):  # Top 10
                p = result["params"]
                m = result["metrics"]
                print(
                    f"{i:<6}{p['k']:<4}{p['theta']:<8.2f}{p['p_max']:<8.2f}"
                    f"{m['total_trades']:<8}{m['up_trades']:<6}{m['down_trades']:<8}"
                    f"{m['avg_confidence']:<10.3f}"
                )

            print("=" * 80)

    else:
        # Single backtest run
        result = run_backtest(
            snapshots=snapshots,
            k=args.k,
            theta=args.theta,
            p_max=args.p_max,
            target_market_substring=args.target,
        )

        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("=" * 80)
            print("ORDERBOOK IMBALANCE STRATEGY - BACKTEST RESULTS")
            print("=" * 80)
            print(f"Snapshots analyzed: {len(snapshots)}")
            print(f"Target market: {args.target}")
            print("\nParameters:")
            print(f"  k (depth levels):     {args.k}")
            print(f"  theta (threshold):    {args.theta:.2f}")
            print(f"  p_max (max price):    {args.p_max:.2f}")

            print("\n--- Results ---")
            print(f"Total trades:     {result.metrics['total_trades']}")
            print(f"UP trades:        {result.metrics['up_trades']}")
            print(f"DOWN trades:      {result.metrics['down_trades']}")
            print(f"Avg confidence:   {result.metrics['avg_confidence']:.3f}")
            print(f"Avg entry price:  {result.metrics['avg_entry_price']:.3f}")

            if result.trades:
                print("\n--- Recent Trades (last 10) ---")
                for t in result.trades[-10:]:
                    print(
                        f"  {t.timestamp.strftime('%H:%M')} | {t.decision:<6} | "
                        f"imb={t.imbalance_value:.3f} | mid={t.mid_yes:.3f} | "
                        f"entry={t.entry_price:.3f} | conf={t.confidence:.2f}"
                    )

            print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(prog="polymarket")
    sub = p.add_subparsers(dest="cmd", required=True)

    p5 = sub.add_parser("markets-5m", help="Heuristic fetch of likely 5-minute markets")
    p5.add_argument("--limit", type=int, default=50)
    p5.add_argument("--search", type=str, default=None)
    p5.set_defaults(func=cmd_markets_5m)

    pb = sub.add_parser("book", help="Fetch CLOB orderbook for a token_id")
    pb.add_argument("token_id")
    pb.set_defaults(func=cmd_book)

    pp = sub.add_parser("price", help="Fetch CLOB price for a token_id")
    pp.add_argument("token_id")
    pp.add_argument("--side", choices=["buy", "sell"], default="buy")
    pp.set_defaults(func=cmd_price)

    pc = sub.add_parser("collect-5m", help="Snapshot /predictions/5M + CLOB orderbooks")
    pc.add_argument("--out", default="data")
    pc.add_argument(
        "--every-seconds",
        type=float,
        default=None,
        help="Enable continuous collection mode (interval in seconds)",
    )
    pc.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=60.0,
        help="Max backoff on errors (default: 60)",
    )
    pc.add_argument(
        "--retention-hours", type=float, default=None, help="Prune snapshots older than N hours"
    )
    pc.set_defaults(func=cmd_collect_5m)

    pu = sub.add_parser("universe-5m", help="Build normalized market universe from /predictions/5M")
    pu.add_argument("--out", default="data/universe.json", help="Output JSON file path")
    pu.add_argument("--cross-check", action="store_true", help="Verify against Gamma API")
    pu.set_defaults(func=cmd_universe_5m)

    p15 = sub.add_parser("markets-15m", help="Heuristic fetch of 15-minute crypto interval markets")
    p15.add_argument("--limit", type=int, default=50)
    p15.set_defaults(func=cmd_markets_15m)

    pc15 = sub.add_parser("collect-15m", help="Snapshot /crypto/15M + CLOB orderbooks")
    pc15.add_argument("--out", default="data")
    pc15.set_defaults(func=cmd_collect_15m)

    pc15l = sub.add_parser(
        "collect-15m-loop", help="Continuously snapshot /crypto/15M + CLOB orderbooks"
    )
    pc15l.add_argument("--out", default="data")
    pc15l.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Collection interval in seconds (default: 60)",
    )
    pc15l.add_argument(
        "--max-backoff-seconds", type=float, default=60.0, help="Max backoff on errors"
    )
    pc15l.add_argument(
        "--retention-hours",
        type=float,
        default=24.0,
        help="Prune snapshots older than N hours (default: 24)",
    )
    pc15l.add_argument(
        "--max-snapshots",
        type=int,
        default=1440,
        help="Max snapshots to retain (default: 1440 ~= 24h at 60s)",
    )
    pc15l.add_argument(
        "--microstructure-interval-seconds",
        type=float,
        default=60.0,
        help="Seconds between microstructure analyses",
    )
    pc15l.add_argument(
        "--microstructure-target",
        type=str,
        default="bitcoin",
        help="Target market substring filter (e.g., 'bitcoin')",
    )
    pc15l.add_argument(
        "--spread-alert-threshold",
        type=float,
        default=DEFAULT_SPREAD_ALERT_THRESHOLD,
        help=f"Alert threshold for spread (default: {DEFAULT_SPREAD_ALERT_THRESHOLD})",
    )
    pc15l.add_argument(
        "--extreme-pin-threshold",
        type=float,
        default=DEFAULT_EXTREME_PIN_THRESHOLD,
        help=f"Alert threshold for extreme price pinning (default: {DEFAULT_EXTREME_PIN_THRESHOLD})",
    )
    pc15l.add_argument(
        "--depth-levels",
        type=int,
        default=DEFAULT_DEPTH_LEVELS,
        help=f"Number of book levels for depth calc (default: {DEFAULT_DEPTH_LEVELS})",
    )
    pc15l.set_defaults(func=cmd_collect_15m_loop)

    # Enhanced PnL verify command
    pnl = sub.add_parser(
        "pnl-verify",
        help="Verify PnL from fills with cash tracking, sanity checks, and liquidation value",
    )
    pnl.add_argument(
        "--input",
        default=None,
        help="Path to fills JSON/JSONL file (alternative to --data-dir)",
    )
    pnl.add_argument(
        "--data-dir",
        default=None,
        help="Data directory containing fills.json or fills.jsonl",
    )
    pnl.add_argument(
        "--books",
        default=None,
        help="Path to orderbooks JSON for liquidation value calculation",
    )
    pnl.add_argument(
        "--snapshot",
        default=None,
        help="Path to collector snapshot for orderbook data",
    )
    pnl.add_argument(
        "--since",
        default=None,
        help="ISO timestamp to filter fills (inclusive)",
    )
    pnl.add_argument(
        "--market",
        default=None,
        help="Market slug filter (substring match)",
    )
    pnl.add_argument(
        "--starting-cash",
        type=float,
        default=0.0,
        help="Starting cash balance (default: 0)",
    )
    pnl.add_argument(
        "--save-daily",
        action="store_true",
        help="Save daily summary to data/pnl/",
    )
    pnl.add_argument(
        "--daily-dir",
        default=None,
        help="Custom directory for daily summaries (default: data/pnl/)",
    )
    pnl.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format",
    )
    pnl.set_defaults(func=cmd_pnl_verify)

    ms = sub.add_parser("microstructure", help="Analyze market microstructure from snapshot")
    ms.add_argument("--snapshot", required=True, help="Path to snapshot JSON file")
    ms.add_argument("--target", type=str, default="bitcoin", help="Target market substring filter")
    ms.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Generate summary with alerts (default)",
    )
    ms.add_argument("--raw", action="store_true", help="Output raw analysis without summary")
    ms.add_argument(
        "--spread-threshold",
        type=float,
        default=DEFAULT_SPREAD_ALERT_THRESHOLD,
        help=f"Alert threshold for spread (default: {DEFAULT_SPREAD_ALERT_THRESHOLD})",
    )
    ms.add_argument(
        "--extreme-pin-threshold",
        type=float,
        default=DEFAULT_EXTREME_PIN_THRESHOLD,
        help=f"Alert threshold for extreme price pinning (default: {DEFAULT_EXTREME_PIN_THRESHOLD})",
    )
    ms.add_argument(
        "--depth-levels",
        type=int,
        default=DEFAULT_DEPTH_LEVELS,
        help=f"Number of book levels for depth calc (default: {DEFAULT_DEPTH_LEVELS})",
    )
    ms.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    ms.set_defaults(func=cmd_microstructure)

    # Binance commands
    bc = sub.add_parser("binance-collect", help="Collect Binance BTC market data (single snapshot)")
    bc.add_argument("--out", default="data/binance", help="Output directory")
    bc.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)")
    bc.add_argument("--intervals", nargs="+", default=["1m", "5m"], help="Kline intervals to fetch")
    bc.set_defaults(func=cmd_binance_collect)

    bcl = sub.add_parser("binance-loop", help="Continuously collect Binance data via WebSocket")
    bcl.add_argument("--out", default="data/binance", help="Output directory")
    bcl.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    bcl.add_argument(
        "--intervals", nargs="+", default=["1m", "5m"], help="Kline intervals to subscribe"
    )
    bcl.add_argument(
        "--snapshot-interval-seconds", type=float, default=5.0, help="Snapshot interval"
    )
    bcl.add_argument(
        "--max-reconnect-delay", type=float, default=60.0, help="Max reconnection delay"
    )
    bcl.add_argument("--retention-hours", type=float, default=None, help="Prune old files")
    bcl.set_defaults(func=cmd_binance_loop)

    bf = sub.add_parser("binance-align", help="Align Binance features to Polymarket snapshots")
    bf.add_argument("--binance-dir", default="data/binance", help="Binance data directory")
    bf.add_argument("--polymarket-dir", default="data", help="Polymarket data directory")
    bf.add_argument("--out", default="data/aligned_features.json", help="Output file")
    bf.add_argument("--tolerance", type=float, default=1.0, help="Alignment tolerance in seconds")
    bf.set_defaults(func=cmd_binance_features)

    hc = sub.add_parser("health-check", help="Check collector health and staleness SLA")
    hc.add_argument("--data-dir", default="data", help="Data directory containing snapshots")
    hc.add_argument(
        "--max-age-seconds",
        type=float,
        default=120.0,
        help="Max acceptable snapshot age (default: 120s)",
    )
    hc.add_argument("--prefix", default="snapshot_15m", help="Snapshot file prefix")
    hc.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    hc.add_argument("--fail", action="store_true", help="Exit with error code if unhealthy")
    hc.set_defaults(func=cmd_health_check)

    # Sports arbitrage commands
    sp = sub.add_parser("sports-scan", help="Scan for sports arbitrage opportunities")
    sp.add_argument("--bankroll", type=float, default=10000.0, help="Paper trading bankroll")
    sp.add_argument("--min-edge", type=float, default=2.0, help="Minimum edge percentage")
    sp.add_argument("--limit", type=int, default=10, help="Max opportunities to display")
    sp.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    sp.set_defaults(func=cmd_sports_scan)

    spt = sub.add_parser("sports-trade", help="Execute paper trades for arbitrage opportunities")
    spt.add_argument("--bankroll", type=float, default=10000.0, help="Paper trading bankroll")
    spt.add_argument("--scan", action="store_true", default=True, help="Scan before trading")
    spt.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    spt.set_defaults(func=cmd_sports_trade)

    sps = sub.add_parser("sports-stats", help="Show sports arbitrage strategy statistics")
    sps.add_argument("--bankroll", type=float, default=10000.0, help="Paper trading bankroll")
    sps.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    sps.set_defaults(func=cmd_sports_stats)

    # Orderbook imbalance backtest command
    ib = sub.add_parser(
        "imbalance-backtest",
        help="Backtest orderbook imbalance strategy on BTC interval markets",
    )
    ib.add_argument("--data-dir", default="data", help="Data directory containing snapshots")
    ib.add_argument(
        "--interval",
        choices=["5m", "15m"],
        default="15m",
        help="Market interval to analyze (default: 15m)",
    )
    ib.add_argument(
        "--k",
        type=int,
        default=3,
        choices=[1, 3, 5],
        help="Depth levels for imbalance calculation (default: 3)",
    )
    ib.add_argument(
        "--theta",
        type=float,
        default=0.70,
        help="Imbalance threshold 0.5-1.0 (default: 0.70)",
    )
    ib.add_argument(
        "--p-max",
        type=float,
        default=0.65,
        help="Max price to pay for position 0.5-1.0 (default: 0.65)",
    )
    ib.add_argument(
        "--target",
        type=str,
        default="bitcoin",
        help="Target market substring filter (default: bitcoin)",
    )
    ib.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep across k/theta/p_max combinations",
    )
    ib.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    ib.set_defaults(func=cmd_imbalance_backtest)

    args = p.parse_args()

    # Handle --raw flag for microstructure command
    if hasattr(args, "raw") and args.raw:
        args.summary = False

    args.func(args)


if __name__ == "__main__":
    main()
