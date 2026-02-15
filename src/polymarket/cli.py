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
        microstructure_interval_seconds=float(args.microstructure_interval_seconds),
        microstructure_target=args.microstructure_target,
        spread_alert_threshold=float(args.spread_alert_threshold),
        extreme_pin_threshold=float(args.extreme_pin_threshold),
        depth_levels=int(args.depth_levels),
    )


def cmd_pnl_verify(args: argparse.Namespace) -> None:
    """Verify PnL from fills data."""
    from .pnl import compute_pnl, load_fills_from_file, load_orderbooks_from_file

    input_path = Path(args.input)
    if not input_path.exists():
        print(json.dumps({"error": f"Input file not found: {args.input}"}), file=__import__("sys").stderr)
        raise SystemExit(1)

    fills = load_fills_from_file(input_path)

    # Load optional orderbooks for liquidation value
    orderbooks = None
    if args.books:
        books_path = Path(args.books)
        if books_path.exists():
            orderbooks = load_orderbooks_from_file(books_path)

    report = compute_pnl(fills, orderbooks=orderbooks)

    # Output format
    output = report.to_dict()

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        # Human-readable format
        print("=" * 60)
        print("PnL VERIFICATION REPORT")
        print("=" * 60)
        print(f"\nTotal Fills:      {output['summary']['total_fills']}")
        print(f"Unique Tokens:    {output['summary']['unique_tokens']}")
        print("\n--- PnL Breakdown ---")
        print(f"Realized PnL:     ${output['pnl']['realized_pnl']:,.2f}")
        print(f"Unrealized PnL:   ${output['pnl']['unrealized_pnl']:,.2f}")
        print(f"Total Fees:       ${output['pnl']['total_fees']:,.2f}")
        print(f"Net PnL:          ${output['pnl']['net_pnl']:,.2f}")
        print("\n--- Liquidation Analysis ---")
        print(f"Mark to Market:   ${output['liquidation']['mark_to_market']:,.2f}")
        print(f"Liquidation Val:  ${output['liquidation']['liquidation_value']:,.2f}")
        print(f"Discount:         ${output['liquidation']['liquidation_discount']:,.2f} ({output['liquidation']['discount_pct']:.1f}%)")

        if output['positions']:
            print(f"\n--- Open Positions ({len(output['positions'])}) ---")
            for pos in output['positions'][:10]:  # Limit to 10
                print(f"  {pos['token_id'][:40]}...")
                print(f"    Size: {pos['net_size']:,.2f} @ ${pos['avg_cost_basis']:.3f}")
                print(f"    Price: ${pos['current_price']:.3f} | Unrealized: ${pos['unrealized_pnl']:,.2f}")

        if output['warnings']:
            print("\n--- Warnings ---")
            for warning in output['warnings']:
                print(f"  ! {warning}")

        print("\n" + "=" * 60)


def cmd_microstructure(args: argparse.Namespace) -> None:
    """Analyze microstructure for markets in a snapshot."""
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(json.dumps({"error": f"Snapshot file not found: {args.snapshot}"}), file=__import__("sys").stderr)
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
            print(f"\nThresholds:")
            print(f"  Spread alert:     > {summary['spread_threshold']:.2f}")
            print(f"  Extreme pin:      <= {summary['extreme_pin_threshold']:.2f} or >= {1.0 - summary['extreme_pin_threshold']:.2f}")
            print(f"  Depth levels:     {summary['depth_levels']}")

            if summary['alerts']:
                print(f"\n--- ALERTS ({summary['alert_count']}) ---")
                for alert in summary['alerts']:
                    print(f"  ⚠ {alert}")
            else:
                print("\n--- ALERTS ---")
                print("  ✓ No alerts. Markets appear healthy.")

            if summary['market_summaries']:
                print(f"\n--- MARKET DETAILS ({len(summary['market_summaries'])}) ---")
                for ms in summary['market_summaries']:
                    print(f"\n  {ms['market_title']}")
                    print(f"    YES: bid={ms.get('yes_best_bid'):.2f} ask={ms.get('yes_best_ask'):.2f} "
                          f"spread={ms.get('yes_spread'):.2f} imbalance={ms.get('yes_imbalance', 0):+.3f}")
                    print(f"    NO:  bid={ms.get('no_best_bid'):.2f} ask={ms.get('no_best_ask'):.2f} "
                          f"spread={ms.get('no_spread'):.2f} imbalance={ms.get('no_imbalance', 0):+.3f}")
                    if ms.get('implied_sum'):
                        print(f"    Implied: YES_mid={ms.get('implied_yes_mid', 0):.2f} "
                              f"NO_mid={ms.get('implied_no_mid', 0):.2f} "
                              f"sum={ms.get('implied_sum', 0):.2f} "
                              f"consistency={ms.get('consistency_diff', 0):.3f}")

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
=======
def cmd_microstructure(args: argparse.Namespace) -> None:
    """Analyze microstructure for markets in a snapshot."""
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(json.dumps({"error": f"Snapshot file not found: {args.snapshot}"}), file=__import__("sys").stderr)
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
            print(f"\nThresholds:")
            print(f"  Spread alert:     > {summary['spread_threshold']:.2f}")
            print(f"  Extreme pin:      <= {summary['extreme_pin_threshold']:.2f} or >= {1.0 - summary['extreme_pin_threshold']:.2f}")
            print(f"  Depth levels:     {summary['depth_levels']}")

            if summary['alerts']:
                print(f"\n--- ALERTS ({summary['alert_count']}) ---")
                for alert in summary['alerts']:
                    print(f"  ⚠ {alert}")
            else:
                print("\n--- ALERTS ---")
                print("  ✓ No alerts. Markets appear healthy.")

            if summary['market_summaries']:
                print(f"\n--- MARKET DETAILS ({len(summary['market_summaries'])}) ---")
                for ms in summary['market_summaries']:
                    print(f"\n  {ms['market_title']}")
                    print(f"    YES: bid={ms.get('yes_best_bid'):.2f} ask={ms.get('yes_best_ask'):.2f} "
                          f"spread={ms.get('yes_spread'):.2f} imbalance={ms.get('yes_imbalance', 0):+.3f}")
                    print(f"    NO:  bid={ms.get('no_best_bid'):.2f} ask={ms.get('no_best_ask'):.2f} "
                          f"spread={ms.get('no_spread'):.2f} imbalance={ms.get('no_imbalance', 0):+.3f}")
                    if ms.get('implied_sum'):
                        print(f"    Implied: YES_mid={ms.get('implied_yes_mid', 0):.2f} "
                              f"NO_mid={ms.get('implied_no_mid', 0):.2f} "
                              f"sum={ms.get('implied_sum', 0):.2f} "
                              f"consistency={ms.get('consistency_diff', 0):.3f}")

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
>>>>>>> e6f5e61 (feat: add BTC 15m microstructure stats + sanity checks)


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
    pc.add_argument("--every-seconds", type=float, default=None, help="Enable continuous collection mode (interval in seconds)")
    pc.add_argument("--max-backoff-seconds", type=float, default=60.0, help="Max backoff on errors (default: 60)")
    pc.add_argument("--retention-hours", type=float, default=None, help="Prune snapshots older than N hours")
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

    pc15l = sub.add_parser("collect-15m-loop", help="Continuously snapshot /crypto/15M + CLOB orderbooks")
    pc15l.add_argument("--out", default="data")
    pc15l.add_argument("--interval-seconds", type=float, default=5.0, help="Collection interval in seconds")
    pc15l.add_argument("--max-backoff-seconds", type=float, default=60.0, help="Max backoff on errors")
    pc15l.add_argument("--retention-hours", type=float, default=None, help="Prune snapshots older than N hours")
    pc15l.add_argument("--microstructure-interval-seconds", type=float, default=60.0, help="Seconds between microstructure analyses")
    pc15l.add_argument("--microstructure-target", type=str, default="bitcoin", help="Target market substring filter (e.g., 'bitcoin')")
    pc15l.add_argument("--spread-alert-threshold", type=float, default=DEFAULT_SPREAD_ALERT_THRESHOLD, help=f"Alert threshold for spread (default: {DEFAULT_SPREAD_ALERT_THRESHOLD})")
    pc15l.add_argument("--extreme-pin-threshold", type=float, default=DEFAULT_EXTREME_PIN_THRESHOLD, help=f"Alert threshold for extreme price pinning (default: {DEFAULT_EXTREME_PIN_THRESHOLD})")
    pc15l.add_argument("--depth-levels", type=int, default=DEFAULT_DEPTH_LEVELS, help=f"Number of book levels for depth calc (default: {DEFAULT_DEPTH_LEVELS})")
    pc15l.set_defaults(func=cmd_collect_15m_loop)

    pnl = sub.add_parser("pnl-verify", help="Verify PnL from fills data (debunk screenshots)")
    pnl.add_argument("--input", required=True, help="Path to fills JSON file")
    pnl.add_argument("--books", default=None, help="Path to orderbooks JSON for liquidation value")
    pnl.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    pnl.set_defaults(func=cmd_pnl_verify)

    ms = sub.add_parser("microstructure", help="Analyze market microstructure from snapshot")
    ms.add_argument("--snapshot", required=True, help="Path to snapshot JSON file")
    ms.add_argument("--target", type=str, default="bitcoin", help="Target market substring filter")
    ms.add_argument("--summary", action="store_true", default=True, help="Generate summary with alerts (default)")
    ms.add_argument("--raw", action="store_true", help="Output raw analysis without summary")
    ms.add_argument("--spread-threshold", type=float, default=DEFAULT_SPREAD_ALERT_THRESHOLD, help=f"Alert threshold for spread (default: {DEFAULT_SPREAD_ALERT_THRESHOLD})")
    ms.add_argument("--extreme-pin-threshold", type=float, default=DEFAULT_EXTREME_PIN_THRESHOLD, help=f"Alert threshold for extreme price pinning (default: {DEFAULT_EXTREME_PIN_THRESHOLD})")
    ms.add_argument("--depth-levels", type=int, default=DEFAULT_DEPTH_LEVELS, help=f"Number of book levels for depth calc (default: {DEFAULT_DEPTH_LEVELS})")
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
    bcl.add_argument("--intervals", nargs="+", default=["1m", "5m"], help="Kline intervals to subscribe")
    bcl.add_argument("--snapshot-interval-seconds", type=float, default=5.0, help="Snapshot interval")
    bcl.add_argument("--max-reconnect-delay", type=float, default=60.0, help="Max reconnection delay")
    bcl.add_argument("--retention-hours", type=float, default=None, help="Prune old files")
    bcl.set_defaults(func=cmd_binance_loop)

    bf = sub.add_parser("binance-align", help="Align Binance features to Polymarket snapshots")
    bf.add_argument("--binance-dir", default="data/binance", help="Binance data directory")
    bf.add_argument("--polymarket-dir", default="data", help="Polymarket data directory")
    bf.add_argument("--out", default="data/aligned_features.json", help="Output file")
    bf.add_argument("--tolerance", type=float, default=1.0, help="Alignment tolerance in seconds")
    bf.set_defaults(func=cmd_binance_features)

    args = p.parse_args()

    # Handle --raw flag for microstructure command
    if hasattr(args, "raw") and args.raw:
        args.summary = False

    args.func(args)


if __name__ == "__main__":
    main()
