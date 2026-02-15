from __future__ import annotations

import argparse
import json

from . import clob


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
    from pathlib import Path

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
    from pathlib import Path

    from .collector import collect_15m_snapshot

    out = collect_15m_snapshot(Path(args.out))
    print(str(out))


def cmd_universe_5m(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .universe import build_universe, save_universe

    data = build_universe(cross_check=args.cross_check)
    out_path = save_universe(data, Path(args.out))
    print(str(out_path))


def cmd_collect_15m_loop(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .collector_loop import collect_15m_loop

    collect_15m_loop(
        out_dir=Path(args.out),
        interval_seconds=float(args.interval_seconds),
        max_backoff_seconds=float(args.max_backoff_seconds),
        retention_hours=args.retention_hours,
    )


def cmd_pnl_verify(args: argparse.Namespace) -> None:
    """Verify PnL from fills data."""
    from pathlib import Path

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


def cmd_hourly_digest(args: argparse.Namespace) -> None:
    """Generate hourly digest report from snapshot data."""
    from pathlib import Path

    from .report import generate_hourly_digest

    data_dir = Path(args.data_dir)
    digest = generate_hourly_digest(data_dir, interval_seconds=args.interval_seconds)
    output = digest.to_dict()

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        # Human-readable format
        health = output["collector_health"]
        btc = output["btc_microstructure"]
        strategy = output["paper_strategy"]

        print("=" * 60)
        print("POLYMARKET HOURLY DIGEST")
        print("=" * 60)
        print(f"Generated: {output['generated_at']}")

        print("\n--- Collector Health ---")
        if health["latest_snapshot_at"]:
            print(f"Latest snapshot: {health['latest_snapshot_at']}")
            if health["freshness_seconds"] is not None:
                print(f"Freshness: {health['freshness_seconds']:.1f}s ago")
        else:
            print("Latest snapshot: None found")
        print(f"Snapshots (last hour): {health['snapshots_last_hour']}/{health['expected_snapshots']}")
        print(f"Capture rate: {health['capture_rate_pct']:.1f}%")
        if health["backoff_evidence"]:
            print("⚠️  Backoff detected (gaps in snapshot sequence)")

        print("\n--- BTC 15m Microstructure ---")
        print(f"Best bid: {btc['best_bid']}")
        print(f"Best ask: {btc['best_ask']}")
        if btc["spread"] is not None:
            print(f"Spread: {btc['spread']:.4f} ({btc['spread_bps']:.2f} bps)")
        print(f"Bid depth (top 5): {btc['best_bid_depth']:,.2f}")
        print(f"Ask depth (top 5): {btc['best_ask_depth']:,.2f}")
        if btc["depth_imbalance"] is not None:
            imbalance_pct = btc["depth_imbalance"] * 100
            side = "bid" if imbalance_pct > 0 else "ask"
            print(f"Depth imbalance: {imbalance_pct:+.1f}% ({side} heavy)")

        print("\n--- Paper Strategy (Momentum) ---")
        print(f"Signal: {strategy['signal'].upper()}")
        print(f"Confidence: {strategy['confidence']*100:.0f}%")
        if strategy["mid_price_change_1h"] is not None:
            print(f"1h price change: {strategy['mid_price_change_1h']:+.2f}%")
        print(f"Reasoning: {strategy['reasoning']}")

        print("\n" + "=" * 60)


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
    pc15l.set_defaults(func=cmd_collect_15m_loop)

    pnl = sub.add_parser("pnl-verify", help="Verify PnL from fills data (debunk screenshots)")
    pnl.add_argument("--input", required=True, help="Path to fills JSON file")
    pnl.add_argument("--books", default=None, help="Path to orderbooks JSON for liquidation value")
    pnl.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    pnl.set_defaults(func=cmd_pnl_verify)

    hd = sub.add_parser("hourly-digest", help="Generate hourly report from 15m snapshots")
    hd.add_argument("--data-dir", default="data", help="Directory containing snapshot files")
    hd.add_argument("--interval-seconds", type=float, default=5.0, help="Expected collection interval")
    hd.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    hd.set_defaults(func=cmd_hourly_digest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
