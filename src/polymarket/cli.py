from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import clob


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_markets_5m(args: argparse.Namespace) -> None:
    from .site import fetch_predictions_page, parse_next_data, extract_5m_markets

    html = fetch_predictions_page("5M")
    data = parse_next_data(html)
    ms = extract_5m_markets(data)
    out = [m.__dict__ for m in ms[: args.limit]]
    _print({"count": len(out), "markets": out})


def cmd_book(args: argparse.Namespace) -> None:
    book = clob.get_book(args.token_id)
    _print(book)


def cmd_price(args: argparse.Namespace) -> None:
    price = clob.get_price(args.token_id, side=args.side)
    _print(price)


def cmd_markets_15m(args: argparse.Namespace) -> None:
    from .site import fetch_crypto_page, parse_next_data, extract_15m_markets

    html = fetch_crypto_page("15M")
    data = parse_next_data(html)
    ms = extract_15m_markets(data)
    out = [m.__dict__ for m in ms[: args.limit]]
    _print({"count": len(out), "markets": out})


def cmd_collect_5m(args: argparse.Namespace) -> None:
    from .collector import collect_5m_snapshot

    out = collect_5m_snapshot(Path(args.out))
    print(str(out))


def cmd_collect_15m(args: argparse.Namespace) -> None:
    from .collector import collect_15m_snapshot, prune_snapshots

    out = collect_15m_snapshot(Path(args.out), use_backoff=args.backoff)
    print(str(out))

    if args.prune_hours > 0:
        deleted = prune_snapshots(
            Path(args.out),
            pattern="snapshot_15m_*.json",
            retention_hours=args.prune_hours,
        )
        if deleted > 0:
            print(f"Pruned {deleted} old snapshots")


def cmd_collect_15m_loop(args: argparse.Namespace) -> None:
    from .collector import collect_15m_snapshot, prune_snapshots

    out_dir = Path(args.out)
    interval = args.interval
    prune_hours = args.prune_hours

    print(f"Starting 15M collection loop: interval={interval}s, out={out_dir}")

    try:
        while True:
            try:
                out = collect_15m_snapshot(out_dir, use_backoff=True)
                print(f"[{time.strftime('%H:%M:%S')}] Saved: {out.name}")

                if prune_hours > 0:
                    deleted = prune_snapshots(
                        out_dir,
                        pattern="snapshot_15m_*.json",
                        retention_hours=prune_hours,
                    )
                    if deleted > 0:
                        print(f"  Pruned {deleted} old snapshots")

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\\nStopped.")


def cmd_universe_5m(args: argparse.Namespace) -> None:
    from .universe import build_universe, save_universe

    data = build_universe(cross_check=args.cross_check)
    out_path = save_universe(data, Path(args.out))
    print(str(out_path))


def main() -> None:
    p = argparse.ArgumentParser(prog="polymarket")
    sub = p.add_subparsers(dest="cmd", required=True)

    p5 = sub.add_parser("markets-5m", help="Heuristic fetch of likely 5-minute markets")
    p5.add_argument("--limit", type=int, default=50)
    p5.add_argument("--search", type=str, default=None)
    p5.set_defaults(func=cmd_markets_5m)

    p15 = sub.add_parser("markets-15m", help="Fetch 15-minute crypto markets from /crypto/15M")
    p15.add_argument("--limit", type=int, default=50)
    p15.set_defaults(func=cmd_markets_15m)

    pb = sub.add_parser("book", help="Fetch CLOB orderbook for a token_id")
    pb.add_argument("token_id")
    pb.set_defaults(func=cmd_book)

    pp = sub.add_parser("price", help="Fetch CLOB price for a token_id")
    pp.add_argument("token_id")
    pp.add_argument("--side", choices=["buy", "sell"], default="buy")
    pp.set_defaults(func=cmd_price)

    pc = sub.add_parser("collect-5m", help="Snapshot /predictions/5M + CLOB orderbooks")
    pc.add_argument("--out", default="data")
    pc.set_defaults(func=cmd_collect_5m)

    pc15 = sub.add_parser("collect-15m", help="Snapshot /crypto/15M + CLOB orderbooks")
    pc15.add_argument("--out", default="data")
    pc15.add_argument(
        "--backoff",
        action="store_true",
        default=True,
        help="Use exponential backoff on rate limits",
    )
    pc15.add_argument(
        "--no-backoff", dest="backoff", action="store_false", help="Disable exponential backoff"
    )
    pc15.add_argument("--prune-hours", type=float, default=0, help="Retention hours (0=disable)")
    pc15.set_defaults(func=cmd_collect_15m)

    pc15l = sub.add_parser("collect-15m-loop", help="Continuous 15M snapshot loop")
    pc15l.add_argument("--out", default="data")
    pc15l.add_argument("--interval", type=float, default=5.0, help="Seconds between snapshots")
    pc15l.add_argument(
        "--prune-hours", type=float, default=24.0, help="Retention hours (0=disable)"
    )
    pc15l.set_defaults(func=cmd_collect_15m_loop)

    pu = sub.add_parser("universe-5m", help="Build normalized market universe from /predictions/5M")
    pu.add_argument("--out", default="data/universe.json", help="Output JSON file path")
    pu.add_argument("--cross-check", action="store_true", help="Verify against Gamma API")
    pu.set_defaults(func=cmd_universe_5m)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
