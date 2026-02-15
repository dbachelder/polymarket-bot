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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
