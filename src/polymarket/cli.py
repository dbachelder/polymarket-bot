from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from . import gamma, clob


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


def cmd_collect_5m(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .collector import collect_5m_snapshot

    out = collect_5m_snapshot(Path(args.out))
    print(str(out))


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
    pc.set_defaults(func=cmd_collect_5m)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
