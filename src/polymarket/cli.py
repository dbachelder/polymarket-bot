from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from . import gamma, clob


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_markets_5m(args: argparse.Namespace) -> None:
    # Best-effort: Gamma doesn't document a canonical "5M" tag in quickstart docs.
    # We'll search by title patterns and let us refine once we inspect real metadata.
    search = args.search or "5m"  # broad
    markets = gamma.get_markets(active=True, closed=False, limit=args.limit, offset=0, search=search)

    # Keep only likely 5-minute crypto markets (heuristic)
    out = []
    for m in markets:
        title = (m.get("question") or m.get("title") or "").lower()
        if "5" in title and "minute" in title:
            out.append(m)
        elif "5m" in title:
            out.append(m)
        elif "up or down" in title and ("5" in title or "5m" in title):
            out.append(m)

    _print(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "count": len(out),
            "markets": out,
        }
    )


def cmd_book(args: argparse.Namespace) -> None:
    book = clob.get_book(args.token_id)
    _print(book)


def cmd_price(args: argparse.Namespace) -> None:
    price = clob.get_price(args.token_id, side=args.side)
    _print(price)


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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
