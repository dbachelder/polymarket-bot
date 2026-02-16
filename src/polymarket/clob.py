from __future__ import annotations

from typing import Any

import httpx

from .endpoints import CLOB_BASE


def _client(timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(
        base_url=CLOB_BASE, timeout=timeout, headers={"User-Agent": "polymarket-bot/0.1"}
    )


def get_price(token_id: str, side: str = "buy") -> dict:
    with _client() as c:
        r = c.get("/price", params={"token_id": token_id, "side": side})
        r.raise_for_status()
        return r.json()


def get_book(token_id: str) -> dict:
    with _client() as c:
        r = c.get("/book", params={"token_id": token_id})
        r.raise_for_status()
        return r.json()


def get_best_prices(book: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract best bid and ask prices from a CLOB book response.

    The CLOB API returns bids sorted ascending (worst first) and asks sorted
    descending (worst first). This function correctly identifies the best
    prices by sorting appropriately.

    Args:
        book: CLOB book response dict with 'bids' and 'asks' lists.
              Each list item has 'price' and 'size' keys.

    Returns:
        Tuple of (best_bid, best_ask) where best_bid is the highest bid
        and best_ask is the lowest ask. None if no orders on that side.
    """
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    # Sort bids descending (highest first) - best bid is the highest price
    sorted_bids = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
    # Sort asks ascending (lowest first) - best ask is the lowest price
    sorted_asks = sorted(asks, key=lambda x: float(x["price"]))

    best_bid = float(sorted_bids[0]["price"]) if sorted_bids else None
    best_ask = float(sorted_asks[0]["price"]) if sorted_asks else None

    return best_bid, best_ask
