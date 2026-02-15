from __future__ import annotations

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
