from __future__ import annotations

import httpx

from .endpoints import GAMMA_BASE


def _client(timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(
        base_url=GAMMA_BASE, timeout=timeout, headers={"User-Agent": "polymarket-bot/0.1"}
    )


def get_events(
    active: bool = True, closed: bool = False, limit: int = 50, offset: int = 0
) -> list[dict]:
    """Fetch events from Gamma API."""
    params = {
        "active": str(active).lower(),
        "closed": str(closed).lower(),
        "limit": limit,
        "offset": offset,
    }
    with _client() as c:
        r = c.get("/events", params=params)
        r.raise_for_status()
        return r.json()


def get_markets(
    *,
    active: bool = True,
    closed: bool = False,
    limit: int = 100,
    offset: int = 0,
    tag: str | None = None,
    slug: str | None = None,
    search: str | None = None,
) -> list[dict]:
    """Fetch markets from Gamma API.

    Note: Gamma supports a bunch of query params; we keep this thin and pass through
    the most useful ones.
    """
    params: dict[str, str | int] = {
        "active": str(active).lower(),
        "closed": str(closed).lower(),
        "limit": limit,
        "offset": offset,
    }
    if tag:
        params["tag"] = tag
    if slug:
        params["slug"] = slug
    if search:
        params["search"] = search

    with _client() as c:
        r = c.get("/markets", params=params)
        r.raise_for_status()
        return r.json()
