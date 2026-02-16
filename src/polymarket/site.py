from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class FiveMMarket:
    market_id: str
    slug: str
    question: str
    end_date: str
    clob_token_ids: tuple[str, str]
    maker_base_fee: int | None = None
    taker_base_fee: int | None = None
    fees_enabled: bool | None = None


@dataclass(frozen=True)
class CryptoIntervalEvent:
    """A live crypto interval event (5m/15m/etc) from /crypto/<interval>."""

    event_id: str
    event_slug: str
    title: str
    end_date: str
    market_id: str
    market_slug: str
    question: str
    clob_token_ids: tuple[str, str]
    maker_base_fee: int | None = None
    taker_base_fee: int | None = None
    fees_enabled: bool | None = None


_NEXT_DATA_RE = re.compile(
    r"<script[^>]*id=\"__NEXT_DATA__\"[^>]*>(.*?)</script>",
    re.DOTALL,
)

# Gamma API base URL
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def fetch_predictions_page(slug: str = "5M") -> dict:
    """Fetch events from the Gamma API by tag.

    Polymarket moved from /predictions/* routes to using the Gamma API.
    This function now returns the parsed JSON API response instead of HTML.

    Examples:
      - 5M  -> https://gamma-api.polymarket.com/events?tag=5M&active=true&closed=false&limit=100
      - weather -> https://gamma-api.polymarket.com/events?tag=weather&active=true&closed=false&limit=100
    """
    url = f"{GAMMA_API_BASE}/events"
    params = {
        "tag": slug,
        "active": "true",
        "closed": "false",
        "limit": "100",
    }
    r = httpx.get(url, params=params, timeout=30, headers={"User-Agent": "polymarket-bot/0.1"})
    r.raise_for_status()
    return r.json()


def fetch_crypto_interval_page(interval_slug: str = "15M") -> dict:
    """Fetch crypto interval events from the Gamma API.

    The /crypto/<interval> page now uses client-side fetching. We use the
    events API with appropriate filtering for crypto interval markets.
    """
    url = f"{GAMMA_API_BASE}/events"
    params = {
        "tag": "crypto",
        "active": "true",
        "closed": "false",
        "limit": "100",
    }
    r = httpx.get(url, params=params, timeout=30, headers={"User-Agent": "polymarket-bot/0.1"})
    r.raise_for_status()
    return r.json()


def parse_next_data(html: str) -> dict:
    """Parse __NEXT_DATA__ from HTML - kept for backward compatibility."""
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise ValueError("Could not find __NEXT_DATA__ JSON")
    return json.loads(m.group(1))


def extract_5m_markets(events_data: list[dict]) -> list[FiveMMarket]:
    """Extract 5M markets from Gamma API events response.

    The API returns a list of events for the 5M tag, each containing markets.
    We flatten all markets from all events.
    """
    out: list[FiveMMarket] = []

    for event in events_data:
        markets = event.get("markets", [])
        for m0 in markets:
            # Parse clobTokenIds from JSON string
            token_ids_str = m0.get("clobTokenIds", "[]")
            if isinstance(token_ids_str, str):
                try:
                    token_ids = json.loads(token_ids_str)
                except json.JSONDecodeError:
                    continue
            else:
                token_ids = token_ids_str

            if not token_ids or len(token_ids) != 2:
                continue

            out.append(
                FiveMMarket(
                    market_id=str(m0.get("id")),
                    slug=str(m0.get("slug")),
                    question=str(m0.get("question")),
                    end_date=str(m0.get("endDate")),
                    clob_token_ids=(str(token_ids[0]), str(token_ids[1])),
                    maker_base_fee=m0.get("makerBaseFee"),
                    taker_base_fee=m0.get("takerBaseFee"),
                    fees_enabled=m0.get("feesEnabled"),
                )
            )

    return out


def extract_crypto_interval_events(
    events_data: list[dict], interval_slug: str = "15M"
) -> list[CryptoIntervalEvent]:
    """Extract crypto interval events from Gamma API response.

    Filters events by title containing the interval phrase (e.g., "15 min").
    """
    interval_phrase = {
        "5M": "5 min",
        "15M": "15 min",
    }.get(interval_slug.upper())

    out: list[CryptoIntervalEvent] = []

    for event in events_data:
        title = str(event.get("title") or "")

        # Filter by interval phrase in title
        if interval_phrase and interval_phrase not in title.lower():
            continue

        markets = event.get("markets", [])
        if not markets:
            continue

        m0 = markets[0]

        # Parse clobTokenIds from JSON string
        token_ids_str = m0.get("clobTokenIds", "[]")
        if isinstance(token_ids_str, str):
            try:
                token_ids = json.loads(token_ids_str)
            except json.JSONDecodeError:
                continue
        else:
            token_ids = token_ids_str

        if not token_ids or len(token_ids) != 2:
            continue

        out.append(
            CryptoIntervalEvent(
                event_id=str(event.get("id")),
                event_slug=str(event.get("slug")),
                title=title,
                end_date=str(event.get("endDate")),
                market_id=str(m0.get("id")),
                market_slug=str(m0.get("slug")),
                question=str(m0.get("question")),
                clob_token_ids=(str(token_ids[0]), str(token_ids[1])),
                maker_base_fee=m0.get("makerBaseFee"),
                taker_base_fee=m0.get("takerBaseFee"),
                fees_enabled=m0.get("feesEnabled"),
            )
        )

    return out
