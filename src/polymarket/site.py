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


def fetch_predictions_page(slug: str = "5M") -> str:
    url = f"https://polymarket.com/predictions/{slug}"
    r = httpx.get(url, timeout=30, headers={"User-Agent": "polymarket-bot/0.1"})
    r.raise_for_status()
    return r.text


def fetch_crypto_interval_page(interval_slug: str = "15M") -> str:
    """Fetch the Polymarket crypto interval hub page, e.g. /crypto/15M."""

    url = f"https://polymarket.com/crypto/{interval_slug}"
    r = httpx.get(url, timeout=30, headers={"User-Agent": "polymarket-bot/0.1"})
    r.raise_for_status()
    return r.text


def parse_next_data(html: str) -> dict:
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise ValueError("Could not find __NEXT_DATA__ JSON")
    return json.loads(m.group(1))


def extract_5m_markets(next_data: dict) -> list[FiveMMarket]:
    # Path discovered empirically from the /predictions/5M page.
    dehydrated = next_data["props"]["pageProps"]["dehydratedState"]["queries"]

    # Find the query that contains pages with tagLabel=5M (stable-ish)
    pages: list[dict] = []
    for q in dehydrated:
        state = q.get("state", {})
        data = state.get("data")
        if not isinstance(data, dict):
            continue
        candidate_pages = data.get("pages")
        if not isinstance(candidate_pages, list) or not candidate_pages:
            continue
        # tagLabel lives at the *page* level
        if candidate_pages[0].get("tagLabel") == "5M":
            pages = candidate_pages
            break

    if not pages:
        raise ValueError("Could not locate 5M pages payload in dehydratedState")

    results: list[dict] = []
    for p in pages:
        results.extend(p.get("results") or [])

    out: list[FiveMMarket] = []
    for item in results:
        markets = item.get("markets") or []
        if not markets:
            continue
        m0 = markets[0]
        token_ids = m0.get("clobTokenIds")
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
    next_data: dict, interval_slug: str = "15M"
) -> list[CryptoIntervalEvent]:
    """Extract live interval events (e.g. 15m) from /crypto/<interval>.

    Observed structure: dehydratedState includes a query whose state.data.pages[*].events
    is a list of events (BTC/ETH/etc). Each event embeds a single market with clobTokenIds.
    """

    dehydrated = next_data["props"]["pageProps"]["dehydratedState"]["queries"]

    interval_phrase = {
        "5M": "5 min",
        "15M": "15 min",
    }.get(interval_slug.upper(), None)

    pages: list[dict] = []
    for q in dehydrated:
        state = q.get("state", {})
        data = state.get("data")
        if not isinstance(data, dict):
            continue
        candidate_pages = data.get("pages")
        if not isinstance(candidate_pages, list) or not candidate_pages:
            continue
        # /crypto/<interval> seems to put events into pages[0].events.
        if isinstance(candidate_pages[0], dict) and isinstance(
            candidate_pages[0].get("events"), list
        ):
            pages = candidate_pages
            break

    if not pages:
        raise ValueError("Could not locate crypto interval events payload in dehydratedState")

    events: list[dict] = []
    for p in pages:
        events.extend(p.get("events") or [])

    out: list[CryptoIntervalEvent] = []
    for ev in events:
        title = str(ev.get("title") or "")
        if interval_phrase and interval_phrase not in title.lower():
            # Defensive: if Polymarket ever includes mixed intervals.
            continue

        markets = ev.get("markets") or []
        if not markets:
            continue
        m0 = markets[0]
        token_ids = m0.get("clobTokenIds")
        if not token_ids or len(token_ids) != 2:
            continue

        out.append(
            CryptoIntervalEvent(
                event_id=str(ev.get("id")),
                event_slug=str(ev.get("slug")),
                title=title,
                end_date=str(ev.get("endDate")),
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
