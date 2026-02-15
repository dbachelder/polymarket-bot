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


_NEXT_DATA_RE = re.compile(
    r"<script[^>]*id=\"__NEXT_DATA__\"[^>]*>(.*?)</script>",
    re.DOTALL,
)


def fetch_predictions_page(slug: str = "5M") -> str:
    url = f"https://polymarket.com/predictions/{slug}"
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
