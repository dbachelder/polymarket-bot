from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .clob import get_book
from .site import fetch_predictions_page
from .strategy_weather import _is_weather_market


def collect_weather_snapshot(out_dir: Path, *, max_markets: int = 15) -> Path:
    """Collect a snapshot of Polymarket weather-related markets + CLOB orderbooks.

    Uses the Gamma API to fetch weather-tagged events.
    NOTE: The existing weather scanner currently looks for the latest
    `snapshot_5m_*.json` file. For compatibility, we write the weather snapshot
    using that naming scheme *and* a clearer `snapshot_weather_*.json` copy.

    Returns:
        Path to the written snapshot JSON file (the snapshot_5m_* path)
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    events = fetch_predictions_page("weather")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path_5m = out_dir / f"snapshot_5m_{ts}.json"
    out_path_weather = out_dir / f"snapshot_weather_{ts}.json"

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {"predictions": {"url": "https://polymarket.com/predictions/weather"}},
        "count": 0,
        "markets": [],
    }

    seen_ids: set[str] = set()

    for event in events:
        if len(payload["markets"]) >= int(max_markets):
            break

        markets = event.get("markets", [])
        for m in markets:
            if len(payload["markets"]) >= int(max_markets):
                break

            market_id = str(m.get("id") or "")
            if not market_id or market_id in seen_ids:
                continue
            seen_ids.add(market_id)

            question = str(m.get("question") or m.get("title") or "")
            if not _is_weather_market(question):
                continue

            # Parse clobTokenIds from JSON string
            token_ids_str = m.get("clobTokenIds", "[]")
            if isinstance(token_ids_str, str):
                try:
                    token_ids = json.loads(token_ids_str)
                except json.JSONDecodeError:
                    continue
            else:
                token_ids = token_ids_str

            if not isinstance(token_ids, list) or len(token_ids) != 2:
                continue

            yes_id, no_id = str(token_ids[0]), str(token_ids[1])
            try:
                books = {"yes": get_book(yes_id), "no": get_book(no_id)}
            except Exception:
                # Some markets show up on the site before CLOB books are available.
                # Keep the market but record missing books.
                books = {"yes": None, "no": None}

            payload["markets"].append(
                {
                    "market_id": market_id,
                    "slug": m.get("slug"),
                    "question": question,
                    "end_date": m.get("endDate"),
                    "clob_token_ids": [yes_id, no_id],
                    "books": books,
                    "fees_enabled": m.get("feesEnabled"),
                    "maker_base_fee": m.get("makerBaseFee"),
                    "taker_base_fee": m.get("takerBaseFee"),
                }
            )

    payload["count"] = len(payload["markets"])

    txt = json.dumps(payload, indent=2, sort_keys=True)
    out_path_5m.write_text(txt)
    out_path_weather.write_text(txt)

    return out_path_5m
