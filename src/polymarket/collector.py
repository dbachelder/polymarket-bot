from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .clob import get_book
from .site import (
    extract_5m_markets,
    extract_crypto_interval_events,
    fetch_crypto_interval_page,
    fetch_predictions_page,
    parse_next_data,
)


def collect_5m_snapshot(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_predictions_page("5M")
    data = parse_next_data(html)
    markets = extract_5m_markets(data)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"snapshot_5m_{ts}.json"

    payload: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_url": "https://polymarket.com/predictions/5M",
        "count": len(markets),
        "markets": [],
    }

    for m in markets:
        yes_id, no_id = m.clob_token_ids
        payload["markets"].append(
            {
                **asdict(m),
                "books": {
                    "yes": get_book(yes_id),
                    "no": get_book(no_id),
                },
            }
        )

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def collect_15m_snapshot(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_crypto_interval_page("15M")
    data = parse_next_data(html)
    events = extract_crypto_interval_events(data, interval_slug="15M")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"snapshot_15m_{ts}.json"

    payload: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_url": "https://polymarket.com/crypto/15M",
        "count": len(events),
        "markets": [],
    }

    for ev in events:
        yes_id, no_id = ev.clob_token_ids
        payload["markets"].append(
            {
                **asdict(ev),
                "books": {
                    "yes": get_book(yes_id),
                    "no": get_book(no_id),
                },
            }
        )

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path
