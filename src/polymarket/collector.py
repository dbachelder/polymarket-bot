from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .clob import get_book
from .site import extract_5m_markets, fetch_predictions_page, parse_next_data


def collect_5m_snapshot(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_predictions_page("5M")
    data = parse_next_data(html)
    markets = extract_5m_markets(data)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"snapshot_5m_{ts}.json"

    payload: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
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
