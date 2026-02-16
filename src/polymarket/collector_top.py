from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .clob import get_book
from .gamma import get_markets


def collect_top_snapshot(
    out_dir: Path,
    *,
    limit: int = 200,
    offset: int = 0,
    search: str | None = None,
) -> Path:
    """Collect a broad snapshot of active markets via Gamma + CLOB books.

    This is intended for "non-crypto vertical" scanners that want a wide view of
    currently active markets.

    Args:
        out_dir: output directory
        limit: number of markets to fetch from Gamma
        offset: Gamma offset
        search: optional Gamma search string

    Returns:
        Path to written snapshot JSON file
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    markets = get_markets(active=True, closed=False, limit=int(limit), offset=int(offset), search=search)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"snapshot_top_{ts}.json"

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "gamma": {
                "endpoint": "/markets",
                "active": True,
                "closed": False,
                "limit": int(limit),
                "offset": int(offset),
                "search": search,
            }
        },
        "count": len(markets),
        "markets": [],
    }

    for m in markets:
        # Gamma returns token ids as a JSON-encoded string list like
        # '["<YES>", "<NO>"]'
        token_ids: list[str] = []
        try:
            raw = m.get("clobTokenIds")
            if isinstance(raw, str):
                token_ids = list(json.loads(raw))
            elif isinstance(raw, list):
                token_ids = raw
        except Exception:
            token_ids = []

        books: dict[str, Any] | None = None
        if len(token_ids) >= 2:
            yes_id, no_id = token_ids[0], token_ids[1]
            books = {"yes": get_book(yes_id), "no": get_book(no_id)}

        payload["markets"].append({**m, "books": books})

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path
