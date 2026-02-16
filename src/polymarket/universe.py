"""Market universe builder for extracting and normalizing Polymarket market data.

Parses /predictions/5M for event/market metadata and optionally cross-checks with Gamma API.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import gamma
from .site import fetch_predictions_page


@dataclass(frozen=True)
class MarketUniverseEntry:
    """Normalized market universe entry with event and market metadata."""

    # Event/EventSeries identifiers
    event_id: str
    event_slug: str
    event_title: str
    series_id: str | None
    series_slug: str | None

    # Market identifiers
    market_id: str
    market_slug: str
    condition_id: str
    question: str

    # Token identifiers (CLOB)
    yes_token_id: str
    no_token_id: str

    # Timing
    start_time: str | None  # When trading starts
    end_time: str  # When market resolves
    created_at: str | None

    # Resolution
    resolution_source: str | None
    description: str | None

    # Fees
    maker_base_fee: int | None
    taker_base_fee: int | None
    fees_enabled: bool | None

    # Gamma cross-check (populated if verified)
    gamma_verified: bool = False
    gamma_market_data: dict[str, Any] | None = None


def _extract_series_info(event_data: dict) -> tuple[str | None, str | None]:
    """Extract series id and slug from event data."""
    series_list = event_data.get("series") or []
    if series_list and isinstance(series_list, list):
        first_series = series_list[0]
        if isinstance(first_series, dict):
            return (
                str(first_series.get("id")),
                str(first_series.get("slug")),
            )
    return None, None


def _parse_event_market_data(events_data: list[dict]) -> list[tuple[dict, dict]]:
    """Extract (event_data, market_data) tuples from Gamma API events.

    Returns a list of tuples where each tuple contains:
    - event_data: The event/series metadata
    - market_data: The associated market (first market in the event)
    """
    event_market_pairs: list[tuple[dict, dict]] = []

    for event_data in events_data:
        markets = event_data.get("markets") or []
        if not markets:
            continue
        market_data = markets[0]

        # Parse clobTokenIds from JSON string
        token_ids_str = market_data.get("clobTokenIds", "[]")
        if isinstance(token_ids_str, str):
            try:
                token_ids = json.loads(token_ids_str)
            except json.JSONDecodeError:
                continue
        else:
            token_ids = token_ids_str

        if not token_ids or len(token_ids) != 2:
            continue
        event_market_pairs.append((event_data, market_data))

    return event_market_pairs


def build_universe_from_site() -> list[MarketUniverseEntry]:
    """Build market universe by fetching from Gamma API.

    Returns a list of normalized MarketUniverseEntry objects.
    """
    events = fetch_predictions_page("5M")
    pairs = _parse_event_market_data(events)

    entries: list[MarketUniverseEntry] = []
    for event_data, market_data in pairs:
        series_id, series_slug = _extract_series_info(event_data)

        # Parse clobTokenIds from JSON string
        token_ids_str = market_data.get("clobTokenIds", "[]")
        if isinstance(token_ids_str, str):
            try:
                token_ids = json.loads(token_ids_str)
            except json.JSONDecodeError:
                continue
        else:
            token_ids = token_ids_str

        if not token_ids or len(token_ids) != 2:
            continue

        entry = MarketUniverseEntry(
            event_id=str(event_data.get("id")),
            event_slug=str(event_data.get("slug")),
            event_title=str(event_data.get("title")),
            series_id=series_id,
            series_slug=series_slug,
            market_id=str(market_data.get("id")),
            market_slug=str(market_data.get("slug")),
            condition_id=str(market_data.get("conditionId")),
            question=str(market_data.get("question")),
            yes_token_id=str(token_ids[0]),
            no_token_id=str(token_ids[1]),
            start_time=_normalize_time(market_data.get("eventStartTime"))
            or _normalize_time(market_data.get("startDate")),
            end_time=str(market_data.get("endDate")),
            created_at=_normalize_time(market_data.get("createdAt")),
            resolution_source=_get_resolution_source(market_data, event_data),
            description=str(market_data.get("description"))
            if market_data.get("description")
            else None,
            maker_base_fee=market_data.get("makerBaseFee"),
            taker_base_fee=market_data.get("takerBaseFee"),
            fees_enabled=market_data.get("feesEnabled"),
        )
        entries.append(entry)

    return entries


def _normalize_time(ts: Any) -> str | None:
    """Normalize timestamp to ISO format string, or None if invalid."""
    if not ts:
        return None
    return str(ts)


def _get_resolution_source(market_data: dict, event_data: dict) -> str | None:
    """Extract resolution source from market or event data."""
    # Try market level first, then event level
    for src in [market_data.get("resolution_source"), market_data.get("resolutionSource")]:
        if src:
            return str(src)
    for src in [event_data.get("resolutionSource"), event_data.get("resolution_source")]:
        if src:
            return str(src)
    return None


def cross_check_with_gamma(
    entries: list[MarketUniverseEntry],
    lookup_by: str = "slug",
) -> list[MarketUniverseEntry]:
    """Cross-check universe entries against Gamma API.

    Args:
        entries: List of universe entries to verify
        lookup_by: How to lookup in Gamma ('slug' or 'id')

    Returns:
        List of entries with gamma_verified and gamma_market_data populated
    """
    verified_entries: list[MarketUniverseEntry] = []

    for entry in entries:
        gamma_data = None
        try:
            if lookup_by == "slug":
                markets = gamma.get_markets(slug=entry.market_slug, limit=1)
            else:
                # Try to get by ID via events endpoint
                events = gamma.get_events(active=True, limit=50)
                markets = []
                for ev in events:
                    for m in ev.get("markets", []):
                        if str(m.get("id")) == entry.market_id:
                            markets.append(m)
                            break

            if markets:
                gamma_data = markets[0]
                verified = str(gamma_data.get("id")) == entry.market_id
            else:
                verified = False

        except Exception:
            verified = False

        # Create new entry with verification data
        verified_entry = MarketUniverseEntry(
            event_id=entry.event_id,
            event_slug=entry.event_slug,
            event_title=entry.event_title,
            series_id=entry.series_id,
            series_slug=entry.series_slug,
            market_id=entry.market_id,
            market_slug=entry.market_slug,
            condition_id=entry.condition_id,
            question=entry.question,
            yes_token_id=entry.yes_token_id,
            no_token_id=entry.no_token_id,
            start_time=entry.start_time,
            end_time=entry.end_time,
            created_at=entry.created_at,
            resolution_source=entry.resolution_source,
            description=entry.description,
            maker_base_fee=entry.maker_base_fee,
            taker_base_fee=entry.taker_base_fee,
            fees_enabled=entry.fees_enabled,
            gamma_verified=verified,
            gamma_market_data=gamma_data if verified else None,
        )
        verified_entries.append(verified_entry)

    return verified_entries


def build_universe(
    *,
    cross_check: bool = False,
) -> dict[str, Any]:
    """Build complete market universe.

    Args:
        cross_check: Whether to verify against Gamma API

    Returns:
        Dictionary with metadata and list of universe entries
    """
    entries = build_universe_from_site()

    verification_stats = {"attempted": cross_check, "verified_count": 0}

    if cross_check:
        entries = cross_check_with_gamma(entries)
        verification_stats["verified_count"] = sum(1 for e in entries if e.gamma_verified)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_url": "https://polymarket.com/predictions/5M",
        "count": len(entries),
        "verification": verification_stats,
        "markets": [asdict(e) for e in entries],
    }


def save_universe(
    data: dict[str, Any],
    out_path: Path,
) -> Path:
    """Save universe data to JSON file.

    Args:
        data: Universe data dictionary
        out_path: Output file path

    Returns:
        Path to saved file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out_path
