from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .paper_trading import PaperTradingEngine
from .site import extract_5m_markets, fetch_predictions_page, parse_next_data


def _best_ask(book: dict) -> Decimal | None:
    asks = (book or {}).get("asks") or []
    if not asks:
        return None
    try:
        return min(Decimal(str(a["price"])) for a in asks)
    except Exception:
        return None


def run_btc_preclose_paper(
    *,
    data_dir: Path,
    window_seconds: int = 120,
    cheap_price: Decimal = Decimal("0.03"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
) -> dict[str, Any]:
    """Paper-trade cheap-side trigger on BTC 5m markets near close.

    For each market on /predictions/5M ending within `window_seconds`, fetch YES/NO books.
    If either side's best ask <= cheap_price, record a paper BUY at that ask.

    This produces fills so we can verify whether near-close "cheap side" has edge.
    """

    now = datetime.now(UTC)

    html = fetch_predictions_page("5M")
    nd = parse_next_data(html)
    markets = extract_5m_markets(nd)

    engine = PaperTradingEngine(data_dir=data_dir, starting_cash=starting_cash)

    scanned = 0
    near_close = 0
    fills = 0
    triggers: list[dict[str, Any]] = []

    # Import here so the module stays pure / testable.
    from .clob import get_book

    for m in markets:
        scanned += 1
        try:
            end_dt = datetime.fromisoformat(str(m.end_date).replace("Z", "+00:00"))
        except Exception:
            continue

        ttc = (end_dt - now).total_seconds()
        if ttc < 0 or ttc > float(window_seconds):
            continue

        near_close += 1

        yes_id, no_id = m.clob_token_ids
        yes_book = get_book(yes_id)
        no_book = get_book(no_id)

        yes_ask = _best_ask(yes_book)
        no_ask = _best_ask(no_book)

        token_id = None
        side_label = None
        px: Decimal | None = None

        if yes_ask is not None and yes_ask <= cheap_price:
            token_id = yes_id
            side_label = "yes"
            px = yes_ask
        if no_ask is not None and no_ask <= cheap_price:
            if px is None or no_ask < px:
                token_id = no_id
                side_label = "no"
                px = no_ask

        if token_id and px is not None:
            fill = engine.record_fill(
                token_id=token_id,
                side="buy",
                size=size,
                price=px,
                fee=Decimal("0"),
                market_slug=m.slug,
                market_question=m.question,
            )
            fills += 1
            triggers.append(
                {
                    "market_slug": m.slug,
                    "question": m.question,
                    "ends_at": m.end_date,
                    "time_to_close_seconds": round(ttc, 3),
                    "side": side_label,
                    "token_id": token_id,
                    "price": str(px),
                    "size": str(size),
                    "fill_timestamp": fill.timestamp,
                }
            )

    return {
        "timestamp": now.isoformat(),
        "window_seconds": window_seconds,
        "cheap_price": str(cheap_price),
        "size": str(size),
        "markets_scanned": scanned,
        "candidates_near_close": near_close,
        "fills_recorded": fills,
        "triggers": triggers,
    }
