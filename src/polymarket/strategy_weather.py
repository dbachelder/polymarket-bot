"""Weather forecast latency arbitrage strategy.

Hypothesis: Weather prediction markets on Polymarket exhibit predictable
mispricing due to forecast latency—the delay between when professional
weather models update and when market prices adjust.

Entry Rules:
1. Poll GFS + ECMWF + ICON models via Open-Meteo API
2. Calculate model consensus: require 3+ models agreeing
3. Compare consensus probability to Polymarket implied probability
4. Buy YES when market price < 0.15 but model suggests >70% probability
5. Buy NO when market price > 0.45 but models show <20% probability

Exit Rules:
1. Sell when market reprices toward model consensus
2. Hold to resolution if no reprice within 12 hours
3. Stop if opposing model run shifts forecast
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .paper_trading import PaperTradingEngine
from .weather import (
    ModelConsensus,
    get_consensus_for_cities,
)

logger = logging.getLogger(__name__)

# Cities to monitor (expanded universe for more opportunities)
DEFAULT_CITIES = [
    "nyc", "london", "chicago", "seattle", "atlanta", "dallas", "miami",
    "los_angeles", "san_francisco", "boston", "washington", "philadelphia",
    "denver", "phoenix", "houston", "detroit", "minneapolis", "portland",
    "las_vegas", "san_diego", "austin", "nashville", "new_orleans",
    "seoul", "tokyo", "paris", "berlin", "sydney", "toronto", "vancouver",
]

# Entry thresholds (LOOSENED to increase signal frequency - per task requirements)
YES_ENTRY_MAX_PRICE = 0.35  # Was 0.25, now 0.35
YES_ENTRY_MIN_PROBABILITY = 0.55  # Was 0.60, now 0.55
NO_ENTRY_MIN_PRICE = 0.30  # Was 0.35, now 0.30
NO_ENTRY_MAX_PROBABILITY = 0.40  # Was 0.30, now 0.40

# Position sizing - reduced to allow more positions
MAX_POSITION_SIZE = 2.0  # contracts per trade (reduced from 5.0)
MAX_POSITIONS_PER_SCAN = 5  # Max trades per scan


def _load_weather_snapshots(snapshots_dir: Path, max_age_seconds: float = 3600) -> list[dict]:
    """Load recent weather snapshots from disk.
    
    Args:
        snapshots_dir: Directory containing snapshot files
        max_age_seconds: Maximum age of snapshots to consider
        
    Returns:
        List of snapshot data, sorted by time (newest last)
    """
    snapshots = []
    cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
    
    if not snapshots_dir.exists():
        return snapshots
    
    for file in snapshots_dir.glob("snapshot_weather_*.json"):
        try:
            # Parse timestamp from filename
            ts_str = file.stem.split("_")[2]  # snapshot_weather_20260216T115006Z
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            
            if ts >= cutoff:
                data = json.loads(file.read_text())
                data["_snapshot_ts"] = ts.isoformat()
                snapshots.append(data)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.debug("Skipping snapshot %s: %s", file.name, e)
            continue
    
    # Sort by timestamp
    snapshots.sort(key=lambda x: x.get("_snapshot_ts", ""))
    return snapshots


@dataclass(frozen=True)
class WeatherMarket:
    """A weather-related market on Polymarket."""

    market_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    city: str | None
    forecast_date: date | None
    market_end_date: datetime
    threshold_temp: float | None
    condition: str | None  # "above", "below", or None
    current_yes_price: float | None = None
    current_no_price: float | None = None
    best_yes_ask: float | None = None
    best_no_ask: float | None = None

    @property
    def implied_probability(self) -> float | None:
        """Market-implied probability of YES outcome."""
        return self.current_yes_price


def _extract_city_from_question(question: str) -> str | None:
    """Extract city name from market question text."""
    question_lower = question.lower()

    # Expanded city patterns
    city_patterns = [
        ("seoul", ["seoul"]),
        ("tokyo", ["tokyo"]),
        ("paris", ["paris"]),
        ("berlin", ["berlin"]),
        ("sydney", ["sydney"]),
        ("toronto", ["toronto"]),
        ("vancouver", ["vancouver"]),
        ("nyc", ["nyc", "new york city", "new york, ny"]),
        ("new_york", ["new york"]),
        ("london", ["london, uk", "london, england", "london"]),
        ("chicago", ["chicago"]),
        ("seattle", ["seattle"]),
        ("atlanta", ["atlanta"]),
        ("dallas", ["dallas"]),
        ("miami", ["miami"]),
        ("los_angeles", ["los angeles", "la, ca", "l.a."]),
        ("san_francisco", ["san francisco", "sf, ca", "sf bay"]),
        ("boston", ["boston"]),
        ("washington", ["washington, dc", "washington dc", "dc"]),
        ("philadelphia", ["philadelphia", "philly"]),
        ("denver", ["denver"]),
        ("phoenix", ["phoenix"]),
        ("houston", ["houston"]),
        ("detroit", ["detroit"]),
        ("minneapolis", ["minneapolis"]),
        ("portland", ["portland, or", "portland, oregon"]),
        ("las_vegas", ["las vegas"]),
        ("san_diego", ["san diego"]),
        ("austin", ["austin, tx", "austin, texas"]),
        ("nashville", ["nashville"]),
        ("new_orleans", ["new orleans"]),
    ]

    for city_id, patterns in city_patterns:
        for pattern in patterns:
            if pattern in question_lower:
                return city_id

    return None


def _extract_date_from_question(question: str) -> date | None:
    """Extract forecast date from market question text."""
    question_lower = question.lower()

    if "tomorrow" in question_lower:
        return date.today() + timedelta(days=1)

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?"
    match = re.search(month_pattern, question_lower)

    if match:
        month_name = match.group(1)
        day = int(match.group(2))
        year_str = match.group(3)

        month = months.get(month_name.lower())
        if month:
            year = int(year_str) if year_str else date.today().year
            try:
                return date(year, month, day)
            except ValueError:
                pass

    return None


def _extract_temperature_threshold(question: str) -> tuple[float | None, str | None]:
    """Extract temperature threshold and condition from question."""
    question_lower = question.lower()

    # Handle Celsius patterns too - only patterns with capture groups
    above_patterns = [
        (r"above\s+(\d+)\s*(?:°|degrees?)?", "above"),
        (r"higher\s+than\s+(\d+)\s*(?:°|degrees?)?", "above"),
        (r"greater\s+than\s+(\d+)\s*(?:°|degrees?)?", "above"),
        (r"exceed\s+(\d+)\s*(?:°|degrees?)?", "above"),
        (r"over\s+(\d+)\s*(?:°|degrees?)?", "above"),
        (r"at\s+least\s+(\d+)\s*(?:°|degrees?)?", "above"),
    ]

    below_patterns = [
        (r"below\s+(\d+)\s*(?:°|degrees?)?", "below"),
        (r"lower\s+than\s+(\d+)\s*(?:°|degrees?)?", "below"),
        (r"less\s+than\s+(\d+)\s*(?:°|degrees?)?", "below"),
        (r"under\s+(\d+)\s*(?:°|degrees?)?", "below"),
        (r"at\s+most\s+(\d+)\s*(?:°|degrees?)?", "below"),
    ]

    for pattern, condition in above_patterns:
        match = re.search(pattern, question_lower)
        if match:
            return float(match.group(1)), condition

    for pattern, condition in below_patterns:
        match = re.search(pattern, question_lower)
        if match:
            return float(match.group(1)), condition

    # Fallback: just find any temperature number
    temp_pattern = r"(\d+)\s*(?:°|degrees?\s*[fc])?"
    match = re.search(temp_pattern, question_lower)
    if match:
        return float(match.group(1)), None

    return None, None


def _is_weather_market(question: str) -> bool:
    """Check if a market question is weather-related."""
    question_lower = question.lower()

    weather_keywords = [
        "temperature",
        "high temp",
        "low temp",
        "degrees",
        "°f",
        "°c",
        "fahrenheit",
        "celsius",
        "weather",
        "forecast",
        "rain",
        "snow",
        "precipitation",
    ]

    return any(kw in question_lower for kw in weather_keywords)


def _best_ask(book: dict | None) -> float | None:
    """Get best ask price from book."""
    if not book:
        return None
    asks = book.get("asks") or []
    if not asks:
        return None
    try:
        return min(float(a["price"]) for a in asks)
    except Exception:
        return None


def _best_bid(book: dict | None) -> float | None:
    """Get best bid price from book."""
    if not book:
        return None
    bids = book.get("bids") or []
    if not bids:
        return None
    try:
        return max(float(b["price"]) for b in bids)
    except Exception:
        return None


def find_weather_markets(snapshots_dir: Path | None = None) -> list[WeatherMarket]:
    """Find weather markets using weather collector snapshots.
    
    Uses the weather snapshot data collected by the collector loop.
    """
    markets: list[WeatherMarket] = []
    
    if snapshots_dir is None:
        snapshots_dir = Path("data")
    
    if not snapshots_dir.exists():
        return markets
    
    # Load weather snapshots
    snapshots = _load_weather_snapshots(snapshots_dir, max_age_seconds=3600)
    
    if not snapshots:
        logger.warning("No recent weather snapshots found in %s", snapshots_dir)
        return markets
    
    # Use the most recent snapshot
    latest = snapshots[-1]
    
    for m in latest.get("markets", []):
        question = m.get("question", "")
        
        if not _is_weather_market(question):
            continue
        
        token_ids = m.get("clob_token_ids", [])
        if len(token_ids) != 2:
            continue
        
        end_date = m.get("end_date", "")
        try:
            market_end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            market_end = datetime.now(UTC) + timedelta(days=1)
        
        city = _extract_city_from_question(question)
        forecast_date = _extract_date_from_question(question)
        threshold, condition = _extract_temperature_threshold(question)
        
        # Get prices from books
        books = m.get("books", {}) or {}
        yes_book = books.get("yes") or {}
        no_book = books.get("no") or {}
        
        yes_bid = _best_bid(yes_book)
        yes_ask = _best_ask(yes_book)
        no_ask = _best_ask(no_book)
        
        # Calculate mid price
        current_yes_price = None
        if yes_bid is not None and yes_ask is not None:
            current_yes_price = (yes_bid + yes_ask) / 2
        elif yes_bid is not None:
            current_yes_price = yes_bid
        elif yes_ask is not None:
            current_yes_price = yes_ask
        
        weather_market = WeatherMarket(
            market_id=str(m.get("market_id", m.get("id", ""))),
            token_id_yes=str(token_ids[0]),
            token_id_no=str(token_ids[1]),
            question=question,
            city=city,
            forecast_date=forecast_date,
            market_end_date=market_end,
            threshold_temp=threshold,
            condition=condition,
            current_yes_price=current_yes_price,
            current_no_price=1.0 - current_yes_price if current_yes_price else None,
            best_yes_ask=yes_ask,
            best_no_ask=no_ask,
        )
        
        markets.append(weather_market)
    
    return markets


@dataclass(frozen=True)
class WeatherSignal:
    """Trading signal from weather forecast analysis."""

    timestamp: datetime
    market: WeatherMarket
    consensus: ModelConsensus | None
    side: str  # "buy_yes", "buy_no", "no_trade"
    market_prob: float
    model_prob: float
    edge: float
    confidence: float
    expected_value: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market": {
                "market_id": self.market.market_id,
                "question": self.market.question,
                "city": self.market.city,
                "threshold_temp": self.market.threshold_temp,
                "condition": self.market.condition,
                "current_yes_price": self.market.current_yes_price,
            },
            "consensus": self.consensus.to_dict() if self.consensus else None,
            "side": self.side,
            "market_prob": self.market_prob,
            "model_prob": self.model_prob,
            "edge": self.edge,
            "confidence": self.confidence,
            "expected_value": self.expected_value,
        }


def generate_signals(
    markets: list[WeatherMarket],
    consensus_data: dict[str, ModelConsensus],
) -> list[WeatherSignal]:
    """Generate trading signals by comparing market prices to model consensus."""
    signals: list[WeatherSignal] = []
    now = datetime.now(UTC)

    for market in markets:
        # Skip if no price data
        if market.current_yes_price is None:
            continue

        # Get consensus for this city (if available)
        consensus = consensus_data.get(market.city) if market.city else None

        market_prob = market.current_yes_price
        
        # Default model probability - assume efficient market baseline
        # If no model data, use 0.5 as neutral
        if consensus and market.threshold_temp is not None and market.condition:
            if consensus.agreement_score >= 0.4 and consensus.model_count >= 2:  # Lowered from 0.5
                if market.condition == "above":
                    model_prob = consensus.probability_above_threshold(market.threshold_temp)
                elif market.condition == "below":
                    model_prob = 1.0 - consensus.probability_above_threshold(market.threshold_temp)
                else:
                    model_prob = 0.5
            else:
                # Insufficient model agreement - skip
                continue
        else:
            # No consensus data - use simple heuristic: look for mispriced extremes
            # Buy cheap sides (< 0.15) or sell expensive sides (> 0.85)
            if market_prob < 0.15:
                model_prob = 0.3  # Assume some value
            elif market_prob > 0.85:
                model_prob = 0.7
            else:
                continue  # No edge without model data

        edge = model_prob - market_prob

        # Decision logic
        side = "no_trade"
        confidence = 0.0
        expected_value = 0.0

        # Buy YES when market price is low but model suggests higher probability
        if market_prob < YES_ENTRY_MAX_PRICE and model_prob > YES_ENTRY_MIN_PROBABILITY:
            side = "buy_yes"
            confidence = min(1.0, model_prob * (1 - market_prob / YES_ENTRY_MAX_PRICE))
            win_amount = 1.0 - market_prob
            lose_amount = market_prob
            expected_value = (model_prob * win_amount) - ((1 - model_prob) * lose_amount)

        # Buy NO when market price implies high probability but model suggests lower
        no_price = market.current_no_price or (1.0 - market_prob)
        if side == "no_trade" and no_price is not None and no_price > 0:
            if no_price > NO_ENTRY_MIN_PRICE and model_prob < NO_ENTRY_MAX_PROBABILITY:
                side = "buy_no"
                confidence = min(1.0, (1 - model_prob) * (no_price / NO_ENTRY_MIN_PRICE))
                model_prob_no = 1.0 - model_prob
                win_amount = 1.0 - no_price
                lose_amount = no_price
                expected_value = (model_prob_no * win_amount) - ((1 - model_prob_no) * lose_amount)

        if side != "no_trade" or edge != 0:
            signals.append(
                WeatherSignal(
                    timestamp=now,
                    market=market,
                    consensus=consensus,
                    side=side,
                    market_prob=market_prob,
                    model_prob=model_prob,
                    edge=edge,
                    confidence=confidence,
                    expected_value=expected_value,
                )
            )

    # Sort by expected value descending
    signals.sort(key=lambda s: s.expected_value, reverse=True)

    return signals


@dataclass(frozen=True)
class WeatherTrade:
    """An executed weather arbitrage trade."""

    timestamp: datetime
    signal: WeatherSignal
    position_size: float
    entry_price: float
    token_id: str
    side: str
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal.to_dict(),
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "token_id": self.token_id,
            "side": self.side,
            "dry_run": self.dry_run,
        }


def record_weather_fill(
    signal: WeatherSignal,
    data_dir: Path,
    dry_run: bool = True,
) -> WeatherTrade | None:
    """Record a paper trade for a weather signal.
    
    Uses the PaperTradingEngine to record fills.
    """
    if signal.side == "no_trade":
        return None

    now = datetime.now(UTC)
    engine = PaperTradingEngine(data_dir=data_dir)

    # Determine order parameters
    if signal.side == "buy_yes":
        token_id = signal.market.token_id_yes
        side = "buy"
        # Use best ask if available, otherwise mid price
        price = signal.market.best_yes_ask or signal.market.current_yes_price or 0.10
    elif signal.side == "buy_no":
        token_id = signal.market.token_id_no
        side = "buy"
        price = signal.market.best_no_ask or signal.market.current_no_price or 0.10
    else:
        return None

    # Position sizing: smaller for more opportunities
    base_size = 1.0
    confidence_multiplier = min(3.0, signal.confidence * 3)
    position_size = min(MAX_POSITION_SIZE, base_size * confidence_multiplier)
    position_size = round(position_size, 2)

    if not dry_run:
        # Would submit order here
        pass

    # Record fill
    try:
        engine.record_fill(
            token_id=token_id,
            side=side,
            size=position_size,
            price=price,
            fee=0,
            market_slug=signal.market.market_id,
            market_question=signal.market.question,
        )
        
        return WeatherTrade(
            timestamp=now,
            signal=signal,
            position_size=position_size,
            entry_price=price,
            token_id=token_id,
            side=side,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.exception("Error recording weather fill: %s", e)
        return None


def run_weather_scan(
    data_dir: Path,
    snapshots_dir: Path | None = None,
    cities: list[str] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run a complete weather arbitrage scan with paper fills.

    Args:
        data_dir: Directory to store paper trading data
        snapshots_dir: Directory with market snapshots
        cities: List of cities to monitor (None = all)
        dry_run: If True, don't actually submit orders (paper trading)

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)

    if snapshots_dir is None:
        snapshots_dir = Path("data")

    logger.info("Starting weather scan at %s", now.isoformat())

    # Step 1: Find weather markets from snapshots
    markets = find_weather_markets(snapshots_dir)
    logger.info("Found %d weather markets from snapshots", len(markets))

    # Filter to target cities if specified
    if cities:
        markets = [m for m in markets if m.city in cities]
        logger.info("%d markets match target cities: %s", len(markets), cities)

    # Step 2: Fetch model forecasts for cities we have markets for
    unique_cities = list(set(m.city for m in markets if m.city))
    target_date = date.today() + timedelta(days=1)
    
    if unique_cities:
        consensus_data = get_consensus_for_cities(unique_cities, target_date, min_models=2)
        logger.info("Got consensus for %d/%d cities", len(consensus_data), len(unique_cities))
    else:
        consensus_data = {}
        logger.warning("No cities identified in markets")

    # Step 3: Generate signals
    signals = generate_signals(markets, consensus_data)
    logger.info("Generated %d signals", len(signals))

    # Step 4: Filter to actionable signals (loosened threshold)
    actionable = [s for s in signals if s.side != "no_trade" and s.expected_value > 0.02]
    logger.info("%d actionable signals (EV > 0.02)", len(actionable))

    # Step 5: Record paper fills
    trades: list[WeatherTrade] = []
    for signal in actionable[:MAX_POSITIONS_PER_SCAN]:
        trade = record_weather_fill(signal, data_dir, dry_run=dry_run)
        if trade:
            trades.append(trade)
            logger.info(
                "Recorded %s fill: %s @ %.3f (EV: %.3f)",
                signal.side,
                signal.market.question[:50],
                trade.entry_price,
                signal.expected_value,
            )

    result = {
        "timestamp": now.isoformat(),
        "markets_scanned": len(markets),
        "signals_generated": len(signals),
        "actionable_signals": len(actionable),
        "fills_recorded": len(trades),
        "dry_run": dry_run,
        "cities_with_consensus": len(consensus_data),
        "markets_by_city": {city: len([m for m in markets if m.city == city]) for city in unique_cities},
        "signals": [s.to_dict() for s in signals],
        "trades": [t.to_dict() for t in trades],
    }
    
    logger.info("Weather scan complete: %d fills recorded", len(trades))
    return result
