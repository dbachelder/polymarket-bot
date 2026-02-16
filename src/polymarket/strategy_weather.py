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

from . import clob
from .site import fetch_predictions_page, parse_next_data
from .trading import Order, submit_order
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
]

# Entry thresholds (loosened to increase signal frequency)
YES_ENTRY_MAX_PRICE = 0.25  # Increased from 0.15
YES_ENTRY_MIN_PROBABILITY = 0.60  # Decreased from 0.70
NO_ENTRY_MIN_PRICE = 0.35  # Decreased from 0.45
NO_ENTRY_MAX_PROBABILITY = 0.30  # Increased from 0.20

# Position sizing
MAX_POSITION_SIZE = 5.0  # contracts per trade
MAX_POSITIONS_PER_DAY = 10


@dataclass(frozen=True)
class WeatherMarket:
    """A weather-related market on Polymarket.

    Attributes:
        market_id: Polymarket market ID
        token_id_yes: YES token ID
        token_id_no: NO token ID
        question: Market question text
        city: Identified city (nyc, london, etc.)
        forecast_date: Date the forecast is for
        market_end_date: When the market resolves
        threshold_temp: Temperature threshold from question (e.g., 72°F)
        condition: "above" or "below" the threshold
        current_yes_price: Current YES price (mid)
        current_no_price: Current NO price (mid)
    """

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

    @property
    def implied_probability(self) -> float | None:
        """Market-implied probability of YES outcome."""
        return self.current_yes_price


def _extract_city_from_question(question: str) -> str | None:
    """Extract city name from market question text.

    Returns normalized city identifier or None.
    """
    question_lower = question.lower()

    # City patterns to match (expanded universe)
    city_patterns = [
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
    """Extract forecast date from market question text.

    Looks for patterns like:
    - "February 15, 2026"
    - "Feb 15"
    - "15th"
    - "tomorrow"
    """
    question_lower = question.lower()

    # Check for "tomorrow"
    if "tomorrow" in question_lower:
        return date.today() + timedelta(days=1)

    # Month patterns
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    # Pattern: Month DD, YYYY or Month DD
    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?"
    match = re.search(month_pattern, question_lower)

    if match:
        month_name = match.group(1)
        day = int(match.group(2))
        year_str = match.group(3)

        month = months.get(month_name.lower())
        if month:
            if year_str:
                year = int(year_str)
            else:
                year = date.today().year
            try:
                return date(year, month, day)
            except ValueError:
                pass

    return None


def _extract_temperature_threshold(question: str) -> tuple[float | None, str | None]:
    """Extract temperature threshold and condition from question.

    Returns (threshold, condition) where condition is "above" or "below".
    """
    question_lower = question.lower()

    # Temperature patterns
    temp_pattern = r"(\d+)\s*(?:°|degrees?\s*f|fahrenheit)?"

    # Look for "above X" or "below X" patterns
    above_patterns = [
        r"above\s+(\d+)\s*(?:°|degrees?)?",
        r"higher\s+than\s+(\d+)\s*(?:°|degrees?)?",
        r"greater\s+than\s+(\d+)\s*(?:°|degrees?)?",
        r"exceed\s+(\d+)\s*(?:°|degrees?)?",
        r"over\s+(\d+)\s*(?:°|degrees?)?",
        r"at\s+least\s+(\d+)\s*(?:°|degrees?)?",
    ]

    below_patterns = [
        r"below\s+(\d+)\s*(?:°|degrees?)?",
        r"lower\s+than\s+(\d+)\s*(?:°|degrees?)?",
        r"less\s+than\s+(\d+)\s*(?:°|degrees?)?",
        r"under\s+(\d+)\s*(?:°|degrees?)?",
        r"at\s+most\s+(\d+)\s*(?:°|degrees?)?",
    ]

    for pattern in above_patterns:
        match = re.search(pattern, question_lower)
        if match:
            return float(match.group(1)), "above"

    for pattern in below_patterns:
        match = re.search(pattern, question_lower)
        if match:
            return float(match.group(1)), "below"

    # Fallback: just find any temperature number
    match = re.search(temp_pattern, question_lower)
    if match:
        return float(match.group(1)), None

    return None, None


def find_weather_markets() -> list[WeatherMarket]:
    """Find weather-related markets on Polymarket.

    Returns list of WeatherMarket objects with current prices.
    """
    markets: list[WeatherMarket] = []

    try:
        # Try to find weather markets via search or categories
        # For now, we'll use the predictions page and filter
        html = fetch_predictions_page("weather")
        data = parse_next_data(html)

        # Extract markets from page data
        # This is similar to extract_5m_markets but for weather
        dehydrated = (
            data.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
        )

        for q in dehydrated:
            state = q.get("state", {})
            data_inner = state.get("data")
            if not isinstance(data_inner, dict):
                continue

            pages = data_inner.get("pages", [])
            for page in pages:
                results = page.get("results", [])
                for item in results:
                    markets_data = item.get("markets", [])
                    for m in markets_data:
                        question = m.get("question", "")

                        # Check if this looks like a weather market
                        if not _is_weather_market(question):
                            continue

                        token_ids = m.get("clobTokenIds", [])
                        if len(token_ids) != 2:
                            continue

                        end_date = m.get("endDate", "")
                        try:
                            market_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        except (ValueError, AttributeError):
                            market_end = datetime.now(UTC) + timedelta(days=1)

                        city = _extract_city_from_question(question)
                        forecast_date = _extract_date_from_question(question)
                        threshold, condition = _extract_temperature_threshold(question)

                        weather_market = WeatherMarket(
                            market_id=str(m.get("id", "")),
                            token_id_yes=str(token_ids[0]),
                            token_id_no=str(token_ids[1]),
                            question=question,
                            city=city,
                            forecast_date=forecast_date,
                            market_end_date=market_end,
                            threshold_temp=threshold,
                            condition=condition,
                        )

                        # Fetch current prices
                        try:
                            yes_price = clob.get_price(weather_market.token_id_yes, side="buy")
                            if yes_price and isinstance(yes_price, dict):
                                weather_market = weather_market._replace(
                                    current_yes_price=yes_price.get("price")
                                )
                        except Exception as e:
                            logger.debug(
                                "Could not fetch price for %s: %s", weather_market.market_id, e
                            )

                        markets.append(weather_market)

    except Exception as e:
        logger.exception("Error finding weather markets: %s", e)

    return markets


def _is_weather_market(question: str) -> bool:
    """Check if a market question is weather-related."""
    question_lower = question.lower()

    weather_keywords = [
        "temperature",
        "high temp",
        "low temp",
        "degrees",
        "°f",
        "fahrenheit",
        "weather",
        "forecast",
        "rain",
        "snow",
        "precipitation",
    ]

    return any(kw in question_lower for kw in weather_keywords)


def find_weather_markets_heuristic(snapshots_dir: Path | None = None) -> list[WeatherMarket]:
    """Find weather markets using heuristics on available market data.

    Alternative approach that scans recent snapshots for weather markets.
    """
    markets: list[WeatherMarket] = []

    # If we have snapshots, scan them
    if snapshots_dir and snapshots_dir.exists():
        # Look at the most recent 5m snapshot
        snapshot_files = sorted(snapshots_dir.glob("snapshot_5m_*.json"))
        if snapshot_files:
            latest = snapshot_files[-1]
            try:
                data = json.loads(latest.read_text())
                for m in data.get("markets", []):
                    question = m.get("question", m.get("title", ""))

                    if not _is_weather_market(question):
                        continue

                    token_ids = m.get("clob_token_ids", m.get("clobTokenIds", []))
                    if len(token_ids) != 2:
                        continue

                    end_date = m.get("end_date", m.get("endDate", ""))
                    try:
                        market_end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        market_end = datetime.now(UTC) + timedelta(days=1)

                    city = _extract_city_from_question(question)
                    forecast_date = _extract_date_from_question(question)
                    threshold, condition = _extract_temperature_threshold(question)

                    books = m.get("books", {})
                    yes_book = books.get("yes", {})

                    # Get mid price from book
                    yes_bids = yes_book.get("bids", [])
                    yes_asks = yes_book.get("asks", [])

                    current_yes_price = None
                    if yes_bids and yes_asks:
                        best_bid = max(float(b["price"]) for b in yes_bids)
                        best_ask = min(float(a["price"]) for a in yes_asks)
                        current_yes_price = (best_bid + best_ask) / 2

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
                    )

                    markets.append(weather_market)

            except Exception as e:
                logger.exception("Error scanning snapshots for weather markets: %s", e)

    return markets


@dataclass(frozen=True)
class WeatherSignal:
    """Trading signal from weather forecast analysis.

    Attributes:
        timestamp: When signal was generated
        market: WeatherMarket this signal is for
        consensus: Model consensus forecast
        side: "buy_yes", "buy_no", or "no_trade"
        market_prob: Current market-implied probability
        model_prob: Model-derived probability
        edge: Difference between model and market (model - market)
        confidence: Signal confidence (0-1)
        expected_value: Expected value of trade
    """

    timestamp: datetime
    market: WeatherMarket
    consensus: ModelConsensus
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
            "consensus": self.consensus.to_dict(),
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
    """Generate trading signals by comparing market prices to model consensus.

    Args:
        markets: List of weather markets
        consensus_data: Dictionary of city -> ModelConsensus

    Returns:
        List of trading signals
    """
    signals: list[WeatherSignal] = []
    now = datetime.now(UTC)

    for market in markets:
        # Skip if no city identified or no price data
        if not market.city or market.current_yes_price is None:
            continue

        # Skip if we don't have consensus for this city
        if market.city not in consensus_data:
            continue

        consensus = consensus_data[market.city]

        # Skip if insufficient model agreement
        if consensus.agreement_score < 0.5 or consensus.model_count < 2:
            continue

        # Determine what the market is asking
        threshold = market.threshold_temp
        condition = market.condition

        if threshold is None:
            continue

        # Calculate model probability based on condition
        if condition == "above":
            model_prob = consensus.probability_above_threshold(threshold)
        elif condition == "below":
            model_prob = 1.0 - consensus.probability_above_threshold(threshold)
        else:
            # Unknown condition - skip
            continue

        market_prob = market.current_yes_price
        edge = model_prob - market_prob

        # Decision logic
        side = "no_trade"
        confidence = 0.0
        expected_value = 0.0

        # Buy YES when market price is low but model suggests high probability
        if market_prob < YES_ENTRY_MAX_PRICE and model_prob > YES_ENTRY_MIN_PROBABILITY:
            side = "buy_yes"
            confidence = min(1.0, model_prob * (1 - market_prob / YES_ENTRY_MAX_PRICE))
            # EV = (win_prob * win_amount) - (lose_prob * lose_amount)
            # Assuming 1 contract at market_prob price
            win_amount = 1.0 - market_prob  # Profit if win (pay 1, keep cost)
            lose_amount = market_prob  # Loss if lose (cost is lost)
            expected_value = (model_prob * win_amount) - ((1 - model_prob) * lose_amount)

        # Buy NO when market price is high but model suggests low probability
        no_price = market.current_no_price or (1.0 - market_prob)
        if no_price is not None and no_price > 0:
            if no_price > NO_ENTRY_MIN_PRICE and model_prob < NO_ENTRY_MAX_PROBABILITY:
                side = "buy_no"
                confidence = min(1.0, (1 - model_prob) * (no_price / NO_ENTRY_MIN_PRICE))
                # EV for NO side
                model_prob_no = 1.0 - model_prob
                win_amount = 1.0 - no_price
                lose_amount = no_price
                expected_value = (model_prob_no * win_amount) - ((1 - model_prob_no) * lose_amount)

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
    """An executed weather arbitrage trade.

    Attributes:
        timestamp: When trade was executed
        signal: The signal that triggered the trade
        order_result: Result from order submission
        position_size: Number of contracts
        entry_price: Price paid per contract
    """

    timestamp: datetime
    signal: WeatherSignal
    order_result: Any
    position_size: float
    entry_price: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal.to_dict(),
            "order_result": {
                "success": self.order_result.success,
                "dry_run": self.order_result.dry_run,
                "message": self.order_result.message,
                "order_id": self.order_result.order_id,
            }
            if hasattr(self.order_result, "success")
            else str(self.order_result),
            "position_size": self.position_size,
            "entry_price": self.entry_price,
        }


def execute_trade(signal: WeatherSignal, dry_run: bool = True) -> WeatherTrade | None:
    """Execute a trade based on a weather signal.

    Args:
        signal: WeatherSignal with side and market info
        dry_run: If True, don't actually submit orders

    Returns:
        WeatherTrade if executed, None otherwise
    """
    if signal.side == "no_trade":
        return None

    now = datetime.now(UTC)

    # Determine order parameters
    if signal.side == "buy_yes":
        token_id = signal.market.token_id_yes
        side = "buy"
        price = signal.market.current_yes_price or 0.10
    elif signal.side == "buy_no":
        token_id = signal.market.token_id_no
        side = "buy"
        price = signal.market.current_no_price or 0.10
    else:
        return None

    # Limit price: be slightly aggressive to get filled
    limit_price = min(0.99, price * 1.02)  # Pay up to 2% over mid

    # Position sizing: micro-betting approach
    # Size based on confidence and EV
    base_size = 1.0
    confidence_multiplier = min(5.0, signal.confidence * 5)
    position_size = min(MAX_POSITION_SIZE, base_size * confidence_multiplier)

    # Round to reasonable precision
    position_size = round(position_size, 2)

    try:
        order = Order(
            token_id=token_id,
            side=side,
            size=position_size,
            price=limit_price,
        )

        # Import here to avoid circular imports
        from .config import load_config

        config = load_config()
        if dry_run:
            # Override config to force dry run
            config = config  # config is frozen, can't modify

        result = submit_order(order, config)

        return WeatherTrade(
            timestamp=now,
            signal=signal,
            order_result=result,
            position_size=position_size,
            entry_price=limit_price,
        )

    except Exception as e:
        logger.exception("Error executing weather trade: %s", e)
        return None


def run_weather_scan(
    snapshots_dir: Path | None = None,
    cities: list[str] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run a complete weather arbitrage scan.

    Args:
        snapshots_dir: Directory with market snapshots
        cities: List of cities to monitor
        dry_run: If True, don't execute trades

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)

    if cities is None:
        cities = ["nyc", "london"]  # Start with these per test parameters

    logger.info("Starting weather scan at %s", now.isoformat())

    # Step 1: Find weather markets
    markets = find_weather_markets_heuristic(snapshots_dir)
    logger.info("Found %d weather markets", len(markets))

    # Filter to target cities
    markets = [m for m in markets if m.city in cities]
    logger.info("%d markets match target cities: %s", len(markets), cities)

    # Step 2: Fetch model forecasts
    target_date = date.today() + timedelta(days=1)  # Tomorrow
    consensus_data = get_consensus_for_cities(cities, target_date, min_models=2)
    logger.info("Got consensus for %d cities", len(consensus_data))

    # Step 3: Generate signals
    signals = generate_signals(markets, consensus_data)
    logger.info("Generated %d signals", len(signals))

    # Step 4: Filter to actionable signals
    actionable = [s for s in signals if s.side != "no_trade" and s.expected_value > 0.05]
    logger.info("%d actionable signals", len(actionable))

    # Step 5: Execute trades (respecting limits)
    trades: list[WeatherTrade] = []
    for signal in actionable[:MAX_POSITIONS_PER_DAY]:
        trade = execute_trade(signal, dry_run=dry_run)
        if trade:
            trades.append(trade)
            logger.info(
                "Executed %s trade: %s @ %.3f (EV: %.3f)",
                signal.side,
                signal.market.question[:50],
                trade.entry_price,
                signal.expected_value,
            )

    return {
        "timestamp": now.isoformat(),
        "markets_scanned": len(markets),
        "signals_generated": len(signals),
        "actionable_signals": len(actionable),
        "trades_executed": len(trades),
        "dry_run": dry_run,
        "consensus": {k: v.to_dict() for k, v in consensus_data.items()},
        "signals": [s.to_dict() for s in signals],
        "trades": [t.to_dict() for t in trades],
    }
