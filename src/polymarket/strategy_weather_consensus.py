"""Weather Data Gap Mispricing Strategy (Model Consensus).

Hypothesis: Weather markets on Polymarket systematically misprice outcomes when
multiple forecast models agree but market prices haven't adjusted. By comparing
GFS/ECMWF consensus to market-implied probabilities, we capture +EV entries in
the 0.01-0.15 price range where casual bettors underprice high-probability events.

Key differences from strategy_weather (forecast latency):
- Focus on model consensus (2-3 models agreeing) vs single-model latency
- Extreme mispricing focus (0.01-0.05 range) for asymmetric upside
- Higher confidence threshold (>75% vs >70%)
- Dynamic position sizing based on edge magnitude
- Cut loss if consensus shifts >50%

Reference Wallets:
- 0xf2e346ab ($24k PnL, 74% WR): Temperature specialist
- 1pixel ($20k PnL, 55% WR): Multi-city approach
- automatedAltradingbot ($65k PnL, 37% WR): High volume, small size
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .strategy_weather import (
    MAX_POSITIONS_PER_DAY,
    WeatherMarket,
    find_weather_markets_heuristic,
)
from .trading import Order, submit_order
from .weather import ModelConsensus, get_consensus_for_cities

logger = logging.getLogger(__name__)

# Entry thresholds - more extreme than base weather strategy
YES_ENTRY_MAX_PRICE = 0.15  # Max price to buy YES
YES_ENTRY_EXTREME_MAX = 0.05  # Extreme mispricing range
YES_ENTRY_MIN_PROBABILITY = 0.70  # Model consensus threshold
YES_ENTRY_HIGH_CONFIDENCE = 0.75  # High confidence threshold for extreme sizing

NO_ENTRY_MIN_PRICE = 0.85  # Min price to buy NO (market implies high YES prob)
NO_ENTRY_MAX_PROBABILITY = 0.30  # Model shows low probability
NO_ENTRY_LOW_CONFIDENCE = 0.25  # Very low confidence for extreme sizing

# Model consensus requirements
MIN_MODELS_FOR_CONSENSUS = 2  # Require at least 2 models
MIN_MODEL_AGREEMENT = 0.75  # 75% agreement score required

# Position sizing - micro-betting approach
MIN_POSITION_SIZE = 5.0  # $5 minimum
MAX_POSITION_SIZE = 20.0  # $20 maximum
EXTREME_POSITION_SIZE = 15.0  # Size for extreme mispricings (0.01-0.05)
MAX_POSITIONS_PER_CITY = 5  # Max concurrent positions per city
MAX_DAILY_EXPOSURE = 500.0  # Total daily exposure cap

# Exit thresholds
TAKE_PROFIT_THRESHOLD = 0.85  # Take profit when price reaches this
CONSENSUS_SHIFT_CUTOFF = 0.50  # Cut loss if consensus shifts by 50%+

# Target cities (high liquidity)
TARGET_CITIES = ["nyc", "chicago", "dallas", "miami", "london"]


@dataclass(frozen=True)
class ConsensusSignal:
    """Trading signal from weather model consensus analysis.

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
        is_extreme_mispricing: Whether this is in the 0.01-0.05 range
        model_agreement: How well models agree (0-1)
        models_used: Number of models in consensus
    """

    timestamp: datetime
    market: WeatherMarket
    consensus: ModelConsensus
    side: str
    market_prob: float
    model_prob: float
    edge: float
    confidence: float
    expected_value: float
    is_extreme_mispricing: bool
    model_agreement: float
    models_used: int

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
            "is_extreme_mispricing": self.is_extreme_mispricing,
            "model_agreement": self.model_agreement,
            "models_used": self.models_used,
        }


@dataclass(frozen=True)
class ConsensusPosition:
    """An open position tracked by the consensus strategy.

    Attributes:
        entry_signal: The signal that triggered the position
        entry_price: Price paid per contract
        position_size: Dollar amount invested
        contracts: Number of contracts owned
        entry_time: When position was opened
        token_id: Token ID for position tracking
        side: "yes" or "no"
        take_profit_price: Price level for take profit
        stop_loss_trigger: Consensus shift threshold for stop
    """

    entry_signal: ConsensusSignal
    entry_price: float
    position_size: float
    contracts: float
    entry_time: datetime
    token_id: str
    side: str
    take_profit_price: float
    stop_loss_trigger: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entry_signal": self.entry_signal.to_dict(),
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "contracts": self.contracts,
            "entry_time": self.entry_time.isoformat(),
            "token_id": self.token_id,
            "side": self.side,
            "take_profit_price": self.take_profit_price,
            "stop_loss_trigger": self.stop_loss_trigger,
        }


def calculate_position_size(
    signal: ConsensusSignal,
    daily_exposure: float,
    city_positions: int,
) -> float:
    """Calculate position size based on signal strength and limits.

    Args:
        signal: The trading signal
        daily_exposure: Current daily exposure
        city_positions: Number of positions already in this city

    Returns:
        Position size in dollars (0 if limits exceeded)
    """
    # Check limits
    if city_positions >= MAX_POSITIONS_PER_CITY:
        return 0.0
    if daily_exposure >= MAX_DAILY_EXPOSURE:
        return 0.0

    remaining_exposure = MAX_DAILY_EXPOSURE - daily_exposure

    # Base size on signal characteristics
    if signal.is_extreme_mispricing and signal.model_agreement >= MIN_MODEL_AGREEMENT:
        # Extreme mispricing with high agreement - max size
        base_size = EXTREME_POSITION_SIZE
    elif signal.expected_value > 0.20:
        # High EV - larger size
        base_size = 12.0
    elif signal.expected_value > 0.10:
        # Medium EV
        base_size = 8.0
    else:
        # Standard EV
        base_size = MIN_POSITION_SIZE

    # Adjust for confidence
    confidence_multiplier = 0.5 + (signal.confidence * 0.5)  # 0.5 to 1.0
    size = base_size * confidence_multiplier

    # Cap at remaining exposure and max position size
    size = min(size, remaining_exposure, MAX_POSITION_SIZE)
    size = max(size, MIN_POSITION_SIZE)  # Don't trade below minimum

    return round(size, 2)


def generate_consensus_signals(
    markets: list[WeatherMarket],
    consensus_data: dict[str, ModelConsensus],
) -> list[ConsensusSignal]:
    """Generate trading signals using model consensus approach.

    Args:
        markets: List of weather markets
        consensus_data: Dictionary of city -> ModelConsensus

    Returns:
        List of consensus-based trading signals
    """
    signals: list[ConsensusSignal] = []
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
        if consensus.agreement_score < MIN_MODEL_AGREEMENT:
            continue
        if consensus.model_count < MIN_MODELS_FOR_CONSENSUS:
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
            continue

        market_prob = market.current_yes_price
        edge = model_prob - market_prob

        # Decision logic
        side = "no_trade"
        confidence = 0.0
        expected_value = 0.0
        is_extreme = False

        # Buy YES when market price is low but model suggests high probability
        if market_prob < YES_ENTRY_MAX_PRICE and model_prob > YES_ENTRY_MIN_PROBABILITY:
            side = "buy_yes"
            confidence = min(1.0, model_prob * (1 - market_prob / YES_ENTRY_MAX_PRICE))

            # Check for extreme mispricing (0.01-0.05 range)
            if market_prob < YES_ENTRY_EXTREME_MAX and model_prob > YES_ENTRY_HIGH_CONFIDENCE:
                is_extreme = True
                confidence = min(1.0, confidence * 1.2)  # Boost confidence

            # EV calculation
            win_amount = 1.0 - market_prob
            lose_amount = market_prob
            expected_value = (model_prob * win_amount) - ((1 - model_prob) * lose_amount)

        # Buy NO when market price is high but model suggests low probability
        no_price = 1.0 - market_prob
        if no_price > (1.0 - NO_ENTRY_MIN_PRICE) and model_prob < NO_ENTRY_MAX_PROBABILITY:
            side = "buy_no"
            model_prob_no = 1.0 - model_prob
            confidence = min(1.0, model_prob_no * (no_price / (1.0 - NO_ENTRY_MIN_PRICE)))

            # Check for extreme mispricing on NO side
            if model_prob < NO_ENTRY_LOW_CONFIDENCE and no_price > 0.70:
                is_extreme = True
                confidence = min(1.0, confidence * 1.2)

            # EV calculation for NO
            win_amount = 1.0 - no_price
            lose_amount = no_price
            expected_value = (model_prob_no * win_amount) - ((1 - model_prob_no) * lose_amount)

        signals.append(
            ConsensusSignal(
                timestamp=now,
                market=market,
                consensus=consensus,
                side=side,
                market_prob=market_prob,
                model_prob=model_prob,
                edge=edge,
                confidence=confidence,
                expected_value=expected_value,
                is_extreme_mispricing=is_extreme,
                model_agreement=consensus.agreement_score,
                models_used=consensus.model_count,
            )
        )

    # Sort by expected value descending
    signals.sort(key=lambda s: s.expected_value, reverse=True)

    return signals


def execute_consensus_trade(
    signal: ConsensusSignal,
    daily_exposure: float,
    city_positions: int,
    dry_run: bool = True,
) -> ConsensusPosition | None:
    """Execute a trade based on a consensus signal.

    Args:
        signal: ConsensusSignal with side and market info
        daily_exposure: Current daily exposure
        city_positions: Number of positions in this city
        dry_run: If True, don't actually submit orders

    Returns:
        ConsensusPosition if executed, None otherwise
    """
    if signal.side == "no_trade":
        return None

    # Calculate position size
    position_size = calculate_position_size(signal, daily_exposure, city_positions)
    if position_size <= 0:
        logger.debug("Position size 0 for %s - limits reached", signal.market.market_id)
        return None

    now = datetime.now(UTC)

    # Determine order parameters
    if signal.side == "buy_yes":
        token_id = signal.market.token_id_yes
        side = "yes"
        entry_price = signal.market.current_yes_price or 0.10
        take_profit = TAKE_PROFIT_THRESHOLD
        # Stop if consensus drops below entry threshold
        stop_trigger = YES_ENTRY_MIN_PROBABILITY - CONSENSUS_SHIFT_CUTOFF
    elif signal.side == "buy_no":
        token_id = signal.market.token_id_no
        side = "no"
        entry_price = signal.market.current_no_price or (1.0 - signal.market.current_yes_price)
        take_profit = TAKE_PROFIT_THRESHOLD
        # Stop if consensus rises above entry threshold
        stop_trigger = NO_ENTRY_MAX_PROBABILITY + CONSENSUS_SHIFT_CUTOFF
    else:
        return None

    # Calculate contracts
    contracts = position_size / entry_price if entry_price > 0 else 0

    # Limit price: be slightly aggressive to get filled
    limit_price = min(0.99, entry_price * 1.02)

    try:
        order = Order(
            token_id=token_id,
            side="buy",
            size=contracts,
            price=limit_price,
        )

        if dry_run:
            logger.info(
                "[DRY RUN] Would execute %s trade: %s @ %.3f "
                "(size: $%.2f, contracts: %.2f, EV: %.3f)",
                signal.side,
                signal.market.question[:50],
                entry_price,
                position_size,
                contracts,
                signal.expected_value,
            )
            # Still create position record for tracking
            result = type(
                "OrderResult",
                (),
                {"success": True, "dry_run": True, "message": "dry_run", "order_id": None},
            )()
        else:
            from .config import load_config

            config = load_config()
            result = submit_order(order, config)

            if result.success:
                logger.info(
                    "Executed %s trade: %s @ %.3f "
                    "(size: $%.2f, contracts: %.2f, EV: %.3f)",
                    signal.side,
                    signal.market.question[:50],
                    entry_price,
                    position_size,
                    contracts,
                    signal.expected_value,
                )
            else:
                logger.warning("Order failed: %s", result.message)
                return None

        return ConsensusPosition(
            entry_signal=signal,
            entry_price=entry_price,
            position_size=position_size,
            contracts=contracts,
            entry_time=now,
            token_id=token_id,
            side=side,
            take_profit_price=take_profit,
            stop_loss_trigger=stop_trigger,
        )

    except Exception as e:
        logger.exception("Error executing consensus trade: %s", e)
        return None


def check_exit_conditions(
    position: ConsensusPosition,
    current_consensus: ModelConsensus | None,
    current_market_price: float | None,
) -> tuple[bool, str]:
    """Check if position should be exited.

    Args:
        position: The open position
        current_consensus: Current model consensus (None if unavailable)
        current_market_price: Current market price for the position side

    Returns:
        Tuple of (should_exit, reason)
    """
    if current_market_price is None:
        return False, "no_price"

    # Take profit condition
    if position.side == "yes" and current_market_price >= TAKE_PROFIT_THRESHOLD:
        return True, f"take_profit_{current_market_price:.2f}"

    if position.side == "no":
        # For NO positions, check if NO price is high (meaning YES price is low)
        no_price = 1.0 - current_market_price if current_market_price else None
        if no_price and no_price >= TAKE_PROFIT_THRESHOLD:
            return True, f"take_profit_{no_price:.2f}"

    # Consensus shift stop loss
    if current_consensus:
        signal = position.entry_signal
        threshold = signal.market.threshold_temp
        condition = signal.market.condition

        if threshold and condition:
            if condition == "above":
                current_prob = current_consensus.probability_above_threshold(threshold)
            else:
                current_prob = 1.0 - current_consensus.probability_above_threshold(threshold)

            original_prob = signal.model_prob
            shift = abs(current_prob - original_prob)

            if shift > CONSENSUS_SHIFT_CUTOFF:
                return True, f"consensus_shift_{shift:.2f}"

    return False, "hold"


def run_consensus_scan(
    snapshots_dir: Path | None = None,
    cities: list[str] | None = None,
    dry_run: bool = True,
    existing_positions: list[ConsensusPosition] | None = None,
) -> dict[str, Any]:
    """Run a complete weather consensus mispricing scan.

    Args:
        snapshots_dir: Directory with market snapshots
        cities: List of cities to monitor
        dry_run: If True, don't execute trades
        existing_positions: List of existing positions to check for exits

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)

    if cities is None:
        cities = TARGET_CITIES

    logger.info("Starting weather consensus scan at %s", now.isoformat())

    # Track positions and exposure
    positions = existing_positions or []
    daily_exposure = sum(p.position_size for p in positions)
    city_position_counts: dict[str, int] = {}
    for p in positions:
        city = p.entry_signal.market.city
        if city:
            city_position_counts[city] = city_position_counts.get(city, 0) + 1

    # Step 1: Find weather markets
    markets = find_weather_markets_heuristic(snapshots_dir)
    logger.info("Found %d weather markets", len(markets))

    # Filter to target cities
    markets = [m for m in markets if m.city in cities]
    logger.info("%d markets match target cities: %s", len(markets), cities)

    # Step 2: Fetch model forecasts
    target_date = date.today() + timedelta(days=1)  # Tomorrow
    consensus_data = get_consensus_for_cities(cities, target_date, min_models=MIN_MODELS_FOR_CONSENSUS)
    logger.info("Got consensus for %d cities", len(consensus_data))

    # Step 3: Check existing positions for exits
    exits: list[dict[str, Any]] = []
    remaining_positions: list[ConsensusPosition] = []

    for position in positions:
        market = position.entry_signal.market
        current_consensus = consensus_data.get(market.city) if market.city else None
        current_price = market.current_yes_price

        should_exit, reason = check_exit_conditions(position, current_consensus, current_price)

        if should_exit:
            exits.append({
                "position": position.to_dict(),
                "exit_reason": reason,
                "exit_time": now.isoformat(),
            })
            logger.info(
                "Exit signal for %s: %s",
                market.question[:50],
                reason,
            )
        else:
            remaining_positions.append(position)

    # Step 4: Generate new signals
    signals = generate_consensus_signals(markets, consensus_data)
    logger.info("Generated %d consensus signals", len(signals))

    # Step 5: Filter to actionable signals
    actionable = [s for s in signals if s.side != "no_trade" and s.expected_value > 0.05]
    logger.info("%d actionable signals", len(actionable))

    # Step 6: Execute trades (respecting limits)
    new_positions: list[ConsensusPosition] = []
    trades_executed = 0

    for signal in actionable[:MAX_POSITIONS_PER_DAY]:
        city = signal.market.city
        city_count = city_position_counts.get(city, 0) if city else 0

        position = execute_consensus_trade(
            signal,
            daily_exposure,
            city_count,
            dry_run=dry_run,
        )

        if position:
            new_positions.append(position)
            daily_exposure += position.position_size
            if city:
                city_position_counts[city] = city_count + 1
            trades_executed += 1

    # Combine positions
    all_positions = remaining_positions + new_positions

    return {
        "timestamp": now.isoformat(),
        "markets_scanned": len(markets),
        "signals_generated": len(signals),
        "actionable_signals": len(actionable),
        "trades_executed": trades_executed,
        "exits_triggered": len(exits),
        "positions_open": len(all_positions),
        "daily_exposure": daily_exposure,
        "dry_run": dry_run,
        "consensus": {k: v.to_dict() for k, v in consensus_data.items()},
        "signals": [s.to_dict() for s in signals],
        "exits": exits,
        "positions": [p.to_dict() for p in all_positions],
    }


def run_consensus_loop(
    snapshots_dir: Path | None = None,
    cities: list[str] | None = None,
    interval_seconds: int = 300,
    dry_run: bool = True,
) -> None:
    """Run continuous consensus scanning loop.

    Args:
        snapshots_dir: Directory with market snapshots
        cities: List of cities to monitor
        interval_seconds: Seconds between scans
        dry_run: If True, don't execute trades
    """
    import time

    positions: list[ConsensusPosition] = []

    logger.info(
        "Starting weather consensus loop (interval=%ds, dry_run=%s)",
        interval_seconds,
        dry_run,
    )

    while True:
        try:
            result = run_consensus_scan(
                snapshots_dir=snapshots_dir,
                cities=cities,
                dry_run=dry_run,
                existing_positions=positions,
            )

            # Update positions for next iteration
            positions = [ConsensusPosition(**p) for p in result["positions"]]

            logger.info(
                "Scan complete: %d signals, %d trades, %d positions open, $%.2f exposure",
                result["signals_generated"],
                result["trades_executed"],
                result["positions_open"],
                result["daily_exposure"],
            )

        except Exception as e:
            logger.exception("Error in consensus scan: %s", e)

        time.sleep(interval_seconds)
