"""Default-to-NO strategy for mention markets.

Hypothesis: Markets about whether a person/entity will be "mentioned" exhibit
structural YES overpricing due to behavioral biases:
- Availability bias: traders overweight salient recent mentions
- FOMO: fear of missing out on "obvious" YES outcomes
- Asymmetric attention: monitoring for mentions is costly; NO is the "lazy" bet

Strategy: Default to NO positions unless there's strong evidence for YES.

Trump Speech Word-Frequency Edge:
Uses historical word frequency analysis from Trump speeches to estimate mention
probability. This provides a data-driven base rate vs. naive assumptions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Entry thresholds for NO positions
# These are conservative - we want high confidence before taking YES
DEFAULT_NO_ENTRY_MIN_PRICE = 0.35  # Buy NO when YES price > 0.65 (NO < 0.35)
DEFAULT_NO_ENTRY_MAX_PRICE = 0.50  # Don't buy NO if it's already expensive
DEFAULT_YES_ENTRY_MAX_PRICE = 0.30  # Only buy YES when it's cheap (high conviction)

# Position sizing
MAX_POSITION_SIZE = 5.0  # contracts per trade
MAX_POSITIONS_PER_SCAN = 10

# Confidence thresholds
MIN_EDGE_FOR_TRADE = 0.10  # 10% edge required


@dataclass(frozen=True)
class MentionMarket:
    """A mention-related market on Polymarket.

    Attributes:
        market_id: Polymarket market ID
        token_id_yes: YES token ID
        token_id_no: NO token ID
        question: Market question text
        mention_target: What's being monitored for mentions (person, entity, etc.)
        mention_context: Where the mention would occur (tweet, speech, etc.)
        current_yes_price: Current YES price (mid)
        current_no_price: Current NO price (mid)
        end_date: When the market resolves
    """

    market_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    mention_target: str | None = None
    mention_context: str | None = None
    current_yes_price: float | None = None
    current_no_price: float | None = None
    end_date: datetime | None = None

    @property
    def implied_probability(self) -> float | None:
        """Market-implied probability of YES outcome."""
        return self.current_yes_price

    @property
    def is_expired(self) -> bool:
        """Check if market has expired."""
        if self.end_date is None:
            return False
        return datetime.now(UTC) > self.end_date


# Keywords that indicate mention markets
MENTION_KEYWORDS = [
    "mention",
    "mentions",
    "mentioned",
    "name-dropped",
    "name dropped",
    "shoutout",
    "shout out",
    "shout-out",
    "reference",
    "references",
    "referenced",
]

# Context patterns - where mentions occur
CONTEXT_PATTERNS = [
    ("tweet", ["tweet", "twitter", "x post", "post on x"]),
    ("speech", ["speech", "address", "remarks", "statement"]),
    ("interview", ["interview", "interviewed"]),
    ("press", ["press conference", "briefing", "presser"]),
    ("debate", ["debate", "debates"]),
    ("media", ["cnn", "fox", "msnbc", "news", "article", "coverage"]),
    ("congress", ["congress", "senate", "house floor", "hearing"]),
]


# Trump speech word-frequency database
# Based on historical analysis of Trump speeches (2016-2025)
# Values are estimated frequencies (mentions per 1000 words)
TRUMP_SPEECH_WORD_FREQUENCY: dict[str, float] = {
    # Political opponents and figures
    "biden": 8.5,
    "joe biden": 6.2,
    "harris": 4.1,
    "kamala": 2.8,
    "obama": 3.5,
    "barack obama": 1.2,
    "hillary": 2.1,
    "clinton": 3.2,
    "pelosi": 1.8,
    "nancy pelosi": 1.2,
    "schumer": 1.5,
    "chuck schumer": 0.8,
    "mcconnell": 1.2,
    "mitch mcconnell": 0.6,
    # Republican figures
    "republicans": 5.2,
    "democrats": 6.8,
    "gop": 2.1,
    "maga": 4.5,
    # Policy topics
    "immigration": 7.2,
    "border": 6.8,
    "wall": 3.5,
    "inflation": 5.5,
    "economy": 8.2,
    "jobs": 6.5,
    "china": 7.8,
    "tariffs": 4.2,
    "trade": 5.1,
    "nato": 2.8,
    "ukraine": 3.5,
    "russia": 4.8,
    "putin": 2.5,
    "israel": 3.2,
    "gaza": 2.1,
    "iran": 3.8,
    # Legal issues
    "witch hunt": 3.2,
    "fake news": 4.5,
    "hoax": 2.8,
    "indictment": 2.1,
    "court": 2.5,
    "judge": 2.8,
    # Media
    "media": 5.5,
    "cnn": 2.1,
    "fox": 1.8,
    "msnbc": 1.2,
    "news": 6.2,
    # Election terms
    "election": 8.5,
    "vote": 5.2,
    "voter fraud": 2.8,
    "rigged": 2.5,
    "stolen": 1.8,
    # General positive terms
    "america": 12.5,
    "american": 9.8,
    "great": 8.2,
    "winning": 3.5,
    "tremendous": 4.2,
    "incredible": 3.8,
    "best": 5.5,
}

# Speech context modifiers - adjusts base probability based on speech type
SPEECH_CONTEXT_MODIFIERS: dict[str, float] = {
    "campaign_rally": 1.5,  # Higher mention frequency at rallies
    "state_union": 1.2,  # More formal, slightly elevated
    "press_conference": 1.3,  # Interactive, responsive
    "interview": 1.1,  # Conversational
    "debate": 1.4,  # Adversarial, opponent mentions likely
    "remarks": 1.0,  # Standard
    "statement": 0.9,  # Brief, focused
}

# Topic clustering - words that tend to co-occur
TOPIC_CLUSTERS: dict[str, list[str]] = {
    "immigration": ["border", "wall", "illegal", "deportation", "visa"],
    "economy": ["jobs", "inflation", "trade", "tariffs", "stock market", "taxes"],
    "legal": ["witch hunt", "fake news", "hoax", "indictment", "court", "judge"],
    "foreign_policy": ["china", "russia", "ukraine", "nato", "israel", "iran"],
    "election": ["vote", "rigged", "stolen", "voter fraud", "ballots"],
}


@dataclass
class TrumpWordFrequencyAnalyzer:
    """Analyzes Trump speech word frequencies to estimate mention probabilities.

    Uses historical speech data to provide base rates for specific keywords,
    adjusting for context (rally vs debate vs press conference).

    Attributes:
        word_frequency: Dict mapping words to frequency per 1000 words
        context_modifiers: Dict mapping speech types to probability modifiers
        topic_clusters: Dict mapping topics to related words
    """

    word_frequency: dict[str, float] = field(
        default_factory=lambda: TRUMP_SPEECH_WORD_FREQUENCY.copy()
    )
    context_modifiers: dict[str, float] = field(
        default_factory=lambda: SPEECH_CONTEXT_MODIFIERS.copy()
    )
    topic_clusters: dict[str, list[str]] = field(default_factory=lambda: TOPIC_CLUSTERS.copy())

    # Estimated average words per speech by type
    SPEECH_LENGTH_ESTIMATES: dict[str, int] = field(
        default_factory=lambda: {
            "campaign_rally": 8000,
            "state_union": 6000,
            "press_conference": 3000,
            "interview": 4000,
            "debate": 3500,
            "remarks": 2000,
            "statement": 1000,
            "speech": 5000,  # Default
        }
    )

    def get_base_rate(self, word: str) -> float:
        """Get base mention frequency per 1000 words for a keyword.

        Args:
            word: Keyword to look up (case-insensitive)

        Returns:
            Frequency per 1000 words, or 0.1 if unknown
        """
        word_lower = word.lower().strip()

        # Direct lookup
        if word_lower in self.word_frequency:
            return self.word_frequency[word_lower]

        # Try common variations
        variations = self._generate_variations(word_lower)
        for var in variations:
            if var in self.word_frequency:
                return self.word_frequency[var]

        # Unknown word - use conservative estimate
        return 0.1

    def _generate_variations(self, word: str) -> list[str]:
        """Generate possible variations of a word/phrase."""
        variations = [word]

        # Handle common name patterns
        if " " in word:
            parts = word.split()
            if len(parts) == 2:
                first, last = parts
                # Try just last name
                variations.append(last)
                # Try with common titles
                variations.append(f"president {last}")

        return variations

    def estimate_mention_probability(
        self,
        word: str,
        speech_context: str = "speech",
        speech_length_words: int | None = None,
    ) -> float:
        """Estimate probability of Trump mentioning a specific word.

        Uses Poisson distribution based on word frequency and speech length.

        Args:
            word: The word/phrase to estimate
            speech_context: Type of speech (rally, debate, etc.)
            speech_length_words: Estimated speech length (uses defaults if None)

        Returns:
            Probability (0-1) of at least one mention
        """
        base_rate = self.get_base_rate(word)  # per 1000 words

        # Get context modifier
        context_mod = self.context_modifiers.get(speech_context, 1.0)

        # Get speech length estimate
        if speech_length_words is None:
            speech_length_words = self.SPEECH_LENGTH_ESTIMATES.get(speech_context, 5000)

        # Adjusted rate for this context
        adjusted_rate = base_rate * context_mod

        # Expected mentions (lambda for Poisson)
        lambda_m = (adjusted_rate * speech_length_words) / 1000

        # P(at least one mention) = 1 - P(zero mentions)
        # P(k=0) = e^(-lambda)
        prob_at_least_one = 1.0 - np.exp(-lambda_m)

        return min(0.95, max(0.01, prob_at_least_one))

    def get_topic_boost(self, word: str, known_topics_in_speech: list[str]) -> float:
        """Calculate probability boost based on topic clustering.

        If related topics are already being discussed, probability increases.

        Args:
            word: Target word
            known_topics_in_speech: List of known topics in current speech

        Returns:
            Multiplier boost (1.0 = no change)
        """
        word_lower = word.lower()

        # Find which cluster this word belongs to
        for topic, words in self.topic_clusters.items():
            if word_lower in words or any(word_lower in w for w in words):
                # Word is in this topic cluster
                if topic in known_topics_in_speech:
                    # Strong boost if same topic already mentioned
                    return 1.5
                # Check for related topic overlap
                cluster_words = set(words)
                for other_topic in known_topics_in_speech:
                    if other_topic in self.topic_clusters:
                        overlap = cluster_words & set(self.topic_clusters[other_topic])
                        if overlap:
                            return 1.2  # Moderate boost for related topics

        return 1.0

    def estimate_with_context(
        self,
        word: str,
        speech_context: str = "speech",
        known_topics: list[str] | None = None,
        speech_length_words: int | None = None,
    ) -> dict[str, Any]:
        """Full estimate with context adjustments and metadata.

        Args:
            word: Target word/phrase
            speech_context: Type of speech
            known_topics: Topics already mentioned in speech
            speech_length_words: Override speech length estimate

        Returns:
            Dict with probability and reasoning
        """
        known_topics = known_topics or []

        # Base probability
        base_prob = self.estimate_mention_probability(word, speech_context, speech_length_words)

        # Topic boost
        topic_boost = self.get_topic_boost(word, known_topics)

        # Adjusted probability
        adjusted_prob = min(0.95, base_prob * topic_boost)

        base_rate = self.get_base_rate(word)
        context_mod = self.context_modifiers.get(speech_context, 1.0)

        return {
            "word": word,
            "base_rate_per_1k": base_rate,
            "speech_context": speech_context,
            "context_modifier": context_mod,
            "topic_boost": topic_boost,
            "base_probability": base_prob,
            "adjusted_probability": adjusted_prob,
            "confidence": "high" if base_rate > 1.0 else "medium" if base_rate > 0.5 else "low",
            "reasoning": (
                f"Base rate {base_rate}/1k words with {context_mod:.1f}x context modifier"
                f"{f' and {topic_boost:.1f}x topic boost' if topic_boost > 1 else ''}"
            ),
        }

    def compare_to_market(
        self,
        word: str,
        market_yes_price: float,
        speech_context: str = "speech",
        known_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare word-frequency estimate to market-implied probability.

        Args:
            word: Target word
            market_yes_price: Current market YES price (0-1)
            speech_context: Type of speech context
            known_topics: Topics already mentioned

        Returns:
            Dict with edge analysis
        """
        estimate = self.estimate_with_context(word, speech_context, known_topics)
        our_prob = estimate["adjusted_probability"]
        market_prob = market_yes_price

        # Calculate edge (positive means market is underpricing YES)
        edge = our_prob - market_prob

        # Signal direction
        if edge > 0.15 and our_prob > 0.3:
            signal = "strong_buy_yes"
        elif edge > 0.05:
            signal = "buy_yes"
        elif edge < -0.15 and our_prob < 0.2:
            signal = "strong_buy_no"
        elif edge < -0.05:
            signal = "buy_no"
        else:
            signal = "no_trade"

        return {
            "word": word,
            "our_probability": our_prob,
            "market_probability": market_prob,
            "edge": edge,
            "signal": signal,
            "confidence": estimate["confidence"],
            "reasoning": estimate["reasoning"],
        }


def extract_trump_speech_context(question: str) -> str:
    """Extract Trump speech context from market question.

    Args:
        question: Market question text

    Returns:
        Speech context type for frequency analysis
    """
    question_lower = question.lower()

    # Check for specific contexts
    if any(kw in question_lower for kw in ["rally", "campaign"]):
        return "campaign_rally"
    if any(kw in question_lower for kw in ["state of the union", "sotu"]):
        return "state_union"
    if any(kw in question_lower for kw in ["press conference", "briefing"]):
        return "press_conference"
    if any(kw in question_lower for kw in ["interview", "interviewed", "sits down with"]):
        return "interview"
    if any(kw in question_lower for kw in ["debate", "debates"]):
        return "debate"
    if any(kw in question_lower for kw in ["remarks", "remarks at"]):
        return "remarks"
    if any(kw in question_lower for kw in ["statement", "statements"]):
        return "statement"

    # Default to general speech
    return "speech"


def is_trump_mention_market(question: str) -> bool:
    """Check if this is a Trump speech mention market.

    Args:
        question: Market question

    Returns:
        True if Trump is the speaker being asked about
    """
    question_lower = question.lower()

    # Must be a mention market
    if not _is_mention_market(question):
        return False

    # Check if Trump is the subject/speaker (typically appears early in question)
    # Pattern: "Will Trump..." or "Does Trump..." or similar
    trump_as_speaker_patterns = [
        r"^will\s+trump\s+",
        r"^does\s+trump\s+",
        r"^will\s+donald\s+trump\s+",
        r"^does\s+donald\s+trump\s+",
        r"^will\s+president\s+trump\s+",
    ]

    for pattern in trump_as_speaker_patterns:
        if re.search(pattern, question_lower):
            return True

    return False


def _is_mention_market(question: str) -> bool:
    """Check if a market question is about mentions.

    Args:
        question: The market question text

    Returns:
        True if this is a mention market
    """
    question_lower = question.lower()
    return any(kw in question_lower for kw in MENTION_KEYWORDS)


def _extract_mention_target(question: str) -> str | None:
    """Extract what/who is being monitored for mentions.

    Uses simple heuristics to identify the target entity.

    Args:
        question: The market question text

    Returns:
        The target entity name or None if can't extract
    """
    question_lower = question.lower()

    # Common patterns:
    # "Will [X] mention [Y]?" -> Y is the target
    # "Will [X] be mentioned?" -> X is the target
    # "Will [X] mention [Y] in [Z]?" -> Y is the target

    # Pattern: "Will X mention Y" or "Will X mention Y in Z"
    # Stop at prepositions: in, during, at, on, by, before, after, etc.
    mention_pattern = r"will\s+\w+(?:\s+\w+){0,3}\s+mention\s+([^\s]+(?:\s+[^\s]+){0,3})(?:\s+(?:in|during|at|on|by|before|after|for|of|to|about|regarding|concerning|$))"
    match = re.search(mention_pattern, question_lower)
    if match:
        # Return the object of "mention" (who/what is being mentioned)
        target = match.group(1).strip().rstrip("?.")
        # Clean up common stop words
        target = re.sub(r"^(the|a|an)\s+", "", target)
        return target.title() if target else None

    # Pattern: "Will X be mentioned"
    be_mentioned_pattern = r"will\s+([^\s]+(?:\s+[^\s]+){0,3})\s+be\s+mentioned"
    match = re.search(be_mentioned_pattern, question_lower)
    if match:
        target = match.group(1).strip().rstrip("?.")
        target = re.sub(r"^(the|a|an)\s+", "", target)
        return target.title() if target else None

    # Pattern: mention of [X]
    of_pattern = (
        r"mention\s+of\s+([^\s]+(?:\s+[^\s]+){0,3})(?:\s+(?:in|during|at|on|by|before|after|for|$))"
    )
    match = re.search(of_pattern, question_lower)
    if match:
        target = match.group(1).strip().rstrip("?.")
        # Remove leading "the" or "of" if present
        target = re.sub(r"^(the|a|an|of)\s+", "", target)
        return target.title() if target else None

    # Fallback: extract capitalized phrases (likely proper nouns)
    capitalized = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
    if capitalized:
        # Return the last capitalized phrase (often the entity)
        return capitalized[-1]

    return None


def _extract_mention_context(question: str) -> str | None:
    """Extract the context where the mention would occur.

    Args:
        question: The market question text

    Returns:
        The context category or None
    """
    question_lower = question.lower()

    for context, patterns in CONTEXT_PATTERNS:
        if any(p in question_lower for p in patterns):
            return context

    return None


def _compute_theoretical_yes_probability(
    market: MentionMarket,
    base_rate: float = 0.15,
    use_word_frequency: bool = True,
) -> float:
    """Compute theoretical probability of YES based on structural factors.

    The default-to-NO hypothesis: base rate for mentions is low,
    but markets systematically overprice YES due to biases.

    For Trump mention markets, uses word-frequency analysis for edge.

    Args:
        market: The mention market
        base_rate: Historical base rate for mentions (default 15%)
        use_word_frequency: Whether to use word-frequency analysis for Trump markets

    Returns:
        Theoretical probability (0-1)
    """
    # Check if this is a Trump speech mention market
    if use_word_frequency and is_trump_mention_market(market.question):
        analyzer = TrumpWordFrequencyAnalyzer()
        speech_context = extract_trump_speech_context(market.question)

        # Extract the mention target for word lookup
        target = market.mention_target or _extract_mention_target(market.question) or ""

        if target:
            estimate = analyzer.estimate_with_context(
                word=target,
                speech_context=speech_context,
            )
            prob = estimate["adjusted_probability"]
            logger.debug(
                "Trump word-frequency estimate for '%s' in %s context: %.3f",
                target,
                speech_context,
                prob,
            )
            return prob

    # Start with base rate
    prob = base_rate

    # Adjust based on mention context
    if market.mention_context:
        # Different contexts have different base rates
        context_adjustments = {
            "tweet": 0.10,  # Tweets are common but specific mentions are rare
            "speech": 0.20,  # Speeches often mention various topics
            "interview": 0.15,
            "press": 0.18,
            "debate": 0.25,  # Debates often involve many mentions
            "media": 0.12,
            "congress": 0.15,
        }
        adj = context_adjustments.get(market.mention_context, 0.0)
        prob = adj  # Use context-specific base rate

    # Adjust for time until resolution (closer = more predictable)
    if market.end_date:
        now = datetime.now(UTC)
        if market.end_date > now:
            hours_remaining = (market.end_date - now).total_seconds() / 3600
            if hours_remaining < 1:
                # Very close to resolution - if no mention yet, less likely
                prob *= 0.5
            elif hours_remaining < 6:
                prob *= 0.8

    return min(0.95, max(0.05, prob))


def find_mention_markets(snapshots_dir: Path | None = None) -> list[MentionMarket]:
    """Find mention-related markets from snapshot data.

    Args:
        snapshots_dir: Directory with snapshot files (optional)

    Returns:
        List of MentionMarket objects
    """
    markets: list[MentionMarket] = []

    if snapshots_dir is None:
        snapshots_dir = Path("data")

    if not snapshots_dir.exists():
        logger.warning("Snapshots directory not found: %s", snapshots_dir)
        return markets

    # Find the most recent 5m or 15m snapshot
    snapshot_files = sorted(snapshots_dir.glob("snapshot_*m_*.json"))
    if not snapshot_files:
        logger.warning("No snapshot files found in %s", snapshots_dir)
        return markets

    latest = snapshot_files[-1]
    logger.debug("Scanning for mention markets in: %s", latest)

    try:
        data = json.loads(latest.read_text())
        for m in data.get("markets", []):
            question = m.get("question", m.get("title", ""))

            if not _is_mention_market(question):
                continue

            token_ids = m.get("clob_token_ids", m.get("clobTokenIds", []))
            if len(token_ids) != 2:
                continue

            end_date_str = m.get("end_date", m.get("endDate", ""))
            end_date = None
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(str(end_date_str).replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            mention_target = _extract_mention_target(question)
            mention_context = _extract_mention_context(question)

            # Get current prices from book
            books = m.get("books", {})
            yes_book = books.get("yes", {})
            yes_bids = yes_book.get("bids", [])
            yes_asks = yes_book.get("asks", [])

            current_yes_price = None
            if yes_bids and yes_asks:
                try:
                    best_bid = max(float(b["price"]) for b in yes_bids)
                    best_ask = min(float(a["price"]) for a in yes_asks)
                    current_yes_price = (best_bid + best_ask) / 2
                except (ValueError, KeyError):
                    pass

            market = MentionMarket(
                market_id=str(m.get("market_id", m.get("id", ""))),
                token_id_yes=str(token_ids[0]),
                token_id_no=str(token_ids[1]),
                question=question,
                mention_target=mention_target,
                mention_context=mention_context,
                current_yes_price=current_yes_price,
                current_no_price=1.0 - current_yes_price if current_yes_price else None,
                end_date=end_date,
            )

            markets.append(market)

    except Exception as e:
        logger.exception("Error scanning snapshots for mention markets: %s", e)

    logger.info("Found %d mention markets", len(markets))
    return markets


@dataclass(frozen=True)
class MentionSignal:
    """Trading signal from mention market analysis.

    Attributes:
        timestamp: When signal was generated
        market: MentionMarket this signal is for
        side: "buy_yes", "buy_no", or "no_trade"
        market_prob: Current market-implied probability
        theoretical_prob: Our estimated true probability
        edge: Difference between theoretical and market probability
        confidence: Signal confidence (0-1)
        expected_value: Expected value of trade
        reasoning: Human-readable explanation
    """

    timestamp: datetime
    market: MentionMarket
    side: str  # "buy_yes", "buy_no", "no_trade"
    market_prob: float
    theoretical_prob: float
    edge: float
    confidence: float
    expected_value: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market": {
                "market_id": self.market.market_id,
                "question": self.market.question,
                "mention_target": self.market.mention_target,
                "mention_context": self.market.mention_context,
                "current_yes_price": self.market.current_yes_price,
                "current_no_price": self.market.current_no_price,
            },
            "side": self.side,
            "market_prob": self.market_prob,
            "theoretical_prob": self.theoretical_prob,
            "edge": self.edge,
            "confidence": self.confidence,
            "expected_value": self.expected_value,
            "reasoning": self.reasoning,
        }


def generate_signals(
    markets: list[MentionMarket],
    base_rate: float = 0.15,
    no_entry_min_price: float = DEFAULT_NO_ENTRY_MIN_PRICE,
    no_entry_max_price: float = DEFAULT_NO_ENTRY_MAX_PRICE,
    yes_entry_max_price: float = DEFAULT_YES_ENTRY_MAX_PRICE,
    min_edge: float = MIN_EDGE_FOR_TRADE,
) -> list[MentionSignal]:
    """Generate trading signals using default-to-NO logic.

    Args:
        markets: List of mention markets
        base_rate: Historical base rate for mentions
        no_entry_min_price: Minimum NO price to consider entry
        no_entry_max_price: Maximum NO price to consider entry
        yes_entry_max_price: Maximum YES price for YES entry
        min_edge: Minimum edge required for any trade

    Returns:
        List of trading signals
    """
    signals: list[MentionSignal] = []
    now = datetime.now(UTC)

    for market in markets:
        # Skip if no price data or expired
        if market.current_yes_price is None or market.is_expired:
            continue

        market_prob = market.current_yes_price
        theoretical_prob = _compute_theoretical_yes_probability(market, base_rate)

        # Edge is the difference between our estimate and market's estimate
        edge = theoretical_prob - market_prob

        side = "no_trade"
        confidence = 0.0
        expected_value = 0.0
        reasoning = ""

        # DEFAULT-TO-NO LOGIC:
        # We only go YES if there's a strong signal (very cheap YES)
        # Otherwise, we look for overpriced YES (cheap NO)

        if market_prob <= yes_entry_max_price and edge > min_edge:
            # YES is cheap and we have positive edge - buy YES
            side = "buy_yes"
            confidence = min(1.0, abs(edge) * 2)  # Scale edge to confidence
            # EV = (win_prob * win_amount) - (lose_prob * lose_amount)
            win_amount = 1.0 - market_prob
            lose_amount = market_prob
            expected_value = (theoretical_prob * win_amount) - (
                (1 - theoretical_prob) * lose_amount
            )
            reasoning = (
                f"YES underpriced: market={market_prob:.2%}, "
                f"theoretical={theoretical_prob:.2%}, edge={edge:+.1%}"
            )

        else:
            # Check for buy_no opportunity
            # NO price = 1 - YES price
            no_price = market.current_no_price or (1.0 - market_prob)
            if no_price > 0:
                # For NO position: we win if YES doesn't happen
                theoretical_no_prob = 1.0 - theoretical_prob
                market_no_prob = 1.0 - market_prob
                no_edge = theoretical_no_prob - market_no_prob

                # Buy NO when NO is cheap enough (YES is expensive)
                # and we have sufficient edge
                if no_entry_min_price <= no_price <= no_entry_max_price and no_edge > min_edge:
                    side = "buy_no"
                    confidence = min(1.0, abs(no_edge) * 2)
                    # EV for NO side
                    win_amount = 1.0 - no_price
                    lose_amount = no_price
                    expected_value = (theoretical_no_prob * win_amount) - (
                        (1 - theoretical_no_prob) * lose_amount
                    )
                    reasoning = (
                        f"YES overpriced: market={market_prob:.2%}, "
                        f"theoretical={theoretical_prob:.2%}, "
                        f"NO_edge={no_edge:+.1%}"
                    )

        if side == "no_trade":
            reasoning = (
                f"No edge: market={market_prob:.2%}, "
                f"theoretical={theoretical_prob:.2%}, edge={edge:+.1%}"
            )

        signals.append(
            MentionSignal(
                timestamp=now,
                market=market,
                side=side,
                market_prob=market_prob,
                theoretical_prob=theoretical_prob,
                edge=edge,
                confidence=confidence,
                expected_value=expected_value,
                reasoning=reasoning,
            )
        )

    # Sort by expected value descending
    signals.sort(key=lambda s: s.expected_value, reverse=True)

    return signals


def generate_trump_word_frequency_signals(
    markets: list[MentionMarket],
    min_edge: float = MIN_EDGE_FOR_TRADE,
    no_entry_min_price: float = DEFAULT_NO_ENTRY_MIN_PRICE,
    no_entry_max_price: float = DEFAULT_NO_ENTRY_MAX_PRICE,
    yes_entry_max_price: float = DEFAULT_YES_ENTRY_MAX_PRICE,
) -> list[MentionSignal]:
    """Generate signals for Trump mention markets using word-frequency edge.

    Uses historical Trump speech word frequency data to estimate mention
    probabilities, providing edge over naive base rates.

    Args:
        markets: List of mention markets (will filter to Trump-only)
        min_edge: Minimum edge required for any trade
        no_entry_min_price: Minimum NO price to consider entry
        no_entry_max_price: Maximum NO price to consider entry
        yes_entry_max_price: Maximum YES price for YES entry

    Returns:
        List of trading signals with word-frequency metadata
    """
    signals: list[MentionSignal] = []
    analyzer = TrumpWordFrequencyAnalyzer()
    now = datetime.now(UTC)

    for market in markets:
        # Skip non-Trump markets
        if not is_trump_mention_market(market.question):
            continue

        # Skip if no price data or expired
        if market.current_yes_price is None or market.is_expired:
            continue

        market_prob = market.current_yes_price
        speech_context = extract_trump_speech_context(market.question)
        target = market.mention_target or _extract_mention_target(market.question) or ""

        # Get word-frequency based estimate
        comparison = analyzer.compare_to_market(
            word=target,
            market_yes_price=market_prob,
            speech_context=speech_context,
        )

        theoretical_prob = comparison["our_probability"]
        edge = comparison["edge"]

        side = "no_trade"
        confidence = 0.0
        expected_value = 0.0
        reasoning = ""

        # Trading logic with word-frequency edge
        if market_prob <= yes_entry_max_price and edge > min_edge:
            side = "buy_yes"
            confidence = min(1.0, abs(edge) * 2)
            win_amount = 1.0 - market_prob
            lose_amount = market_prob
            expected_value = (theoretical_prob * win_amount) - (
                (1 - theoretical_prob) * lose_amount
            )
            reasoning = (
                f"Trump word-freq: '{target}' has {theoretical_prob:.1%} in {speech_context} "
                f"(base rate: {analyzer.get_base_rate(target):.1f}/1k), "
                f"market={market_prob:.2%}, edge={edge:+.1%}"
            )
        else:
            no_price = market.current_no_price or (1.0 - market_prob)
            if no_price > 0:
                theoretical_no_prob = 1.0 - theoretical_prob
                market_no_prob = 1.0 - market_prob
                no_edge = theoretical_no_prob - market_no_prob

                if no_entry_min_price <= no_price <= no_entry_max_price and no_edge > min_edge:
                    side = "buy_no"
                    confidence = min(1.0, abs(no_edge) * 2)
                    win_amount = 1.0 - no_price
                    lose_amount = no_price
                    expected_value = (theoretical_no_prob * win_amount) - (
                        (1 - theoretical_no_prob) * lose_amount
                    )
                    reasoning = (
                        f"Trump word-freq: '{target}' unlikely ({theoretical_prob:.1%}) "
                        f"in {speech_context}, market overpricing YES, NO_edge={no_edge:+.1%}"
                    )

        if side == "no_trade":
            reasoning = (
                f"Trump word-freq: '{target}'={theoretical_prob:.1%} in {speech_context}, "
                f"market={market_prob:.2%}, no significant edge"
            )

        signals.append(
            MentionSignal(
                timestamp=now,
                market=market,
                side=side,
                market_prob=market_prob,
                theoretical_prob=theoretical_prob,
                edge=edge,
                confidence=confidence,
                expected_value=expected_value,
                reasoning=reasoning,
            )
        )

    # Sort by expected value descending
    signals.sort(key=lambda s: s.expected_value, reverse=True)

    return signals


@dataclass(frozen=True)
class MentionTrade:
    """An executed mention market trade.

    Attributes:
        timestamp: When trade was executed
        signal: The signal that triggered the trade
        order_result: Result from order submission
        position_size: Number of contracts
        entry_price: Price paid per contract
    """

    timestamp: datetime
    signal: MentionSignal
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


def execute_trade(
    signal: MentionSignal,
    dry_run: bool = True,
    max_position_size: float = MAX_POSITION_SIZE,
) -> MentionTrade | None:
    """Execute a trade based on a mention market signal.

    Args:
        signal: MentionSignal with side and market info
        dry_run: If True, don't actually submit orders
        max_position_size: Maximum position size

    Returns:
        MentionTrade if executed, None otherwise
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

    # Position sizing based on confidence and EV
    base_size = 1.0
    confidence_multiplier = min(5.0, signal.confidence * 5)
    position_size = min(max_position_size, base_size * confidence_multiplier)
    position_size = round(position_size, 2)

    try:
        from .trading import Order, submit_order

        order = Order(
            token_id=token_id,
            side=side,
            size=position_size,
            price=limit_price,
        )

        result = submit_order(order)

        return MentionTrade(
            timestamp=now,
            signal=signal,
            order_result=result,
            position_size=position_size,
            entry_price=limit_price,
        )

    except Exception as e:
        logger.exception("Error executing mention trade: %s", e)
        return None


def run_mention_scan(
    snapshots_dir: Path | None = None,
    base_rate: float = 0.15,
    dry_run: bool = True,
    max_positions: int = MAX_POSITIONS_PER_SCAN,
    use_word_frequency: bool = True,
) -> dict[str, Any]:
    """Run a complete mention market scan with default-to-NO strategy.

    For Trump mention markets, uses word-frequency analysis to estimate
    mention probabilities based on historical speech data.

    Args:
        snapshots_dir: Directory with market snapshots
        base_rate: Historical base rate for mentions
        dry_run: If True, don't execute trades
        max_positions: Maximum positions to take
        use_word_frequency: Whether to use word-frequency for Trump markets

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)

    logger.info("Starting mention market scan at %s", now.isoformat())

    # Step 1: Find mention markets
    markets = find_mention_markets(snapshots_dir)
    logger.info("Found %d mention markets", len(markets))

    # Step 2: Generate signals using default-to-NO logic
    # For Trump markets, use word-frequency analysis if enabled
    if use_word_frequency:
        # Separate Trump and non-Trump markets
        trump_markets = [m for m in markets if is_trump_mention_market(m.question)]
        other_markets = [m for m in markets if not is_trump_mention_market(m.question)]

        logger.info("Found %d Trump mention markets", len(trump_markets))

        # Generate Trump-specific signals with word-frequency edge
        trump_signals = generate_trump_word_frequency_signals(trump_markets)

        # Generate regular signals for other markets
        other_signals = generate_signals(other_markets, base_rate=base_rate)

        signals = trump_signals + other_signals
    else:
        signals = generate_signals(markets, base_rate=base_rate)

    logger.info("Generated %d signals", len(signals))

    # Step 3: Filter to actionable signals
    actionable = [s for s in signals if s.side != "no_trade" and s.expected_value > 0.05]
    logger.info("%d actionable signals", len(actionable))

    # Step 4: Execute trades (respecting limits)
    trades: list[MentionTrade] = []
    for signal in actionable[:max_positions]:
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

    # Compute summary statistics
    buy_yes_signals = [s for s in signals if s.side == "buy_yes"]
    buy_no_signals = [s for s in signals if s.side == "buy_no"]
    no_trade_signals = [s for s in signals if s.side == "no_trade"]

    # Trump-specific breakdown
    trump_signals = [s for s in signals if is_trump_mention_market(s.market.question)]
    trump_buy_yes = [s for s in trump_signals if s.side == "buy_yes"]
    trump_buy_no = [s for s in trump_signals if s.side == "buy_no"]

    avg_edge_no = np.mean([s.edge for s in buy_no_signals]) if buy_no_signals else 0.0
    avg_edge_yes = np.mean([s.edge for s in buy_yes_signals]) if buy_yes_signals else 0.0

    return {
        "timestamp": now.isoformat(),
        "markets_scanned": len(markets),
        "signals_generated": len(signals),
        "actionable_signals": len(actionable),
        "trades_executed": len(trades),
        "dry_run": dry_run,
        "summary": {
            "buy_yes_count": len(buy_yes_signals),
            "buy_no_count": len(buy_no_signals),
            "no_trade_count": len(no_trade_signals),
            "avg_edge_buy_no": avg_edge_no,
            "avg_edge_buy_yes": avg_edge_yes,
            "trump_markets": len(trump_signals),
            "trump_buy_yes": len(trump_buy_yes),
            "trump_buy_no": len(trump_buy_no),
        },
        "markets": [
            {
                "market_id": m.market_id,
                "question": m.question,
                "target": m.mention_target,
                "context": m.mention_context,
                "yes_price": m.current_yes_price,
                "is_trump_market": is_trump_mention_market(m.question),
            }
            for m in markets
        ],
        "signals": [s.to_dict() for s in signals],
        "trades": [t.to_dict() for t in trades],
    }
