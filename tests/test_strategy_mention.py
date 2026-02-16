"""Tests for strategy_mention module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polymarket.strategy_mention import (
    MENTION_KEYWORDS,
    CONTEXT_PATTERNS,
    MentionMarket,
    MentionSignal,
    _compute_theoretical_yes_probability,
    _extract_mention_context,
    _extract_mention_target,
    _is_mention_market,
    execute_trade,
    find_mention_markets,
    generate_signals,
    run_mention_scan,
)


class TestIsMentionMarket:
    """Tests for _is_mention_market function."""

    def test_detects_mention_keyword(self) -> None:
        assert _is_mention_market("Will Trump mention Biden in his speech?")

    def test_detects_mentions_keyword(self) -> None:
        assert _is_mention_market("Will the Fed mention inflation in their statement?")

    def test_detects_mentioned_keyword(self) -> None:
        assert _is_mention_market("Will Elon be mentioned at the conference?")

    def test_detects_name_dropped(self) -> None:
        assert _is_mention_market("Will Bitcoin be name-dropped in the debate?")

    def test_detects_shoutout(self) -> None:
        assert _is_mention_market("Will the CEO give a shoutout to the team?")

    def test_detects_reference(self) -> None:
        assert _is_mention_market("Will the report reference previous findings?")

    def test_rejects_non_mention(self) -> None:
        assert not _is_mention_market("Will Bitcoin go up in the next hour?")

    def test_rejects_partial_match(self) -> None:
        # "mention" must be a separate word
        assert not _is_mention_market("Will the dimension change?")

    def test_all_keywords_present(self) -> None:
        for kw in MENTION_KEYWORDS:
            assert _is_mention_market(f"Will X {kw} Y?")


class TestExtractMentionTarget:
    """Tests for _extract_mention_target function."""

    def test_extracts_simple_target(self) -> None:
        result = _extract_mention_target("Will Trump mention Biden in his speech?")
        assert result == "Biden"

    def test_extracts_target_after_mention(self) -> None:
        result = _extract_mention_target("Will Biden mention student loans tonight?")
        assert result == "Student Loans"

    def test_extracts_be_mentioned_target(self) -> None:
        result = _extract_mention_target("Will Bitcoin be mentioned in the debate?")
        assert result == "Bitcoin"

    def test_extracts_mention_of_target(self) -> None:
        result = _extract_mention_target("Will there be a mention of tariffs in the speech?")
        # The regex extracts "tariffs" which gets title-cased
        assert "Tariffs" in result

    def test_cleans_stop_words(self) -> None:
        result = _extract_mention_target("Will Biden mention the economy?")
        # "Will Biden mention" pattern: "Will" followed by "Biden" then "mention"
        # The mention_pattern extracts what's after "mention"
        # In this case, "the economy" -> after cleaning stop words -> "Economy"
        assert result is not None

    def test_extracts_multi_word_target(self) -> None:
        result = _extract_mention_target("Will Trump mention climate change in his speech?")
        # Should extract up to 5 words after "mention"
        assert "Climate" in result

    def test_fallback_to_capitalized(self) -> None:
        result = _extract_mention_target("Will Someone mention Something Important?")
        assert result is not None

    def test_returns_none_for_no_match(self) -> None:
        # This should fall back to capitalized words since there's no "mention" pattern
        result = _extract_mention_target("will it rain tomorrow?")
        # "Will" is capitalized in the original, but lowercase here so no match
        assert result is None


class TestExtractMentionContext:
    """Tests for _extract_mention_context function."""

    def test_detects_tweet_context(self) -> None:
        assert _extract_mention_context("Will Trump mention Biden in his tweet?") == "tweet"
        assert _extract_mention_context("Will he post on X?") == "tweet"
        assert _extract_mention_context("Will she post on Twitter?") == "tweet"

    def test_detects_speech_context(self) -> None:
        assert _extract_mention_context("Will he mention it in his speech?") == "speech"
        assert _extract_mention_context("Will the address cover it?") == "speech"

    def test_detects_interview_context(self) -> None:
        assert _extract_mention_context("Will he mention it in the interview?") == "interview"

    def test_detects_press_context(self) -> None:
        assert _extract_mention_context("Will the press conference cover it?") == "press"

    def test_detects_debate_context(self) -> None:
        assert _extract_mention_context("Will the debate mention it?") == "debate"

    def test_detects_media_context(self) -> None:
        assert _extract_mention_context("Will CNN mention it?") == "media"
        assert _extract_mention_context("Will the article reference it?") == "media"

    def test_detects_congress_context(self) -> None:
        assert _extract_mention_context("Will Congress mention it?") == "congress"

    def test_returns_none_for_no_match(self) -> None:
        assert _extract_mention_context("Will it happen?") is None

    def test_all_context_patterns(self) -> None:
        for context, patterns in CONTEXT_PATTERNS:
            for pattern in patterns:
                test_q = f"Will he mention it in {pattern}?"
                result = _extract_mention_context(test_q)
                assert result == context, f"Failed for pattern: {pattern}"


class TestComputeTheoreticalProbability:
    """Tests for _compute_theoretical_yes_probability function."""

    def test_default_base_rate(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
        )
        result = _compute_theoretical_yes_probability(market, base_rate=0.15)
        assert result == 0.15

    def test_context_adjustments(self) -> None:
        for context, expected_base in [
            ("tweet", 0.10),
            ("speech", 0.20),
            ("debate", 0.25),
            ("interview", 0.15),
            ("press", 0.18),
            ("media", 0.12),
            ("congress", 0.15),
        ]:
            market = MentionMarket(
                market_id="test",
                token_id_yes="yes",
                token_id_no="no",
                question="Will X mention Y?",
                mention_target="Y",
                mention_context=context,
            )
            result = _compute_theoretical_yes_probability(market)
            assert result == expected_base, f"Failed for context: {context}"

    def test_time_decay_under_1_hour(self) -> None:
        end_date = datetime.now(UTC) + timedelta(minutes=30)
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="tweet",
            end_date=end_date,
        )
        result = _compute_theoretical_yes_probability(market)
        # Base rate for tweet is 0.10, with < 1 hour multiplier of 0.5
        assert result == pytest.approx(0.05, abs=0.01)

    def test_time_decay_under_6_hours(self) -> None:
        end_date = datetime.now(UTC) + timedelta(hours=3)
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="tweet",
            end_date=end_date,
        )
        result = _compute_theoretical_yes_probability(market)
        # Base rate 0.10 with 0.8 multiplier
        assert result == pytest.approx(0.08, abs=0.01)

    def test_clamps_to_bounds(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
        )
        # Very high base rate should be clamped to 0.95
        result = _compute_theoretical_yes_probability(market, base_rate=0.99)
        assert result == 0.95

        # Very low base rate should be clamped to 0.05
        result = _compute_theoretical_yes_probability(market, base_rate=0.01)
        assert result == 0.05


class TestGenerateSignals:
    """Tests for generate_signals function."""

    def test_no_trade_when_no_edge(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="tweet",  # 0.10 base rate
            current_yes_price=0.10,  # Matches base rate, no edge
        )
        signals = generate_signals([market])
        assert len(signals) == 1
        assert signals[0].side == "no_trade"
        assert signals[0].edge == pytest.approx(0.0, abs=0.01)

    def test_buy_no_when_yes_overpriced(self) -> None:
        # With default thresholds: NO_ENTRY_MIN_PRICE=0.35, NO_ENTRY_MAX_PRICE=0.50
        # We buy NO when NO price is between 0.35-0.50 (YES between 0.50-0.65)
        # This captures overpriced YES without paying too much for NO
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="tweet",  # 0.10 base rate
            current_yes_price=0.60,  # YES at 0.60 -> NO at 0.40 (within entry range)
            current_no_price=0.40,
        )
        signals = generate_signals([market], min_edge=0.05)
        assert len(signals) == 1
        # With YES at 0.60 and base rate 0.10, NO edge = (1-0.10) - (1-0.60) = 0.90 - 0.40 = 0.50
        # This is a strong buy_no signal
        assert signals[0].side == "buy_no"
        assert signals[0].edge < 0  # Negative edge for YES = positive for NO

    def test_buy_yes_when_underpriced(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="debate",  # 0.25 base rate
            current_yes_price=0.15,  # YES underpriced
            current_no_price=0.85,
        )
        signals = generate_signals([market], min_edge=0.05)
        assert len(signals) == 1
        assert signals[0].side == "buy_yes"
        assert signals[0].edge > 0

    def test_respects_min_edge_threshold(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="tweet",  # 0.10 base rate
            current_yes_price=0.50,  # Some mispricing
            current_no_price=0.50,
        )
        # With high min_edge, should not trade
        signals = generate_signals([market], min_edge=0.50)
        assert signals[0].side == "no_trade"

    def test_skips_expired_markets(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="tweet",
            current_yes_price=0.80,
            current_no_price=0.20,
            end_date=datetime.now(UTC) - timedelta(hours=1),  # Expired
        )
        signals = generate_signals([market])
        # Should skip entirely (no signal generated for expired)
        assert len([s for s in signals if s.market.market_id == "test"]) == 0

    def test_sorts_by_expected_value(self) -> None:
        markets = [
            MentionMarket(
                market_id=f"test{i}",
                token_id_yes="yes",
                token_id_no="no",
                question="Will X mention Y?",
                mention_target="Y",
                mention_context="tweet",
                current_yes_price=price,
                current_no_price=1.0 - price,
            )
            for i, price in enumerate([0.15, 0.90, 0.50])  # Low, high, mid
        ]
        signals = generate_signals(markets, min_edge=0.05)
        evs = [s.expected_value for s in signals]
        # Should be sorted descending
        assert evs == sorted(evs, reverse=True)

    def test_expected_value_calculation(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="debate",  # 0.25 base rate
            current_yes_price=0.20,  # Undervalued
        )
        signals = generate_signals([market], min_edge=0.01)
        signal = signals[0]
        if signal.side == "buy_yes":
            # EV = (0.25 * 0.80) - (0.75 * 0.20) = 0.20 - 0.15 = 0.05
            assert signal.expected_value > 0


class TestMentionSignal:
    """Tests for MentionSignal dataclass."""

    def test_to_dict(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context="tweet",
            current_yes_price=0.15,
        )
        signal = MentionSignal(
            timestamp=datetime.now(UTC),
            market=market,
            side="buy_yes",
            market_prob=0.15,
            theoretical_prob=0.25,
            edge=0.10,
            confidence=0.50,
            expected_value=0.05,
            reasoning="Test reasoning",
        )
        d = signal.to_dict()
        assert d["side"] == "buy_yes"
        assert d["market_prob"] == 0.15
        assert d["edge"] == 0.10
        assert d["reasoning"] == "Test reasoning"


class TestMentionMarket:
    """Tests for MentionMarket dataclass."""

    def test_implied_probability(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
            current_yes_price=0.35,
        )
        assert market.implied_probability == 0.35

    def test_is_expired_true(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
            end_date=datetime.now(UTC) - timedelta(hours=1),
        )
        assert market.is_expired

    def test_is_expired_false(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
            end_date=datetime.now(UTC) + timedelta(hours=1),
        )
        assert not market.is_expired

    def test_is_expired_no_end_date(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
            end_date=None,
        )
        assert not market.is_expired


class TestExecuteTrade:
    """Tests for execute_trade function."""

    def test_returns_none_for_no_trade(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
        )
        signal = MentionSignal(
            timestamp=datetime.now(UTC),
            market=market,
            side="no_trade",
            market_prob=0.50,
            theoretical_prob=0.50,
            edge=0.0,
            confidence=0.0,
            expected_value=0.0,
            reasoning="No trade",
        )
        result = execute_trade(signal, dry_run=True)
        assert result is None

    def test_returns_none_for_invalid_side(self) -> None:
        market = MentionMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Will X mention Y?",
            mention_target="Y",
            mention_context=None,
        )
        signal = MentionSignal(
            timestamp=datetime.now(UTC),
            market=market,
            side="invalid_side",
            market_prob=0.50,
            theoretical_prob=0.50,
            edge=0.0,
            confidence=0.0,
            expected_value=0.0,
            reasoning="Invalid",
        )
        result = execute_trade(signal, dry_run=True)
        assert result is None


class TestFindMentionMarkets:
    """Tests for find_mention_markets function."""

    def test_returns_empty_list_for_missing_dir(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent"
        result = find_mention_markets(nonexistent)
        assert result == []

    def test_finds_mention_markets_in_snapshot(self, tmp_path: Path) -> None:
        # Create a test snapshot
        snapshot = {
            "markets": [
                {
                    "market_id": "test1",
                    "question": "Will Trump mention Biden in his speech?",
                    "clobTokenIds": ["yes1", "no1"],
                    "endDate": "2026-02-16T00:00:00Z",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.35", "size": "100"}],
                            "asks": [{"price": "0.37", "size": "100"}],
                        }
                    },
                },
                {
                    "market_id": "test2",
                    "question": "Will Bitcoin go up in the next hour?",
                    "clobTokenIds": ["yes2", "no2"],
                    "endDate": "2026-02-16T00:00:00Z",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.50", "size": "100"}],
                            "asks": [{"price": "0.52", "size": "100"}],
                        }
                    },
                },
            ]
        }

        # Write snapshot file
        snapshot_file = tmp_path / "snapshot_5m_20260215T120000Z.json"
        snapshot_file.write_text(json.dumps(snapshot))

        result = find_mention_markets(tmp_path)

        # Should find only the mention market
        assert len(result) == 1
        assert result[0].market_id == "test1"
        assert result[0].mention_target == "Biden"
        assert result[0].mention_context == "speech"
        assert result[0].current_yes_price == pytest.approx(0.36, abs=0.01)

    def test_handles_empty_markets(self, tmp_path: Path) -> None:
        snapshot = {"markets": []}
        snapshot_file = tmp_path / "snapshot_5m_20260215T120000Z.json"
        snapshot_file.write_text(json.dumps(snapshot))

        result = find_mention_markets(tmp_path)
        assert result == []

    def test_handles_missing_books(self, tmp_path: Path) -> None:
        snapshot = {
            "markets": [
                {
                    "market_id": "test1",
                    "question": "Will Trump mention Biden?",
                    "clobTokenIds": ["yes1", "no1"],
                    "endDate": "2026-02-16T00:00:00Z",
                    # No books key
                },
            ]
        }
        snapshot_file = tmp_path / "snapshot_5m_20260215T120000Z.json"
        snapshot_file.write_text(json.dumps(snapshot))

        result = find_mention_markets(tmp_path)
        assert len(result) == 1
        assert result[0].current_yes_price is None


class TestRunMentionScan:
    """Tests for run_mention_scan function."""

    def test_returns_scan_results(self, tmp_path: Path) -> None:
        # Create test snapshot
        snapshot = {
            "markets": [
                {
                    "market_id": "test1",
                    "question": "Will Trump mention Biden in his speech?",
                    "clobTokenIds": ["yes1", "no1"],
                    "endDate": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.70", "size": "100"}],
                            "asks": [{"price": "0.72", "size": "100"}],
                        }
                    },
                },
            ]
        }
        snapshot_file = tmp_path / "snapshot_5m_20260215T120000Z.json"
        snapshot_file.write_text(json.dumps(snapshot))

        result = run_mention_scan(snapshots_dir=tmp_path, dry_run=True)

        assert result["markets_scanned"] == 1
        assert result["signals_generated"] == 1
        assert result["dry_run"] is True
        assert "summary" in result
        assert "markets" in result
        assert "signals" in result
        assert "trades" in result

    def test_respects_max_positions(self, tmp_path: Path) -> None:
        # Create multiple markets
        markets = []
        for i in range(5):
            markets.append(
                {
                    "market_id": f"test{i}",
                    "question": f"Will X{i} mention Y{i}?",
                    "clobTokenIds": [f"yes{i}", f"no{i}"],
                    "endDate": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.80", "size": "100"}],
                            "asks": [{"price": "0.82", "size": "100"}],
                        }
                    },
                }
            )

        snapshot = {"markets": markets}
        snapshot_file = tmp_path / "snapshot_5m_20260215T120000Z.json"
        snapshot_file.write_text(json.dumps(snapshot))

        result = run_mention_scan(snapshots_dir=tmp_path, dry_run=True, max_positions=2)

        # Should limit trades to max_positions
        assert result["trades_executed"] <= 2

    def test_dry_run_flag(self, tmp_path: Path) -> None:
        snapshot = {
            "markets": [
                {
                    "market_id": "test1",
                    "question": "Will Trump mention Biden?",
                    "clobTokenIds": ["yes1", "no1"],
                    "endDate": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.80", "size": "100"}],
                            "asks": [{"price": "0.82", "size": "100"}],
                        }
                    },
                },
            ]
        }
        snapshot_file = tmp_path / "snapshot_5m_20260215T120000Z.json"
        snapshot_file.write_text(json.dumps(snapshot))

        # Test dry_run=True
        result = run_mention_scan(snapshots_dir=tmp_path, dry_run=True)
        assert result["dry_run"] is True

    def test_returns_summary_stats(self, tmp_path: Path) -> None:
        # Create markets with different signals
        # Note: "Will X mention Y" has tweet context (0.10 base rate)
        # "Will A mention B" also has tweet context (0.10 base rate)
        # debate context has 0.25 base rate
        markets = [
            {
                "market_id": "cheap_yes",
                "question": "Will X mention Y during the debate?",  # debate context = 0.25 base
                "clobTokenIds": ["yes1", "no1"],
                "endDate": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                "books": {
                    "yes": {
                        "bids": [{"price": "0.10", "size": "100"}],  # YES at 0.10
                        "asks": [{"price": "0.12", "size": "100"}],
                    }
                },
            },
            {
                "market_id": "expensive_yes",
                "question": "Will A mention B?",  # tweet context = 0.10 base
                "clobTokenIds": ["yes2", "no2"],
                "endDate": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                "books": {
                    "yes": {
                        # YES at 0.60 -> NO at 0.40 (within NO entry range 0.35-0.50)
                        "bids": [{"price": "0.59", "size": "100"}],
                        "asks": [{"price": "0.61", "size": "100"}],
                    }
                },
            },
            {
                "market_id": "no_trade",
                "question": "Will C mention D?",  # tweet context = 0.10 base
                "clobTokenIds": ["yes3", "no3"],
                "endDate": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                "books": {
                    "yes": {
                        # YES at 0.40 -> NO at 0.60 (above NO entry max of 0.50)
                        # Price is close to fair value but NO too expensive to buy
                        "bids": [{"price": "0.39", "size": "100"}],
                        "asks": [{"price": "0.41", "size": "100"}],
                    }
                },
            },
        ]

        snapshot = {"markets": markets}
        snapshot_file = tmp_path / "snapshot_5m_20260215T120000Z.json"
        snapshot_file.write_text(json.dumps(snapshot))

        result = run_mention_scan(snapshots_dir=tmp_path, dry_run=True, base_rate=0.15)

        # cheap_yes: debate (0.25) vs YES at 0.11 -> buy_yes
        # expensive_yes: tweet (0.15) vs YES at 0.60 -> buy_no (NO at 0.40)
        # no_trade: tweet (0.15) vs YES at 0.40 -> no_trade (NO at 0.60 > 0.50 max)
        assert result["summary"]["buy_yes_count"] >= 1
        assert result["summary"]["buy_no_count"] >= 1
        assert result["summary"]["no_trade_count"] >= 1


class TestTrumpWordFrequencyAnalyzer:
    """Tests for TrumpWordFrequencyAnalyzer class."""

    def test_get_base_rate_known_word(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        rate = analyzer.get_base_rate("biden")
        assert rate == 8.5  # From TRUMP_SPEECH_WORD_FREQUENCY

    def test_get_base_rate_unknown_word(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        rate = analyzer.get_base_rate("xyzunknown")
        assert rate == 0.1  # Default for unknown words

    def test_estimate_mention_probability(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        # Use a word with moderate frequency (tariffs = 4.2 per 1k)
        prob = analyzer.estimate_mention_probability("tariffs", speech_context="speech")

        # Should have moderate probability
        assert 0.05 < prob <= 0.95
        assert isinstance(prob, float)

    def test_context_modifiers_affect_probability(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()

        # Use words with lower frequency to avoid capping at 0.95
        # nato has 2.8 per 1k words, which should give ~75% probability for 5000 words
        # with modifiers 1.5 vs 0.9, we should see a difference
        prob_rally = analyzer.estimate_mention_probability("nato", speech_context="campaign_rally")
        prob_statement = analyzer.estimate_mention_probability("nato", speech_context="statement")

        # Rally context has higher modifier (1.5) than statement (0.9)
        assert prob_rally > prob_statement

    def test_compare_to_market_buy_signal(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        result = analyzer.compare_to_market(
            word="biden",
            market_yes_price=0.05,  # Market thinks 5% chance
            speech_context="campaign_rally",
        )

        assert result["word"] == "biden"
        assert result["market_probability"] == 0.05
        assert result["edge"] > 0  # We think it's more likely than market
        assert result["signal"] in ["buy_yes", "strong_buy_yes"]

    def test_compare_to_market_sell_signal(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        result = analyzer.compare_to_market(
            word="xyzunknown",  # Very unlikely word
            market_yes_price=0.80,  # Market thinks 80% chance
            speech_context="speech",
        )

        assert result["edge"] < 0  # We think it's less likely than market
        assert result["signal"] in ["buy_no", "strong_buy_no"]

    def test_estimate_with_context(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        result = analyzer.estimate_with_context(
            word="biden",
            speech_context="campaign_rally",
            known_topics=["immigration"],
        )

        assert result["word"] == "biden"
        assert result["speech_context"] == "campaign_rally"
        assert result["base_rate_per_1k"] == 8.5
        assert result["context_modifier"] == 1.5
        assert "base_probability" in result
        assert "adjusted_probability" in result
        assert "confidence" in result
        assert "reasoning" in result

    def test_topic_boost_for_related_topics(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        # "border" is in the immigration topic cluster with "immigration"
        boost = analyzer.get_topic_boost("border", known_topics_in_speech=["immigration"])
        assert boost == 1.5

    def test_no_topic_boost_for_unrelated(self) -> None:
        from polymarket.strategy_mention import TrumpWordFrequencyAnalyzer

        analyzer = TrumpWordFrequencyAnalyzer()
        boost = analyzer.get_topic_boost("biden", known_topics_in_speech=["economy"])
        assert boost == 1.0


class TestTrumpSpeechContextExtraction:
    """Tests for extract_trump_speech_context function."""

    def test_extract_rally_context(self) -> None:
        from polymarket.strategy_mention import extract_trump_speech_context

        question = "Will Trump mention Biden at his campaign rally?"
        assert extract_trump_speech_context(question) == "campaign_rally"

    def test_extract_debate_context(self) -> None:
        from polymarket.strategy_mention import extract_trump_speech_context

        question = "Will Trump mention tariffs during the debate?"
        assert extract_trump_speech_context(question) == "debate"

    def test_extract_interview_context(self) -> None:
        from polymarket.strategy_mention import extract_trump_speech_context

        question = "Will Trump mention China in his interview?"
        assert extract_trump_speech_context(question) == "interview"

    def test_default_speech_context(self) -> None:
        from polymarket.strategy_mention import extract_trump_speech_context

        question = "Will Trump mention Biden in his speech?"
        assert extract_trump_speech_context(question) == "speech"


class TestIsTrumpMentionMarket:
    """Tests for is_trump_mention_market function."""

    def test_detects_trump_mention_market(self) -> None:
        from polymarket.strategy_mention import is_trump_mention_market

        assert is_trump_mention_market("Will Trump mention Biden in his speech?")
        assert is_trump_mention_market("Will Donald Trump mention tariffs?")

    def test_rejects_non_trump_market(self) -> None:
        from polymarket.strategy_mention import is_trump_mention_market

        assert not is_trump_mention_market("Will Biden mention Trump in his speech?")

    def test_rejects_non_mention_market(self) -> None:
        from polymarket.strategy_mention import is_trump_mention_market

        assert not is_trump_mention_market("Will Trump win the election?")


class TestTrumpWordFrequencySignals:
    """Tests for generate_trump_word_frequency_signals function."""

    def test_generates_trump_specific_signals(self) -> None:
        from polymarket.strategy_mention import (
            MentionMarket,
            generate_trump_word_frequency_signals,
        )

        markets = [
            MentionMarket(
                market_id="test1",
                token_id_yes="yes1",
                token_id_no="no1",
                question="Will Trump mention Biden in his speech?",
                mention_target="Biden",
                mention_context="speech",
                current_yes_price=0.80,  # Market overpricing
                current_no_price=0.20,
            ),
            MentionMarket(
                market_id="test2",
                token_id_yes="yes2",
                token_id_no="no2",
                question="Will Trump mention tariffs at his rally?",
                mention_target="Tariffs",
                mention_context="speech",
                current_yes_price=0.10,  # Market underpricing
                current_no_price=0.90,
            ),
        ]

        signals = generate_trump_word_frequency_signals(markets)

        # Should generate signals for both Trump markets
        assert len(signals) == 2

        # Check that reasoning includes word-frequency info
        for signal in signals:
            assert "Trump word-freq" in signal.reasoning or "word-freq" in signal.reasoning

    def test_skips_non_trump_markets(self) -> None:
        from polymarket.strategy_mention import (
            MentionMarket,
            generate_trump_word_frequency_signals,
        )

        markets = [
            MentionMarket(
                market_id="test1",
                token_id_yes="yes1",
                token_id_no="no1",
                question="Will Biden mention Trump in his speech?",  # Biden speaking
                mention_target="Trump",
                mention_context="speech",
                current_yes_price=0.50,
                current_no_price=0.50,
            ),
        ]

        signals = generate_trump_word_frequency_signals(markets)
        assert len(signals) == 0  # No Trump markets = no signals

    def test_trump_signals_sorted_by_ev(self) -> None:
        from polymarket.strategy_mention import (
            MentionMarket,
            generate_trump_word_frequency_signals,
        )

        markets = [
            MentionMarket(
                market_id="test1",
                token_id_yes="yes1",
                token_id_no="no1",
                question="Will Trump mention Biden?",
                mention_target="Biden",
                current_yes_price=0.20,
                current_no_price=0.80,
            ),
            MentionMarket(
                market_id="test2",
                token_id_yes="yes2",
                token_id_no="no2",
                question="Will Trump mention xyzunknown?",
                mention_target="Xyzunknown",
                current_yes_price=0.90,  # Very overpriced
                current_no_price=0.10,
            ),
        ]

        signals = generate_trump_word_frequency_signals(markets)

        # Should be sorted by expected value descending
        if len(signals) >= 2:
            evs = [s.expected_value for s in signals]
            assert evs == sorted(evs, reverse=True)
