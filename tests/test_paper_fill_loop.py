"""Tests for paper_fill_loop module."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch


from polymarket.paper_fill_loop import (
    DAILY_METRIC_HOUR,
    DEFAULT_CHEAP_PRICE,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_SIZE,
    DEFAULT_WINDOW_SECONDS,
    MAX_RELAXATION_STEPS,
    MIN_CHEAP_PRICE,
    MIN_WINDOW_SECONDS,
    RELAXATION_FACTOR,
    WINDOW_EXTENSION_FACTOR,
    calculate_relaxed_thresholds,
    count_fills_last_24h,
    emit_daily_metric,
    get_daily_metric,
    run_paper_fill_iteration,
    run_paper_fill_testbed_loop,
)


class TestCountFillsLast24h:
    """Tests for count_fills_last_24h function."""

    def test_no_file_returns_zero(self):
        """Test returns 0 when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "nonexistent.jsonl"
            assert count_fills_last_24h(fills_path) == 0

    def test_empty_file_returns_zero(self):
        """Test returns 0 for empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text("")
            assert count_fills_last_24h(fills_path) == 0

    def test_counts_fills_within_24h(self):
        """Test counts only fills within last 24 hours."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills = [
                {"timestamp": (now - timedelta(hours=2)).isoformat(), "token_id": "1"},
                {"timestamp": (now - timedelta(hours=5)).isoformat(), "token_id": "2"},
                {"timestamp": (now - timedelta(hours=23)).isoformat(), "token_id": "3"},
                {"timestamp": (now - timedelta(hours=25)).isoformat(), "token_id": "4"},
                {"timestamp": (now - timedelta(days=2)).isoformat(), "token_id": "5"},
            ]
            fills_path.write_text("\n".join(json.dumps(f) for f in fills))

            count = count_fills_last_24h(fills_path)

            assert count == 3  # Only first 3 are within 24h

    def test_uses_created_at_fallback(self):
        """Test uses created_at when timestamp not present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills = [
                {"created_at": (now - timedelta(hours=2)).isoformat(), "token_id": "1"},
                {"timestamp": (now - timedelta(hours=5)).isoformat(), "token_id": "2"},
            ]
            fills_path.write_text("\n".join(json.dumps(f) for f in fills))

            count = count_fills_last_24h(fills_path)

            assert count == 2

    def test_skips_invalid_json(self):
        """Test skips lines with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills_path.write_text(
                "not json\n"
                + json.dumps({"timestamp": (now - timedelta(hours=2)).isoformat()})
                + "\ninvalid again\n"
            )

            count = count_fills_last_24h(fills_path)

            assert count == 1

    def test_skips_invalid_timestamp(self):
        """Test skips lines with invalid timestamp format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills_path.write_text(
                json.dumps({"timestamp": "invalid-date"})
                + "\n"
                + json.dumps({"timestamp": (now - timedelta(hours=2)).isoformat()})
                + "\n"
            )

            count = count_fills_last_24h(fills_path)

            assert count == 1

    def test_skips_lines_without_timestamp(self):
        """Test skips lines without timestamp or created_at."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills_path.write_text(
                json.dumps({"token_id": "no-timestamp"})
                + "\n"
                + json.dumps({"timestamp": (now - timedelta(hours=2)).isoformat()})
                + "\n"
            )

            count = count_fills_last_24h(fills_path)

            assert count == 1

    def test_handles_z_timezone(self):
        """Test handles Z timezone format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills_path.write_text(
                json.dumps({"timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ")}) + "\n"
            )

            count = count_fills_last_24h(fills_path)

            assert count == 1

    def test_blank_lines_ignored(self):
        """Test blank lines are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills_path.write_text(
                "\n\n" + json.dumps({"timestamp": (now - timedelta(hours=2)).isoformat()}) + "\n\n"
            )

            count = count_fills_last_24h(fills_path)

            assert count == 1


class TestGetDailyMetric:
    """Tests for get_daily_metric function."""

    @patch("polymarket.paper_fill_loop.get_fills_summary")
    def test_returns_metric_dict(self, mock_summary):
        """Test returns proper metric dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            # Create one fill within 24h
            fills_path.write_text(
                json.dumps({"timestamp": (now - timedelta(hours=2)).isoformat()}) + "\n"
            )

            mock_summary.return_value = {
                "total_fills": 10,
                "last_fill_at": (now - timedelta(hours=1)).isoformat(),
            }

            metric = get_daily_metric(fills_path)

            assert metric["fills_appended_last_24h"] == 1
            assert metric["total_fills"] == 10
            assert metric["alert"] is False
            assert metric["fills_path"] == str(fills_path)
            assert "timestamp" in metric

    @patch("polymarket.paper_fill_loop.get_fills_summary")
    def test_alert_when_zero_fills(self, mock_summary):
        """Test alert flag set when no fills in 24h."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            # Create only old fills
            fills_path.write_text(
                json.dumps({"timestamp": (now - timedelta(days=2)).isoformat()}) + "\n"
            )

            mock_summary.return_value = {"total_fills": 5, "last_fill_at": None}

            metric = get_daily_metric(fills_path)

            assert metric["fills_appended_last_24h"] == 0
            assert metric["alert"] is True


class TestEmitDailyMetric:
    """Tests for emit_daily_metric function."""

    @patch("polymarket.paper_fill_loop.get_fills_summary")
    @patch("polymarket.paper_fill_loop.logger")
    def test_logs_metric(self, mock_logger, mock_summary):
        """Test logs daily metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills_path.write_text(
                json.dumps({"timestamp": (now - timedelta(hours=2)).isoformat()}) + "\n"
            )

            mock_summary.return_value = {
                "total_fills": 10,
                "last_fill_at": (now - timedelta(hours=1)).isoformat(),
            }

            emit_daily_metric(fills_path)

            # Check that info was logged with DAILY_METRIC prefix
            info_calls = [c for c in mock_logger.info.call_args_list if "DAILY_METRIC" in str(c)]
            assert len(info_calls) > 0

    @patch("polymarket.paper_fill_loop.get_fills_summary")
    @patch("polymarket.paper_fill_loop.logger")
    def test_triggers_alert_when_no_fills(self, mock_logger, mock_summary):
        """Test triggers alert when no fills in 24h."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"

            mock_summary.return_value = {"total_fills": 0, "last_fill_at": None}

            emit_daily_metric(fills_path)

            # Check that warning was logged
            warning_calls = [c for c in mock_logger.warning.call_args_list if "ALERT" in str(c)]
            assert len(warning_calls) > 0

    @patch("polymarket.paper_fill_loop.get_fills_summary")
    def test_calls_alert_callback(self, mock_summary):
        """Test calls on_alert callback when alert triggered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            mock_alert_callback = MagicMock()

            mock_summary.return_value = {"total_fills": 0, "last_fill_at": None}

            emit_daily_metric(fills_path, on_alert=mock_alert_callback)

            mock_alert_callback.assert_called_once()
            assert "ALERT" in mock_alert_callback.call_args[0][0]

    @patch("polymarket.paper_fill_loop.get_fills_summary")
    def test_returns_metric(self, mock_summary):
        """Test returns metric dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            now = datetime.now(UTC)

            fills_path.write_text(
                json.dumps({"timestamp": (now - timedelta(hours=2)).isoformat()}) + "\n"
            )

            mock_summary.return_value = {"total_fills": 5, "last_fill_at": None}

            result = emit_daily_metric(fills_path)

            assert result["fills_appended_last_24h"] == 1
            assert result["total_fills"] == 5


class TestCalculateRelaxedThresholds:
    """Tests for calculate_relaxed_thresholds function."""

    def test_no_relaxation_at_step_zero(self):
        """Test no change at step 0."""
        price, window = calculate_relaxed_thresholds(Decimal("0.15"), 1800, 0)

        assert price == Decimal("0.15")
        assert window == 1800

    def test_relaxation_reduces_price(self):
        """Test relaxation reduces price threshold."""
        base_price = Decimal("0.15")
        price, window = calculate_relaxed_thresholds(base_price, 1800, 1)

        # Price should be reduced by 15%
        expected_price = base_price * RELAXATION_FACTOR
        assert price == expected_price

    def test_relaxation_increases_window(self):
        """Test relaxation increases window."""
        base_window = 1800
        price, window = calculate_relaxed_thresholds(Decimal("0.15"), base_window, 1)

        # Window should be increased by 20%
        expected_window = int(base_window * WINDOW_EXTENSION_FACTOR)
        assert window == expected_window

    def test_multiple_relaxation_steps(self):
        """Test cumulative effect of multiple steps."""
        base_price = Decimal("0.10")
        base_window = 1000

        for step in range(1, 4):
            price, window = calculate_relaxed_thresholds(base_price, base_window, step)

            expected_price = max(
                base_price * (RELAXATION_FACTOR**step),
                MIN_CHEAP_PRICE,
            )
            expected_window = min(
                int(base_window * (WINDOW_EXTENSION_FACTOR**step)),
                3600,
            )

            assert price == expected_price
            assert window == expected_window

    def test_price_bounded_at_minimum(self):
        """Test price doesn't go below MIN_CHEAP_PRICE."""
        price, window = calculate_relaxed_thresholds(
            Decimal("0.03"),
            1800,
            10,  # High step count
        )

        assert price >= MIN_CHEAP_PRICE
        assert price == MIN_CHEAP_PRICE  # Should hit floor

    def test_window_bounded_at_maximum(self):
        """Test window doesn't exceed 3600 seconds."""
        price, window = calculate_relaxed_thresholds(
            Decimal("0.15"),
            3000,
            10,  # High step count
        )

        assert window <= 3600
        assert window == 3600  # Should hit ceiling

    def test_negative_step_treated_as_zero(self):
        """Test negative step treated as no relaxation."""
        price, window = calculate_relaxed_thresholds(Decimal("0.15"), 1800, -1)

        assert price == Decimal("0.15")
        assert window == 1800


class TestRunPaperFillIteration:
    """Tests for run_paper_fill_iteration function."""

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    def test_calls_btc_preclose_paper(self, mock_run):
        """Test calls run_btc_preclose_paper with correct args."""
        mock_run.return_value = {
            "fills_recorded": 1,
            "markets_scanned": 5,
            "candidates_near_close": 2,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            snapshots_dir = Path(tmpdir) / "snapshots"

            run_paper_fill_iteration(
                data_dir=data_dir,
                snapshots_dir=snapshots_dir,
                cheap_price=Decimal("0.15"),
                window_seconds=1800,
                size=Decimal("1"),
                relaxation_step=0,
            )

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["data_dir"] == data_dir
            assert call_kwargs["window_seconds"] == 1800
            assert call_kwargs["cheap_price"] == Decimal("0.15")
            assert call_kwargs["size"] == Decimal("1")
            assert call_kwargs["snapshots_dir"] == snapshots_dir
            assert call_kwargs["verbose_tick"] is False

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    def test_applies_relaxation(self, mock_run):
        """Test applies relaxation to thresholds."""
        mock_run.return_value = {"fills_recorded": 0}

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            snapshots_dir = Path(tmpdir) / "snapshots"

            run_paper_fill_iteration(
                data_dir=data_dir,
                snapshots_dir=snapshots_dir,
                cheap_price=Decimal("0.15"),
                window_seconds=1800,
                size=Decimal("1"),
                relaxation_step=2,
            )

            call_kwargs = mock_run.call_args[1]
            # At step 2, price should be reduced
            expected_price = Decimal("0.15") * (RELAXATION_FACTOR**2)
            assert call_kwargs["cheap_price"] == expected_price

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    def test_returns_result_with_relaxation_info(self, mock_run):
        """Test result includes relaxation info."""
        mock_run.return_value = {
            "fills_recorded": 2,
            "markets_scanned": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            snapshots_dir = Path(tmpdir) / "snapshots"

            result = run_paper_fill_iteration(
                data_dir=data_dir,
                snapshots_dir=snapshots_dir,
                cheap_price=Decimal("0.15"),
                window_seconds=1800,
                size=Decimal("1"),
                relaxation_step=1,
            )

            assert result["relaxation_step"] == 1
            assert "thresholds_used" in result
            assert "cheap_price" in result["thresholds_used"]
            assert "window_seconds" in result["thresholds_used"]


class TestRunPaperFillTestbedLoop:
    """Tests for run_paper_fill_testbed_loop function."""

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_loop_runs_iterations(self, mock_time, mock_sleep, mock_run):
        """Test loop runs multiple iterations."""
        mock_time.side_effect = [0, 1, 2, 3, 4, 5]  # Simulate time passing
        mock_sleep.side_effect = [None, None, KeyboardInterrupt()]  # Stop after 2 iterations
        mock_run.return_value = {"fills_recorded": 0}

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            paper_dir = Path(tmpdir) / "paper"
            snapshots_dir = Path(tmpdir) / "snapshots"

            run_paper_fill_testbed_loop(
                data_dir=data_dir,
                paper_dir=paper_dir,
                snapshots_dir=snapshots_dir,
                interval_seconds=60,
            )

            assert mock_run.call_count == 2

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_resets_relaxation_on_success(self, mock_time, mock_sleep, mock_run):
        """Test relaxation resets when fills are recorded."""
        mock_time.side_effect = [0, 1, 2, 3, KeyboardInterrupt()]
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        # First no fills, then fills
        mock_run.side_effect = [
            {"fills_recorded": 0, "markets_scanned": 5, "candidates_near_close": 1},
            {"fills_recorded": 1, "markets_scanned": 5, "candidates_near_close": 2},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "paper"
            snapshots_dir = Path(tmpdir) / "snapshots"

            run_paper_fill_testbed_loop(
                paper_dir=paper_dir,
                snapshots_dir=snapshots_dir,
                interval_seconds=60,
            )

            # First call: relaxation_step=0, Second call: relaxation_step=0 (reset after success)
            calls = mock_run.call_args_list
            assert calls[0][1].get("relaxation_step", 0) == 0

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_increases_relaxation_after_no_fills(self, mock_time, mock_sleep, mock_run):
        """Test relaxation increases after consecutive no-fills."""
        mock_time.side_effect = list(range(20))  # Plenty of time values
        mock_sleep.side_effect = [None, None, None, KeyboardInterrupt()]

        # 3 iterations with no fills
        mock_run.side_effect = [
            {"fills_recorded": 0, "markets_scanned": 5, "candidates_near_close": 0},
            {"fills_recorded": 0, "markets_scanned": 5, "candidates_near_close": 0},
            {"fills_recorded": 0, "markets_scanned": 5, "candidates_near_close": 0},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "paper"
            snapshots_dir = Path(tmpdir) / "snapshots"

            run_paper_fill_testbed_loop(
                paper_dir=paper_dir,
                snapshots_dir=snapshots_dir,
                interval_seconds=60,
            )

            calls = mock_run.call_args_list
            # After 3 consecutive no-fills, relaxation should have increased
            # First call: step 0, Second: step 0 (only 1 no-fill), Third: step 1 (3 no-fills trigger)
            assert len(calls) == 3

    @patch("polymarket.paper_fill_loop.emit_daily_metric")
    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_emits_daily_metric(self, mock_time, mock_sleep, mock_run, mock_emit):
        """Test daily metric is emitted."""
        mock_time.side_effect = [0, 1, KeyboardInterrupt()]
        mock_sleep.side_effect = KeyboardInterrupt()
        mock_run.return_value = {"fills_recorded": 0}

        # Simulate midnight UTC (hour 0)
        mock_now = MagicMock()
        mock_now.date.return_value = datetime(2024, 1, 1).date()
        mock_now.hour = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "paper"
            snapshots_dir = Path(tmpdir) / "snapshots"

            with patch("polymarket.paper_fill_loop.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_now
                mock_datetime.UTC = UTC

                run_paper_fill_testbed_loop(
                    paper_dir=paper_dir,
                    snapshots_dir=snapshots_dir,
                    interval_seconds=60,
                )

                mock_emit.assert_called_once()

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_creates_directories(self, mock_time, mock_sleep, mock_run):
        """Test creates paper directory if needed."""
        mock_time.side_effect = [0, 1, KeyboardInterrupt()]
        mock_sleep.side_effect = KeyboardInterrupt()
        mock_run.return_value = {"fills_recorded": 0}

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "nested" / "paper"
            snapshots_dir = Path(tmpdir) / "snapshots"

            assert not paper_dir.exists()

            run_paper_fill_testbed_loop(
                paper_dir=paper_dir,
                snapshots_dir=snapshots_dir,
                interval_seconds=60,
            )

            assert paper_dir.exists()

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_handles_iteration_exception(self, mock_time, mock_sleep, mock_run):
        """Test continues loop after iteration exception."""
        mock_time.side_effect = list(range(10))
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        # First call raises exception, second succeeds
        mock_run.side_effect = [Exception("Test error"), {"fills_recorded": 0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "paper"
            snapshots_dir = Path(tmpdir) / "snapshots"

            # Should not raise, should continue loop
            run_paper_fill_testbed_loop(
                paper_dir=paper_dir,
                snapshots_dir=snapshots_dir,
                interval_seconds=60,
            )

            assert mock_run.call_count == 2

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_uses_default_paths(self, mock_time, mock_sleep, mock_run):
        """Test uses default paths when not specified."""
        mock_time.side_effect = [0, 1, KeyboardInterrupt()]
        mock_sleep.side_effect = KeyboardInterrupt()
        mock_run.return_value = {"fills_recorded": 0}

        run_paper_fill_testbed_loop()

        # Should not raise and should call run_btc_preclose_paper
        assert mock_run.called

    @patch("polymarket.paper_fill_loop.run_btc_preclose_paper")
    @patch("polymarket.paper_fill_loop.time.sleep")
    @patch("polymarket.paper_fill_loop.time.time")
    def test_daily_metric_only_once_per_day(self, mock_time, mock_sleep, mock_run, mock_emit):
        """Test daily metric only emitted once per day."""
        mock_time.side_effect = list(range(20))
        mock_sleep.side_effect = [None, None, KeyboardInterrupt()]
        mock_run.return_value = {"fills_recorded": 0}

        # Same day for all iterations
        same_day = datetime(2024, 1, 1).date()

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "paper"
            snapshots_dir = Path(tmpdir) / "snapshots"

            call_count = [0]

            def mock_now(*args, **kwargs):
                mock = MagicMock()
                mock.date.return_value = same_day
                mock.hour = 0
                call_count[0] += 1
                # Only first call should trigger metric
                return mock

            with patch("polymarket.paper_fill_loop.datetime") as mock_datetime:
                mock_datetime.now = mock_now
                mock_datetime.UTC = UTC

                run_paper_fill_testbed_loop(
                    paper_dir=paper_dir,
                    snapshots_dir=snapshots_dir,
                    interval_seconds=60,
                )

                # emit_daily_metric should only be called once (first iteration)
                assert mock_emit.call_count == 1


class TestConstants:
    """Tests for module constants."""

    def test_default_values(self):
        """Test default constant values are as expected."""
        assert DEFAULT_INTERVAL_SECONDS == 60
        assert DEFAULT_WINDOW_SECONDS == 1800
        assert DEFAULT_CHEAP_PRICE == Decimal("0.15")
        assert DEFAULT_SIZE == Decimal("1")
        assert MIN_CHEAP_PRICE == Decimal("0.02")
        assert MIN_WINDOW_SECONDS == 300
        assert MAX_RELAXATION_STEPS == 5
        assert RELAXATION_FACTOR == Decimal("0.85")
        assert WINDOW_EXTENSION_FACTOR == 1.2
        assert DAILY_METRIC_HOUR == 0
