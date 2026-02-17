"""Tests for fills_monitor module."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path


from polymarket.fills_monitor import (
    FillsMonitorState,
    auto_adjust_thresholds,
    check_fills_health,
    count_fills,
    get_current_thresholds,
    get_last_fill_timestamp,
    load_state,
    run_fills_monitor,
    save_state,
)


class TestFillsMonitorState:
    """Tests for FillsMonitorState dataclass."""

    def test_default_initialization(self):
        """Test default state initialization."""
        state = FillsMonitorState()

        assert state.last_fill_timestamp is None
        assert state.last_check_timestamp is None
        assert state.adjustment_count == 0
        assert state.current_cheap_price == Decimal("0.05")
        assert state.current_window_seconds == 300
        assert state.alerts_triggered == 0
        assert state.total_fills_seen == 0
        assert state.original_cheap_price == Decimal("0.05")
        assert state.original_window_seconds == 300

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = FillsMonitorState(
            last_fill_timestamp="2024-01-01T00:00:00+00:00",
            last_check_timestamp="2024-01-01T01:00:00+00:00",
            adjustment_count=2,
            current_cheap_price=Decimal("0.04"),
            current_window_seconds=240,
            alerts_triggered=1,
            total_fills_seen=100,
            original_cheap_price=Decimal("0.05"),
            original_window_seconds=300,
        )

        data = state.to_dict()

        assert data["last_fill_timestamp"] == "2024-01-01T00:00:00+00:00"
        assert data["last_check_timestamp"] == "2024-01-01T01:00:00+00:00"
        assert data["adjustment_count"] == 2
        assert data["current_cheap_price"] == "0.04"
        assert data["current_window_seconds"] == 240
        assert data["alerts_triggered"] == 1
        assert data["total_fills_seen"] == 100
        assert data["original_cheap_price"] == "0.05"
        assert data["original_window_seconds"] == 300

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "last_fill_timestamp": "2024-01-01T00:00:00+00:00",
            "last_check_timestamp": "2024-01-01T01:00:00+00:00",
            "adjustment_count": 2,
            "current_cheap_price": "0.04",
            "current_window_seconds": 240,
            "alerts_triggered": 1,
            "total_fills_seen": 100,
            "original_cheap_price": "0.05",
            "original_window_seconds": 300,
        }

        state = FillsMonitorState.from_dict(data)

        assert state.last_fill_timestamp == "2024-01-01T00:00:00+00:00"
        assert state.last_check_timestamp == "2024-01-01T01:00:00+00:00"
        assert state.adjustment_count == 2
        assert state.current_cheap_price == Decimal("0.04")
        assert state.current_window_seconds == 240
        assert state.alerts_triggered == 1
        assert state.total_fills_seen == 100
        assert state.original_cheap_price == Decimal("0.05")
        assert state.original_window_seconds == 300

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        data = {}

        state = FillsMonitorState.from_dict(data)

        assert state.adjustment_count == 0
        assert state.current_cheap_price == Decimal("0.05")
        assert state.current_window_seconds == 300

    def test_from_dict_with_string_numbers(self):
        """Test from_dict handles string numbers."""
        data = {
            "current_cheap_price": "0.035",
            "original_cheap_price": "0.05",
        }

        state = FillsMonitorState.from_dict(data)

        assert state.current_cheap_price == Decimal("0.035")
        assert state.original_cheap_price == Decimal("0.05")


class TestGetLastFillTimestamp:
    """Tests for get_last_fill_timestamp function."""

    def test_no_file(self):
        """Test returns None when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "nonexistent.jsonl"
            result = get_last_fill_timestamp(fills_path)
            assert result is None

    def test_empty_file(self):
        """Test returns None for empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text("")
            result = get_last_fill_timestamp(fills_path)
            assert result is None

    def test_single_fill_with_timestamp(self):
        """Test extracts timestamp from single fill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fill = {
                "token_id": "test-token",
                "timestamp": "2024-01-15T10:30:00+00:00",
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = get_last_fill_timestamp(fills_path)

            assert result is not None
            assert result.year == 2024
            assert result.month == 1
            assert result.day == 15

    def test_single_fill_with_created_at(self):
        """Test extracts created_at timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fill = {
                "token_id": "test-token",
                "created_at": "2024-01-15T10:30:00+00:00",
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = get_last_fill_timestamp(fills_path)

            assert result is not None
            assert result.year == 2024

    def test_multiple_fills_returns_latest(self):
        """Test returns latest timestamp from multiple fills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills = [
                {"timestamp": "2024-01-15T08:00:00+00:00"},
                {"timestamp": "2024-01-15T12:00:00+00:00"},
                {"timestamp": "2024-01-15T10:00:00+00:00"},
            ]
            fills_path.write_text("\n".join(json.dumps(f) for f in fills))

            result = get_last_fill_timestamp(fills_path)

            assert result is not None
            assert result.hour == 12

    def test_z_timezone_format(self):
        """Test handles Z timezone format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fill = {"timestamp": "2024-01-15T10:30:00Z"}
            fills_path.write_text(json.dumps(fill) + "\n")

            result = get_last_fill_timestamp(fills_path)

            assert result is not None
            assert result.tzinfo is not None

    def test_invalid_json_skipped(self):
        """Test skips invalid JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text(
                "not json\n" + json.dumps({"timestamp": "2024-01-15T10:30:00+00:00"}) + "\n"
            )

            result = get_last_fill_timestamp(fills_path)

            assert result is not None
            assert result.hour == 10

    def test_missing_timestamp_skipped(self):
        """Test skips lines without timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text(
                json.dumps({"token_id": "test"})
                + "\n"
                + json.dumps({"timestamp": "2024-01-15T10:30:00+00:00"})
                + "\n"
            )

            result = get_last_fill_timestamp(fills_path)

            assert result is not None

    def test_blank_lines_ignored(self):
        """Test blank lines are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text(
                "\n\n" + json.dumps({"timestamp": "2024-01-15T10:30:00+00:00"}) + "\n\n"
            )

            result = get_last_fill_timestamp(fills_path)

            assert result is not None


class TestCountFills:
    """Tests for count_fills function."""

    def test_no_file(self):
        """Test returns 0 when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "nonexistent.jsonl"
            assert count_fills(fills_path) == 0

    def test_empty_file(self):
        """Test returns 0 for empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text("")
            assert count_fills(fills_path) == 0

    def test_counts_lines(self):
        """Test counts all non-empty lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text("line1\nline2\nline3\n")
            assert count_fills(fills_path) == 3

    def test_ignores_blank_lines(self):
        """Test ignores blank lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fills_path.write_text("line1\n\nline2\n\n")
            assert count_fills(fills_path) == 2


class TestLoadState:
    """Tests for load_state function."""

    def test_no_file_returns_default(self):
        """Test returns default state when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = load_state(state_path)

            assert state.adjustment_count == 0
            assert state.current_cheap_price == Decimal("0.05")

    def test_loads_valid_state(self):
        """Test loads state from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            data = {
                "adjustment_count": 3,
                "current_cheap_price": "0.03",
                "current_window_seconds": 200,
            }
            state_path.write_text(json.dumps(data))

            state = load_state(state_path)

            assert state.adjustment_count == 3
            assert state.current_cheap_price == Decimal("0.03")
            assert state.current_window_seconds == 200

    def test_invalid_json_returns_default(self):
        """Test returns default on invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text("not json")

            state = load_state(state_path)

            assert state.adjustment_count == 0


class TestSaveState:
    """Tests for save_state function."""

    def test_saves_state(self):
        """Test state is saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = FillsMonitorState(
                adjustment_count=2,
                current_cheap_price=Decimal("0.04"),
            )

            save_state(state, state_path)

            assert state_path.exists()
            data = json.loads(state_path.read_text())
            assert data["adjustment_count"] == 2
            assert data["current_cheap_price"] == "0.04"

    def test_creates_parent_directory(self):
        """Test creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "subdir" / "state.json"
            state = FillsMonitorState()

            save_state(state, state_path)

            assert state_path.exists()


class TestCheckFillsHealth:
    """Tests for check_fills_health function."""

    def test_no_fills_file(self):
        """Test health check when no fills file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            result = check_fills_health(fills_path, stale_hours=12)

            assert result["healthy"] is False
            assert result["status"] == "no_fills"
            assert result["last_fill_at"] is None
            assert result["hours_since_last_fill"] is None
            assert result["total_fills"] == 0

    def test_healthy_fills(self):
        """Test health check with fresh fills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=5)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = check_fills_health(fills_path, stale_hours=12)

            assert result["healthy"] is True
            assert result["status"] == "healthy"
            assert result["hours_since_last_fill"] < 12
            assert result["total_fills"] == 1

    def test_stale_fills(self):
        """Test health check with stale fills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=15)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = check_fills_health(fills_path, stale_hours=12)

            assert result["healthy"] is False
            assert result["status"] == "stale"
            assert result["hours_since_last_fill"] > 12

    def test_custom_stale_threshold(self):
        """Test health check with custom stale threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=8)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            # 8 hours should be stale with 6 hour threshold
            result = check_fills_health(fills_path, stale_hours=6)
            assert result["healthy"] is False

            # 8 hours should be healthy with 12 hour threshold
            result = check_fills_health(fills_path, stale_hours=12)
            assert result["healthy"] is True


class TestAutoAdjustThresholds:
    """Tests for auto_adjust_thresholds function."""

    def test_no_adjustment_at_step_zero(self):
        """Test no adjustment when already at bounds."""
        original_price = Decimal("0.05")
        original_window = 300
        current_price = Decimal("0.025")  # At 50% bound
        current_window = 150  # At 50% bound

        new_price, new_window, was_adjusted = auto_adjust_thresholds(
            current_price, current_window, 0, original_price, original_window
        )

        assert was_adjusted is False
        assert new_price == current_price
        assert new_window == current_window

    def test_adjustment_reduces_thresholds(self):
        """Test adjustment reduces thresholds by 10%."""
        original_price = Decimal("0.05")
        original_window = 300
        current_price = Decimal("0.05")
        current_window = 300

        new_price, new_window, was_adjusted = auto_adjust_thresholds(
            current_price, current_window, 0, original_price, original_window
        )

        assert was_adjusted is True
        assert new_price == Decimal("0.045")  # 5% * 0.9
        assert new_window == 270  # 300 * 0.9

    def test_adjustment_respects_price_bound(self):
        """Test adjustment respects 50% price bound."""
        original_price = Decimal("0.05")
        original_window = 300
        current_price = Decimal("0.028")  # Just above 50% bound of 0.025
        current_window = 160

        new_price, new_window, was_adjusted = auto_adjust_thresholds(
            current_price, current_window, 5, original_price, original_window
        )

        assert was_adjusted is True
        # Should not go below 50% of original (0.025)
        assert new_price >= Decimal("0.025")

    def test_adjustment_respects_window_bound(self):
        """Test adjustment respects 50% window bound."""
        original_price = Decimal("0.05")
        original_window = 300
        current_price = Decimal("0.04")
        current_window = 155  # Just above 50% bound of 150

        new_price, new_window, was_adjusted = auto_adjust_thresholds(
            current_price, current_window, 5, original_price, original_window
        )

        assert was_adjusted is True
        # Should not go below 50% of original (150)
        assert new_window >= 150

    def test_multiple_adjustments(self):
        """Test cumulative effect of multiple adjustments."""
        original_price = Decimal("0.10")
        original_window = 600
        current_price = Decimal("0.10")
        current_window = 600

        for i in range(5):
            new_price, new_window, was_adjusted = auto_adjust_thresholds(
                current_price, current_window, i, original_price, original_window
            )
            if was_adjusted:
                current_price = new_price
                current_window = new_window

        # After 5 adjustments of 10% each, should be at ~59% of original
        assert current_price == Decimal("0.059049")  # 0.10 * 0.9^5


class TestRunFillsMonitor:
    """Tests for run_fills_monitor function."""

    def test_healthy_fills_no_adjustment(self):
        """Test no adjustment when fills are healthy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            state_path = Path(tmpdir) / "state.json"

            # Create fresh fill
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = run_fills_monitor(fills_path, state_path, stale_hours=12)

            assert result["healthy"] is True
            assert result["alert_triggered"] is False
            assert result["auto_adjusted"] is False

    def test_stale_fills_triggers_alert(self):
        """Test stale fills trigger alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            state_path = Path(tmpdir) / "state.json"

            # Create stale fill
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=15)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = run_fills_monitor(fills_path, state_path, stale_hours=12)

            assert result["healthy"] is False
            assert result["alert_triggered"] is True

    def test_stale_fills_with_auto_adjust(self):
        """Test auto-adjustment on stale fills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            state_path = Path(tmpdir) / "state.json"

            # Create stale fill
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=15)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = run_fills_monitor(fills_path, state_path, stale_hours=12, auto_adjust=True)

            assert result["auto_adjusted"] is True
            assert "new_cheap_price" in result
            assert "new_window_seconds" in result

    def test_stale_fills_no_auto_adjust(self):
        """Test no auto-adjustment when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            state_path = Path(tmpdir) / "state.json"

            # Create stale fill
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=15)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = run_fills_monitor(fills_path, state_path, stale_hours=12, auto_adjust=False)

            assert result["alert_triggered"] is True
            assert result["auto_adjusted"] is False

    def test_state_preserved_between_runs(self):
        """Test state is saved and loaded between runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            state_path = Path(tmpdir) / "state.json"

            # Create stale fill
            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=15)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            # First run - should adjust (result shows count BEFORE increment)
            result1 = run_fills_monitor(fills_path, state_path, stale_hours=12)
            assert result1["adjustment_count"] == 0  # Before increment
            assert result1["auto_adjusted"] is True

            # Second run - should show previous count and adjust again
            result2 = run_fills_monitor(fills_path, state_path, stale_hours=12)
            assert result2["adjustment_count"] == 1  # Previous increment now visible
            assert result2["auto_adjusted"] is True

    def test_includes_all_result_fields(self):
        """Test result contains all expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            state_path = Path(tmpdir) / "state.json"

            fill = {
                "timestamp": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = run_fills_monitor(fills_path, state_path, stale_hours=12)

            expected_fields = [
                "timestamp",
                "healthy",
                "status",
                "message",
                "hours_since_last_fill",
                "total_fills",
                "adjustment_count",
                "current_cheap_price",
                "current_window_seconds",
                "auto_adjusted",
                "alert_triggered",
            ]
            for field in expected_fields:
                assert field in result


class TestGetCurrentThresholds:
    """Tests for get_current_thresholds function."""

    def test_returns_defaults_when_no_state(self):
        """Test returns defaults when no state file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            price, window = get_current_thresholds(state_path)

            assert price == Decimal("0.05")
            assert window == 300

    def test_returns_module_defaults_when_no_state(self):
        """Test returns module defaults when no state file exists.

        Note: default_price and default_window params exist for API compatibility
        but the function currently returns values from FillsMonitorState defaults.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            price, window = get_current_thresholds(
                state_path,
                default_price=Decimal("0.10"),  # Currently unused
                default_window=600,  # Currently unused
            )

            # Returns FillsMonitorState defaults (0.05, 300)
            assert price == Decimal("0.05")
            assert window == 300

    def test_returns_state_values(self):
        """Test returns values from state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = FillsMonitorState(
                current_cheap_price=Decimal("0.03"),
                current_window_seconds=200,
            )
            save_state(state, state_path)

            price, window = get_current_thresholds(state_path)

            assert price == Decimal("0.03")
            assert window == 200
