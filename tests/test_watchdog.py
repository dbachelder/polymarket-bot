"""Tests for collector watchdog."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from polymarket.watchdog import (
    DEFAULT_MAX_AGE_SECONDS,
    CollectorWatchdog,
    run_watchdog,
)


class TestCollectorWatchdogInit:
    """Test watchdog initialization."""

    def test_default_paths(self):
        """Test default path configuration."""
        wd = CollectorWatchdog()
        assert wd.data_dir == Path("data")
        assert wd.latest_path == Path("data/latest_15m.json")
        assert wd.pidfile_path == Path("data/collect-15m-loop.pid")
        assert wd.log_path == Path("data/collector_watchdog.log")
        assert wd.max_age_seconds == DEFAULT_MAX_AGE_SECONDS

    def test_custom_paths(self, tmp_path):
        """Test custom path configuration."""
        wd = CollectorWatchdog(
            data_dir=tmp_path,
            latest_file="custom_latest.json",
            pidfile="custom.pid",
            log_file="custom.log",
            max_age_seconds=300.0,
        )
        assert wd.data_dir == tmp_path
        assert wd.latest_path == tmp_path / "custom_latest.json"
        assert wd.pidfile_path == tmp_path / "custom.pid"
        assert wd.log_path == tmp_path / "custom.log"
        assert wd.max_age_seconds == 300.0

    def test_script_path_detection(self, tmp_path):
        """Test script path auto-detection."""
        # Create a fake run.sh in tmp_path
        run_sh = tmp_path / "run.sh"
        run_sh.write_text("#!/bin/bash\necho test")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            wd = CollectorWatchdog(data_dir=tmp_path)
            assert wd.script_path == run_sh


class TestGetLatestFileAge:
    """Test latest file age checking."""

    def test_returns_none_when_file_missing(self, tmp_path):
        """Test returns None when latest file doesn't exist."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        assert wd.get_latest_file_age() is None

    def test_returns_age_when_file_exists(self, tmp_path):
        """Test returns correct age when file exists."""
        wd = CollectorWatchdog(data_dir=tmp_path)

        # Create latest file with recent mtime
        wd.latest_path.write_text(json.dumps({"test": "data"}))

        age = wd.get_latest_file_age()
        assert age is not None
        assert 0 <= age < 1  # Less than 1 second old

    def test_returns_correct_age_for_old_file(self, tmp_path):
        """Test returns correct age for older file."""
        wd = CollectorWatchdog(data_dir=tmp_path)

        # Create latest file
        wd.latest_path.write_text(json.dumps({"test": "data"}))

        # Set mtime to 60 seconds ago
        old_time = (datetime.now(UTC) - timedelta(seconds=60)).timestamp()
        os.utime(wd.latest_path, (old_time, old_time))

        age = wd.get_latest_file_age()
        assert age is not None
        assert 59 <= age <= 61  # Approximately 60 seconds


class TestIsFresh:
    """Test freshness checking."""

    def test_fresh_when_under_max_age(self, tmp_path):
        """Test file is fresh when under max age."""
        wd = CollectorWatchdog(data_dir=tmp_path, max_age_seconds=120.0)

        # Create recent file
        wd.latest_path.write_text(json.dumps({"test": "data"}))

        is_fresh, age, message = wd.is_fresh()
        assert is_fresh is True
        assert age is not None
        assert "age" in message.lower()

    def test_stale_when_over_max_age(self, tmp_path):
        """Test file is stale when over max age."""
        wd = CollectorWatchdog(data_dir=tmp_path, max_age_seconds=10.0)

        # Create old file (60 seconds ago)
        wd.latest_path.write_text(json.dumps({"test": "data"}))
        old_time = (datetime.now(UTC) - timedelta(seconds=60)).timestamp()
        os.utime(wd.latest_path, (old_time, old_time))

        is_fresh, age, message = wd.is_fresh()
        assert is_fresh is False
        assert age is not None
        assert age > 10.0

    def test_stale_when_file_missing(self, tmp_path):
        """Test file is stale when missing."""
        wd = CollectorWatchdog(data_dir=tmp_path)

        is_fresh, age, message = wd.is_fresh()
        assert is_fresh is False
        assert age is None
        assert "not found" in message.lower()


class TestGetCollectorPid:
    """Test PID file checking."""

    def test_returns_none_when_no_pidfile(self, tmp_path):
        """Test returns None when pidfile doesn't exist."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        assert wd.get_collector_pid() is None

    def test_returns_none_for_invalid_pid(self, tmp_path):
        """Test returns None for invalid pidfile content."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        wd.pidfile_path.write_text("not_a_number")
        assert wd.get_collector_pid() is None
        # Should clean up stale pidfile
        assert not wd.pidfile_path.exists()

    def test_returns_none_for_dead_process(self, tmp_path):
        """Test returns None for dead process."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        # Use a very high PID that won't exist
        wd.pidfile_path.write_text("999999")
        assert wd.get_collector_pid() is None
        # Should clean up stale pidfile
        assert not wd.pidfile_path.exists()

    def test_returns_pid_for_running_process(self, tmp_path):
        """Test returns PID for running process."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        # Use our own PID (always exists)
        our_pid = os.getpid()
        wd.pidfile_path.write_text(str(our_pid))
        assert wd.get_collector_pid() == our_pid


class TestIsCollectorRunning:
    """Test collector running check."""

    def test_not_running_when_no_pidfile(self, tmp_path):
        """Test reports not running when no pidfile."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        is_running, pid = wd.is_collector_running()
        assert is_running is False
        assert pid is None

    def test_running_when_valid_pidfile(self, tmp_path):
        """Test reports running when valid pidfile."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        our_pid = os.getpid()
        wd.pidfile_path.write_text(str(our_pid))
        is_running, pid = wd.is_collector_running()
        assert is_running is True
        assert pid == our_pid


class TestStartCollector:
    """Test collector starting."""

    def test_returns_existing_if_already_running(self, tmp_path):
        """Test returns existing PID if collector already running."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        our_pid = os.getpid()
        wd.pidfile_path.write_text(str(our_pid))

        success, pid, message = wd.start_collector()
        assert success is True
        assert pid == our_pid
        assert "already running" in message.lower()

    @patch("subprocess.Popen")
    def test_starts_new_collector(self, mock_popen, tmp_path):
        """Test starts new collector process."""
        # Create fake run.sh
        run_sh = tmp_path / "run.sh"
        run_sh.write_text("#!/bin/bash\necho test")

        wd = CollectorWatchdog(data_dir=tmp_path, script_path=run_sh)

        # Mock the process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        success, pid, message = wd.start_collector()
        assert success is True
        assert pid == 12345
        assert "started" in message.lower()

        # Verify pidfile was written
        assert wd.pidfile_path.exists()
        assert wd.pidfile_path.read_text() == "12345"

    @patch("subprocess.Popen", side_effect=OSError("Failed to start"))
    def test_handles_start_failure(self, mock_popen, tmp_path):
        """Test handles start failure gracefully."""
        run_sh = tmp_path / "run.sh"
        run_sh.write_text("#!/bin/bash\necho test")

        wd = CollectorWatchdog(data_dir=tmp_path, script_path=run_sh)
        success, pid, message = wd.start_collector()
        assert success is False
        assert pid is None
        assert "failed" in message.lower()


class TestStopCollector:
    """Test collector stopping."""

    def test_returns_ok_when_not_running(self, tmp_path):
        """Test returns OK when collector not running."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        success, message = wd.stop_collector()
        assert success is True
        assert "not running" in message.lower()

    def test_stops_running_collector(self, tmp_path):
        """Test stops running collector."""
        wd = CollectorWatchdog(data_dir=tmp_path)
        # We can't actually kill ourselves, so use a mock
        with patch("polymarket.watchdog.os.kill") as mock_kill:
            our_pid = os.getpid()
            wd.pidfile_path.write_text(str(our_pid))
            success, message = wd.stop_collector()
            assert success is True
            # Called twice: once for get_collector_pid (signal 0) and once for stop (signal 15)
            assert mock_kill.call_count == 2
            mock_kill.assert_any_call(our_pid, 0)  # Check process exists
            mock_kill.assert_any_call(our_pid, 15)  # SIGTERM


class TestCheckAndRestart:
    """Test check and restart functionality."""

    def test_no_action_when_fresh(self, tmp_path):
        """Test no action when data is fresh."""
        wd = CollectorWatchdog(data_dir=tmp_path, max_age_seconds=120.0)

        # Create recent file
        wd.latest_path.write_text(json.dumps({"test": "data"}))

        result = wd.check_and_restart(dry_run=False)
        assert result["fresh"] is True
        assert result["action_taken"] == "none"
        assert result["age_seconds"] is not None

    def test_restarts_when_stale_and_not_running(self, tmp_path):
        """Test restarts when data is stale and collector not running."""
        run_sh = tmp_path / "run.sh"
        run_sh.write_text("#!/bin/bash\necho test")

        wd = CollectorWatchdog(data_dir=tmp_path, max_age_seconds=10.0, script_path=run_sh)

        # Create old file
        wd.latest_path.write_text(json.dumps({"test": "data"}))
        old_time = (datetime.now(UTC) - timedelta(seconds=60)).timestamp()
        os.utime(wd.latest_path, (old_time, old_time))

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            result = wd.check_and_restart(dry_run=False)
            assert result["fresh"] is False
            assert result["action_taken"] == "restarted"
            assert result["collector_pid"] == 12345

    def test_dry_run_does_not_restart(self, tmp_path):
        """Test dry run doesn't actually restart."""
        wd = CollectorWatchdog(data_dir=tmp_path, max_age_seconds=10.0)

        # Create old file
        wd.latest_path.write_text(json.dumps({"test": "data"}))
        old_time = (datetime.now(UTC) - timedelta(seconds=60)).timestamp()
        os.utime(wd.latest_path, (old_time, old_time))

        result = wd.check_and_restart(dry_run=True)
        assert result["fresh"] is False
        assert result["action_taken"] == "would_restart"
        # Pidfile should not exist
        assert not wd.pidfile_path.exists()

    def test_logs_warning_when_running_but_stale(self, tmp_path):
        """Test logs warning when collector running but data stale."""
        wd = CollectorWatchdog(data_dir=tmp_path, max_age_seconds=10.0)

        # Create old file
        wd.latest_path.write_text(json.dumps({"test": "data"}))
        old_time = (datetime.now(UTC) - timedelta(seconds=60)).timestamp()
        os.utime(wd.latest_path, (old_time, old_time))

        # Simulate running collector
        wd.pidfile_path.write_text(str(os.getpid()))

        result = wd.check_and_restart(dry_run=False)
        assert result["fresh"] is False
        assert result["collector_running"] is True
        assert result["action_taken"] == "logged_warning"


class TestRunWatchdog:
    """Test the run_watchdog convenience function."""

    def test_returns_result_dict(self, tmp_path):
        """Test returns properly formatted result dict."""
        # Create fresh file
        latest = tmp_path / "latest_15m.json"
        latest.write_text(json.dumps({"test": "data"}))

        result = run_watchdog(data_dir=tmp_path, max_age_seconds=120.0)
        assert "timestamp" in result
        assert "fresh" in result
        assert "age_seconds" in result
        assert "collector_running" in result
        assert "action_taken" in result
        assert "message" in result
