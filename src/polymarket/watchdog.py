"""Collector watchdog - ensures data freshness and auto-restarts collector."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data")
DEFAULT_LATEST_FILE = "latest_15m.json"
DEFAULT_PIDFILE = "collect-15m-loop.pid"
DEFAULT_LOG_FILE = "collector_watchdog.log"
DEFAULT_MAX_AGE_SECONDS = 120.0


class CollectorWatchdog:
    """Watchdog for the 15m collector loop.

    Ensures latest_15m.json is fresh and restarts collector if needed.
    Uses pidfile/lock to prevent duplicate collectors.
    """

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        latest_file: str = DEFAULT_LATEST_FILE,
        pidfile: str = DEFAULT_PIDFILE,
        log_file: str = DEFAULT_LOG_FILE,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        script_path: Path | str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.latest_path = self.data_dir / latest_file
        self.pidfile_path = self.data_dir / pidfile
        self.log_path = self.data_dir / log_file
        self.max_age_seconds = max_age_seconds
        self.script_path = Path(script_path) if script_path else self._find_script_path()

    def _find_script_path(self) -> Path:
        """Find the run.sh script path."""
        # Look in parent directories
        cwd = Path.cwd()
        for path in [cwd, cwd.parent, cwd.parent.parent]:
            script = path / "run.sh"
            if script.exists():
                return script
        # Default to cwd/run.sh
        return cwd / "run.sh"

    def _log(self, message: str, level: str = "info", extra: dict[str, Any] | None = None) -> None:
        """Log to watchdog log file."""
        timestamp = datetime.now(UTC).isoformat()
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
        }
        if extra:
            entry.update(extra)

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Append to log file
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning("Failed to write to watchdog log: %s", e)

        # Also log to standard logger
        log_func = getattr(logger, level, logger.info)
        log_func(message)

    def get_latest_file_age(self) -> float | None:
        """Get age of latest_15m.json in seconds.

        Returns:
            Age in seconds, or None if file doesn't exist.
        """
        if not self.latest_path.exists():
            return None

        try:
            mtime = datetime.fromtimestamp(self.latest_path.stat().st_mtime, tz=UTC)
            return (datetime.now(UTC) - mtime).total_seconds()
        except (OSError, ValueError) as e:
            self._log(f"Error checking latest file: {e}", level="error")
            return None

    def is_fresh(self) -> tuple[bool, float | None, str]:
        """Check if latest_15m.json is fresh.

        Returns:
            Tuple of (is_fresh, age_seconds, message)
        """
        age = self.get_latest_file_age()

        if age is None:
            return False, None, f"Latest file not found: {self.latest_path}"

        is_fresh = age <= self.max_age_seconds
        message = f"Latest file age: {age:.1f}s (max: {self.max_age_seconds:.1f}s)"

        return is_fresh, age, message

    def get_collector_pid(self) -> int | None:
        """Get PID from pidfile if collector is running.

        Returns:
            PID if running and valid, None otherwise.
        """
        if not self.pidfile_path.exists():
            return None

        try:
            pid = int(self.pidfile_path.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return pid
        except (ValueError, OSError, ProcessLookupError):
            # Stale pidfile
            try:
                self.pidfile_path.unlink()
            except OSError:
                pass
            return None

    def is_collector_running(self) -> tuple[bool, int | None]:
        """Check if collector is currently running.

        Returns:
            Tuple of (is_running, pid)
        """
        pid = self.get_collector_pid()
        return pid is not None, pid

    def start_collector(self) -> tuple[bool, int | None, str]:
        """Start the collector loop with pidfile.

        Returns:
            Tuple of (success, pid, message)
        """
        # Double-check lock before starting
        is_running, existing_pid = self.is_collector_running()
        if is_running:
            return True, existing_pid, f"Collector already running (PID {existing_pid})"

        try:
            # Start collector as subprocess
            # Use nohup to keep running after parent exits
            cmd = [
                "nohup",
                str(self.script_path),
                "collect-15m-loop",
                "--out",
                str(self.data_dir),
            ]

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

            pid = process.pid

            # Write pidfile
            self.pidfile_path.write_text(str(pid))

            self._log(
                "Collector started",
                level="info",
                extra={"pid": pid, "script": str(self.script_path)},
            )

            return True, pid, f"Collector started (PID {pid})"

        except Exception as e:
            error_msg = f"Failed to start collector: {e}"
            self._log(error_msg, level="error", extra={"error": str(e)})
            return False, None, error_msg

    def stop_collector(self) -> tuple[bool, str]:
        """Stop the collector if running.

        Returns:
            Tuple of (success, message)
        """
        pid = self.get_collector_pid()
        if pid is None:
            return True, "Collector not running"

        try:
            os.kill(pid, 15)  # SIGTERM
            self._log("Collector stopped", level="info", extra={"pid": pid})
            return True, f"Collector stopped (PID {pid})"
        except OSError as e:
            error_msg = f"Failed to stop collector (PID {pid}): {e}"
            self._log(error_msg, level="error", extra={"pid": pid, "error": str(e)})
            return False, error_msg

    def check_and_restart(self, dry_run: bool = False) -> dict[str, Any]:
        """Check freshness and restart collector if needed.

        Args:
            dry_run: If True, don't actually restart, just report what would happen.

        Returns:
            Dict with check results and actions taken.
        """
        result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "fresh": False,
            "age_seconds": None,
            "collector_running": False,
            "collector_pid": None,
            "action_taken": None,
            "message": "",
        }

        # Check freshness
        is_fresh, age, message = self.is_fresh()
        result["fresh"] = is_fresh
        result["age_seconds"] = age
        result["message"] = message

        self._log(
            f"Freshness check: {message}", level="info", extra={"fresh": is_fresh, "age": age}
        )

        if is_fresh:
            # Data is fresh, nothing to do
            result["action_taken"] = "none"
            result["message"] = f"Data is fresh ({age:.1f}s old)"
            return result

        # Data is stale or missing
        self._log(f"Data stale or missing: {message}", level="warning")

        # Check if collector is running
        is_running, pid = self.is_collector_running()
        result["collector_running"] = is_running
        result["collector_pid"] = pid

        if is_running:
            # Collector is running but data is stale - something is wrong
            result["action_taken"] = "logged_warning"
            result["message"] = f"Collector running (PID {pid}) but data is stale"
            self._log(
                "Collector running but data stale",
                level="warning",
                extra={"pid": pid, "age": age},
            )
            return result

        # Collector not running and data is stale - need to restart
        if dry_run:
            result["action_taken"] = "would_restart"
            result["message"] = "Would restart collector (dry run)"
            self._log("Would restart collector (dry run)", level="info")
        else:
            success, new_pid, msg = self.start_collector()
            if success:
                result["action_taken"] = "restarted"
                result["collector_pid"] = new_pid
                result["message"] = msg
            else:
                result["action_taken"] = "failed"
                result["message"] = msg

        return result


def run_watchdog(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
    dry_run: bool = False,
    script_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the watchdog check.

    Args:
        data_dir: Directory containing data files
        max_age_seconds: Maximum acceptable age in seconds
        dry_run: If True, don't actually restart collector
        script_path: Path to run.sh script

    Returns:
        Dict with check results
    """
    watchdog = CollectorWatchdog(
        data_dir=data_dir,
        max_age_seconds=max_age_seconds,
        script_path=script_path,
    )
    return watchdog.check_and_restart(dry_run=dry_run)
