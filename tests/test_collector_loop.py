"""Tests for collector loop functions."""

from __future__ import annotations

import time
from pathlib import Path


from polymarket.collector_loop import (
    _prune_by_count,
    _prune_old_files,
    check_staleness_sla,
    get_latest_snapshot_age_seconds,
)


class TestPruneOldFiles:
    def test_prunes_files_older_than_retention(self, tmp_path: Path):
        """Files older than retention_hours should be deleted."""
        # Create test files with old and new mtimes
        old_file = tmp_path / "snapshot_15m_20260101T000000Z.json"
        new_file = tmp_path / "snapshot_15m_20260214T220000Z.json"

        old_file.write_text("{}")
        new_file.write_text("{}")

        # Set old file mtime to 48 hours ago
        old_mtime = time.time() - (48 * 3600)
        old_file.touch()
        import os

        os.utime(old_file, (old_mtime, old_mtime))

        # Prune with 24 hour retention
        deleted = _prune_old_files(tmp_path, "snapshot_15m", 24.0)

        assert deleted == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_no_files_to_prune(self, tmp_path: Path):
        """When all files are fresh, nothing should be deleted."""
        new_file = tmp_path / "snapshot_15m_20260214T220000Z.json"
        new_file.write_text("{}")

        deleted = _prune_old_files(tmp_path, "snapshot_15m", 24.0)

        assert deleted == 0
        assert new_file.exists()

    def test_different_prefix_not_pruned(self, tmp_path: Path):
        """Files with different prefix should not be affected."""
        file_5m = tmp_path / "snapshot_5m_20260101T000000Z.json"
        file_5m.write_text("{}")

        # Set old mtime
        old_mtime = time.time() - (48 * 3600)
        import os

        os.utime(file_5m, (old_mtime, old_mtime))

        # Prune 15m snapshots
        deleted = _prune_old_files(tmp_path, "snapshot_15m", 24.0)

        assert deleted == 0
        assert file_5m.exists()


class TestPruneByCount:
    def test_prunes_excess_files(self, tmp_path: Path):
        """Oldest files should be pruned when count exceeds max."""
        # Create 5 snapshot files
        for i in range(5):
            f = tmp_path / f"snapshot_15m_20260214T22000{i}Z.json"
            f.write_text("{}")
            time.sleep(0.01)  # Ensure different mtimes

        # Prune to keep only 3
        deleted = _prune_by_count(tmp_path, "snapshot_15m", 3)

        assert deleted == 2
        remaining = list(tmp_path.glob("snapshot_15m_*.json"))
        assert len(remaining) == 3

    def test_no_pruning_when_under_limit(self, tmp_path: Path):
        """Nothing pruned when file count is under limit."""
        for i in range(3):
            f = tmp_path / f"snapshot_15m_20260214T22000{i}Z.json"
            f.write_text("{}")

        deleted = _prune_by_count(tmp_path, "snapshot_15m", 5)

        assert deleted == 0
        remaining = list(tmp_path.glob("snapshot_15m_*.json"))
        assert len(remaining) == 3

    def test_max_snapshots_zero_disables_pruning(self, tmp_path: Path):
        """When max_snapshots is 0, no pruning occurs."""
        for i in range(3):
            f = tmp_path / f"snapshot_15m_20260214T22000{i}Z.json"
            f.write_text("{}")

        deleted = _prune_by_count(tmp_path, "snapshot_15m", 0)

        assert deleted == 0
        remaining = list(tmp_path.glob("snapshot_15m_*.json"))
        assert len(remaining) == 3


class TestGetLatestSnapshotAge:
    def test_returns_age_of_latest_snapshot(self, tmp_path: Path):
        """Should return age of most recent snapshot."""
        # Create a file with current time
        f = tmp_path / "snapshot_15m_20260214T220000Z.json"
        f.write_text("{}")

        age = get_latest_snapshot_age_seconds(tmp_path, "snapshot_15m")

        assert age is not None
        assert age >= 0
        assert age < 1  # Should be very fresh

    def test_returns_none_when_no_snapshots(self, tmp_path: Path):
        """Should return None when no snapshots exist."""
        age = get_latest_snapshot_age_seconds(tmp_path, "snapshot_15m")

        assert age is None

    def test_different_prefix_not_considered(self, tmp_path: Path):
        """Should only consider files with matching prefix."""
        f = tmp_path / "snapshot_5m_20260214T220000Z.json"
        f.write_text("{}")

        age = get_latest_snapshot_age_seconds(tmp_path, "snapshot_15m")

        assert age is None


class TestCheckStalenessSla:
    def test_healthy_when_under_sla(self, tmp_path: Path):
        """Should report healthy when snapshot is fresh."""
        f = tmp_path / "snapshot_15m_20260214T220000Z.json"
        f.write_text("{}")

        result = check_staleness_sla(tmp_path, max_age_seconds=120.0, prefix="snapshot_15m")

        assert result["healthy"] is True
        assert result["age_seconds"] is not None
        assert result["age_seconds"] < 120.0
        assert "OK" in result["message"]

    def test_unhealthy_when_over_sla(self, tmp_path: Path):
        """Should report unhealthy when snapshot is stale."""
        f = tmp_path / "snapshot_15m_20260214T220000Z.json"
        f.write_text("{}")

        # Set mtime to 3 minutes ago
        old_mtime = time.time() - 180
        import os

        os.utime(f, (old_mtime, old_mtime))

        result = check_staleness_sla(tmp_path, max_age_seconds=120.0, prefix="snapshot_15m")

        assert result["healthy"] is False
        assert result["age_seconds"] is not None
        assert result["age_seconds"] > 120.0
        assert "EXCEEDS" in result["message"]

    def test_unhealthy_when_no_snapshots(self, tmp_path: Path):
        """Should report unhealthy when no snapshots exist."""
        result = check_staleness_sla(tmp_path, max_age_seconds=120.0, prefix="snapshot_15m")

        assert result["healthy"] is False
        assert result["age_seconds"] is None
        assert "No snapshots found" in result["message"]
