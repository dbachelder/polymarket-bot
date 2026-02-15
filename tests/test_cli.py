"""Tests for CLI utility functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polymarket.cli import resolve_snapshot_path


class TestResolveSnapshotPath:
    """Tests for resolve_snapshot_path function."""

    def test_returns_original_path_for_regular_snapshot(self, tmp_path: Path):
        """If file is a regular snapshot (not a pointer), return it as-is."""
        snapshot_path = tmp_path / "snapshot_15m_20250215T120000Z.json"
        snapshot_data = {
            "generated_at": "2025-02-15T12:00:00Z",
            "markets": [{"id": "market1"}],
        }
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = resolve_snapshot_path(snapshot_path)

        assert result == snapshot_path

    def test_dereferences_pointer_file(self, tmp_path: Path):
        """If file is a pointer, return the path it points to."""
        # Create the actual snapshot
        snapshot_path = tmp_path / "snapshot_15m_20250215T120000Z.json"
        snapshot_data = {"generated_at": "2025-02-15T12:00:00Z", "markets": []}
        snapshot_path.write_text(json.dumps(snapshot_data))

        # Create the pointer file
        pointer_path = tmp_path / "latest_15m.json"
        pointer_data = {
            "path": str(snapshot_path),
            "generated_at": "2025-02-15T12:00:00Z",
        }
        pointer_path.write_text(json.dumps(pointer_data))

        result = resolve_snapshot_path(pointer_path)

        assert result == snapshot_path

    def test_returns_original_if_pointer_target_missing(self, tmp_path: Path):
        """If pointer points to non-existent file, return original path."""
        pointer_path = tmp_path / "latest_15m.json"
        pointer_data = {
            "path": str(tmp_path / "nonexistent.json"),
            "generated_at": "2025-02-15T12:00:00Z",
        }
        pointer_path.write_text(json.dumps(pointer_data))

        result = resolve_snapshot_path(pointer_path)

        # Falls back to original path since target doesn't exist
        assert result == pointer_path

    def test_returns_original_for_nonexistent_file(self, tmp_path: Path):
        """If file doesn't exist, return the path as-is."""
        nonexistent_path = tmp_path / "does_not_exist.json"

        result = resolve_snapshot_path(nonexistent_path)

        assert result == nonexistent_path

    def test_returns_original_for_invalid_json(self, tmp_path: Path):
        """If file has invalid JSON, return original path."""
        bad_json_path = tmp_path / "bad.json"
        bad_json_path.write_text("not valid json")

        result = resolve_snapshot_path(bad_json_path)

        assert result == bad_json_path

    def test_returns_original_for_json_without_path_key(self, tmp_path: Path):
        """If JSON doesn't have 'path' key, return original path."""
        json_path = tmp_path / "other.json"
        json_path.write_text(json.dumps({"foo": "bar", "generated_at": "2025-02-15T12:00:00Z"}))

        result = resolve_snapshot_path(json_path)

        assert result == json_path

    def test_returns_original_for_non_dict_json(self, tmp_path: Path):
        """If JSON is not a dict (e.g., array), return original path."""
        json_path = tmp_path / "array.json"
        json_path.write_text(json.dumps(["item1", "item2"]))

        result = resolve_snapshot_path(json_path)

        assert result == json_path
