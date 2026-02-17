"""Tests for fills collector module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from polymarket.fills_collector import (
    _mask_value,
    append_fills,
    check_api_auth,
    collect_fills,
    get_existing_tx_hashes,
    get_fills_summary,
    get_last_fill_timestamp,
    load_paper_fills,
    validate_credentials,
)
from polymarket.pnl import Fill


class TestMaskValue:
    """Test credential masking."""

    def test_mask_value_normal(self) -> None:
        """Mask middle of value, show last 4 chars."""
        assert _mask_value("abcdefghij", visible_chars=4) == "****ghij"

    def test_mask_value_short(self) -> None:
        """Handle short values."""
        assert _mask_value("abc", visible_chars=4) == "****abc"

    def test_mask_value_empty(self) -> None:
        """Handle None/empty."""
        assert _mask_value(None) == "<not set>"
        assert _mask_value("") == "<not set>"

    def test_mask_value_custom_visible(self) -> None:
        """Custom visible character count."""
        assert _mask_value("secret123", visible_chars=3) == "****123"


class TestValidateCredentials:
    """Test credential validation."""

    def test_validate_credentials_complete(self) -> None:
        """All credentials present."""
        config = MagicMock()
        config.api_key = "key123"
        config.api_secret = "secret456"
        config.api_passphrase = "passphrase789"
        config.has_credentials = True
        config.can_trade = True
        config.dry_run = False

        result = validate_credentials(config)

        assert result["has_credentials"] is True
        assert result["can_trade"] is True
        assert result["dry_run"] is False
        assert result["api_key"] == "****y123"
        assert result["api_secret"] == "****t456"
        assert result["api_passphrase"] == "****e789"
        assert result["api_key_length"] == 6
        assert result["api_secret_length"] == 9
        assert result["api_passphrase_length"] == 13
        assert result["warnings"] == []

    def test_validate_credentials_missing_all(self) -> None:
        """All credentials missing."""
        config = MagicMock()
        config.api_key = None
        config.api_secret = None
        config.api_passphrase = None
        config.has_credentials = False
        config.can_trade = False
        config.dry_run = True

        result = validate_credentials(config)

        assert result["has_credentials"] is False
        assert result["api_key"] == "<not set>"
        assert result["api_secret"] == "<not set>"
        assert result["api_passphrase"] == "<not set>"
        assert len(result["warnings"]) == 1
        assert "No API credentials configured" in result["warnings"][0]

    def test_validate_credentials_partial(self) -> None:
        """Some credentials missing."""
        config = MagicMock()
        config.api_key = "key123"
        config.api_secret = None
        config.api_passphrase = "passphrase"
        config.has_credentials = False
        config.can_trade = False
        config.dry_run = True

        result = validate_credentials(config)

        assert result["has_credentials"] is False
        assert len(result["warnings"]) == 1
        assert "POLYMARKET_API_SECRET is missing" in result["warnings"][0]


class TestCheckApiAuth:
    """Test API auth testing."""

    def test_check_api_auth_no_credentials(self) -> None:
        """Return early if no credentials."""
        config = MagicMock()
        config.has_credentials = False

        result = check_api_auth(config)

        assert result["success"] is False
        assert result["error"] == "Cannot test auth: no credentials configured"

    def test_check_api_auth_success(self) -> None:
        """Successful auth test."""
        config = MagicMock()
        config.has_credentials = True
        config.api_key = "key"
        config.api_passphrase = "pass"

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.Client.get", return_value=mock_response):
            result = check_api_auth(config)

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["endpoint"] == "/orders"

    def test_check_api_auth_401(self) -> None:
        """Auth failure (invalid credentials)."""
        config = MagicMock()
        config.has_credentials = True
        config.api_key = "key"
        config.api_passphrase = "pass"

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.Client.get", return_value=mock_response):
            result = check_api_auth(config)

        assert result["success"] is False
        assert result["status_code"] == 401
        assert "Invalid credentials" in result["error"]

    def test_check_api_auth_403(self) -> None:
        """Auth failure (insufficient permissions)."""
        config = MagicMock()
        config.has_credentials = True
        config.api_key = "key"
        config.api_passphrase = "pass"

        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("httpx.Client.get", return_value=mock_response):
            result = check_api_auth(config)

        assert result["success"] is False
        assert result["status_code"] == 403
        assert "Insufficient permissions" in result["error"]

    def test_check_api_auth_connection_error(self) -> None:
        """Connection error handling."""
        config = MagicMock()
        config.has_credentials = True
        config.api_key = "key"
        config.api_passphrase = "pass"

        with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
            result = check_api_auth(config)

        assert result["success"] is False
        assert "Connection error" in result["error"]

    def test_check_api_auth_timeout(self) -> None:
        """Timeout error handling."""
        config = MagicMock()
        config.has_credentials = True
        config.api_key = "key"
        config.api_passphrase = "pass"

        with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Timeout")):
            result = check_api_auth(config)

        assert result["success"] is False
        assert "Timeout error" in result["error"]


class TestFillPersistence:
    """Test fill loading and persistence."""

    def test_load_paper_fills_empty(self, tmp_path: Path) -> None:
        """Loading from non-existent file returns empty list."""
        fills_path = tmp_path / "nonexistent.jsonl"
        fills = load_paper_fills(fills_path)
        assert fills == []

    def test_load_paper_fills(self, tmp_path: Path) -> None:
        """Load fills from JSONL file."""
        fills_path = tmp_path / "fills.jsonl"
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc123",
                    "side": "buy",
                    "size": "100",
                    "price": "0.55",
                    "fee": "0.50",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xabc",
                    "market_slug": "test-market",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def456",
                    "side": "sell",
                    "size": "50",
                    "price": "0.60",
                    "fee": "0.25",
                    "timestamp": "2024-01-15T13:00:00+00:00",
                    "transaction_hash": "0xdef",
                    "market_slug": "test-market-2",
                }
            )
            + "\n"
        )

        fills = load_paper_fills(fills_path)
        assert len(fills) == 2

        assert fills[0].token_id == "abc123"
        assert fills[0].side == "buy"
        assert fills[0].size == Decimal("100")
        assert fills[0].price == Decimal("0.55")

        assert fills[1].token_id == "def456"
        assert fills[1].side == "sell"

    def test_get_existing_tx_hashes(self, tmp_path: Path) -> None:
        """Extract transaction hashes from fills file."""
        fills_path = tmp_path / "fills.jsonl"
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xabc",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def",
                    "side": "sell",
                    "size": "50",
                    "price": "0.6",
                    "fee": "0",
                    "timestamp": "2024-01-15T13:00:00+00:00",
                    "transaction_hash": "0xdef",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "ghi",
                    "side": "buy",
                    "size": "25",
                    "price": "0.7",
                    "fee": "0",
                    "timestamp": "2024-01-15T14:00:00+00:00",
                    # No transaction_hash
                }
            )
            + "\n"
        )

        hashes = get_existing_tx_hashes(fills_path)
        assert hashes == {"0xabc", "0xdef"}

    def test_get_existing_tx_hashes_empty(self, tmp_path: Path) -> None:
        """Empty file returns empty set."""
        fills_path = tmp_path / "fills.jsonl"
        hashes = get_existing_tx_hashes(fills_path)
        assert hashes == set()

    def test_append_fills(self, tmp_path: Path) -> None:
        """Append fills to file."""
        fills_path = tmp_path / "fills.jsonl"

        fill1 = Fill(
            token_id="abc",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.5"),
            fee=Decimal("0.5"),
            timestamp="2024-01-15T12:00:00+00:00",
            transaction_hash="0xabc",
            market_slug="test-market",
        )
        fill2 = Fill(
            token_id="def",
            side="sell",
            size=Decimal("50"),
            price=Decimal("0.6"),
            fee=Decimal("0.25"),
            timestamp="2024-01-15T13:00:00+00:00",
            transaction_hash="0xdef",
            market_slug="test-market",
        )

        count = append_fills([fill1, fill2], fills_path)
        assert count == 2

        # Verify file contents
        lines = fills_path.read_text().strip().split("\n")
        assert len(lines) == 2

        data1 = json.loads(lines[0])
        assert data1["token_id"] == "abc"
        assert data1["side"] == "buy"
        assert data1["size"] == "100"

        data2 = json.loads(lines[1])
        assert data2["token_id"] == "def"
        assert data2["side"] == "sell"

    def test_append_fills_empty(self, tmp_path: Path) -> None:
        """Appending empty list returns 0."""
        fills_path = tmp_path / "fills.jsonl"
        count = append_fills([], fills_path)
        assert count == 0
        assert not fills_path.exists()


class TestFillsSummary:
    """Test fills summary functions."""

    def test_get_fills_summary_empty(self, tmp_path: Path) -> None:
        """Summary for non-existent file."""
        fills_path = tmp_path / "fills.jsonl"
        summary = get_fills_summary(fills_path)

        assert summary["exists"] is False
        assert summary["total_fills"] == 0

    def test_get_fills_summary(self, tmp_path: Path) -> None:
        """Get summary of fills file."""
        fills_path = tmp_path / "fills.jsonl"
        now = datetime.now(UTC)

        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc123",
                    "side": "buy",
                    "size": "100",
                    "price": "0.55",
                    "fee": "0.50",
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                    "transaction_hash": "0xabc",
                    "market_slug": "market-1",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def456",
                    "side": "sell",
                    "size": "50",
                    "price": "0.60",
                    "fee": "0.25",
                    "timestamp": (now - timedelta(hours=1)).isoformat(),
                    "transaction_hash": "0xdef",
                    "market_slug": "market-2",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "abc123",  # Same token
                    "side": "buy",
                    "size": "25",
                    "price": "0.65",
                    "fee": "0.10",
                    "timestamp": now.isoformat(),
                    "transaction_hash": "0xghi",
                    "market_slug": "market-1",  # Same market
                }
            )
            + "\n"
        )

        summary = get_fills_summary(fills_path)

        assert summary["exists"] is True
        assert summary["total_fills"] == 3
        assert summary["unique_tokens"] == 2  # abc123, def456
        assert summary["unique_markets"] == 2  # market-1, market-2
        assert summary["last_fill_at"] is not None
        assert summary["age_seconds"] is not None

    def test_get_last_fill_timestamp(self, tmp_path: Path) -> None:
        """Extract last fill timestamp."""
        fills_path = tmp_path / "fills.jsonl"
        now = datetime.now(UTC)

        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc",
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def",
                    "timestamp": now.isoformat(),
                }
            )
            + "\n"
        )

        last_ts = get_last_fill_timestamp(fills_path)
        assert last_ts is not None
        # Allow small difference due to rounding
        assert abs((last_ts - now).total_seconds()) < 1

    def test_get_last_fill_timestamp_empty(self, tmp_path: Path) -> None:
        """Empty file returns None."""
        fills_path = tmp_path / "fills.jsonl"
        assert get_last_fill_timestamp(fills_path) is None


class TestCollectFills:
    """Test the collect_fills function."""

    def test_collect_fills_from_paper(self, tmp_path: Path) -> None:
        """Collect fills from paper trading."""
        paper_dir = tmp_path / "paper_trading"
        paper_dir.mkdir()
        paper_fills_path = paper_dir / "fills.jsonl"

        paper_fills_path.write_text(
            json.dumps(
                {
                    "token_id": "paper1",
                    "side": "buy",
                    "size": "100",
                    "price": "0.55",
                    "fee": "0.50",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xpaper1",
                    "market_slug": "paper-market",
                }
            )
            + "\n"
        )

        fills_path = tmp_path / "fills.jsonl"

        result = collect_fills(
            fills_path=fills_path,
            paper_fills_path=paper_fills_path,
            include_account=False,
            include_paper=True,
        )

        assert result["paper_fills"] == 1
        assert result["account_fills"] == 0
        assert result["total_appended"] == 1
        assert fills_path.exists()

    def test_collect_fills_deduplication(self, tmp_path: Path) -> None:
        """Duplicate fills are skipped."""
        fills_path = tmp_path / "fills.jsonl"

        # Pre-populate with existing fill
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "existing",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xexisting",
                }
            )
            + "\n"
        )

        paper_dir = tmp_path / "paper_trading"
        paper_dir.mkdir()
        paper_fills_path = paper_dir / "fills.jsonl"

        paper_fills_path.write_text(
            json.dumps(
                {
                    "token_id": "existing",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xexisting",  # Same tx hash
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "new",
                    "side": "sell",
                    "size": "50",
                    "price": "0.6",
                    "fee": "0",
                    "timestamp": "2024-01-15T13:00:00+00:00",
                    "transaction_hash": "0xnew",
                }
            )
            + "\n"
        )

        result = collect_fills(
            fills_path=fills_path,
            paper_fills_path=paper_fills_path,
            include_account=False,
            include_paper=True,
        )

        assert result["paper_fills"] == 2
        assert result["duplicates_skipped"] == 1
        assert result["total_appended"] == 1

        # Verify only new fill was added
        lines = fills_path.read_text().strip().split("\n")
        assert len(lines) == 2
