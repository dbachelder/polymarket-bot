"""Tests for polymarket.trading module."""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest

from polymarket.config import PolymarketConfig
from polymarket.trading import (
    Order,
    _build_auth_headers,
    _generate_signature,
    submit_order,
    submit_order_safe,
)


class TestOrder:
    """Tests for Order dataclass."""

    def test_valid_order(self) -> None:
        """Can create a valid order."""
        order = Order(
            token_id="0x123",
            side="buy",
            size=Decimal("10"),
            price=Decimal("0.55"),
        )
        assert order.token_id == "0x123"
        assert order.side == "buy"
        assert order.size == Decimal("10")
        assert order.price == Decimal("0.55")

    def test_invalid_side_raises(self) -> None:
        """Invalid side raises ValueError."""
        with pytest.raises(ValueError, match="side must be 'buy' or 'sell'"):
            Order(
                token_id="0x123",
                side="invalid",
                size=Decimal("10"),
                price=Decimal("0.55"),
            )

    def test_zero_size_raises(self) -> None:
        """Zero size raises ValueError."""
        with pytest.raises(ValueError, match="size must be positive"):
            Order(
                token_id="0x123",
                side="buy",
                size=Decimal("0"),
                price=Decimal("0.55"),
            )

    def test_negative_size_raises(self) -> None:
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="size must be positive"):
            Order(
                token_id="0x123",
                side="buy",
                size=Decimal("-1"),
                price=Decimal("0.55"),
            )

    def test_price_too_low_raises(self) -> None:
        """Price below 0.01 raises ValueError."""
        with pytest.raises(ValueError, match="price must be between 0.01 and 0.99"):
            Order(
                token_id="0x123",
                side="buy",
                size=Decimal("10"),
                price=Decimal("0.001"),
            )

    def test_price_too_high_raises(self) -> None:
        """Price above 0.99 raises ValueError."""
        with pytest.raises(ValueError, match="price must be between 0.01 and 0.99"):
            Order(
                token_id="0x123",
                side="buy",
                size=Decimal("10"),
                price=Decimal("1.00"),
            )


class TestGenerateSignature:
    """Tests for signature generation."""

    def test_signature_format(self) -> None:
        """Signature is 64-character hex string."""
        sig = _generate_signature(
            secret="test_secret",
            timestamp="1234567890000",
            method="POST",
            request_path="/order",
            body='{"test": true}',
        )
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_signature_deterministic(self) -> None:
        """Same inputs produce same signature."""
        sig1 = _generate_signature(
            secret="test_secret",
            timestamp="1234567890000",
            method="POST",
            request_path="/order",
        )
        sig2 = _generate_signature(
            secret="test_secret",
            timestamp="1234567890000",
            method="POST",
            request_path="/order",
        )
        assert sig1 == sig2

    def test_different_secrets_produce_different_sigs(self) -> None:
        """Different secrets produce different signatures."""
        sig1 = _generate_signature(
            secret="secret1",
            timestamp="1234567890000",
            method="POST",
            request_path="/order",
        )
        sig2 = _generate_signature(
            secret="secret2",
            timestamp="1234567890000",
            method="POST",
            request_path="/order",
        )
        assert sig1 != sig2


class TestBuildAuthHeaders:
    """Tests for building authentication headers."""

    def test_builds_all_required_headers(self) -> None:
        """All required headers are present."""
        config = PolymarketConfig(
            api_key="test_key",
            api_secret="test_secret",
            api_passphrase="test_pass",
            dry_run=True,
        )
        headers = _build_auth_headers(
            config,
            method="POST",
            request_path="/order",
            body='{"test": true}',
        )

        assert "POLYMARKET-API-KEY" in headers
        assert "POLYMARKET-SIGNATURE" in headers
        assert "POLYMARKET-TIMESTAMP" in headers
        assert "POLYMARKET-PASSPHRASE" in headers
        assert headers["POLYMARKET-API-KEY"] == "test_key"
        assert headers["POLYMARKET-PASSPHRASE"] == "test_pass"

    def test_raises_without_credentials(self) -> None:
        """Raises error if credentials missing."""
        config = PolymarketConfig()
        with pytest.raises(ValueError, match="credentials missing"):
            _build_auth_headers(config, method="GET", request_path="/markets")


class TestSubmitOrder:
    """Tests for submit_order function."""

    def test_dry_run_without_credentials(self) -> None:
        """Dry-run works without credentials."""
        order = Order(
            token_id="0x123",
            side="buy",
            size=Decimal("10"),
            price=Decimal("0.55"),
        )
        config = PolymarketConfig(dry_run=True)

        result = submit_order(order, config)

        assert result.success is True
        assert result.dry_run is True
        assert "[DRY-RUN]" in result.message
        assert "buy order" in result.message

    def test_dry_run_with_credentials(self) -> None:
        """Dry-run works even with credentials present."""
        order = Order(
            token_id="0x123",
            side="sell",
            size=Decimal("5"),
            price=Decimal("0.45"),
        )
        config = PolymarketConfig(
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
            dry_run=True,
        )

        result = submit_order(order, config)

        assert result.success is True
        assert result.dry_run is True
        assert "[DRY-RUN]" in result.message
        assert "sell order" in result.message

    def test_live_without_credentials_raises(self) -> None:
        """Live trading without credentials raises error."""
        order = Order(
            token_id="0x123",
            side="buy",
            size=Decimal("10"),
            price=Decimal("0.55"),
        )
        config = PolymarketConfig(dry_run=False)

        with pytest.raises(ValueError, match="Live trading requested"):
            submit_order(order, config)

    def test_loads_config_if_not_provided(self) -> None:
        """Auto-loads config if not provided."""
        order = Order(
            token_id="0x123",
            side="buy",
            size=Decimal("10"),
            price=Decimal("0.55"),
        )

        with mock.patch("polymarket.config.load_config") as mock_load:
            mock_load.return_value = PolymarketConfig(dry_run=True)
            result = submit_order(order)

        assert result.dry_run is True
        mock_load.assert_called_once()


class TestSubmitOrderSafe:
    """Tests for submit_order_safe convenience function."""

    def test_converts_float_to_decimal(self) -> None:
        """Converts float inputs to Decimal."""
        config = PolymarketConfig(dry_run=True)

        result = submit_order_safe(
            token_id="0x123",
            side="buy",
            size=10.5,
            price=0.55,
            config=config,
        )

        assert result.success is True
        assert result.dry_run is True

    def test_converts_string_to_decimal(self) -> None:
        """Converts string inputs to Decimal."""
        config = PolymarketConfig(dry_run=True)

        result = submit_order_safe(
            token_id="0x123",
            side="buy",
            size="10.5",
            price="0.55",
            config=config,
        )

        assert result.success is True
        assert result.dry_run is True
