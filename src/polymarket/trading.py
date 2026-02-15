"""Trading operations for Polymarket CLOB.

This module provides order submission with mandatory dry-run safety.
By default, all orders are logged but NOT submitted to the exchange.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx

from .endpoints import CLOB_BASE

if TYPE_CHECKING:
    from .config import PolymarketConfig


@dataclass(frozen=True)
class Order:
    """Represents a CLOB order.

    Attributes:
        token_id: The token ID to trade
        side: 'buy' or 'sell'
        size: Order size (number of contracts)
        price: Limit price (0.01 to 0.99)
    """

    token_id: str
    side: str  # 'buy' or 'sell'
    size: Decimal
    price: Decimal

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.side not in ("buy", "sell"):
            msg = f"side must be 'buy' or 'sell', got {self.side}"
            raise ValueError(msg)
        if self.size <= 0:
            msg = f"size must be positive, got {self.size}"
            raise ValueError(msg)
        if not (Decimal("0.01") <= self.price <= Decimal("0.99")):
            msg = f"price must be between 0.01 and 0.99, got {self.price}"
            raise ValueError(msg)


@dataclass(frozen=True)
class OrderResult:
    """Result of an order submission attempt.

    Attributes:
        success: Whether the order was accepted
        order_id: The order ID (if submitted successfully)
        dry_run: Whether this was a dry-run (not actually submitted)
        message: Human-readable result message
        raw_response: Raw API response (if live submission)
    """

    success: bool
    order_id: str | None = None
    dry_run: bool = False
    message: str = ""
    raw_response: dict | None = None


def _generate_signature(
    *,
    secret: str,
    timestamp: str,
    method: str,
    request_path: str,
    body: str = "",
) -> str:
    """Generate HMAC-SHA256 signature for CLOB API authentication.

    Args:
        secret: API secret key
        timestamp: Unix timestamp in milliseconds as string
        method: HTTP method (GET, POST, etc.)
        request_path: API endpoint path
        body: Request body (for POST requests)

    Returns:
        Hex-encoded signature string.
    """
    message = timestamp + method.upper() + request_path + body
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return signature


def _build_auth_headers(
    config: PolymarketConfig,
    *,
    method: str,
    request_path: str,
    body: str = "",
) -> dict[str, str]:
    """Build authentication headers for CLOB API request.

    Args:
        config: Polymarket configuration with credentials
        method: HTTP method
        request_path: API endpoint path
        body: Request body

    Returns:
        Dictionary of HTTP headers.

    Raises:
        ValueError: If credentials are missing.
    """
    if not config.has_credentials:
        msg = "Cannot build auth headers: credentials missing"
        raise ValueError(msg)

    timestamp = str(int(time.time() * 1000))
    signature = _generate_signature(
        secret=config.api_secret or "",
        timestamp=timestamp,
        method=method,
        request_path=request_path,
        body=body,
    )

    return {
        "POLYMARKET-API-KEY": config.api_key or "",
        "POLYMARKET-SIGNATURE": signature,
        "POLYMARKET-TIMESTAMP": timestamp,
        "POLYMARKET-PASSPHRASE": config.api_passphrase or "",
        "Content-Type": "application/json",
    }


def submit_order(
    order: Order,
    config: PolymarketConfig | None = None,
) -> OrderResult:
    """Submit an order to Polymarket CLOB.

    By default, this operates in dry-run mode where orders are validated
    and logged but NOT actually submitted to the exchange. This is a
    safety feature to prevent accidental live trading.

    To enable live trading, set POLYMARKET_DRY_RUN=false and provide
    valid credentials.

    Args:
        order: The order to submit
        config: Configuration (loads from env if not provided)

    Returns:
        OrderResult with submission status

    Example:
        >>> from decimal import Decimal
        >>> from polymarket.trading import Order, submit_order
        >>> order = Order(
        ...     token_id="...",
        ...     side="buy",
        ...     size=Decimal("10"),
        ...     price=Decimal("0.55"),
        ... )
        >>> result = submit_order(order)
        >>> print(result.message)
        [DRY-RUN] Would submit buy order: 10 @ 0.55
    """
    if config is None:
        from .config import load_config

        config = load_config()

    # Validate the order can be submitted
    config.validate_or_raise()

    # Calculate order value for logging
    order_value = order.size * order.price

    if config.dry_run:
        # Dry-run: log what we would do but don't submit
        msg = (
            f"[DRY-RUN] Would submit {order.side} order: "
            f"{order.size} @ {order.price} "
            f"(value: ${order_value:.2f})"
        )
        return OrderResult(
            success=True,
            dry_run=True,
            message=msg,
        )

    # Live trading: submit to CLOB
    request_path = "/order"
    body_dict = {
        "tokenID": order.token_id,
        "side": order.side.upper(),
        "size": str(order.size),
        "price": str(order.price),
    }
    body = json.dumps(body_dict)

    headers = _build_auth_headers(
        config,
        method="POST",
        request_path=request_path,
        body=body,
    )

    try:
        with httpx.Client(base_url=CLOB_BASE, timeout=30.0) as client:
            response = client.post(request_path, headers=headers, content=body)
            response.raise_for_status()
            data = response.json()

            return OrderResult(
                success=True,
                order_id=data.get("orderID") or data.get("orderId"),
                dry_run=False,
                message=f"Order submitted: {order.side} {order.size} @ {order.price}",
                raw_response=data,
            )
    except httpx.HTTPStatusError as e:
        error_data = e.response.json() if e.response.text else {}
        return OrderResult(
            success=False,
            dry_run=False,
            message=f"Order failed: {e.response.status_code} - {error_data}",
            raw_response=error_data,
        )
    except Exception as e:
        return OrderResult(
            success=False,
            dry_run=False,
            message=f"Order error: {e!s}",
        )


def submit_order_safe(
    token_id: str,
    side: str,
    size: float | Decimal,
    price: float | Decimal,
    config: PolymarketConfig | None = None,
) -> OrderResult:
    """Convenience function to submit an order with primitive types.

    Args:
        token_id: Token ID to trade
        side: 'buy' or 'sell'
        size: Order size
        price: Limit price
        config: Optional config (loads from env if not provided)

    Returns:
        OrderResult with submission status
    """
    order = Order(
        token_id=token_id,
        side=side,
        size=Decimal(str(size)),
        price=Decimal(str(price)),
    )
    return submit_order(order, config)
