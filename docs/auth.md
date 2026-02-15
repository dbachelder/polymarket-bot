# Polymarket CLOB Authentication Guide

This document describes the credentials required for trading on Polymarket's CLOB (Central Limit Order Book) and how to configure them safely.

## Overview

Polymarket CLOB uses API key authentication with cryptographic signing. To place orders, you need:

1. **API Key** - Identifies your account
2. **API Secret** - Used to sign requests (keep this secure!)
3. **Passphrase** - Additional authentication factor

## Required Credentials

### From Polymarket UI

1. Log into [polymarket.com](https://polymarket.com)
2. Go to **Settings** → **API Keys**
3. Generate a new API key pair
4. Save all three values:
   - **API Key** (starts with something like `0x...`)
   - **API Secret** (long hex string)
   - **Passphrase** (user-defined or auto-generated)

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `POLYMARKET_API_KEY` | Your API key | Live trading |
| `POLYMARKET_API_SECRET` | Your API secret | Live trading |
| `POLYMARKET_API_PASSPHRASE` | Your API passphrase | Live trading |
| `POLYMARKET_DRY_RUN` | Set to `true` to disable live trading | Always (safety) |

## Safe Configuration

### Option 1: Environment File (Local Development)

Create a `.env` file in the project root (already in `.gitignore`):

```bash
# .env
POLYMARKET_API_KEY=your_api_key_here
POLYMARKET_API_SECRET=your_api_secret_here
POLYMARKET_API_PASSPHRASE=your_passphrase_here

# Safety: always dry-run unless explicitly disabled
POLYMARKET_DRY_RUN=true
```

Load with:
```python
from polymarket.config import load_config

config = load_config()  # Auto-loads from .env
```

### Option 2: 1Password (Recommended for Production)

Store credentials in 1Password and inject at runtime:

```bash
# Using 1Password CLI
export POLYMARKET_API_KEY=$(op read "op://Private/Polymarket/API Key")
export POLYMARKET_API_SECRET=$(op read "op://Private/Polymarket/API Secret")
export POLYMARKET_API_PASSPHRASE=$(op read "op://Private/Polymarket/Passphrase")
export POLYMARKET_DRY_RUN=true
```

## Dry-Run Mode (Safety First)

By default, the bot operates in **dry-run mode** — it will:
- ✅ Log what orders would be placed
- ✅ Validate order parameters
- ✅ Calculate expected costs
- ❌ Never actually submit orders to the exchange

To enable live trading, explicitly set:
```bash
POLYMARKET_DRY_RUN=false
```

**⚠️ WARNING:** Only disable dry-run when you are absolutely ready to trade with real funds.

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Fill in your credentials from the Polymarket UI

3. Test in dry-run mode:
   ```bash
   python -c "from polymarket.trading import submit_order; submit_order(...)"
   # Will log: "[DRY-RUN] Would submit order: ..."
   ```

4. When ready, disable dry-run:
   ```bash
   export POLYMARKET_DRY_RUN=false
   ```

## Security Best Practices

1. **Never commit credentials** - `.env` is in `.gitignore`, but double-check
2. **Use dry-run by default** - Only disable when actively trading
3. **Rotate keys regularly** - Generate new API keys every 90 days
4. **Limit key permissions** - Use read-only keys for data collection, trading keys only for orders
5. **Monitor usage** - Check Polymarket UI for unexpected API activity

## Troubleshooting

### "Missing required credentials" error
- Ensure all three credentials are set in environment or `.env`
- Run `python -c "from polymarket.config import load_config; print(load_config())"` to verify

### Orders not being submitted
- Check `POLYMARKET_DRY_RUN` is set to `false` (not just unset)
- The order submitter will log `[DRY-RUN]` for all orders when enabled

### Invalid signature errors
- Verify your API secret is complete (not truncated)
- Check system clock is synchronized (NTP)
