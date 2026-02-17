# Polymarket Credential Setup Guide

## Problem

The fills collector requires authenticated API access to fetch account fills. Without credentials, it will log:

```
FILLS AUTH MISSING: No API credentials configured (POLYMARKET_API_KEY / POLYMARKET_API_PASSPHRASE)
```

## Required Credentials

You need three values from Polymarket:
1. **API Key** - Your public API identifier
2. **API Secret** - Used for signing requests
3. **API Passphrase** - Additional authentication factor

## Getting Credentials from Polymarket

1. Sign in to https://polymarket.com
2. Go to **Settings** → **API Keys**
3. Click **Generate New Key**
4. Select permissions (at minimum "Read" for fills collection)
5. Copy all three values (Key, Secret, Passphrase)

## Setup Options

### Option 1: Manual .env file

```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

### Option 2: Environment variables

```bash
export POLYMARKET_API_KEY="your-api-key"
export POLYMARKET_API_SECRET="your-api-secret"
export POLYMARKET_API_PASSPHRASE="your-passphrase"
```

### Option 3: 1Password (Recommended)

If you store credentials in 1Password:

```bash
# Sign in to 1Password
eval $(op signin)

# Load credentials into environment
source ./scripts/load-env-from-1password.sh

# Run the fills collector
./run.sh collect-fills-loop
```

**1Password Item Setup:**
- Create an item named "Polymarket API"
- Fields:
  - `password`: API Key
  - `API Secret`: API Secret
  - `Passphrase`: API Passphrase

## Verification

After setting credentials, verify they work:

```bash
./run.sh collect-fills --account --paper
```

You should see fills being collected instead of the auth missing warning.

## Security Notes

- `.env` is gitignored - never commit credentials
- The helper script only loads credentials into environment, doesn't write to disk
- Use read-only API keys when possible (sufficient for fills collection)
