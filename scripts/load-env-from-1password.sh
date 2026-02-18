#!/bin/bash
# Load Polymarket API credentials from 1Password
# Usage: source ./scripts/load-env-from-1password.sh

set -e

# Check if op CLI is available
if ! command -v op &> /dev/null; then
    echo "ERROR: 1Password CLI (op) not found" >&2
    return 1
fi

# Check if already signed in to 1Password
if ! op whoami &> /dev/null; then
    echo "ERROR: Not signed in to 1Password. Run: eval \$(op signin)" >&2
    return 1
fi

# Try to find Polymarket API item
ITEM_NAME="Polymarket API"
if ! op item list | grep -q "$ITEM_NAME"; then
    echo "ERROR: 1Password item '$ITEM_NAME' not found" >&2
    return 1
fi

# Load credentials
export POLYMARKET_API_KEY=$(op item get "$ITEM_NAME" --field password)
export POLYMARKET_API_SECRET=$(op item get "$ITEM_NAME" --field "API Secret")
export POLYMARKET_API_PASSPHRASE=$(op item get "$ITEM_NAME" --field "Passphrase")

echo "Loaded Polymarket API credentials from 1Password"
