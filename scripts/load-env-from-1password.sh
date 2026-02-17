#!/bin/bash
# Load Polymarket API credentials from 1Password into environment
# Usage: source ./scripts/load-env-from-1password.sh

set -e

# Check if op CLI is installed
if ! command -v op &> /dev/null; then
    echo "Error: 1Password CLI (op) is not installed"
    echo "Install from: https://1password.com/downloads/command-line/"
    return 1
fi

# Check if signed in
if ! op account list &> /dev/null; then
    echo "Error: Not signed in to 1Password"
    echo "Run: eval \$(op signin)"
    return 1
fi

# Item name in 1Password (adjust if different)
ITEM_NAME="Polymarket API"

echo "Loading Polymarket credentials from 1Password..."

# Load credentials
export POLYMARKET_API_KEY=$(op item get "$ITEM_NAME" --field password 2>/dev/null || echo "")
export POLYMARKET_API_SECRET=$(op item get "$ITEM_NAME" --field "API Secret" 2>/dev/null || echo "")
export POLYMARKET_API_PASSPHRASE=$(op item get "$ITEM_NAME" --field "Passphrase" 2>/dev/null || echo "")

# Verify credentials were loaded
if [ -z "$POLYMARKET_API_KEY" ] || [ -z "$POLYMARKET_API_SECRET" ] || [ -z "$POLYMARKET_API_PASSPHRASE" ]; then
    echo "Warning: Some credentials could not be loaded from 1Password"
    echo "Check that item '$ITEM_NAME' exists with fields: password, API Secret, Passphrase"
    return 1
fi

echo "Credentials loaded successfully!"
echo "POLYMARKET_API_KEY: ***${POLYMARKET_API_KEY: -4}"
echo "POLYMARKET_API_SECRET: ***${POLYMARKET_API_SECRET: -4}"
echo "POLYMARKET_API_PASSPHRASE: ***${POLYMARKET_API_PASSPHRASE: -4}"
