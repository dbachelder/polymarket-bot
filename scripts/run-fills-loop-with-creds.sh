#!/bin/bash
# Wrapper to start fills collection loop with 1Password credentials
# Usage: ./scripts/run-fills-loop-with-creds.sh [args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Check if 1Password CLI is available and signed in
if command -v op &> /dev/null && op account list &> /dev/null; then
    echo "Loading credentials from 1Password..."
    ITEM_NAME="Polymarket API"
    
    # Try to load credentials from 1Password
    API_KEY=$(op item get "$ITEM_NAME" --field password 2>/dev/null || echo "")
    API_SECRET=$(op item get "$ITEM_NAME" --field "API Secret" 2>/dev/null || echo "")
    API_PASSPHRASE=$(op item get "$ITEM_NAME" --field "Passphrase" 2>/dev/null || echo "")
    
    if [ -n "$API_KEY" ] && [ -n "$API_SECRET" ] && [ -n "$API_PASSPHRASE" ]; then
        export POLYMARKET_API_KEY="$API_KEY"
        export POLYMARKET_API_SECRET="$API_SECRET"
        export POLYMARKET_API_PASSPHRASE="$API_PASSPHRASE"
        echo "Credentials loaded from 1Password"
    else
        echo "Warning: Could not load all credentials from 1Password (item: $ITEM_NAME)"
        echo "Falling back to environment/.env file"
    fi
else
    echo "1Password CLI not available or not signed in, using environment/.env file"
fi

# Run the fills loop with all passed arguments
exec ./run.sh collect-fills-loop "$@"
