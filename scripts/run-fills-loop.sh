#!/bin/bash
# Wrapper script to run collect-fills-loop with 1Password credentials
# Usage: ./scripts/run-fills-loop.sh

set -e

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)"

# Check if 1Password is signed in
if ! op whoami &> /dev/null; then
    echo "Signing in to 1Password..."
    eval $(op signin --account my)
fi

# Source the credentials
source "$SCRIPT_DIR/scripts/load-env-from-1password.sh"

# Run the fills loop
echo "Starting collect-fills-loop with authenticated credentials..."
exec ./run.sh collect-fills-loop --interval-seconds 300 --stale-alert-hours 6 --no-account --paper
