#!/bin/bash
# Validate Polymarket API credentials are properly configured
# Usage: ./scripts/check-credentials.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Ensure venv is available
if [[ ! -d ".venv" ]]; then
    echo "Error: Virtual environment not found. Run: ./run.sh venv"
    exit 1
fi

PYTHON="$SCRIPT_DIR/.venv/bin/python"

echo "Checking Polymarket API credentials..."
echo ""

# Run Python credential check
$PYTHON -c "
from polymarket.config import load_config
from polymarket.fills_collector import validate_credentials, check_api_auth

config = load_config()
validation = validate_credentials(config)

print('=== Credential Status ===')
print(f'Has credentials: {validation[\"has_credentials\"]}')
print(f'Can trade:       {validation[\"can_trade\"]}')
print(f'Dry run:         {validation[\"dry_run\"]}')
print(f'API Key:         {validation[\"api_key\"]} (length: {validation[\"api_key_length\"]})')
print(f'API Secret:      {validation[\"api_secret\"]} (length: {validation[\"api_secret_length\"]})')
print(f'API Passphrase:  {validation[\"api_passphrase\"]} (length: {validation[\"api_passphrase_length\"]})')

if validation['warnings']:
    print()
    print('=== Warnings ===')
    for warning in validation['warnings']:
        print(f'  - {warning}')

if validation['has_credentials']:
    print()
    print('=== Testing API Authentication ===')
    auth_test = check_api_auth(config)
    print(f'Endpoint:    {auth_test[\"endpoint\"]}')
    print(f'Status:      {auth_test[\"status_code\"]}')
    print(f'Success:     {auth_test[\"success\"]}')
    if auth_test['error']:
        print(f'Error:       {auth_test[\"error\"]}')
else:
    print()
    print('=== How to Fix ===')
    print('1. Set credentials in environment:')
    print('   export POLYMARKET_API_KEY=your_key')
    print('   export POLYMARKET_API_SECRET=your_secret')
    print('   export POLYMARKET_API_PASSPHRASE=your_passphrase')
    print()
    print('2. Or load from 1Password:')
    print('   source ./scripts/load-env-from-1password.sh')
    print()
    print('3. Or create a .env file:')
    print('   cp .env.example .env')
    print('   # Edit .env with your credentials')
"

echo ""
echo "Credential check complete."
