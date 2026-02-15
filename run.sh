#!/bin/bash
# polymarket-bot runner script
# Provides canonical invocations for common operations

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure virtual environment exists
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Run command using the venv python
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Install/sync dependencies if needed
if [[ ! -f ".venv/.deps-synced" ]] || [[ "pyproject.toml" -nt ".venv/.deps-synced" ]]; then
    echo "Installing dependencies..."
    uv pip install -e .
    touch .venv/.deps-synced
fi

# Dispatch commands
case "${1:-}" in
    collect-15m-loop)
        shift
        $PYTHON -m polymarket.cli collect-15m-loop "$@"
        ;;
    markets-15m)
        shift
        $PYTHON -m polymarket.cli markets-15m "$@"
        ;;
    pnl-verify)
        shift
        $PYTHON -m polymarket.cli pnl-verify "$@"
        ;;
    copytrade-loop)
        shift
        $PYTHON -m polymarket.cli copytrade-loop "$@"
        ;;
    copytrade-pnl)
        shift
        $PYTHON -m polymarket.cli copytrade-pnl "$@"
        ;;
    copytrade-collect)
        shift
        $PYTHON -m polymarket.cli copytrade-collect "$@"
        ;;
    tests|test)
        shift
        $PYTHON -m pytest "$@"
        ;;
    health-check)
        shift
        $PYTHON -m polymarket.cli health-check "$@"
        ;;
    collect-5m)
        shift
        $PYTHON -m polymarket.cli collect-5m "$@"
        ;;
    collect-15m)
        shift
        $PYTHON -m polymarket.cli collect-15m "$@"
        ;;
    markets-5m)
        shift
        $PYTHON -m polymarket.cli markets-5m "$@"
        ;;
    universe-5m)
        shift
        $PYTHON -m polymarket.cli universe-5m "$@"
        ;;
    microstructure)
        shift
        $PYTHON -m polymarket.cli microstructure "$@"
        ;;
    binance-collect)
        shift
        $PYTHON -m polymarket.cli binance-collect "$@"
        ;;
    binance-loop)
        shift
        $PYTHON -m polymarket.cli binance-loop "$@"
        ;;
    binance-align)
        shift
        $PYTHON -m polymarket.cli binance-align "$@"
        ;;
    weather-consensus-scan)
        shift
        $PYTHON -m polymarket.cli weather-consensus-scan "$@"
        ;;
    weather-consensus-loop)
        shift
        $PYTHON -m polymarket.cli weather-consensus-loop "$@"
        ;;
    shell)
        # Drop into a shell with the venv activated
        exec bash --rcfile <(echo 'source "$SCRIPT_DIR/.venv/bin/activate"; cd "$SCRIPT_DIR"')
        ;;
    venv)
        # Just ensure venv exists and is up to date
        echo "Virtual environment ready at: $SCRIPT_DIR/.venv"
        ;;
    *)
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  collect-15m-loop   Continuously collect 15m market snapshots"
        echo "  markets-15m        Fetch 15-minute crypto interval markets"
        echo "  pnl-verify         Verify PnL from fills data"
        echo "  copytrade-loop     Run copytrade accounting loop (collect + daily PnL)"
        echo "  copytrade-pnl      Run single PnL verification on copytrade fills"
        echo "  copytrade-collect  Collect fills for a wallet (one-time)"
        echo "  tests              Run pytest test suite"
        echo "  health-check       Check collector health and staleness"
        echo ""
        echo "Additional commands:"
        echo "  collect-5m         Snapshot 5M predictions + CLOB orderbooks"
        echo "  collect-15m        Single 15m snapshot"
        echo "  markets-5m         Fetch 5-minute markets"
        echo "  universe-5m        Build normalized market universe"
        echo "  microstructure     Analyze market microstructure"
        echo "  binance-collect    Collect Binance BTC data (single snapshot)"
        echo "  binance-loop       Run Binance WebSocket collector loop"
        echo "  binance-align      Align Binance features to Polymarket"
        echo "  weather-consensus-scan  Scan for weather model consensus mispricing"
        echo "  weather-consensus-loop  Run continuous weather consensus scanning"
        echo ""
        echo "Utility commands:"
        echo "  shell              Open shell with venv activated"
        echo "  venv               Ensure venv exists and is up to date"
        echo ""
        echo "Examples:"
        echo "  $0 collect-15m-loop --out data --interval-seconds 60"
        echo "  $0 markets-15m --limit 20"
        echo "  $0 pnl-verify --input data/fills.json --books data/books.json"
        echo "  $0 copytrade-loop --wallet 0x... --interval-seconds 300"
        echo "  $0 copytrade-pnl --data-dir data/copytrade --snapshot data/latest_15m.json"
        exit 1
        ;;
esac
