# polymarket-bot

R&D repo for Polymarket market data collection, backtests, and paper/live execution.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or Python's built-in `venv`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd polymarket-bot

# Create virtual environment and install dependencies
uv venv .venv
uv pip install -e .

# Or use the provided runner which handles this automatically:
./run.sh venv
```

### Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your credentials:
# - POLYMARKET_API_KEY (optional, for authenticated endpoints)
# - Other service-specific keys as needed
```

## Usage

The project provides a `./run.sh` script for all common operations. This ensures commands run in the correct virtual environment.

### Common Commands

```bash
# Continuously collect 15m market snapshots (main collector loop)
./run.sh collect-15m-loop --out data --interval-seconds 60

# Fetch current 15-minute crypto interval markets
./run.sh markets-15m --limit 50

# Verify PnL from fills data
./run.sh pnl-verify --input data/fills.json --books data/books.json

# Run tests
./run.sh tests

# Check collector health
./run.sh health-check --data-dir data
```

### Pointer File Support

Commands that accept `--snapshot` (like `microstructure` and `pnl-verify`) support pointer files for convenience. Pointer files are JSON files containing `{"path": "...", "generated_at": "..."}` that point to actual snapshot files.

The collector automatically creates `data/latest_15m.json` as a pointer to the most recent snapshot:

```bash
# These are equivalent:
./run.sh microstructure --snapshot data/latest_15m.json
./run.sh microstructure --snapshot data/snapshot_15m_20250215T120000Z.json

# PnL verification also supports pointer files:
./run.sh pnl-verify --data-dir data --snapshot data/latest_15m.json
```

This is useful for:
- Scripts that always want the latest snapshot without parsing filenames
- Cron jobs that run analysis on the most recent data
- Avoiding race conditions when snapshots are being rotated

### Cron Usage

When running from cron, use the dedicated polymarket-bot venv (not axiom-trader's):

```bash
# Example crontab entry (runs every minute)
* * * * * cd /home/dan/src/polymarket-bot && /home/dan/src/polymarket-bot/.venv/bin/python -m polymarket.cli collect-15m-loop --interval-seconds 60 --max-backoff-seconds 60 >> /tmp/polymarket-collector.log 2>&1
```

Or use the run.sh script (recommended - handles venv automatically):

```bash
* * * * * cd /home/dan/src/polymarket-bot && ./run.sh collect-15m-loop --interval-seconds 60 >> /tmp/polymarket-collector.log 2>&1
```

See [docs/ops.md](docs/ops.md) for complete cron setup, monitoring, and troubleshooting.

### All Available Commands

Run `./run.sh` without arguments to see the full list of commands.

| Command | Description |
|---------|-------------|
| `collect-15m-loop` | Continuous 15m market data collection |
| `collect-fills-loop` | Continuous fills collection (requires API credentials) |
| `collect-5m` | Single 5m predictions snapshot |
| `collect-15m` | Single 15m snapshot |
| `markets-5m` | Fetch 5-minute prediction markets |
| `markets-15m` | Fetch 15-minute crypto interval markets |
| `universe-5m` | Build normalized market universe |
| `microstructure` | Analyze order book microstructure |
| `pnl-verify` | Verify PnL from fills JSON |
| `health-check` | Check collector health/staleness |
| `binance-collect` | Collect Binance BTC market data |
| `binance-loop` | Binance WebSocket collector loop |
| `binance-align` | Align Binance features to Polymarket |
| `tests` | Run pytest test suite |
| `shell` | Open shell with venv activated |
| `venv` | Ensure venv exists and is up to date |

## Project Structure

```
.
├── src/polymarket/       # Main source code
│   ├── cli.py            # CLI entry point
│   ├── collector.py      # Data collection logic
│   ├── collector_loop.py # Continuous collection loops
│   ├── pnl.py            # PnL verification
│   └── ...
├── tests/                # Test suite
├── data/                 # Collected data (gitignored)
├── docs/                 # Documentation
├── vendor/               # Git submodules
├── run.sh                # Canonical runner script
├── pyproject.toml        # Python project config
└── uv.lock              # Locked dependencies
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run linting
ruff check .
ruff format .

# Run tests
./run.sh tests

# Run specific test file
./run.sh tests tests/test_collector.py -v
```

## Data Collection

The collector stores snapshots in `data/`:

- `snapshot_15m_*.json` - 15m interval market snapshots
- `snapshot_5m_*.json` - 5m prediction market snapshots
- `binance/` - Binance market data
- `universe.json` - Normalized market universe

Old snapshots are automatically pruned based on retention settings (default: 24 hours).

## License

Private - For R&D use only.
