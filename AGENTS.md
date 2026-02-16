# AGENTS.md - Polymarket Bot

Guide for AI agents working on this codebase.

## Quick Start

```bash
./run.sh venv           # Ensure venv exists
./run.sh tests          # Run tests
./run.sh shell          # Shell with venv activated
```

## Project Overview

R&D repo for Polymarket market data collection, backtests, and paper/live execution.
Focuses on 15-minute crypto interval markets (BTC price predictions) and 5-minute prediction markets.

## Structure

```
src/polymarket/          # Main source code
├── cli.py               # CLI entry point (all commands)
├── collector.py         # Data collection logic
├── collector_loop.py    # Continuous collection loops
├── pnl.py              # PnL verification
└── ...
tests/                   # Test suite
data/                    # Collected data (gitignored)
docs/                    # Documentation
```

## Key Commands

```bash
./run.sh collect-15m-loop  # Continuous 15m market data collection
./run.sh markets-15m       # Fetch current 15m crypto markets
./run.sh binance-collect   # Collect Binance BTC data
./run.sh binance-align     # Align Binance features to Polymarket
./run.sh pnl-verify        # Verify PnL from fills
./run.sh health-check      # Check collector health
./run.sh tests             # Run pytest suite
```

## Development

```bash
uv pip install -e ".[dev]"   # Dev dependencies
uv run ruff check src tests  # Linting
uv run ruff format src tests # Formatting
./run.sh tests               # Tests
```

## Conventions

- **Python 3.11+**, managed via `uv`
- **Linting:** ruff (check + format)
- **Tests:** pytest
- **Data:** stored in `data/` (gitignored), JSON + parquet formats
- **Timezone:** UTC everywhere, convert at display only

## GitHub Identity

Use bot identity for all GitHub operations:
```bash
gh-as-ada issue create ...
gh-as-ada pr create ...
```

Git config (per-repo):
```bash
git config user.name "ada-codesushi[bot]"
git config user.email "ada-codesushi[bot]@users.noreply.github.com"
```
