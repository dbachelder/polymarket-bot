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
├── odds_api.py         # Sharp book odds (Pinnacle/Betfair via OddsAPI)
├── sports_markets.py   # Polymarket sports market identification
└── sports_arbitrage.py # Cross-platform sports arbitrage strategy
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

## Sports Arbitrage Strategy

Cross-platform arbitrage between Polymarket sports markets and sharp sportsbooks.

```bash
# Scan for arbitrage opportunities
python -m polymarket.cli sports-scan

# Execute paper trades
python -m polymarket.cli sports-trade

# View strategy statistics
python -m polymarket.cli sports-stats
```

### Strategy Details

- **Data Sources**: Polymarket Gamma API + The Odds API (Pinnacle/Betfair)
- **Minimum Edge**: 2% after Polymarket withdrawal fees
- **Sports**: NFL, NBA, MLB, Premier League, NHL
- **Sizing**: Kelly/4, max 5% bankroll per trade
- **Paper Trading**: Logs to `data/sports_arb/opportunities.jsonl` and `trades.jsonl`

### Configuration

Set `ODDS_API_KEY` environment variable for sharp book data (free tier at https://the-odds-api.com).

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
