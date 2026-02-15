# polymarket-bot

R&D repo for Polymarket market data, backtests, and paper/live execution.

## Quick Start

```bash
# Fetch 15M crypto markets
uv run polymarket markets-15m

# Single snapshot with orderbooks
uv run polymarket collect-15m --out data

# Continuous 5s loop with retention pruning (default: 24h)
uv run polymarket collect-15m-loop --out data --interval 5 --prune-hours 24

# Smoke test (fetch + snapshot one cycle)
uv run polymarket markets-15m --limit 5 && uv run polymarket collect-15m --out data
```
