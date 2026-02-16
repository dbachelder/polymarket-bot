# Operations Guide

This document covers running polymarket-bot in production, including cron setup, monitoring, and troubleshooting.

## Virtual Environment

The project uses a dedicated virtual environment at `.venv/` in the project root. **Do not use axiom-trader's venv.**

### Quick Setup

```bash
# From project root
./run.sh venv
```

This creates `.venv/` and installs all dependencies from `uv.lock`.

### Manual Activation

If you need to run Python directly (e.g., from cron):

```bash
cd /home/dan/src/polymarket-bot
source .venv/bin/activate
python -m polymarket.cli <command>
```

Or use the full path without activating:

```bash
/home/dan/src/polymarket-bot/.venv/bin/python -m polymarket.cli <command>
```

## Cron Setup

### Recommended: Use run.sh

The simplest approach is using `./run.sh` which handles venv activation automatically:

```bash
# Example crontab entries

# Collect 15m market data every minute
* * * * * cd /home/dan/src/polymarket-bot && ./run.sh collect-15m-loop --interval-seconds 60 --max-backoff-seconds 60 >> /tmp/polymarket-collector.log 2>&1

# Run PnL verification every 5 minutes
*/5 * * * * cd /home/dan/src/polymarket-bot && ./run.sh pnl-loop >> /tmp/polymarket-pnl.log 2>&1

# Health check every 10 minutes
*/10 * * * * cd /home/dan/src/polymarket-bot && ./run.sh health-check --data-dir data >> /tmp/polymarket-health.log 2>&1

# Copy trading fills collection every 2 minutes
*/2 * * * * cd /home/dan/src/polymarket-bot && ./run.sh collect-fills >> /tmp/polymarket-fills.log 2>&1
```

### Alternative: Direct Python Path

If you prefer explicit paths:

```bash
* * * * * /home/dan/src/polymarket-bot/.venv/bin/python -m polymarket.cli collect-15m-loop --interval-seconds 60 --max-backoff-seconds 60 >> /tmp/polymarket-collector.log 2>&1
```

### Verifying the Venv

To confirm cron jobs use the correct venv:

```bash
# Check which Python is being used
ps aux | grep polymarket

# Should show: /home/dan/src/polymarket-bot/.venv/bin/python
# NOT: /home/dan/src/axiom-trader/.venv/bin/python
```

## Dependency Management

Dependencies are locked in `uv.lock`. To update:

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package httpx

# Sync venv with lock file
uv sync
```

### Adding New Dependencies

Edit `pyproject.toml` then regenerate the lock file:

```bash
uv lock
uv sync
```

## Monitoring

### Log Files

| Service | Log Location |
|---------|-------------|
| 15m Collector | `/tmp/polymarket-collector.log` |
| PnL Loop | `/tmp/polymarket-pnl.log` |
| Health Check | `/tmp/polymarket-health.log` |
| Fills Collection | `/tmp/polymarket-fills.log` |

### Health Checks

```bash
# Check collector is producing fresh data
./run.sh health-check --data-dir data --max-age-seconds 120

# Check PnL data freshness
./run.sh pnl-health --data-dir data
```

### Watchdog

The watchdog can monitor and restart stale collectors:

```bash
# Run once (useful for cron)
./run.sh watchdog --data-dir data --max-age-seconds 180

# Dry-run mode (log only, don't restart)
./run.sh watchdog --dry-run
```

## Troubleshooting

### "No module named polymarket"

The venv isn't activated. Use `./run.sh` or activate manually:

```bash
source .venv/bin/activate
```

### Dependencies out of sync

If you see import errors after pulling:

```bash
./run.sh venv  # Re-creates .venv/.deps-synced trigger
```

Or force reinstall:

```bash
rm .venv/.deps-synced
./run.sh venv
```

### Stale data

Check if the collector is running:

```bash
ps aux | grep collect-15m-loop
./run.sh health-check
```

If stale, the watchdog (if configured in cron) will restart it. Or manually:

```bash
./run.sh watchdog
```

## Environment Variables

Create `.env` in the project root:

```bash
POLYMARKET_API_KEY=your_key_here
POLYMARKET_API_SECRET=your_secret_here
POLYMARKET_API_PASSPHRASE=your_passphrase_here
POLYMARKET_DRY_RUN=true
```

For cron jobs, ensure the environment is loaded. `run.sh` handles this automatically.

## Migration from axiom-trader venv

If you have existing cron jobs using axiom-trader's venv:

**Before:**
```bash
* * * * * /home/dan/src/axiom-trader/.venv/bin/python -m polymarket.cli collect-15m-loop
```

**After:**
```bash
* * * * * cd /home/dan/src/polymarket-bot && ./run.sh collect-15m-loop
```

Or if you must use direct paths:
```bash
* * * * * /home/dan/src/polymarket-bot/.venv/bin/python -m polymarket.cli collect-15m-loop
```
