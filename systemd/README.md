# Systemd Services

This directory contains systemd service and timer files for running Polymarket Bot services.

## Services Overview

| Service | Purpose | Credential Required |
|---------|---------|---------------------|
| `polymarket-watchdog` | Monitors collector health and restarts if stale | No |
| `collect-fills-loop` | Continuously collects trade fills from Polymarket | Yes (API Key) |

## Collect Fills Loop Service

**IMPORTANT:** This service requires Polymarket API credentials. See [docs/ops.md](../docs/ops.md) for credential setup instructions.

### Installation

1. Copy the service file to your systemd user directory:
   ```bash
   cp systemd/collect-fills-loop.service ~/.config/systemd/user/
   ```

2. **Configure credentials**: The service loads credentials from 1Password. Ensure you have:
   - 1Password CLI installed and configured
   - A "Polymarket API" item in your personal vault with fields: password, API Secret, Passphrase

3. Reload systemd:
   ```bash
   systemctl --user daemon-reload
   ```

4. Enable and start the service:
   ```bash
   systemctl --user enable collect-fills-loop.service
   systemctl --user start collect-fills-loop.service
   ```

### Verification

```bash
# Check service status
systemctl --user status collect-fills-loop.service

# View logs
journalctl --user -u collect-fills-loop.service -f

# Check if fills are being collected
ls -la data/fills.json
tail data/fills.json
```

### Manual Operation

For interactive use with credential loading:

```bash
./scripts/run-fills-loop.sh
```

## Watchdog Service

This directory contains systemd service and timer files for running the Polymarket Bot collector watchdog.

## Files

- `polymarket-watchdog.service` - The watchdog service that checks collector health
- `polymarket-watchdog.timer` - Timer that runs the service every 30 seconds

## Installation

1. Copy the service and timer files to your systemd user directory:
   ```bash
   cp systemd/polymarket-watchdog.service ~/.config/systemd/user/
   cp systemd/polymarket-watchdog.timer ~/.config/systemd/user/
   ```

2. Reload systemd:
   ```bash
   systemctl --user daemon-reload
   ```

3. Enable and start the timer:
   ```bash
   systemctl --user enable polymarket-watchdog.timer
   systemctl --user start polymarket-watchdog.timer
   ```

## Verification

Check timer status:
```bash
systemctl --user status polymarket-watchdog.timer
```

View recent watchdog logs:
```bash
journalctl --user -u polymarket-watchdog.service -n 20
```

View watchdog log file:
```bash
tail -f data/collector_watchdog.log
```

## Manual Testing

Run the watchdog manually:
```bash
./run.sh watchdog --data-dir data --max-age-seconds 120
```

Dry run (don't actually restart):
```bash
./run.sh watchdog --data-dir data --max-age-seconds 120 --dry-run
```

## Behavior

- The watchdog checks `data/latest_15m.json` every 30 seconds
- If the file is older than 120 seconds, the watchdog restarts the collector
- Restart reason and last successful snapshot time are logged to:
  - `data/collector_watchdog.log` (JSON format)
  - systemd journal (via `journalctl`)
