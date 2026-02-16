# Systemd Watchdog Setup

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
