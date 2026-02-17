# Systemd Services

This directory contains systemd service files for running the Polymarket Bot components.

## Files

- `polymarket-watchdog.service` - The watchdog service that checks collector health
- `polymarket-watchdog.timer` - Timer that runs the service every 30 seconds
- `polymarket-fills-loop.service` - Continuous fills collection service

## Fills Loop Service

The fills collection service requires API credentials to fetch account fills from Polymarket.

### Prerequisites

1. Ensure you have Polymarket API credentials configured in one of:
   - Environment variables: `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE`
   - `.env` file in the project root
   - 1Password (if using the credential helper)

2. Verify credentials are working:
   ```bash
   ./scripts/check-credentials.sh
   ```

### Installation

1. Copy the service file to your systemd user directory:
   ```bash
   cp systemd/polymarket-fills-loop.service ~/.config/systemd/user/
   ```

2. Reload systemd:
   ```bash
   systemctl --user daemon-reload
   ```

3. Enable and start the service:
   ```bash
   systemctl --user enable polymarket-fills-loop.service
   systemctl --user start polymarket-fills-loop.service
   ```

### Credential Loading Options

The service supports multiple credential loading methods:

**Option 1: Environment Variables**
Set credentials in your shell or systemd user session before starting the service.

**Option 2: 1Password (Recommended)**
Use the wrapper script that loads credentials from 1Password:
```bash
# The service uses run-fills-loop-with-creds.sh which auto-loads from 1Password
# Ensure you're signed in: eval $(op signin)
```

**Option 3: .env File**
Create a `.env` file with your credentials (see `.env.example`).

### Verification

Check service status:
```bash
systemctl --user status polymarket-fills-loop.service
```

View recent logs:
```bash
journalctl --user -u polymarket-fills-loop.service -n 50
```

Check fills data:
```bash
tail -f data/fills.jsonl
./scripts/check-credentials.sh
```

## Watchdog Service

### Installation

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

### Verification

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

### Manual Testing

Run the watchdog manually:
```bash
./run.sh watchdog --data-dir data --max-age-seconds 120
```

Dry run (don't actually restart):
```bash
./run.sh watchdog --data-dir data --max-age-seconds 120 --dry-run
```

### Behavior

- The watchdog checks `data/latest_15m.json` every 30 seconds
- If the file is older than 120 seconds, the watchdog restarts the collector
- Restart reason and last successful snapshot time are logged to:
  - `data/collector_watchdog.log` (JSON format)
  - systemd journal (via `journalctl`)
