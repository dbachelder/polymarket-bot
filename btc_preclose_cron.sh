#!/bin/bash
set -euo pipefail

PY=/home/dan/src/axiom-trader/.venv/bin/python

# Check for existing daily PnL files before running
DAILY_FILES_BEFORE=$(find data/ -name "*daily*" -type f 2>/dev/null | wc -l)

echo "=== BTC Pre-close Paper Trading ==="
# Expanded window to 30min (1800s) and lowered cheap-price threshold to 0.02 for more fills
# Use monitor thresholds if available (--use-monitor-thresholds)
OUTPUT=$($PY -m polymarket.cli btc-preclose-paper --window-seconds 1800 --cheap-price 0.02 --size 1 --data-dir data/paper_trading --format human --use-monitor-thresholds)
echo "$OUTPUT"

# Parse fills_recorded from output
FILLS_RECORDED=$(echo "$OUTPUT" | grep -o "fills_recorded: [0-9]*" | head -1 | cut -d' ' -f2 || echo "0")
echo "Detected fills_recorded: $FILLS_RECORDED"

# Weather market fallback: if BTC has no opportunities, scan weather markets
if [ "$FILLS_RECORDED" -eq 0 ]; then
    echo ""
    echo "=== BTC No Fills - Running Weather Market Fallback ==="
    WEATHER_OUTPUT=$($PY -m polymarket.cli weather-scan --live --format human)
    echo "$WEATHER_OUTPUT"
    
    # Parse weather fills
    WEATHER_FILLS=$(echo "$WEATHER_OUTPUT" | grep -o "fills_recorded: [0-9]*" | head -1 | cut -d' ' -f2 || echo "0")
    if [ "$WEATHER_FILLS" -gt 0 ]; then
        echo "Weather markets generated $WEATHER_FILLS fills"
        FILLS_RECORDED=$WEATHER_FILLS
    fi
fi

if [ "$FILLS_RECORDED" -gt 0 ]; then
    echo ""
    echo "=== Collecting Fills ==="
    $PY -m polymarket.cli collect-fills --format human
    
    echo ""
    echo "=== PnL Verification ==="
    SINCE_TIME=$(date -u -d '6 hours ago' +%Y-%m-%dT%H:%M:%SZ)
    PNL_OUTPUT=$($PY -m polymarket.cli pnl-verify --data-dir data --snapshot data/latest_15m.json --since "$SINCE_TIME" --save-daily --format human)
    echo "$PNL_OUTPUT"
    
    # Check if daily files were created/updated
    DAILY_FILES_AFTER=$(find data/ -name "*daily*" -type f 2>/dev/null | wc -l)
    ARTIFACTS_UPDATED=false
    
    if [ "$DAILY_FILES_AFTER" -gt "$DAILY_FILES_BEFORE" ]; then
        ARTIFACTS_UPDATED=true
        echo "New daily artifacts detected"
    elif echo "$PNL_OUTPUT" | grep -q -E "(saved|wrote|created)"; then
        ARTIFACTS_UPDATED=true
        echo "PnL artifacts updated based on output"
    fi
    
    echo "REPORT_NEEDED=true"
    echo "FILLS_RECORDED=$FILLS_RECORDED"
    echo "ARTIFACTS_UPDATED=$ARTIFACTS_UPDATED"
else
    echo "REPORT_NEEDED=false"
fi
