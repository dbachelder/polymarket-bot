# st1ne/polymarket-assistant Analysis

**Reference:** https://github.com/st1ne/polymarket-assistant  
**Analysis Date:** 2026-02-14  
**Ticket:** 1d4356e5-1fa7-4488-9b1a-c0c86fccd61b

---

## Overview

A real-time terminal dashboard combining Binance order flow with Polymarket prediction market prices to generate actionable crypto signals. The system computes 11 technical indicators and aggregates them into a BULLISH/BEARISH/NEUTRAL trend score.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Binance WS     │     │  Binance REST   │     │  Polymarket WS  │
│  (trades+klines)│     │  (orderbook)    │     │  (up/down prices)│
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │      State          │
                    │  - bids/asks/mid    │
                    │  - trades[]         │
                    │  - klines[]         │
                    │  - pm_up/pm_dn      │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Indicators Module  │
                    │  (11 calculations)  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Trend Scoring      │
                    │  (-9 to +9 range)   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Dashboard (Rich)   │
                    └─────────────────────┘
```

---

## Data Feeds

### Binance (Primary Market Data)
| Feed | Type | Purpose |
|------|------|---------|
| `wss://stream.binance.com/stream` | WebSocket | Trades + klines (candles) |
| `api.binance.com/api/v3/depth` | REST (2s poll) | Orderbook (20 levels) |
| `api.binance.com/api/v3/klines` | REST (bootstrap) | Historical candles |

### Polymarket (Prediction Market Overlay)
| Feed | Type | Purpose |
|------|------|---------|
| `gamma-api.polymarket.com/events` | REST | Resolve market slug → token IDs |
| `wss://ws-subscriptions-clob.polymarket.com/ws/market` | WebSocket | Live up/down prices |

**Slug construction logic** (critical for mapping):
- 5m: `{coin}-updown-5m-{truncated_ts}`
- 15m: `{coin}-updown-15m-{truncated_ts}`
- 4h: `{coin}-updown-4h-{truncated_ts}`
- 1h: `{full-coin-name}-up-or-down-{month}-{day}-{12h-hour}-et`
- daily: `{full-coin-name}-up-or-down-on-{month}-{day}`

---

## Indicators (11 Total)

### Order Book (3 indicators)
| Indicator | Calculation | Parameters |
|-----------|-------------|------------|
| **OBI** (Order Book Imbalance) | `(bid_vol - ask_vol) / total_vol` within ±1% band | Band: 1% of mid, Threshold: ±10% for signal |
| **Buy/Sell Walls** | Levels with qty > 5× avg level qty | Multiplier: 5× |
| **Liquidity Depth** | Sum of (price × qty) within 0.1%/0.5%/1.0% bands | Bands: [0.1, 0.5, 1.0] |

### Flow & Volume (3 indicators)
| Indicator | Calculation | Parameters |
|-----------|-------------|------------|
| **CVD** (Cumulative Volume Delta) | Σ(qty × price × direction) over trailing window | Windows: 60s, 180s, 300s |
| **Delta** (1m) | CVD over 60s | Window: 60s |
| **Volume Profile** | Price distribution into 30 bins, POC = max volume price | Bins: 30, Display: 9 rows |

### Technical Analysis (5 indicators)
| Indicator | Calculation | Parameters |
|-----------|-------------|------------|
| **RSI(14)** | Standard RSI with smoothing | Period: 14, OB: 70, OS: 30 |
| **MACD** | EMA(12) - EMA(26), Signal = EMA(9) of MACD | Fast: 12, Slow: 26, Signal: 9 |
| **VWAP** | Σ(typical_price × volume) / Σ(volume) | Rolling from klines |
| **EMA Crossover** | EMA(5) vs EMA(20) | Short: 5, Long: 20 |
| **Heikin Ashi** | Modified candlesticks for trend detection | Streak: 3 candles |

---

## Trend Scoring Algorithm

The system aggregates all indicators into a single score (-9 to +9) with threshold-based signals:

```python
def _score_trend(state):
    score = 0
    
    # OBI: +1 if > 10%, -1 if < -10%
    if obi > OBI_THRESH: score += 1
    elif obi < -OBI_THRESH: score -= 1
    
    # CVD 5m: +1 if positive, -1 if negative
    score += 1 if cvd_5m > 0 else -1 if cvd_5m < 0 else 0
    
    # RSI: -1 if > 70 (overbought), +1 if < 30 (oversold)
    if rsi > RSI_OB: score -= 1
    elif rsi < RSI_OS: score += 1
    
    # MACD histogram: +1 if positive, -1 if negative
    score += 1 if macd_hist > 0 else -1
    
    # VWAP: +1 if price above, -1 if below
    score += 1 if mid > vwap else -1
    
    # EMA: +1 if EMA5 > EMA20, -1 if below
    score += 1 if ema_5 > ema_20 else -1
    
    # Walls: +min(buy_walls, 2), -min(sell_walls, 2)
    score += min(len(buy_walls), 2)
    score -= min(len(sell_walls), 2)
    
    # Heikin Ashi: +1 if last 3 all green, -1 if all red
    if all_green_last_3: score += 1
    elif all_red_last_3: score -= 1
    
    # Signal thresholds
    if score >= 3: return "BULLISH"
    elif score <= -3: return "BEARISH"
    else: return "NEUTRAL"
```

**Key insight:** The scoring is unweighted—all indicators contribute equally (±1 or ±2 for walls). This simplicity is likely intentional for interpretability.

---

## Reusability Assessment

### Direct Port Candidates (High Value)

#### 1. **CVD (Cumulative Volume Delta)** ⭐ HIGHEST PRIORITY
- **Why:** Best leading indicator for short-term price direction on 5M markets
- **Implementation:** Simple running sum with time-windowed decay
- **Backtestable:** Yes—can replay trade streams
- **Effort:** Low (~20 lines)

#### 2. **OBI (Order Book Imbalance)** ⭐ HIGH PRIORITY
- **Why:** Strong predictor of immediate price pressure
- **Implementation:** Band-filtered bid/ask volume ratio
- **Backtestable:** Partially—requires orderbook snapshots (we have via CLOB)
- **Effort:** Low (~15 lines)

#### 3. **Volume Profile + POC** ⭐ HIGH PRIORITY
- **Why:** Identifies key support/resistance levels (POC = most traded price)
- **Implementation:** Histogram binning of volume by price
- **Backtestable:** Yes—from trade history or klines
- **Effort:** Medium (~40 lines)

### Partial Port Candidates (Medium Value)

#### 4. **Heikin Ashi Streak**
- **Why:** Good for filtering noise in trend detection
- **Caveat:** Requires reliable kline data (we have this)
- **Effort:** Low (~25 lines)

#### 5. **EMA Crossover (5/20)**
- **Why:** Classic trend confirmation
- **Caveat:** Lagging indicator, less useful for 5M resolution
- **Effort:** Very Low (~10 lines with existing EMA util)

### Skip / Low Priority

| Indicator | Reason |
|-----------|--------|
| MACD | Too lagging for 5M markets |
| RSI | Already commonly available; slow for 5M |
| VWAP | Requires intraday trade data we may not persist |
| Buy/Sell Walls | Very noisy; thresholds are arbitrary |
| Liquidity Depth | More useful for execution than prediction |

---

## Integration Strategy

### Option A: Module Import (Recommended)

Create `src/polymarket/indicators/st1ne.py` with pure functions:

```python
# src/polymarket/indicators/st1ne.py
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TradeTick:
    timestamp: float
    price: float
    qty: float
    is_buy: bool

@dataclass  
class Kline:
    open: float
    high: float
    low: float
    close: float
    volume: float

def cumulative_volume_delta(
    trades: List[TradeTick], 
    window_seconds: float
) -> float: ...

def order_book_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    mid: float,
    band_pct: float = 1.0
) -> float: ...

def volume_profile(
    klines: List[Kline],
    bins: int = 30
) -> Tuple[float, List[Tuple[float, float]]]:  # (poc, profile)
    ...
```

**Pros:** Clean separation, testable, no deps on st1ne code  
**Cons:** Need to maintain ourselves

### Option B: Git Submodule

Add st1ne repo as vendor dependency:

```bash
git submodule add https://github.com/st1ne/polymarket-assistant.git vendor/st1ne
```

**Pros:** Gets updates automatically  
**Cons:** Repo may change/break; not packaged for import; GPL risk unknown

### Decision: Option A

Implement key indicators directly—cleaner dependencies, better testing, easier to adapt for Polymarket-specific data shapes.

---

## Backtesting on 5M Markets

### Data Requirements

| Indicator | Required Data | Source |
|-----------|---------------|--------|
| CVD | Trade stream (price, qty, side, ts) | CLOB trades endpoint |
| OBI | Orderbook snapshots (bids, asks) | CLOB book endpoint |
| Volume Profile | Klines (OHLCV) or trades | CLOB trades aggregated |

### Recommended Approach

1. **Collector Enhancement**: Extend existing collector to capture:
   - Periodic orderbook snapshots (every 5s)
   - Trade stream via WebSocket
   - Kline aggregation from trades

2. **Backtest Framework**:
   ```python
   # Replay captured data
   for timestamp in range(start, end, step):
       state = load_state_at(timestamp)
       cvd = indicators.cvd(state.trades, window=300)
       obi = indicators.obi(state.bids, state.asks, state.mid)
       signal = scoring.evaluate(cvd, obi, ...)
       
       # Evaluate against actual market outcome
       pnl = simulate_trade(signal, next_period_data)
   ```

3. **Validation Metrics**:
   - Win rate by signal strength (|score| >= 3 vs >= 5)
   - Sharpe ratio of signal-based strategy
   - Drawdown analysis

---

## Implementation Phases

### Phase 1: Core Indicators (This Week)
- [ ] Implement `cvd()` with configurable windows
- [ ] Implement `obi()` with band parameter
- [ ] Unit tests with synthetic data

### Phase 2: Volume Analysis (Next Week)
- [ ] Implement `volume_profile()` with POC
- [ ] Add Heikin Ashi utility
- [ ] Integration with collector for live data

### Phase 3: Scoring & Backtest (Following Week)
- [ ] Port trend scoring algorithm
- [ ] Build backtest harness
- [ ] Validate on 1 week of 5M market data

---

## Open Questions

1. **Polymarket CLOB trade data**: Do we have historical trade streams, or only current book?
2. **Signal latency**: st1ne's system is ~2-3s delayed by REST polling. Can we improve with WS?
3. **Market coverage**: st1ne supports BTC/ETH/SOL/XRP. Our 5M markets—same coverage?
4. **Scoring weights**: Should we keep equal weighting or optimize weights via backtest?

---

## Code Samples from Reference

### CVD Calculation
```python
def cvd(trades, secs):
    cut = time.time() - secs
    return sum(
        t["qty"] * t["price"] * (1 if t["is_buy"] else -1)
        for t in trades
        if t["t"] >= cut
    )
```

### OBI Calculation
```python
def obi(bids, asks, mid):
    band = mid * config.OBI_BAND_PCT / 100
    bv = sum(q for p, q in bids if p >= mid - band)
    av = sum(q for p, q in asks if p <= mid + band)
    tot = bv + av
    return (bv - av) / tot if tot else 0.0
```

### Volume Profile
```python
def vol_profile(klines):
    lo = min(k["l"] for k in klines)
    hi = max(k["h"] for k in klines)
    if hi == lo:
        return lo, [(lo, sum(k["v"] for k in klines))]
    
    n = config.VP_BINS
    bsz = (hi - lo) / n
    bins = [0.0] * n
    
    for k in klines:
        b_lo = max(0, int((k["l"] - lo) / bsz))
        b_hi = min(n - 1, int((k["h"] - lo) / bsz))
        share = k["v"] / max(1, b_hi - b_lo + 1)
        for b in range(b_lo, b_hi + 1):
            bins[b] += share
    
    poci = bins.index(max(bins))
    poc = lo + (poci + 0.5) * bsz
    return poc, [(lo + (i + 0.5) * bsz, bins[i]) for i in range(n)]
```

---

## Summary

**Recommendation:** Port CVD, OBI, and Volume Profile as standalone indicators. These three provide the highest signal-to-noise ratio for 5M prediction markets. The trend scoring system is worth replicating after validating individual indicator performance.

**Estimated LOE:** 
- Phase 1: 2-3 hours
- Phase 2: 3-4 hours  
- Phase 3: 4-6 hours
- Total: ~2 days of focused work
