from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import clob
from .microstructure import (
    DEFAULT_CONSISTENCY_THRESHOLD,
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_EXTREME_PIN_THRESHOLD,
    DEFAULT_SPREAD_ALERT_THRESHOLD,
    DEFAULT_TIGHT_SPREAD_THRESHOLD,
    analyze_snapshot_microstructure,
    generate_microstructure_summary,
    log_microstructure_alerts,
)

logger = logging.getLogger(__name__)


def resolve_snapshot_path(path: Path) -> Path:
    """Resolve a snapshot path, auto-dereferencing pointer files.

    Pointer files (e.g., data/latest_15m.json) contain {"path": "...", "generated_at": "..."}.
    If the file is a pointer, return the path it points to. Otherwise return the original path.

    Args:
        path: Path to snapshot file or pointer file

    Returns:
        Resolved path to the actual snapshot file
    """
    if not path.exists():
        return path

    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "path" in data:
            resolved = Path(data["path"])
            if resolved.exists():
                return resolved
    except (json.JSONDecodeError, OSError):
        pass

    return path


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_markets_5m(args: argparse.Namespace) -> None:
    from .site import extract_5m_markets, fetch_predictions_page, parse_next_data

    html = fetch_predictions_page("5M")
    data = parse_next_data(html)
    ms = extract_5m_markets(data)
    out = [m.__dict__ for m in ms[: args.limit]]
    _print({"count": len(out), "markets": out})


def cmd_markets_15m(args: argparse.Namespace) -> None:
    from .site import (
        extract_crypto_interval_events,
        fetch_crypto_interval_page,
        parse_next_data,
    )

    html = fetch_crypto_interval_page("15M")
    data = parse_next_data(html)
    ms = extract_crypto_interval_events(data, interval_slug="15M")
    out = [m.__dict__ for m in ms[: args.limit]]
    _print({"count": len(out), "markets": out})


def cmd_book(args: argparse.Namespace) -> None:
    book = clob.get_book(args.token_id)
    _print(book)


def cmd_price(args: argparse.Namespace) -> None:
    price = clob.get_price(args.token_id, side=args.side)
    _print(price)


def cmd_collect_5m(args: argparse.Namespace) -> None:
    from .collector import collect_5m_snapshot
    from .collector_loop import collect_5m_loop

    out_dir = Path(args.out)

    if args.every_seconds is not None:
        collect_5m_loop(
            out_dir=out_dir,
            interval_seconds=float(args.every_seconds),
            max_backoff_seconds=float(args.max_backoff_seconds),
            retention_hours=args.retention_hours,
        )
    else:
        out = collect_5m_snapshot(out_dir)
        print(str(out))


def cmd_collect_15m(args: argparse.Namespace) -> None:
    from .collector import collect_15m_snapshot

    out = collect_15m_snapshot(Path(args.out))
    print(str(out))


def cmd_universe_5m(args: argparse.Namespace) -> None:
    from .universe import build_universe, save_universe

    data = build_universe(cross_check=args.cross_check)
    out_path = save_universe(data, Path(args.out))
    print(str(out_path))


def cmd_collect_15m_loop(args: argparse.Namespace) -> None:
    from .collector_loop import collect_15m_loop

    collect_15m_loop(
        out_dir=Path(args.out),
        interval_seconds=float(args.interval_seconds),
        max_backoff_seconds=float(args.max_backoff_seconds),
        retention_hours=args.retention_hours,
        max_snapshots=args.max_snapshots,
        microstructure_interval_seconds=float(args.microstructure_interval_seconds),
        microstructure_target=args.microstructure_target,
        spread_alert_threshold=float(args.spread_alert_threshold),
        extreme_pin_threshold=float(args.extreme_pin_threshold),
        depth_levels=int(args.depth_levels),
    )


def cmd_pnl_verify(args: argparse.Namespace) -> None:
    """Verify PnL from fills data with cash tracking and sanity checks."""
    from decimal import Decimal

    from .pnl import (
        PnLVerifier,
        load_fills_from_file,
        load_orderbooks_from_file,
        load_orderbooks_from_snapshot,
        save_daily_summary,
    )

    # Determine fills source
    fills: list = []

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(
                json.dumps({"error": f"Input file not found: {args.input}"}),
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)
        fills = load_fills_from_file(input_path)
    elif args.data_dir:
        # Load from data directory - look for fills.json or fills.jsonl
        data_dir = Path(args.data_dir)
        fills_path = data_dir / "fills.json"
        fills_jsonl_path = data_dir / "fills.jsonl"

        if fills_path.exists():
            fills = load_fills_from_file(fills_path)
        elif fills_jsonl_path.exists():
            fills = load_fills_from_file(fills_jsonl_path)
        else:
            print(
                json.dumps({"error": f"No fills.json or fills.jsonl found in {args.data_dir}"}),
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)
    else:
        print(
            json.dumps({"error": "Must specify --input or --data-dir"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    # Load optional orderbooks for liquidation value
    orderbooks = None
    if args.books:
        books_path = Path(args.books)
        if books_path.exists():
            orderbooks = load_orderbooks_from_file(books_path)
    elif args.snapshot:
        snapshot_path = resolve_snapshot_path(Path(args.snapshot))
        if snapshot_path.exists():
            orderbooks = load_orderbooks_from_snapshot(snapshot_path)
    elif args.data_dir:
        # Try to load latest snapshot from data directory
        data_dir = Path(args.data_dir)
        snapshot_files = sorted(data_dir.glob("snapshot_*.json"))
        if snapshot_files:
            orderbooks = load_orderbooks_from_snapshot(snapshot_files[-1])

    # Build verifier
    starting_cash = Decimal(str(args.starting_cash)) if args.starting_cash else Decimal("0")
    verifier = PnLVerifier(starting_cash=starting_cash)
    verifier.add_fills(fills)

    # Compute report
    report = verifier.compute_pnl(
        orderbooks=orderbooks,
        since=args.since,
        market_filter=args.market,
    )

    # Save daily summary if requested
    if args.save_daily:
        summary_path = save_daily_summary(
            report,
            out_dir=Path(args.daily_dir) if args.daily_dir else None,
        )
        if args.format == "human":
            print(f"\nDaily summary saved: {summary_path}")

    # Output format
    if args.format == "json":
        print(report.to_json())
    else:
        # Human-readable format
        output = report.to_dict()

        print("=" * 70)
        print("PnL VERIFICATION REPORT")
        print("=" * 70)

        # Metadata
        if output["metadata"]["since"]:
            print(f"Since:            {output['metadata']['since']}")
        if output["metadata"]["market_filter"]:
            print(f"Market Filter:    {output['metadata']['market_filter']}")
        print(f"Generated:        {output['metadata']['generated_at']}")

        print("\n--- Summary ---")
        print(f"Total Fills:      {output['summary']['total_fills']}")
        print(f"Unique Tokens:    {output['summary']['unique_tokens']}")
        print(f"Unique Markets:   {output['summary']['unique_markets']}")

        print("\n--- Cash Tracking ---")
        print(f"Starting Cash:    ${output['cash']['starting_cash']:,.2f}")
        print(f"Cash Flow:        ${output['cash']['cash_flow_from_fills']:,.2f}")
        print(f"Ending Cash:      ${output['cash']['ending_cash']:,.2f}")

        print("\n--- PnL Breakdown ---")
        print(f"Realized PnL:     ${output['pnl']['realized_pnl']:,.2f}")
        print(f"Unrealized PnL:   ${output['pnl']['unrealized_pnl']:,.2f}")
        print(f"Total Fees:       ${output['pnl']['total_fees']:,.2f}")
        print(f"Net PnL:          ${output['pnl']['net_pnl']:,.2f}")

        print("\n--- Liquidation Analysis ---")
        print(f"Mark to Mid:      ${output['liquidation']['mark_to_mid']:,.2f}")
        print(f"Liquidation Val:  ${output['liquidation']['liquidation_value']:,.2f}")
        print(
            f"Discount:         ${output['liquidation']['liquidation_discount']:,.2f} "
            f"({output['liquidation']['discount_pct']:.1f}%)"
        )

        print("\n--- Verification ---")
        cash_ok = "✓" if output["verification"]["cashflow_conserved"] else "✗"
        pos_ok = "✓" if output["verification"]["position_verified"] else "✗"
        print(f"Cashflow:         {cash_ok} Conserved")
        print(f"Positions:        {pos_ok} Verified")

        if output["positions"]:
            print(f"\n--- Open Positions ({len(output['positions'])}) ---")
            for pos in output["positions"][:10]:  # Limit to 10
                slug = pos.get("market_slug", "")
                token_display = slug if slug else pos["token_id"][:20] + "..."
                print(f"\n  {token_display}")
                print(f"    Size: {pos['net_size']:,.2f} @ ${pos['avg_cost_basis']:.3f}")
                print(
                    f"    Mid:  ${pos['current_price']:.3f} | "
                    f"Unrealized: ${pos['unrealized_pnl']:,.2f}"
                )
                if pos.get("liquidation_value") != pos.get("mark_to_mid"):
                    print(
                        f"    Liquid: ${pos['liquidation_value']:,.2f} "
                        f"(discount: ${pos['mark_to_mid'] - pos['liquidation_value']:,.2f})"
                    )

        if output["verification"]["warnings"]:
            print(f"\n--- Warnings ({len(output['verification']['warnings'])}) ---")
            for warning in output["verification"]["warnings"]:
                print(f"  ! {warning}")

        print("\n" + "=" * 70)


def cmd_microstructure(args: argparse.Namespace) -> None:
    """Analyze microstructure for markets in a snapshot."""
    snapshot_path = resolve_snapshot_path(Path(args.snapshot))
    if not snapshot_path.exists():
        print(
            json.dumps({"error": f"Snapshot file not found: {args.snapshot}"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    # Load optional BTC context from aligned features file
    btc_context = None
    if args.btc_features:
        btc_features_path = Path(args.btc_features)
        if btc_features_path.exists():
            try:
                btc_context = _load_btc_context(btc_features_path, args.btc_horizon)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load BTC context from %s: %s", btc_features_path, e)

    if args.summary:
        # Generate summary with alerts
        summary = generate_microstructure_summary(
            snapshot_path=snapshot_path,
            target_market_substring=args.target,
            spread_threshold=float(args.spread_threshold),
            extreme_pin_threshold=float(args.extreme_pin_threshold),
            depth_levels=int(args.depth_levels),
            tight_spread_threshold=float(args.tight_spread_threshold),
            consistency_threshold=float(args.consistency_threshold),
            btc_context=btc_context,
        )

        if args.format == "json":
            print(json.dumps(summary, indent=2))
        else:
            # Human-readable format
            print("=" * 70)
            print("MARKET MICROSTRUCTURE SUMMARY")
            print("=" * 70)
            print(f"Generated:      {summary['generated_at']}")
            print(f"Snapshot:       {summary['snapshot_path']}")
            print(f"Markets:        {summary['markets_analyzed']}")
            print("\nThresholds:")
            print(f"  Spread alert:       > {summary['spread_threshold']:.2f}")
            print(
                f"  Extreme pin:        <= {summary['extreme_pin_threshold']:.2f} "
                f"or >= {1.0 - summary['extreme_pin_threshold']:.2f}"
            )
            print(f"  Tight spread:       <= {summary['tight_spread_threshold']:.2f}")
            print(f"  Consistency:        <= {summary['consistency_threshold']:.2f}")
            print(f"  Depth levels:       {summary['depth_levels']}")

            if summary["alerts"]:
                print(f"\n--- ALERTS ({summary['alert_count']}) ---")
                for alert in summary["alerts"]:
                    print(f"  ⚠ {alert}")
            else:
                print("\n--- ALERTS ---")
                print("  ✓ No alerts. Markets appear healthy.")

            if summary["market_summaries"]:
                print(f"\n--- MARKET DETAILS ({len(summary['market_summaries'])}) ---")
                for ms in summary["market_summaries"]:
                    print(f"\n  {ms['market_title']}")
                    print(
                        f"    YES: bid={ms.get('yes_best_bid'):.2f} ask={ms.get('yes_best_ask'):.2f} "
                        f"spread={ms.get('yes_spread'):.2f} imbalance={ms.get('yes_imbalance', 0):+.3f}"
                    )
                    print(
                        f"    NO:  bid={ms.get('no_best_bid'):.2f} ask={ms.get('no_best_ask'):.2f} "
                        f"spread={ms.get('no_spread'):.2f} imbalance={ms.get('no_imbalance', 0):+.3f}"
                    )
                    if ms.get("implied_sum"):
                        print(
                            f"    Implied: YES_mid={ms.get('implied_yes_mid', 0):.2f} "
                            f"NO_mid={ms.get('implied_no_mid', 0):.2f} "
                            f"sum={ms.get('implied_sum', 0):.2f} "
                            f"consistency={ms.get('consistency_diff', 0):.3f}"
                        )

            print("\n" + "=" * 70)

        # Log alerts to stderr as well
        log_microstructure_alerts(summary)
    else:
        # Raw analysis output
        analyses = analyze_snapshot_microstructure(
            snapshot_path=snapshot_path,
            target_market_substring=args.target,
            depth_levels=int(args.depth_levels),
        )
        print(json.dumps(analyses, indent=2))


def _load_btc_context(features_path: Path, horizon: str) -> dict[str, Any] | None:
    """Load BTC context from aligned features file.

    Args:
        features_path: Path to aligned_features.json
        horizon: Which return horizon to use (e.g., '5m', '1h', '24h')

    Returns:
        Dict with BTC returns for various horizons, or None if not found.
    """
    data = json.loads(features_path.read_text())
    if not data or not isinstance(data, list):
        return None

    # Get the most recent record
    latest = data[-1]
    binance_features = latest.get("binance_features", {})
    returns = binance_features.get("returns", [])

    context = {}
    for ret in returns:
        h = ret.get("horizon_seconds", 0)
        simple_ret = ret.get("simple_return")
        if simple_ret is None:
            continue
        # Map horizon seconds to key names
        if h == 300:  # 5 minutes
            context["return_5m"] = simple_ret
        elif h == 3600:  # 1 hour
            context["return_1h"] = simple_ret
        elif h == 86400:  # 24 hours
            context["return_24h"] = simple_ret

    return context if context else None


def cmd_binance_collect(args: argparse.Namespace) -> None:
    """Collect Binance BTC market data (REST API single snapshot)."""
    from pathlib import Path

    from .binance_collector import collect_snapshot_rest

    out_dir = Path(args.out)
    base_urls = args.base_url if args.base_url else None
    out_path = collect_snapshot_rest(
        out_dir=out_dir,
        symbol=args.symbol,
        kline_intervals=args.intervals,
        base_urls=base_urls,
    )
    print(str(out_path))


def cmd_binance_loop(args: argparse.Namespace) -> None:
    """Run Binance WebSocket collector loop."""
    from pathlib import Path

    from .binance_collector import run_collector_loop

    ws_bases = args.ws_base if args.ws_base else None
    run_collector_loop(
        out_dir=Path(args.out),
        symbol=args.symbol,
        kline_intervals=args.intervals,
        snapshot_interval_seconds=float(args.snapshot_interval_seconds),
        max_reconnect_delay=float(args.max_reconnect_delay),
        retention_hours=args.retention_hours,
        ws_bases=ws_bases,
    )


def cmd_binance_features(args: argparse.Namespace) -> None:
    """Build features from Binance data and align to Polymarket snapshots."""
    from pathlib import Path

    from .binance_features import (
        align_to_polymarket_snapshots,
        save_aligned_features,
    )

    binance_dir = Path(args.binance_dir)
    polymarket_dir = Path(args.polymarket_dir)
    out_path = Path(args.out)

    aligned = align_to_polymarket_snapshots(
        binance_data_dir=binance_dir,
        polymarket_data_dir=polymarket_dir,
        tolerance_seconds=float(args.tolerance),
    )

    save_aligned_features(aligned, out_path)
    print(f"Aligned {len(aligned)} records to {out_path}")


def cmd_health_check(args: argparse.Namespace) -> None:
    """Check collector health and staleness SLA."""
    from pathlib import Path

    from .collector_loop import check_staleness_sla

    out_dir = Path(args.data_dir)
    result = check_staleness_sla(
        out_dir=out_dir,
        max_age_seconds=float(args.max_age_seconds),
        prefix=args.prefix,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        status = "✓ HEALTHY" if result["healthy"] else "✗ UNHEALTHY"
        print(f"{status}: {result['message']}")
        if result["age_seconds"] is not None:
            print(f"  Age: {result['age_seconds']:.1f}s (max: {result['max_age_seconds']:.1f}s)")

    # Exit with error code if unhealthy and --fail is set
    if args.fail and not result["healthy"]:
        raise SystemExit(1)


def cmd_weather_forecast(args: argparse.Namespace) -> None:
    """Fetch weather forecasts for specified cities."""
    from datetime import date, timedelta

    from .weather import get_consensus_for_cities

    cities = args.cities.split(",") if args.cities else ["nyc", "london"]
    target_date = date.fromisoformat(args.date) if args.date else date.today() + timedelta(days=1)

    consensus = get_consensus_for_cities(
        cities=cities,
        target_date=target_date,
        min_models=args.min_models,
    )

    if args.format == "json":
        print(json.dumps({k: v.to_dict() for k, v in consensus.items()}, indent=2))
    else:
        print("=" * 70)
        print("WEATHER FORECAST CONSENSUS")
        print("=" * 70)
        print(f"Target date: {target_date}")
        print(f"Cities: {', '.join(cities)}")
        print()

        for city, cons in consensus.items():
            print(f"\n{city.upper()}")
            print(f"  Consensus High: {cons.consensus_high:.1f}°F")
            print(f"  Consensus Low:  {cons.consensus_low:.1f}°F")
            print(f"  Models used:    {cons.model_count}")
            print(f"  Agreement:      {cons.agreement_score:.2f}")
            print("  Model breakdown:")
            for model in cons.models:
                print(
                    f"    {model.model:12} (via {model.source}): "
                    f"high={model.temp_high:.1f}°F, low={model.temp_low:.1f}°F"
                )

        print("\n" + "=" * 70)


def cmd_weather_scan(args: argparse.Namespace) -> None:
    """Scan for weather market arbitrage opportunities."""
    from pathlib import Path

    from .strategy_weather import run_weather_scan

    snapshots_dir = Path(args.snapshots_dir) if args.snapshots_dir else None
    cities = args.cities.split(",") if args.cities else None

    result = run_weather_scan(
        snapshots_dir=snapshots_dir,
        cities=cities,
        dry_run=not args.live,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("WEATHER ARBITRAGE SCAN RESULTS")
        print("=" * 70)
        print(f"Scan time: {result['timestamp']}")
        print(f"Markets scanned: {result['markets_scanned']}")
        print(f"Signals generated: {result['signals_generated']}")
        print(f"Actionable signals: {result['actionable_signals']}")
        print(f"Trades executed: {result['trades_executed']}")
        print(f"Dry run: {result['dry_run']}")

        # Show consensus
        if result["consensus"]:
            print("\n--- Model Consensus ---")
            for city, cons in result["consensus"].items():
                print(
                    f"  {city}: high={cons['consensus_high']:.1f}°F, models={cons['model_count']}"
                )

        # Show signals
        if result["signals"]:
            print("\n--- All Signals ---")
            for sig in result["signals"]:
                market_q = (
                    sig["market"]["question"][:50] if sig["market"]["question"] else "Unknown"
                )
                print(
                    f"  {sig['side']:<12} | edge={sig['edge']:+.2f} | "
                    f"EV={sig['expected_value']:.3f} | {market_q}..."
                )

        # Show trades
        if result["trades"]:
            print("\n--- Executed Trades ---")
            for trade in result["trades"]:
                sig = trade["signal"]
                print(
                    f"  {sig['side']:<12} | size={trade['position_size']:.2f} | "
                    f"price={trade['entry_price']:.3f} | "
                    f"EV={sig['expected_value']:.3f}"
                )

        print("\n" + "=" * 70)


def cmd_cross_market_scan(args: argparse.Namespace) -> None:
    """Run cross-market arbitrage scan across Polymarket and Kalshi."""
    from pathlib import Path

    from .cross_market.strategy import CrossMarketArbitrage

    data_dir = Path(args.out) if args.out else None

    with CrossMarketArbitrage(
        min_gross_spread=float(args.min_gross_spread),
        min_net_spread=float(args.min_net_spread),
        max_positions=int(args.max_positions),
        position_size=float(args.position_size),
        data_dir=data_dir,
    ) as strategy:
        result = strategy.scan(
            categories=args.categories.split(",") if args.categories else None,
            dry_run=not args.live,
        )

        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("=" * 70)
            print("CROSS-MARKET ARBITRAGE SCAN")
            print("=" * 70)
            print(f"Scan time:       {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"Polymarket:      {result.polymarket_markets} markets")
            print(f"Kalshi:          {result.kalshi_markets} markets")
            print(f"Matched events:  {result.matched_events}")
            print(f"Opportunities:   {result.opportunities}")
            print(f"Trades entered:  {result.trades_entered}")
            print(f"Mode:            {'LIVE' if not result.dry_run else 'DRY RUN'}")

            if result.opportunities_list:
                print("\n--- Top Opportunities ---")
                for opp in result.opportunities_list[:10]:
                    print(
                        f"  {opp.venue_yes} YES @ {opp.yes_price:.3f} + "
                        f"{opp.venue_no} NO @ {opp.no_price:.3f} | "
                        f"Gross: {opp.gross_spread * 100:.2f}% | "
                        f"Net: {opp.net_spread * 100:.2f}% | "
                        f"Conf: {opp.confidence:.2f}"
                    )

            # Show performance summary
            summary = strategy.get_performance_report()
            if summary["total_trades"] > 0:
                print("\n--- Performance Summary ---")
                print(f"Total trades:    {summary['total_trades']}")
                print(f"Open positions:  {summary['open_positions']}")
                print(f"Realized PnL:    ${summary['total_realized_pnl']:.2f}")
                print(f"Theoretical PnL: ${summary['total_theoretical_pnl']:.2f}")

            print("=" * 70)


def cmd_cross_market_report(args: argparse.Namespace) -> None:
    """Generate cross-market arbitrage performance report."""
    from pathlib import Path

    from .cross_market.tracker import PaperTradeTracker

    data_dir = Path(args.data_dir) if args.data_dir else None
    tracker = PaperTradeTracker(data_dir=data_dir)

    summary = tracker.get_performance_summary()

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print("=" * 70)
        print("CROSS-MARKET ARBITRAGE PERFORMANCE REPORT")
        print("=" * 70)
        print(f"Total trades:         {summary['total_trades']}")
        print(f"Open positions:       {summary['open_positions']}")
        print(f"Closed positions:     {summary['closed_positions']}")
        print(f"Win rate:             {summary['win_rate'] * 100:.1f}%")
        print(f"\nTotal realized PnL:   ${summary['total_realized_pnl']:.2f}")
        print(f"Total theoretical:    ${summary['total_theoretical_pnl']:.2f}")
        print(f"Combined PnL:         ${summary['total_pnl']:.2f}")
        print(f"\nAvg realized PnL:     ${summary['avg_realized_pnl']:.3f}")
        print(f"Avg spread captured:  {summary['avg_spread_captured'] * 100:.2f}%")

        status = summary.get("trades_by_status", {})
        if status:
            print("\n--- By Status ---")
            print(f"Open:                 {status.get('open', 0)}")
            print(f"Closed early:         {status.get('closed', 0)}")
            print(f"Held to resolution:   {status.get('held_to_resolution', 0)}")

        print("=" * 70)


def cmd_both_sides_scan(args: argparse.Namespace) -> None:
    """Scan for both-sides mispricing arbitrage opportunities."""
    from decimal import Decimal

    from .btc_both_sides_arb import BothSidesArbitrageStrategy

    strategy = BothSidesArbitrageStrategy(
        position_size=Decimal(str(args.position_size)),
        check_alignment=args.check_alignment,
    )

    opportunities = strategy.scan(
        interval=args.interval,
        min_spread=Decimal(str(args.min_spread)),
    )

    if args.format == "json":
        print(json.dumps([opp.to_dict() for opp in opportunities], indent=2))
    else:
        print("=" * 80)
        print("BOTH-SIDES MISPRICING ARBITRAGE SCAN")
        print("=" * 80)
        print(f"Interval: {args.interval}")
        print(f"Check 15m alignment: {args.check_alignment}")
        print(f"Min spread: {args.min_spread * 100:.1f}%")
        print(f"Opportunities found: {len(opportunities)}")
        print()

        for i, opp in enumerate(opportunities[: args.limit], 1):
            print(f"--- Opportunity {i} ---")
            print(f"Market: {opp.market_metadata.get('title', 'N/A')}")
            print(f"Interval: {opp.interval}")
            print(f"UP price:   ${float(opp.up_price):.4f}")
            print(f"DOWN price: ${float(opp.down_price):.4f}")
            print(f"Sum:        ${float(opp.price_sum):.4f}")
            print(f"Spread:     {float(opp.spread) * 100:.2f}%")
            print(f"After fees: {float(opp.spread_after_fees) * 100:.2f}%")
            if opp.aligned_15m is not None:
                print(f"15m aligned: {'Yes' if opp.aligned_15m else 'No'}")
            print(f"Confidence: {float(opp.confidence) * 100:.1f}%")
            print()

        print("=" * 80)


def cmd_both_sides_trade(args: argparse.Namespace) -> None:
    """Execute paper trades for both-sides arbitrage opportunities."""
    from decimal import Decimal

    from .btc_both_sides_arb import BothSidesArbitrageStrategy

    strategy = BothSidesArbitrageStrategy(
        position_size=Decimal(str(args.position_size)),
        check_alignment=args.check_alignment,
    )

    if args.scan:
        # Scan and trade all valid opportunities
        opportunities = strategy.scan(
            interval=args.interval,
            min_spread=Decimal(str(args.min_spread)),
        )
        trades = []
        for opp in opportunities:
            if opp.is_valid:
                trade = strategy.paper_trade(opp)
                trades.append(trade)

        if args.format == "json":
            print(json.dumps([t.to_dict() for t in trades], indent=2))
        else:
            print("=" * 80)
            print("BOTH-SIDES ARBITRAGE PAPER TRADES")
            print("=" * 80)
            print(f"Opportunities found: {len(opportunities)}")
            print(f"Trades executed: {len(trades)}")
            print()

            for t in trades:
                print(f"Trade: {t.trade_id}")
                print(f"  Market: {t.market_id}")
                print(f"  Interval: {t.interval}")
                print(f"  UP entry:   ${float(t.up_entry_price):.4f}")
                print(f"  DOWN entry: ${float(t.down_entry_price):.4f}")
                print(f"  Position size: ${float(t.position_size):.2f} per side")
                print(f"  Total cost: ${float(t.total_cost):.4f}")
                print(f"  Spread captured: {float(t.spread_at_entry) * 100:.2f}%")
                if t.aligned_15m is not None:
                    print(f"  15m aligned: {'Yes' if t.aligned_15m else 'No'}")
                print()

            print("=" * 80)


def cmd_both_sides_stats(args: argparse.Namespace) -> None:
    """Show both-sides arbitrage strategy statistics."""
    from decimal import Decimal

    from .btc_both_sides_arb import BothSidesArbitrageStrategy

    strategy = BothSidesArbitrageStrategy(
        position_size=Decimal(str(args.position_size)),
    )

    stats = strategy.get_stats()

    if args.format == "json":
        # Convert Decimals to strings for JSON serialization
        json_stats = {k: str(v) if isinstance(v, Decimal) else v for k, v in stats.items()}
        print(json.dumps(json_stats, indent=2))
    else:
        print("=" * 80)
        print("BOTH-SIDES ARBITRAGE STATISTICS")
        print("=" * 80)
        print(f"Total opportunities detected: {stats['total_opportunities']}")
        print(f"Total trades: {stats['total_trades']}")
        print(f"Open trades: {stats['open_trades']}")
        print(f"Closed trades: {stats['closed_trades']}")
        print(f"Total PnL: ${float(stats['total_pnl']):,.2f}")
        print(f"Average spread: {float(stats['avg_spread']) * 100:.2f}%")
        if stats["aligned_trades"] + stats["non_aligned_trades"] > 0:
            print(f"15m aligned trades: {stats['aligned_trades']}")
            print(f"Non-aligned trades: {stats['non_aligned_trades']}")
        print("=" * 80)


def cmd_combinatorial_scan(args: argparse.Namespace) -> None:
    """Scan for combinatorial arbitrage (Dutch book) opportunities."""
    from .combinatorial import run_combinatorial_scan

    result = run_combinatorial_scan(
        event_limit=args.event_limit,
        fee_rate=args.fee_rate,
        min_edge=args.min_edge,
        max_basket_size=args.max_basket_size,
        min_liquidity=args.min_liquidity,
        detailed=args.detailed,
    )

    if args.format == "json":
        print(json.dumps(result["result"], indent=2))
    else:
        print(result["report"])


def cmd_mention_scan(args: argparse.Namespace) -> None:
    """Scan for mention market opportunities with default-to-NO strategy."""
    from pathlib import Path

    from .strategy_mention import run_mention_scan

    snapshots_dir = Path(args.snapshots_dir) if args.snapshots_dir else None

    result = run_mention_scan(
        snapshots_dir=snapshots_dir,
        base_rate=args.base_rate,
        dry_run=not args.live,
        max_positions=args.max_positions,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("MENTION MARKET SCAN RESULTS")
        print("=" * 70)
        print(f"Scan time: {result['timestamp']}")
        print(f"Markets scanned: {result['markets_scanned']}")
        print(f"Signals generated: {result['signals_generated']}")
        print(f"Actionable signals: {result['actionable_signals']}")
        print(f"Trades executed: {result['trades_executed']}")
        print(f"Dry run: {result['dry_run']}")

        if result["summary"]:
            print("\n--- Summary ---")
            print(f"  Buy YES signals: {result['summary']['buy_yes_count']}")
            print(f"  Buy NO signals:  {result['summary']['buy_no_count']}")
            print(f"  No trade:        {result['summary']['no_trade_count']}")
            print(f"  Avg edge (NO):   {result['summary']['avg_edge_buy_no']:+.1%}")
            print(f"  Avg edge (YES):  {result['summary']['avg_edge_buy_yes']:+.1%}")

        if result["markets"]:
            print(f"\n--- Mention Markets Found ({len(result['markets'])}) ---")
            for m in result["markets"][:10]:  # Show first 10
                target = m["target"] or "Unknown"
                print(f"  {target:<20} | {m['yes_price']:.2f} | {m['question'][:40]}...")

        if result["signals"]:
            print("\n--- All Signals ---")
            for sig in result["signals"]:
                market_q = (
                    sig["market"]["question"][:40] if sig["market"]["question"] else "Unknown"
                )
                print(
                    f"  {sig['side']:<12} | edge={sig['edge']:+.2f} | "
                    f"market={sig['market_prob']:.2f} | theo={sig['theoretical_prob']:.2f} | {market_q}..."
                )

        if result["trades"]:
            print("\n--- Executed Trades ---")
            for trade in result["trades"]:
                sig = trade["signal"]
                print(
                    f"  {sig['side']:<12} | size={trade['position_size']:.2f} | "
                    f"price={trade['entry_price']:.3f} | "
                    f"EV={sig['expected_value']:.3f}"
                )

        print("\n" + "=" * 70)


def cmd_news_momentum_scan(args: argparse.Namespace) -> None:
    """Scan for news-driven momentum trading opportunities."""
    from pathlib import Path

    from .strategy_news_momentum import NewsItem, SourceReliability, run_news_momentum_scan

    snapshots_dir = Path(args.snapshots_dir) if args.snapshots_dir else None

    # Build news items from CLI arguments or use example
    if args.headline:
        news_items = [
            NewsItem(
                timestamp=datetime.now(UTC) - timedelta(seconds=args.seconds_ago),
                source=args.source or "CLI",
                source_reliability=SourceReliability[args.source_reliability],
                headline=args.headline,
                category=None,
            ),
        ]
    else:
        news_items = None  # Will use example data

    # Build config from CLI args
    config = {
        "max_time_since_news_seconds": args.max_time_seconds,
        "min_edge_for_entry": args.min_edge,
        "min_confidence": args.min_confidence,
        "base_position_size": args.base_position_size,
        "scaled_position_size": args.scaled_position_size,
        "max_position_size": args.max_position_size,
        "stop_loss_pct": args.stop_loss,
        "profit_target_pct": args.profit_target,
        "max_hold_hours": args.max_hold_hours,
        "momentum_exit_enabled": args.momentum_exit,
    }

    result = run_news_momentum_scan(
        news_items=news_items,
        snapshots_dir=snapshots_dir,
        capital=args.capital,
        dry_run=not args.live,
        max_positions=args.max_positions,
        config=config,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("NEWS-DRIVEN MOMENTUM SCAN RESULTS")
        print("=" * 70)
        print(f"Scan time: {result['timestamp']}")
        print(f"News items analyzed: {result['news_items_analyzed']}")
        print(f"Markets available: {result['markets_available']}")
        print(f"Signals generated: {result['signals_generated']}")
        print(f"Actionable signals: {result['actionable_signals']}")
        print(f"Trades executed: {result['trades_executed']}")
        print(f"Open positions: {result['open_positions']}")
        print(f"Dry run: {result['dry_run']}")

        if result["summary"]:
            print("\n--- Summary ---")
            print(f"  Buy YES signals: {result['summary']['buy_yes_count']}")
            print(f"  Buy NO signals:  {result['summary']['buy_no_count']}")
            print(f"  Avg edge:        {result['summary']['avg_edge']:+.2%}")
            print(f"  Avg confidence:  {result['summary']['avg_confidence']:.1%}")

        if result["signals"]:
            print("\n--- Top Signals ---")
            for sig in result["signals"][:10]:
                market_q = sig["market_question"][:40] if sig["market_question"] else "Unknown"
                print(
                    f"  {sig['side']:<12} | edge={sig['edge']:+.2f} | "
                    f"conf={sig['confidence']:.1%} | {market_q}..."
                )
                print(f"    Current: {sig['current_price']:.3f} -> Target: {sig['target_price']:.3f}")
                print(f"    Source: {sig['news_source']} | {sig['time_since_news_seconds']:.0f}s ago")

        if result["trades"]:
            print("\n--- Executed Trades ---")
            for trade in result["trades"]:
                sig = trade["signal"]
                print(
                    f"  {sig['side']:<12} | size={trade['position_size']:.2f} | "
                    f"price={trade['entry_price']:.3f} | pos_id={trade['position_id']}"
                )

        print("\n" + "=" * 70)


def cmd_news_momentum_positions(args: argparse.Namespace) -> None:
    """Show news-driven momentum positions and check for exits."""
    from pathlib import Path

    from .strategy_news_momentum import NewsMomentumTracker, check_position_exits

    data_dir = Path(args.data_dir) if args.data_dir else None
    tracker = NewsMomentumTracker(data_dir=data_dir)

    # Check for exits if snapshot provided
    exits = []
    if args.snapshot:
        snapshot_path = Path(args.snapshot)
        exits = check_position_exits(tracker, snapshot_path)

    summary = tracker.get_performance_summary()
    open_positions = tracker.get_open_positions()

    if args.format == "json":
        print(json.dumps({
            "summary": summary,
            "open_positions": [
                {
                    "position_id": p.position_id,
                    "market": p.market_question,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "position_size": p.position_size,
                }
                for p in open_positions
            ],
            "exits_today": exits,
        }, indent=2))
    else:
        print("=" * 70)
        print("NEWS-DRIVEN MOMENTUM POSITIONS")
        print("=" * 70)

        print("\n--- Performance Summary ---")
        print(f"Total trades:     {summary['total_trades']}")
        print(f"Win rate:         {summary['win_rate']:.1%}")
        print(f"Total PnL:        ${summary['total_pnl']:,.2f}")
        print(f"Avg hold time:    {summary['avg_hold_time_hours']:.1f}h")

        print(f"\n--- Open Positions ({len(open_positions)}) ---")
        for p in open_positions:
            print(f"\n  {p.position_id}")
            print(f"    Market: {p.market_question[:50]}...")
            print(f"    Side: {p.side} | Entry: {p.entry_price:.3f} | Size: {p.position_size:.2f}")
            if p.peak_price:
                print(f"    Peak: {p.peak_price:.3f}")

        if exits:
            print(f"\n--- Exits Triggered ({len(exits)}) ---")
            for exit_info in exits:
                print(f"  {exit_info['position_id']}: {exit_info['reason']} | PnL: ${exit_info['pnl']:.2f}")

        print("\n" + "=" * 70)


def cmd_watchdog(args: argparse.Namespace) -> None:
    """Run collector watchdog to ensure data freshness."""
    from pathlib import Path

    from .watchdog import run_watchdog

    result = run_watchdog(
        data_dir=Path(args.data_dir),
        max_age_seconds=float(args.max_age_seconds),
        dry_run=args.dry_run,
        script_path=Path(args.script) if args.script else None,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        status = "✓ FRESH" if result["fresh"] else "✗ STALE"
        print(f"{status}: {result['message']}")
        if result["age_seconds"] is not None:
            print(f"  Age: {result['age_seconds']:.1f}s (max: {args.max_age_seconds}s)")
        print(f"  Collector running: {result['collector_running']}")
        if result["collector_pid"]:
            print(f"  PID: {result['collector_pid']}")
        print(f"  Action: {result['action_taken']}")

    # Exit with error code if data is stale and --fail is set
    if args.fail and not result["fresh"]:
        raise SystemExit(1)


def cmd_paper_record_fill(args: argparse.Namespace) -> None:
    """Record a paper trade fill."""
    from decimal import Decimal
    from pathlib import Path

    from .paper_trading import PaperTradingEngine

    engine = PaperTradingEngine(
        data_dir=Path(args.data_dir),
        starting_cash=Decimal(str(args.starting_cash)),
    )

    fill = engine.record_fill(
        token_id=args.token_id,
        side=args.side,
        size=Decimal(str(args.size)),
        price=Decimal(str(args.price)),
        fee=Decimal(str(args.fee)) if args.fee else Decimal("0"),
        market_slug=args.market_slug,
        market_question=args.market_question,
    )

    if args.format == "json":
        print(
            json.dumps(
                {
                    "status": "recorded",
                    "fill": {
                        "token_id": fill.token_id,
                        "side": fill.side,
                        "size": str(fill.size),
                        "price": str(fill.price),
                        "timestamp": fill.timestamp,
                    },
                },
                indent=2,
            )
        )
    else:
        print(f"✓ Recorded {fill.side.upper()} {fill.size} @ {fill.price}")
        print(f"  Token: {fill.token_id}")
        print(f"  Cash impact: ${fill.cash_flow:,.2f}")


def cmd_paper_positions(args: argparse.Namespace) -> None:
    """Show current paper trading positions."""
    from pathlib import Path

    from .paper_trading import PaperTradingEngine

    engine = PaperTradingEngine(data_dir=Path(args.data_dir))
    positions = engine.get_positions()

    # Filter to open positions only unless --all
    if not args.all:
        positions = {k: v for k, v in positions.items() if v.net_size != 0}

    if args.format == "json":
        print(
            json.dumps(
                {
                    "positions": [p.to_dict() for p in positions.values()],
                    "count": len(positions),
                },
                indent=2,
            )
        )
    else:
        print("=" * 70)
        print("PAPER TRADING POSITIONS")
        print("=" * 70)
        print(f"Total positions: {len(positions)}")
        print(f"Open positions: {len([p for p in positions.values() if p.net_size != 0])}")
        print()

        for pos in sorted(positions.values(), key=lambda x: abs(x.net_size), reverse=True):
            status = "OPEN" if pos.net_size != 0 else "CLOSED"
            slug = pos.market_slug or pos.token_id[:30]
            print(f"{status:<7} | {slug:<40}")
            print(f"        Size: {pos.net_size:>12,.2f} | Avg cost: ${pos.avg_cost_basis:.3f}")
            print(
                f"        Realized PnL: ${pos.realized_pnl:>10,.2f} | Fees: ${pos.total_fees:.2f}"
            )
            print()


def cmd_paper_equity(args: argparse.Namespace) -> None:
    """Show current paper trading equity."""
    from pathlib import Path

    from .paper_trading import PaperTradingEngine

    engine = PaperTradingEngine(data_dir=Path(args.data_dir))

    # Use snapshot if provided
    snapshot_path = Path(args.snapshot) if args.snapshot else None

    equity = engine.compute_equity(snapshot_path=snapshot_path)

    if args.format == "json":
        print(json.dumps(equity.to_dict(), indent=2))
    else:
        print("=" * 70)
        print("PAPER TRADING EQUITY")
        print("=" * 70)
        print(f"Timestamp: {equity.timestamp}")
        print()
        print(f"Cash Balance:       ${equity.cash_balance:>12,.2f}")
        print(f"Mark to Market:     ${equity.mark_to_market:>12,.2f}")
        print(f"Liquidation Value:  ${equity.liquidation_value:>12,.2f}")
        print(f"Net Equity:         ${equity.net_equity:>12,.2f}")
        print()
        print(f"Realized PnL:       ${equity.realized_pnl:>12,.2f}")
        print(f"Unrealized PnL:     ${equity.unrealized_pnl:>12,.2f}")
        print(f"Total Fees:         ${equity.total_fees:>12,.2f}")
        print()
        print(f"Positions:          {equity.position_count}")
        print(f"Open Positions:     {equity.open_position_count}")
        print("=" * 70)


def cmd_paper_reconcile(args: argparse.Namespace) -> None:
    """Reconcile paper positions against a collector snapshot."""
    from decimal import Decimal
    from pathlib import Path

    from .paper_trading import PaperTradingEngine

    engine = PaperTradingEngine(data_dir=Path(args.data_dir))
    snapshot_path = Path(args.snapshot)

    if not snapshot_path.exists():
        print(
            json.dumps({"error": f"Snapshot not found: {args.snapshot}"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    result = engine.reconcile_against_snapshot(
        snapshot_path=snapshot_path,
        drift_threshold_usd=Decimal(str(args.drift_threshold)),
        drift_threshold_pct=Decimal(str(args.drift_pct)),
    )

    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("=" * 70)
        print("PAPER TRADING RECONCILIATION")
        print("=" * 70)
        print(f"Snapshot: {result.snapshot_timestamp}")
        print()
        print(f"Positions Reconciled: {result.positions_reconciled}")
        print(f"Positions with Drift: {result.positions_with_drift}")
        print(f"Total Drift (USD):    ${result.total_drift_usd:.2f}")
        print(f"Max Drift (%):        {result.max_drift_pct:.2f}%")
        print()

        if result.position_drifts:
            print("--- Position Drifts ---")
            for drift in result.position_drifts:
                print(f"  {drift['market_slug'] or drift['token_id'][:30]}")
                print(
                    f"    Size: {drift['net_size']:.2f} | Drift: ${drift['drift_usd']:.2f} ({drift['drift_pct']:.2f}%)"
                )

        if result.warnings:
            print("\n--- Warnings ---")
            for warning in result.warnings:
                print(f"  ! {warning}")

        print("=" * 70)


def cmd_paper_backtest(args: argparse.Namespace) -> None:
    """Run paper trading equity calculation against all 15m snapshots."""
    from pathlib import Path

    from .paper_trading import run_equity_calculation_against_snapshots

    data_dir = Path(args.data_dir)
    snapshot_dir = Path(args.snapshot_dir)
    output_file = Path(args.output) if args.output else None

    if not snapshot_dir.exists():
        print(
            json.dumps({"error": f"Snapshot directory not found: {args.snapshot_dir}"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    summary = run_equity_calculation_against_snapshots(
        data_dir=data_dir,
        snapshot_dir=snapshot_dir,
        output_file=output_file,
    )

    if "error" in summary:
        print(json.dumps(summary, indent=2), file=__import__("sys").stderr)
        raise SystemExit(1)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print("=" * 70)
        print("PAPER TRADING BACKTEST RESULTS")
        print("=" * 70)
        print(f"Snapshots Processed: {summary['snapshots_processed']}")
        print(f"Data Points:         {summary['data_points']}")
        print()
        print(f"Starting Equity:     ${summary['starting_equity']:,.2f}")
        print(f"Current Equity:      ${summary['current_equity']:,.2f}")
        print(
            f"Total Return:        ${summary['total_return']:,.2f} ({summary['total_return_pct']:.2f}%)"
        )
        print()
        print(
            f"Max Drawdown:        ${summary['max_drawdown']:,.2f} ({summary['max_drawdown_pct']:.2f}%)"
        )
        print(f"Realized PnL:        ${summary['realized_pnl']:,.2f}")
        print(f"Unrealized PnL:      ${summary['unrealized_pnl']:,.2f}")
        print(f"Total Fees:          ${summary['total_fees']:,.2f}")

        if output_file:
            print(f"\nEquity curve saved to: {output_file}")

        print("=" * 70)


def cmd_collect_fills(args: argparse.Namespace) -> None:
    """Collect fills from paper trading and/or real account."""
    from pathlib import Path

    from .fills_collector import collect_fills

    fills_path = Path(args.out) / "fills.jsonl" if args.out else Path(args.fills_path)
    paper_fills_path = Path(args.paper_fills_path) if args.paper_fills_path else None

    # Parse since timestamp
    since = None
    if args.since:
        from datetime import datetime
        since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))

    result = collect_fills(
        fills_path=fills_path,
        paper_fills_path=paper_fills_path,
        include_account=args.account,
        include_paper=args.paper,
        since=since,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("FILLS COLLECTION RESULTS")
        print("=" * 70)
        print(f"Fills file:       {result['fills_path']}")
        print(f"Since:            {result['since'] or 'beginning'}")
        print(f"Account fills:    {result['account_fills']}")
        print(f"Paper fills:      {result['paper_fills']}")
        print(f"Duplicates:       {result['duplicates_skipped']}")
        print(f"Total appended:   {result['total_appended']}")
        print("=" * 70)


def cmd_pnl_loop(args: argparse.Namespace) -> None:
    """Run PnL collection and verification loop."""
    from pathlib import Path
    from decimal import Decimal
    from .pnl_loop import collect_and_verify_loop

    collect_and_verify_loop(
        data_dir=Path(args.data_dir),
        snapshot_path=Path(args.snapshot) if args.snapshot else None,
        pnl_dir=Path(args.pnl_dir) if args.pnl_dir else None,
        interval_seconds=float(args.interval_seconds),
        verify_time=args.verify_time,
        starting_cash=Decimal(str(args.starting_cash)) if args.starting_cash else None,
        include_account=args.account,
        include_paper=args.paper,
    )


def cmd_pnl_health(args: argparse.Namespace) -> None:
    """Check health of fills and PnL data."""
    from pathlib import Path
    from .pnl_loop import pnl_health_check

    result = pnl_health_check(
        data_dir=Path(args.data_dir),
        max_fills_age_seconds=float(args.max_fills_age),
        max_pnl_age_seconds=float(args.max_pnl_age),
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        status = "✓ HEALTHY" if result["healthy"] else "✗ UNHEALTHY"
        print(f"{status}: PnL data health check")
        print()

        fills = result["fills"]
        print("--- Fills ---")
        print(f"  Exists:         {fills['exists']}")
        print(f"  Total fills:    {fills.get('total_fills', 0)}")
        print(f"  Last fill:      {fills.get('last_fill_at') or 'N/A'}")
        if fills.get('age_seconds') is not None:
            age_hours = fills['age_seconds'] / 3600
            print(f"  Age:            {age_hours:.1f}h (max: {fills['max_age_seconds']/3600:.1f}h)")
        print()

        pnl = result["pnl"]
        print("--- PnL Summaries ---")
        print(f"  Exists:         {pnl['exists']}")
        print(f"  Latest file:    {pnl.get('latest_file') or 'N/A'}")
        print(f"  Latest date:    {pnl.get('latest_date') or 'N/A'}")
        if pnl.get('age_seconds') is not None:
            age_hours = pnl['age_seconds'] / 3600
            print(f"  Age:            {age_hours:.1f}h (max: {pnl['max_age_seconds']/3600:.1f}h)")
        print()

        if result["warnings"]:
            print("--- Warnings ---")
            for warning in result["warnings"]:
                print(f"  ! {warning}")

    # Exit with error code if unhealthy and --fail is set
    if args.fail and not result["healthy"]:
        raise SystemExit(1)


def cmd_accounting_init(args: argparse.Namespace) -> None:
    """Initialize accounting database."""
    from decimal import Decimal
    from pathlib import Path

    from .accounting import init_accounting_db

    db_path = Path(args.db_path) if args.db_path else None
    db = init_accounting_db(db_path)

    starting_cash = Decimal(str(args.starting_cash))
    db.set_initial_cash(starting_cash)

    if args.format == "json":
        print(json.dumps({
            "status": "initialized",
            "db_path": str(db.db_path),
            "starting_cash": float(starting_cash),
        }, indent=2))
    else:
        print("=" * 70)
        print("ACCOUNTING DATABASE INITIALIZED")
        print("=" * 70)
        print(f"Database: {db.db_path}")
        print(f"Starting cash: ${starting_cash:,.2f}")
        print("=" * 70)


def cmd_accounting_record_fill(args: argparse.Namespace) -> None:
    """Record a paper trade fill in accounting database."""
    from decimal import Decimal
    from pathlib import Path
    from uuid import uuid4

    from .accounting import AccountingDB, AccountingFill

    db = AccountingDB(args.db_path)
    db.init_schema()

    # Calculate cash flow
    size = Decimal(str(args.size))
    price = Decimal(str(args.price))
    fee = Decimal(str(args.fee))
    notional = size * price

    if args.side == "buy":
        cash_flow = -(notional + fee)
    else:
        cash_flow = notional - fee

    # Parse timestamp
    timestamp = datetime.now(UTC)
    if args.timestamp:
        timestamp = datetime.fromisoformat(args.timestamp.replace("Z", "+00:00"))

    fill = AccountingFill(
        fill_id=str(uuid4()),
        timestamp=timestamp,
        token_id=args.token_id,
        side=args.side,
        size=size,
        price=price,
        fee=fee,
        cash_flow=cash_flow,
        market_slug=args.market_slug,
        market_question=args.market_question,
        trader_source=args.trader_source,
        original_tx_hash=args.original_tx,
    )

    db.record_fill(fill)

    if args.format == "json":
        print(json.dumps({
            "status": "recorded",
            "fill": fill.to_dict(),
            "cash_balance": float(db.get_cash_balance()),
        }, indent=2))
    else:
        print("=" * 70)
        print("FILL RECORDED")
        print("=" * 70)
        print(f"Fill ID: {fill.fill_id}")
        print(f"Token: {fill.token_id}")
        print(f"Side: {fill.side.upper()}")
        print(f"Size: {fill.size}")
        print(f"Price: ${fill.price}")
        print(f"Fee: ${fill.fee}")
        print(f"Cash Flow: ${fill.cash_flow}")
        print(f"Cash Balance: ${db.get_cash_balance():,.2f}")
        print("=" * 70)


def cmd_accounting_snapshot(args: argparse.Namespace) -> None:
    """Record portfolio snapshot."""
    from pathlib import Path

    from .accounting import AccountingDB

    db = AccountingDB(args.db_path)
    db.init_schema()

    # Load prices from snapshot file if provided
    prices = {}
    if args.snapshot_file:
        snapshot_path = Path(args.snapshot_file)
        if snapshot_path.exists():
            data = json.loads(snapshot_path.read_text())
            # Extract prices from snapshot format
            if "markets" in data:
                for market in data["markets"]:
                    if "books" in market:
                        for side in ["yes", "no"]:
                            if side in market["books"]:
                                book = market["books"][side]
                                token_id = f"{market.get('condition_id', 'unknown')}_{side}"
                                if "bids" in book and "asks" in book:
                                    best_bid = book["bids"][0]["price"] if book["bids"] else 0
                                    best_ask = book["asks"][0]["price"] if book["asks"] else 0
                                    if best_bid and best_ask:
                                        prices[token_id] = Decimal(str((best_bid + best_ask) / 2))

    snapshot = db.record_portfolio_snapshot(prices)

    if args.format == "json":
        print(json.dumps(snapshot.to_dict(), indent=2))
    else:
        print("=" * 70)
        print("PORTFOLIO SNAPSHOT")
        print("=" * 70)
        print(f"Snapshot ID: {snapshot.snapshot_id}")
        print(f"Time: {snapshot.snapshot_time}")
        print()
        print(f"Cash Balance:      ${snapshot.cash_balance:>12,.2f}")
        print(f"Positions Value:   ${snapshot.positions_value:>12,.2f}")
        print(f"Liquidation Value: ${snapshot.liquidation_value:>12,.2f}")
        print(f"NAV:               ${snapshot.nav:>12,.2f}")
        print()
        print(f"Realized PnL:      ${snapshot.realized_pnl:>12,.2f}")
        print(f"Unrealized PnL:    ${snapshot.unrealized_pnl:>12,.2f}")
        print(f"Total Fees:        ${snapshot.total_fees:>12,.2f}")
        print()
        print(f"Open Positions:    {snapshot.open_position_count}")
        print("=" * 70)


def cmd_accounting_summary(args: argparse.Namespace) -> None:
    """Show account summary."""
    from .accounting import AccountingDB

    db = AccountingDB(args.db_path)
    db.init_schema()

    summary = db.get_account_summary(days=args.days)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print("=" * 70)
        print(f"ACCOUNT SUMMARY (Last {args.days} days)")
        print("=" * 70)
        print(f"Period Start: {summary['period_start']}")
        print()
        print("--- NAV ---")
        print(f"Starting NAV:     ${summary['starting_nav']:>12,.2f}")
        print(f"Current NAV:      ${summary['current_nav']:>12,.2f}")
        print(f"Change:           ${summary['nav_change']:>12,.2f} ({summary['nav_change_pct']:+.2f}%)")
        print(f"Max NAV:          ${summary['max_nav']:>12,.2f}")
        print(f"Min NAV:          ${summary['min_nav']:>12,.2f}")
        print(f"Max Drawdown:     ${summary['max_drawdown']:>12,.2f} ({summary['max_drawdown_pct']:.2f}%)")
        print()
        print("--- PnL ---")
        print(f"Realized PnL:     ${summary['realized_pnl']:>12,.2f}")
        print(f"Unrealized PnL:   ${summary['unrealized_pnl']:>12,.2f}")
        print(f"Total Fees:       ${summary['total_fees']:>12,.2f}")
        print()
        print("--- Activity ---")
        print(f"Total Fills:      {summary['total_fills']}")
        print(f"Open Positions:   {summary['open_positions']}")
        print(f"Cash Balance:     ${summary['cash_balance']:>12,.2f}")
        print(f"Liquidation Val:  ${summary['liquidation_value']:>12,.2f}")
        print("=" * 70)


def cmd_accounting_positions(args: argparse.Namespace) -> None:
    """Show current positions."""
    from .accounting import AccountingDB

    db = AccountingDB(args.db_path)
    db.init_schema()

    positions = db.get_positions(open_only=not args.all)

    if args.format == "json":
        print(json.dumps({
            "positions": [p.to_dict() for p in positions],
            "count": len(positions),
        }, indent=2))
    else:
        print("=" * 70)
        print("POSITIONS")
        print("=" * 70)
        print(f"Total positions: {len(positions)}")
        print()

        open_count = 0
        for pos in positions:
            status = "OPEN" if pos.net_size != 0 else "CLOSED"
            if pos.net_size != 0:
                open_count += 1

            print(f"Token: {pos.token_id}")
            if pos.market_slug:
                print(f"  Market: {pos.market_slug}")
            print(f"  Status: {status}")
            print(f"  Net Size: {pos.net_size}")
            print(f"  Avg Cost: ${pos.avg_cost_basis:.4f}")
            print(f"  Total Cost: ${pos.total_cost:,.2f}")
            print(f"  Realized PnL: ${pos.realized_pnl:,.2f}")
            print(f"  Total Fees: ${pos.total_fees:,.2f}")
            print(f"  Fills: {pos.fill_count}")
            if pos.last_fill_at:
                print(f"  Last Fill: {pos.last_fill_at}")
            print()

        print(f"Open positions: {open_count}")
        print("=" * 70)


def cmd_accounting_exposures(args: argparse.Namespace) -> None:
    """Show current exposures breakdown."""
    from .accounting import AccountingDB

    db = AccountingDB(args.db_path)
    db.init_schema()

    exposures = db.get_exposures()
    latest = db.get_latest_snapshot()
    total_nav = latest.nav if latest else Decimal("0")

    if args.format == "json":
        print(json.dumps({
            "exposures": exposures,
            "total_nav": float(total_nav),
            "count": len(exposures),
        }, indent=2))
    else:
        print("=" * 70)
        print("CURRENT EXPOSURES")
        print("=" * 70)
        print(f"Total NAV: ${total_nav:,.2f}")
        print(f"Open positions: {len(exposures)}")
        print()

        total_exposure = 0.0
        for exp in exposures:
            slug = exp.get("market_slug") or exp["token_id"][:40]
            print(f"{slug}")
            print(f"  Size: {exp['net_size']:>10.2f} @ ${exp['avg_cost_basis']:.3f}")
            print(f"  Mark: ${exp['mark_price']:>8.3f} | MtM: ${exp['mark_to_market']:>10,.2f}")
            print(f"  Unrealized: ${exp['unrealized_pnl']:>10,.2f} | Realized: ${exp['realized_pnl']:>10,.2f}")
            print(f"  NAV %: {exp['nav_pct']:>6.2f}%")
            print()
            total_exposure += exp["nav_pct"]

        print(f"Total Exposure: {total_exposure:.2f}%")
        print("=" * 70)


def cmd_accounting_cash(args: argparse.Namespace) -> None:
    """Show cash ledger."""
    from datetime import UTC, datetime, timedelta

    from .accounting import AccountingDB

    db = AccountingDB(args.db_path)
    db.init_schema()

    since = datetime.now(UTC) - timedelta(days=args.days)
    entries = db.get_cash_ledger(since=since, limit=args.limit)

    if args.format == "json":
        print(json.dumps({
            "entries": [e.to_dict() for e in entries],
            "current_balance": float(db.get_cash_balance()),
            "count": len(entries),
        }, indent=2))
    else:
        print("=" * 70)
        print("CASH LEDGER")
        print("=" * 70)
        print(f"Current Balance: ${db.get_cash_balance():,.2f}")
        print()

        for entry in entries[:args.limit]:
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M") if entry.timestamp else "N/A"
            amount_str = f"${entry.amount:>10,.2f}"
            print(f"{ts} | {entry.entry_type:12} | {amount_str} | ${entry.balance_after:>12,.2f}")
            if entry.description:
                print(f"  {entry.description}")

        print()
        print(f"Showing {min(len(entries), args.limit)} entries")
        print("=" * 70)


def cmd_accounting_import_fills(args: argparse.Namespace) -> None:
    """Import fills from JSON/JSONL file."""
    from pathlib import Path
    from uuid import uuid4

    from .accounting import AccountingDB, AccountingFill

    db = AccountingDB(args.db_path)
    db.init_schema()

    input_path = Path(args.input)
    if not input_path.exists():
        print(json.dumps({"error": f"File not found: {args.input}"}), file=__import__("sys").stderr)
        raise SystemExit(1)

    # Load fills from file
    fills_data = []
    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    fills_data.append(json.loads(line))
    else:
        data = json.loads(input_path.read_text())
        if isinstance(data, list):
            fills_data = data
        elif isinstance(data, dict) and "fills" in data:
            fills_data = data["fills"]
        else:
            fills_data = [data]

    # Convert and record fills
    recorded = 0
    for fill_data in fills_data:
        # Map common field names
        token_id = fill_data.get("token_id") or fill_data.get("asset_id") or ""
        side = fill_data.get("side", "buy").lower()
        size = Decimal(str(fill_data.get("size") or fill_data.get("amount") or "0"))
        price = Decimal(str(fill_data.get("price") or fill_data.get("execution_price") or "0"))
        fee = Decimal(str(fill_data.get("fee") or fill_data.get("trade_fee") or "0"))
        timestamp_str = fill_data.get("timestamp") or fill_data.get("created_at")
        timestamp = datetime.now(UTC)
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

        # Calculate cash flow
        notional = size * price
        if side == "buy":
            cash_flow = -(notional + fee)
        else:
            cash_flow = notional - fee

        fill = AccountingFill(
            fill_id=str(uuid4()),
            timestamp=timestamp,
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            fee=fee,
            cash_flow=cash_flow,
            market_slug=fill_data.get("market_slug") or fill_data.get("slug"),
            market_question=fill_data.get("market_question"),
            trader_source=fill_data.get("trader_source"),
            original_tx_hash=fill_data.get("original_tx_hash") or fill_data.get("transaction_hash"),
        )

        db.record_fill(fill)
        recorded += 1

    if args.format == "json":
        print(json.dumps({
            "status": "imported",
            "recorded": recorded,
            "current_balance": float(db.get_cash_balance()),
        }, indent=2))
    else:
        print("=" * 70)
        print("FILLS IMPORTED")
        print("=" * 70)
        print(f"Recorded: {recorded} fills")
        print(f"Cash Balance: ${db.get_cash_balance():,.2f}")
        print("=" * 70)


def cmd_dataset_join(args: argparse.Namespace) -> None:
    """Align Polymarket 15m snapshots with Binance BTC features for lead/lag analysis."""
    from pathlib import Path

    from .dataset_join import build_aligned_dataset, save_report

    pm_dir = Path(args.polymarket_dir)
    bn_dir = Path(args.binance_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pm_dir

    report = build_aligned_dataset(
        polymarket_data_dir=pm_dir,
        binance_data_dir=bn_dir,
        hours=args.hours,
        tolerance_seconds=args.tolerance,
        horizons=args.horizons,
    )

    # Generate output filenames
    from datetime import UTC, datetime

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"leadlag_{timestamp}.json"
    text_path = out_dir / f"leadlag_{timestamp}.txt" if args.text else None

    # Save report
    save_report(report, json_path, text_path)

    # Output
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_text())

        if text_path:
            print(f"\nReport saved to: {json_path}")
            if text_path:
                print(f"Text report saved to: {text_path}")


def cmd_imbalance_backtest(args: argparse.Namespace) -> None:
    """Run orderbook imbalance strategy backtest."""
    from pathlib import Path

    from .strategy_imbalance import (
        load_snapshots_for_backtest,
        parameter_sweep,
        run_backtest,
    )

    data_dir = Path(args.data_dir)

    # Load snapshots
    snapshots = load_snapshots_for_backtest(
        data_dir=data_dir,
        interval=args.interval,
    )

    if not snapshots:
        print(
            json.dumps({"error": f"No snapshots found in {data_dir} for interval {args.interval}"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    if args.sweep:
        # Parameter sweep
        results = parameter_sweep(
            snapshots=snapshots,
            target_market_substring=args.target,
            horizon=args.horizon,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )

        if args.format == "json":
            output = {
                "config": {
                    "horizon": args.horizon,
                    "fee_bps": args.fee_bps,
                    "slippage_bps": args.slippage_bps,
                    "snapshots": len(snapshots),
                },
                "results": results,
            }
            print(json.dumps(output, indent=2))
        else:
            print("=" * 100)
            print("ORDERBOOK IMBALANCE STRATEGY - PARAMETER SWEEP")
            print("=" * 100)
            print(f"Snapshots analyzed: {len(snapshots)}")
            print(f"Target market: {args.target}")
            print(f"Horizon: {args.horizon} snapshot(s) ahead")
            print(f"Fee: {args.fee_bps} bps | Slippage: {args.slippage_bps} bps")
            print(
                f"\n{'Rank':<6}{'k':<4}{'theta':<8}{'p_max':<8}{'Trades':<8}{'Hit%':<8}{'Win%':<8}{'Total PnL':<12}{'EV/Trade':<10}{'Brier':<8}"
            )
            print("-" * 100)

            for i, result in enumerate(results[:15], 1):  # Top 15
                p = result["params"]
                m = result["metrics"]
                hit_pct = m.get("hit_rate", 0) * 100
                win_pct = m.get("win_rate", 0) * 100
                total_pnl = m.get("total_pnl", 0)
                ev = m.get("ev_per_trade", 0)
                brier = m.get("avg_brier_score", 0)
                print(
                    f"{i:<6}{p['k']:<4}{p['theta']:<8.2f}{p['p_max']:<8.2f}"
                    f"{m['total_trades']:<8}{hit_pct:<8.1f}{win_pct:<8.1f}"
                    f"{total_pnl:<12.4f}{ev:<10.4f}{brier:<8.4f}"
                )

            print("=" * 100)

    else:
        # Single backtest run
        result = run_backtest(
            snapshots=snapshots,
            k=args.k,
            theta=args.theta,
            p_max=args.p_max,
            target_market_substring=args.target,
            horizon=args.horizon,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )

        if args.format == "json":
            print(json.dumps(result.to_dict(include_trades=args.include_trades), indent=2))
        else:
            print("=" * 80)
            print("ORDERBOOK IMBALANCE STRATEGY - BACKTEST RESULTS")
            print("=" * 80)
            print(f"Snapshots analyzed: {len(snapshots)}")
            print(f"Target market: {args.target}")
            print(f"Horizon: {args.horizon} snapshot(s) ahead")
            print("\nParameters:")
            print(f"  k (depth levels):     {args.k}")
            print(f"  theta (threshold):    {args.theta:.2f}")
            print(f"  p_max (max price):    {args.p_max:.2f}")
            print(f"  fee_bps:              {args.fee_bps}")
            print(f"  slippage_bps:         {args.slippage_bps}")

            print("\n--- Trade Summary ---")
            print(f"Total trades:         {result.metrics['total_trades']}")
            print(f"UP trades:            {result.metrics['up_trades']}")
            print(f"DOWN trades:          {result.metrics['down_trades']}")
            print(f"Trades with outcome:  {result.metrics.get('trades_with_outcome', 0)}")

            print("\n--- Accuracy Metrics ---")
            hit_rate = result.metrics.get("hit_rate", 0)
            up_hit = result.metrics.get("up_hit_rate", 0)
            down_hit = result.metrics.get("down_hit_rate", 0)
            print(f"Hit rate (direction): {hit_rate:.1%}")
            print(f"  UP predictions:     {up_hit:.1%}")
            print(f"  DOWN predictions:   {down_hit:.1%}")

            print("\n--- Calibration Metrics ---")
            brier = result.metrics.get("avg_brier_score", 0)
            logloss = result.metrics.get("avg_log_loss", 0)
            print(f"Avg Brier score:      {brier:.4f} (lower=better, 0=perfect)")
            print(f"Avg log loss:         {logloss:.4f} (lower=better)")

            print("\n--- PnL Metrics ---")
            total_pnl = result.metrics.get("total_pnl", 0)
            win_rate = result.metrics.get("win_rate", 0)
            ev = result.metrics.get("ev_per_trade", 0)
            sharpe = result.metrics.get("sharpe_ratio", 0)
            print(f"Total PnL:            {total_pnl:+.4f}")
            print(f"Win rate (PnL>0):     {win_rate:.1%}")
            print(f"EV per trade:         {ev:+.4f}")
            print(f"Sharpe ratio:         {sharpe:.3f}")

            if result.trades and args.show_trades:
                print("\n--- Recent Trades (last 10) ---")
                for t in result.trades[-10:]:
                    outcome_str = ""
                    if t.outcome_up is not None:
                        correct = (t.decision == "UP" and t.outcome_up) or (
                            t.decision == "DOWN" and not t.outcome_up
                        )
                        outcome_str = f" | outcome={'UP' if t.outcome_up else 'DOWN'} {'✓' if correct else '✗'}"
                    pnl_str = f" | pnl={t.pnl:+.4f}" if t.pnl is not None else ""
                    print(
                        f"  {t.timestamp.strftime('%H:%M')} | {t.decision:<6} | "
                        f"imb={t.imbalance_value:.3f} | mid={t.mid_yes:.3f} | "
                        f"entry={t.entry_price:.3f} | conf={t.confidence:.2f}"
                        f"{outcome_str}{pnl_str}"
                    )

            print("=" * 80)


def cmd_marketdata_collect(args: argparse.Namespace) -> None:
    """Collect market data using provider abstraction with auto-fallback."""
    from pathlib import Path

    from polymarket.marketdata.collector import collect_snapshot

    out_dir = Path(args.out)

    # Use provider-specific subdirectory
    provider = args.provider
    data_dir = out_dir / "marketdata" / provider

    out_path = collect_snapshot(
        out_dir=data_dir,
        provider_name=provider,
        symbol=args.symbol,
        kline_intervals=args.intervals,
        timeout=args.timeout,
    )

    if args.verbose:
        print(f"Provider: {provider}")
        print(f"Symbol: {args.symbol}")
        print(f"Output: {out_path}")
    else:
        print(str(out_path))


def main() -> None:
    p = argparse.ArgumentParser(prog="polymarket")
    sub = p.add_subparsers(dest="cmd", required=True)

    p5 = sub.add_parser("markets-5m", help="Heuristic fetch of likely 5-minute markets")
    p5.add_argument("--limit", type=int, default=50)
    p5.add_argument("--search", type=str, default=None)
    p5.set_defaults(func=cmd_markets_5m)

    pb = sub.add_parser("book", help="Fetch CLOB orderbook for a token_id")
    pb.add_argument("token_id")
    pb.set_defaults(func=cmd_book)

    pp = sub.add_parser("price", help="Fetch CLOB price for a token_id")
    pp.add_argument("token_id")
    pp.add_argument("--side", choices=["buy", "sell"], default="buy")
    pp.set_defaults(func=cmd_price)

    pc = sub.add_parser("collect-5m", help="Snapshot /predictions/5M + CLOB orderbooks")
    pc.add_argument("--out", default="data")
    pc.add_argument(
        "--every-seconds",
        type=float,
        default=None,
        help="Enable continuous collection mode (interval in seconds)",
    )
    pc.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=60.0,
        help="Max backoff on errors (default: 60)",
    )
    pc.add_argument(
        "--retention-hours", type=float, default=None, help="Prune snapshots older than N hours"
    )
    pc.set_defaults(func=cmd_collect_5m)

    pu = sub.add_parser("universe-5m", help="Build normalized market universe from /predictions/5M")
    pu.add_argument("--out", default="data/universe.json", help="Output JSON file path")
    pu.add_argument("--cross-check", action="store_true", help="Verify against Gamma API")
    pu.set_defaults(func=cmd_universe_5m)

    p15 = sub.add_parser("markets-15m", help="Heuristic fetch of 15-minute crypto interval markets")
    p15.add_argument("--limit", type=int, default=50)
    p15.set_defaults(func=cmd_markets_15m)

    pc15 = sub.add_parser("collect-15m", help="Snapshot /crypto/15M + CLOB orderbooks")
    pc15.add_argument("--out", default="data")
    pc15.set_defaults(func=cmd_collect_15m)

    pc15l = sub.add_parser(
        "collect-15m-loop", help="Continuously snapshot /crypto/15M + CLOB orderbooks"
    )
    pc15l.add_argument("--out", default="data")
    pc15l.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Collection interval in seconds (default: 60)",
    )
    pc15l.add_argument(
        "--max-backoff-seconds", type=float, default=60.0, help="Max backoff on errors"
    )
    pc15l.add_argument(
        "--retention-hours",
        type=float,
        default=24.0,
        help="Prune snapshots older than N hours (default: 24)",
    )
    pc15l.add_argument(
        "--max-snapshots",
        type=int,
        default=1440,
        help="Max snapshots to retain (default: 1440 ~= 24h at 60s)",
    )
    pc15l.add_argument(
        "--microstructure-interval-seconds",
        type=float,
        default=60.0,
        help="Seconds between microstructure analyses",
    )
    pc15l.add_argument(
        "--microstructure-target",
        type=str,
        default="bitcoin",
        help="Target market substring filter (e.g., 'bitcoin')",
    )
    pc15l.add_argument(
        "--spread-alert-threshold",
        type=float,
        default=DEFAULT_SPREAD_ALERT_THRESHOLD,
        help=f"Alert threshold for spread (default: {DEFAULT_SPREAD_ALERT_THRESHOLD})",
    )
    pc15l.add_argument(
        "--extreme-pin-threshold",
        type=float,
        default=DEFAULT_EXTREME_PIN_THRESHOLD,
        help=f"Alert threshold for extreme price pinning (default: {DEFAULT_EXTREME_PIN_THRESHOLD})",
    )
    pc15l.add_argument(
        "--depth-levels",
        type=int,
        default=DEFAULT_DEPTH_LEVELS,
        help=f"Number of book levels for depth calc (default: {DEFAULT_DEPTH_LEVELS})",
    )
    pc15l.set_defaults(func=cmd_collect_15m_loop)

    # Enhanced PnL verify command
    pnl = sub.add_parser(
        "pnl-verify",
        help="Verify PnL from fills with cash tracking, sanity checks, and liquidation value",
    )
    pnl.add_argument(
        "--input",
        default=None,
        help="Path to fills JSON/JSONL file (alternative to --data-dir)",
    )
    pnl.add_argument(
        "--data-dir",
        default=None,
        help="Data directory containing fills.json or fills.jsonl",
    )
    pnl.add_argument(
        "--books",
        default=None,
        help="Path to orderbooks JSON for liquidation value calculation",
    )
    pnl.add_argument(
        "--snapshot",
        default=None,
        help=(
            "Path to collector snapshot for orderbook data. "
            "Supports pointer files like data/latest_15m.json"
        ),
    )
    pnl.add_argument(
        "--since",
        default=None,
        help="ISO timestamp to filter fills (inclusive)",
    )
    pnl.add_argument(
        "--market",
        default=None,
        help="Market slug filter (substring match)",
    )
    pnl.add_argument(
        "--starting-cash",
        type=float,
        default=0.0,
        help="Starting cash balance (default: 0)",
    )
    pnl.add_argument(
        "--save-daily",
        action="store_true",
        help="Save daily summary to data/pnl/",
    )
    pnl.add_argument(
        "--daily-dir",
        default=None,
        help="Custom directory for daily summaries (default: data/pnl/)",
    )
    pnl.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format",
    )
    pnl.set_defaults(func=cmd_pnl_verify)

    # Collect fills command
    cf = sub.add_parser(
        "collect-fills",
        help="Collect fills from paper trading and/or real account",
    )
    cf.add_argument(
        "--fills-path",
        default="data/fills.jsonl",
        help="Output path for fills.jsonl (default: data/fills.jsonl)",
    )
    cf.add_argument(
        "--paper-fills-path",
        default=None,
        help="Path to paper trading fills.jsonl (default: data/paper_trading/fills.jsonl)",
    )
    cf.add_argument(
        "--out",
        default=None,
        help="Output directory (alternative to --fills-path)",
    )
    cf.add_argument(
        "--since",
        default=None,
        help="ISO timestamp to collect from (default: last fill timestamp)",
    )
    cf.add_argument(
        "--account",
        action="store_true",
        default=True,
        help="Include real account fills (default: True)",
    )
    cf.add_argument(
        "--no-account",
        action="store_false",
        dest="account",
        help="Skip account fills",
    )
    cf.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Include paper trading fills (default: True)",
    )
    cf.add_argument(
        "--no-paper",
        action="store_false",
        dest="paper",
        help="Skip paper trading fills",
    )
    cf.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format",
    )
    cf.set_defaults(func=cmd_collect_fills)

    # PnL loop command
    pl = sub.add_parser(
        "pnl-loop",
        help="Run continuous PnL collection and verification loop",
    )
    pl.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    pl.add_argument(
        "--snapshot",
        default=None,
        help="Path to snapshot file (default: data/latest_15m.json)",
    )
    pl.add_argument(
        "--pnl-dir",
        default=None,
        help="Directory for PnL summaries (default: data/pnl)",
    )
    pl.add_argument(
        "--interval-seconds",
        type=float,
        default=3600.0,
        help="Collection interval in seconds (default: 3600 = 1 hour)",
    )
    pl.add_argument(
        "--verify-time",
        default=None,
        help="Time of day to run verification (HH:MM format, e.g., '00:00' for midnight)",
    )
    pl.add_argument(
        "--starting-cash",
        type=float,
        default=0.0,
        help="Starting cash balance",
    )
    pl.add_argument(
        "--account",
        action="store_true",
        default=True,
        help="Include account fills (default: True)",
    )
    pl.add_argument(
        "--no-account",
        action="store_false",
        dest="account",
        help="Skip account fills",
    )
    pl.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Include paper fills (default: True)",
    )
    pl.add_argument(
        "--no-paper",
        action="store_false",
        dest="paper",
        help="Skip paper fills",
    )
    pl.set_defaults(func=cmd_pnl_loop)

    # PnL health command
    ph = sub.add_parser(
        "pnl-health",
        help="Check health of fills and PnL data",
    )
    ph.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)",
    )
    ph.add_argument(
        "--max-fills-age",
        type=float,
        default=86400.0,
        help="Max fills age in seconds (default: 86400 = 24h)",
    )
    ph.add_argument(
        "--max-pnl-age",
        type=float,
        default=86400.0,
        help="Max PnL summary age in seconds (default: 86400 = 24h)",
    )
    ph.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format",
    )
    ph.add_argument(
        "--fail",
        action="store_true",
        help="Exit with error code if unhealthy",
    )
    ph.set_defaults(func=cmd_pnl_health)

    ms = sub.add_parser("microstructure", help="Analyze market microstructure from snapshot")
    ms.add_argument(
        "--snapshot",
        required=True,
        help=("Path to snapshot JSON file. Supports pointer files like data/latest_15m.json"),
    )
    ms.add_argument("--target", type=str, default="bitcoin", help="Target market substring filter")
    ms.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Generate summary with alerts (default)",
    )
    ms.add_argument("--raw", action="store_true", help="Output raw analysis without summary")
    ms.add_argument(
        "--spread-threshold",
        type=float,
        default=DEFAULT_SPREAD_ALERT_THRESHOLD,
        help=f"Alert threshold for spread (default: {DEFAULT_SPREAD_ALERT_THRESHOLD})",
    )
    ms.add_argument(
        "--extreme-pin-threshold",
        type=float,
        default=DEFAULT_EXTREME_PIN_THRESHOLD,
        help=f"Alert threshold for extreme price pinning (default: {DEFAULT_EXTREME_PIN_THRESHOLD})",
    )
    ms.add_argument(
        "--tight-spread-threshold",
        type=float,
        default=DEFAULT_TIGHT_SPREAD_THRESHOLD,
        help=f"Tight spread threshold for suppressing alerts (default: {DEFAULT_TIGHT_SPREAD_THRESHOLD})",
    )
    ms.add_argument(
        "--consistency-threshold",
        type=float,
        default=DEFAULT_CONSISTENCY_THRESHOLD,
        help=f"Probability consistency threshold for suppressing alerts (default: {DEFAULT_CONSISTENCY_THRESHOLD})",
    )
    ms.add_argument(
        "--depth-levels",
        type=int,
        default=DEFAULT_DEPTH_LEVELS,
        help=f"Number of book levels for depth calc (default: {DEFAULT_DEPTH_LEVELS})",
    )
    ms.add_argument(
        "--btc-features",
        type=str,
        default=None,
        help="Path to aligned BTC features JSON for alert enrichment",
    )
    ms.add_argument(
        "--btc-horizon",
        type=str,
        default="5m",
        choices=["5m", "1h", "24h"],
        help="BTC return horizon to display in alerts (default: 5m)",
    )
    ms.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    ms.set_defaults(func=cmd_microstructure)

    # Binance commands
    bc = sub.add_parser("binance-collect", help="Collect Binance BTC market data (single snapshot)")
    bc.add_argument("--out", default="data/binance", help="Output directory")
    bc.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)")
    bc.add_argument("--intervals", nargs="+", default=["1m", "5m"], help="Kline intervals to fetch")
    bc.add_argument(
        "--base-url",
        nargs="+",
        default=None,
        help="Override base URL(s) for Binance API (default: auto-failover list)",
    )
    bc.set_defaults(func=cmd_binance_collect)

    bcl = sub.add_parser("binance-loop", help="Continuously collect Binance data via WebSocket")
    bcl.add_argument("--out", default="data/binance", help="Output directory")
    bcl.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    bcl.add_argument(
        "--intervals", nargs="+", default=["1m", "5m"], help="Kline intervals to subscribe"
    )
    bcl.add_argument(
        "--snapshot-interval-seconds", type=float, default=5.0, help="Snapshot interval"
    )
    bcl.add_argument(
        "--max-reconnect-delay", type=float, default=60.0, help="Max reconnection delay"
    )
    bcl.add_argument("--retention-hours", type=float, default=None, help="Prune old files")
    bcl.add_argument(
        "--ws-base",
        nargs="+",
        default=None,
        help="Override WebSocket base URL(s) (default: auto-failover list)",
    )
    bcl.set_defaults(func=cmd_binance_loop)

    bf = sub.add_parser("binance-align", help="Align Binance features to Polymarket snapshots")
    bf.add_argument("--binance-dir", default="data/binance", help="Binance data directory")
    bf.add_argument("--polymarket-dir", default="data", help="Polymarket data directory")
    bf.add_argument("--out", default="data/aligned_features.json", help="Output file")
    bf.add_argument("--tolerance", type=float, default=1.0, help="Alignment tolerance in seconds")
    bf.set_defaults(func=cmd_binance_features)

    wd = sub.add_parser(
        "watchdog",
        help="Run collector watchdog to ensure data freshness and auto-restart",
    )
    wd.add_argument("--data-dir", default="data", help="Data directory containing snapshots")
    wd.add_argument(
        "--max-age-seconds",
        type=float,
        default=120.0,
        help="Max acceptable snapshot age (default: 120s)",
    )
    wd.add_argument(
        "--script",
        default=None,
        help="Path to run.sh script (default: auto-detect)",
    )
    wd.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without taking action",
    )
    wd.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    wd.add_argument("--fail", action="store_true", help="Exit with error code if data is stale")
    wd.set_defaults(func=cmd_watchdog)

    hc = sub.add_parser("health-check", help="Check collector health and staleness SLA")
    hc.add_argument("--data-dir", default="data", help="Data directory containing snapshots")
    hc.add_argument(
        "--max-age-seconds",
        type=float,
        default=120.0,
        help="Max acceptable snapshot age (default: 120s)",
    )
    hc.add_argument("--prefix", default="snapshot_15m", help="Snapshot file prefix")
    hc.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    hc.add_argument("--fail", action="store_true", help="Exit with error code if unhealthy")
    hc.set_defaults(func=cmd_health_check)

    # Weather forecast command
    wf = sub.add_parser(
        "weather-forecast",
        help="Fetch weather forecasts for specified cities",
    )
    wf.add_argument(
        "--cities",
        type=str,
        default="nyc,london",
        help="Comma-separated list of cities (default: nyc,london)",
    )
    wf.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (ISO format, default: tomorrow)",
    )
    wf.add_argument(
        "--min-models",
        type=int,
        default=2,
        help="Minimum models required for consensus (default: 2)",
    )
    wf.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    wf.set_defaults(func=cmd_weather_forecast)

    # Weather scan command
    ws = sub.add_parser(
        "weather-scan",
        help="Scan for weather market arbitrage opportunities",
    )
    ws.add_argument(
        "--snapshots-dir",
        type=str,
        default=None,
        help="Directory containing market snapshots",
    )
    ws.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated list of cities (default: nyc,london)",
    )
    ws.add_argument(
        "--live",
        action="store_true",
        help="Execute live trades (default: dry-run)",
    )
    ws.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    ws.set_defaults(func=cmd_weather_scan)

    # Dataset join command for lead/lag analysis
    dj = sub.add_parser(
        "dataset-join",
        help="Align Polymarket 15m snapshots with Binance BTC features for lead/lag analysis",
    )
    dj.add_argument(
        "--polymarket-dir",
        default="data",
        help="Polymarket data directory containing 15m snapshots (default: data)",
    )
    dj.add_argument(
        "--binance-dir",
        default="data/binance",
        help="Binance data directory (default: data/binance)",
    )
    dj.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for reports (default: same as --polymarket-dir)",
    )
    dj.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Hours of data to analyze (default: 24)",
    )
    dj.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Alignment tolerance in seconds (default: 5.0)",
    )
    dj.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Return horizons in seconds (default: 5 15 30 60 300 900)",
    )
    dj.add_argument(
        "--text",
        action="store_true",
        help="Also save human-readable text report",
    )
    dj.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    dj.set_defaults(func=cmd_dataset_join)

    # Orderbook imbalance backtest command
    ib = sub.add_parser(
        "imbalance-backtest",
        help="Backtest orderbook imbalance strategy on BTC interval markets",
    )
    ib.add_argument("--data-dir", default="data", help="Data directory containing snapshots")
    ib.add_argument(
        "--interval",
        choices=["5m", "15m"],
        default="15m",
        help="Market interval to analyze (default: 15m)",
    )
    ib.add_argument(
        "--k",
        type=int,
        default=3,
        choices=[1, 3, 5],
        help="Depth levels for imbalance calculation (default: 3)",
    )
    ib.add_argument(
        "--theta",
        type=float,
        default=0.70,
        help="Imbalance threshold 0.5-1.0 (default: 0.70)",
    )
    ib.add_argument(
        "--p-max",
        type=float,
        default=0.65,
        help="Max price to pay for position 0.5-1.0 (default: 0.65)",
    )
    ib.add_argument(
        "--target",
        type=str,
        default="bitcoin",
        help="Target market substring filter (default: bitcoin)",
    )
    ib.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep across k/theta/p_max combinations",
    )
    ib.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of snapshots ahead to evaluate outcome (default: 1)",
    )
    ib.add_argument(
        "--fee-bps",
        type=float,
        default=50.0,
        help="Taker fee in basis points (default: 50 = 0.5%%)",
    )
    ib.add_argument(
        "--slippage-bps",
        type=float,
        default=10.0,
        help="Slippage in basis points (default: 10 = 0.1%%)",
    )
    ib.add_argument(
        "--show-trades",
        action="store_true",
        help="Show individual trades in human output",
    )
    ib.add_argument(
        "--include-trades",
        action="store_true",
        help="Include full trade list in JSON output",
    )
    ib.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    ib.set_defaults(func=cmd_imbalance_backtest)

    # Market data provider command (with auto-fallback)
    mdc = sub.add_parser(
        "marketdata-collect",
        help="Collect BTC market data with provider fallback (binance/coinbase/kraken/auto)",
    )
    mdc.add_argument(
        "--out",
        default="data",
        help="Output directory (default: data)",
    )
    mdc.add_argument(
        "--provider",
        choices=["binance", "coinbase", "kraken", "auto"],
        default="auto",
        help="Data provider (default: auto - tries binance, then coinbase, then kraken)",
    )
    mdc.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)",
    )
    mdc.add_argument(
        "--intervals",
        nargs="+",
        default=["1m", "5m"],
        help="Kline intervals to fetch (default: 1m 5m)",
    )
    mdc.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    mdc.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    mdc.set_defaults(func=cmd_marketdata_collect)

    # Cross-market arbitrage commands
    cm = sub.add_parser(
        "cross-market-scan",
        help="Scan for cross-market arbitrage opportunities (Polymarket vs Kalshi)",
    )
    cm.add_argument("--out", default="data/cross_market", help="Output directory for trade data")
    cm.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories (politics,crypto,sports,finance)",
    )
    cm.add_argument(
        "--min-gross-spread",
        type=float,
        default=0.01,
        help="Minimum gross spread before fees (default: 0.01 = 1%%)",
    )
    cm.add_argument(
        "--min-net-spread",
        type=float,
        default=0.005,
        help="Minimum net spread after fees (default: 0.005 = 0.5%%)",
    )
    cm.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum concurrent positions (default: 10)",
    )
    cm.add_argument(
        "--position-size",
        type=float,
        default=1.0,
        help="Position size in contracts per side (default: 1.0)",
    )
    cm.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: dry-run)",
    )
    cm.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    cm.set_defaults(func=cmd_cross_market_scan)

    cmr = sub.add_parser(
        "cross-market-report",
        help="Generate performance report for cross-market arbitrage",
    )
    cmr.add_argument(
        "--data-dir",
        default="data/cross_market",
        help="Data directory containing trade data",
    )
    cmr.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    cmr.set_defaults(func=cmd_cross_market_report)

    # Both-sides arbitrage commands
    bs = sub.add_parser(
        "both-sides-scan",
        help="Scan for both-sides mispricing arbitrage opportunities on BTC markets",
    )
    bs.add_argument(
        "--interval",
        choices=["5m", "15m"],
        default="5m",
        help="Market interval to analyze (default: 5m)",
    )
    bs.add_argument(
        "--check-alignment",
        action="store_true",
        help="Check 15m alignment for 5m signals",
    )
    bs.add_argument(
        "--min-spread",
        type=float,
        default=0.02,
        help="Minimum spread after fees (default: 0.02 = 2%%)",
    )
    bs.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max opportunities to display",
    )
    bs.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    bs.set_defaults(func=cmd_both_sides_scan)

    bst = sub.add_parser(
        "both-sides-trade",
        help="Execute paper trades for both-sides arbitrage opportunities",
    )
    bst.add_argument(
        "--interval",
        choices=["5m", "15m"],
        default="5m",
        help="Market interval (default: 5m)",
    )
    bst.add_argument(
        "--check-alignment",
        action="store_true",
        help="Check 15m alignment for 5m signals",
    )
    bst.add_argument(
        "--min-spread",
        type=float,
        default=0.02,
        help="Minimum spread after fees (default: 0.02 = 2%%)",
    )
    bst.add_argument(
        "--position-size",
        type=float,
        default=100.0,
        help="Position size per side in $ (default: 100)",
    )
    bst.add_argument(
        "--scan",
        action="store_true",
        default=True,
        help="Scan before trading",
    )
    bst.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    bst.set_defaults(func=cmd_both_sides_trade)

    bss = sub.add_parser(
        "both-sides-stats",
        help="Show both-sides arbitrage strategy statistics",
    )
    bss.add_argument(
        "--position-size",
        type=float,
        default=100.0,
        help="Position size per side in $ (default: 100)",
    )
    bss.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    bss.set_defaults(func=cmd_both_sides_stats)

    # Mention market scan command
    mm = sub.add_parser(
        "mention-scan",
        help="Scan for mention market opportunities with default-to-NO strategy",
    )
    mm.add_argument(
        "--snapshots-dir",
        type=str,
        default=None,
        help="Directory containing market snapshots",
    )
    mm.add_argument(
        "--base-rate",
        type=float,
        default=0.15,
        help="Historical base rate for mentions (default: 0.15)",
    )
    mm.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum positions to take (default: 10)",
    )
    mm.add_argument(
        "--live",
        action="store_true",
        help="Execute live trades (default: dry-run)",
    )
    mm.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    mm.set_defaults(func=cmd_mention_scan)

    # Paper trading commands
    paper = sub.add_parser("paper", help="Paper trading commands")
    paper_sub = paper.add_subparsers(dest="paper_cmd", required=True)

    # paper record-fill
    paper_fill = paper_sub.add_parser("record-fill", help="Record a paper trade fill")
    paper_fill.add_argument("token_id", help="Token ID (YES/NO token)")
    paper_fill.add_argument("side", choices=["buy", "sell"], help="Trade side")
    paper_fill.add_argument("size", type=float, help="Number of shares")
    paper_fill.add_argument("price", type=float, help="Execution price (0.0-1.0)")
    paper_fill.add_argument("--fee", type=float, default=0.0, help="Trading fee")
    paper_fill.add_argument("--market-slug", default=None, help="Market slug")
    paper_fill.add_argument("--market-question", default=None, help="Market question")
    paper_fill.add_argument("--data-dir", default="data/paper_trading", help="Data directory")
    paper_fill.add_argument("--starting-cash", type=float, default=10000.0, help="Starting cash")
    paper_fill.add_argument(
        "--format", choices=["json", "human"], default="human", help="Output format"
    )
    paper_fill.set_defaults(func=cmd_paper_record_fill)

    # paper positions
    paper_pos = paper_sub.add_parser("positions", help="Show current positions")
    paper_pos.add_argument("--data-dir", default="data/paper_trading", help="Data directory")
    paper_pos.add_argument("--all", action="store_true", help="Show all positions including closed")
    paper_pos.add_argument(
        "--format", choices=["json", "human"], default="human", help="Output format"
    )
    paper_pos.set_defaults(func=cmd_paper_positions)

    # paper equity
    paper_eq = paper_sub.add_parser("equity", help="Show current equity")
    paper_eq.add_argument("--data-dir", default="data/paper_trading", help="Data directory")
    paper_eq.add_argument("--snapshot", default=None, help="Path to collector snapshot for prices")
    paper_eq.add_argument(
        "--format", choices=["json", "human"], default="human", help="Output format"
    )
    paper_eq.set_defaults(func=cmd_paper_equity)

    # paper reconcile
    paper_rec = paper_sub.add_parser("reconcile", help="Reconcile against collector snapshot")
    paper_rec.add_argument("snapshot", help="Path to collector snapshot")
    paper_rec.add_argument("--data-dir", default="data/paper_trading", help="Data directory")
    paper_rec.add_argument(
        "--drift-threshold", type=float, default=0.01, help="USD drift threshold"
    )
    paper_rec.add_argument(
        "--drift-pct", type=float, default=0.01, help="Percentage drift threshold"
    )
    paper_rec.add_argument(
        "--format", choices=["json", "human"], default="human", help="Output format"
    )
    paper_rec.set_defaults(func=cmd_paper_reconcile)

    # paper backtest
    paper_bt = paper_sub.add_parser("backtest", help="Run backtest against 15m snapshots")
    paper_bt.add_argument("--data-dir", default="data/paper_trading", help="Data directory")
    paper_bt.add_argument(
        "--snapshot-dir", default="data", help="Directory with collector snapshots"
    )
    paper_bt.add_argument("--output", default=None, help="Output file for equity curve JSON")
    paper_bt.add_argument(
        "--format", choices=["json", "human"], default="human", help="Output format"
    )
    paper_bt.set_defaults(func=cmd_paper_backtest)
    # Combinatorial arbitrage command
    cb = sub.add_parser(
        "combinatorial-scan",
        help="Scan for combinatorial arbitrage (Dutch book) opportunities",
    )
    cb.add_argument(
        "--event-limit",
        type=int,
        default=100,
        help="Maximum events to scan (default: 100)",
    )
    cb.add_argument(
        "--fee-rate",
        type=float,
        default=0.0315,
        help="Settlement fee rate (default: 0.0315 = 3.15%%)",
    )
    cb.add_argument(
        "--min-edge",
        type=float,
        default=0.015,
        help="Minimum edge after fees (default: 0.015 = 1.5%%)",
    )
    cb.add_argument(
        "--max-basket-size",
        type=int,
        default=4,
        help="Maximum outcomes per basket (default: 4)",
    )
    cb.add_argument(
        "--min-liquidity",
        type=float,
        default=100.0,
        help="Minimum liquidity per outcome (default: 100)",
    )
    cb.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed output for all baskets",
    )
    cb.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    cb.set_defaults(func=cmd_combinatorial_scan)

    # News-driven momentum commands
    nm = sub.add_parser(
        "news-momentum-scan",
        help="Scan for news-driven momentum trading opportunities",
    )
    nm.add_argument(
        "--snapshots-dir",
        type=str,
        default=None,
        help="Directory containing market snapshots",
    )
    nm.add_argument(
        "--headline",
        type=str,
        default=None,
        help="News headline to analyze (if not provided, uses example data)",
    )
    nm.add_argument(
        "--source",
        type=str,
        default=None,
        help="News source name",
    )
    nm.add_argument(
        "--source-reliability",
        type=str,
        default="MAJOR_OUTLET",
        choices=["VERIFIED", "MAJOR_OUTLET", "ESTABLISHED", "AGGREGATOR", "RUMOR"],
        help="Source reliability tier (default: MAJOR_OUTLET)",
    )
    nm.add_argument(
        "--seconds-ago",
        type=float,
        default=60,
        help="How many seconds ago the news broke (default: 60)",
    )
    nm.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Available capital in USD (default: 10000)",
    )
    nm.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum positions to take (default: 5)",
    )
    nm.add_argument(
        "--live",
        action="store_true",
        help="Execute live trades (default: dry-run)",
    )
    nm.add_argument(
        "--max-time-seconds",
        type=float,
        default=120,
        help="Max time since news to enter (default: 120)",
    )
    nm.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge for entry (default: 0.05 = 5%%)",
    )
    nm.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
    )
    nm.add_argument(
        "--base-position-size",
        type=float,
        default=2.0,
        help="Base position size %% of capital (default: 2.0)",
    )
    nm.add_argument(
        "--scaled-position-size",
        type=float,
        default=10.0,
        help="Scaled position size %% for high confidence (default: 10.0)",
    )
    nm.add_argument(
        "--max-position-size",
        type=float,
        default=15.0,
        help="Maximum position size %% hard cap (default: 15.0)",
    )
    nm.add_argument(
        "--stop-loss",
        type=float,
        default=0.15,
        help="Stop loss %% (default: 0.15 = 15%%)",
    )
    nm.add_argument(
        "--profit-target",
        type=float,
        default=0.20,
        help="Profit target %% (default: 0.20 = 20%%)",
    )
    nm.add_argument(
        "--max-hold-hours",
        type=int,
        default=24,
        help="Maximum hold time in hours (default: 24)",
    )
    nm.add_argument(
        "--momentum-exit",
        action="store_true",
        default=True,
        help="Enable momentum-based exits (default: True)",
    )
    nm.add_argument(
        "--no-momentum-exit",
        action="store_false",
        dest="momentum_exit",
        help="Disable momentum-based exits",
    )
    nm.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    nm.set_defaults(func=cmd_news_momentum_scan)

    nmp = sub.add_parser(
        "news-momentum-positions",
        help="Show news-driven momentum positions and check for exits",
    )
    nmp.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for position tracking",
    )
    nmp.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to snapshot for current prices (triggers exit check)",
    )
    nmp.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    nmp.set_defaults(func=cmd_news_momentum_positions)

    # Add trader profiling commands
    from .trader_cli import add_trader_commands

    add_trader_commands(sub)

    # Accounting commands
    acct = sub.add_parser("accounting", help="PnL/NAV accounting for paper copytrade")
    acct_sub = acct.add_subparsers(dest="accounting_cmd", help="Accounting commands")

    # accounting init
    acct_init = acct_sub.add_parser("init", help="Initialize accounting database")
    acct_init.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_init.add_argument("--starting-cash", type=float, default=10000.0, help="Initial cash balance")
    acct_init.set_defaults(func=cmd_accounting_init)

    # accounting record-fill
    acct_fill = acct_sub.add_parser("record-fill", help="Record a paper trade fill")
    acct_fill.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_fill.add_argument("--token-id", type=str, required=True, help="Token ID")
    acct_fill.add_argument("--side", type=str, required=True, choices=["buy", "sell"], help="Trade side")
    acct_fill.add_argument("--size", type=float, required=True, help="Position size")
    acct_fill.add_argument("--price", type=float, required=True, help="Execution price")
    acct_fill.add_argument("--fee", type=float, default=0.0, help="Trading fee")
    acct_fill.add_argument("--market-slug", type=str, default=None, help="Market slug")
    acct_fill.add_argument("--market-question", type=str, default=None, help="Market question")
    acct_fill.add_argument("--trader-source", type=str, default=None, help="Original trader (for copy trading)")
    acct_fill.add_argument("--original-tx", type=str, default=None, help="Original transaction hash")
    acct_fill.add_argument("--timestamp", type=str, default=None, help="ISO timestamp (default: now)")
    acct_fill.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    acct_fill.set_defaults(func=cmd_accounting_record_fill)

    # accounting snapshot
    acct_snap = acct_sub.add_parser("snapshot", help="Record portfolio snapshot")
    acct_snap.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_snap.add_argument("--snapshot-file", type=str, default=None, help="Snapshot JSON file with prices")
    acct_snap.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    acct_snap.set_defaults(func=cmd_accounting_snapshot)

    # accounting summary
    acct_summary = acct_sub.add_parser("summary", help="Show account summary")
    acct_summary.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_summary.add_argument("--days", type=int, default=7, help="Number of days for summary (default: 7)")
    acct_summary.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    acct_summary.set_defaults(func=cmd_accounting_summary)

    # accounting positions
    acct_pos = acct_sub.add_parser("positions", help="Show current positions")
    acct_pos.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_pos.add_argument("--all", action="store_true", help="Show all positions including closed")
    acct_pos.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    acct_pos.set_defaults(func=cmd_accounting_positions)

    # accounting exposures
    acct_exp = acct_sub.add_parser("exposures", help="Show current exposures breakdown")
    acct_exp.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_exp.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    acct_exp.set_defaults(func=cmd_accounting_exposures)

    # accounting cash
    acct_cash = acct_sub.add_parser("cash", help="Show cash ledger")
    acct_cash.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_cash.add_argument("--days", type=int, default=7, help="Number of days (default: 7)")
    acct_cash.add_argument("--limit", type=int, default=50, help="Maximum entries (default: 50)")
    acct_cash.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    acct_cash.set_defaults(func=cmd_accounting_cash)

    # accounting import-fills
    acct_import = acct_sub.add_parser("import-fills", help="Import fills from JSON/JSONL file")
    acct_import.add_argument("--db-path", type=str, default=None, help="Database file path")
    acct_import.add_argument("--input", type=str, required=True, help="Input file (JSON or JSONL)")
    acct_import.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    acct_import.set_defaults(func=cmd_accounting_import_fills)

    args = p.parse_args()

    # Handle --raw flag for microstructure command
    if hasattr(args, "raw") and args.raw:
        args.summary = False

    args.func(args)


if __name__ == "__main__":
    main()
