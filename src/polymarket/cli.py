from __future__ import annotations

import argparse
import json
import logging
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
        )

        if args.format == "json":
            print(json.dumps({"results": results}, indent=2))
        else:
            print("=" * 80)
            print("ORDERBOOK IMBALANCE STRATEGY - PARAMETER SWEEP")
            print("=" * 80)
            print(f"Snapshots analyzed: {len(snapshots)}")
            print(f"Target market: {args.target}")
            print(
                f"\n{'Rank':<6}{'k':<4}{'theta':<8}{'p_max':<8}{'Trades':<8}{'UP':<6}{'DOWN':<8}{'Avg Conf':<10}"
            )
            print("-" * 80)

            for i, result in enumerate(results[:10], 1):  # Top 10
                p = result["params"]
                m = result["metrics"]
                print(
                    f"{i:<6}{p['k']:<4}{p['theta']:<8.2f}{p['p_max']:<8.2f}"
                    f"{m['total_trades']:<8}{m['up_trades']:<6}{m['down_trades']:<8}"
                    f"{m['avg_confidence']:<10.3f}"
                )

            print("=" * 80)

    else:
        # Single backtest run
        result = run_backtest(
            snapshots=snapshots,
            k=args.k,
            theta=args.theta,
            p_max=args.p_max,
            target_market_substring=args.target,
        )

        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("=" * 80)
            print("ORDERBOOK IMBALANCE STRATEGY - BACKTEST RESULTS")
            print("=" * 80)
            print(f"Snapshots analyzed: {len(snapshots)}")
            print(f"Target market: {args.target}")
            print("\nParameters:")
            print(f"  k (depth levels):     {args.k}")
            print(f"  theta (threshold):    {args.theta:.2f}")
            print(f"  p_max (max price):    {args.p_max:.2f}")

            print("\n--- Results ---")
            print(f"Total trades:     {result.metrics['total_trades']}")
            print(f"UP trades:        {result.metrics['up_trades']}")
            print(f"DOWN trades:      {result.metrics['down_trades']}")
            print(f"Avg confidence:   {result.metrics['avg_confidence']:.3f}")
            print(f"Avg entry price:  {result.metrics['avg_entry_price']:.3f}")

            if result.trades:
                print("\n--- Recent Trades (last 10) ---")
                for t in result.trades[-10:]:
                    print(
                        f"  {t.timestamp.strftime('%H:%M')} | {t.decision:<6} | "
                        f"imb={t.imbalance_value:.3f} | mid={t.mid_yes:.3f} | "
                        f"entry={t.entry_price:.3f} | conf={t.confidence:.2f}"
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

    ms = sub.add_parser("microstructure", help="Analyze market microstructure from snapshot")
    ms.add_argument(
        "--snapshot",
        required=True,
        help=(
            "Path to snapshot JSON file. "
            "Supports pointer files like data/latest_15m.json"
        ),
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

    args = p.parse_args()

    # Handle --raw flag for microstructure command
    if hasattr(args, "raw") and args.raw:
        args.summary = False

    args.func(args)


if __name__ == "__main__":
    main()
