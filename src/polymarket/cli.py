from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import clob
from .microstructure import (
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_EXTREME_PIN_THRESHOLD,
    DEFAULT_SPREAD_ALERT_THRESHOLD,
    analyze_snapshot_microstructure,
    generate_microstructure_summary,
    log_microstructure_alerts,
)


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
                json.dumps(
                    {"error": f"No fills.json or fills.jsonl found in {args.data_dir}"}
                ),
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)
    else:
        print(
            json.dumps(
                {"error": "Must specify --input or --data-dir"}
            ),
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
        snapshot_path = Path(args.snapshot)
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
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(
            json.dumps({"error": f"Snapshot file not found: {args.snapshot}"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    if args.summary:
        # Generate summary with alerts
        summary = generate_microstructure_summary(
            snapshot_path=snapshot_path,
            target_market_substring=args.target,
            spread_threshold=float(args.spread_threshold),
            extreme_pin_threshold=float(args.extreme_pin_threshold),
            depth_levels=int(args.depth_levels),
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
            print(f"  Spread alert:     > {summary['spread_threshold']:.2f}")
            print(
                f"  Extreme pin:      <= {summary['extreme_pin_threshold']:.2f} "
                f"or >= {1.0 - summary['extreme_pin_threshold']:.2f}"
            )
            print(f"  Depth levels:     {summary['depth_levels']}")

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


def cmd_binance_collect(args: argparse.Namespace) -> None:
    """Collect Binance BTC market data (REST API single snapshot)."""
    from pathlib import Path

    from .binance_collector import collect_snapshot_rest

    out_dir = Path(args.out)
    out_path = collect_snapshot_rest(
        out_dir=out_dir,
        symbol=args.symbol,
        kline_intervals=args.intervals,
    )
    print(str(out_path))


def cmd_binance_loop(args: argparse.Namespace) -> None:
    """Run Binance WebSocket collector loop."""
    from pathlib import Path

    from .binance_collector import run_collector_loop

    run_collector_loop(
        out_dir=Path(args.out),
        symbol=args.symbol,
        kline_intervals=args.intervals,
        snapshot_interval_seconds=float(args.snapshot_interval_seconds),
        max_reconnect_delay=float(args.max_reconnect_delay),
        retention_hours=args.retention_hours,
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


def cmd_sports_scan(args: argparse.Namespace) -> None:
    """Scan for sports arbitrage opportunities."""
    from decimal import Decimal

    from .sports_arbitrage import SportsArbitrageStrategy

    strategy = SportsArbitrageStrategy(
        bankroll=Decimal(str(args.bankroll)),
    )

    opportunities = strategy.scan()

    if args.format == "json":
        print(json.dumps([opp.to_dict() for opp in opportunities], indent=2))
    else:
        print("=" * 80)
        print("SPORTS ARBITRAGE SCAN")
        print("=" * 80)
        print(f"Opportunities found: {len(opportunities)}")
        print(f"Min edge: {args.min_edge}%")
        print()

        for i, opp in enumerate(opportunities[: args.limit], 1):
            print(f"--- Opportunity {i} ---")
            print(f"Market: {opp.pm_market.get('question', 'N/A')}")
            print(f"Side: {opp.side.upper()}")
            print(f"Polymarket implied: {float(opp.pm_implied):.2%}")
            print(f"Sharp book implied: {float(opp.sharp_implied):.2%}")
            print(f"Edge: {float(opp.edge):.2%}")
            print(f"Edge after fees: {float(opp.edge_after_fees):.2%}")
            print(f"Confidence: {float(opp.confidence):.2%}")
            print()

        print("=" * 80)


def cmd_sports_trade(args: argparse.Namespace) -> None:
    """Execute paper trades for arbitrage opportunities."""
    from decimal import Decimal

    from .sports_arbitrage import SportsArbitrageStrategy

    strategy = SportsArbitrageStrategy(
        bankroll=Decimal(str(args.bankroll)),
    )

    if args.scan:
        # Scan and trade all opportunities
        opportunities = strategy.scan()
        trades = []
        for opp in opportunities:
            if opp.is_valid:
                trade = strategy.paper_trade(opp)
                trades.append(trade)

        if args.format == "json":
            print(json.dumps([t.to_dict() for t in trades], indent=2))
        else:
            print("=" * 80)
            print("SPORTS ARBITRAGE PAPER TRADES")
            print("=" * 80)
            print(f"Opportunities found: {len(opportunities)}")
            print(f"Trades executed: {len(trades)}")
            print()

            for t in trades:
                print(f"Trade: {t.trade_id}")
                print(f"  Market: {t.pm_market_id}")
                print(f"  Side: {t.side.upper()}")
                print(f"  Size: ${float(t.size):.2f}")
                print(f"  Entry: {float(t.entry_price):.4f}")
                print(f"  Edge: {float(t.edge_at_entry):.2%}")
                print()

            print("=" * 80)


def cmd_sports_stats(args: argparse.Namespace) -> None:
    """Show sports arbitrage strategy statistics."""
    from decimal import Decimal

    from .sports_arbitrage import SportsArbitrageStrategy

    strategy = SportsArbitrageStrategy(
        bankroll=Decimal(str(args.bankroll)),
    )

    stats = strategy.get_stats()

    if args.format == "json":
        # Convert Decimals to strings for JSON serialization
        json_stats = {k: str(v) if isinstance(v, Decimal) else v for k, v in stats.items()}
        print(json.dumps(json_stats, indent=2))
    else:
        print("=" * 80)
        print("SPORTS ARBITRAGE STATISTICS")
        print("=" * 80)
        print(f"Total opportunities detected: {stats['total_opportunities']}")
        print(f"Total trades: {stats['total_trades']}")
        print(f"Open trades: {stats['open_trades']}")
        print(f"Closed trades: {stats['closed_trades']}")
        print(f"Total PnL: ${float(stats['total_pnl']):,.2f}")
        print(f"Average edge: {float(stats['avg_edge']):.2%}")
        print("=" * 80)


def cmd_weather_consensus_scan(args: argparse.Namespace) -> None:
    """Scan for weather consensus mispricing opportunities."""
    from pathlib import Path

    from .strategy_weather_consensus import run_consensus_scan

    snapshots_dir = Path(args.snapshots_dir) if args.snapshots_dir else None
    cities = args.cities.split(",") if args.cities else None

    result = run_consensus_scan(
        snapshots_dir=snapshots_dir,
        cities=cities,
        dry_run=not args.live,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print("WEATHER CONSENSUS MISPRICING SCAN")
        print("=" * 70)
        print(f"Scan time: {result['timestamp']}")
        print(f"Markets scanned: {result['markets_scanned']}")
        print(f"Signals generated: {result['signals_generated']}")
        print(f"Actionable signals: {result['actionable_signals']}")
        print(f"Trades executed: {result['trades_executed']}")
        print(f"Exits triggered: {result['exits_triggered']}")
        print(f"Positions open: {result['positions_open']}")
        print(f"Daily exposure: ${result['daily_exposure']:.2f}")
        print(f"Dry run: {result['dry_run']}")

        # Show consensus
        if result["consensus"]:
            print("\n--- Model Consensus ---")
            for city, cons in result["consensus"].items():
                print(
                    f"  {city}: high={cons['consensus_high']:.1f}°F, "
                    f"models={cons['model_count']}, agreement={cons['agreement_score']:.2f}"
                )

        # Show signals
        if result["signals"]:
            print("\n--- All Signals ---")
            for sig in result["signals"]:
                market_q = (
                    sig["market"]["question"][:45] if sig["market"]["question"] else "Unknown"
                )
                extreme = " [!]" if sig["is_extreme_mispricing"] else ""
                print(
                    f"  {sig['side']:<12} | edge={sig['edge']:+.2f} | "
                    f"EV={sig['expected_value']:.3f} | agr={sig['model_agreement']:.2f}{extreme}"
                )
                print(f"               | {market_q}...")

        # Show positions
        if result["positions"]:
            print("\n--- Open Positions ---")
            for pos in result["positions"]:
                sig = pos["entry_signal"]
                print(
                    f"  {sig['side']:<12} | ${pos['position_size']:.2f} | "
                    f"@{pos['entry_price']:.3f} | {sig['market']['city']}"
                )

        print("\n" + "=" * 70)


def cmd_weather_consensus_loop(args: argparse.Namespace) -> None:
    """Run continuous weather consensus scanning loop."""
    from pathlib import Path

    from .strategy_weather_consensus import run_consensus_loop

    snapshots_dir = Path(args.snapshots_dir) if args.snapshots_dir else None
    cities = args.cities.split(",") if args.cities else None

    run_consensus_loop(
        snapshots_dir=snapshots_dir,
        cities=cities,
        interval_seconds=args.interval,
        dry_run=not args.live,
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


def cmd_copytrade_loop(args: argparse.Namespace) -> None:
    """Run copytrade accounting loop."""
    from decimal import Decimal
    from pathlib import Path

    from .copytrade_loop import CopytradeConfig, copytrade_loop

    config = CopytradeConfig(
        wallet_address=args.wallet,
        data_dir=Path(args.data_dir),
        starting_cash=Decimal(str(args.starting_cash)),
    )

    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None

    try:
        copytrade_loop(
            config=config,
            fill_collection_interval_seconds=args.interval_seconds,
            pnl_verification_time_utc=args.pnl_time,
            snapshot_dir=snapshot_dir,
            max_backoff_seconds=args.max_backoff_seconds,
        )
    except KeyboardInterrupt:
        print("\nCopytrade loop stopped.")


def cmd_copytrade_pnl(args: argparse.Namespace) -> None:
    """Run single PnL verification on copytrade fills."""
    from decimal import Decimal
    from pathlib import Path

    from .copytrade_loop import run_single_pnl_verify

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    snapshot_path = Path(args.snapshot) if args.snapshot else None

    summary_path = run_single_pnl_verify(
        data_dir=data_dir,
        output_dir=output_dir,
        snapshot_path=snapshot_path,
        starting_cash=Decimal(str(args.starting_cash)),
    )

    if summary_path is None:
        print(
            json.dumps({"error": "PnL verification failed - check logs"}),
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    if args.format == "json":
        print(json.dumps({"summary_path": str(summary_path)}, indent=2))
    else:
        print(f"PnL summary saved to: {summary_path}")
        # Also print a summary
        import json as json_mod

        data = json_mod.loads(summary_path.read_text())
        print("\n" + "=" * 70)
        print("COPYTRADE PnL SUMMARY")
        print("=" * 70)
        print(f"Generated: {data['metadata']['generated_at']}")
        print(f"Fills: {data['summary']['total_fills']}")
        print(f"Positions: {data['summary']['unique_tokens']}")
        print()
        print(f"Realized PnL:   ${data['pnl']['realized_pnl']:,.2f}")
        print(f"Unrealized PnL: ${data['pnl']['unrealized_pnl']:,.2f}")
        print(f"Total Fees:     ${data['pnl']['total_fees']:,.2f}")
        print(f"Net PnL:        ${data['pnl']['net_pnl']:,.2f}")
        print("=" * 70)


def cmd_copytrade_collect(args: argparse.Namespace) -> None:
    """Collect fills for a wallet (one-time)."""
    from datetime import datetime
    from pathlib import Path

    from .copytrade_loop import CopytradeConfig, collect_fills

    config = CopytradeConfig(
        wallet_address=args.wallet,
        data_dir=Path(args.data_dir),
    )

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))

    result = collect_fills(config, since=since)

    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        status = "✓ SUCCESS" if not result.errors else "✗ ERRORS"
        print(f"{status}: Fill collection complete")
        print(f"  New fills: {result.new_fills}")
        print(f"  Total fills: {result.total_fills}")
        if result.last_fill_timestamp:
            print(f"  Latest fill: {result.last_fill_timestamp}")
        if result.errors:
            print(f"  Errors: {result.errors}")


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
            print(f"\n{'Rank':<6}{'k':<4}{'theta':<8}{'p_max':<8}{'Trades':<8}{'UP':<6}{'DOWN':<8}{'Avg Conf':<10}")
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
        help="Path to collector snapshot for orderbook data",
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
    ms.add_argument("--snapshot", required=True, help="Path to snapshot JSON file")
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
        "--depth-levels",
        type=int,
        default=DEFAULT_DEPTH_LEVELS,
        help=f"Number of book levels for depth calc (default: {DEFAULT_DEPTH_LEVELS})",
    )
    ms.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    ms.set_defaults(func=cmd_microstructure)

    # Binance commands
    bc = sub.add_parser("binance-collect", help="Collect Binance BTC market data (single snapshot)")
    bc.add_argument("--out", default="data/binance", help="Output directory")
    bc.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)")
    bc.add_argument("--intervals", nargs="+", default=["1m", "5m"], help="Kline intervals to fetch")
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
    bcl.set_defaults(func=cmd_binance_loop)

    bf = sub.add_parser("binance-align", help="Align Binance features to Polymarket snapshots")
    bf.add_argument("--binance-dir", default="data/binance", help="Binance data directory")
    bf.add_argument("--polymarket-dir", default="data", help="Polymarket data directory")
    bf.add_argument("--out", default="data/aligned_features.json", help="Output file")
    bf.add_argument("--tolerance", type=float, default=1.0, help="Alignment tolerance in seconds")
    bf.set_defaults(func=cmd_binance_features)

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

    # Sports arbitrage commands
    sp = sub.add_parser("sports-scan", help="Scan for sports arbitrage opportunities")
    sp.add_argument("--bankroll", type=float, default=10000.0, help="Paper trading bankroll")
    sp.add_argument("--min-edge", type=float, default=2.0, help="Minimum edge percentage")
    sp.add_argument("--limit", type=int, default=10, help="Max opportunities to display")
    sp.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    sp.set_defaults(func=cmd_sports_scan)

    spt = sub.add_parser("sports-trade", help="Execute paper trades for arbitrage opportunities")
    spt.add_argument("--bankroll", type=float, default=10000.0, help="Paper trading bankroll")
    spt.add_argument("--scan", action="store_true", default=True, help="Scan before trading")
    spt.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    spt.set_defaults(func=cmd_sports_trade)

    sps = sub.add_parser("sports-stats", help="Show sports arbitrage strategy statistics")
    sps.add_argument("--bankroll", type=float, default=10000.0, help="Paper trading bankroll")
    sps.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    sps.set_defaults(func=cmd_sports_stats)

    # Weather consensus scan command
    wcs = sub.add_parser(
        "weather-consensus-scan",
        help="Scan for weather model consensus mispricing opportunities",
    )
    wcs.add_argument(
        "--snapshots-dir",
        type=str,
        default=None,
        help="Directory containing market snapshots",
    )
    wcs.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated list of cities (default: nyc,chicago,dallas,miami,london)",
    )
    wcs.add_argument(
        "--live",
        action="store_true",
        help="Execute live trades (default: dry-run)",
    )
    wcs.add_argument("--format", choices=["json", "human"], default="human", help="Output format")
    wcs.set_defaults(func=cmd_weather_consensus_scan)

    # Weather consensus loop command
    wcl = sub.add_parser(
        "weather-consensus-loop",
        help="Run continuous weather consensus scanning loop",
    )
    wcl.add_argument(
        "--snapshots-dir",
        type=str,
        default=None,
        help="Directory containing market snapshots",
    )
    wcl.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated list of cities (default: nyc,chicago,dallas,miami,london)",
    )
    wcl.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between scans (default: 300)",
    )
    wcl.add_argument(
        "--live",
        action="store_true",
        help="Execute live trades (default: dry-run)",
    )
    wcl.set_defaults(func=cmd_weather_consensus_loop)

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

    # Copytrade commands
    ct = sub.add_parser(
        "copytrade-loop",
        help="Run copytrade accounting loop (collect fills + daily PnL)",
    )
    ct.add_argument(
        "--wallet",
        required=True,
        help="Wallet address to copytrade",
    )
    ct.add_argument(
        "--data-dir",
        default="data/copytrade",
        help="Data directory for fills (default: data/copytrade)",
    )
    ct.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Fill collection interval in seconds (default: 300 = 5min)",
    )
    ct.add_argument(
        "--pnl-time",
        type=str,
        default="00:00",
        help="Daily PnL verification time UTC (HH:MM format, default: 00:00)",
    )
    ct.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Directory with collector snapshots for price data",
    )
    ct.add_argument(
        "--starting-cash",
        type=float,
        default=10000.0,
        help="Starting cash balance (default: 10000)",
    )
    ct.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=300.0,
        help="Max backoff on errors (default: 300)",
    )
    ct.set_defaults(func=cmd_copytrade_loop)

    ct_pnl = sub.add_parser(
        "copytrade-pnl",
        help="Run single PnL verification on copytrade fills",
    )
    ct_pnl.add_argument(
        "--data-dir",
        default="data/copytrade",
        help="Data directory containing fills.jsonl (default: data/copytrade)",
    )
    ct_pnl.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PnL report (default: data-dir/pnl)",
    )
    ct_pnl.add_argument(
        "--snapshot",
        default=None,
        help="Path to collector snapshot for price data",
    )
    ct_pnl.add_argument(
        "--starting-cash",
        type=float,
        default=0.0,
        help="Starting cash balance (default: 0)",
    )
    ct_pnl.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    ct_pnl.set_defaults(func=cmd_copytrade_pnl)

    ct_collect = sub.add_parser(
        "copytrade-collect",
        help="Collect fills for a wallet (one-time)",
    )
    ct_collect.add_argument(
        "--wallet",
        required=True,
        help="Wallet address to collect fills for",
    )
    ct_collect.add_argument(
        "--data-dir",
        default="data/copytrade",
        help="Data directory for fills (default: data/copytrade)",
    )
    ct_collect.add_argument(
        "--since",
        default=None,
        help="ISO timestamp to collect from (inclusive)",
    )
    ct_collect.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum fills to fetch (default: 100)",
    )
    ct_collect.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    ct_collect.set_defaults(func=cmd_copytrade_collect)

    # Handle --raw flag for microstructure command
    if hasattr(args, "raw") and args.raw:
        args.summary = False

    args.func(args)


if __name__ == "__main__":
    main()
