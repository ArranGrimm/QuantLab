#![recursion_limit = "256"]

//! B3 AMV bull 波段接力策略回测引擎

mod components;
mod resources;
mod systems;

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use clap::Parser;
use polars::prelude::*;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use bt_core::{BacktestStats, Portfolio, SignalArtifactMeta};
use components::{ClosedTrade, ExitReason, Position};
use resources::{BacktestConfig, ConfigFile, DailyData, MarketData, PriceBar};
use systems::{check_sell_conditions, process_buy_signals, update_stats};

#[derive(Parser, Debug)]
#[command(author, version, about = "B3 AMV bull 波段接力策略回测")]
struct Args {
    #[arg(
        short,
        long,
        default_value = "../artifacts/b3_tdx_signals/signal.parquet"
    )]
    data: PathBuf,

    #[arg(short, long, default_value = "crates/b3/config.toml")]
    config: PathBuf,

    #[arg(short, long, default_value = "../results")]
    output_dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let signal_meta = bt_core::load_signal_meta(&args.data);

    println!("========================================");
    println!("   B3 Backtest Engine (Bevy ECS)");
    println!("========================================");

    let config: BacktestConfig = match ConfigFile::load(&args.config) {
        Ok(cfg) => {
            println!("Loaded config from: {:?}", args.config);
            cfg.into()
        }
        Err(e) => {
            println!("Warning: {}. Using defaults.", e);
            BacktestConfig::default()
        }
    };
    print_config(&config);

    println!("\nLoading data from: {:?}", args.data);
    if let Some(meta) = &signal_meta {
        if let Some(signal_run_id) = &meta.signal_run_id {
            println!("Signal Run ID: {}", signal_run_id);
        }
    }

    let df = LazyFrame::scan_parquet(&args.data, Default::default())?.collect()?;
    println!("Loaded {} rows", df.height());

    let (market_data, mut trading_dates) = build_market_data(&df, &config.sort_field)?;
    let original_len = trading_dates.len();
    if let Some(start) = config.start_date {
        trading_dates.retain(|d| *d >= start);
    }
    if let Some(end) = config.end_date {
        trading_dates.retain(|d| *d <= end);
    }

    println!(
        "Stocks: {}, Trading days: {} (filtered from {})",
        market_data.prices.len(),
        trading_dates.len(),
        original_len
    );

    let mut app = App::new();
    let initial_capital = config.initial_capital;
    app.insert_resource(config);
    app.insert_resource(Portfolio::new(initial_capital));
    app.insert_resource(BacktestStats::default());
    app.insert_resource(market_data);
    app.insert_resource(DailyData::default());
    app.add_systems(
        Update,
        (process_buy_signals, check_sell_conditions, update_stats).chain(),
    );

    println!(
        "\nRunning backtest over {} trading days...\n",
        trading_dates.len()
    );
    for date in &trading_dates {
        let world = app.world_mut();
        world.resource_mut::<Portfolio>().current_date = Some(*date);

        let market_data = world.resource::<MarketData>();
        let config = world.resource::<BacktestConfig>();
        let entry_rank_limit = config.entry_rank_limit;
        let min_score = config.min_score;
        let require_bull_regime = config.require_bull_regime;
        let stop_buffer = config.structural_stop_buffer_pct;
        let sort_ascending = config.sort_ascending;

        let mut candidates: Vec<_> = market_data
            .prices
            .iter()
            .filter_map(|(code, dates)| {
                dates.get(date).and_then(|bar| {
                    let regime_ok = !require_bull_regime || bar.is_bull_regime;
                    if bar.is_signal
                        && regime_ok
                        && bar.rank <= entry_rank_limit
                        && bar.score >= min_score
                        && bar.open > 0.0
                        && bar.trigger_low > 0.0
                    {
                        let structural_stop_price = bar.trigger_low * (1.0 - stop_buffer);
                        Some((
                            code.clone(),
                            bar.sort_value,
                            bar.open,
                            structural_stop_price,
                        ))
                    } else {
                        None
                    }
                })
            })
            .collect();
        if sort_ascending {
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        world.resource_mut::<DailyData>().buy_candidates = candidates;
        app.update();
    }

    if let Some(end_date) = trading_dates.last() {
        force_close_all_positions(&mut app, *end_date);
    }
    let (trade_summary_text, trade_summary_json) = build_trade_summary(app.world_mut());

    let stats = app.world().resource::<BacktestStats>();
    let portfolio = app.world().resource::<Portfolio>();
    bt_core::print_results(stats, portfolio);

    if !args.output_dir.is_empty() {
        let config = app.world().resource::<BacktestConfig>();
        let config_text = format_config(config, trading_dates.len());
        let report_paths = bt_core::write_report_bundle(
            &args.output_dir,
            "b3",
            &args.data,
            signal_meta.as_ref(),
            &config_text,
            json!({
                "initial_capital": config.initial_capital,
                "max_positions": config.max_positions,
                "max_daily_buys": config.max_daily_buys,
                "position_size_pct": config.position_size_pct,
                "max_hold_trading_days": config.max_hold_trading_days,
                "start_date": config.start_date.map(|d| d.to_string()),
                "end_date": config.end_date.map(|d| d.to_string()),
                "entry_rank_limit": config.entry_rank_limit,
                "min_score": config.min_score,
                "require_bull_regime": config.require_bull_regime,
                "sort_field": config.sort_field,
                "sort_ascending": config.sort_ascending,
                "structural_stop_enabled": config.structural_stop_enabled,
                "structural_stop_buffer_pct": config.structural_stop_buffer_pct,
                "break_white_line_enabled": config.break_white_line_enabled,
                "break_white_line_buffer_pct": config.break_white_line_buffer_pct,
                "fast_fail_enabled": config.fast_fail_enabled,
                "fast_fail_days": config.fast_fail_days,
                "fast_fail_loss_pct": config.fast_fail_loss_pct,
                "weak_enabled": config.weak_enabled,
                "weak_days": config.weak_days,
                "weak_min_gain_pct": config.weak_min_gain_pct,
                "trailing_enabled": config.trailing_enabled,
                "trailing_activation_pct": config.trailing_activation_pct,
                "trailing_pct": config.trailing_pct,
                "commission_rate": config.commission_rate,
                "stamp_duty_rate": config.stamp_duty_rate,
                "slippage_pct": config.slippage_pct,
            }),
            Some(&trade_summary_text),
            Some(trade_summary_json),
            stats,
            portfolio,
            trading_dates.len(),
        )?;
        append_b3_registry_entry(
            signal_meta.as_ref(),
            &args.data,
            config,
            stats,
            portfolio,
            trading_dates.len(),
            &report_paths.txt_path,
            &report_paths.json_path,
        )?;
    }

    Ok(())
}

#[derive(Default)]
struct ReasonStats {
    count: u32,
    wins: u32,
    total_pnl: f64,
    total_pnl_pct: f64,
    total_hold_trading_days: u32,
}

fn build_trade_summary(world: &mut World) -> (String, serde_json::Value) {
    let mut by_reason: BTreeMap<String, ReasonStats> = BTreeMap::new();
    let mut max_hold_trading_days = 0u32;
    let mut query = world.query::<&ClosedTrade>();

    for trade in query.iter(world) {
        let reason = trade.exit_reason.to_string();
        let stats = by_reason.entry(reason).or_default();
        stats.count += 1;
        stats.wins += u32::from(trade.pnl > 0.0);
        stats.total_pnl += trade.pnl;
        stats.total_pnl_pct += trade.pnl_pct;
        stats.total_hold_trading_days += trade.hold_trading_days;
        max_hold_trading_days = max_hold_trading_days.max(trade.hold_trading_days);
    }

    let mut lines = String::from("--- Trade Summary ---\n");
    lines.push_str(&format!(
        "Max Hold Observed:   {}td\n",
        max_hold_trading_days
    ));
    let mut reason_json = serde_json::Map::new();

    for (reason, stats) in &by_reason {
        let avg_pnl_pct = if stats.count > 0 {
            stats.total_pnl_pct / stats.count as f64 * 100.0
        } else {
            0.0
        };
        let avg_hold = if stats.count > 0 {
            stats.total_hold_trading_days as f64 / stats.count as f64
        } else {
            0.0
        };
        let win_rate = if stats.count > 0 {
            stats.wins as f64 / stats.count as f64 * 100.0
        } else {
            0.0
        };
        lines.push_str(&format!(
            "{:<16} {:>4} trades | Win {:>5.1}% | AvgPnL {:+6.2}% | AvgHold {:>5.1}td | PnL {:+.2}\n",
            reason, stats.count, win_rate, avg_pnl_pct, avg_hold, stats.total_pnl
        ));
        reason_json.insert(
            reason.clone(),
            json!({
                "count": stats.count,
                "win_rate_pct": win_rate,
                "avg_pnl_pct": avg_pnl_pct,
                "avg_hold_trading_days": avg_hold,
                "total_pnl": stats.total_pnl,
            }),
        );
    }

    (
        lines,
        json!({
            "max_hold_observed_trading_days": max_hold_trading_days,
            "exit_reason": reason_json,
        }),
    )
}

fn print_config(config: &BacktestConfig) {
    println!("\n--- B3 Configuration ---");
    println!("Initial Capital: {:.0}", config.initial_capital);
    println!(
        "Max Positions: {} (Daily: {})",
        config.max_positions, config.max_daily_buys
    );
    println!("Position Size: {:.0}%", config.position_size_pct * 100.0);
    println!("Max Hold TD: {}", config.max_hold_trading_days);
    println!("Entry Rank Limit: Top{}", config.entry_rank_limit);
    println!(
        "Sort Field: {} ({})",
        config.sort_field,
        if config.sort_ascending { "ASC" } else { "DESC" }
    );
    println!("Require Bull Regime: {}", config.require_bull_regime);
    println!("Structural Stop: {}", config.structural_stop_enabled);
    println!("Break White Line: {}", config.break_white_line_enabled);
    println!(
        "Fast Fail: {}td @ -{:.1}% ({})",
        config.fast_fail_days,
        config.fast_fail_loss_pct * 100.0,
        if config.fast_fail_enabled {
            "ON"
        } else {
            "OFF"
        }
    );
    println!(
        "Weak Exit: {}td @ +{:.1}% ({})",
        config.weak_days,
        config.weak_min_gain_pct * 100.0,
        if config.weak_enabled { "ON" } else { "OFF" }
    );
    println!(
        "Trailing Stop: Activate={:.0}%, Trail={:.0}% ({})",
        config.trailing_activation_pct * 100.0,
        config.trailing_pct * 100.0,
        if config.trailing_enabled { "ON" } else { "OFF" }
    );
    println!("------------------------");
}

fn format_config(config: &BacktestConfig, trading_days: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "--- Configuration ---").unwrap();
    writeln!(s, "Initial Capital:      {:.0}", config.initial_capital).unwrap();
    writeln!(
        s,
        "Max Positions:        {} (Daily: {})",
        config.max_positions, config.max_daily_buys
    )
    .unwrap();
    writeln!(
        s,
        "Position Size:        {:.0}%",
        config.position_size_pct * 100.0
    )
    .unwrap();
    writeln!(s, "Max Hold TD:          {}", config.max_hold_trading_days).unwrap();
    writeln!(s, "Trading Days:         {}", trading_days).unwrap();
    writeln!(s, "Entry Rank Limit:     Top{}", config.entry_rank_limit).unwrap();
    writeln!(
        s,
        "Sort Field:           {} ({})",
        config.sort_field,
        if config.sort_ascending { "ASC" } else { "DESC" }
    )
    .unwrap();
    writeln!(s, "Require Bull Regime:  {}", config.require_bull_regime).unwrap();
    writeln!(
        s,
        "Structural Stop:      {}",
        config.structural_stop_enabled
    )
    .unwrap();
    writeln!(
        s,
        "Break White Line:     {}",
        config.break_white_line_enabled
    )
    .unwrap();
    writeln!(
        s,
        "Fast Fail:            {}td @ -{:.1}% ({})",
        config.fast_fail_days,
        config.fast_fail_loss_pct * 100.0,
        if config.fast_fail_enabled {
            "ON"
        } else {
            "OFF"
        }
    )
    .unwrap();
    writeln!(
        s,
        "Weak Exit:            {}td @ +{:.1}% ({})",
        config.weak_days,
        config.weak_min_gain_pct * 100.0,
        if config.weak_enabled { "ON" } else { "OFF" }
    )
    .unwrap();
    writeln!(
        s,
        "Trailing Stop:        Activate={:.0}%, Trail={:.0}% ({})",
        config.trailing_activation_pct * 100.0,
        config.trailing_pct * 100.0,
        if config.trailing_enabled { "ON" } else { "OFF" }
    )
    .unwrap();
    writeln!(
        s,
        "Commission:           {:.4}%",
        config.commission_rate * 100.0
    )
    .unwrap();
    writeln!(
        s,
        "Stamp Duty:           {:.3}%",
        config.stamp_duty_rate * 100.0
    )
    .unwrap();
    writeln!(
        s,
        "Slippage:             {:.2}%",
        config.slippage_pct * 100.0
    )
    .unwrap();
    s
}

fn append_b3_registry_entry(
    signal_meta: Option<&SignalArtifactMeta>,
    data_path: &Path,
    config: &BacktestConfig,
    stats: &BacktestStats,
    portfolio: &Portfolio,
    trading_days: usize,
    report_txt_path: &Path,
    report_json_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use chrono::Local;

    let registry_path =
        bt_core::resolve_registry_path(signal_meta, "../artifacts/b3/backtest.jsonl");
    let derived = bt_core::calc_derived_metrics(stats, portfolio, trading_days);
    let train_run_dir = registry_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let signal_path_resolved = signal_meta.and_then(bt_core::resolve_signal_path);
    let signal_meta_resolved = signal_meta.and_then(|meta| {
        bt_core::resolve_meta_relative_path(meta, meta.signal_meta_path.as_deref())
    });
    let report_dir = report_json_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let backtest_id = report_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| Local::now().format("%Y%m%d_%H%M%S_%3f").to_string());

    let record = json!({
        "record_type": "backtest_run",
        "recorded_at": Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        "strategy": "b3",
        "signal_run_id": signal_meta.and_then(|meta| meta.signal_run_id.clone()),
        "label": signal_meta.and_then(|meta| meta.label.clone()),
        "backtest_id": backtest_id,
        "input_signal_file": bt_core::relative_portable_path(&train_run_dir, data_path),
        "signal_path": signal_path_resolved.as_ref().map(|p| bt_core::relative_portable_path(&train_run_dir, p)),
        "signal_meta_path": signal_meta_resolved.as_ref().map(|p| bt_core::relative_portable_path(&train_run_dir, p)),
        "backtest_dir": bt_core::relative_portable_path(&train_run_dir, &report_dir),
        "git_commit": signal_meta.and_then(|meta| meta.git_commit.clone()),
        "max_positions": config.max_positions,
        "max_daily_buys": config.max_daily_buys,
        "position_size_pct": config.position_size_pct,
        "max_hold_trading_days": config.max_hold_trading_days,
        "entry_rank_limit": config.entry_rank_limit,
        "min_score": config.min_score,
        "require_bull_regime": config.require_bull_regime,
        "sort_field": config.sort_field,
        "sort_ascending": config.sort_ascending,
        "structural_stop_enabled": config.structural_stop_enabled,
        "break_white_line_enabled": config.break_white_line_enabled,
        "fast_fail_enabled": config.fast_fail_enabled,
        "weak_enabled": config.weak_enabled,
        "trailing_enabled": config.trailing_enabled,
        "commission_rate": config.commission_rate,
        "stamp_duty_rate": config.stamp_duty_rate,
        "slippage_pct": config.slippage_pct,
        "gross_return_pct": derived.gross_return_pct,
        "net_return_pct": derived.total_return_pct,
        "max_drawdown_pct": stats.max_drawdown * 100.0,
        "win_rate_pct": stats.win_rate() * 100.0,
        "avg_trades_per_day": derived.avg_trades_per_day,
        "total_trades": stats.total_trades,
        "report_txt_path": bt_core::relative_portable_path(&train_run_dir, report_txt_path),
        "report_json_path": bt_core::relative_portable_path(&train_run_dir, report_json_path),
    });

    bt_core::append_jsonl_record(&registry_path, &record)?;
    println!("Registry appended: {}", registry_path.display());
    Ok(())
}

fn build_market_data(
    df: &DataFrame,
    sort_field: &str,
) -> Result<(MarketData, Vec<NaiveDate>), Box<dyn std::error::Error>> {
    let mut market_data = MarketData::default();
    let mut all_dates = Vec::new();

    let codes = df.column("code")?.str()?;
    let dates = df.column("date")?.date()?;
    let open_adj = df.column("open_adj")?.f64()?;
    let high_adj = df.column("high_adj")?.f64()?;
    let low_adj = df.column("low_adj")?.f64()?;
    let close_adj = df.column("close_adj")?.f64()?;
    let pre_close_adj = df.column("pre_close_adj")?.f64()?;
    let white_line = df.column("white_line")?.f64()?;
    let yellow_line = df.column("yellow_line")?.f64()?;
    let is_signal = df.column("is_signal")?.bool()?;
    let is_bull_regime = df.column("is_bull_regime")?.bool()?;
    let score = df.column("score")?.f64()?;
    let sort_col = df
        .column(sort_field)
        .map_err(|_| format!("Sort field '{}' not found in parquet", sort_field))?
        .f64()?;
    let rank = df.column("rank")?.u32()?;
    let trigger_low = df.column("trigger_low")?.f64()?;
    let trigger_high = df.column("trigger_high")?.f64()?;

    for i in 0..df.height() {
        let code = codes.get(i).ok_or("Missing code")?;
        let date_days = dates.get(i).ok_or("Missing date")?;
        let date = bt_core::epoch_days_to_date(date_days).ok_or("Invalid date")?;
        let bar = PriceBar {
            open: open_adj.get(i).unwrap_or(0.0),
            high: high_adj.get(i).unwrap_or(0.0),
            low: low_adj.get(i).unwrap_or(0.0),
            close: close_adj.get(i).unwrap_or(0.0),
            pre_close: pre_close_adj.get(i).unwrap_or(0.0),
            white_line: white_line.get(i).unwrap_or(0.0),
            yellow_line: yellow_line.get(i).unwrap_or(0.0),
            is_signal: is_signal.get(i).unwrap_or(false),
            is_bull_regime: is_bull_regime.get(i).unwrap_or(false),
            score: score.get(i).unwrap_or(0.0),
            sort_value: sort_col.get(i).unwrap_or(0.0),
            rank: rank.get(i).unwrap_or(9999),
            trigger_low: trigger_low.get(i).unwrap_or(0.0),
            trigger_high: trigger_high.get(i).unwrap_or(0.0),
        };
        market_data
            .prices
            .entry(code.to_string())
            .or_insert_with(HashMap::new)
            .insert(date, bar);
        all_dates.push(date);
    }

    all_dates.sort();
    all_dates.dedup();
    Ok((market_data, all_dates))
}

fn force_close_all_positions(app: &mut App, end_date: NaiveDate) {
    let world = app.world_mut();
    let market_data = world.resource::<MarketData>().clone();
    let config = world.resource::<BacktestConfig>().clone();

    let mut to_close: Vec<(Entity, Position, f64)> = Vec::new();
    {
        let mut query = world.query::<(Entity, &Position)>();
        for (entity, position) in query.iter(world) {
            if let Some(prices) = market_data.prices.get(&position.code) {
                let exit_price = prices
                    .iter()
                    .filter(|(d, _)| **d <= end_date)
                    .max_by_key(|(d, _)| *d)
                    .map(|(_, bar)| bar.close);
                if let Some(price) = exit_price {
                    to_close.push((entity, position.clone(), price));
                }
            }
        }
    }

    for (entity, position, exit_price) in to_close {
        let gross = position.shares as f64 * exit_price;
        let commission = gross * config.commission_rate;
        let stamp_duty = gross * config.stamp_duty_rate;
        let slippage = gross * config.slippage_pct;
        let net = gross - commission - stamp_duty - slippage;
        let trade_pnl = net - position.cost;
        let initial_cost = position.initial_shares as f64 * position.entry_price;
        let trade_pnl_pct = trade_pnl / initial_cost;

        world.resource_mut::<Portfolio>().cash += net;
        world
            .resource_mut::<BacktestStats>()
            .record_trade(trade_pnl, commission, stamp_duty, slippage);

        println!(
            "[{}] [CLOSE] {} @ {:.2} | PnL {:+.2}% | Hold {}td | EndOfBacktest",
            end_date,
            position.code,
            exit_price,
            trade_pnl_pct * 100.0,
            position.hold_trading_days
        );

        world.entity_mut(entity).insert(ClosedTrade {
            code: position.code.clone(),
            entry_date: position.entry_date,
            exit_date: end_date,
            entry_price: position.entry_price,
            exit_price,
            shares: position.initial_shares,
            pnl: trade_pnl,
            pnl_pct: trade_pnl_pct,
            hold_trading_days: position.hold_trading_days,
            exit_reason: ExitReason::EndOfBacktest,
        });
        world.entity_mut(entity).remove::<Position>();
    }
}
