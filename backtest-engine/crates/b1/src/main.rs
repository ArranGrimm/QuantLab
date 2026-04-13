#![recursion_limit = "256"]

//! B1 超跌反转策略回测引擎
//!
//! Usage:
//!   cargo run -p bt-b1 --release -- --data ../../data/signals/market_data.parquet
//!   cargo run -p bt-b1 --release -- --config config.toml --data ../../data/signals/market_data.parquet

mod components;
mod resources;
mod systems;

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use clap::Parser;
use polars::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bt_core::{BacktestStats, Portfolio, SignalArtifactMeta};
use components::Position;
use resources::{BacktestConfig, ConfigFile, DailyData, MarketData, PriceBar};
use systems::{check_sell_conditions, process_buy_signals, update_stats};

#[derive(Parser, Debug)]
#[command(author, version, about = "B1 超跌反转策略回测")]
struct Args {
    #[arg(short, long, default_value = "../data/signals/market_data.parquet")]
    data: PathBuf,

    #[arg(short, long, default_value = "crates/b1/config.toml")]
    config: PathBuf,

    /// 结果输出目录 (留空则不保存)
    #[arg(short, long, default_value = "../results")]
    output_dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let signal_meta = bt_core::load_signal_meta(&args.data);

    println!("========================================");
    println!("   B1 Backtest Engine (Bevy ECS)");
    println!("========================================");

    // 1. Load config
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

    // 2. Load market data
    let df = LazyFrame::scan_parquet(&args.data, Default::default())?.collect()?;
    println!("Loaded {} rows", df.height());

    // 3. Build market data structure
    let (market_data, mut trading_dates) = build_market_data(&df, &config.sort_field)?;

    // 4. Apply date range filter
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

    // 5. Initialize Bevy App
    let mut app = App::new();
    let initial_capital = config.initial_capital;
    app.insert_resource(config);
    app.insert_resource(Portfolio::new(initial_capital));
    app.insert_resource(BacktestStats::default());
    app.insert_resource(market_data);
    app.insert_resource(DailyData::default());

    // 先买入(开盘)，再卖出(收盘)
    app.add_systems(
        Update,
        (process_buy_signals, check_sell_conditions, update_stats).chain(),
    );

    // 6. Run backtest
    println!("\nRunning backtest over {} trading days...\n", trading_dates.len());

    for date in &trading_dates {
        let world = app.world_mut();
        world.resource_mut::<Portfolio>().current_date = Some(*date);

        let market_data = world.resource::<MarketData>();
        let config = world.resource::<BacktestConfig>();
        let sort_ascending = config.sort_ascending;
        let stop_loss_pct = config.stop_loss_pct;
        let min_score = config.min_score;

        let mut candidates: Vec<_> = market_data
            .prices
            .iter()
            .filter_map(|(code, dates)| {
                dates.get(date).and_then(|bar| {
                    if bar.pre_b1_signal && bar.is_loose && bar.sort_value >= min_score {
                        let stop_price = bar.low * (1.0 - stop_loss_pct);
                        Some((code.clone(), bar.sort_value, bar.open, stop_price))
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

    // 7. Force close remaining positions
    if let Some(end_date) = trading_dates.last() {
        force_close_all_positions(&mut app, *end_date);
    }

    // 8. Print results
    let stats = app.world().resource::<BacktestStats>();
    let portfolio = app.world().resource::<Portfolio>();
    bt_core::print_results(stats, portfolio);

    // 9. Save report
    if !args.output_dir.is_empty() {
        let config = app.world().resource::<BacktestConfig>();
        let config_text = format_config(config, trading_dates.len());
        let report_paths = bt_core::write_report_bundle(
            &args.output_dir,
            "b1",
            &args.data,
            signal_meta.as_ref(),
            &config_text,
            json!({
                "initial_capital": config.initial_capital,
                "max_positions": config.max_positions,
                "max_daily_buys": config.max_daily_buys,
                "position_size_pct": config.position_size_pct,
                "max_hold_days": config.max_hold_days,
                "start_date": config.start_date.map(|d| d.to_string()),
                "end_date": config.end_date.map(|d| d.to_string()),
                "sort_field": config.sort_field,
                "sort_ascending": config.sort_ascending,
                "min_position_ratio": config.min_position_ratio,
                "min_score": config.min_score,
                "stop_loss_enabled": config.stop_loss_enabled,
                "stop_loss_pct": config.stop_loss_pct,
                "tp1_pct": config.tp1_pct,
                "tp2_pct": config.tp2_pct,
                "tp_sell_ratio": config.tp_sell_ratio,
                "sell_on_break_wl": config.sell_on_break_wl,
                "sell_on_break_yl": config.sell_on_break_yl,
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
            None,
            None,
            stats,
            portfolio,
            trading_dates.len(),
        )?;
        append_b1_registry_entry(
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

fn print_config(config: &BacktestConfig) {
    println!("\n--- B1 Configuration ---");
    println!("Initial Capital: {:.0}", config.initial_capital);
    println!("Max Positions: {} (Daily: {})", config.max_positions, config.max_daily_buys);
    println!("Position Size: {:.0}%", config.position_size_pct * 100.0);
    println!("Max Hold Days: {}", config.max_hold_days);
    let start_str = config.start_date.map(|d| d.to_string()).unwrap_or_else(|| "auto".to_string());
    let end_str = config.end_date.map(|d| d.to_string()).unwrap_or_else(|| "auto".to_string());
    println!("Date Range: {} ~ {}", start_str, end_str);
    println!("Sort Field: {} ({})", config.sort_field, if config.sort_ascending { "ASC" } else { "DESC" });
    println!("Min Score: {} ({})", config.min_score, if config.min_score > 0.0 { "ON" } else { "OFF" });
    println!("Stop Loss: {:.1}% ({})", config.stop_loss_pct * 100.0, if config.stop_loss_enabled { "ON" } else { "OFF" });
    println!("Take Profit: TP1={:.0}%, TP2={:.0}%", config.tp1_pct * 100.0, config.tp2_pct * 100.0);
    println!("Weak Filter: {} days @ {:.0}% ({})", config.weak_days, config.weak_min_gain_pct * 100.0, if config.weak_enabled { "ON" } else { "OFF" });
    println!("Trailing Stop: Activate={:.0}%, Trail={:.0}% ({})",
        config.trailing_activation_pct * 100.0,
        config.trailing_pct * 100.0,
        if config.trailing_enabled { "ON" } else { "OFF" }
    );
    println!("Costs: Commission={:.4}%, Stamp={:.3}%, Slippage={:.2}%",
        config.commission_rate * 100.0,
        config.stamp_duty_rate * 100.0,
        config.slippage_pct * 100.0
    );
    println!("------------------------");
}

fn format_config(config: &BacktestConfig, trading_days: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();

    writeln!(s, "--- Configuration ---").unwrap();
    writeln!(s, "Initial Capital:  {:.0}", config.initial_capital).unwrap();
    writeln!(s, "Max Positions:    {} (Daily: {})", config.max_positions, config.max_daily_buys).unwrap();
    writeln!(s, "Position Size:    {:.0}%", config.position_size_pct * 100.0).unwrap();
    writeln!(s, "Max Hold Days:    {}", config.max_hold_days).unwrap();
    let start_str = config.start_date.map(|d| d.to_string()).unwrap_or_else(|| "auto".into());
    let end_str = config.end_date.map(|d| d.to_string()).unwrap_or_else(|| "auto".into());
    writeln!(s, "Date Range:       {} ~ {}", start_str, end_str).unwrap();
    writeln!(s, "Trading Days:     {}", trading_days).unwrap();
    writeln!(s, "Sort Field:       {} ({})", config.sort_field, if config.sort_ascending { "ASC" } else { "DESC" }).unwrap();
    writeln!(s, "Min Score:        {} ({})", config.min_score, if config.min_score > 0.0 { "ON" } else { "OFF" }).unwrap();
    writeln!(s, "Stop Loss:        {:.1}% ({})",
        config.stop_loss_pct * 100.0,
        if config.stop_loss_enabled { "ON" } else { "OFF" }
    ).unwrap();
    writeln!(s, "Take Profit:      TP1={:.0}%, TP2={:.0}%, Sell Ratio={:.1}%",
        config.tp1_pct * 100.0, config.tp2_pct * 100.0, config.tp_sell_ratio * 100.0
    ).unwrap();
    writeln!(s, "Break WL/YL:      WL={}, YL={}",
        if config.sell_on_break_wl { "ON" } else { "OFF" },
        if config.sell_on_break_yl { "ON" } else { "OFF" }
    ).unwrap();
    writeln!(s, "Weak Filter:      {} days @ {:.0}% ({})",
        config.weak_days, config.weak_min_gain_pct * 100.0,
        if config.weak_enabled { "ON" } else { "OFF" }
    ).unwrap();
    writeln!(s, "Trailing Stop:    Activate={:.0}%, Trail={:.0}% ({})",
        config.trailing_activation_pct * 100.0,
        config.trailing_pct * 100.0,
        if config.trailing_enabled { "ON" } else { "OFF" }
    ).unwrap();
    writeln!(s, "Commission:       {:.4}%", config.commission_rate * 100.0).unwrap();
    writeln!(s, "Stamp Duty:       {:.3}%", config.stamp_duty_rate * 100.0).unwrap();
    writeln!(s, "Slippage:         {:.2}%", config.slippage_pct * 100.0).unwrap();

    s
}

fn append_b1_registry_entry(
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

    let registry_path = bt_core::resolve_registry_path(
        signal_meta,
        "../artifacts/b1/backtest.jsonl",
    );
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
        "strategy": "b1",
        "train_run_id": signal_meta.and_then(|meta| meta.train_run_id.clone()),
        "signal_id": signal_meta.and_then(|meta| meta.signal_id.clone()),
        "signal_run_id": signal_meta.and_then(|meta| meta.signal_run_id.clone()),
        "label": signal_meta.and_then(|meta| meta.label.clone()),
        "model_name": signal_meta.and_then(|meta| meta.model_name.clone()),
        "feature_mode": signal_meta.and_then(|meta| meta.feature_mode.clone()),
        "feature_hash": signal_meta.and_then(|meta| meta.feature_hash.clone()),
        "feature_count": signal_meta.and_then(|meta| meta.feature_count),
        "export_ema_alpha": signal_meta.and_then(|meta| meta.export_ema_alpha),
        "backtest_id": backtest_id,
        "input_signal_file": bt_core::relative_portable_path(&train_run_dir, data_path),
        "signal_path": signal_path_resolved.as_ref().map(|p| bt_core::relative_portable_path(&train_run_dir, p)),
        "signal_meta_path": signal_meta_resolved.as_ref().map(|p| bt_core::relative_portable_path(&train_run_dir, p)),
        "backtest_dir": bt_core::relative_portable_path(&train_run_dir, &report_dir),
        "git_commit": signal_meta.and_then(|meta| meta.git_commit.clone()),
        "max_positions": config.max_positions,
        "max_daily_buys": config.max_daily_buys,
        "position_size_pct": config.position_size_pct,
        "max_hold_days": config.max_hold_days,
        "sort_field": config.sort_field,
        "sort_ascending": config.sort_ascending,
        "min_position_ratio": config.min_position_ratio,
        "min_score": config.min_score,
        "stop_loss_enabled": config.stop_loss_enabled,
        "stop_loss_pct": config.stop_loss_pct,
        "tp1_pct": config.tp1_pct,
        "tp2_pct": config.tp2_pct,
        "tp_sell_ratio": config.tp_sell_ratio,
        "sell_on_break_wl": config.sell_on_break_wl,
        "sell_on_break_yl": config.sell_on_break_yl,
        "weak_enabled": config.weak_enabled,
        "weak_days": config.weak_days,
        "weak_min_gain_pct": config.weak_min_gain_pct,
        "trailing_enabled": config.trailing_enabled,
        "trailing_activation_pct": config.trailing_activation_pct,
        "trailing_pct": config.trailing_pct,
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
    println!("🗂️ Registry appended: {}", registry_path.display());
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
    let wl = df.column("WL")?.f64()?;
    let yl = df.column("YL")?.f64()?;
    let b1_signal = df.column("b1_signal")?.bool()?;
    let pre_b1_signal = df.column("pre_b1_signal")?.bool()?;
    let is_loose = df.column("is_loose")?.bool()?;
    let sort_col = df.column(sort_field)
        .map_err(|_| format!("Sort field '{}' not found in parquet", sort_field))?
        .f64()?;

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
            wl: wl.get(i).unwrap_or(0.0),
            yl: yl.get(i).unwrap_or(0.0),
            b1_signal: b1_signal.get(i).unwrap_or(false),
            pre_b1_signal: pre_b1_signal.get(i).unwrap_or(false),
            is_loose: is_loose.get(i).unwrap_or(false),
            sort_value: sort_col.get(i).unwrap_or(999.0),
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
    use components::{ClosedTrade, ExitReason};

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

    let mut running_total_asset = world.resource::<Portfolio>().cash;
    for (_, position, exit_price) in &to_close {
        running_total_asset += position.shares as f64 * exit_price;
    }

    for (entity, position, exit_price) in to_close {
        let gross = position.shares as f64 * exit_price;
        let commission = gross * config.commission_rate;
        let stamp_duty = gross * config.stamp_duty_rate;
        let slippage = gross * config.slippage_pct;
        let net = gross - commission - stamp_duty - slippage;

        let exit_pnl = net - position.cost;
        let trade_pnl = exit_pnl + position.realized_pnl;
        let total_initial_cost = position.initial_shares as f64 * position.entry_price;
        let trade_pnl_pct = trade_pnl / total_initial_cost;
        let hold_days = (end_date - position.entry_date).num_days() as i32;

        world.resource_mut::<Portfolio>().cash += net;
        running_total_asset -= commission + stamp_duty + slippage;
        {
            let mut stats = world.resource_mut::<BacktestStats>();
            stats.record_trade(trade_pnl, commission, stamp_duty, slippage);
        }

        let stage_label = match position.take_profit_stage {
            0 => "None",
            1 => "TP1",
            2 => "TP2",
            _ => "Unknown",
        };
        println!(
            "[{}] [CLOSE] {} @ {:.2} | ExitPnL: {:+.2} | TradePnL: {:+.2} ({:+.2}%) | Stage: {} | Hold: {}d | EndOfBacktest | Asset: {:.2}",
            end_date,
            position.code,
            exit_price,
            exit_pnl,
            trade_pnl,
            trade_pnl_pct * 100.0,
            stage_label,
            hold_days,
            running_total_asset
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
            hold_days,
            exit_reason: ExitReason::EndOfBacktest,
        });
        world.entity_mut(entity).remove::<Position>();
    }
}
