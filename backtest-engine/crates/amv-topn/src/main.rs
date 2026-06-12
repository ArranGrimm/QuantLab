//! AMV TopN strategy backtest engine.
//!
//! The input parquet is expected to contain T+1 shifted signals:
//! `is_signal=true` on the execution day, with `score` and `rank` copied from the
//! prior close signal.

#![recursion_limit = "256"]

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
use components::{ClosedTrade, Position};
use resources::{AmvTopnConfig, ConfigFile, DailyData, MarketData, PriceBar};
use systems::{check_exit_conditions, process_buy_signals, update_stats};

#[derive(Parser, Debug)]
#[command(author, version, about = "AMV TopN 策略回测")]
struct Args {
    #[arg(short, long)]
    data: PathBuf,

    #[arg(short, long, default_value = "crates/amv-topn/config.toml")]
    config: PathBuf,

    #[arg(short, long, default_value = "")]
    output_dir: String,
}

#[derive(Debug, Clone)]
struct DailyEquity {
    date: NaiveDate,
    cash: f64,
    positions_value: f64,
    total_value: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let signal_meta = bt_core::load_signal_meta(&args.data);

    println!("========================================");
    println!("   AMV TopN Backtest Engine (Bevy ECS)");
    println!("========================================");

    let config: AmvTopnConfig = match ConfigFile::load(&args.config) {
        Ok(cfg) => {
            println!("Loaded config from: {:?}", args.config);
            cfg.into()
        }
        Err(e) => {
            println!("Warning: {}. Using defaults.", e);
            AmvTopnConfig::default()
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
    let (market_data, mut trading_dates) = build_market_data(&df)?;
    println!("Execution price basis: {}", market_data.price_basis);

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
    let top_n = config.top_n;
    let entry_rank_limit = config.entry_rank_limit as u32;
    let min_score = config.min_score;
    let require_bull_regime = config.require_bull_regime;
    let max_open_gap_pct = config.max_open_gap_pct;
    app.insert_resource(config);
    app.insert_resource(Portfolio::new(initial_capital));
    app.insert_resource(BacktestStats::default());
    app.insert_resource(market_data);
    app.insert_resource(DailyData::default());
    app.add_systems(
        Update,
        (process_buy_signals, check_exit_conditions, update_stats).chain(),
    );

    println!(
        "\nRunning AMV TopN backtest over {} trading days...\n",
        trading_dates.len()
    );

    let mut daily_equity: Vec<DailyEquity> = Vec::with_capacity(trading_dates.len());
    let mut signal_rows: u32 = 0;
    let mut limit_up_blocked: u32 = 0;
    let mut limit_up_days: u32 = 0;
    let mut bull_regime_blocked_signals: u32 = 0;
    let mut bull_regime_blocked_days: u32 = 0;
    let mut open_gap_blocked: u32 = 0;
    let mut open_gap_blocked_days: u32 = 0;

    for date in &trading_dates {
        let world = app.world_mut();
        world.resource_mut::<Portfolio>().current_date = Some(*date);

        let market_data = world.resource::<MarketData>();
        let mut raw_candidates: Vec<_> = market_data
            .prices
            .iter()
            .filter_map(|(code, dates)| {
                dates.get(date).and_then(|bar| {
                    if !bar.is_signal || bar.score < min_score || bar.rank > entry_rank_limit {
                        return None;
                    }
                    Some((
                        code.clone(),
                        bar.score,
                        bar.open,
                        bar.pre_close,
                        bar.rank,
                        bar.is_bull_regime,
                    ))
                })
            })
            .collect();
        raw_candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.4.cmp(&b.4))
        });
        raw_candidates.truncate(top_n);

        signal_rows += raw_candidates.len() as u32;
        let before_regime = raw_candidates.len();
        if require_bull_regime {
            raw_candidates.retain(|(_, _, _, _, _, is_bull_regime)| *is_bull_regime);
        }
        let regime_blocked = (before_regime - raw_candidates.len()) as u32;
        if regime_blocked > 0 {
            bull_regime_blocked_signals += regime_blocked;
            bull_regime_blocked_days += 1;
        }

        let before_limit = raw_candidates.len();
        raw_candidates.retain(|(code, _score, open, pre_close, _rank, _)| {
            let limit = bt_core::price_limit_pct(code);
            !bt_core::is_limit_up(*open, *pre_close, limit)
        });
        let limit_blocked = (before_limit - raw_candidates.len()) as u32;
        if limit_blocked > 0 {
            limit_up_blocked += limit_blocked;
            limit_up_days += 1;
        }

        let before_open_gap = raw_candidates.len();
        if let Some(max_open_gap_pct) = max_open_gap_pct {
            raw_candidates.retain(|(_, _, open, pre_close, _, _)| {
                *pre_close > 0.0 && (*open / *pre_close - 1.0) <= max_open_gap_pct
            });
        }
        let open_gap_blocked_count = (before_open_gap - raw_candidates.len()) as u32;
        if open_gap_blocked_count > 0 {
            open_gap_blocked += open_gap_blocked_count;
            open_gap_blocked_days += 1;
        }

        // --- Hook-level rejection (is_rejected flag) ---
        raw_candidates.retain(|(code, _score, _open, _pre_close, _rank, _)| {
            if let Some(prices) = market_data.prices.get(code) {
                if let Some(bar) = prices.get(date) {
                    return !bar.is_rejected;
                }
            }
            true
        });

        let candidates: Vec<(String, f64, f64)> = raw_candidates
            .into_iter()
            .map(|(code, score, open, _, _, _)| (code, score, open))
            .collect();
        world.resource_mut::<DailyData>().buy_candidates = candidates;
        app.update();
        daily_equity.push(snapshot_daily_equity(app.world_mut(), *date));
    }

    if let Some(end_date) = trading_dates.last() {
        force_close_all_positions(&mut app, *end_date);
        if let Some(last_equity) = daily_equity.last_mut() {
            if last_equity.date == *end_date {
                *last_equity = snapshot_daily_equity(app.world_mut(), *end_date);
            }
        }
    }

    println!("\n--- AMV TopN 过滤统计 ---");
    println!(
        "执行日信号行: {}, 涨停过滤: {} 次 ({} 天)",
        signal_rows, limit_up_blocked, limit_up_days
    );
    if let Some(max_open_gap_pct) = max_open_gap_pct {
        println!(
            "高开过滤: {} 次 ({} 天, 阈值 +{:.1}%)",
            open_gap_blocked,
            open_gap_blocked_days,
            max_open_gap_pct * 100.0
        );
    }
    if require_bull_regime {
        println!(
            "AMV bear/非 bull 阻止开仓: {} 次 ({} 天)",
            bull_regime_blocked_signals, bull_regime_blocked_days
        );
    }

    let stats = app.world().resource::<BacktestStats>();
    let portfolio = app.world().resource::<Portfolio>();
    bt_core::print_results(stats, portfolio);

    if !args.output_dir.is_empty() {
        let config = app.world().resource::<AmvTopnConfig>();
        let market_data = app.world().resource::<MarketData>();
        let config_text = format_config(config, trading_dates.len());
        let extra_text = format!(
            "--- Strategy Stats ---\n\
             Execution Price Basis: {}\n\
             Signal Rows:       {}\n\
             Limit Up Blocked:  {} ({} days)\n\
             Open Gap Blocked:  {} ({} days)\n\
             Require Bull Regime: {}\n\
             Bull Regime Blocked Signals: {} ({} days)\n",
            market_data.price_basis,
            signal_rows,
            limit_up_blocked,
            limit_up_days,
            open_gap_blocked,
            open_gap_blocked_days,
            if require_bull_regime { "true" } else { "false" },
            bull_regime_blocked_signals,
            bull_regime_blocked_days,
        );
        let report_paths = bt_core::write_report_bundle(
            &args.output_dir,
            "amv_topn",
            &args.data,
            signal_meta.as_ref(),
            &config_text,
            json!({
                "initial_capital": config.initial_capital,
                "max_positions": config.max_positions,
                "max_daily_buys": config.max_daily_buys,
                "position_size_pct": config.position_size_pct,
                "max_hold_trading_days": config.max_hold_trading_days,
                "allow_duplicate_positions": config.allow_duplicate_positions,
                "start_date": config.start_date.map(|d| d.to_string()),
                "end_date": config.end_date.map(|d| d.to_string()),
                "top_n": config.top_n,
                "entry_rank_limit": config.entry_rank_limit,
                "min_score": config.min_score,
                "require_bull_regime": config.require_bull_regime,
                "max_open_gap_pct": config.max_open_gap_pct,
                "sell_on_bear_regime": config.sell_on_bear_regime,
                "stop_loss_enabled": config.stop_loss_enabled,
                "stop_loss_pct": config.stop_loss_pct,
                "early_stop_enabled": config.early_stop_enabled,
                "early_stop_trigger_hold_trading_days": config.early_stop_trigger_hold_trading_days,
                "early_stop_loss_pct": config.early_stop_loss_pct,
                                                "trailing_enabled": config.trailing_enabled,
                "trailing_activation_pct": config.trailing_activation_pct,
                "trailing_pct": config.trailing_pct,
                "commission_rate": config.cost_model.commission_rate,
                "stamp_duty_rate": config.cost_model.stamp_duty_rate,
                "slippage_pct": config.cost_model.slippage_pct,
                "execution_price_basis": market_data.price_basis.as_str(),
            }),
            Some(extra_text.as_str()),
            Some(json!({
                "execution_price_basis": market_data.price_basis.as_str(),
                "signal_rows": signal_rows,
                "limit_up_blocked": limit_up_blocked,
                "limit_up_days": limit_up_days,
                "open_gap_blocked": open_gap_blocked,
                "open_gap_blocked_days": open_gap_blocked_days,
                "require_bull_regime": require_bull_regime,
                "bull_regime_blocked_signals": bull_regime_blocked_signals,
                "bull_regime_blocked_days": bull_regime_blocked_days,
            })),
            stats,
            portfolio,
            trading_dates.len(),
        )?;
        let report_dir = report_paths.json_path.parent().map(|p| p.to_path_buf());
        append_amv_topn_registry_entry(
            signal_meta.as_ref(),
            &args.data,
            config,
            stats,
            portfolio,
            trading_dates.len(),
            &report_paths.txt_path,
            &report_paths.json_path,
        )?;
        if let Some(report_dir) = report_dir {
            let trades_path = report_dir.join("trades.csv");
            let equity_path = report_dir.join("daily_equity.csv");
            write_closed_trades_csv(app.world_mut(), &trades_path)?;
            write_daily_equity_csv(&daily_equity, &equity_path)?;
            println!("📈 Daily equity CSV saved: {}", equity_path.display());
            println!("📋 Trades CSV saved: {}", trades_path.display());
        }
    }

    Ok(())
}

fn print_config(config: &AmvTopnConfig) {
    println!("\n--- AMV TopN Configuration ---");
    println!("Initial Capital: {:.0}", config.initial_capital);
    println!("Max Positions: {}", config.max_positions);
    println!("Max Daily Buys: {}", config.max_daily_buys);
    println!("Position Size: {:.1}%", config.position_size_pct * 100.0);
    println!("Max Hold Trading Days: {}", config.max_hold_trading_days);
    println!(
        "Allow Duplicate Positions: {}",
        if config.allow_duplicate_positions {
            "ON (lot-level)"
        } else {
            "OFF"
        }
    );
    let start_str = config
        .start_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "auto".to_string());
    let end_str = config
        .end_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "auto".to_string());
    println!("Date Range: {} ~ {}", start_str, end_str);
    println!(
        "Top-N: {} (Entry Rank Limit: {})",
        config.top_n, config.entry_rank_limit
    );
    println!("Min Score: {}", config.min_score);
    println!(
        "Require Bull Regime: {}",
        if config.require_bull_regime {
            "ON (entry-only)"
        } else {
            "OFF"
        }
    );
    match config.max_open_gap_pct {
        Some(pct) => println!("Max Open Gap: +{:.1}%", pct * 100.0),
        None => println!("Max Open Gap: OFF"),
    }
    println!(
        "Sell On Bear Regime: {}",
        if config.sell_on_bear_regime {
            "ON"
        } else {
            "OFF"
        }
    );
    println!(
        "Stop Loss: {:.1}% ({})",
        config.stop_loss_pct * 100.0,
        if config.stop_loss_enabled {
            "ON"
        } else {
            "OFF"
        }
    );
    println!(
        "Early Stop: d{}+ ATR×{:.1} ({})",
        config.early_stop_trigger_hold_trading_days,
        config.early_stop_atr_multiple,
        if config.early_stop_enabled { "ON" } else { "OFF" }
    );
    println!(
        "Trailing Stop: Activate={:.0}%, Trail={:.0}% ({})",
        config.trailing_activation_pct * 100.0,
        config.trailing_pct * 100.0,
        if config.trailing_enabled { "ON" } else { "OFF" }
    );
    let cm = &config.cost_model;
    println!(
        "Costs: Commission={:.4}%, Stamp={:.3}%, Slippage={:.2}%",
        cm.commission_rate * 100.0,
        cm.stamp_duty_rate * 100.0,
        cm.slippage_pct * 100.0
    );
    println!("------------------------------");
}

fn build_market_data(
    df: &DataFrame,
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
    let has_raw_prices = ["open_raw", "high_raw", "low_raw", "close_raw", "pre_close_raw"]
        .iter()
        .all(|name| df.get_column_names().iter().any(|col| col.as_str() == *name));
    let open_raw = if has_raw_prices {
        Some(df.column("open_raw")?.f64()?)
    } else {
        None
    };
    let high_raw = if has_raw_prices {
        Some(df.column("high_raw")?.f64()?)
    } else {
        None
    };
    let low_raw = if has_raw_prices {
        Some(df.column("low_raw")?.f64()?)
    } else {
        None
    };
    let close_raw = if has_raw_prices {
        Some(df.column("close_raw")?.f64()?)
    } else {
        None
    };
    let pre_close_raw = if has_raw_prices {
        Some(df.column("pre_close_raw")?.f64()?)
    } else {
        None
    };
    market_data.price_basis = if has_raw_prices {
        "raw_ohlc_pre_close".to_string()
    } else {
        "adjusted_ohlc_fallback".to_string()
    };
    let score = df.column("score")?.f64()?;
    let rank_casted = df.column("rank")?.cast(&DataType::UInt32)?;
    let rank = rank_casted.u32()?;
    let is_signal = df.column("is_signal")?.bool()?;
    let is_bull_regime = df.column("is_bull_regime")?.bool()?;
    let is_rejected = if df.get_column_names().iter().any(|c| c.as_str() == "is_rejected") {
        Some(df.column("is_rejected")?.bool()?)
    } else {
        None
    };

    for i in 0..df.height() {
        let code = codes.get(i).ok_or("Missing code")?;
        let date_days = dates.get(i).ok_or("Missing date")?;
        let date = bt_core::epoch_days_to_date(date_days).ok_or("Invalid date")?;
        let open_adj_value = open_adj.get(i).unwrap_or(0.0);
        let high_adj_value = high_adj.get(i).unwrap_or(0.0);
        let close_adj_value = close_adj.get(i).unwrap_or(0.0);
        let pre_close_adj_value = pre_close_adj.get(i).unwrap_or(0.0);
        let bar = PriceBar {
            open: open_raw
                .as_ref()
                .and_then(|col| col.get(i))
                .unwrap_or(open_adj_value),
            high: high_raw
                .as_ref()
                .and_then(|col| col.get(i))
                .unwrap_or(high_adj_value),
            low: low_raw
                .as_ref()
                .and_then(|col| col.get(i))
                .unwrap_or(low_adj.get(i).unwrap_or(0.0)),
            close: close_raw
                .as_ref()
                .and_then(|col| col.get(i))
                .unwrap_or(close_adj_value),
            pre_close: pre_close_raw
                .as_ref()
                .and_then(|col| col.get(i))
                .unwrap_or(pre_close_adj_value),
            open_adj: open_adj_value,
            high_adj: high_adj_value,
            low_adj: low_adj.get(i).unwrap_or(0.0),
            close_adj: close_adj_value,
            pre_close_adj: pre_close_adj_value,
            score: score.get(i).unwrap_or(0.0),
            rank: rank.get(i).unwrap_or(9999u32),
            is_signal: is_signal.get(i).unwrap_or(false),
            is_bull_regime: is_bull_regime.get(i).unwrap_or(false),
            is_rejected: is_rejected
                .as_ref()
                .and_then(|col| col.get(i))
                .unwrap_or(false),
            atr_14: 0.0,
        };

        market_data
            .prices
            .entry(code.to_string())
            .or_default()
            .insert(date, bar);
        all_dates.push(date);
    }

    // Compute ATR_14 for each stock after all bars are loaded.
    let mut atr_values: HashMap<String, HashMap<NaiveDate, f64>> = HashMap::new();
    for (code_str, price_map) in market_data.prices.iter() {
        let mut sorted_dates: Vec<NaiveDate> = price_map.keys().cloned().collect();
        sorted_dates.sort();
        let mut prev_close = 0.0;
        let mut tr_buffer: Vec<f64> = Vec::new();
        let mut code_atr: HashMap<NaiveDate, f64> = HashMap::new();
        for date in &sorted_dates {
            if let Some(bar) = price_map.get(date) {
                let tr = if prev_close > 0.0 {
                    let h_l = bar.high - bar.low;
                    let h_pc = (bar.high - prev_close).abs();
                    let l_pc = (bar.low - prev_close).abs();
                    h_l.max(h_pc).max(l_pc)
                } else {
                    bar.high - bar.low
                };
                tr_buffer.push(tr);
                if tr_buffer.len() > 14 {
                    tr_buffer.remove(0);
                }
                let atr: f64 = tr_buffer.iter().sum::<f64>() / tr_buffer.len() as f64;
                code_atr.insert(*date, atr);
                prev_close = bar.close;
            }
        }
        atr_values.insert(code_str.clone(), code_atr);
    }
    for (code_str, code_atr) in atr_values {
        if let Some(price_map) = market_data.prices.get_mut(&code_str) {
            for (date, atr) in code_atr {
                if let Some(bar) = price_map.get_mut(&date) {
                    bar.atr_14 = atr;
                }
            }
        }
    }

    all_dates.sort();
    all_dates.dedup();
    market_data.date_index = all_dates
        .iter()
        .enumerate()
        .map(|(idx, date)| (*date, idx as i32))
        .collect();
    market_data.trading_dates = all_dates.clone();
    Ok((market_data, all_dates))
}

fn format_config(config: &AmvTopnConfig, trading_days: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();

    writeln!(s, "--- Configuration ---").unwrap();
    writeln!(s, "Initial Capital:  {:.0}", config.initial_capital).unwrap();
    writeln!(s, "Max Positions:    {}", config.max_positions).unwrap();
    writeln!(s, "Max Daily Buys:   {}", config.max_daily_buys).unwrap();
    writeln!(
        s,
        "Position Size:    {:.1}%",
        config.position_size_pct * 100.0
    )
    .unwrap();
    writeln!(s, "Max Hold Trading Days: {}", config.max_hold_trading_days).unwrap();
    writeln!(
        s,
        "Allow Duplicate Positions: {}",
        if config.allow_duplicate_positions {
            "true (lot-level)"
        } else {
            "false"
        }
    )
    .unwrap();
    let start_str = config
        .start_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "auto".into());
    let end_str = config
        .end_date
        .map(|d| d.to_string())
        .unwrap_or_else(|| "auto".into());
    writeln!(s, "Date Range:       {} ~ {}", start_str, end_str).unwrap();
    writeln!(s, "Trading Days:     {}", trading_days).unwrap();
    writeln!(s, "Top-N:            {}", config.top_n).unwrap();
    writeln!(s, "Entry Rank Limit: {}", config.entry_rank_limit).unwrap();
    writeln!(s, "Min Score:        {}", config.min_score).unwrap();
    writeln!(
        s,
        "Require Bull Regime: {}",
        if config.require_bull_regime {
            "true (entry-only gate)"
        } else {
            "false"
        }
    )
    .unwrap();
    match config.max_open_gap_pct {
        Some(pct) => writeln!(s, "Max Open Gap:     +{:.1}%", pct * 100.0).unwrap(),
        None => writeln!(s, "Max Open Gap:     OFF").unwrap(),
    }
    writeln!(
        s,
        "Sell On Bear Regime: {}",
        if config.sell_on_bear_regime {
            "true"
        } else {
            "false"
        }
    )
    .unwrap();
    writeln!(
        s,
        "Stop Loss:        {:.1}% ({})",
        config.stop_loss_pct * 100.0,
        if config.stop_loss_enabled {
            "ON"
        } else {
            "OFF"
        }
    )
    .unwrap();
    writeln!(
        s,
        "Early Stop:       d{}+ ATR×{:.1} ({})",
        config.early_stop_trigger_hold_trading_days,
        config.early_stop_atr_multiple,
        if config.early_stop_enabled { "ON" } else { "OFF" }
    )
    .unwrap();
    writeln!(
        s,
        "Trailing Stop:    Activate={:.0}%, Trail={:.0}% ({})",
        config.trailing_activation_pct * 100.0,
        config.trailing_pct * 100.0,
        if config.trailing_enabled { "ON" } else { "OFF" }
    )
    .unwrap();
    let cm = &config.cost_model;
    writeln!(s, "Commission:       {:.4}%", cm.commission_rate * 100.0).unwrap();
    writeln!(s, "Stamp Duty:       {:.3}%", cm.stamp_duty_rate * 100.0).unwrap();
    writeln!(s, "Slippage:         {:.2}%", cm.slippage_pct * 100.0).unwrap();
    s
}

fn append_amv_topn_registry_entry(
    signal_meta: Option<&SignalArtifactMeta>,
    data_path: &Path,
    config: &AmvTopnConfig,
    stats: &BacktestStats,
    portfolio: &Portfolio,
    trading_days: usize,
    report_txt_path: &Path,
    report_json_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use chrono::Local;
    let registry_path =
        bt_core::resolve_registry_path(signal_meta, "../artifacts/amv_topn/backtest.jsonl");
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
        "strategy": "amv_topn",
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
        "max_hold_trading_days": config.max_hold_trading_days,
        "allow_duplicate_positions": config.allow_duplicate_positions,
        "top_n": config.top_n,
        "entry_rank_limit": config.entry_rank_limit,
        "min_score": config.min_score,
        "require_bull_regime": config.require_bull_regime,
        "max_open_gap_pct": config.max_open_gap_pct,
        "sell_on_bear_regime": config.sell_on_bear_regime,
        "stop_loss_enabled": config.stop_loss_enabled,
        "stop_loss_pct": config.stop_loss_pct,
        "early_stop_enabled": config.early_stop_enabled,
        "early_stop_trigger_hold_trading_days": config.early_stop_trigger_hold_trading_days,
        "early_stop_loss_pct": config.early_stop_loss_pct,
                        "trailing_enabled": config.trailing_enabled,
        "trailing_activation_pct": config.trailing_activation_pct,
        "trailing_pct": config.trailing_pct,
        "commission_rate": config.cost_model.commission_rate,
        "stamp_duty_rate": config.cost_model.stamp_duty_rate,
        "slippage_pct": config.cost_model.slippage_pct,
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

fn force_close_all_positions(app: &mut App, end_date: NaiveDate) {
    use components::{ClosedTrade, ExitReason};

    let world = app.world_mut();
    let market_data = world.resource::<MarketData>().clone();
    let config = world.resource::<AmvTopnConfig>().clone();

    let mut to_close: Vec<(Entity, Position, f64)> = Vec::new();
    {
        let mut query = world.query::<(Entity, &Position)>();
        for (entity, position) in query.iter(world) {
            if position.shares == 0 {
                continue;
            }
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
        let cm = &config.cost_model;
        let commission = gross * cm.commission_rate;
        let stamp_duty = gross * cm.stamp_duty_rate;
        let slippage = gross * cm.slippage_pct;
        let net = gross - commission - stamp_duty - slippage;
        let pnl = net - position.cost;
        let pnl_pct = pnl / position.cost;
        let exit_trade_index = market_data
            .date_index
            .get(&end_date)
            .copied()
            .unwrap_or(position.entry_trade_index);
        let hold_trading_days = exit_trade_index - position.entry_trade_index;

        world.resource_mut::<Portfolio>().cash += net;
        {
            let mut stats = world.resource_mut::<BacktestStats>();
            stats.record_trade(pnl, commission, stamp_duty, slippage);
        }

        println!(
            "[{}] [CLOSE] {} @ {:.2} | PnL: {:+.1}% | Hold: {}td",
            end_date,
            position.code,
            exit_price,
            pnl_pct * 100.0,
            hold_trading_days
        );

        world.entity_mut(entity).insert(ClosedTrade {
            code: position.code.clone(),
            entry_date: position.entry_date,
            exit_date: end_date,
            entry_price: position.entry_price,
            exit_price,
            shares: position.shares,
            cost: position.cost,
            exit_value: net,
            pnl,
            pnl_pct,
            hold_trading_days,
            exit_reason: ExitReason::EndOfBacktest,
        });
        world.entity_mut(entity).remove::<Position>();
    }
}

fn snapshot_daily_equity(world: &mut World, date: NaiveDate) -> DailyEquity {
    let cash = world.resource::<Portfolio>().cash;
    let mut query = world.query::<&Position>();
    let positions: Vec<(String, u32)> = query
        .iter(world)
        .map(|position| (position.code.clone(), position.shares))
        .collect();
    let market_data = world.resource::<MarketData>();
    let positions_value = positions
        .iter()
        .filter_map(|(code, shares)| {
            market_data
                .prices
                .get(code)
                .and_then(|prices| prices.get(&date))
                .map(|bar| *shares as f64 * bar.close)
        })
        .sum::<f64>();
    DailyEquity {
        date,
        cash,
        positions_value,
        total_value: cash + positions_value,
    }
}

fn write_closed_trades_csv(
    world: &mut World,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut query = world.query::<&ClosedTrade>();
    let mut trades: Vec<ClosedTrade> = query.iter(world).cloned().collect();
    trades.sort_by(|a, b| {
        a.exit_date
            .cmp(&b.exit_date)
            .then_with(|| a.entry_date.cmp(&b.entry_date))
            .then_with(|| a.code.cmp(&b.code))
    });

    let mut file = std::fs::File::create(path)?;
    writeln!(
        file,
        "code,entry_date,exit_date,entry_price,exit_price,shares,cost,exit_value,pnl,pnl_pct,hold_trading_days,exit_reason"
    )?;
    for trade in trades {
        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.8},{},{}",
            csv_escape(&trade.code),
            trade.entry_date,
            trade.exit_date,
            trade.entry_price,
            trade.exit_price,
            trade.shares,
            trade.cost,
            trade.exit_value,
            trade.pnl,
            trade.pnl_pct,
            trade.hold_trading_days,
            trade.exit_reason
        )?;
    }
    Ok(())
}

fn write_daily_equity_csv(
    rows: &[DailyEquity],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "date,cash,positions_value,total_value")?;
    for row in rows {
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6}",
            row.date, row.cash, row.positions_value, row.total_value
        )?;
    }
    Ok(())
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
