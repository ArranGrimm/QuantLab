//! Renko 截面短线回测引擎
//!
//! 读取 Python LightGBM 模型生成的 renko_scores.parquet，
//! 以 T 日 score/rank 决定 T+1 日 open 买入，收盘按排名/止损退出。
//!
//! Usage:
//!   cargo run -p bt-renko --release -- --data ../../data/signals/renko_scores.parquet
//!   cargo run -p bt-renko --release -- --config config.toml --data ../../data/signals/renko_scores.parquet

mod components;
mod resources;
mod systems;

use bevy_app::{App, Update};
use bevy_ecs::schedule::IntoSystemConfigs;
use chrono::NaiveDate;
use clap::Parser;
use polars::prelude::*;
use std::path::PathBuf;

use bt_core::{BacktestStats, Portfolio};
use components::Position;
use resources::{ConfigFile, DailyData, MarketData, PriceBar, RenkoConfig};
use systems::{check_exit_conditions, process_buy_signals, update_stats};

#[derive(Parser, Debug)]
#[command(author, version, about = "Renko 截面短线回测")]
struct Args {
    #[arg(short, long, default_value = "../data/signals/renko_scores.parquet")]
    data: PathBuf,

    #[arg(short, long, default_value = "crates/renko/config.toml")]
    config: PathBuf,

    /// 结果输出目录 (留空则不保存)
    #[arg(short, long, default_value = "../results")]
    output_dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("========================================");
    println!("   Renko Cross-Section Backtest Engine");
    println!("========================================");

    let config: RenkoConfig = match ConfigFile::load(&args.config) {
        Ok(cfg) => {
            println!("Loaded config from: {:?}", args.config);
            cfg.into()
        }
        Err(e) => {
            println!("Warning: {}. Using defaults.", e);
            RenkoConfig::default()
        }
    };

    print_config(&config);

    println!("\nLoading data from: {:?}", args.data);
    let df = LazyFrame::scan_parquet(&args.data, Default::default())?.collect()?;
    println!("Loaded {} rows", df.height());

    let (market_data, mut trading_dates) = build_market_data(&df)?;

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
    let min_score = config.min_score;
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
        "\nRunning Renko cross-sectional backtest over {} trading days...\n",
        trading_dates.len()
    );

    let mut limit_up_blocked: u32 = 0;
    let mut limit_up_days: u32 = 0;

    for date in &trading_dates {
        let world = app.world_mut();
        world.resource_mut::<Portfolio>().current_date = Some(*date);

        let market_data = world.resource::<MarketData>();
        let mut candidates: Vec<_> = market_data
            .prices
            .iter()
            .filter_map(|(code, dates)| {
                dates.get(date).and_then(|bar| {
                    if bar.pre_score < min_score || bar.pre_rank > top_n as u32 {
                        return None;
                    }
                    Some((code.clone(), bar.pre_score, bar.open, bar.pre_close))
                })
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let before = candidates.len();
        candidates.retain(|(code, _pre_score, open, pre_close)| {
            let limit = bt_core::price_limit_pct(code);
            !bt_core::is_limit_up(*open, *pre_close, limit)
        });
        let blocked = (before - candidates.len()) as u32;
        if blocked > 0 {
            limit_up_blocked += blocked;
            limit_up_days += 1;
        }

        let candidates: Vec<(String, f64, f64)> = candidates
            .into_iter()
            .map(|(code, pre_score, open, _)| (code, pre_score, open))
            .collect();

        world.resource_mut::<DailyData>().buy_candidates = candidates;
        app.update();
    }

    if let Some(end_date) = trading_dates.last() {
        force_close_all_positions(&mut app, *end_date);
    }

    println!("\n--- 开盘涨停过滤统计 ---");
    println!(
        "Pre Top-{} 中被开盘涨停过滤: {} 次 ({} 天有过滤, 日均 {:.1})",
        top_n,
        limit_up_blocked,
        limit_up_days,
        if limit_up_days > 0 {
            limit_up_blocked as f64 / limit_up_days as f64
        } else {
            0.0
        }
    );

    let stats = app.world().resource::<BacktestStats>();
    let portfolio = app.world().resource::<Portfolio>();
    bt_core::print_results(stats, portfolio);

    if !args.output_dir.is_empty() {
        let config = app.world().resource::<RenkoConfig>();
        let config_text = format_config(config, trading_dates.len());
        bt_core::write_report(
            "renko",
            &config_text,
            stats,
            portfolio,
            trading_dates.len(),
            &args.output_dir,
        )?;
    }

    Ok(())
}

fn print_config(config: &RenkoConfig) {
    println!("\n--- Renko Configuration ---");
    println!("Initial Capital: {:.0}", config.initial_capital);
    println!("Max Positions: {}", config.max_positions);
    println!("Position Size: {:.1}%", config.position_size_pct * 100.0);
    println!("Max Hold Days: {}", config.max_hold_days);
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
        "Top-N: {} (Hold Buffer: {})",
        config.top_n, config.hold_buffer
    );
    println!("Min Score: {}", config.min_score);
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
    println!("---------------------------");
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
    let score = df.column("score")?.f64()?;
    let rank_casted = df.column("rank")?.cast(&DataType::UInt32)?;
    let rank = rank_casted.u32()?;
    let is_top_n = df.column("is_top_n")?.bool()?;
    let pre_score = df.column("pre_score")?.f64()?;
    let pre_rank_casted = df.column("pre_rank")?.cast(&DataType::UInt32)?;
    let pre_rank = pre_rank_casted.u32()?;
    let pre_is_top_n = df.column("pre_is_top_n")?.bool()?;

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
            score: score.get(i).unwrap_or(-999.0),
            rank: rank.get(i).unwrap_or(9999u32),
            is_top_n: is_top_n.get(i).unwrap_or(false),
            pre_score: pre_score.get(i).unwrap_or(-999.0),
            pre_rank: pre_rank.get(i).unwrap_or(9999u32),
            pre_is_top_n: pre_is_top_n.get(i).unwrap_or(false),
        };

        market_data
            .prices
            .entry(code.to_string())
            .or_default()
            .insert(date, bar);

        all_dates.push(date);
    }

    all_dates.sort();
    all_dates.dedup();
    Ok((market_data, all_dates))
}

fn format_config(config: &RenkoConfig, trading_days: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();

    writeln!(s, "--- Configuration ---").unwrap();
    writeln!(s, "Initial Capital:  {:.0}", config.initial_capital).unwrap();
    writeln!(s, "Max Positions:    {}", config.max_positions).unwrap();
    writeln!(
        s,
        "Position Size:    {:.1}%",
        config.position_size_pct * 100.0
    )
    .unwrap();
    writeln!(s, "Max Hold Days:    {}", config.max_hold_days).unwrap();
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
    writeln!(s, "Hold Buffer:      {}", config.hold_buffer).unwrap();
    writeln!(s, "Min Score:        {}", config.min_score).unwrap();
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

fn force_close_all_positions(app: &mut App, end_date: NaiveDate) {
    use components::{ClosedTrade, ExitReason};

    let world = app.world_mut();
    let market_data = world.resource::<MarketData>().clone();
    let config = world.resource::<RenkoConfig>().clone();

    let mut to_close: Vec<(bevy_ecs::entity::Entity, Position, f64)> = Vec::new();
    {
        let mut query = world.query::<(bevy_ecs::entity::Entity, &Position)>();
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
        let commission = gross * config.cost_model.commission_rate;
        let stamp_duty = gross * config.cost_model.stamp_duty_rate;
        let slippage = gross * config.cost_model.slippage_pct;
        let net = gross - commission - stamp_duty - slippage;

        let pnl = net - position.cost;
        let pnl_pct = pnl / position.cost;
        let hold_days = (end_date - position.entry_date).num_days() as i32;

        world.resource_mut::<Portfolio>().cash += net;
        {
            let mut stats = world.resource_mut::<BacktestStats>();
            stats.record_trade(pnl, commission, stamp_duty, slippage);
        }

        println!(
            "[{}] [CLOSE] {} @ {:.2} | PnL: {:+.1}% | Hold: {}d",
            end_date,
            position.code,
            exit_price,
            pnl_pct * 100.0,
            hold_days
        );

        world.entity_mut(entity).insert(ClosedTrade {
            code: position.code.clone(),
            entry_date: position.entry_date,
            exit_date: end_date,
            entry_price: position.entry_price,
            exit_price,
            shares: position.shares,
            pnl,
            pnl_pct,
            hold_days,
            exit_reason: ExitReason::EndOfBacktest,
        });
        world.entity_mut(entity).remove::<Position>();
    }
}
