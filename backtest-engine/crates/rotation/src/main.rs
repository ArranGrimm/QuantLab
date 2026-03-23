//! 截面轮动策略回测引擎
//!
//! 读取 Python LightGBM 模型生成的 rotation_scores.parquet,
//! 每日按 score 排名选 Top-N 买入, 排名退出/止损/移动止损卖出.
//!
//! Usage:
//!   cargo run -p bt-rotation --release -- --data ../../data/signals/rotation_scores.parquet
//!   cargo run -p bt-rotation --release -- --config config.toml --data ../../data/signals/rotation_scores.parquet

mod components;
mod resources;
mod systems;

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use clap::Parser;
use polars::prelude::*;
use std::path::PathBuf;

use bt_core::{BacktestStats, Portfolio};
use components::Position;
use resources::{ConfigFile, DailyData, MarketData, PriceBar, RotationConfig};
use systems::{check_exit_conditions, fill_positions, update_stats};

#[derive(Parser, Debug)]
#[command(author, version, about = "截面轮动策略回测")]
struct Args {
    #[arg(short, long, default_value = "../data/signals/rotation_scores.parquet")]
    data: PathBuf,

    #[arg(short, long, default_value = "crates/rotation/config.toml")]
    config: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("========================================");
    println!("   Rotation Backtest Engine (Bevy ECS)");
    println!("========================================");

    // 1. Load config
    let config: RotationConfig = match ConfigFile::load(&args.config) {
        Ok(cfg) => {
            println!("Loaded config from: {:?}", args.config);
            cfg.into()
        }
        Err(e) => {
            println!("Warning: {}. Using defaults.", e);
            RotationConfig::default()
        }
    };

    print_config(&config);

    println!("\nLoading data from: {:?}", args.data);

    // 2. Load market data
    let df = LazyFrame::scan_parquet(&args.data, Default::default())?.collect()?;
    println!("Loaded {} rows", df.height());

    // 3. Build market data
    let (market_data, mut trading_dates) = build_market_data(&df)?;

    // 4. Date range filter
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
    let top_n = config.top_n;
    app.insert_resource(config);
    app.insert_resource(Portfolio::new(initial_capital));
    app.insert_resource(BacktestStats::default());
    app.insert_resource(market_data);
    app.insert_resource(DailyData::default());

    // 先卖出(排名退出/止损)，再买入(填补空位)
    app.add_systems(
        Update,
        (check_exit_conditions, fill_positions, update_stats).chain(),
    );

    // 6. Run backtest
    println!(
        "\nRunning rotation backtest over {} trading days...\n",
        trading_dates.len()
    );

    for date in &trading_dates {
        let world = app.world_mut();
        world.resource_mut::<Portfolio>().current_date = Some(*date);

        // 构建当日候选: score 前 top_n 的股票
        let market_data = world.resource::<MarketData>();
        let mut candidates: Vec<_> = market_data
            .prices
            .iter()
            .filter_map(|(code, dates)| {
                dates.get(date).map(|bar| (code.clone(), bar.score, bar.close))
            })
            .collect();

        // 按 score 降序
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_n);

        world.resource_mut::<DailyData>().candidates = candidates;
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

    Ok(())
}

fn print_config(config: &RotationConfig) {
    println!("\n--- Rotation Configuration ---");
    println!("Initial Capital: {:.0}", config.initial_capital);
    println!("Max Positions: {}", config.max_positions);
    println!("Position Size: {:.1}%", config.position_size_pct * 100.0);
    println!("Max Hold Days: {}", config.max_hold_days);
    let start_str = config.start_date.map(|d| d.to_string()).unwrap_or_else(|| "auto".to_string());
    let end_str = config.end_date.map(|d| d.to_string()).unwrap_or_else(|| "auto".to_string());
    println!("Date Range: {} ~ {}", start_str, end_str);
    println!("Top-N: {} (Hold Buffer: {})", config.top_n, config.hold_buffer);
    println!(
        "Stop Loss: {:.1}% ({})",
        config.stop_loss_pct * 100.0,
        if config.stop_loss_enabled { "ON" } else { "OFF" }
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
    let score = df.column("score")?.f64()?;
    let rank_casted = df.column("rank")?.cast(&DataType::UInt32)?;
    let rank = rank_casted.u32()?;
    let is_top_n = df.column("is_top_n")?.bool()?;

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
            score: score.get(i).unwrap_or(0.0),
            rank: rank.get(i).unwrap_or(9999u32),
            is_top_n: is_top_n.get(i).unwrap_or(false),
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

fn force_close_all_positions(app: &mut App, end_date: NaiveDate) {
    use components::{ClosedTrade, ExitReason};

    let world = app.world_mut();
    let market_data = world.resource::<MarketData>().clone();
    let config = world.resource::<RotationConfig>().clone();

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
        let cm = &config.cost_model;
        let commission = gross * cm.commission_rate;
        let stamp_duty = gross * cm.stamp_duty_rate;
        let slippage = gross * cm.slippage_pct;
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
            end_date, position.code, exit_price, pnl_pct * 100.0, hold_days
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
