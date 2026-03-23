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
use std::collections::HashMap;
use std::path::PathBuf;

use bt_core::{BacktestStats, Portfolio};
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

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

        let mut candidates: Vec<_> = market_data
            .prices
            .iter()
            .filter_map(|(code, dates)| {
                dates.get(date).and_then(|bar| {
                    if bar.pre_b1_signal && bar.is_loose {
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

    for (entity, position, exit_price) in to_close {
        let gross = position.shares as f64 * exit_price;
        let commission = gross * config.commission_rate;
        let stamp_duty = gross * config.stamp_duty_rate;
        let slippage = gross * config.slippage_pct;
        let net = gross - commission - stamp_duty - slippage;

        let pnl = (net - position.cost) + position.realized_pnl;
        let total_initial_cost = position.initial_shares as f64 * position.entry_price;
        let pnl_pct = pnl / total_initial_cost;
        let hold_days = (end_date - position.entry_date).num_days() as i32;

        world.resource_mut::<Portfolio>().cash += net;
        {
            let mut stats = world.resource_mut::<BacktestStats>();
            stats.record_trade(pnl, commission, stamp_duty, slippage);
        }

        let stage_info = match position.take_profit_stage {
            0 => "",
            1 => " (TP1)",
            2 => " (TP2)",
            _ => "",
        };
        println!(
            "[{}] [CLOSE] {} @ {:.2} | PnL: {:+.2}%{} | Hold: {} days",
            end_date, position.code, exit_price, pnl_pct * 100.0, stage_info, hold_days
        );

        world.entity_mut(entity).insert(ClosedTrade {
            code: position.code.clone(),
            entry_date: position.entry_date,
            exit_date: end_date,
            entry_price: position.entry_price,
            exit_price,
            shares: position.initial_shares,
            pnl,
            pnl_pct,
            hold_days,
            exit_reason: ExitReason::EndOfBacktest,
        });
        world.entity_mut(entity).remove::<Position>();
    }
}
