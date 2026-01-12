//! Quant Backtest Engine - ECS-based backtesting using Bevy
//!
//! 设计理念：
//! - 加载完整市场数据（包含 b1_signal, pre_b1_signal, is_loose 标记）
//! - 每日工作流：先买入(开盘)，再卖出(收盘)
//! - 支持 TOML 配置文件
//!
//! Usage:
//!   cargo run --release -- --data ../data/signals/market_data.parquet
//!   cargo run --release -- --config config.toml --data ../data/signals/market_data.parquet

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

use components::Position;
use resources::{BacktestConfig, BacktestStats, ConfigFile, DailyData, MarketData, Portfolio, PriceBar};
use systems::{check_sell_conditions, process_buy_signals, update_stats};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to market data parquet file
    #[arg(short, long, default_value = "../data/signals/market_data.parquet")]
    data: PathBuf,

    /// Path to config file (TOML)
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("========================================");
    println!("   Quant Backtest Engine (Bevy ECS)");
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
    let (market_data, trading_dates) = build_market_data(&df)?;
    println!(
        "Stocks: {}, Trading days: {}",
        market_data.prices.len(),
        trading_dates.len()
    );

    // 4. Initialize Bevy App
    let mut app = App::new();

    // 5. Add resources
    let initial_capital = config.initial_capital;
    app.insert_resource(config);
    app.insert_resource(Portfolio::new(initial_capital));
    app.insert_resource(BacktestStats::default());
    app.insert_resource(market_data);
    app.insert_resource(DailyData::default());

    // 6. Add systems
    // 重要：先买入(开盘)，再卖出(收盘)，这样卖出释放的仓位要到下一天才能用
    app.add_systems(
        Update,
        (process_buy_signals, check_sell_conditions, update_stats).chain(),
    );

    // 7. Run backtest by date
    println!("\nRunning backtest over {} trading days...\n", trading_dates.len());

    for date in &trading_dates {
        // Update current date and daily candidates
        {
            let world = app.world_mut();
            world.resource_mut::<Portfolio>().current_date = Some(*date);

            // Build today's buy candidates (pre_b1_signal && is_loose)
            let market_data = world.resource::<MarketData>();
            let mut candidates: Vec<_> = market_data
                .prices
                .iter()
                .filter_map(|(code, dates)| {
                    dates.get(date).and_then(|bar| {
                        if bar.pre_b1_signal && bar.is_loose {
                            Some((code.clone(), bar.vol_ratio, bar.open, bar.stop_price))
                        } else {
                            None
                        }
                    })
                })
                .collect();

            // Sort by vol_ratio (ascending = more shrinkage = priority)
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            world.resource_mut::<DailyData>().buy_candidates = candidates;
        }

        // Run one tick
        app.update();
    }

    // 8. Force close remaining positions
    if let Some(end_date) = trading_dates.last() {
        force_close_all_positions(&mut app, *end_date);
    }

    // 9. Print results
    print_results(&app, initial_capital);

    Ok(())
}

/// Print current config
fn print_config(config: &BacktestConfig) {
    println!("\n--- Configuration ---");
    println!("Initial Capital: {:.0}", config.initial_capital);
    println!("Max Positions: {} (Daily: {})", config.max_positions, config.max_daily_buys);
    println!("Position Size: {:.0}%", config.position_size_pct * 100.0);
    println!("Max Hold Days: {}", config.max_hold_days);
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
    println!("----------------------");
}

/// Build market data from DataFrame
fn build_market_data(
    df: &DataFrame,
) -> Result<(MarketData, Vec<NaiveDate>), Box<dyn std::error::Error>> {
    let mut market_data = MarketData::default();
    let mut all_dates = Vec::new();

    // Get columns
    let codes = df.column("code")?.str()?;
    let dates = df.column("date")?.date()?;
    let open_adj = df.column("open_adj")?.f64()?;
    let high_adj = df.column("high_adj")?.f64()?;
    let low_adj = df.column("low_adj")?.f64()?;
    let close_adj = df.column("close_adj")?.f64()?;
    let wl = df.column("WL")?.f64()?;
    let yl = df.column("YL")?.f64()?;
    let b1_signal = df.column("b1_signal")?.bool()?;
    let pre_b1_signal = df.column("pre_b1_signal")?.bool()?;
    let is_loose = df.column("is_loose")?.bool()?;
    let vol_ratio = df.column("vol_ratio")?.f64()?;
    let stop_price = df.column("stop_price")?.f64()?;

    for i in 0..df.height() {
        let code = codes.get(i).ok_or("Missing code")?;
        let date_days = dates.get(i).ok_or("Missing date")?;
        let date =
            NaiveDate::from_num_days_from_ce_opt(date_days + 719163).ok_or("Invalid date")?;

        let bar = PriceBar {
            open: open_adj.get(i).unwrap_or(0.0),
            high: high_adj.get(i).unwrap_or(0.0),
            low: low_adj.get(i).unwrap_or(0.0),
            close: close_adj.get(i).unwrap_or(0.0),
            wl: wl.get(i).unwrap_or(0.0),
            yl: yl.get(i).unwrap_or(0.0),
            b1_signal: b1_signal.get(i).unwrap_or(false),
            pre_b1_signal: pre_b1_signal.get(i).unwrap_or(false),
            is_loose: is_loose.get(i).unwrap_or(false),
            vol_ratio: vol_ratio.get(i).unwrap_or(999.0),
            stop_price: stop_price.get(i).unwrap_or(0.0),
        };

        market_data
            .prices
            .entry(code.to_string())
            .or_insert_with(HashMap::new)
            .insert(date, bar);

        all_dates.push(date);
    }

    // Get unique sorted trading dates
    all_dates.sort();
    all_dates.dedup();

    Ok((market_data, all_dates))
}

/// Force close all remaining positions
fn force_close_all_positions(app: &mut App, end_date: NaiveDate) {
    use components::{ClosedTrade, ExitReason};

    let world = app.world_mut();

    let market_data = world.resource::<MarketData>().clone();
    let config = world.resource::<BacktestConfig>().clone();

    // Find all positions
    let mut to_close: Vec<(Entity, Position, f64)> = Vec::new();
    {
        let mut query = world.query::<(Entity, &Position)>();
        for (entity, position) in query.iter(world) {
            if let Some(prices) = market_data.prices.get(&position.code) {
                // Find latest price <= end_date
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

    // Close positions
    for (entity, position, exit_price) in to_close {
        let gross = position.shares as f64 * exit_price;
        let commission = gross * config.commission_rate;
        let stamp_duty = gross * config.stamp_duty_rate;
        let slippage = gross * config.slippage_pct;
        let net = gross - commission - stamp_duty - slippage;
        
        // 总 PnL = 当前持仓盈亏 + 已实现盈亏 (分批止盈)
        let pnl = (net - position.cost) + position.realized_pnl;
        let total_initial_cost = position.initial_shares as f64 * position.entry_price;
        let pnl_pct = pnl / total_initial_cost;
        let hold_days = (end_date - position.entry_date).num_days() as i32;

        world.resource_mut::<Portfolio>().cash += net;

        {
            let mut stats = world.resource_mut::<BacktestStats>();
            stats.total_trades += 1;
            stats.total_pnl += pnl;
            stats.total_commission += commission;
            stats.total_stamp_duty += stamp_duty;
            stats.total_slippage += slippage;
            if pnl > 0.0 {
                stats.winning_trades += 1;
            } else {
                stats.losing_trades += 1;
            }
        }

        let stage_info = match position.take_profit_stage {
            0 => "",
            1 => " (TP1)",
            2 => " (TP2)",
            _ => "",
        };
        println!(
            "[{}] [CLOSE] {} @ {:.2} | PnL: {:+.2}%{} | Hold: {} days",
            end_date,
            position.code,
            exit_price,
            pnl_pct * 100.0,
            stage_info,
            hold_days
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

fn print_results(app: &App, initial_capital: f64) {
    let stats = app.world().resource::<BacktestStats>();
    let portfolio = app.world().resource::<Portfolio>();

    println!("\n========================================");
    println!("           Backtest Results");
    println!("========================================");
    println!("Total Trades: {}", stats.total_trades);
    println!("Win Rate: {:.1}%", stats.win_rate() * 100.0);
    println!("Total PnL: {:+.2}", stats.total_pnl);
    println!("Final Portfolio: {:.2}", portfolio.cash);
    println!(
        "Total Return: {:+.2}%",
        (portfolio.cash / initial_capital - 1.0) * 100.0
    );
    println!("Max Drawdown: {:.2}%", stats.max_drawdown * 100.0);
    println!("----------------------------------------");
    println!("Trading Costs:");
    println!("  Commission: {:.2}", stats.total_commission);
    println!("  Stamp Duty: {:.2}", stats.total_stamp_duty);
    println!("  Slippage:   {:.2}", stats.total_slippage);
    println!("  Total:      {:.2}", stats.total_costs());
    println!("========================================");
}
