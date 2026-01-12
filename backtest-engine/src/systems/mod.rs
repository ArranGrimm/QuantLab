//! ECS Systems for the backtesting engine
//!
//! 每日工作流 (模拟真实交易时序)：
//! 1. process_buy_signals   - [开盘] 根据昨日信号决定今日买入
//! 2. check_sell_conditions - [收盘] 根据今日收盘价决定是否卖出
//! 3. update_stats          - 更新统计数据
//!
//! 分批止盈策略：
//! - Stage 0: 涨 TP1% → 卖 1/3，进入 Stage 1
//! - Stage 1: 涨 TP2% → 卖 1/3，进入 Stage 2
//! - Stage 2: 跌破 WL → 卖出剩余
//!
//! 移动止损 (Trailing Stop)：
//! - 涨幅达到 activation_pct 后激活
//! - 从最高点回撤 trailing_pct 止损

use bevy_ecs::prelude::*;

use crate::components::{ClosedTrade, ExitReason, Position};
use crate::resources::{BacktestConfig, BacktestStats, DailyData, MarketData, Portfolio};

/// System 2: Check sell conditions for all positions [收盘时执行]
pub fn check_sell_conditions(
    mut commands: Commands,
    config: Res<BacktestConfig>,
    market_data: Res<MarketData>,
    mut portfolio: ResMut<Portfolio>,
    mut stats: ResMut<BacktestStats>,
    mut positions: Query<(Entity, &mut Position)>,
) {
    let current_date = match portfolio.current_date {
        Some(d) => d,
        None => return,
    };

    for (entity, mut position) in positions.iter_mut() {
        let bar = match market_data.prices.get(&position.code) {
            Some(prices) => match prices.get(&current_date) {
                Some(b) => b,
                None => continue,
            },
            None => continue,
        };

        // 更新最高价 (用于移动止损)
        position.update_high(bar.high);

        let hold_days = (current_date - position.entry_date).num_days() as i32;
        let current_gain_pct = (bar.close - position.entry_price) / position.entry_price;

        // ========== 激活移动止损 ==========
        if config.trailing_enabled 
            && !position.trailing_stop_active 
            && current_gain_pct >= config.trailing_activation_pct 
        {
            position.trailing_stop_active = true;
            println!(
                "[{}] [TRAILING ACTIVATED] {} | Gain: +{:.1}% | High: {:.2}",
                current_date, position.code, current_gain_pct * 100.0, position.high_since_entry
            );
        }

        // ========== 分批止盈检查 ==========
        // Stage 0 → 1: 涨 TP1%，卖 1/3
        if position.take_profit_stage == 0 && current_gain_pct >= config.tp1_pct {
            let sell_shares = position.initial_shares / 3;
            if sell_shares > 0 {
                let gross = sell_shares as f64 * bar.close;
                let commission = gross * config.commission_rate;
                let stamp_duty = gross * config.stamp_duty_rate;
                let slippage = gross * config.slippage_pct;
                let net = gross - commission - stamp_duty - slippage;
                
                let sell_cost = (sell_shares as f64 / position.shares as f64) * position.cost;
                let pnl = net - sell_cost;

                portfolio.cash += net;
                position.shares -= sell_shares;
                position.cost -= sell_cost;
                position.realized_pnl += pnl;
                position.take_profit_stage = 1;

                // 统计交易成本
                stats.total_commission += commission;
                stats.total_stamp_duty += stamp_duty;
                stats.total_slippage += slippage;

                println!(
                    "[{}] [TP1] {} @ {:.2} | +{:.1}% | Sold {}/{} shares | PnL: {:+.2}",
                    current_date, position.code, bar.close, current_gain_pct * 100.0,
                    sell_shares, position.initial_shares, pnl
                );
            }
        }
        // Stage 1 → 2: 涨 TP2%，再卖 1/3
        else if position.take_profit_stage == 1 && current_gain_pct >= config.tp2_pct {
            let sell_shares = position.initial_shares / 3;
            if sell_shares > 0 && position.shares >= sell_shares {
                let gross = sell_shares as f64 * bar.close;
                let commission = gross * config.commission_rate;
                let stamp_duty = gross * config.stamp_duty_rate;
                let slippage = gross * config.slippage_pct;
                let net = gross - commission - stamp_duty - slippage;
                
                let sell_cost = (sell_shares as f64 / position.shares as f64) * position.cost;
                let pnl = net - sell_cost;

                portfolio.cash += net;
                position.shares -= sell_shares;
                position.cost -= sell_cost;
                position.realized_pnl += pnl;
                position.take_profit_stage = 2;

                // 统计交易成本
                stats.total_commission += commission;
                stats.total_stamp_duty += stamp_duty;
                stats.total_slippage += slippage;

                println!(
                    "[{}] [TP2] {} @ {:.2} | +{:.1}% | Sold {}/{} shares | PnL: {:+.2}",
                    current_date, position.code, bar.close, current_gain_pct * 100.0,
                    sell_shares, position.initial_shares, pnl
                );
            }
        }

        // ========== 完全卖出检查 ==========
        let mut should_sell_all = false;
        let mut exit_reason = ExitReason::MaxHoldDays;

        // 1. 固定止损 (适用于所有阶段)
        if config.stop_loss_enabled && bar.close <= position.stop_price {
            should_sell_all = true;
            exit_reason = ExitReason::StopLoss;
        }
        // 2. 移动止损 (如果已激活)
        else if config.trailing_enabled && position.trailing_stop_active {
            let trailing_stop_price = position.trailing_stop_price(config.trailing_pct);
            if bar.close <= trailing_stop_price {
                should_sell_all = true;
                exit_reason = ExitReason::TrailingStop;
            }
        }
        // 3. Stage 2 跌破 WL → 卖出剩余
        else if config.sell_on_break_wl && position.take_profit_stage == 2 && bar.close < bar.wl {
            should_sell_all = true;
            exit_reason = ExitReason::BreakWL;
        }
        // 4. 弱势清仓: N 天后涨幅不足 (仅 Stage 0，未触发过止盈)
        else if config.weak_enabled 
            && position.take_profit_stage == 0 
            && hold_days >= config.weak_days 
            && current_gain_pct < config.weak_min_gain_pct 
        {
            should_sell_all = true;
            exit_reason = ExitReason::WeakPerformance;
        }
        // 5. 最大持有期 (适用于 Stage 0, 1)
        else if position.take_profit_stage < 2 && hold_days >= config.max_hold_days {
            should_sell_all = true;
            exit_reason = ExitReason::MaxHoldDays;
        }

        if should_sell_all && position.shares > 0 {
            let gross = position.shares as f64 * bar.close;
            let commission = gross * config.commission_rate;
            let stamp_duty = gross * config.stamp_duty_rate;
            let slippage = gross * config.slippage_pct;
            let net = gross - commission - stamp_duty - slippage;
            
            let pnl = net - position.cost + position.realized_pnl;
            let total_initial_cost = position.initial_shares as f64 * position.entry_price;
            let pnl_pct = pnl / total_initial_cost;

            portfolio.cash += net;
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

            let stage_info = match position.take_profit_stage {
                0 => "",
                1 => " (TP1)",
                2 => " (TP2)",
                _ => "",
            };

            println!(
                "[{}] [SELL] {} @ {:.2} | PnL: {:+.2}%{} | Hold: {} days | {}",
                current_date, position.code, bar.close, pnl_pct * 100.0, stage_info, hold_days, exit_reason
            );

            commands.entity(entity).insert(ClosedTrade {
                code: position.code.clone(),
                entry_date: position.entry_date,
                exit_date: current_date,
                entry_price: position.entry_price,
                exit_price: bar.close,
                shares: position.initial_shares,
                pnl,
                pnl_pct,
                hold_days,
                exit_reason,
            });
            commands.entity(entity).remove::<Position>();
        }
    }
}

/// System 1: Process buy signals from daily candidates [开盘时执行]
pub fn process_buy_signals(
    mut commands: Commands,
    config: Res<BacktestConfig>,
    mut portfolio: ResMut<Portfolio>,
    mut stats: ResMut<BacktestStats>,
    daily_data: Res<DailyData>,
    positions: Query<&Position>,
) {
    let current_date = match portfolio.current_date {
        Some(d) => d,
        None => return,
    };

    let current_positions = positions.iter().count();
    if current_positions >= config.max_positions {
        return;
    }

    let existing_codes: std::collections::HashSet<_> =
        positions.iter().map(|p| p.code.clone()).collect();

    // 每天可买数量 = min(每日上限, 剩余仓位)
    let available_slots = (config.max_positions - current_positions).min(config.max_daily_buys);
    let mut bought_count = 0;

    for (code, _vol_ratio, open_price, stop_price) in daily_data.buy_candidates.iter() {
        if bought_count >= available_slots {
            break;
        }

        if existing_codes.contains(code) {
            continue;
        }

        let position_value = portfolio.cash * config.position_size_pct;
        if position_value < open_price * 100.0 {
            continue;
        }

        // 股数取整到 300 的倍数，方便分批卖出 (1/3)
        let shares = ((position_value / open_price / 300.0).floor() as u32) * 300;
        if shares == 0 {
            continue;
        }

        // 计算买入成本 (含佣金和滑点)
        let gross = shares as f64 * open_price;
        let commission = gross * config.commission_rate;
        let slippage = gross * config.slippage_pct;
        let cost = gross + commission + slippage;
        
        if cost > portfolio.cash {
            continue;
        }

        portfolio.cash -= cost;
        
        // 统计交易成本
        stats.total_commission += commission;
        stats.total_slippage += slippage;

        commands.spawn(Position {
            code: code.clone(),
            entry_date: current_date,
            entry_price: *open_price,
            stop_price: *stop_price,
            shares,
            initial_shares: shares,
            cost,
            realized_pnl: 0.0,
            take_profit_stage: 0,
            high_since_entry: *open_price,  // 初始最高价 = 买入价
            trailing_stop_active: false,
        });

        println!("[{}] [BUY] {} @ {:.2} x {} shares", current_date, code, open_price, shares);

        bought_count += 1;
    }
}

/// System 3: Update statistics [收盘后]
pub fn update_stats(
    portfolio: Res<Portfolio>,
    mut stats: ResMut<BacktestStats>,
    positions: Query<&Position>,
    market_data: Res<MarketData>,
) {
    let current_date = match portfolio.current_date {
        Some(d) => d,
        None => return,
    };

    let mut positions_value = 0.0;
    for position in positions.iter() {
        if let Some(prices) = market_data.prices.get(&position.code) {
            if let Some(bar) = prices.get(&current_date) {
                positions_value += position.shares as f64 * bar.close;
            }
        }
    }

    let total_value = portfolio.total_value(positions_value);

    if total_value > stats.peak_value {
        stats.peak_value = total_value;
    }

    if stats.peak_value > 0.0 {
        let drawdown = (stats.peak_value - total_value) / stats.peak_value;
        if drawdown > stats.max_drawdown {
            stats.max_drawdown = drawdown;
        }
    }
}
