//! B3 strategy ECS Systems

use bevy_ecs::prelude::*;
use bt_core::{BacktestStats, Portfolio};

use crate::components::{ClosedTrade, ExitReason, Position};
use crate::resources::{BacktestConfig, DailyData, MarketData};

pub fn process_buy_signals(
    mut commands: Commands,
    config: Res<BacktestConfig>,
    market_data: Res<MarketData>,
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

    let mut positions_value = 0.0;
    for position in positions.iter() {
        if let Some(prices) = market_data.prices.get(&position.code) {
            if let Some(bar) = prices.get(&current_date) {
                positions_value += position.shares as f64 * bar.pre_close;
            }
        }
    }
    let total_value = portfolio.cash + positions_value;
    let available_slots = (config.max_positions - current_positions).min(config.max_daily_buys);
    let mut bought_count = 0;

    for (code, _score, open_price, structural_stop_price) in daily_data.buy_candidates.iter() {
        if bought_count >= available_slots {
            break;
        }
        if existing_codes.contains(code) || *open_price <= 0.0 {
            continue;
        }

        let target_position_value = total_value * config.position_size_pct;
        let ideal_shares = bt_core::round_to_lot(target_position_value / open_price);
        if ideal_shares == 0 {
            continue;
        }
        let buy_cost_per_share = open_price * (1.0 + config.commission_rate + config.slippage_pct);
        let affordable_shares = bt_core::round_to_lot(portfolio.cash / buy_cost_per_share);
        let shares = ideal_shares.min(affordable_shares);
        let min_shares = (ideal_shares as f64 * config.min_position_ratio) as u32;
        if shares == 0 || shares < min_shares {
            continue;
        }

        let gross = shares as f64 * open_price;
        let commission = gross * config.commission_rate;
        let slippage = gross * config.slippage_pct;
        let total_cost = gross + commission + slippage;
        portfolio.cash -= total_cost;
        stats.record_costs(commission, 0.0, slippage);

        commands.spawn(Position {
            code: code.clone(),
            entry_date: current_date,
            entry_price: *open_price,
            structural_stop_price: *structural_stop_price,
            shares,
            initial_shares: shares,
            cost: total_cost,
            high_since_entry: *open_price,
            hold_trading_days: 0,
            trailing_stop_active: false,
        });
        println!(
            "[{}] [BUY] {} @ {:.2} x {} | StructuralStop {:.2}",
            current_date, code, open_price, shares, structural_stop_price
        );
        bought_count += 1;
    }
}

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

        position.update_high(bar.high);
        if current_date <= position.entry_date {
            continue;
        }
        position.hold_trading_days += 1;

        let current_gain_pct = (bar.close - position.entry_price) / position.entry_price;
        if config.trailing_enabled
            && !position.trailing_stop_active
            && current_gain_pct >= config.trailing_activation_pct
        {
            position.trailing_stop_active = true;
        }

        let mut should_sell = false;
        let mut exit_reason = ExitReason::MaxHoldDays;
        let structural_stop = position.structural_stop_price * (1.0 - config.structural_stop_buffer_pct);
        let white_line_stop = bar.white_line * (1.0 - config.break_white_line_buffer_pct);

        if config.structural_stop_enabled && bar.close <= structural_stop {
            should_sell = true;
            exit_reason = ExitReason::StructuralStop;
        }
        if !should_sell
            && config.fast_fail_enabled
            && position.hold_trading_days <= config.fast_fail_days
            && (
                current_gain_pct <= -config.fast_fail_loss_pct
                    || (config.break_white_line_enabled && bar.close < white_line_stop)
            )
        {
            should_sell = true;
            exit_reason = ExitReason::FastFail;
        }
        if !should_sell && config.break_white_line_enabled && bar.close < white_line_stop {
            should_sell = true;
            exit_reason = ExitReason::BreakWhiteLine;
        }
        if !should_sell && config.trailing_enabled && position.trailing_stop_active {
            let trailing_stop = position.trailing_stop_price(config.trailing_pct);
            if bar.close <= trailing_stop {
                should_sell = true;
                exit_reason = ExitReason::TrailingStop;
            }
        }
        if !should_sell
            && config.weak_enabled
            && position.hold_trading_days >= config.weak_days
            && current_gain_pct < config.weak_min_gain_pct
        {
            should_sell = true;
            exit_reason = ExitReason::WeakPerformance;
        }
        if !should_sell && position.hold_trading_days >= config.max_hold_trading_days {
            should_sell = true;
            exit_reason = ExitReason::MaxHoldDays;
        }

        if should_sell && position.shares > 0 {
            let gross = position.shares as f64 * bar.close;
            let commission = gross * config.commission_rate;
            let stamp_duty = gross * config.stamp_duty_rate;
            let slippage = gross * config.slippage_pct;
            let net = gross - commission - stamp_duty - slippage;
            let trade_pnl = net - position.cost;
            let initial_cost = position.initial_shares as f64 * position.entry_price;
            let trade_pnl_pct = trade_pnl / initial_cost;

            portfolio.cash += net;
            stats.record_trade(trade_pnl, commission, stamp_duty, slippage);
            println!(
                "[{}] [SELL] {} @ {:.2} | PnL {:+.2}% | Hold {}td | {}",
                current_date,
                position.code,
                bar.close,
                trade_pnl_pct * 100.0,
                position.hold_trading_days,
                exit_reason
            );

            commands.entity(entity).insert(ClosedTrade {
                code: position.code.clone(),
                entry_date: position.entry_date,
                exit_date: current_date,
                entry_price: position.entry_price,
                exit_price: bar.close,
                shares: position.initial_shares,
                pnl: trade_pnl,
                pnl_pct: trade_pnl_pct,
                hold_trading_days: position.hold_trading_days,
                exit_reason,
            });
            commands.entity(entity).remove::<Position>();
        }
    }
}

pub fn update_stats(
    portfolio: Res<Portfolio>,
    mut stats: ResMut<BacktestStats>,
    positions: Query<&Position>,
    market_data: Res<MarketData>,
) {
    if let Some(date) = portfolio.current_date {
        let mut total_value = portfolio.cash;
        for position in positions.iter() {
            if let Some(prices) = market_data.prices.get(&position.code) {
                if let Some(bar) = prices.get(&date) {
                    total_value += position.shares as f64 * bar.close;
                }
            }
        }
        stats.update_drawdown(total_value, date);
    }
}
