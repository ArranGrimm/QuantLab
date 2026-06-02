//! AMV TopN strategy ECS systems.
//!
//! Daily workflow:
//! 1. process_buy_signals - [open] buy T+1 shifted signals.
//! 2. check_exit_conditions - [close] sell by risk or max hold days.
//! 3. update_stats - [close] update portfolio drawdown.

use bevy_ecs::prelude::*;
use bt_core::{BacktestStats, Portfolio};
use std::collections::HashSet;

use crate::components::{ClosedTrade, ExitReason, Position};
use crate::resources::{AmvTopnConfig, DailyData, MarketData, PriceBar};

pub fn process_buy_signals(
    mut commands: Commands,
    config: Res<AmvTopnConfig>,
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
    let entry_trade_index = match market_data.date_index.get(&current_date) {
        Some(idx) => *idx,
        None => return,
    };

    let current_positions = positions.iter().count();
    if current_positions >= config.max_positions {
        return;
    }

    let existing_codes: HashSet<_> = positions.iter().map(|p| p.code.clone()).collect();

    let mut positions_value = 0.0;
    for position in positions.iter() {
        if let Some(prices) = market_data.prices.get(&position.code) {
            if let Some(bar) = prices.get(&current_date) {
                positions_value += position.shares as f64 * bar.open;
            }
        }
    }
    let total_value = portfolio.cash + positions_value;
    let available_slots = (config.max_positions - current_positions).min(config.max_daily_buys);
    let cost_model = &config.cost_model;
    let mut bought_count = 0;

    for (code, _score, open_price) in daily_data.buy_candidates.iter() {
        if bought_count >= available_slots {
            break;
        }
        if !config.allow_duplicate_positions && existing_codes.contains(code) {
            continue;
        }

        let target_value = total_value * config.position_size_pct;
        let ideal_shares = bt_core::round_to_lot(code, target_value / open_price);
        if ideal_shares == 0 {
            continue;
        }

        let cost_per_share =
            open_price * (1.0 + cost_model.commission_rate + cost_model.slippage_pct);
        let affordable_shares = bt_core::round_to_lot(code, portfolio.cash / cost_per_share);
        let shares = ideal_shares.min(affordable_shares);
        let min_shares = (ideal_shares as f64 * config.min_position_ratio) as u32;
        if shares < min_shares || shares == 0 {
            continue;
        }

        let gross = shares as f64 * open_price;
        let commission = gross * cost_model.commission_rate;
        let slippage = gross * cost_model.slippage_pct;
        let cost = gross + commission + slippage;
        if cost > portfolio.cash {
            continue;
        }

        portfolio.cash -= cost;
        stats.record_costs(commission, 0.0, slippage);
        commands.spawn(Position {
            code: code.clone(),
            entry_date: current_date,
            entry_trade_index,
            entry_price: *open_price,
            shares,
            cost,
            high_since_entry: *open_price,
            trailing_stop_active: false,
        });

        println!(
            "[{}] [BUY] {} @ {:.2} x {} shares",
            current_date, code, open_price, shares
        );
        bought_count += 1;
    }
}

pub fn check_exit_conditions(
    mut commands: Commands,
    config: Res<AmvTopnConfig>,
    market_data: Res<MarketData>,
    mut portfolio: ResMut<Portfolio>,
    mut stats: ResMut<BacktestStats>,
    mut positions: Query<(Entity, &mut Position)>,
) {
    let current_date = match portfolio.current_date {
        Some(d) => d,
        None => return,
    };
    let cost_model = &config.cost_model;

    for (entity, mut position) in positions.iter_mut() {
        let bar = match market_data.prices.get(&position.code) {
            Some(prices) => match prices.get(&current_date) {
                Some(b) => b,
                None => continue,
            },
            None => continue,
        };

        position.update_high(bar.high);
        let current_trade_index = match market_data.date_index.get(&current_date) {
            Some(idx) => *idx,
            None => continue,
        };
        let hold_trading_days = current_trade_index - position.entry_trade_index;
        if position.shares == 0 {
            if hold_trading_days >= config.max_hold_trading_days {
                commands.entity(entity).remove::<Position>();
            }
            continue;
        }
        // A-share T+1: a position opened this morning cannot be sold at today's close.
        if hold_trading_days <= 0 {
            continue;
        }

        let current_gain = (bar.close - position.entry_price) / position.entry_price;
        if config.trailing_enabled
            && !position.trailing_stop_active
            && current_gain >= config.trailing_activation_pct
        {
            position.trailing_stop_active = true;
        }

        let mut should_sell = false;
        let mut exit_reason = ExitReason::MaxHoldDays;
        if config.early_stop_enabled
            && hold_trading_days >= config.early_stop_trigger_hold_trading_days
            && is_early_stop_triggered(&config, bar, position.entry_price)
        {
            should_sell = true;
            exit_reason = ExitReason::EarlyStop;
        } else if config.stop_loss_enabled
            && bar.close <= position.entry_price * (1.0 - config.stop_loss_pct)
        {
            should_sell = true;
            exit_reason = ExitReason::StopLoss;
        } else if config.trailing_enabled && position.trailing_stop_active {
            if bar.close <= position.trailing_stop_price(config.trailing_pct) {
                should_sell = true;
                exit_reason = ExitReason::TrailingStop;
            }
        } else if config.sell_on_bear_regime && !bar.is_bull_regime {
            should_sell = true;
            exit_reason = ExitReason::BearRegime;
        } else if hold_trading_days >= config.max_hold_trading_days {
            should_sell = true;
            exit_reason = ExitReason::MaxHoldDays;
        }

        if should_sell && position.shares > 0 {
            let limit = bt_core::price_limit_pct(&position.code);
            if bt_core::is_limit_down(bar.close, bar.pre_close, limit) {
                println!(
                    "[{}] [LOCKED] {} 跌停 {:.2}, 无法卖出",
                    current_date, position.code, bar.close
                );
                continue;
            }

            let gross = position.shares as f64 * bar.close;
            let commission = gross * cost_model.commission_rate;
            let stamp_duty = gross * cost_model.stamp_duty_rate;
            let slippage = gross * cost_model.slippage_pct;
            let net = gross - commission - stamp_duty - slippage;
            let pnl = net - position.cost;
            let pnl_pct = pnl / position.cost;

            portfolio.cash += net;
            stats.record_trade(pnl, commission, stamp_duty, slippage);

            println!(
                "[{}] [SELL] {} @ {:.2} | PnL: {:+.1}% | Hold: {}td | {}",
                current_date,
                position.code,
                bar.close,
                pnl_pct * 100.0,
                hold_trading_days,
                exit_reason
            );

            commands.entity(entity).insert(ClosedTrade {
                code: position.code.clone(),
                entry_date: position.entry_date,
                exit_date: current_date,
                entry_price: position.entry_price,
                exit_price: bar.close,
                shares: position.shares,
                cost: position.cost,
                exit_value: net,
                pnl,
                pnl_pct,
                hold_trading_days,
                exit_reason,
            });
            commands.entity(entity).remove::<Position>();
        }
    }
}

fn is_early_stop_triggered(
    config: &AmvTopnConfig,
    bar: &PriceBar,
    entry_price: f64,
) -> bool {
    // ATR-based stop: close < entry_price - N × ATR_14
    bar.atr_14 > 0.0
        && bar.close < entry_price - config.early_stop_atr_multiple * bar.atr_14
}

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
    stats.update_drawdown(total_value, current_date);
}
