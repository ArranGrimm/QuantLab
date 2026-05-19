//! AMV close-to-close cohort diagnostic ECS systems.

use bevy_ecs::prelude::*;
use bt_core::{BacktestStats, Portfolio};
use std::collections::HashSet;

use crate::components::{ClosedTrade, ExitReason, Position};
use crate::resources::{CohortDiagnosticConfig, DailyData, MarketData};

pub fn process_buy_signals(
    mut commands: Commands,
    config: Res<CohortDiagnosticConfig>,
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
                positions_value += position.shares as f64 * bar.close;
            }
        }
    }

    let total_value = portfolio.cash + positions_value;
    let available_slots = (config.max_positions - current_positions).min(config.max_daily_buys);
    let cost_model = &config.cost_model;
    let mut bought_count = 0;

    for (code, _score, close_price) in daily_data.buy_candidates.iter() {
        if bought_count >= available_slots {
            break;
        }
        if existing_codes.contains(code) {
            continue;
        }

        let target_value = total_value * config.position_size_pct;
        let ideal_shares = bt_core::round_to_lot(code, target_value / close_price);
        if ideal_shares == 0 {
            continue;
        }

        let cost_per_share =
            close_price * (1.0 + cost_model.commission_rate + cost_model.slippage_pct);
        let affordable_shares = bt_core::round_to_lot(code, portfolio.cash / cost_per_share);
        let shares = ideal_shares.min(affordable_shares);
        let min_shares = (ideal_shares as f64 * config.min_position_ratio) as u32;
        if shares < min_shares || shares == 0 {
            continue;
        }

        let gross = shares as f64 * close_price;
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
            entry_price: *close_price,
            shares,
            cost,
        });

        println!(
            "[{}] [BUY-CLOSE] {} @ {:.2} x {} shares",
            current_date, code, close_price, shares
        );
        bought_count += 1;
    }
}

pub fn check_exit_conditions(
    mut commands: Commands,
    config: Res<CohortDiagnosticConfig>,
    market_data: Res<MarketData>,
    mut portfolio: ResMut<Portfolio>,
    mut stats: ResMut<BacktestStats>,
    positions: Query<(Entity, &Position)>,
) {
    let current_date = match portfolio.current_date {
        Some(d) => d,
        None => return,
    };
    let current_trade_index = match market_data.date_index.get(&current_date) {
        Some(idx) => *idx,
        None => return,
    };
    let cost_model = &config.cost_model;

    for (entity, position) in positions.iter() {
        let hold_trading_days = current_trade_index - position.entry_trade_index;
        if hold_trading_days <= 0 || hold_trading_days < config.max_hold_trading_days {
            continue;
        }

        let bar = match market_data.prices.get(&position.code) {
            Some(prices) => match prices.get(&current_date) {
                Some(b) => b,
                None => continue,
            },
            None => continue,
        };

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
            "[{}] [SELL-CLOSE] {} @ {:.2} | PnL: {:+.1}% | Hold: {}td",
            current_date,
            position.code,
            bar.close,
            pnl_pct * 100.0,
            hold_trading_days
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
            exit_reason: ExitReason::MaxHoldDays,
        });
        commands.entity(entity).remove::<Position>();
    }
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
