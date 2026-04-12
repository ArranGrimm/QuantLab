//! B1 strategy ECS Systems
//!
//! 每日工作流 (模拟真实交易时序)：
//! 1. process_buy_signals   - [开盘] 根据昨日信号决定今日买入
//! 2. check_sell_conditions - [收盘] 根据今日收盘价决定是否卖出
//! 3. update_stats          - 更新统计数据

use bevy_ecs::prelude::*;
use bt_core::{BacktestStats, Portfolio};

use crate::components::{ClosedTrade, ExitReason, Position};
use crate::resources::{BacktestConfig, DailyData, MarketData};

/// System 1: Process buy signals [开盘]
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

    // 用昨日收盘价估算持仓市值
    let mut positions_value = 0.0;
    for position in positions.iter() {
        if let Some(prices) = market_data.prices.get(&position.code) {
            if let Some(bar) = prices.get(&current_date) {
                positions_value += position.shares as f64 * bar.pre_close;
            }
        }
    }
    let total_value = portfolio.cash + positions_value;
    let mut running_total_asset = portfolio.cash;
    for position in positions.iter() {
        if let Some(prices) = market_data.prices.get(&position.code) {
            if let Some(bar) = prices.get(&current_date) {
                running_total_asset += position.shares as f64 * bar.open;
            }
        }
    }

    let available_slots = (config.max_positions - current_positions).min(config.max_daily_buys);
    let mut bought_count = 0;

    for (code, _vol_ratio, open_price, stop_price) in daily_data.buy_candidates.iter() {
        if bought_count >= available_slots {
            break;
        }
        if existing_codes.contains(code) {
            continue;
        }

        let target_position_value = total_value * config.position_size_pct;
        let ideal_shares = bt_core::round_to_lot(target_position_value / open_price);
        if ideal_shares == 0 {
            continue;
        }

        let cost_per_share = open_price * (1.0 + config.commission_rate + config.slippage_pct);
        let affordable_shares = bt_core::round_to_lot(portfolio.cash / cost_per_share);
        let shares = ideal_shares.min(affordable_shares);

        let min_shares = (ideal_shares as f64 * config.min_position_ratio) as u32;
        if shares < min_shares || shares == 0 {
            continue;
        }

        let gross = shares as f64 * open_price;
        let commission = gross * config.commission_rate;
        let slippage = gross * config.slippage_pct;
        let cost = gross + commission + slippage;

        if cost > portfolio.cash {
            continue;
        }

        portfolio.cash -= cost;
        running_total_asset -= commission + slippage;
        stats.record_costs(commission, 0.0, slippage);

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
            high_since_entry: *open_price,
            trailing_stop_active: false,
        });

        println!(
            "[{}] [BUY] {} @ {:.2} x {} shares | Total Asset: {:.2}",
            current_date,
            code,
            open_price,
            shares,
            running_total_asset
        );
        bought_count += 1;
    }
}

/// System 2: Check sell conditions [收盘]
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

    let mut running_total_asset = portfolio.cash;
    for (_, position) in positions.iter_mut() {
        if let Some(prices) = market_data.prices.get(&position.code) {
            if let Some(bar) = prices.get(&current_date) {
                running_total_asset += position.shares as f64 * bar.close;
            }
        }
    }

    for (entity, mut position) in positions.iter_mut() {
        let bar = match market_data.prices.get(&position.code) {
            Some(prices) => match prices.get(&current_date) {
                Some(b) => b,
                None => continue,
            },
            None => continue,
        };

        position.update_high(bar.high);

        let hold_days = (current_date - position.entry_date).num_days() as i32;
        // A 股 T+1: 当天开仓的仓位不允许在同日收盘触发任何卖出逻辑。
        if hold_days <= 0 {
            continue;
        }
        let current_gain_pct = (bar.close - position.entry_price) / position.entry_price;

        // ── 激活移动止损 ──
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

        // ── 分批止盈 ──
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
                running_total_asset -= commission + stamp_duty + slippage;
                stats.record_costs(commission, stamp_duty, slippage);

                println!(
                    "[{}] [TP1] {} @ {:.2} | +{:.1}% | Sold {} shares | Remaining {}/{} | PnL: {:+.2} | Total Asset: {:.2}",
                    current_date,
                    position.code,
                    bar.close,
                    current_gain_pct * 100.0,
                    sell_shares,
                    position.shares,
                    position.initial_shares,
                    pnl,
                    running_total_asset
                );
            }
        } else if position.take_profit_stage == 1 && current_gain_pct >= config.tp2_pct {
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
                running_total_asset -= commission + stamp_duty + slippage;
                stats.record_costs(commission, stamp_duty, slippage);

                println!(
                    "[{}] [TP2] {} @ {:.2} | +{:.1}% | Sold {} shares | Remaining {}/{} | PnL: {:+.2} | Total Asset: {:.2}",
                    current_date,
                    position.code,
                    bar.close,
                    current_gain_pct * 100.0,
                    sell_shares,
                    position.shares,
                    position.initial_shares,
                    pnl,
                    running_total_asset
                );
            }
        }

        // ── 完全卖出检查 ──
        let mut should_sell_all = false;
        let mut exit_reason = ExitReason::MaxHoldDays;

        if config.stop_loss_enabled && bar.close <= position.stop_price {
            should_sell_all = true;
            exit_reason = ExitReason::StopLoss;
        } else if config.trailing_enabled && position.trailing_stop_active {
            let trailing_stop_price = position.trailing_stop_price(config.trailing_pct);
            if bar.close <= trailing_stop_price {
                should_sell_all = true;
                exit_reason = ExitReason::TrailingStop;
            }
        } else if config.sell_on_break_wl && position.take_profit_stage == 2 && bar.close < bar.wl {
            should_sell_all = true;
            exit_reason = ExitReason::BreakWL;
        } else if config.weak_enabled
            && position.take_profit_stage == 0
            && hold_days >= config.weak_days
            && current_gain_pct < config.weak_min_gain_pct
        {
            should_sell_all = true;
            exit_reason = ExitReason::WeakPerformance;
        } else if position.take_profit_stage < 2 && hold_days >= config.max_hold_days {
            should_sell_all = true;
            exit_reason = ExitReason::MaxHoldDays;
        }

        if should_sell_all && position.shares > 0 {
            let gross = position.shares as f64 * bar.close;
            let commission = gross * config.commission_rate;
            let stamp_duty = gross * config.stamp_duty_rate;
            let slippage = gross * config.slippage_pct;
            let net = gross - commission - stamp_duty - slippage;

            let exit_pnl = net - position.cost;
            let trade_pnl = exit_pnl + position.realized_pnl;
            let total_initial_cost = position.initial_shares as f64 * position.entry_price;
            let trade_pnl_pct = trade_pnl / total_initial_cost;

            portfolio.cash += net;
            running_total_asset -= commission + stamp_duty + slippage;
            stats.record_trade(trade_pnl, commission, stamp_duty, slippage);

            let stage_label = match position.take_profit_stage {
                0 => "None",
                1 => "TP1",
                2 => "TP2",
                _ => "Unknown",
            };

            println!(
                "[{}] [SELL] {} @ {:.2} | ExitPnL: {:+.2} | TradePnL: {:+.2} ({:+.2}%) | Stage: {} | Hold: {}d | {} | Asset: {:.2}",
                current_date,
                position.code,
                bar.close,
                exit_pnl,
                trade_pnl,
                trade_pnl_pct * 100.0,
                stage_label,
                hold_days,
                exit_reason,
                running_total_asset
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
                hold_days,
                exit_reason,
            });
            commands.entity(entity).remove::<Position>();
        }
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
    stats.update_drawdown(total_value, current_date);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::{App, Update};
    use chrono::NaiveDate;
    use std::collections::HashMap;

    fn build_market_data_for_day(
        code: &str,
        date: NaiveDate,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> MarketData {
        let mut by_date = HashMap::new();
        by_date.insert(
            date,
            crate::resources::PriceBar {
                open,
                high,
                low,
                close,
                pre_close: open,
                wl: close * 0.95,
                yl: close * 0.90,
                b1_signal: true,
                pre_b1_signal: true,
                is_loose: true,
                sort_value: 1.0,
            },
        );

        let mut prices = HashMap::new();
        prices.insert(code.to_string(), by_date);
        MarketData { prices }
    }

    #[test]
    fn does_not_sell_on_entry_day_even_if_take_profit_hits() {
        let trade_date = NaiveDate::from_ymd_opt(2026, 1, 22).unwrap();
        let mut app = App::new();
        let mut portfolio = Portfolio::new(100_000.0);
        portfolio.current_date = Some(trade_date);

        app.insert_resource(BacktestConfig::default());
        app.insert_resource(portfolio);
        app.insert_resource(BacktestStats::default());
        app.insert_resource(build_market_data_for_day(
            "sh.603778",
            trade_date,
            14.73,
            18.50,
            14.50,
            18.01,
        ));
        app.add_systems(Update, check_sell_conditions);
        app.world_mut().spawn(Position {
            code: "sh.603778".to_string(),
            entry_date: trade_date,
            entry_price: 14.73,
            stop_price: 14.00,
            shares: 2700,
            initial_shares: 2700,
            cost: 2700.0 * 14.73,
            realized_pnl: 0.0,
            take_profit_stage: 0,
            high_since_entry: 14.73,
            trailing_stop_active: false,
        });

        app.update();

        let world = app.world_mut();
        let mut query = world.query::<&Position>();
        let positions: Vec<Position> = query.iter(world).cloned().collect();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].shares, 2700);
        assert_eq!(positions[0].take_profit_stage, 0);
        assert_eq!(world.resource::<BacktestStats>().total_trades, 0);
    }

    #[test]
    fn can_take_profit_after_entry_day() {
        let entry_date = NaiveDate::from_ymd_opt(2026, 1, 21).unwrap();
        let trade_date = NaiveDate::from_ymd_opt(2026, 1, 22).unwrap();
        let mut app = App::new();
        let mut portfolio = Portfolio::new(100_000.0);
        portfolio.current_date = Some(trade_date);

        app.insert_resource(BacktestConfig::default());
        app.insert_resource(portfolio);
        app.insert_resource(BacktestStats::default());
        app.insert_resource(build_market_data_for_day(
            "sh.603778",
            trade_date,
            14.73,
            18.50,
            14.50,
            18.01,
        ));
        app.add_systems(Update, check_sell_conditions);
        app.world_mut().spawn(Position {
            code: "sh.603778".to_string(),
            entry_date,
            entry_price: 14.73,
            stop_price: 14.00,
            shares: 2700,
            initial_shares: 2700,
            cost: 2700.0 * 14.73,
            realized_pnl: 0.0,
            take_profit_stage: 0,
            high_since_entry: 14.73,
            trailing_stop_active: false,
        });

        app.update();

        let world = app.world_mut();
        let mut query = world.query::<&Position>();
        let positions: Vec<Position> = query.iter(world).cloned().collect();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].shares, 1800);
        assert_eq!(positions[0].take_profit_stage, 1);
    }
}
