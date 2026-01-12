//! ECS Resources for the backtesting engine

use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use std::collections::HashMap;

/// Backtest Configuration
#[derive(Resource, Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub position_size_pct: f64,
    pub stop_loss_pct: f64,
    pub max_hold_days: i32,
    pub sell_on_break_wl: bool,
    pub sell_on_break_yl: bool,
    pub slippage_pct: f64,
    pub commission_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            max_positions: 5,
            position_size_pct: 0.2,
            stop_loss_pct: 0.03,
            max_hold_days: 30,
            sell_on_break_wl: false,
            sell_on_break_yl: false,
            slippage_pct: 0.001,
            commission_pct: 0.0003,
        }
    }
}

/// Portfolio state
#[derive(Resource, Debug)]
pub struct Portfolio {
    pub cash: f64,
    pub initial_capital: f64,
    pub current_date: Option<NaiveDate>,
}

impl Portfolio {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            initial_capital,
            current_date: None,
        }
    }

    pub fn total_value(&self, positions_value: f64) -> f64 {
        self.cash + positions_value
    }
}

/// Price bar with all signals
#[derive(Debug, Clone)]
pub struct PriceBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub wl: f64,
    pub yl: f64,
    pub b1_signal: bool,
    pub pre_b1_signal: bool,
    pub is_loose: bool,
    pub vol_ratio: f64,
    pub stop_price: f64,
}

/// Market data: code -> date -> PriceBar
#[derive(Resource, Default, Clone)]
pub struct MarketData {
    pub prices: HashMap<String, HashMap<NaiveDate, PriceBar>>,
}

/// Daily data (buy candidates for today)
#[derive(Resource, Default)]
pub struct DailyData {
    /// (code, vol_ratio, open_price, stop_price)
    pub buy_candidates: Vec<(String, f64, f64, f64)>,
}

/// Backtest statistics
#[derive(Resource, Default, Debug)]
pub struct BacktestStats {
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub peak_value: f64,
}

impl BacktestStats {
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.winning_trades as f64 / self.total_trades as f64
        }
    }
}
