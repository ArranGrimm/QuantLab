//! B1 strategy ECS Components

use bevy_ecs::prelude::*;
use chrono::NaiveDate;

/// 持仓组件
#[derive(Component, Debug, Clone)]
pub struct Position {
    pub code: String,
    pub entry_date: NaiveDate,
    pub entry_price: f64,
    pub stop_price: f64,
    pub shares: u32,
    pub initial_shares: u32,
    pub cost: f64,
    pub realized_pnl: f64,
    pub take_profit_stage: u8, // 0=未止盈, 1=已TP1, 2=已TP2
    pub high_since_entry: f64,
    pub trailing_stop_active: bool,
}

impl Position {
    pub fn update_high(&mut self, current_high: f64) {
        if current_high > self.high_since_entry {
            self.high_since_entry = current_high;
        }
    }

    pub fn trailing_stop_price(&self, trailing_pct: f64) -> f64 {
        self.high_since_entry * (1.0 - trailing_pct)
    }
}

/// 已平仓记录
#[derive(Component, Debug, Clone)]
pub struct ClosedTrade {
    pub code: String,
    pub entry_date: NaiveDate,
    pub exit_date: NaiveDate,
    pub entry_price: f64,
    pub exit_price: f64,
    pub shares: u32,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub hold_days: i32,
    pub exit_reason: ExitReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    StopLoss,
    TrailingStop,
    BreakWL,
    BreakYL,
    MaxHoldDays,
    WeakPerformance,
    TakeProfit1,
    TakeProfit2,
    EndOfBacktest,
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::StopLoss => write!(f, "StopLoss"),
            ExitReason::TrailingStop => write!(f, "TrailingStop"),
            ExitReason::BreakWL => write!(f, "BreakWL"),
            ExitReason::BreakYL => write!(f, "BreakYL"),
            ExitReason::MaxHoldDays => write!(f, "MaxHoldDays"),
            ExitReason::WeakPerformance => write!(f, "Weak"),
            ExitReason::TakeProfit1 => write!(f, "TP1"),
            ExitReason::TakeProfit2 => write!(f, "TP2"),
            ExitReason::EndOfBacktest => write!(f, "EndOfBacktest"),
        }
    }
}
