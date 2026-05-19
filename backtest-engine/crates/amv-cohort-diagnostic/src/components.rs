//! AMV close-to-close cohort diagnostic ECS components.

use bevy_ecs::prelude::*;
use chrono::NaiveDate;

#[derive(Component, Debug, Clone)]
pub struct Position {
    pub code: String,
    pub entry_date: NaiveDate,
    pub entry_trade_index: i32,
    pub entry_price: f64,
    pub shares: u32,
    pub cost: f64,
}

#[allow(dead_code)]
#[derive(Component, Debug, Clone)]
pub struct ClosedTrade {
    pub code: String,
    pub entry_date: NaiveDate,
    pub exit_date: NaiveDate,
    pub entry_price: f64,
    pub exit_price: f64,
    pub shares: u32,
    pub cost: f64,
    pub exit_value: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub hold_trading_days: i32,
    pub exit_reason: ExitReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    MaxHoldDays,
    EndOfBacktest,
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::MaxHoldDays => write!(f, "MaxHoldDays"),
            ExitReason::EndOfBacktest => write!(f, "EndOfBacktest"),
        }
    }
}
