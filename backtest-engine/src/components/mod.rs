//! ECS Components for the backtesting engine

use bevy_ecs::prelude::*;
use chrono::NaiveDate;

/// Position Component - currently held stock
#[derive(Component, Debug, Clone)]
pub struct Position {
    pub code: String,
    pub entry_date: NaiveDate,
    pub entry_price: f64,
    pub stop_price: f64,
    pub shares: u32,           // 当前持有股数
    pub initial_shares: u32,   // 初始股数 (用于计算 1/3)
    pub cost: f64,             // 当前持仓成本
    pub realized_pnl: f64,     // 已实现盈亏 (分批止盈累计)
    pub take_profit_stage: u8, // 止盈阶段: 0=未止盈, 1=已止盈15%, 2=已止盈30%
}

/// Closed Trade Record
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

/// Exit Reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    StopLoss,
    BreakWL,
    BreakYL,
    MaxHoldDays,
    EndOfBacktest,
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::StopLoss => write!(f, "StopLoss"),
            ExitReason::BreakWL => write!(f, "BreakWL"),
            ExitReason::BreakYL => write!(f, "BreakYL"),
            ExitReason::MaxHoldDays => write!(f, "MaxHoldDays"),
            ExitReason::EndOfBacktest => write!(f, "EndOfBacktest"),
        }
    }
}
