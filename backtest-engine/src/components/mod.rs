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
    
    // 移动止损相关
    pub high_since_entry: f64,     // 买入后的最高价
    pub trailing_stop_active: bool, // 移动止损是否激活
}

impl Position {
    /// 更新最高价
    pub fn update_high(&mut self, current_high: f64) {
        if current_high > self.high_since_entry {
            self.high_since_entry = current_high;
        }
    }
    
    /// 计算移动止损价
    pub fn trailing_stop_price(&self, trailing_pct: f64) -> f64 {
        self.high_since_entry * (1.0 - trailing_pct)
    }
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
    TrailingStop,     // 移动止损
    BreakWL,
    BreakYL,
    MaxHoldDays,
    WeakPerformance,  // N 天内涨幅不足
    TakeProfit1,      // 第一阶段止盈
    TakeProfit2,      // 第二阶段止盈
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
