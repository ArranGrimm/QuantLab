//! B3 strategy ECS Components

use bevy_ecs::prelude::*;
use chrono::NaiveDate;

#[derive(Component, Debug, Clone)]
pub struct Position {
    pub code: String,
    pub entry_date: NaiveDate,
    pub entry_price: f64,
    pub structural_stop_price: f64,
    pub shares: u32,
    pub initial_shares: u32,
    pub cost: f64,
    pub high_since_entry: f64,
    pub hold_trading_days: u32,
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

#[allow(dead_code)]
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
    pub hold_trading_days: u32,
    pub exit_reason: ExitReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    StructuralStop,
    BreakWhiteLine,
    FastFail,
    WeakPerformance,
    TrailingStop,
    MaxHoldDays,
    EndOfBacktest,
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::StructuralStop => write!(f, "StructuralStop"),
            ExitReason::BreakWhiteLine => write!(f, "BreakWhiteLine"),
            ExitReason::FastFail => write!(f, "FastFail"),
            ExitReason::WeakPerformance => write!(f, "Weak"),
            ExitReason::TrailingStop => write!(f, "TrailingStop"),
            ExitReason::MaxHoldDays => write!(f, "MaxHoldDays"),
            ExitReason::EndOfBacktest => write!(f, "EndOfBacktest"),
        }
    }
}
