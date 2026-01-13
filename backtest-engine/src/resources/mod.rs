//! ECS Resources for the backtesting engine

use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// Configuration Structures (for TOML)
// ============================================================================

#[derive(Debug, Deserialize, Clone)]
pub struct ConfigFile {
    pub backtest: BacktestSection,
    pub stop_loss: StopLossSection,
    pub take_profit: TakeProfitSection,
    pub weak_performance: WeakPerformanceSection,
    pub trailing_stop: TrailingStopSection,
    pub costs: CostsSection,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BacktestSection {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub max_daily_buys: usize,
    pub position_size_pct: f64,
    pub max_hold_days: i32,
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StopLossSection {
    pub enabled: bool,
    pub pct: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TakeProfitSection {
    pub tp1_pct: f64,
    pub tp2_pct: f64,
    pub sell_ratio: f64,
    pub sell_on_break_wl: bool,
    pub sell_on_break_yl: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WeakPerformanceSection {
    pub enabled: bool,
    pub days: i32,
    pub min_gain_pct: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TrailingStopSection {
    pub enabled: bool,
    pub activation_pct: f64,
    pub trailing_pct: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CostsSection {
    pub commission_rate: f64,
    pub stamp_duty_rate: f64,
    pub slippage_pct: f64,
}

impl ConfigFile {
    /// Load config from TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse config: {}", e))
    }
}

// ============================================================================
// Backtest Configuration (ECS Resource)
// ============================================================================

/// Backtest Configuration (flattened for easy access in systems)
#[derive(Resource, Debug, Clone)]
pub struct BacktestConfig {
    // Basic
    pub initial_capital: f64,
    pub max_positions: usize,
    pub max_daily_buys: usize,
    pub position_size_pct: f64,
    pub max_hold_days: i32,
    
    // Date Range (optional)
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    
    // Stop Loss
    pub stop_loss_enabled: bool,
    pub stop_loss_pct: f64,
    
    // Take Profit
    pub tp1_pct: f64,
    pub tp2_pct: f64,
    pub tp_sell_ratio: f64,
    pub sell_on_break_wl: bool,
    pub sell_on_break_yl: bool,
    
    // Weak Performance
    pub weak_enabled: bool,
    pub weak_days: i32,
    pub weak_min_gain_pct: f64,
    
    // Trailing Stop
    pub trailing_enabled: bool,
    pub trailing_activation_pct: f64,
    pub trailing_pct: f64,
    
    // Costs
    pub commission_rate: f64,
    pub stamp_duty_rate: f64,
    pub slippage_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            max_positions: 5,
            max_daily_buys: 5,
            position_size_pct: 0.2,
            max_hold_days: 30,
            
            start_date: None,
            end_date: None,
            
            stop_loss_enabled: true,
            stop_loss_pct: 0.03,
            
            tp1_pct: 0.15,
            tp2_pct: 0.30,
            tp_sell_ratio: 0.333,
            sell_on_break_wl: true,
            sell_on_break_yl: false,
            
            weak_enabled: true,
            weak_days: 10,
            weak_min_gain_pct: 0.05,
            
            trailing_enabled: true,
            trailing_activation_pct: 0.10,
            trailing_pct: 0.05,
            
            commission_rate: 0.00025,
            stamp_duty_rate: 0.001,
            slippage_pct: 0.001,
        }
    }
}

impl From<ConfigFile> for BacktestConfig {
    fn from(cfg: ConfigFile) -> Self {
        // 解析日期字符串
        let parse_date = |s: &Option<String>| -> Option<NaiveDate> {
            s.as_ref()
                .filter(|d| !d.is_empty())
                .and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
        };
        
        Self {
            initial_capital: cfg.backtest.initial_capital,
            max_positions: cfg.backtest.max_positions,
            max_daily_buys: cfg.backtest.max_daily_buys,
            position_size_pct: cfg.backtest.position_size_pct,
            max_hold_days: cfg.backtest.max_hold_days,
            
            start_date: parse_date(&cfg.backtest.start_date),
            end_date: parse_date(&cfg.backtest.end_date),
            
            stop_loss_enabled: cfg.stop_loss.enabled,
            stop_loss_pct: cfg.stop_loss.pct,
            
            tp1_pct: cfg.take_profit.tp1_pct,
            tp2_pct: cfg.take_profit.tp2_pct,
            tp_sell_ratio: cfg.take_profit.sell_ratio,
            sell_on_break_wl: cfg.take_profit.sell_on_break_wl,
            sell_on_break_yl: cfg.take_profit.sell_on_break_yl,
            
            weak_enabled: cfg.weak_performance.enabled,
            weak_days: cfg.weak_performance.days,
            weak_min_gain_pct: cfg.weak_performance.min_gain_pct,
            
            trailing_enabled: cfg.trailing_stop.enabled,
            trailing_activation_pct: cfg.trailing_stop.activation_pct,
            trailing_pct: cfg.trailing_stop.trailing_pct,
            
            commission_rate: cfg.costs.commission_rate,
            stamp_duty_rate: cfg.costs.stamp_duty_rate,
            slippage_pct: cfg.costs.slippage_pct,
        }
    }
}

impl BacktestConfig {
    /// Calculate buy cost (commission + slippage)
    pub fn buy_cost(&self, amount: f64) -> f64 {
        amount * (self.commission_rate + self.slippage_pct)
    }
    
    /// Calculate sell cost (commission + stamp duty + slippage)
    pub fn sell_cost(&self, amount: f64) -> f64 {
        amount * (self.commission_rate + self.stamp_duty_rate + self.slippage_pct)
    }
}

// ============================================================================
// Other Resources
// ============================================================================

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
    pub total_commission: f64,
    pub total_stamp_duty: f64,
    pub total_slippage: f64,
}

impl BacktestStats {
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.winning_trades as f64 / self.total_trades as f64
        }
    }
    
    pub fn total_costs(&self) -> f64 {
        self.total_commission + self.total_stamp_duty + self.total_slippage
    }
}
