//! B3 strategy ECS Resources

use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct ConfigFile {
    pub backtest: BacktestSection,
    pub exit: ExitSection,
    pub trailing_stop: TrailingStopSection,
    pub costs: CostsSection,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BacktestSection {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub max_daily_buys: usize,
    pub position_size_pct: f64,
    pub max_hold_trading_days: u32,
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default = "default_entry_rank_limit")]
    pub entry_rank_limit: u32,
    #[serde(default)]
    pub min_score: f64,
    #[serde(default = "default_require_bull_regime")]
    pub require_bull_regime: bool,
    #[serde(default = "default_min_position_ratio")]
    pub min_position_ratio: f64,
    #[serde(default = "default_sort_field")]
    pub sort_field: String,
    #[serde(default = "default_sort_ascending")]
    pub sort_ascending: bool,
}

fn default_entry_rank_limit() -> u32 { 3 }
fn default_require_bull_regime() -> bool { true }
fn default_min_position_ratio() -> f64 { 0.5 }
fn default_sort_field() -> String { "score".to_string() }
fn default_sort_ascending() -> bool { false }

#[derive(Debug, Deserialize, Clone)]
pub struct ExitSection {
    pub structural_stop_enabled: bool,
    pub structural_stop_buffer_pct: f64,
    pub break_white_line_enabled: bool,
    pub break_white_line_buffer_pct: f64,
    pub fast_fail_enabled: bool,
    pub fast_fail_days: u32,
    pub fast_fail_loss_pct: f64,
    pub weak_enabled: bool,
    pub weak_days: u32,
    pub weak_min_gain_pct: f64,
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
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse config: {}", e))
    }
}

#[derive(Resource, Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub max_daily_buys: usize,
    pub position_size_pct: f64,
    pub max_hold_trading_days: u32,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub entry_rank_limit: u32,
    pub min_score: f64,
    pub require_bull_regime: bool,
    pub min_position_ratio: f64,
    pub sort_field: String,
    pub sort_ascending: bool,

    pub structural_stop_enabled: bool,
    pub structural_stop_buffer_pct: f64,
    pub break_white_line_enabled: bool,
    pub break_white_line_buffer_pct: f64,
    pub fast_fail_enabled: bool,
    pub fast_fail_days: u32,
    pub fast_fail_loss_pct: f64,
    pub weak_enabled: bool,
    pub weak_days: u32,
    pub weak_min_gain_pct: f64,

    pub trailing_enabled: bool,
    pub trailing_activation_pct: f64,
    pub trailing_pct: f64,

    pub commission_rate: f64,
    pub stamp_duty_rate: f64,
    pub slippage_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 500_000.0,
            max_positions: 3,
            max_daily_buys: 3,
            position_size_pct: 0.333333333333,
            max_hold_trading_days: 30,
            start_date: None,
            end_date: None,
            entry_rank_limit: 3,
            min_score: 0.0,
            require_bull_regime: true,
            min_position_ratio: 0.5,
            sort_field: "score".to_string(),
            sort_ascending: false,
            structural_stop_enabled: true,
            structural_stop_buffer_pct: 0.0,
            break_white_line_enabled: true,
            break_white_line_buffer_pct: 0.0,
            fast_fail_enabled: true,
            fast_fail_days: 3,
            fast_fail_loss_pct: 0.03,
            weak_enabled: true,
            weak_days: 10,
            weak_min_gain_pct: 0.02,
            trailing_enabled: true,
            trailing_activation_pct: 0.12,
            trailing_pct: 0.08,
            commission_rate: 0.00025,
            stamp_duty_rate: 0.001,
            slippage_pct: 0.001,
        }
    }
}

impl From<ConfigFile> for BacktestConfig {
    fn from(cfg: ConfigFile) -> Self {
        Self {
            initial_capital: cfg.backtest.initial_capital,
            max_positions: cfg.backtest.max_positions,
            max_daily_buys: cfg.backtest.max_daily_buys,
            position_size_pct: cfg.backtest.position_size_pct,
            max_hold_trading_days: cfg.backtest.max_hold_trading_days,
            start_date: bt_core::parse_date_opt(&cfg.backtest.start_date),
            end_date: bt_core::parse_date_opt(&cfg.backtest.end_date),
            entry_rank_limit: cfg.backtest.entry_rank_limit,
            min_score: cfg.backtest.min_score,
            require_bull_regime: cfg.backtest.require_bull_regime,
            min_position_ratio: cfg.backtest.min_position_ratio,
            sort_field: cfg.backtest.sort_field,
            sort_ascending: cfg.backtest.sort_ascending,
            structural_stop_enabled: cfg.exit.structural_stop_enabled,
            structural_stop_buffer_pct: cfg.exit.structural_stop_buffer_pct,
            break_white_line_enabled: cfg.exit.break_white_line_enabled,
            break_white_line_buffer_pct: cfg.exit.break_white_line_buffer_pct,
            fast_fail_enabled: cfg.exit.fast_fail_enabled,
            fast_fail_days: cfg.exit.fast_fail_days,
            fast_fail_loss_pct: cfg.exit.fast_fail_loss_pct,
            weak_enabled: cfg.exit.weak_enabled,
            weak_days: cfg.exit.weak_days,
            weak_min_gain_pct: cfg.exit.weak_min_gain_pct,
            trailing_enabled: cfg.trailing_stop.enabled,
            trailing_activation_pct: cfg.trailing_stop.activation_pct,
            trailing_pct: cfg.trailing_stop.trailing_pct,
            commission_rate: cfg.costs.commission_rate,
            stamp_duty_rate: cfg.costs.stamp_duty_rate,
            slippage_pct: cfg.costs.slippage_pct,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PriceBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub pre_close: f64,
    pub white_line: f64,
    pub yellow_line: f64,
    pub is_signal: bool,
    pub is_bull_regime: bool,
    pub score: f64,
    pub sort_value: f64,
    pub rank: u32,
    pub trigger_low: f64,
    pub trigger_high: f64,
}

#[derive(Resource, Default, Clone)]
pub struct MarketData {
    pub prices: HashMap<String, HashMap<NaiveDate, PriceBar>>,
}

/// 当日买入候选 (code, score, open_price, structural_stop_price)
#[derive(Resource, Default)]
pub struct DailyData {
    pub buy_candidates: Vec<(String, f64, f64, f64)>,
}
