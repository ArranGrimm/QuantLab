//! 截面轮动策略 ECS Resources

use bevy_ecs::prelude::*;
use bt_core::CostModel;
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// TOML Config Structures
// ============================================================================

#[derive(Debug, Deserialize, Clone)]
pub struct ConfigFile {
    pub backtest: BacktestSection,
    pub entry: EntrySection,
    pub stop_loss: StopLossSection,
    pub trailing_stop: TrailingStopSection,
    pub costs: CostModel,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BacktestSection {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub position_size_pct: f64,
    pub max_hold_days: i32,
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default = "default_min_position_ratio")]
    pub min_position_ratio: f64,
}

fn default_min_position_ratio() -> f64 { 0.5 }

#[derive(Debug, Deserialize, Clone)]
pub struct EntrySection {
    pub top_n: usize,
    pub hold_buffer: usize,
    #[serde(default)]
    pub min_score: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StopLossSection {
    pub enabled: bool,
    pub pct: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TrailingStopSection {
    pub enabled: bool,
    pub activation_pct: f64,
    pub trailing_pct: f64,
}

impl ConfigFile {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse config: {}", e))
    }
}

// ============================================================================
// Flattened Rotation Config (ECS Resource)
// ============================================================================

#[derive(Resource, Debug, Clone)]
pub struct RotationConfig {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub position_size_pct: f64,
    pub max_hold_days: i32,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub min_position_ratio: f64,

    pub top_n: usize,
    pub hold_buffer: usize,
    pub min_score: f64,

    pub stop_loss_enabled: bool,
    pub stop_loss_pct: f64,

    pub trailing_enabled: bool,
    pub trailing_activation_pct: f64,
    pub trailing_pct: f64,

    pub cost_model: CostModel,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            max_positions: 20,
            position_size_pct: 0.05,
            max_hold_days: 30,
            start_date: None,
            end_date: None,
            min_position_ratio: 0.5,
            top_n: 20,
            hold_buffer: 50,
            min_score: 0.0,
            stop_loss_enabled: true,
            stop_loss_pct: 0.05,
            trailing_enabled: false,
            trailing_activation_pct: 0.10,
            trailing_pct: 0.05,
            cost_model: CostModel::default(),
        }
    }
}

impl From<ConfigFile> for RotationConfig {
    fn from(cfg: ConfigFile) -> Self {
        Self {
            initial_capital: cfg.backtest.initial_capital,
            max_positions: cfg.backtest.max_positions,
            position_size_pct: cfg.backtest.position_size_pct,
            max_hold_days: cfg.backtest.max_hold_days,
            start_date: bt_core::parse_date_opt(&cfg.backtest.start_date),
            end_date: bt_core::parse_date_opt(&cfg.backtest.end_date),
            min_position_ratio: cfg.backtest.min_position_ratio,
            top_n: cfg.entry.top_n,
            hold_buffer: cfg.entry.hold_buffer,
            min_score: cfg.entry.min_score,
            stop_loss_enabled: cfg.stop_loss.enabled,
            stop_loss_pct: cfg.stop_loss.pct,
            trailing_enabled: cfg.trailing_stop.enabled,
            trailing_activation_pct: cfg.trailing_stop.activation_pct,
            trailing_pct: cfg.trailing_stop.trailing_pct,
            cost_model: cfg.costs,
        }
    }
}

// ============================================================================
// Market Data
// ============================================================================

#[derive(Debug, Clone)]
pub struct PriceBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub pre_close: f64,
    pub score: f64,
    pub rank: u32,
    pub is_top_n: bool,
}

/// code → date → PriceBar
#[derive(Resource, Default, Clone)]
pub struct MarketData {
    pub prices: HashMap<String, HashMap<NaiveDate, PriceBar>>,
}

/// 当日买入候选 (code, score, close_price)，按 score 降序排列
#[derive(Resource, Default)]
pub struct DailyData {
    pub candidates: Vec<(String, f64, f64)>,
}
