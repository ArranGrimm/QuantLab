//! AMV TopN strategy ECS resources.

use bevy_ecs::prelude::*;
use bt_core::CostModel;
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct ConfigFile {
    pub backtest: BacktestSection,
    pub entry: EntrySection,
    #[serde(default)]
    pub exit: ExitSection,
    pub stop_loss: StopLossSection,
    #[serde(default)]
    pub early_stop: EarlyStopSection,
    pub trailing_stop: TrailingStopSection,
    pub costs: CostModel,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BacktestSection {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub max_daily_buys: usize,
    pub position_size_pct: f64,
    pub max_hold_trading_days: i32,
    #[serde(default)]
    pub allow_duplicate_positions: bool,
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default = "default_min_position_ratio")]
    pub min_position_ratio: f64,
}

fn default_min_position_ratio() -> f64 {
    0.5
}

#[derive(Debug, Deserialize, Clone)]
pub struct EntrySection {
    pub top_n: usize,
    #[serde(default)]
    pub entry_rank_limit: Option<usize>,
    #[serde(default)]
    pub min_score: f64,
    #[serde(default = "default_require_bull_regime")]
    pub require_bull_regime: bool,
    #[serde(default)]
    pub max_open_gap_pct: Option<f64>,
}

fn default_require_bull_regime() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ExitSection {
    #[serde(default)]
    pub sell_on_bear_regime: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StopLossSection {
    pub enabled: bool,
    pub pct: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EarlyStopSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_early_stop_trigger_hold_trading_days")]
    pub trigger_hold_trading_days: i32,
    #[serde(default = "default_early_stop_loss_pct")]
    pub loss_pct: f64,
    #[serde(default)]
    pub require_previous_close_below_entry: bool,
    #[serde(default)]
    pub reserve_slot_until_max_hold: bool,
}

impl Default for EarlyStopSection {
    fn default() -> Self {
        Self {
            enabled: false,
            trigger_hold_trading_days: default_early_stop_trigger_hold_trading_days(),
            loss_pct: default_early_stop_loss_pct(),
            require_previous_close_below_entry: false,
            reserve_slot_until_max_hold: false,
        }
    }
}

fn default_early_stop_trigger_hold_trading_days() -> i32 {
    2
}

fn default_early_stop_loss_pct() -> f64 {
    0.03
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
        toml::from_str(&content).map_err(|e| format!("Failed to parse config: {}", e))
    }
}

#[derive(Resource, Debug, Clone)]
pub struct AmvTopnConfig {
    pub initial_capital: f64,
    pub max_positions: usize,
    pub max_daily_buys: usize,
    pub position_size_pct: f64,
    pub max_hold_trading_days: i32,
    pub allow_duplicate_positions: bool,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub min_position_ratio: f64,

    pub top_n: usize,
    pub entry_rank_limit: usize,
    pub min_score: f64,
    pub require_bull_regime: bool,
    pub max_open_gap_pct: Option<f64>,

    pub sell_on_bear_regime: bool,

    pub stop_loss_enabled: bool,
    pub stop_loss_pct: f64,

    pub early_stop_enabled: bool,
    pub early_stop_trigger_hold_trading_days: i32,
    pub early_stop_loss_pct: f64,
    pub early_stop_require_previous_close_below_entry: bool,
    pub early_stop_reserve_slot_until_max_hold: bool,

    pub trailing_enabled: bool,
    pub trailing_activation_pct: f64,
    pub trailing_pct: f64,

    pub cost_model: CostModel,
}

impl Default for AmvTopnConfig {
    fn default() -> Self {
        Self {
            initial_capital: 500_000.0,
            max_positions: 3,
            max_daily_buys: 3,
            position_size_pct: 1.0 / 3.0,
            max_hold_trading_days: 10,
            allow_duplicate_positions: false,
            start_date: None,
            end_date: None,
            min_position_ratio: 0.5,
            top_n: 3,
            entry_rank_limit: 3,
            min_score: 0.0,
            require_bull_regime: true,
            max_open_gap_pct: None,
            sell_on_bear_regime: false,
            stop_loss_enabled: true,
            stop_loss_pct: 0.05,
            early_stop_enabled: false,
            early_stop_trigger_hold_trading_days: 2,
            early_stop_loss_pct: 0.03,
            early_stop_require_previous_close_below_entry: false,
            early_stop_reserve_slot_until_max_hold: false,
            trailing_enabled: false,
            trailing_activation_pct: 0.10,
            trailing_pct: 0.05,
            cost_model: CostModel::default(),
        }
    }
}

impl From<ConfigFile> for AmvTopnConfig {
    fn from(cfg: ConfigFile) -> Self {
        let max_daily_buys = cfg.backtest.max_daily_buys.min(cfg.backtest.max_positions);
        let entry_rank_limit = cfg
            .entry
            .entry_rank_limit
            .unwrap_or(cfg.entry.top_n)
            .min(cfg.entry.top_n);
        Self {
            initial_capital: cfg.backtest.initial_capital,
            max_positions: cfg.backtest.max_positions,
            max_daily_buys,
            position_size_pct: cfg.backtest.position_size_pct,
            max_hold_trading_days: cfg.backtest.max_hold_trading_days,
            allow_duplicate_positions: cfg.backtest.allow_duplicate_positions,
            start_date: bt_core::parse_date_opt(&cfg.backtest.start_date),
            end_date: bt_core::parse_date_opt(&cfg.backtest.end_date),
            min_position_ratio: cfg.backtest.min_position_ratio,
            top_n: cfg.entry.top_n,
            entry_rank_limit,
            min_score: cfg.entry.min_score,
            require_bull_regime: cfg.entry.require_bull_regime,
            max_open_gap_pct: cfg.entry.max_open_gap_pct,
            sell_on_bear_regime: cfg.exit.sell_on_bear_regime,
            stop_loss_enabled: cfg.stop_loss.enabled,
            stop_loss_pct: cfg.stop_loss.pct,
            early_stop_enabled: cfg.early_stop.enabled,
            early_stop_trigger_hold_trading_days: cfg.early_stop.trigger_hold_trading_days,
            early_stop_loss_pct: cfg.early_stop.loss_pct,
            early_stop_require_previous_close_below_entry: cfg
                .early_stop
                .require_previous_close_below_entry,
            early_stop_reserve_slot_until_max_hold: cfg.early_stop.reserve_slot_until_max_hold,
            trailing_enabled: cfg.trailing_stop.enabled,
            trailing_activation_pct: cfg.trailing_stop.activation_pct,
            trailing_pct: cfg.trailing_stop.trailing_pct,
            cost_model: cfg.costs,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PriceBar {
    pub open: f64,
    pub high: f64,
    pub close: f64,
    pub pre_close: f64,
    pub score: f64,
    pub rank: u32,
    pub is_signal: bool,
    pub is_bull_regime: bool,
}

#[derive(Resource, Default, Clone)]
pub struct MarketData {
    pub prices: HashMap<String, HashMap<NaiveDate, PriceBar>>,
    pub date_index: HashMap<NaiveDate, i32>,
    pub trading_dates: Vec<NaiveDate>,
}

#[derive(Resource, Default)]
pub struct DailyData {
    pub buy_candidates: Vec<(String, f64, f64)>,
}
