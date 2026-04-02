//! bt-core: Shared types for the QuantLab backtesting engine
//!
//! Provides strategy-agnostic building blocks:
//! - Portfolio (cash management)
//! - BacktestStats (PnL, drawdown, win rate, cost tracking)
//! - CostModel (A-share commission / stamp duty / slippage)
//! - Utility functions (date parsing, lot sizing, result printing)

use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// A-share lot size (每手 300 股，科创板/北交所除外)
pub const LOT_SIZE: u32 = 300;

// ============================================================================
// Portfolio
// ============================================================================

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

// ============================================================================
// Cost Model
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct CostModel {
    pub commission_rate: f64,
    pub stamp_duty_rate: f64,
    pub slippage_pct: f64,
}

impl CostModel {
    /// 买入成本 (佣金 + 滑点，无印花税)
    pub fn buy_cost(&self, amount: f64) -> f64 {
        amount * (self.commission_rate + self.slippage_pct)
    }

    /// 卖出成本 (佣金 + 印花税 + 滑点)
    pub fn sell_cost(&self, amount: f64) -> f64 {
        amount * (self.commission_rate + self.stamp_duty_rate + self.slippage_pct)
    }

    /// 卖出净收入
    pub fn sell_net(&self, gross: f64) -> f64 {
        gross - self.sell_cost(gross)
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            commission_rate: 0.00025,
            stamp_duty_rate: 0.001,
            slippage_pct: 0.001,
        }
    }
}

// ============================================================================
// Backtest Stats
// ============================================================================

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

#[derive(Debug, Clone, Serialize)]
pub struct BacktestDerivedMetrics {
    pub total_return_pct: f64,
    pub gross_pnl: f64,
    pub gross_return_pct: f64,
    pub avg_trades_per_day: f64,
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

    /// 记录一笔完整交易 (平仓时调用)
    pub fn record_trade(&mut self, pnl: f64, commission: f64, stamp_duty: f64, slippage: f64) {
        self.total_trades += 1;
        self.total_pnl += pnl;
        self.total_commission += commission;
        self.total_stamp_duty += stamp_duty;
        self.total_slippage += slippage;
        if pnl > 0.0 {
            self.winning_trades += 1;
        } else {
            self.losing_trades += 1;
        }
    }

    /// 仅记录成本 (买入时 / 分批止盈时)
    pub fn record_costs(&mut self, commission: f64, stamp_duty: f64, slippage: f64) {
        self.total_commission += commission;
        self.total_stamp_duty += stamp_duty;
        self.total_slippage += slippage;
    }

    /// 更新最大回撤
    pub fn update_drawdown(&mut self, total_value: f64) {
        if total_value > self.peak_value {
            self.peak_value = total_value;
        }
        if self.peak_value > 0.0 {
            let drawdown = (self.peak_value - total_value) / self.peak_value;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SignalArtifactMeta {
    pub strategy: Option<String>,
    pub train_run_id: Option<String>,
    pub signal_run_id: Option<String>,
    pub export_token: Option<String>,
    pub label: Option<String>,
    pub model_name: Option<String>,
    pub feature_mode: Option<String>,
    pub feature_hash: Option<String>,
    pub feature_count: Option<usize>,
    pub export_ema_alpha: Option<f64>,
    pub registry_path: Option<String>,
    pub canonical_signal_path: Option<String>,
    pub latest_alias_path: Option<String>,
    pub train_meta_path: Option<String>,
    pub signal_meta_path: Option<String>,
    pub git_commit: Option<String>,
    pub notebook: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ReportBundlePaths {
    pub txt_path: std::path::PathBuf,
    pub json_path: std::path::PathBuf,
}

// ============================================================================
// A-Share Price Limits (涨跌停)
// ============================================================================

/// 根据股票代码判断涨跌幅限制
/// 主板 (60/00) → 10%, 创业板 (300/301) → 20%, 科创板 (688/689) → 20%
pub fn price_limit_pct(code: &str) -> f64 {
    if code.starts_with("300")
        || code.starts_with("301")
        || code.starts_with("688")
        || code.starts_with("689")
    {
        0.20
    } else {
        0.10
    }
}

/// 涨停容差: 复权价四舍五入精度误差, 0.1% 足以覆盖绝大多数股价
const LIMIT_TOLERANCE: f64 = 0.001;

/// 判断是否涨停 (无法买入)
pub fn is_limit_up(close: f64, pre_close: f64, limit_pct: f64) -> bool {
    if pre_close <= 0.0 {
        return false;
    }
    close / pre_close - 1.0 >= limit_pct - LIMIT_TOLERANCE
}

/// 判断是否跌停 (无法卖出)
pub fn is_limit_down(close: f64, pre_close: f64, limit_pct: f64) -> bool {
    if pre_close <= 0.0 {
        return false;
    }
    close / pre_close - 1.0 <= -(limit_pct - LIMIT_TOLERANCE)
}

// ============================================================================
// Utilities
// ============================================================================

/// Parse optional date string "YYYY-MM-DD" → NaiveDate
pub fn parse_date_opt(s: &Option<String>) -> Option<NaiveDate> {
    s.as_ref()
        .filter(|d| !d.is_empty())
        .and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
}

/// Convert Polars date value (days since 1970-01-01) → NaiveDate
pub fn epoch_days_to_date(days: i32) -> Option<NaiveDate> {
    NaiveDate::from_num_days_from_ce_opt(days + 719163)
}

/// Round down to A-share lot size
pub fn round_to_lot(shares_f64: f64) -> u32 {
    ((shares_f64 / LOT_SIZE as f64).floor() as u32) * LOT_SIZE
}

/// Format backtest results as text (strategy-agnostic)
pub fn format_results(stats: &BacktestStats, portfolio: &Portfolio, trading_days: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();

    let total_return = (portfolio.cash / portfolio.initial_capital - 1.0) * 100.0;
    let gross_pnl = stats.total_pnl + stats.total_costs();
    let gross_return = gross_pnl / portfolio.initial_capital * 100.0;

    writeln!(s, "--- Results ---").unwrap();
    writeln!(s, "Total Trades:     {}", stats.total_trades).unwrap();
    writeln!(s, "Win Rate:         {:.1}%", stats.win_rate() * 100.0).unwrap();
    writeln!(s, "Total PnL:        {:+.2}", stats.total_pnl).unwrap();
    writeln!(s, "Final Portfolio:   {:.2}", portfolio.cash).unwrap();
    writeln!(s, "Total Return:     {:+.2}%", total_return).unwrap();
    writeln!(s, "Max Drawdown:     {:.2}%", stats.max_drawdown * 100.0).unwrap();
    writeln!(s).unwrap();
    writeln!(s, "--- Trading Costs ---").unwrap();
    writeln!(s, "Commission:       {:.2}", stats.total_commission).unwrap();
    writeln!(s, "Stamp Duty:       {:.2}", stats.total_stamp_duty).unwrap();
    writeln!(s, "Slippage:         {:.2}", stats.total_slippage).unwrap();
    writeln!(s, "Total Costs:      {:.2}", stats.total_costs()).unwrap();
    writeln!(s).unwrap();
    writeln!(s, "--- Derived ---").unwrap();
    writeln!(s, "Gross PnL:        {:+.2}", gross_pnl).unwrap();
    writeln!(s, "Gross Return:     {:+.2}%", gross_return).unwrap();
    writeln!(s, "Avg Trades/Day:   {:.1}", stats.total_trades as f64 / trading_days.max(1) as f64).unwrap();

    s
}

pub fn calc_derived_metrics(
    stats: &BacktestStats,
    portfolio: &Portfolio,
    trading_days: usize,
) -> BacktestDerivedMetrics {
    let total_return_pct = (portfolio.cash / portfolio.initial_capital - 1.0) * 100.0;
    let gross_pnl = stats.total_pnl + stats.total_costs();
    let gross_return_pct = gross_pnl / portfolio.initial_capital * 100.0;
    let avg_trades_per_day = stats.total_trades as f64 / trading_days.max(1) as f64;
    BacktestDerivedMetrics {
        total_return_pct,
        gross_pnl,
        gross_return_pct,
        avg_trades_per_day,
    }
}

pub fn build_metrics_json(
    stats: &BacktestStats,
    portfolio: &Portfolio,
    trading_days: usize,
) -> serde_json::Value {
    let derived = calc_derived_metrics(stats, portfolio, trading_days);
    serde_json::json!({
        "total_trades": stats.total_trades,
        "win_rate_pct": stats.win_rate() * 100.0,
        "total_pnl": stats.total_pnl,
        "final_portfolio": portfolio.cash,
        "total_return_pct": derived.total_return_pct,
        "max_drawdown_pct": stats.max_drawdown * 100.0,
        "gross_pnl": derived.gross_pnl,
        "gross_return_pct": derived.gross_return_pct,
        "total_commission": stats.total_commission,
        "total_stamp_duty": stats.total_stamp_duty,
        "total_slippage": stats.total_slippage,
        "total_costs": stats.total_costs(),
        "avg_trades_per_day": derived.avg_trades_per_day,
        "trading_days": trading_days,
    })
}

/// Print backtest results summary to stdout
pub fn print_results(stats: &BacktestStats, portfolio: &Portfolio) {
    println!("\n========================================");
    println!("           Backtest Results");
    println!("========================================");
    println!("Total Trades: {}", stats.total_trades);
    println!("Win Rate: {:.1}%", stats.win_rate() * 100.0);
    println!("Total PnL: {:+.2}", stats.total_pnl);
    println!("Final Portfolio: {:.2}", portfolio.cash);
    println!(
        "Total Return: {:+.2}%",
        (portfolio.cash / portfolio.initial_capital - 1.0) * 100.0
    );
    println!("Max Drawdown: {:.2}%", stats.max_drawdown * 100.0);
    println!("----------------------------------------");
    println!("Trading Costs:");
    println!("  Commission: {:.2}", stats.total_commission);
    println!("  Stamp Duty: {:.2}", stats.total_stamp_duty);
    println!("  Slippage:   {:.2}", stats.total_slippage);
    println!("  Total:      {:.2}", stats.total_costs());
    println!("========================================");
}

/// Save a backtest report file. Each strategy provides its own config text.
pub fn write_report(
    strategy_name: &str,
    config_text: &str,
    stats: &BacktestStats,
    portfolio: &Portfolio,
    trading_days: usize,
    output_dir: &str,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    use chrono::Local;
    use std::io::Write;

    std::fs::create_dir_all(output_dir)?;

    let now = Local::now();
    let filename = format!("{}_{}.txt", strategy_name, now.format("%Y%m%d_%H%M%S"));
    let filepath = std::path::Path::new(output_dir).join(&filename);

    let mut f = std::fs::File::create(&filepath)?;

    writeln!(f, "========================================")?;
    writeln!(f, "   {} Backtest Report", strategy_name)?;
    writeln!(f, "   {}", now.format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(f, "========================================")?;
    writeln!(f)?;
    write!(f, "{}", config_text)?;
    writeln!(f)?;
    write!(f, "{}", format_results(stats, portfolio, trading_days))?;
    writeln!(f, "========================================")?;

    println!("\n📄 Report saved: {}", filepath.display());
    Ok(filepath)
}

pub fn load_signal_meta(data_path: &std::path::Path) -> Option<SignalArtifactMeta> {
    let meta_path = data_path.with_extension("meta.json");
    if !meta_path.exists() {
        return None;
    }
    let raw = std::fs::read_to_string(&meta_path).ok()?;
    let mut meta: SignalArtifactMeta = serde_json::from_str(&raw).ok()?;
    if meta.signal_meta_path.is_none() {
        meta.signal_meta_path = Some(meta_path.to_string_lossy().to_string());
    }
    Some(meta)
}

pub fn format_float_token(value: f64, precision: usize) -> String {
    let mut s = format!("{value:.precision$}");
    while s.contains('.') && s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s.is_empty() {
        s.push('0');
    }
    s.replace('-', "m").replace('.', "p")
}

pub fn build_report_stem(
    strategy_name: &str,
    signal_meta: Option<&SignalArtifactMeta>,
    suffix: &str,
) -> String {
    let signal_id = signal_meta
        .and_then(|meta| meta.signal_run_id.as_deref())
        .unwrap_or(strategy_name);
    if suffix.trim().is_empty() {
        format!("{}__{}", strategy_name, signal_id)
    } else {
        format!("{}__{}__{}", strategy_name, signal_id, suffix)
    }
}

pub fn resolve_registry_path(
    signal_meta: Option<&SignalArtifactMeta>,
    fallback_registry_path: &str,
) -> std::path::PathBuf {
    signal_meta
        .and_then(|meta| meta.registry_path.as_ref())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(fallback_registry_path))
}

pub fn append_jsonl_record(
    path: &std::path::Path,
    record: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::OpenOptions;
    use std::io::Write;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(record)?)?;
    Ok(())
}

pub fn write_report_bundle(
    output_dir: &str,
    strategy_name: &str,
    report_stem: &str,
    data_path: &std::path::Path,
    signal_meta: Option<&SignalArtifactMeta>,
    config_text: &str,
    backtest_config_json: serde_json::Value,
    extra_text: Option<&str>,
    extra_json: Option<serde_json::Value>,
    stats: &BacktestStats,
    portfolio: &Portfolio,
    trading_days: usize,
) -> Result<ReportBundlePaths, Box<dyn std::error::Error>> {
    use chrono::Local;
    use std::io::Write;

    std::fs::create_dir_all(output_dir)?;

    let now = Local::now();
    let stem = format!("{}__{}", report_stem, now.format("%Y%m%d_%H%M%S"));
    let txt_path = std::path::Path::new(output_dir).join(format!("{}.txt", stem));
    let json_path = std::path::Path::new(output_dir).join(format!("{}.json", stem));

    let mut f = std::fs::File::create(&txt_path)?;
    writeln!(f, "========================================")?;
    writeln!(f, "   {} Backtest Report", strategy_name)?;
    writeln!(f, "   {}", now.format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(f, "========================================")?;
    writeln!(f)?;
    if let Some(meta) = signal_meta {
        writeln!(f, "--- Signal Artifact ---")?;
        if let Some(signal_run_id) = &meta.signal_run_id {
            writeln!(f, "Signal Run ID:    {}", signal_run_id)?;
        }
        if let Some(label) = &meta.label {
            writeln!(f, "Label:            {}", label)?;
        }
        if let Some(model_name) = &meta.model_name {
            writeln!(f, "Model:            {}", model_name)?;
        }
        if let Some(feature_mode) = &meta.feature_mode {
            writeln!(f, "Feature Mode:     {}", feature_mode)?;
        }
        if let Some(feature_count) = meta.feature_count {
            writeln!(f, "Feature Count:    {}", feature_count)?;
        }
        if let Some(export_ema_alpha) = meta.export_ema_alpha {
            writeln!(f, "Export EMA:       {}", export_ema_alpha)?;
        }
        writeln!(f, "Signal File:      {}", data_path.display())?;
        writeln!(f)?;
    }
    if let Some(extra_text) = extra_text {
        write!(f, "{}", extra_text)?;
        if !extra_text.ends_with('\n') {
            writeln!(f)?;
        }
        writeln!(f)?;
    }
    write!(f, "{}", config_text)?;
    writeln!(f)?;
    write!(f, "{}", format_results(stats, portfolio, trading_days))?;
    writeln!(f, "========================================")?;

    let signal_json = match signal_meta {
        Some(meta) => serde_json::to_value(meta)?,
        None => serde_json::Value::Null,
    };
    let report_json = serde_json::json!({
        "report_version": 1,
        "strategy": strategy_name,
        "generated_at": now.format("%Y-%m-%d %H:%M:%S").to_string(),
        "signal_file": data_path.to_string_lossy().to_string(),
        "signal": signal_json,
        "backtest_config": backtest_config_json,
        "metrics": build_metrics_json(stats, portfolio, trading_days),
        "extra": extra_json.unwrap_or(serde_json::Value::Null),
        "report_txt_path": txt_path.to_string_lossy().to_string(),
        "report_json_path": json_path.to_string_lossy().to_string(),
    });

    std::fs::write(&json_path, serde_json::to_string_pretty(&report_json)?)?;

    println!("\n📄 Report saved: {}", txt_path.display());
    println!("🧾 Report JSON saved: {}", json_path.display());
    Ok(ReportBundlePaths { txt_path, json_path })
}
