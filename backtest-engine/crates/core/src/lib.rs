//! bt-core: Shared types for the QuantLab backtesting engine
//!
//! Provides strategy-agnostic building blocks:
//! - Portfolio (cash management)
//! - BacktestStats (PnL, drawdown, win rate, cost tracking)
//! - CostModel (A-share commission / stamp duty / slippage)
//! - Utility functions (date parsing, lot sizing, result printing)

use bevy_ecs::prelude::*;
use chrono::NaiveDate;
use serde::Deserialize;

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
