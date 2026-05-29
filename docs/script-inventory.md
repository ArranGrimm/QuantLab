# 脚本状态清单

这份清单是为第二阶段清理服务的，不是“推荐脚本索引”。每个脚本只标一个 operational status。

## 状态定义

- `canonical`: 明确推荐的直接入口。
- `implementation`: 被 canonical workflow 或当前 runner 调用，用户一般不直接运行。
- `diagnostic`: 可复现实验诊断，通常绑定稳定报告。
- `historical`: 用于追溯历史路线，不属于当前 workflow。
- `deprecated`: 第二阶段删除或迁移候选。

## Canonical

- `scripts/qlab.py` - QuantLab 日常 CLI，用于 status、export、backtest、compare、attribution。

## Implementation

- `scripts/amv_bull_pool_export_signals.py` - 当前静态 sleeve 共享的 AMV feature export 基础实现。
- `scripts/backtest_trade_attribution.py` - 通用 `bt-amv-topn` 交易归因实现。
- `scripts/amv_p3_raw_vs_adjusted_trade_attribution.py` - P3 adjusted-vs-raw canonical 归因实现。
- `scripts/amv_raw_execution_ground_truth_summary.py` - raw execution 状态报告生成脚本，为 `qlab status` 提供输入。

## Diagnostic

- `scripts/amv_allocation_diagnostic.py` - 组合 allocation 诊断；做组合权重决策前仍需要。
- `scripts/amv_annual_restart_cadence.py` - annual restart cadence 敏感性分析。
- `scripts/amv_close_to_close_diagnostic_signal_export.py` - close-to-close 诊断信号导出，明确不可当可交易口径。
- `scripts/amv_cohort_diagnostic_backtest.py` - executable-aware cohort 诊断 runner。
- `scripts/amv_exit_logic_whatif.py` - exit logic what-if 诊断。
- `scripts/amv_limit_ecology_diagnostic.py` - 第三阶段涨停生态特征诊断。
- `scripts/amv_limit_ecology_drawdown_attribution.py` - 首板后回踩 MaxDD 归因。
- `scripts/amv_limit_first_board_medium128_diagnostic.py` - limit first-board sleeve 的 medium128 复核。
- `scripts/amv_limit_first_board_variant_summary.py` - first-board variant Rust 汇总。
- `scripts/amv_limit_first_board_weak_window_diagnostic.py` - first-board sleeve 弱窗口诊断。
- `scripts/amv_liquidity_trend_refinement_diagnostic.py` - 趋势 / 流动性细分诊断。
- `scripts/amv_market_sentiment_diagnostic.py` - 市场情绪诊断。
- `scripts/amv_medium_trend_quality_diagnostic.py` - medium128 / 趋势质量诊断。
- `scripts/amv_pb3_gating_robustness.py` - PB3 gating 稳健性诊断。
- `scripts/amv_regime_phase_diagnostic.py` - AMV regime phase 诊断 helper。
- `scripts/amv_sector_breadth_diagnostic.py` - 行业 breadth 诊断。
- `scripts/amv_sector_tailwind_cadence.py` - sector tailwind cadence 敏感性 runner。
- `scripts/amv_sector_tailwind_diagnostic.py` - sector tailwind 诊断。
- `scripts/amv_signal_cohort_stats.py` - signal cohort 统计 helper。
- `scripts/amv_topn_segment_analysis.py` - AMV TopN 分段诊断。
- `scripts/amv_topn_trade_analysis.py` - AMV TopN 交易诊断。

## Historical

- `scripts/amv_attack_ok_lab.py` - 较早的 attack/OK 探索路线。
- `scripts/amv_bull_pool_combo_grid.py` - 当前 canonical export 前的早期 AMV combo grid。
- `scripts/amv_bull_pool_factor_regime_analysis.py` - 历史 factor/regime 分析。
- `scripts/amv_bull_pool_horizon_curve.py` - 历史 horizon curve 分析。
- `scripts/amv_bull_pool_listwise_ranker_lab.py` - 历史 listwise ranker lab。
- `scripts/amv_bull_pool_random_baseline.py` - 历史 random baseline 检查。
- `scripts/amv_bull_pool_ranker_lab.py` - 历史 ranker lab。
- `scripts/amv_bull_pool_regime_sleeve_lab.py` - 历史 regime sleeve lab。
- `scripts/amv_bull_pool_yearly_factor_analysis.py` - 历史 yearly factor 分析。
- `scripts/amv_constrained_oracle_lab.py` - 历史 oracle 上限 lab。
- `scripts/amv_executable_factor_scan.py` - 历史 executable-aware broad factor scan。
- `scripts/amv_executable_pullback_grid.py` - 历史 pullback grid scan。
- `scripts/amv_executable_rsrs_scan.py` - 历史 RSRS scan。
- `scripts/amv_executable_trend_filter_grid.py` - 历史 trend filter grid。
- `scripts/amv_executable_weight_grid.py` - 历史 executable-aware weight grid。
- `scripts/amv_horizon_aware_oracle_lab.py` - 历史 horizon-aware oracle lab。
- `scripts/amv_horizon_oracle_explainability.py` - 历史 oracle explainability lab。
- `scripts/amv_ltr_selection_analysis.py` - 历史 learning-to-rank selection 分析。
- `scripts/amv_limit_ecology_signal_export.py` - 历史涨停生态变体 export；当前 `limit-weakgate` 已由 `qlab export limit-weakgate` native workflow 复现。
- `scripts/amv_market_sentiment_signal_export.py` - 历史 sentiment export 候选，不是当前 CLI 入口。
- `scripts/amv_medium_trend_quality_signal_export.py` - 历史 medium/trend export 候选；当前 workflow 优先用 context combo export。
- `scripts/amv_oracle_sleeve_signal_export.py` - 历史 oracle sleeve export。
- `scripts/amv_sector_tailwind_signal_export.py` - 历史 standalone sector export；当前优先用 combined context export。
- `scripts/amv_yearly_weight_grid.py` - 历史 yearly weight grid。
- `scripts/b1_executable_base_lab.py` - 历史 B1 executable lab，仍被部分诊断 import。
- `scripts/b3_candidate_ranking_lab.py` - 历史 B3 candidate ranking lab。
- `scripts/b3_tdx_signal_export.py` - 历史 B3 TDX export。

## Deprecated

- `scripts/b1_backtest.py` - 当前 AMV workflow 之外的旧策略专用 backtest 入口。
- `scripts/rotation_backtest.py` - 当前 AMV workflow 之外的旧 rotation 入口。
- `scripts/amv_static_sleeve_signal_export.py` - Ref/P3/context/PB3 已由 `qlab export` native workflow 复现 raw ground truth；剩余用途主要是历史 sleeve export / 对照迁移候选。

## Removed

- `scripts/amv_topn_backtest.py` - 已由 `uv run python scripts/qlab.py backtest <signal_dir> --preset <preset>` 替代。
- `scripts/amv_limit_refill_rolling_nav.py` - Python rolling NAV 上限诊断，结论已归档且不是可交易口径。
- `scripts/amv_ltr_signal_export.py` - LTR signal export 路线已降级，不属于当前 workflow。
- `scripts/amv_topn_enhancement_sweep.py` - 较早 enhancement sweep，当前由 raw execution 报告和 `qlab.py` 工作流替代。
- `scripts/amv_context_combo_signal_export.py` - 已由 `qlab export context` native workflow 复现 raw ground truth 后删除。

## 第一阶段规则

不要仅凭这份清单删除脚本。删除前必须满足 `docs/cleanup-plan.md` 的门禁。
