import { Callout, Divider, Grid, H1, H2, H3, Stack, Stat, Table, Text } from 'cursor/canvas';

const strategyRows = [
  ['LTR no_risk', '+2.523pp', '+3.275%', '0.045', '0.657'],
  ['LTR kbar_momentum_state', '+2.386pp', '+3.137%', '0.036', '0.507'],
  ['LTR core_state', '+1.880pp', '+2.631%', '0.036', '0.446'],
  ['Baseline KLEN', '+1.818pp', '+2.569%', '0.024', '0.138'],
  ['Baseline 5日动量', '+1.373pp', '+2.124%', '0.049', '0.310'],
  ['LTR full', '+1.153pp', '+1.904%', '0.040', '0.299'],
  ['Baseline 当前组合 P2/K0.5/R0', '+0.665pp', '+1.416%', '0.009', '-0.000'],
  ['LTR momentum_state', '+0.411pp', '+1.162%', '0.043', '0.249'],
];

const yearlyRows = [
  ['LTR no_risk', '-0.292pp', '+2.731pp', '+1.292pp', '+6.362pp'],
  ['LTR kbar_momentum_state', '-0.186pp', '+1.665pp', '+1.224pp', '+6.842pp'],
  ['LTR core_state', '+0.378pp', '+3.268pp', '-0.321pp', '+4.196pp'],
  ['Baseline KLEN', '+0.469pp', '+2.438pp', '+1.560pp', '+2.804pp'],
  ['LTR full', '+0.817pp', '+0.118pp', '+1.538pp', '+2.140pp'],
  ['LTR momentum_state', '+0.416pp', '+1.208pp', '+0.457pp', '-0.437pp'],
];

const featureRows = [
  ['no_risk', 'rank_ret_20d', '8430'],
  ['no_risk', 'rank_KLEN', '5449'],
  ['no_risk', 'rank_amount_ma20', '1351'],
  ['kbar_momentum_state', 'rank_ret_20d', '8331'],
  ['kbar_momentum_state', 'rank_KLEN', '4042'],
  ['kbar_momentum_state', 'rank_ret_5d', '2879'],
  ['full', 'rank_ret_20d', '6775'],
  ['full', 'rank_atr_14_pct', '5460'],
  ['full', 'rank_ret_5d', '1072'],
];

export default function AmvLtrVariantAblation() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>AMV Bull Pool LTR 特征消融</H1>
        <Text tone="secondary">
          Source: artifacts/amv_bull_pool_listwise_ranker/20260516_105316 · 2019-01-29 至 2026-04-27 · 6d/top3 · LightGBM lambdarank
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="+2.523pp" label="最佳平均 top3 edge" tone="success" />
        <Stat value="no_risk" label="最佳 variant" />
        <Stat value="+6.362pp" label="no_risk 2026 edge" tone="success" />
        <Stat value="827,388" label="训练样本行数" />
      </Grid>

      <Callout tone="success" title="关键变化">
        <Text>
          去掉 ATR 与 panic sell pressure 后, LTR 从 full 的 +1.153pp 提升到 +2.523pp, 已超过 KLEN 单因子。
          这说明 LTR 方向可继续推进, 但风险类特征当前更像噪声或错误约束。
        </Text>
      </Callout>

      <H2>策略平均表现</H2>
      <Table
        headers={['策略', '平均 edge', '平均收益', 'Precision@3', 'Hit15 edge']}
        rows={strategyRows}
        rowTone={['success', 'success', undefined, undefined, undefined, undefined, undefined, 'warning']}
      />

      <Divider />

      <H2>分年 edge 对比</H2>
      <Text tone="secondary">
        单位为百分点。2023 是主要短板; 2026 明显偏向去风险后的 K线加动量组合。
      </Text>
      <Table
        headers={['策略', '2023', '2024', '2025', '2026']}
        rows={yearlyRows}
        rowTone={['success', 'success', undefined, undefined, undefined, 'warning']}
      />

      <Divider />

      <Grid columns={2} gap={18}>
        <Stack gap={10}>
          <H3>重要特征信号</H3>
          <Table headers={['Variant', 'Feature', 'Mean gain']} rows={featureRows} />
        </Stack>
        <Stack gap={10}>
          <H3>当前判断</H3>
          <Text>
            no_risk 和 kbar_momentum_state 都能把 2024/2026 拉起来, 但 2023 仍不稳。
            这意味着模型不是简单记住 2024, 而是在较新的市场状态中更会用 KLEN 与动量。
          </Text>
          <Text>
            下一步不应马上接 Rust, 应先做选股重合度和入场后路径归因, 确认 2026 的高 edge 不是少数样本偶然贡献。
          </Text>
        </Stack>
      </Grid>
    </Stack>
  );
}
