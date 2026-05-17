// @ts-nocheck
import { Callout, Divider, Grid, H1, H2, Stack, Stat, Table, Text } from 'cursor/canvas';

const robustnessRows = [
  ['kbar_momentum_old_state', '+7.89%', '+4.50%', '+6.57%', '+5.49%', '+5.15%'],
  ['no_risk_old_state', '+7.41%', '+3.98%', '+6.11%', '+4.91%', '+4.67%'],
  ['kbar_momentum_state', '+5.36%', '+3.13%', '+5.03%', '+3.67%', '+3.95%'],
  ['no_risk', '+4.52%', '+2.29%', '+4.19%', '+2.89%', '+3.18%'],
];

const concentrationRows = [
  ['kbar_momentum_old_state', 'sz.002931', '8', '47.6%', '+371.6pp'],
  ['no_risk_old_state', 'sz.002931', '8', '50.6%', '+371.6pp'],
  ['kbar_momentum_state', 'sz.002931', '6', '45.1%', '+239.5pp'],
  ['no_risk', 'sz.002931', '5', '51.9%', '+232.5pp'],
];

const overlapRows = [
  ['no_risk vs no_risk_old_state', '1.18 / 3', '39.4%', '29.7%'],
  ['kbar_momentum_state vs old_state', '1.88 / 3', '62.6%', '53.6%'],
  ['no_risk_old_state vs kbar_momentum_old_state', '1.61 / 3', '53.5%', '42.7%'],
  ['no_risk vs kbar_momentum_state', '1.42 / 3', '47.5%', '38.2%'],
];

export default function AmvLtrRobustnessTest() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>AMV LTR 稳健性测试</H1>
        <Text tone="secondary">
          Source: artifacts/amv_bull_pool_listwise_ranker/20260516_112948/selection_analysis_20260516_113241 · 2026 · top3 · 6d 标签
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="+4.50%" label="old_state 去最大票后均值" tone="success" />
        <Stat value="+6.57%" label="old_state 限制重复后均值" tone="success" />
        <Stat value="47%-52%" label="sz.002931 净贡献占比" tone="warning" />
        <Stat value="old_state" label="当前主候选" />
      </Grid>

      <Callout tone="success" title="结论">
        <Text>
          去掉新增状态特征的 old_state 版本不只是 2026 爆发更强, 在去最大贡献股票、限制重复入选、
          单笔收益截断和单票贡献封顶后仍然领先。当前更适合作为 Rust 回测前的主候选。
        </Text>
      </Callout>

      <H2>2026 稳健性敏感性</H2>
      <Text tone="secondary">
        去最大票: 删除当年累计贡献最高股票。限制重复: 同一股票每年最多保留 3 次。单笔截断: fwd_ret_6d 上限 30%。单票封顶: 单股票年度累计贡献上限 100pp。
      </Text>
      <Table
        headers={['Variant', '原始均值', '去最大票', '限制重复', '单笔30%截断', '单票100pp封顶']}
        rows={robustnessRows}
        rowTone={['success', 'success', undefined, undefined]}
      />

      <Divider />

      <H2>集中度仍需控制</H2>
      <Table
        headers={['Variant', '最大贡献股票', '入选次数', '占净收益', '累计贡献']}
        rows={concentrationRows}
        rowTone={['warning', 'warning', 'warning', 'warning']}
      />

      <Divider />

      <H2>2026 选股重合度</H2>
      <Table headers={['对比', '日均重合', '重合比例', 'Jaccard']} rows={overlapRows} />

      <Callout tone="info" title="下一步建议">
        <Text>
          主线切到 kbar_momentum_old_state 与 no_risk_old_state, 但接 Rust 前仍要加入候选层风控:
          同票重复约束、单票贡献监控, 以及 2026 top/worst 入场路径复盘。
        </Text>
      </Callout>
    </Stack>
  );
}
