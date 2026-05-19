import { Callout, Divider, Grid, H1, H2, H3, Stack, Stat, Table, Text } from 'cursor/canvas';

const coverageRows = [
  ['AMV bull 持续天数', 'bull_day_scaled', '已使用'],
  ['AMV ret_1d / ret_2d', 'amv_ret_1d_scaled / amv_ret_2d_scaled', '已使用'],
  ['AMV bull 初期 / 中期 / 后期', 'bull_phase_code', '已使用'],
  ['宽池近 5 日平均收益', 'trail_pool_ret_5d', '已使用'],
  ['宽池上涨比例', 'pool_up_ratio_5d', '已使用'],
  ['宽池 topN 动量强度', 'pool_topn_ret_5d / pool_topn_ret_20d', '本次补齐'],
  ['宽池成交额变化', 'pool_amount_ma5_vs_20', '本次补齐'],
];

const strategyRows = [
  ['LTR no_risk', '+2.304pp', '+3.056%', '0.051', '+0.054pp', '+2.821pp', '+2.872pp', '+3.471pp'],
  ['LTR kbar_momentum_state', '+1.953pp', '+2.704%', '0.041', '+0.411pp', '+0.910pp', '+2.186pp', '+4.307pp'],
  ['Baseline KLEN', '+1.818pp', '+2.569%', '0.024', '+0.469pp', '+2.438pp', '+1.560pp', '+2.804pp'],
  ['Baseline 5日动量', '+1.373pp', '+2.124%', '0.049', '+0.835pp', '-0.527pp', '+1.053pp', '+4.130pp'],
];

const importanceRows = [
  ['no_risk', 'rank_ret_20d', '10498'],
  ['no_risk', 'rank_KLEN', '2437'],
  ['no_risk', 'pool_topn_ret_20d', '412'],
  ['no_risk', 'pool_amount_ma5_vs_20', '291'],
  ['kbar_momentum_state', 'rank_ret_20d', '8853'],
  ['kbar_momentum_state', 'rank_KLEN', '3855'],
  ['kbar_momentum_state', 'pool_topn_ret_20d', '435'],
  ['kbar_momentum_state', 'pool_amount_ma5_vs_20', '330'],
];

export default function AmvLtrStateFeatureCompletion() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>AMV LTR 状态特征补齐</H1>
        <Text tone="secondary">
          Source: artifacts/amv_bull_pool_listwise_ranker/20260516_111818 · 2019-01-29 至 2026-04-27 · 6d/top3 · state_top_n=20
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="+2.304pp" label="no_risk 平均 edge" tone="success" />
        <Stat value="+1.953pp" label="kbar+momentum 平均 edge" tone="success" />
        <Stat value="24" label="完整特征数" />
        <Stat value="2" label="本次新增状态量" />
      </Grid>

      <Callout tone="info" title="回答你的问题">
        <Text>
          之前版本并不是完全没用状态特征, 但少了两个明确状态量: 宽池 topN 动量强度和宽池成交额变化。
          现在已经补齐, 并重新跑了 no_risk 与 kbar_momentum_state 两个核心候选。
        </Text>
      </Callout>

      <H2>状态特征覆盖</H2>
      <Table headers={['讨论过的状态特征', '脚本字段', '当前状态']} rows={coverageRows} />

      <Divider />

      <H2>补齐后表现</H2>
      <Text tone="secondary">
        单位: edge 为百分点; 平均收益为 top3 6日未来收益。补齐状态后 no_risk 仍超过 KLEN, 但 2026 的极端优势有所回落。
      </Text>
      <Table
        headers={['策略', '平均 edge', '平均收益', 'Precision@3', '2023', '2024', '2025', '2026']}
        rows={strategyRows}
        rowTone={['success', 'success', undefined, undefined]}
      />

      <Divider />

      <Grid columns={2} gap={18}>
        <Stack gap={10}>
          <H3>新增状态特征重要性</H3>
          <Table headers={['Variant', 'Feature', 'Mean gain']} rows={importanceRows} />
        </Stack>
        <Stack gap={10}>
          <H3>当前判断</H3>
          <Text>
            新增状态特征不是单纯放大收益, 更像稳定器: no_risk 的 2023 从负 edge 转为微正,
            2025 明显增强, 但 2026 从前一版的极端高 edge 回落。
          </Text>
          <Text>
            下一步应比较“补齐前后”的选股重合度和 2026 贡献来源, 判断回落是减少偶然暴露,
            还是错过真正强势票。
          </Text>
        </Stack>
      </Grid>
    </Stack>
  );
}
