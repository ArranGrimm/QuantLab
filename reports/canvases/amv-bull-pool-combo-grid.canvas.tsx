import {
  BarChart,
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
} from 'cursor/canvas';

const bestRows = [
  ['top3', 'P2/K0.5/R0', '+2.625%', '+1.276pp', '+102.97%', '-2.90%', '+100.07%'],
  ['top5', 'P3/K0.5/R0', '+2.398%', '+1.050pp', '+85.53%', '-3.59%', '+81.94%'],
  ['top10', 'P2/K0.5/R1.5', '+2.232%', '+0.884pp', '+77.42%', '-3.36%', '+74.06%'],
];

const top20Rows = [
  ['top3', 'P2/K0.5/R0', '+2.625%', '+102.97%', '-2.90%', '+100.07%'],
  ['top3', 'P3/K0.5/R0', '+2.679%', '+102.10%', '-3.21%', '+98.88%'],
  ['top3', 'P3/K1/R0', '+2.520%', '+96.24%', '-3.85%', '+92.39%'],
  ['top3', 'P1/K0.5/R0', '+2.403%', '+91.53%', '-3.54%', '+87.99%'],
  ['top5', 'P3/K0.5/R0', '+2.398%', '+85.53%', '-3.59%', '+81.94%'],
  ['top5', 'P1/K0.5/R0', '+2.296%', '+81.64%', '-2.42%', '+79.22%'],
];

export default function AMVBullPoolComboGrid() {
  return (
    <Stack gap={18}>
      <Stack gap={6}>
        <H1>AMV 多头宽池权重网格</H1>
        <Text tone="secondary">
          对 <Code>高位+K线确认</Code> 做 36 组权重 × top3/top5/top10 测试。
          排序目标从单笔均值转为 <Code>tradeoff_score = rolling NAV + MaxDD</Code>，
          用来优先挑净值高且回撤浅的组合。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="top3" label="当前最优持仓宽度" tone="success" />
        <Stat value="P2/K0.5/R0" label="最佳权重结构" tone="success" />
        <Stat value="+102.97%" label="20d rolling NAV" tone="success" />
        <Stat value="-2.90%" label="20d 最大回撤" tone="info" />
      </Grid>

      <Callout tone="success" title="阶段结论">
        最优结果集中在 <Code>top3</Code>，说明当前信号更适合集中买最强的少数票。
        风险权重 <Code>R=0</Code> 排在前列，暂时不适合把 ATR/卖压直接混入主排序分数。
      </Callout>

      <Divider />

      <H2>topN 最优对比</H2>
      <Table
        headers={['持仓宽度', '最佳权重', '20d 单笔均值', '相对随机', '滚动净值', '最大回撤', 'Tradeoff']}
        rows={bestRows}
        rowTone={['success', 'info', 'info']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>20d Tradeoff Top 结果</H2>
      <BarChart
        categories={['top3 P2/K0.5/R0', 'top3 P3/K0.5/R0', 'top3 P3/K1/R0', 'top5 P3/K0.5/R0', 'top10 P2/K0.5/R1.5']}
        valueSuffix="%"
        height={260}
        series={[
          { name: '20d rolling NAV', data: [102.97, 102.10, 96.24, 85.53, 77.42], tone: 'success' },
          { name: '最大回撤绝对值', data: [2.90, 3.21, 3.85, 3.59, 3.36], tone: 'neutral' },
        ]}
      />

      <Table
        headers={['topN', '权重', '20d 单笔均值', '滚动净值', '最大回撤', 'Tradeoff']}
        rows={top20Rows}
        rowTone={['success', 'success', 'success', 'info', 'info', 'info']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right']}
      />

      <H2>怎么读权重</H2>
      <Text>
        <Code>P</Code> 是价格位置权重，包含 <Code>price_pos_20d↑</Code> 与
        <Code>close_to_high_20d↓</Code>；<Code>K</Code> 是 K线确认权重，包含
        <Code>KLEN↓</Code> 与 <Code>KMID2↑</Code>；<Code>R</Code> 是风险权重，包含
        <Code>atr_14_pct↓</Code> 与 <Code>panic_vol_ratio_20d↓</Code>。
      </Text>
      <Text>
        当前最优 <Code>P2/K0.5/R0</Code> 表示：价格位置是主信号，K线形态只做轻确认，
        风险因子暂时不进入主分数。下一步应把风险因子改成交易后的过滤或仓位控制再测。
      </Text>

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_bull_pool_combo_grid/20260503_172052/summary.json</Code>.
        口径: AMV bull + LF2, 每日 top3/top5/top10, 持仓 5/10/20 天。
      </Text>
    </Stack>
  );
}
