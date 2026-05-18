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

const referenceRows = [
  ['2021', '+1.10%', '+0.39pp', '+16.55%', '-4.00%'],
  ['2022', '+1.17%', '+0.80pp', '+23.28%', '-2.72%'],
  ['2023', '+0.66%', '+1.26pp', '+14.73%', '-4.10%'],
  ['2024', '+3.34%', '+2.17pp', '+80.67%', '-1.95%'],
  ['2025', '+0.69%', '-0.70pp', '+8.00%', '-3.82%'],
  ['2026', '+1.10%', '+0.03pp', '+7.05%', '-1.58%'],
];

const bestByYearRows = [
  ['2024', 'KLEN 单因子', 'K线振幅收缩', '+3.81%', '+2.64pp', '+86.78%', '-3.62%'],
  ['2025', 'P1/K0.5/M1', '价格轻 + K线轻 + 动量', '+4.76%', '+3.37pp', '+86.31%', '-4.93%'],
  ['2026', 'P1/K0.5/M1', '价格轻 + K线轻 + 动量', '+7.05%', '+5.98pp', '+39.09%', '0.00%'],
];

const weakYearRows = [
  ['P1/K0.5/M1', '+4.68pp', '+2.06pp', '6/6', '+3.37pp', '+5.98pp'],
  ['P3/K1/M2', '+4.61pp', '+2.11pp', '6/6', '+3.35pp', '+5.87pp'],
  ['P1/K1/M2', '+4.40pp', '+1.93pp', '6/6', '+3.03pp', '+5.77pp'],
  ['P2/K0.5/M1', '+4.25pp', '+2.03pp', '6/6', '+2.93pp', '+5.56pp'],
  ['P3/K0.5/M1', '+4.24pp', '+2.04pp', '6/6', '+2.76pp', '+5.72pp'],
  ['当前 P2/K0.5/R0', '-0.33pp', '+0.82pp', '5/6', '-0.70pp', '+0.03pp'],
];

const walkForwardRows = [
  ['2025', '历史 edge 均值', 'P2/K0.5/M0.5', '+1.78pp', '+2.07pp', '+3.45%'],
  ['2025', '历史 tradeoff', 'P2/K0.5/M0.5', '+1.78pp', '+2.07pp', '+3.45%'],
  ['2026', '历史 edge 均值', 'P3/K1/M2', '+2.03pp', '+5.87pp', '+6.94%'],
  ['2026', '历史 tradeoff', 'P1/K0.5/M1', '+1.98pp', '+5.98pp', '+7.05%'],
];

export default function AmvYearlyWeightGrid() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV 年度权重网格诊断</H1>
        <Text tone="secondary">
          回到 AMV bull 宽池 Top3 / 6d 标签侧框架，比较旧 <Code>P/K/R</Code>
          权重和新增 <Code>P/K/M</Code> 动量权重。目标是解释 2025/2026 弱势，不是直接给交易参数。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="-0.70pp" label="当前基线 2025 edge" tone="warning" />
        <Stat value="+0.03pp" label="当前基线 2026 edge" tone="warning" />
        <Stat value="P1/K0.5/M1" label="2025/2026 最强权重" tone="success" />
        <Stat value="6/6" label="P/K/M 弱年候选正 edge 年份" tone="success" />
      </Grid>

      <Callout tone="success" title="核心发现">
        2025/2026 不是整个 AMV bull 宽池 Top3 框架失效，而是旧 <Code>P2/K0.5/R0</Code>
        价格位置权重过重、缺少动量项。加入 <Code>ret_5d / ret_20d</Code> 后，
        弱年份 edge 从接近 0 或负值变成明显正值。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>当前基线年度 edge</H2>
          <BarChart
            categories={['2021', '2022', '2023', '2024', '2025', '2026']}
            valueSuffix="pp"
            height={240}
            series={[{ name: 'P2/K0.5/R0 edge', data: [0.39, 0.8, 1.26, 2.17, -0.7, 0.03], tone: 'warning' }]}
          />
        </Stack>
        <Stack gap={10}>
          <H2>P/K/M 最强弱年 edge</H2>
          <BarChart
            categories={['2021', '2022', '2023', '2024', '2025', '2026']}
            valueSuffix="pp"
            height={240}
            series={[{ name: 'P1/K0.5/M1 edge', data: [3.01, 0.9, 0.56, 2.07, 3.37, 5.98], tone: 'success' }]}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>当前基线分年份</H2>
      <Table
        headers={['年份', '单笔均值', '相对随机', 'rolling NAV', 'MaxDD']}
        rows={referenceRows}
        rowTone={[undefined, undefined, undefined, 'success', 'warning', 'warning']}
        columnAlign={['left', 'right', 'right', 'right', 'right']}
      />

      <H2>逐年最佳</H2>
      <Table
        headers={['年份', '最佳权重/因子', '解释', '单笔均值', 'edge', 'rolling NAV', 'MaxDD']}
        rows={bestByYearRows}
        rowTone={['success', 'success', 'success']}
        columnAlign={['left', 'left', 'left', 'right', 'right', 'right', 'right']}
      />

      <H2>弱年份综合排序</H2>
      <Table
        headers={['权重', '2025/2026 平均 edge', '全样本 edge', '正 edge 年份', '2025 edge', '2026 edge']}
        rows={weakYearRows}
        rowTone={['success', 'success', 'success', 'success', 'success', 'warning']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>Walk-forward 检查</H2>
      <Table
        headers={['测试年', '选择口径', '历史选中权重', '训练 edge', '测试 edge', '测试单笔均值']}
        rows={walkForwardRows}
        rowTone={['success', 'success', 'success', 'success']}
        columnAlign={['left', 'left', 'left', 'right', 'right', 'right']}
      />

      <Callout tone="warning" title="怎么读">
        这些仍是 6d 标签侧结果，尚未等同于 Rust 真实交易。下一步更适合把
        <Code>P/K/M</Code> 候选导出成静态 sleeve，与 <Code>manual_p2_k0p5_r0_6td</Code>
        做同一套 T+1 open / 6td / Top3 Rust 对比。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_bull_pool_yearly_weight_grid/20260518_100342/summary.json</Code>.
        口径: AMV bull + LF2, Top3, 6d close-to-close 标签侧诊断, 覆盖 2021-04-20 到 2026-04-27。
      </Text>
    </Stack>
  );
}
