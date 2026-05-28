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

const standaloneRows = [
  ['P3 static strict', '+201.69%', '-13.52%', '+13.89%', '-0.77%', '主线收益最高，2026 略弱'],
  ['PB3 rolling raw', '+99.62%', '-20.70%', '+29.73%', '+15.15%', '互补强，但自身回撤较深'],
  ['PB3 rolling gated', '+109.73%', '-16.20%', '+25.15%', '+14.28%', '自身收益/回撤优于 raw，2023 修复明显'],
];

const gatedRows = [
  ['P3 100 / PB3 gated 0', '+201.69%', '-13.52%', '+13.89%', '-0.77%'],
  ['P3 90 / PB3 gated 10', '+193.66%', '-12.41%', '+15.09%', '+0.71%'],
  ['P3 80 / PB3 gated 20', '+185.25%', '-11.64%', '+16.29%', '+2.20%'],
  ['P3 70 / PB3 gated 30', '+176.50%', '-10.87%', '+17.46%', '+3.70%'],
  ['P3 60 / PB3 gated 40', '+167.47%', '-10.10%', '+18.62%', '+5.20%'],
  ['P3 50 / PB3 gated 50', '+158.19%', '-9.33%', '+19.76%', '+6.70%'],
];

const rawVsGatedRows = [
  ['90 / 10', '+1.31pp', '-0.02pp', '-0.42pp', '-0.08pp'],
  ['80 / 20', '+2.56pp', '-0.04pp', '-0.84pp', '-0.15pp'],
  ['70 / 30', '+3.76pp', '-0.06pp', '-1.28pp', '-0.24pp'],
  ['60 / 40', '+4.90pp', '-0.08pp', '-1.73pp', '-0.32pp'],
  ['50 / 50', '+5.96pp', '+0.66pp', '-2.18pp', '-0.40pp'],
];

const corrRows = [
  ['P3 static vs PB3 raw', '0.255', '低相关，组合互补成立'],
  ['P3 static vs PB3 gated', '0.260', 'gating 后与 P3 相关性基本不变'],
  ['PB3 raw vs PB3 gated', '0.958', 'gating 是风控增强，不是新 sleeve'],
];

const annualizedRows = [
  ['P3 static strict', '+22.98%', '19.10%', '1.22', '-13.52%', '1.70'],
  ['PB3 rolling raw', '+13.82%', '14.46%', '1.00', '-20.70%', '0.67'],
  ['PB3 rolling gated', '+14.88%', '13.46%', '1.14', '-16.20%', '0.92'],
  ['P3 90 / PB3 gated 10', '+22.36%', '17.58%', '1.28', '-12.41%', '1.80'],
  ['P3 80 / PB3 gated 20', '+21.69%', '16.19%', '1.34', '-11.64%', '1.86'],
  ['P3 70 / PB3 gated 30', '+20.99%', '14.93%', '1.40', '-10.87%', '1.93'],
  ['P3 60 / PB3 gated 40', '+20.24%', '13.87%', '1.45', '-10.10%', '2.00'],
  ['P3 50 / PB3 gated 50', '+19.44%', '13.03%', '1.48', '-9.33%', '2.08'],
];

export default function AmvP3Pb3GatedAllocation() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>P3 + PB3 Gated Allocation Diagnostic</H1>
        <Text tone="secondary">
          基于 Rust <Code>daily_equity.csv</Code> 的 daily rebalanced synthetic allocation。
          比较 <Code>P3 static strict</Code>、<Code>PB3 rolling raw</Code> 与 <Code>PB3 rolling gated</Code>。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+21.69%" label="80/20 gated CAGR" tone="success" />
        <Stat value="1.34" label="80/20 gated Sharpe" tone="success" />
        <Stat value="-11.64%" label="80/20 gated MaxDD" tone="success" />
        <Stat value="1.86" label="80/20 gated Calmar" tone="info" />
      </Grid>

      <Callout tone="success" title="组合层结论">
        10-30% 的 PB3 gated 配置能把 P3 的 2026 从负收益转正，并把 MaxDD 从 13.52% 降到约 10.87%-12.41%。
        代价是全周期收益从 P3 单独的 201.69% 降到 176.50%-193.66%。
      </Callout>

      <Callout tone="info" title="Raw vs gated 的真实含义">
        PB3 gating 在组合层并不是显著提升 2025/2026 年度收益，而是改善 PB3 自身全周期收益和回撤。
        同权重下 gated 比 raw 全周期收益更高，但 2025/2026 年度收益略低；所以它更像风险质量提升，而不是进攻增强。
      </Callout>

      <Stack gap={10}>
        <H2>Annualized Risk Metrics</H2>
        <Table
          headers={['Strategy', 'CAGR', 'Ann. Vol', 'Sharpe', 'MaxDD', 'Calmar']}
          rows={annualizedRows}
          rowTone={['success', 'warning', 'success', 'success', 'success', 'success', 'info', 'info']}
          columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
        />
        <Text size="small" tone="tertiary">
          CAGR uses calendar span 2021-01-04 to 2026-05-08. Ann. Vol and Sharpe use 252 trading days and zero
          risk-free rate.
        </Text>
      </Stack>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Gated Allocation: Total Return vs MaxDD</H2>
          <BarChart
            categories={['100/0', '90/10', '80/20', '70/30', '60/40', '50/50']}
            valueSuffix="%"
            height={270}
            series={[
              { name: 'Total return', data: [201.69, 193.66, 185.25, 176.5, 167.47, 158.19], tone: 'success' },
              { name: 'MaxDD absolute', data: [13.52, 12.41, 11.64, 10.87, 10.1, 9.33], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: Rust daily equity · 2021-01-04 to 2026-05-08 · daily rebalanced synthetic mix.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>2025/2026: Gated Allocation</H2>
          <BarChart
            categories={['100/0', '90/10', '80/20', '70/30', '60/40', '50/50']}
            valueSuffix="%"
            height={270}
            series={[
              { name: '2025 return', data: [13.89, 15.09, 16.29, 17.46, 18.62, 19.76], tone: 'success' },
              { name: '2026 YTD return', data: [-0.77, 0.71, 2.2, 3.7, 5.2, 6.7], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Transformation: calendar-year compounded daily returns from the mixed daily return series.
          </Text>
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Standalone Sleeves</H2>
          <Table
            headers={['Sleeve', 'Total', 'MaxDD', '2025', '2026', 'Interpretation']}
            rows={standaloneRows}
            rowTone={['success', 'warning', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>Daily Return Correlation</H2>
          <Table
            headers={['Pair', 'Correlation', 'Interpretation']}
            rows={corrRows}
            rowTone={['success', 'success', 'info']}
            columnAlign={['left', 'right', 'left']}
          />
        </Stack>
      </Grid>

      <H2>Gated Allocation Sensitivity</H2>
      <Table
        headers={['Mix', 'Total', 'MaxDD', '2025', '2026']}
        rows={gatedRows}
        rowTone={['success', 'success', 'success', 'success', 'info', 'info']}
        columnAlign={['left', 'right', 'right', 'right', 'right']}
      />

      <H2>Same Weight: Gated PB3 minus Raw PB3</H2>
      <Table
        headers={['Mix', 'Total Return Delta', 'MaxDD Delta', '2025 Delta', '2026 Delta']}
        rows={rawVsGatedRows}
        rowTone={['success', 'success', 'success', 'success', 'warning']}
        columnAlign={['left', 'right', 'right', 'right', 'right']}
      />

      <Divider />

      <Callout tone="info" title="Practical default candidate">
        80/20 gated 是当前较自然的起点：保留大部分 P3 收益，把 2026 转正到 +2.20%，同时将 MaxDD 降至 11.64%。
        若更重视回撤与 2026 修复，70/30 gated 更均衡，但全周期收益牺牲更明显。
      </Callout>

      <Text size="small" tone="tertiary">
        Data artifact: <Code>reports/amv_p3_pb3_gated_allocation.json</Code>. P3 source:
        <Code>20260520_092049_candidate_p3_k0p5_b0_c0_r0</Code>. PB3 raw source:
        <Code>20260521_090945_pullback_p0_k0_pb3_cp1_rv0</Code>. PB3 gated source:
        <Code>20260526_184023_pullback_p0_k0_pb3_cp1_rv0</Code>.
      </Text>
    </Stack>
  );
}
