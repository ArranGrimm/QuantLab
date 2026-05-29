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

const staticRows = [
  ['P3/K0.5', '+201.69%', '-13.52%', '+8.61%', '+40.45%', '+19.51%', '+46.43%', '+13.89%', '-0.77%'],
  ['PB1/CP0/RV0', '+190.28%', '-43.43%', '-17.69%', '+29.62%', '+2.82%', '+40.06%', '+22.83%', '+53.80%'],
  ['Ref P2/K0.5', '+170.80%', '-15.30%', '+10.37%', '+39.70%', '+14.31%', '+49.66%', '+12.57%', '-8.80%'],
  ['PB3/CP1/RV0', '+152.84%', '-41.85%', '-29.83%', '+57.17%', '+7.56%', '+30.27%', '+23.82%', '+32.15%'],
  ['P2/CP0.5/RV0.5', '+72.34%', '-48.92%', '+29.55%', '+6.05%', '-0.06%', '+39.49%', '-10.35%', '+0.38%'],
  ['PB2/CP0.5/RV0', '+66.25%', '-45.69%', '-40.52%', '+34.14%', '-4.86%', '+32.19%', '+24.16%', '+33.44%'],
];

const rollingRows = [
  ['PB3/CP1/RV0', '+99.62%', '-20.70%', '-0.41%', '+8.20%', '-3.37%', '+28.34%', '+29.73%', '+15.15%'],
  ['PB2/CP0.5/RV0', '+96.06%', '-22.74%', '-1.59%', '+5.02%', '-3.92%', '+29.57%', '+32.21%', '+15.25%'],
  ['PB1/CP0/RV0', '+89.41%', '-21.28%', '-2.32%', '+5.39%', '+7.08%', '+26.31%', '+21.11%', '+12.32%'],
  ['P3/K0.5', '+22.19%', '-11.93%', '+3.34%', '-1.50%', '+5.63%', '+9.72%', '+2.76%', '+0.80%'],
  ['Ref P2/K0.5', '+21.44%', '-10.89%', '+0.90%', '-1.09%', '+3.87%', '+13.05%', '+4.86%', '-1.17%'],
  ['P2/CP0.5/RV0.5', '+21.11%', '-14.03%', '+4.48%', '+1.01%', '-1.12%', '+17.41%', '+4.98%', '-5.84%'],
];

const correlationRows = [
  ['Ref static vs P3 static', '0.916', '同源高位/K线，更多是替换关系'],
  ['P3 static vs PB3/CP1 rolling', '0.255', '低相关，具备互补性'],
  ['P3 static vs PB2/CP0.5 rolling', '0.243', '低相关，具备互补性'],
  ['P3 static vs PB1/CP0 rolling', '0.214', '低相关，具备互补性'],
  ['Ref static vs PB3/CP1 rolling', '0.247', '低相关，能补 2026'],
  ['PB3/CP1 rolling vs PB2/CP0.5 rolling', '0.988', 'pullback 内部高度重叠'],
  ['PB3/CP1 rolling vs PB1/CP0 rolling', '0.921', 'pullback 内部高度重叠'],
];

const mixRows = [
  ['P3 100%', '+201.69%', '-13.52%', '-0.77%', '收益最高，2026 仍略弱'],
  ['P3 80% / PB3 rolling 20%', '+183.05%', '-11.60%', '+2.35%', '小幅牺牲收益，修复 2026'],
  ['P3 70% / PB3 rolling 30%', '+173.03%', '-10.81%', '+3.93%', '更均衡'],
  ['P3 60% / PB3 rolling 40%', '+162.79%', '-10.02%', '+5.51%', '回撤显著降低'],
  ['P3 50% / PB3 rolling 50%', '+152.38%', '-9.99%', '+7.11%', '防守更强，收益牺牲明显'],
];

export default function AmvExecutableSleeveRustComplement() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Executable Sleeves: Rust 回测与互补性</H1>
        <Text tone="secondary">
          基于 2026-05-20 的 <Code>bt-amv-topn</Code> 24 组真实回测：6 个 sleeve × static/rolling × strict/refill。
          年度收益来自 <Code>daily_equity.csv</Code>，相关性来自日收益序列。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+201.69%" label="P3/K0.5 静态 strict 净收益" tone="success" />
        <Stat value="-13.52%" label="P3/K0.5 静态 strict MaxDD" tone="success" />
        <Stat value="+99.62%" label="PB3/CP1/RV0 rolling refill 净收益" tone="info" />
        <Stat value="0.255" label="P3 vs PB3 rolling 日收益相关" tone="success" />
      </Grid>

      <Callout tone="info" title="命名约定">
        <Code>PB</Code> = <Code>ma_bias_20 + disp_bias_20</Code> 回调偏离；
        <Code>CP</Code> = <Code>KSFT + intraday_pos</Code> 收盘位置回调；
        <Code>RV</Code> = <Code>atr_14_pct + panic_vol_ratio_20d</Code> 风险/波动。后续 pullback
        文档、导出和 canvas 统一使用这套命名，避免和 TDX B1/B2/B3 策略混淆。
      </Callout>

      <Callout tone="success" title="核心判断">
        <Code>P3/K0.5/R0</Code> 是主基线替换候选；pullback 与 P/K 主线低相关，确实能补 2026。
        但 pullback 家族内部高度相关，所以不要同时堆多个 pullback，优先选一个代表 sleeve 做组合。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>静态 strict Top3: 净收益与回撤</H2>
          <BarChart
            categories={['P3/K0.5', 'PB1', 'Ref P2/K0.5', 'PB3/CP1', 'P2/CP0.5/RV0.5', 'PB2/CP0.5']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Total return', data: [201.69, 190.28, 170.8, 152.84, 72.34, 66.25], tone: 'success' },
              { name: 'MaxDD absolute', data: [13.52, 43.43, 15.3, 41.85, 48.92, 45.69], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: Rust static strict Top3, <Code>T+1 open / 6td / no-stop</Code>, 2021-01-01 to 2026-05-08.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>rolling21 refill Top10: 净收益与回撤</H2>
          <BarChart
            categories={['PB3/CP1', 'PB2/CP0.5', 'PB1', 'P3/K0.5', 'Ref P2/K0.5', 'P2/CP0.5/RV0.5']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Total return', data: [99.62, 96.06, 89.41, 22.19, 21.44, 21.11], tone: 'info' },
              { name: 'MaxDD absolute', data: [20.7, 22.74, 21.28, 11.93, 10.89, 14.03], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: Rust rolling21 refill Top10, daily max buys 3, position size 1/21.
          </Text>
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>2025/2026 年度互补</H2>
          <BarChart
            categories={['Ref static', 'P3 static', 'PB1 static', 'PB3 rolling', 'PB2 rolling', 'PB1 rolling']}
            valueSuffix="%"
            height={270}
            series={[
              { name: '2025 annual return', data: [12.57, 13.89, 22.83, 29.73, 32.21, 21.11], tone: 'success' },
              { name: '2026 YTD return', data: [-8.8, -0.77, 53.8, 15.15, 15.25, 12.32], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: annual return from daily equity, calendar-year end equity over previous year-end equity.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>P3 + PB3 rolling 简单权重混合</H2>
          <BarChart
            categories={['P3 100', '80/20', '70/30', '60/40', '50/50']}
            valueSuffix="%"
            height={270}
            series={[
              { name: 'Total return', data: [201.69, 183.05, 173.03, 162.79, 152.38], tone: 'success' },
              { name: 'MaxDD absolute', data: [13.52, 11.6, 10.81, 10.02, 9.99], tone: 'danger' },
              { name: '2026 YTD', data: [-0.77, 2.35, 3.93, 5.51, 7.11], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Transformation: daily rebalanced synthetic mix of P3 static strict and PB3/CP1/RV0 rolling refill.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>静态 strict Top3 年度收益</H2>
      <Table
        headers={['Sleeve', 'Total', 'MaxDD', '2021', '2022', '2023', '2024', '2025', '2026']}
        rows={staticRows}
        rowTone={['success', 'warning', 'success', 'warning', 'danger', 'danger']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>rolling21 refill Top10 年度收益</H2>
      <Table
        headers={['Sleeve', 'Total', 'MaxDD', '2021', '2022', '2023', '2024', '2025', '2026']}
        rows={rollingRows}
        rowTone={['success', 'success', 'success', undefined, undefined, undefined]}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>日收益相关性</H2>
      <Table
        headers={['Pair', 'Correlation', 'Interpretation']}
        rows={correlationRows}
        rowTone={['warning', 'success', 'success', 'success', 'success', 'danger', 'danger']}
        columnAlign={['left', 'right', 'left']}
      />

      <H2>简单组合敏感性</H2>
      <Table
        headers={['Mix', 'Total', 'MaxDD', '2026 YTD', 'Interpretation']}
        rows={mixRows}
        rowTone={['success', 'success', 'success', 'success', 'info']}
        columnAlign={['left', 'right', 'right', 'right', 'left']}
      />

      <Callout tone="success" title="Pullback 代表选择">
        后续 allocation / gating 暂以 <Code>PB3/CP1/RV0 rolling21 refill</Code> 作为 pullback 代表。
        它相对 <Code>PB2/CP0.5/RV0</Code> 全周期收益更高、MaxDD 更低，但两者高度重叠；
        <Code>PB2/CP0.5/RV0</Code> 保留为 forward challenger，不与 PB3 同时堆叠。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        Data artifact: <Code>reports/amv_executable_sleeve_rust_yearly.json</Code>. Backtest artifacts:
        P/K uses <Code>20260520_092047_*</Code> / <Code>20260520_092049_*</Code>;
        renamed pullback uses <Code>20260520_105222_*</Code> to <Code>20260520_105228_*</Code>.
      </Text>
    </Stack>
  );
}
