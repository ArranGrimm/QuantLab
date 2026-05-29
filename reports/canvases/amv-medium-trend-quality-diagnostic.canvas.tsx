// @ts-nocheck
import {
  BarChart,
  Callout,
  Code,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
} from 'cursor/canvas';

const p3Buckets = [
  ['Structure 64 high', '124', '+1,075.7K', '+2.80%', '55.6%'],
  ['Structure 64 low', '33', '-167.0K', '-1.37%', '39.4%'],
  ['Quality 128 high', '37', '+626.1K', '+5.57%', '64.9%'],
  ['Quality 128 low', '66', '+64.5K', '+0.38%', '47.0%'],
  ['Position 128 high', '150', '+979.3K', '+2.22%', '55.3%'],
  ['Position 128 low', '43', '-122.7K', '-0.75%', '39.5%'],
];

const p3Rules = [
  ['Structure + quality weak', '90', '-97.2K', '+97.2K', '3', '9', '-28.7K'],
  ['Low structure 128', '52', '+33.1K', '-33.1K', '2', '4', '-108.9K'],
  ['Low quality 128', '67', '+72.1K', '-72.1K', '3', '6', '-108.8K'],
  ['Low ret/vol 128', '73', '+63.8K', '-63.8K', '3', '5', '-103.4K'],
  ['High pos + low quality', '70', '+157.7K', '-157.7K', '5', '3', '+7.7K'],
];

const pb3Buckets = [
  ['Structure 128 high', '589', '+333.7K', '+1.80%', '50.3%'],
  ['Structure 128 low', '609', '+50.8K', '+0.29%', '45.0%'],
  ['Quality 128 high', '326', '+225.2K', '+1.99%', '52.1%'],
  ['Quality 128 low', '348', '+10.5K', '+0.10%', '44.5%'],
  ['Ret/vol 128 high', '643', '+335.1K', '+1.68%', '50.2%'],
  ['Ret/vol 128 low', '616', '+46.9K', '+0.20%', '44.0%'],
];

const rustGrid = [
  ['Raw P3', '+201.69%', '13.52%', '52.6%', 'baseline'],
  ['128d structure/quality p0.025', '+251.32%', '15.33%', '54.7%', '+49.63pp'],
  ['128d structure/quality p0.01', '+201.10%', '14.71%', '54.7%', '-0.59pp'],
  ['128d structure/quality p0.02', '+261.77%', '15.33%', '54.7%', '+60.08pp'],
  ['128d structure/quality p0.03', '+264.90%', '14.05%', '54.4%', '+63.21pp'],
  ['128d structure/quality p0.035', '+240.97%', '16.07%', '53.6%', '+39.27pp'],
  ['128d structure/quality p0.04', '+225.43%', '16.13%', '53.6%', '+23.74pp'],
];

const comboGrid = [
  ['128d structure/quality p0.03', '+264.90%', '14.05%', '+63.21pp', 'baseline challenger'],
  ['Sector 0.02 + 128d structure/quality 0.03', '+272.06%', '14.05%', '+70.37pp', 'best combo'],
  ['Sector 0.03 + 128d structure/quality 0.03', '+247.51%', '14.05%', '+45.82pp', 'too much sector penalty'],
];

const yearlyReturns = [
  ['2021', '+8.61%', '+13.25%', '+4.64pp', 'improved'],
  ['2022', '+40.45%', '+39.66%', '-0.79pp', 'slightly worse'],
  ['2023', '+19.51%', '+22.95%', '+3.44pp', 'improved'],
  ['2024', '+46.43%', '+57.50%', '+11.07pp', 'main contributor'],
  ['2025', '+13.89%', '+23.12%', '+9.24pp', 'main contributor'],
  ['2026', '-0.77%', '-1.34%', '-0.57pp', 'still weak'],
];

const annualRestart2026 = [
  ['Raw P3', '-8.55%', '-2.16%', '+14.57%', '2/7'],
  ['Sector complete', '-8.49%', '-4.67%', '+15.68%', '2/7'],
  ['128d structure/quality p0.03', '-9.72%', '+1.85%', '+16.20%', '5/7'],
  ['Context combo', '-9.72%', '+1.85%', '+16.20%', '5/7'],
];

export default function AmvMediumTrendQualityDiagnostic() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV 128d Structure / Trend Quality Diagnostic</H1>
        <Text tone="secondary">
          第二阶段首轮：在 signal_date 为 P3 static / PB3 rolling 交易拼接 64/128 日中期结构与趋势质量特征。
          结果是 trade-level diagnostic，尚未进入 Rust 信号导出。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+272.06%" label="Best Rust total return" tone="success" />
        <Stat value="+70.37pp" label="Delta vs raw P3" tone="success" />
        <Stat value="7/7" label="Positive cadence offsets" tone="success" />
        <Stat value="14.05%" label="Best Rust MaxDD" tone="warning" />
      </Grid>

      <Callout tone="success" title="Rust 结论">
        第二阶段复核后，最佳是 <Code>sector mix10/20 linear p0.02 + 128d structure/quality p0.03</Code>：
        raw P3 <Code>+201.69%</Code> / MaxDD <Code>13.52%</Code> 提升到 <Code>+272.06%</Code> /
        MaxDD <Code>14.05%</Code>。no-cost cadence 为 <Code>7/7</Code> 个 offset 优于 raw，median delta <Code>+31.69pp</Code>。
      </Callout>

      <Grid columns="1.2fr 0.8fr" gap={18}>
        <Stack gap={10}>
          <H2>Yearly Rust Returns</H2>
          <Table
            headers={['Year', 'Raw P3', 'Best combo', 'Delta', 'Comment']}
            rows={yearlyReturns}
            rowTone={['success', 'warning', 'success', 'success', 'success', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'left']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_p3_context_combo_attribution.json</Code>; account-level yearly returns from Rust daily equity.
          </Text>
        </Stack>

        <Callout tone="warning" title="分年解释">
          组合不是单年撑起来：2024 和 2025 是主贡献，2021 / 2023 也正贡献。
          2026 仍小幅弱于 raw，关键拖累是 2026-01 trade delta 约 <Code>-110.4K</Code>。
        </Callout>
      </Grid>

      <Grid columns="1.2fr 0.8fr" gap={18}>
        <Stack gap={10}>
          <H2>Annual Restart 2026</H2>
          <Table
            headers={['Variant', 'Worst', 'Median', 'Best', 'Positive offsets']}
            rows={annualRestart2026}
            rowTone={['danger', 'danger', 'warning', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_p3_annual_restart_cadence_context_combo.json</Code>; each year restarts independently, no costs, 7 entry offsets.
          </Text>
        </Stack>

        <Callout tone="warning" title="Annual Restart 结论">
          Context combo 改善了 2026 median 和正收益 offset 数量，但没有修复最差 offset：
          raw worst <Code>-8.55%</Code>，combo worst <Code>-9.72%</Code>。因此它仍是强 challenger，不宜直接升级默认 P3。
        </Callout>
      </Grid>

      <Callout tone="warning" title="诊断层解释">
        中期结构和趋势质量确实能解释 P3 的赢家分布：高 64 日结构、高 128 日质量、128 日位置高的交易明显更强。
        但单个低结构 / 低质量 hard skip 会误杀 2026 赢家，当前更适合做 <Code>soft penalty / rerank</Code>，不是直接删信号。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 Bucket PnL</H2>
          <BarChart
            categories={['Struct64 high', 'Struct64 low', 'Qual128 high', 'Qual128 low', 'Pos128 high', 'Pos128 low']}
            valueSuffix="K"
            height={260}
            series={[
              { name: 'Total PnL', data: [1075.7, -167.0, 626.1, 64.5, 979.3, -122.7], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_medium_trend_quality_diagnostic.json</Code>; P3 static strict trades, 2019-2026.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>P3 Bucket Detail</H2>
          <Table
            headers={['Bucket', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
            rows={p3Buckets}
            rowTone={['success', 'danger', 'success', 'info', 'success', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Rust Static Strict Grid</H2>
        <Table
          headers={['Variant', 'Total return', 'MaxDD', 'Win rate', 'Return delta']}
          rows={rustGrid}
          rowTone={['info', 'warning', 'success', 'success']}
          columnAlign={['left', 'right', 'right', 'right', 'right']}
        />
      </Stack>

      <Stack gap={10}>
        <H2>Context Combo Validation</H2>
        <Table
          headers={['Variant', 'Total return', 'MaxDD', 'Return delta', 'Comment']}
          rows={comboGrid}
          rowTone={['success', 'success', 'warning']}
          columnAlign={['left', 'right', 'right', 'right', 'left']}
        />
        <Text size="small" tone="tertiary">
          Source: <Code>reports/amv_p3_context_combo_validation.json</Code>. Here <Code>128d structure/quality</Code> means the 128-day medium-term structure score plus 128-day trend-quality score.
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>P3 Candidate Rules</H2>
        <Table
          headers={['Rule', 'Skipped', 'Skipped PnL', 'Delta', 'Big winners killed', 'Big losers avoided', '2026 delta']}
          rows={p3Rules}
          rowTone={['success', 'danger', 'danger', 'danger', 'danger']}
          columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
        />
      </Stack>

      <Stack gap={10}>
        <H2>PB3 Contrast</H2>
        <Table
          headers={['Bucket', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
          rows={pb3Buckets}
          rowTone={['success', 'info', 'success', 'warning', 'success', 'info']}
          columnAlign={['left', 'right', 'right', 'right', 'right']}
        />
      </Stack>

      <Callout tone="info" title="下一步建议">
        第二阶段验收复核通过，可以阶段性收口。剩余风险是 2026-01 仍被牺牲，后续进入第三阶段前只需保留 forward 监控和最终路线对比；
        PB3 方向暂不优先，因 trade-level delta 很小。
      </Callout>
    </Stack>
  );
}
