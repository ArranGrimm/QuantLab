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

const gridRows = [
  ['P3 raw', '0.00', '+201.69%', '+22.98%', '1.22', '-13.52%', '-0.77%', '100.0%'],
  ['Penalty 0.01', '0.01', '+196.62%', '+22.59%', '1.20', '-15.84%', '-1.86%', '85.5%'],
  ['Penalty 0.02', '0.02', '+242.10%', '+25.91%', '1.32', '-13.44%', '+0.63%', '78.1%'],
  ['Penalty 0.03', '0.03', '+223.56%', '+24.60%', '1.26', '-13.47%', '-5.50%', '74.1%'],
  ['Penalty 0.05', '0.05', '+226.92%', '+24.84%', '1.27', '-13.47%', '-5.48%', '72.5%'],
];

const yearlyRows = [
  ['2021', '+8.61%', '+20.84%', '+12.22pp'],
  ['2022', '+40.45%', '+29.27%', '-11.18pp'],
  ['2023', '+19.51%', '+14.11%', '-5.40pp'],
  ['2024', '+46.43%', '+57.64%', '+11.21pp'],
  ['2025', '+13.89%', '+20.98%', '+7.10pp'],
  ['2026', '-0.77%', '+0.63%', '+1.40pp'],
];

const attributionRows = [
  ['Exact overlap', '216 / 274', '78.83% of both sides'],
  ['Raw-only trades', '58', '-60.4K PnL'],
  ['Rerank-only trades', '58', '+84.0K PnL'],
  ['Unique trade delta', '+144.4K', 'main direct swap contribution'],
  ['Common trade delta', '+57.7K', 'capital path / sizing effect'],
  ['Cost delta', '+15.5K', 'acceptable versus gross gain'],
];

export default function AmvP3SectorTailwindRerank() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>P3 Sector Tailwind Rerank</H1>
        <Text tone="secondary">
          对 P3 候选池中行业 10 日收益排名底部 40% 的股票施加 soft penalty，并重新排序 Top3 后接
          <Code>bt-amv-topn</Code> 静态 strict 6td 回测。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+242.10%" label="best total return" tone="success" />
        <Stat value="+25.91%" label="best CAGR" tone="success" />
        <Stat value="1.32" label="best Sharpe" tone="success" />
        <Stat value="-13.44%" label="best MaxDD" tone="info" />
      </Grid>

      <Callout tone="success" title="结论">
        <Code>0.02</Code> 的行业弱势扣分是当前最佳点：总收益比 P3 raw 高 <Code>+40.41pp</Code>，
        Sharpe 从 <Code>1.22</Code> 到 <Code>1.32</Code>，2026 从负收益转正，同时 MaxDD 基本不变。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Total Return and MaxDD</H2>
          <BarChart
            categories={['Raw', '0.01', '0.02', '0.03', '0.05']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Total return', data: [201.69, 196.62, 242.1, 223.56, 226.92], tone: 'success' },
              { name: 'MaxDD absolute', data: [13.52, 15.84, 13.44, 13.47, 13.47], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: Rust static strict Top3 · 2021-01-04 to 2026-05-08.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Sharpe and 2026 YTD</H2>
          <BarChart
            categories={['Raw', '0.01', '0.02', '0.03', '0.05']}
            height={260}
            series={[
              { name: 'Sharpe', data: [1.22, 1.2, 1.32, 1.26, 1.27], tone: 'success' },
              { name: '2026 return (%)', data: [-0.77, -1.86, 0.63, -5.5, -5.48], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Sharpe uses 252 trading days and zero risk-free rate.
          </Text>
        </Stack>
      </Grid>

      <H2>Penalty Grid</H2>
      <Table
        headers={['Strategy', 'Penalty', 'Total', 'CAGR', 'Sharpe', 'MaxDD', '2026', 'Top3 Overlap']}
        rows={gridRows}
        rowTone={['info', 'warning', 'success', 'info', 'info']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Yearly: 0.02 vs Raw</H2>
          <Table
            headers={['Year', 'P3 Raw', 'Penalty 0.02', 'Delta']}
            rows={yearlyRows}
            rowTone={['success', 'warning', 'warning', 'success', 'success', 'success']}
            columnAlign={['left', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>Trade Attribution</H2>
          <Table
            headers={['Metric', 'Value', 'Interpretation']}
            rows={attributionRows}
            rowTone={['info', 'warning', 'success', 'success', 'success', 'warning']}
            columnAlign={['left', 'right', 'left']}
          />
        </Stack>
      </Grid>

      <Callout tone="warning" title="下一步约束">
        <Code>0.02</Code> 不是最终参数。它需要做起始 cadence、行业映射版本、行业 rank 窗口和 threshold
        的稳健性验证；当前数据仍是静态东方财富行业映射，存在历史分类偏差。
      </Callout>

      <Text size="small" tone="tertiary">
        Reports: <Code>reports/amv_p3_sector_tailwind_rerank_summary.json</Code> and
        <Code>reports/amv_p3_sector_tailwind_rerank_attribution.json</Code>.
      </Text>
    </Stack>
  );
}
