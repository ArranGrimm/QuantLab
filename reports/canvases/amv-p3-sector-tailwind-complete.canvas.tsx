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
  ['mix10/20 linear rel20 p0.03', '+219.55%', '1.279', '-14.07%', '+17.86pp', '96.27%', '3.11%'],
  ['mix10/20 linear rel20 p0.04', '+219.55%', '1.279', '-14.07%', '+17.86pp', '95.95%', '2.74%'],
  ['10d linear p0.04', '+218.69%', '1.265', '-13.51%', '+17.00pp', '81.33%', '9.87%'],
  ['Raw P3', '+201.69%', '1.225', '-13.52%', '+0.00pp', '-', '-'],
  ['mix10/20 linear p0.04', '+206.02%', '1.192', '-15.80%', '+4.32pp', '82.84%', '12.08%'],
  ['20d linear p0.03', '+198.90%', '1.166', '-17.71%', '-2.79pp', '82.02%', '15.56%'],
];

const cadenceRows = [
  ['0', '+294.77%', '+316.20%', '+21.43pp', '-8.93%', '-8.93%'],
  ['1', '+293.16%', '+314.51%', '+21.35pp', '-8.93%', '-8.93%'],
  ['2', '+260.74%', '+280.32%', '+19.59pp', '-8.93%', '-8.93%'],
  ['3', '+277.32%', '+297.81%', '+20.49pp', '-8.93%', '-8.93%'],
  ['4', '+270.42%', '+290.54%', '+20.11pp', '-8.93%', '-8.93%'],
  ['5', '+285.60%', '+306.53%', '+20.94pp', '-8.93%', '-8.93%'],
  ['6', '+297.79%', '+319.38%', '+21.60pp', '-8.93%', '-8.93%'],
];

export default function AmvP3SectorTailwindComplete() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>P3 Sector Tailwind Complete Check</H1>
        <Text tone="secondary">
          Rust 6td static strict Top3 grid + Python-side no-cost 7-offset cadence. 目标是把行业因子从离散扣分改成可升级的连续表达。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+219.55%" label="Best Rust total return" tone="success" />
        <Stat value="+17.86pp" label="Excess return vs raw P3" tone="success" />
        <Stat value="7 / 7" label="Positive cadence offsets" tone="success" />
        <Stat value="+20.94pp" label="Median cadence delta" tone="success" />
      </Grid>

      <Callout tone="success" title="升级候选">
        推荐把 <Code>mix10/20 + linear penalty + rel20_under0 + p0.03</Code> 作为行业因子的主线候选。
        它只在“行业处于弱势且个股相对行业也弱”的交集上扣分，约 3.1% 入选样本被实际 penalty，默认 Rust 起点和 7 个 cadence offset 都优于 raw P3。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Rust Grid: Total Return</H2>
          <BarChart
            categories={['Raw', '10d linear p0.04', 'mix rel20 p0.03', 'mix rel20 p0.04']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Total return', data: [201.69, 218.69, 219.55, 219.55], tone: 'success' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_p3_sector_tailwind_complete_grid.json</Code>; Rust account NAV, costs included.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Cadence: Offset Total Return</H2>
          <BarChart
            categories={['0', '1', '2', '3', '4', '5', '6']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Raw P3', data: [294.77, 293.16, 260.74, 277.32, 270.42, 285.6, 297.79], tone: 'info' },
              { name: 'mix10/20 linear rel20 p0.03', data: [316.2, 314.51, 280.32, 297.81, 290.54, 306.53, 319.38], tone: 'success' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_p3_sector_tailwind_complete_cadence_p0p03.json</Code>; costs excluded, cadence sensitivity only.
          </Text>
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Focused Grid Summary</H2>
        <Table
          headers={['Variant', 'Return', 'Sharpe', 'MaxDD', 'Excess', 'Raw overlap', 'Penalized']}
          rows={gridRows}
          rowTone={['success', 'success', 'success', 'info', 'warning', 'danger']}
          columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
        />
      </Stack>

      <Stack gap={10}>
        <H2>Best Candidate Cadence</H2>
        <Table
          headers={['Offset', 'Raw', 'Rerank', 'Delta', 'Raw DD', 'Rerank DD']}
          rows={cadenceRows}
          rowTone={['success', 'success', 'success', 'success', 'success', 'success', 'success']}
          columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
        />
      </Stack>

      <Callout tone="info" title="为什么选 p0.03">
        <Code>p0.03</Code> 和 <Code>p0.04</Code> 在本轮 Rust 与 cadence 上表现相同；优先选择更保守的 <Code>p0.03</Code>，
        因为它的最大扣分更低，同时保持同样的收益和稳定性。
      </Callout>
    </Stack>
  );
}
