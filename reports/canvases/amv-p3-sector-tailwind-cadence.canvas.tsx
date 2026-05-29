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

const tenDayRows = [
  ['0', '+294.77%', '+318.06%', '+23.30pp', '-8.93%', '-10.56%'],
  ['1', '+293.16%', '+282.98%', '-10.18pp', '-8.93%', '-10.56%'],
  ['2', '+260.74%', '+237.56%', '-23.18pp', '-8.93%', '-11.96%'],
  ['3', '+277.32%', '+265.13%', '-12.19pp', '-8.93%', '-10.56%'],
  ['4', '+270.42%', '+246.26%', '-24.16pp', '-8.93%', '-10.56%'],
  ['5', '+285.60%', '+275.03%', '-10.56pp', '-8.93%', '-10.56%'],
  ['6', '+297.79%', '+321.57%', '+23.78pp', '-8.93%', '-10.56%'],
];

const twentyDayRows = [
  ['0', '+294.77%', '+329.50%', '+34.73pp', '-8.93%', '-8.62%'],
  ['1', '+293.16%', '+292.07%', '-1.09pp', '-8.93%', '-8.62%'],
  ['2', '+260.74%', '+238.49%', '-22.25pp', '-8.93%', '-15.42%'],
  ['3', '+277.32%', '+286.42%', '+9.10pp', '-8.93%', '-8.62%'],
  ['4', '+270.42%', '+324.47%', '+54.04pp', '-8.93%', '-8.62%'],
  ['5', '+285.60%', '+291.48%', '+5.89pp', '-8.93%', '-8.62%'],
  ['6', '+297.79%', '+339.65%', '+41.87pp', '-8.93%', '-8.62%'],
];

export default function AmvP3SectorTailwindCadence() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>P3 Sector Tailwind Cadence Check</H1>
        <Text tone="secondary">
          Python-side no-cost static cadence sensitivity: raw P3 vs sector rerank, 7 个起始 offset，6td 静态节奏。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="2 / 7" label="10d/b40/0.02 positive offsets" tone="danger" />
        <Stat value="-10.56pp" label="10d median delta" tone="danger" />
        <Stat value="5 / 7" label="20d/b40/0.02 positive offsets" tone="success" />
        <Stat value="+9.10pp" label="20d median delta" tone="info" />
      </Grid>

      <Callout tone="warning" title="结论修正">
        <Code>10d / bottom40 / 0.02</Code> 在默认 Rust 起点很好，但没有通过 cadence 检查：
        只有 2/7 个 offset 优于 raw P3，median 与 worst path 都更差。<Code>20d</Code> 更稳一些，
        但仍有一个 offset 明显受伤，暂时只能作为 challenger。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>10d Rerank: Offset Total Return</H2>
          <BarChart
            categories={['0', '1', '2', '3', '4', '5', '6']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Raw P3', data: [294.77, 293.16, 260.74, 277.32, 270.42, 285.6, 297.79], tone: 'info' },
              { name: '10d rerank', data: [318.06, 282.98, 237.56, 265.13, 246.26, 275.03, 321.57], tone: 'warning' },
            ]}
          />
        </Stack>

        <Stack gap={10}>
          <H2>20d Rerank: Offset Total Return</H2>
          <BarChart
            categories={['0', '1', '2', '3', '4', '5', '6']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Raw P3', data: [294.77, 293.16, 260.74, 277.32, 270.42, 285.6, 297.79], tone: 'info' },
              { name: '20d rerank', data: [329.5, 292.07, 238.49, 286.42, 324.47, 291.48, 339.65], tone: 'success' },
            ]}
          />
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>10d / bottom40 / 0.02</H2>
          <Table
            headers={['Offset', 'Raw', 'Rerank', 'Delta', 'Raw DD', 'Rerank DD']}
            rows={tenDayRows}
            rowTone={['success', 'warning', 'danger', 'warning', 'danger', 'warning', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>20d / bottom40 / 0.02</H2>
          <Table
            headers={['Offset', 'Raw', 'Rerank', 'Delta', 'Raw DD', 'Rerank DD']}
            rows={twentyDayRows}
            rowTone={['success', 'warning', 'danger', 'success', 'success', 'success', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Callout tone="info" title="下一步含义">
        不建议直接把 <Code>10d/bottom40/0.02</Code> 升级为默认 P3。更合理的是继续寻找更平滑的行业项，
        例如行业 rank 连续 penalty、10d/20d 混合、或只在特定 AMV/行业状态下启用。
      </Callout>

      <Text size="small" tone="tertiary">
        Reports: <Code>reports/amv_p3_sector_tailwind_cadence.json</Code> and
        <Code>reports/amv_p3_sector_tailwind_cadence_w20.json</Code>. Costs excluded; this is a cadence sensitivity
        diagnostic, not a replacement for Rust account NAV.
      </Text>
    </Stack>
  );
}
