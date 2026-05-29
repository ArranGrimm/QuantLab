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

const p3RankRows = [
  ['Top 30% industry rank', '123', '+661,883', '+1.78%', '52.85%'],
  ['Middle 30%', '84', '+390,436', '+1.60%', '54.76%'],
  ['Bottom 40%', '67', '-43,860', '+0.09%', '49.25%'],
];

const p3RuleRows = [
  ['Skip bottom 40% industry rank', '+43,860', '67', '2', '6', '+10,837'],
  ['Skip aged/old + bottom industry', '+24,054', '24', '2', '4', 'n/a'],
  ['Skip cold breadth <35%', '-96,528', '49', '3', '2', '+28,761'],
  ['Keep only strict tailwind_ok', '-369,167', '169', '12', '10', '+38,838'],
];

const pb3RuleRows = [
  ['Skip bottom 40% industry rank', '-257,039', '1,045', '2', '0', '-42,921'],
  ['Skip aged/old + bottom industry', '-6,523', '465', '0', '0', '-8,187'],
  ['Skip cold breadth <35%', '-161,315', '719', '1', '0', '-64,329'],
  ['Keep only strict tailwind_ok', '-385,255', '1,443', '2', '0', '-92,299'],
];

const industryRows = [
  ['家电行业', '+208,240', '4', '+15.80%'],
  ['有色金属', '+157,226', '7', '+8.45%'],
  ['通用设备', '+146,525', '4', '+8.34%'],
  ['贸易行业', '-50,078', '4', '-2.94%'],
  ['医疗服务', '-47,490', '2', '-6.18%'],
  ['化纤行业', '-44,172', '3', '-3.57%'],
  ['中药', '-31,935', '9', '-0.61%'],
  ['航天航空', '-26,308', '2', '-2.51%'],
  ['水泥建材', '-23,247', '1', '-4.27%'],
];

export default function AmvSectorTailwindDiagnostic() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Sector Tailwind Diagnostic</H1>
        <Text tone="secondary">
          使用静态东方财富行业映射 + QMT 日线价量，在 <Code>signal_date</Code> 合成行业顺风因子并归因
          P3 / PB3 交易表现。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="5,549" label="industry mappings" tone="info" />
        <Stat value="86" label="industries" tone="info" />
        <Stat value="+43.9K" label="P3 bottom-rank skip delta" tone="success" />
        <Stat value="-257.0K" label="PB3 bottom-rank skip delta" tone="danger" />
      </Grid>

      <Callout tone="warning" title="结论">
        行业顺风对 P3 有价值，但第一版不适合做强硬 gate。P3 的行业 10 日收益排名底部 40% 区间合计亏损
        43.9K，说明它能帮助识别部分假突破；但严格只保留 <Code>tailwind_ok</Code> 会误杀大量赢家。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 by Industry Rank Bucket</H2>
          <BarChart
            categories={['Top 30%', 'Middle 30%', 'Bottom 40%']}
            valueSuffix="K"
            height={260}
            series={[{ name: 'Trade PnL', data: [661.9, 390.4, -43.9], tone: 'success' }]}
          />
          <Text size="small" tone="tertiary">
            Metric: trade-level PnL grouped by sector 10d return rank at signal_date.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>P3 Rank Buckets</H2>
          <Table
            headers={['Bucket', 'Trades', 'PnL', 'Avg PnL%', 'Win Rate']}
            rows={p3RankRows}
            rowTone={['success', 'success', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 Gating What-if</H2>
          <Table
            headers={['Rule', 'Delta', 'Skipped', 'Big Winners', 'Big Losers', '2026 Delta']}
            rows={p3RuleRows}
            rowTone={['success', 'info', 'warning', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>PB3 Gating What-if</H2>
          <Table
            headers={['Rule', 'Delta', 'Skipped', 'Big Winners', 'Big Losers', '2026 Delta']}
            rows={pb3RuleRows}
            rowTone={['danger', 'warning', 'danger', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>P3 Industry Attribution Sample</H2>
        <Table
          headers={['Industry', 'PnL', 'Trades', 'Avg PnL%']}
          rows={industryRows}
          rowTone={['success', 'success', 'success', 'danger', 'danger', 'danger', 'warning', 'warning', 'warning']}
          columnAlign={['left', 'right', 'right', 'right']}
        />
        <Text size="small" tone="tertiary">
          Source: <Code>reports/amv_sector_tailwind_diagnostic.json</Code>. Static industry mapping has historical
          classification bias and should be treated as prototype data.
        </Text>
      </Stack>
    </Stack>
  );
}
