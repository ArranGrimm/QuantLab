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

const p3Rules = [
  ['Hot yday premium + high new-high', '54', '-132.5K', '+132.5K', '3', '6', '+62.9K'],
  ['Hot yday limit-up premium', '95', '-120.6K', '+120.6K', '5', '8', '+73.3K'],
  ['Low market breadth', '78', '+495.3K', '-495.3K', '9', '2', '+29.2K'],
  ['Low 20d new-high breadth', '63', '+497.2K', '-497.2K', '8', '1', '0.0K'],
];

const pb3Rules = [
  ['Low limit-up count', '396', '-24.0K', '+24.0K', '0', '0', '-3.3K'],
  ['Low market breadth', '455', '+177.3K', '-177.3K', '0', '0', '-68.8K'],
  ['Hot yday premium + high new-high', '318', '+90.6K', '-90.6K', '1', '0', '-3.8K'],
];

const p3Buckets = [
  ['Yday LU premium high', '95', '-120.6K', '-0.36%', '44.2%'],
  ['Yday LU premium mid', '104', '+757.3K', '+2.66%', '56.7%'],
  ['Yday LU premium low', '75', '+371.8K', '+1.56%', '57.3%'],
  ['20d new-high high', '124', '+28.0K', '+0.48%', '49.2%'],
  ['20d new-high mid', '87', '+483.3K', '+1.56%', '51.7%'],
  ['20d new-high low', '63', '+497.2K', '+2.62%', '60.3%'],
];

export default function AmvMarketSentimentDiagnostic() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Market Sentiment Diagnostic</H1>
        <Text tone="secondary">
          Signal-date whole-market emotion factors for P3 static and PB3 rolling. Features use QMT adjusted daily bars and are observable before T+1 open execution.
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+132.5K" label="P3 trade-level hot rule delta" tone="success" />
        <Stat value="+182.02%" label="Rust hard-gate total return" tone="danger" />
        <Stat value="-19.67pp" label="Rust delta vs raw P3" tone="danger" />
        <Stat value="13.53%" label="Rust hard-gate MaxDD" tone="warning" />
      </Grid>

      <Callout tone="danger" title="Rust 结论">
        <Code>hot_yday_premium_and_new_high</Code> 在 trade-level 看似有效，但 date-level hard gate 没有兑现到真实账户路径：
        raw P3 <Code>+201.69%</Code> / MaxDD <Code>13.52%</Code> 降到 gated <Code>+182.02%</Code> / MaxDD <Code>13.53%</Code>。
        当前 hard gate 被否决。
      </Callout>

      <Callout tone="warning" title="诊断层发现">
        P3 的简单“冷市场过滤”不成立。低涨停数、低新高家数、低市场上涨占比这些环境反而承载了很多大赢家。
        更值得保留为线索的是 <Code>昨日涨停溢价过热 + 20日新高家数偏高</Code> 的拥挤环境，但只能考虑 soft/rerank 或与行业因子交互。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 Rule Delta</H2>
          <BarChart
            categories={['Hot premium + new-high', 'Hot premium', 'Low breadth', 'Low new-high']}
            valueSuffix="K"
            height={260}
            series={[
              { name: 'Trade-level delta', data: [132.5, 120.6, -495.3, -497.2], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_market_sentiment_diagnostic.json</Code>; trade-level what-if, not Rust account NAV.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>P3 Bucket PnL</H2>
          <Table
            headers={['Bucket', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
            rows={p3Buckets}
            rowTone={['danger', 'success', 'success', 'info', 'success', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>P3 Candidate Rules</H2>
        <Table
          headers={['Rule', 'Skipped', 'Skipped PnL', 'Delta', 'Big winners killed', 'Big losers avoided', '2026 delta']}
          rows={p3Rules}
          rowTone={['success', 'success', 'danger', 'danger']}
          columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
        />
      </Stack>

      <Stack gap={10}>
        <H2>PB3 Contrast</H2>
        <Table
          headers={['Rule', 'Skipped', 'Skipped PnL', 'Delta', 'Big winners killed', 'Big losers avoided', '2026 delta']}
          rows={pb3Rules}
          rowTone={['info', 'danger', 'danger']}
          columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
        />
      </Stack>

      <Callout tone="info" title="下一步">
        市场情绪对 P3 更像“过热/拥挤线索”，不是简单的冷市场过滤。严格删除整天信号已经被 Rust 否决；
        后续如果继续，只应尝试更温和的 soft penalty / rerank，或与 sector-tailwind 交互。PB3 的低涨停数过滤只有小幅 trade-level 正贡献，暂不优先升级。
      </Callout>
    </Stack>
  );
}
