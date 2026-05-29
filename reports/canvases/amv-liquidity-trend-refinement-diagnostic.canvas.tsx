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
  ['Drawdown128 quality high', '182', '+1,064.0K', '+1.97%', '54.9%'],
  ['Drawdown128 quality low', '17', '-36.3K', '-0.63%', '29.4%'],
  ['MA stack high', '65', '+706.7K', '+3.68%', '58.5%'],
  ['MA stack low', '98', '+119.7K', '+0.34%', '45.9%'],
  ['Amount 20/60 high', '75', '+551.0K', '+2.44%', '53.3%'],
  ['Amount 20/60 low', '78', '+41.1K', '+0.40%', '53.8%'],
];

const p3Events = [
  ['Amount expansion 5d', '92', '+618.9K', '+2.26%', '48.9%'],
  ['Breakout volume confirmed', '102', '+643.5K', '+2.04%', '53.9%'],
  ['Liquidity recovery', '57', '+360.1K', '+2.28%', '57.9%'],
  ['Dry pullback 5d', '0', '+0.0K', '+0.00%', '0.0%'],
];

const p3Rules = [
  ['Skip deep drawdown 128', '18', '-28.7K', '+28.7K', '+17.9K', '-42.9K', 'weak positive'],
  ['Skip breakout without volume', '23', '+29.2K', '-29.2K', '+21.1K', '-44.7K', 'reject'],
  ['Skip low medium liquidity', '79', '+48.7K', '-48.7K', '-46.9K', '-6.2K', 'reject'],
  ['Skip unconfirmed breakout', '172', '+365.0K', '-365.0K', '+6.3K', '-98.9K', 'reject'],
  ['Skip liquidity recovery', '57', '+360.1K', '-360.1K', '-112.6K', '-46.6K', 'reject'],
];

const pb3Buckets = [
  ['MA stack high', '767', '+301.9K', '+1.22%', '49.3%'],
  ['MA stack mid', '500', '+230.3K', '+1.96%', '52.2%'],
  ['MA stack low', '383', '-34.0K', '-0.56%', '40.2%'],
  ['Dry pullback true', '549', '+240.6K', '+1.31%', '45.7%'],
  ['Amount expansion true', '326', '+115.6K', '+1.40%', '54.0%'],
  ['Volume no price true', '299', '+49.3K', '+0.69%', '50.2%'],
];

const pb3Rules = [
  ['Skip unstable MA stack', '386', '-35.5K', '+35.5K', '-0.4K', '-50.3K', 'not robust enough'],
  ['Skip breakout without volume', '2', '-4.8K', '+4.8K', '+0.0K', '+0.0K', 'too small'],
  ['Skip amount spike no price', '299', '+49.3K', '-49.3K', '+2.4K', '-21.7K', 'reject'],
  ['Skip low medium liquidity', '395', '+115.3K', '-115.3K', '-35.0K', '-57.0K', 'reject'],
  ['Keep dry pullback only', '549 kept', '+240.6K', '-257.5K', '-40.1K', '-94.2K', 'reject as gate'],
];

export default function AmvLiquidityTrendRefinementDiagnostic() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Liquidity / Trend Refinement Diagnostic</H1>
        <Text tone="secondary">
          第二阶段剩余项收口：补做趋势质量细分和流动性 / 成交额异动。所有特征按 <Code>signal_date</Code> 拼到已成交交易，
          本页是 trade-level diagnostic，不是 Rust account backtest。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+28.7K" label="Best P3 skip delta" tone="warning" />
        <Stat value="+35.5K" label="Best PB3 skip delta" tone="warning" />
        <Stat value="0" label="New Rust candidates" tone="info" />
        <Stat value="Done" label="Stage 2 remaining checks" tone="success" />
      </Grid>

      <Callout tone="warning" title="收口结论">
        P3 的成交额扩张 / 放量确认 / 流动性恢复是正向事件，不适合 hard skip；唯一正向 skip 是 128 日深回撤过滤，
        但只有 <Code>18</Code> 笔、delta <Code>+28.7K</Code>，且 2025 反向。PB3 的低均线排列稳定性过滤 delta
        <Code>+35.5K</Code>，但 2025 反向 <Code>-50.3K</Code>。因此本轮不接 signal export / Rust。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 Event PnL</H2>
          <Table
            headers={['Event', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
            rows={p3Events}
            rowTone={['success', 'success', 'success', 'info']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_liquidity_trend_refinement_diagnostic.json</Code>; P3 static strict trades, 2019-2026.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>P3 Bucket Contrast</H2>
          <BarChart
            categories={['DD high', 'DD low', 'MA high', 'MA low', 'Amt20/60 high', 'Amt20/60 low']}
            valueSuffix="K"
            height={260}
            series={[
              { name: 'Total PnL', data: [1064.0, -36.3, 706.7, 119.7, 551.0, 41.1], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            P3 的好交易集中在强 MA stack 和成交额中期扩张，但这些更像确认条件，不像可独立删除信号的硬规则。
          </Text>
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>P3 Candidate Skip Rules</H2>
        <Table
          headers={['Rule', 'Skipped', 'Skipped PnL', 'Delta', '2026 delta', '2025 delta', 'Decision']}
          rows={p3Rules}
          rowTone={['warning', 'danger', 'danger', 'danger', 'danger']}
          columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'left']}
        />
      </Stack>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>PB3 Bucket / Event Contrast</H2>
          <Table
            headers={['Bucket / Event', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
            rows={pb3Buckets}
            rowTone={['success', 'success', 'danger', 'success', 'success', 'info']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>PB3 Candidate Skip Rules</H2>
          <Table
            headers={['Rule', 'Skipped', 'Skipped PnL', 'Delta', '2026 delta', '2025 delta', 'Decision']}
            rows={pb3Rules}
            rowTone={['warning', 'info', 'danger', 'danger', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'left']}
          />
        </Stack>
      </Grid>

      <Callout tone="info" title="第二阶段剩余项定义">
        已补齐：回撤深度、显式趋势线性度、均线排列稳定性、成交额 1/20、5/20、20/60 相对扩张、放量不涨、缩量回调、
        突破日量能确认、流动性枯竭后恢复。当前最强 P3 challenger 仍是
        <Code>sector mix10/20 p0.02 + 128d structure/quality p0.03</Code>，本轮新增细分不叠加。
      </Callout>
    </Stack>
  );
}
