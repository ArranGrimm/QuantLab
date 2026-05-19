// @ts-nocheck
import { Callout, Divider, Grid, H1, H2, H3, Stack, Stat, Table, Text } from 'cursor/canvas';

const backtestRows = [
  ['1td oracle', '+15,806,636%', '+16,785,773%', '-31.76%', '825', '73.6%', '+4.59%'],
  ['2td oracle', '+891,251%', '+945,000%', '-37.37%', '571', '70.8%', '+5.12%'],
  ['3td oracle', '+639,900%', '+671,457%', '-41.78%', '436', '71.8%', '+6.49%'],
  ['规则基线 6td no stop', '+144%', '+195%', '-14.71%', '264', '52.3%', 'N/A'],
];

const annualRows = [
  ['1td', '+746.8%', '+489.3%', '+1078.3%', '+899.7%', '+755.2%', '+206.2%'],
  ['2td', '+446.7%', '+357.5%', '+446.9%', '+483.5%', '+436.7%', '+106.3%'],
  ['3td', '+552.6%', '+179.2%', '+487.7%', '+443.4%', '+459.8%', '+95.5%'],
];

const choiceRows = [
  ['1td', 'ret_20d', '213', '+6.73%'],
  ['1td', 'ret_5d', '181', '+7.91%'],
  ['1td', 'kmid2', '126', '+4.06%'],
  ['2td', 'ret_20d', '196', '+8.32%'],
  ['2td', 'ret_5d', '188', '+9.30%'],
  ['2td', 'kmid2', '130', '+5.30%'],
  ['3td', 'ret_5d', '191', '+10.51%'],
  ['3td', 'ret_20d', '184', '+9.92%'],
  ['3td', 'kmid2', '141', '+6.41%'],
];

export default function AmvOracleSleeveRustBacktest() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>AMV Oracle Sleeve Rust Backtest</H1>
        <Text tone="secondary">
          Source: artifacts/amv_oracle_sleeve_signals · bt-amv-topn · T+1 open execution · no stop.
          This is a hindsight oracle using future returns.
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="+15.8M%" label="1td oracle net return" tone="success" />
        <Stat value="73.6%" label="1td win rate" tone="success" />
        <Stat value="-31.8%" label="1td max drawdown" tone="warning" />
        <Stat value="not tradable" label="Hindsight oracle" tone="danger" />
      </Grid>

      <Callout tone="danger" title="Interpretation">
        <Text>
          这个结果不是策略收益, 而是可交易约束下的事后天花板。它说明“因子袖子切换”有巨大空间,
          但我们仍需要找到能在交易前预测 sleeve 的状态变量或规则。
        </Text>
      </Callout>

      <H2>Rust Backtest Metrics</H2>
      <Table
        headers={['Signal', 'Net return', 'Gross return', 'MaxDD', 'Trades', 'Win rate', 'Avg trade PnL']}
        rows={backtestRows}
      />

      <Divider />

      <Grid columns={2} gap={18}>
        <Stack gap={10}>
          <H3>Annual Returns</H3>
          <Table
            headers={['Horizon', '2021', '2022', '2023', '2024', '2025', '2026 YTD']}
            rows={annualRows}
          />
        </Stack>
        <Stack gap={10}>
          <H3>Most Chosen Sleeves</H3>
          <Text tone="secondary">
            Days and average oracle choice return before T+1 shift. Only the top three sleeves per horizon are shown.
          </Text>
          <Table headers={['Horizon', 'Sleeve', 'Days', 'Mean choice return']} rows={choiceRows} />
        </Stack>
      </Grid>
    </Stack>
  );
}
