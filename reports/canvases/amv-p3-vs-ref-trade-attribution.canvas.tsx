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

const yearlyRows = [
  ['2021', '+10.37%', '+8.61%', '-1.76pp'],
  ['2022', '+39.70%', '+40.45%', '+0.75pp'],
  ['2023', '+14.31%', '+19.51%', '+5.20pp'],
  ['2024', '+49.66%', '+46.43%', '-3.23pp'],
  ['2025', '+12.57%', '+13.89%', '+1.32pp'],
  ['2026', '-8.80%', '-0.77%', '+8.03pp'],
];

const topMonthRows = [
  ['2026-01', '-2.08%', '+7.14%', '+9.23pp', '+25.85pp by trade PnL'],
  ['2023-04', '+1.89%', '+11.72%', '+9.83pp', '+14.64pp by trade PnL'],
  ['2021-04', '-3.60%', '+3.07%', '+6.67pp', '+6.95pp by trade PnL'],
  ['2024-09', '+17.49%', '+20.61%', '+3.12pp', '+4.70pp by trade PnL'],
  ['2021-05', '-0.62%', '+1.70%', '+2.32pp', '+3.24pp by trade PnL'],
  ['2023-12', '-0.28%', '+1.77%', '+2.06pp', 'positive replacement month'],
];

const weakMonthRows = [
  ['2021-01', '+9.16%', '-2.10%', '-11.27pp', '-11.90pp by trade PnL'],
  ['2023-03', '+5.60%', '+1.83%', '-3.78pp', '-5.74pp by trade PnL'],
  ['2024-11', '+5.19%', '+3.44%', '-1.75pp', '-11.29pp by trade PnL'],
  ['2024-05', '-0.19%', '-1.90%', '-1.71pp', '-3.26pp by trade PnL'],
  ['2024-10', '+27.71%', '+26.08%', '-1.63pp', '+7.17pp by trade PnL'],
  ['2023-08', '-1.80%', '-3.01%', '-1.21pp', '-2.25pp by trade PnL'],
];

const replacementRows = [
  ['P3 unique pnl', '+170,899', '30 trades only in P3'],
  ['Ref unique pnl', '+29,737', '30 trades only in Ref'],
  ['Replacement delta', '+141,161', '91.4% of total pnl delta'],
  ['Common P3 pnl', '+837,561', '244 overlapping entry-date/code trades'],
  ['Common Ref pnl', '+824,275', 'same overlapping trades in Ref'],
  ['Common delta', '+13,286', 'from different sizing/equity path'],
];

const p3UniqueWinners = [
  ['sz.003035', '2026-01-16', '2026-01-26', '+92,771', '+18.19%'],
  ['sh.600667', '2023-03-31', '2023-04-11', '+53,322', '+20.76%'],
  ['sh.601127', '2024-11-05', '2024-11-13', '+50,207', '+11.57%'],
  ['sh.601127', '2021-04-21', '2021-04-29', '+35,937', '+21.93%'],
  ['sh.601688', '2024-09-12', '2024-09-24', '+16,828', '+5.83%'],
  ['sh.600795', '2023-12-06', '2023-12-14', '+16,204', '+5.39%'],
];

const removedRefRows = [
  ['sh.688789', '2026-01-16', '2026-01-26', '-35,846', '-7.19%', 'avoided loss'],
  ['sz.300919', '2023-03-31', '2023-04-11', '-19,365', '-7.14%', 'avoided loss'],
  ['sz.000559', '2024-10-31', '2024-11-08', '+83,373', '+21.18%', 'missed winner'],
  ['sz.002271', '2021-01-04', '2021-01-12', '+51,608', '+31.25%', 'missed winner'],
  ['sh.600085', '2024-11-11', '2024-11-19', '-10,526', '-2.21%', 'avoided loss'],
  ['sh.600511', '2026-01-27', '2026-02-04', '-8,671', '-1.77%', 'avoided loss'],
];

const topCodeRows = [
  ['sh.600839', '+203,967', '1', '+195,261', '1'],
  ['sh.603667', '+97,216', '1', '+95,496', '1'],
  ['sz.003035', '+92,771', '1', '-', '-'],
  ['sh.600221', '+92,363', '1', '+88,308', '1'],
  ['sh.601127', '+86,144', '2', '-', '-'],
  ['sh.601609', '+71,518', '1', '+69,705', '1'],
];

export default function AmvP3VsRefTradeAttribution() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV P3/K0.5 vs Ref P2/K0.5: 主基线替换归因</H1>
        <Text tone="secondary">
          对比 <Code>bt-amv-topn</Code> 静态 strict Top3 / 6td / no-stop 真实回测。
          目标是判断 <Code>P3/K0.5/R0</Code> 是否足以替换当前 reference。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+201.69%" label="P3 total return" tone="success" />
        <Stat value="+30.89pp" label="P3 over Ref" tone="success" />
        <Stat value="-13.52%" label="P3 MaxDD" tone="success" />
        <Stat value="89.1%" label="trade overlap" tone="info" />
      </Grid>

      <Callout tone="warning" title="结论">
        P3 确实优于 Ref，但优势主要来自 30 笔替换交易，而不是整体交易池大幅变化。
        其中 2026-01 和 2023-04 是最关键的正贡献月份。它是强替换候选，但切换前应继续确认这些替换票不是偶然单点贡献。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>年度收益对比</H2>
          <BarChart
            categories={['2021', '2022', '2023', '2024', '2025', '2026']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Ref P2/K0.5', data: [10.37, 39.7, 14.31, 49.66, 12.57, -8.8], tone: 'info' },
              { name: 'P3/K0.5', data: [8.61, 40.45, 19.51, 46.43, 13.89, -0.77], tone: 'success' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>daily_equity.csv</Code>, calendar-year end equity over previous year-end equity.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>关键指标</H2>
          <BarChart
            categories={['Total return', 'MaxDD abs', 'Win rate', 'Costs / initial', 'Limit-up blocked']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Ref P2/K0.5', data: [170.8, 15.3, 51.09, 58.22, 48], tone: 'info' },
              { name: 'P3/K0.5', data: [201.69, 13.52, 52.55, 59.17, 60], tone: 'success' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Limit-up blocked is count, shown on same chart only for compact comparison.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>年度收益明细</H2>
      <Table
        headers={['Year', 'Ref P2/K0.5', 'P3/K0.5', 'Delta']}
        rows={yearlyRows}
        rowTone={['warning', 'success', 'success', 'warning', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right']}
      />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 优势月份</H2>
          <Table
            headers={['Month', 'Ref', 'P3', 'Equity Delta', 'Trade PnL Delta']}
            rows={topMonthRows}
            rowTone={['success', 'success', 'success', 'success', 'success', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'left']}
          />
        </Stack>
        <Stack gap={10}>
          <H2>P3 弱于 Ref 的月份</H2>
          <Table
            headers={['Month', 'Ref', 'P3', 'Equity Delta', 'Trade PnL Delta']}
            rows={weakMonthRows}
            rowTone={['danger', 'danger', 'warning', 'warning', 'warning', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'left']}
          />
        </Stack>
      </Grid>

      <Divider />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>交易替换贡献</H2>
          <Table
            headers={['Bucket', 'PnL', 'Meaning']}
            rows={replacementRows}
            rowTone={['success', 'warning', 'success', 'info', 'info', 'success']}
            columnAlign={['left', 'right', 'left']}
          />
          <Text size="small" tone="tertiary">
            Exact overlap key: <Code>entry_date + code</Code>. Both strategies have 274 trades.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>换票结构</H2>
          <BarChart
            categories={['Overlap trades', 'P3-only trades', 'Ref-only trades', 'Code overlap']}
            valueSuffix="%"
            height={250}
            series={[
              { name: 'Share', data: [89.05, 10.95, 10.95, 92.34], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            P3 不是重写策略，主要是少量排序边际换票。
          </Text>
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3-only 关键赢家</H2>
          <Table
            headers={['Code', 'Entry', 'Exit', 'PnL', 'Return']}
            rows={p3UniqueWinners}
            rowTone={['success', 'success', 'success', 'success', 'success', 'success']}
            columnAlign={['left', 'left', 'left', 'right', 'right']}
          />
        </Stack>
        <Stack gap={10}>
          <H2>Ref-only 被替换交易</H2>
          <Table
            headers={['Code', 'Entry', 'Exit', 'PnL', 'Return', 'Effect']}
            rows={removedRefRows}
            rowTone={['success', 'success', 'danger', 'danger', 'success', 'success']}
            columnAlign={['left', 'left', 'left', 'right', 'right', 'left']}
          />
        </Stack>
      </Grid>

      <H2>盈利代码集中度</H2>
      <Table
        headers={['Code', 'P3 PnL', 'P3 trades', 'Ref PnL', 'Ref trades']}
        rows={topCodeRows}
        rowTone={['success', 'success', 'success', 'success', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right', 'right']}
      />

      <Callout tone="info" title="决策建议">
        P3 可以进入“准新 reference”状态，但还不建议一句话定版。下一步应检查 2026-01 的换票机制，
        特别是 <Code>sz.003035</Code> 替代 <Code>sh.688789</Code> 这一组是否代表可重复的排序改善。
        如果这个机制解释得通，P3 就可以正式替换 P2。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        Data artifact: <Code>reports/amv_p3_vs_ref_trade_attribution.json</Code>.
        Ref report: <Code>20260520_092047_reference_p2_k0p5_b0_c0_r0</Code>.
        P3 report: <Code>20260520_092049_candidate_p3_k0p5_b0_c0_r0</Code>.
      </Text>
    </Stack>
  );
}
