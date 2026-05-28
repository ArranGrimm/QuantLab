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

const penaltyRows = [
  ['Raw P3', '-', '+201.69%', '1.22', '-13.52%', '-0.77%'],
  ['10d / bottom40 / 0.015', '+0.015', '+191.28%', '1.18', '-15.79%', '-1.86%'],
  ['10d / bottom40 / 0.018', '+0.018', '+227.55%', '1.28', '-13.44%', '-1.85%'],
  ['10d / bottom40 / 0.020', '+0.020', '+242.10%', '1.32', '-13.44%', '+0.63%'],
  ['10d / bottom40 / 0.022', '+0.022', '+239.74%', '1.31', '-13.47%', '+0.64%'],
  ['10d / bottom40 / 0.025', '+0.025', '+239.74%', '1.31', '-13.47%', '+0.64%'],
];

const thresholdRows = [
  ['10d / bottom30 / 0.020', '+226.99%', '+25.30pp', '1.31', '-13.52%', '+1.29%'],
  ['10d / bottom40 / 0.020', '+242.10%', '+40.41pp', '1.32', '-13.44%', '+0.63%'],
  ['10d / bottom50 / 0.020', '+138.70%', '-63.00pp', '0.96', '-20.65%', '+0.71%'],
];

const windowRows = [
  ['5d / bottom40 / 0.020', '+222.04%', '+20.35pp', '1.26', '-16.62%', '+1.56%'],
  ['10d / bottom40 / 0.020', '+242.10%', '+40.41pp', '1.32', '-13.44%', '+0.63%'],
  ['20d / bottom40 / 0.020', '+264.26%', '+62.57pp', '1.36', '-16.09%', '+0.41%'],
];

export default function AmvP3SectorTailwindRobustness() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>P3 Sector Tailwind Robustness</H1>
        <Text tone="secondary">
          对 P3 sector tailwind rerank 做 focused robustness：penalty 邻域、bottom threshold、行业收益 rank window。
          全部结果来自 <Code>bt-amv-topn</Code> static strict 6td。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="0.018-0.025" label="stable useful penalty band" tone="success" />
        <Stat value="+40.41pp" label="10d/b40/0.02 return delta" tone="success" />
        <Stat value="1.32" label="10d/b40/0.02 Sharpe" tone="success" />
        <Stat value="-20.65%" label="bottom50 over-penalty MaxDD" tone="danger" />
      </Grid>

      <Callout tone="success" title="稳健性判断">
        方向不是单点好运气：<Code>0.018</Code> 到 <Code>0.025</Code> 都明显优于 raw P3。
        但阈值不能过宽，bottom 50% 会把收益和回撤都打坏。当前最均衡仍是 <Code>10d / bottom40 / 0.02</Code>。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Penalty Locality: Total Return vs MaxDD</H2>
          <BarChart
            categories={['Raw', '0.015', '0.018', '0.020', '0.022', '0.025']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Total return', data: [201.69, 191.28, 227.55, 242.1, 239.74, 239.74], tone: 'success' },
              { name: 'MaxDD absolute', data: [13.52, 15.79, 13.44, 13.47, 13.47], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Fixed setting: 10d industry rank, bottom 40% bucket.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Window Sensitivity</H2>
          <BarChart
            categories={['5d', '10d', '20d']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Total return', data: [222.04, 242.1, 264.26], tone: 'success' },
              { name: 'MaxDD absolute', data: [16.62, 13.44, 16.09], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Fixed setting: bottom 40%, penalty 0.02. 20d is more aggressive but has deeper drawdown.
          </Text>
        </Stack>
      </Grid>

      <H2>Penalty Locality</H2>
      <Table
        headers={['Strategy', 'Penalty', 'Total', 'Sharpe', 'MaxDD', '2026']}
        rows={penaltyRows}
        rowTone={['info', 'warning', 'success', 'success', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
      />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Threshold Sensitivity</H2>
          <Table
            headers={['Setting', 'Total', 'Delta vs Raw', 'Sharpe', 'MaxDD', '2026']}
            rows={thresholdRows}
            rowTone={['success', 'success', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>Rank Window Sensitivity</H2>
          <Table
            headers={['Setting', 'Total', 'Delta vs Raw', 'Sharpe', 'MaxDD', '2026']}
            rows={windowRows}
            rowTone={['warning', 'success', 'info']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Callout tone="warning" title="仍未完成的验证">
        这轮验证覆盖了参数邻域、阈值、窗口，但还没有覆盖 static cadence offset，也没有更换历史行业分类源。
        因此 <Code>10d / bottom40 / 0.02</Code> 可以升级为强候选，但还不应直接定版。
      </Callout>

      <Text size="small" tone="tertiary">
        Data: <Code>reports/amv_p3_sector_tailwind_robustness.json</Code>. Source industry map is static East Money;
        this has historical classification bias.
      </Text>
    </Stack>
  );
}
