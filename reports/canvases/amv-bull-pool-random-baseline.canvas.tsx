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

const poolRows = [
  ['LF2 全部交易日', '1,250', '1,117', '+0.124%', '+0.251%', '+0.520%', '+20.90%', '-34.70%'],
  ['LF2 AMV 多头日', '542', '1,104', '+0.399%', '+0.842%', '+1.337%', '+38.27%', '-19.08%'],
  ['LF2 非多头日', '708', '1,127', '-0.088%', '-0.206%', '-0.121%', '-10.62%', '-24.15%'],
];

const stopRows = [
  ['LF2 全部交易日', '+0.274%', '+16.31%', '-16.41%'],
  ['LF2 AMV 多头日', '+0.616%', '+16.42%', '-9.87%'],
  ['LF2 非多头日', '-0.000%', '-1.67%', '-10.23%'],
];

export default function AMVBullPoolRandomBaseline() {
  return (
    <Stack gap={18}>
      <Stack gap={6}>
        <H1>AMV 多头宽池随机基准</H1>
        <Text tone="secondary">
          机械 AMV 状态定义: <Code>bull=max(ret_1d, ret_2d) &gt;= 4.0%</Code>,{' '}
          <Code>bear=ret_1d &lt;= -2.3%</Code>, 生效延迟 1 个交易日。股票池为 LF2:
          流通市值 &gt;= 100 亿, 20 日均成交额 &gt;= 5000 万。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+1.337%" label="AMV 多头随机 5 只 20d 单笔均值" tone="success" />
        <Stat value="-0.121%" label="非多头随机 5 只 20d 单笔均值" tone="danger" />
        <Stat value="+38.27%" label="AMV 多头 20d 错峰滚动净值" tone="success" />
        <Stat value="-19.08%" label="AMV 多头 20d 错峰滚动回撤" tone="warning" />
      </Grid>

      <Callout tone="success" title="核心结论">
        AMV 多头作为市场 gate 是成立的。只用 LF2 宽池、每天随机选 5 只，在多头日是正收益，
        在非多头日接近负收益。下一步的关键不再是证明 AMV 有没有择时价值，而是能否降低回撤，
        并找到能打败“随机 5 只”下限的池内排序信号。
      </Callout>

      <Divider />

      <H2>不同持仓周期的单笔平均收益</H2>
      <BarChart
        categories={['5d', '10d', '20d']}
        valueSuffix="%"
        height={260}
        series={[
          { name: 'LF2 全部交易日', data: [0.124, 0.251, 0.520], tone: 'neutral' },
          { name: 'LF2 AMV 多头日', data: [0.399, 0.842, 1.337], tone: 'success' },
          { name: 'LF2 非多头日', data: [-0.088, -0.206, -0.121], tone: 'danger' },
        ]}
      />

      <H2>Monte Carlo 汇总</H2>
      <Table
        headers={[
          '股票池',
          '可交易天数',
          '每日候选中位数',
          '5d 单笔均值',
          '10d 单笔均值',
          '20d 单笔均值',
          '20d 错峰滚动净值',
          '20d 最大回撤',
        ]}
        rows={poolRows}
        rowTone={[undefined, 'success', 'danger']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>3% 止损版本</H2>
      <Text tone="secondary">
        止损能降低回撤，但也明显削弱收益。第一轮结果里，最干净的信号仍然是“不止损”的
        10d / 20d AMV 多头宽池优势。
      </Text>
      <Table
        headers={['股票池', '20d 止损后单笔均值', '20d 止损后滚动净值', '20d 止损后最大回撤']}
        rows={stopRows}
        rowTone={[undefined, 'success', 'warning']}
        columnAlign={['left', 'right', 'right', 'right']}
      />

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_bull_pool_random_baseline/20260503_125323/summary.json</Code>.
        参数: 1,000 次 Monte Carlo, 每个可交易日随机选 5 只, seed 42。
      </Text>
    </Stack>
  );
}
