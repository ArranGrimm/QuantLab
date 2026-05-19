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

const horizonRows = [
  ['5d', '接近20日新高', '+1.427%', '+1.029pp', '+332.18%', '-38.06%'],
  ['10d', '接近20日新高', '+1.665%', '+0.819pp', '+129.28%', '-23.26%'],
  ['20d', '20日高位强势', '+2.242%', '+0.894pp', '+74.42%', '-5.76%'],
];

const top20Rows = [
  ['20日高位强势', '+2.242%', '+0.894pp', '+74.42%', '-5.76%', '24.3%'],
  ['接近20日新高', '+2.091%', '+0.742pp', '+58.18%', '-14.66%', '38.2%'],
  ['收盘靠低', '+1.670%', '+0.321pp', '+52.07%', '-9.98%', '22.2%'],
  ['20日反转', '+1.669%', '+0.321pp', '+42.60%', '-22.16%', '31.4%'],
  ['成本线下回归', '+1.627%', '+0.279pp', '+41.47%', '-19.81%', '31.0%'],
];

export default function AMVBullPoolRankerLab() {
  return (
    <Stack gap={18}>
      <Stack gap={6}>
        <H1>AMV 多头宽池排序实验</H1>
        <Text tone="secondary">
          在 <Code>mechanical AMV bull + LF2 宽池</Code> 内，每天按单因子排序选 top5。
          对照基准是同一天宽池“随机 5 只”的日期等权期望。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+2.242%" label="最佳 20d 单笔均值" tone="success" />
        <Stat value="+0.894pp" label="相对随机 20d 超额" tone="success" />
        <Stat value="+74.42%" label="最佳 20d 错峰滚动净值" tone="success" />
        <Stat value="-5.76%" label="最佳 20d 最大回撤" tone="info" />
      </Grid>

      <Callout tone="success" title="第一轮结论">
        单因子排序能打败随机 5 只。最有价值的方向不是 B1 形态规则，而是“AMV 多头宽池里的价格位置”：
        接近 20 日新高、20 日高位强势、以及日内收盘靠低这几类信号最值得继续挖。
      </Callout>

      <Divider />

      <H2>各周期冠军</H2>
      <Table
        headers={['周期', '最佳排序', '单笔均值', '相对随机', '滚动净值', '最大回撤']}
        rows={horizonRows}
        rowTone={['success', 'success', 'success']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right']}
      />

      <H2>20d Top 排序信号</H2>
      <BarChart
        categories={['20日高位强势', '接近20日新高', '收盘靠低', '20日反转', '成本线下回归']}
        valueSuffix="%"
        height={260}
        series={[
          { name: '20d 单笔均值', data: [2.242, 2.091, 1.670, 1.669, 1.627], tone: 'success' },
          { name: '随机基准', data: [1.348, 1.348, 1.348, 1.348, 1.348], tone: 'neutral' },
        ]}
      />

      <Table
        headers={['排序信号', '20d 单笔均值', '相对随机', '滚动净值', '最大回撤', '触及 +15%']}
        rows={top20Rows}
        rowTone={['success', 'success', 'info', 'info', 'info']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>怎么读</H2>
      <Text>
        <Code>接近20日新高</Code> 在 5d/10d 最强，但 5d 回撤很深，说明短周期追强会很刺激。
        <Code>20日高位强势</Code> 在 20d 最干净：收益高、超额大、回撤显著低。
      </Text>
      <Text>
        下一步应该围绕价格位置做二阶验证：把 <Code>20日高位强势</Code> 和
        <Code>接近20日新高</Code> 合并成候选 score，再加一个风险过滤，目标是保留 20d 收益同时压低回撤。
      </Text>

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_bull_pool_ranker_lab/20260503_130223/summary.json</Code>.
        口径: AMV bull + LF2, 每日 top5, 持仓 5/10/20 天。
      </Text>
    </Stack>
  );
}
