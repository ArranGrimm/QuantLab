import {
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  LineChart,
  Stack,
  Stat,
  Table,
  Text,
} from 'cursor/canvas';

const horizons = ['1d', '2d', '3d', '5d', '10d', '15d', '20d', '30d'];

const curveRows = [
  ['1d', '+0.413%', '+0.340pp', '+0.413%', 'n/a', '+746.31%', '-21.45%'],
  ['2d', '+0.588%', '+0.451pp', '+0.294%', '+0.175%', '+350.19%', '-13.22%'],
  ['3d', '+0.833%', '+0.617pp', '+0.278%', '+0.245%', '+299.67%', '-7.22%'],
  ['5d', '+1.304%', '+0.906pp', '+0.261%', '+0.471%', '+287.31%', '-4.05%'],
  ['10d', '+1.789%', '+0.943pp', '+0.179%', '+0.485%', '+153.04%', '-4.27%'],
  ['15d', '+2.228%', '+1.068pp', '+0.149%', '+0.438%', '+115.35%', '-3.56%'],
  ['20d', '+2.625%', '+1.276pp', '+0.131%', '+0.397%', '+102.97%', '-2.90%'],
  ['30d', '+3.173%', '+1.581pp', '+0.106%', '+0.549%', '+72.07%', '-2.46%'],
];

export default function AMVBullPoolHorizonCurve() {
  return (
    <Stack gap={18}>
      <Stack gap={6}>
        <H1>AMV Top3 持有期兑现曲线</H1>
        <Text tone="secondary">
          固定信号 <Code>top3 高位+K线确认 P2/K0.5/R0</Code>，只改变观察窗口。
          目的不是重新调参，而是看收益在 1d 到 30d 之间如何累积。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+3.173%" label="30d 单笔均值最高" tone="success" />
        <Stat value="+0.413%" label="1d 单日效率最高" tone="info" />
        <Stat value="5-10d" label="更适合交易候选" tone="success" />
        <Stat value="20-30d" label="更像研究观察窗" tone="info" />
      </Grid>

      <Callout tone="warning" title="最重要的读法">
        绝对收益到 30d 仍在增加，说明没有明显 20d 前回吐；但单位时间收益一路下降，
        说明越往后资金效率越低。真实回测不应只押 20d。
      </Callout>

      <Divider />

      <H2>收益与效率</H2>
      <LineChart
        categories={horizons}
        valueSuffix="%"
        height={280}
        series={[
          { name: '单笔均值', data: [0.413, 0.588, 0.833, 1.304, 1.789, 2.228, 2.625, 3.173], tone: 'success' },
          { name: '单位时间收益', data: [0.413, 0.294, 0.278, 0.261, 0.179, 0.149, 0.131, 0.106], tone: 'info' },
        ]}
      />

      <H2>完整曲线</H2>
      <Table
        headers={['窗口', '单笔均值', '相对随机', '日均收益', '边际新增', '滚动净值', '最大回撤']}
        rows={curveRows}
        rowTone={['warning', 'info', 'info', 'success', 'success', 'info', 'info', undefined]}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>怎么解读</H2>
      <Text>
        <Code>1d</Code> 的滚动净值很高，但最大回撤也深，而且真实交易成本会很重；
        它说明信号短期有效，不等于应该日内式高频换仓。
      </Text>
      <Text>
        <Code>5d/10d</Code> 是更值得先做真实回测的区间：收益已经兑现一大段，
        持仓时间没有太长，资金效率明显高于 20d/30d。
      </Text>
      <Text>
        <Code>20d/30d</Code> 更适合当研究标签或上限观察窗；如果真实策略持到这么久，
        需要加入 AMV 转空、排名跌出、止损或止盈退出，否则资金占用可能不划算。
      </Text>

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_bull_pool_horizon_curve/20260503_215103/summary.json</Code>.
        口径: AMV bull + LF2, 每日 top3, 信号固定为 P2/K0.5/R0。
      </Text>
    </Stack>
  );
}
