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
  ['5d', '接近20日新高', '单因子', '+1.427%', '+1.029pp', '+332.18%', '-38.06%'],
  ['10d', '高位+K线确认', '组合', '+1.551%', '+0.705pp', '+123.74%', '-4.57%'],
  ['20d', '高位+K线确认', '组合', '+2.296%', '+0.948pp', '+81.64%', '-2.42%'],
];

const top20Rows = [
  ['高位+K线确认', '+2.296%', '+0.948pp', '+81.64%', '-2.42%', '17.9%'],
  ['20日高位强势', '+2.242%', '+0.894pp', '+74.42%', '-5.76%', '24.3%'],
  ['新高+缩振+低风险', '+2.223%', '+0.875pp', '+75.86%', '-2.80%', '12.2%'],
  ['高位+实体强+低风险', '+2.217%', '+0.869pp', '+77.18%', '-3.81%', '17.2%'],
  ['接近20日新高', '+2.091%', '+0.742pp', '+58.18%', '-14.66%', '38.2%'],
  ['实体占比偏强', '+1.969%', '+0.620pp', '+57.86%', '-16.87%', '35.5%'],
];

const comboDefinitions = [
  ['高位+K线确认', 'price_pos_20d↑, close_to_high_20d↓, KLEN↓, KMID2↑'],
  ['新高+缩振+低风险', 'close_to_high_20d↓, KLEN↓, atr_14_pct↓, panic_vol_ratio_20d↓'],
  ['高位+实体强+低风险', 'price_pos_20d↑, KMID2↑, atr_14_pct↓, panic_vol_ratio_20d↓'],
  ['新高+实体强+短上影', 'close_to_high_20d↓, KMID2↑, KUP2↓, panic_vol_ratio_20d↓'],
  ['高位+收盘强+低波', 'price_pos_20d↑, KSFT2↑, vol_20d↓, atr_14_pct↓'],
];

export default function AMVBullPoolComboRanker() {
  return (
    <Stack gap={18}>
      <Stack gap={6}>
        <H1>AMV 多头宽池组合排序</H1>
        <Text tone="secondary">
          在 <Code>mechanical AMV bull + LF2 宽池</Code> 内，每天用组合分位评分选 top5。
          组合分数把价格位置、Alpha158 K线形态、风险/卖压因子转成每日截面分位后加权平均。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+2.296%" label="最佳 20d 单笔均值" tone="success" />
        <Stat value="+0.948pp" label="相对随机 20d 超额" tone="success" />
        <Stat value="+81.64%" label="最佳 20d 错峰滚动净值" tone="success" />
        <Stat value="-2.42%" label="最佳 20d 最大回撤" tone="info" />
      </Grid>

      <Callout tone="success" title="组合实验结论">
        20d 维度已经从单因子 <Code>20日高位强势</Code> 升级到组合 <Code>高位+K线确认</Code>。
        收益从 +2.242% 提升到 +2.296%，最大回撤从 -5.76% 降到 -2.42%。
      </Callout>

      <Divider />

      <H2>各周期冠军</H2>
      <Table
        headers={['周期', '最佳排序', '类型', '单笔均值', '相对随机', '滚动净值', '最大回撤']}
        rows={horizonRows}
        rowTone={['success', 'success', 'success']}
        columnAlign={['left', 'left', 'left', 'right', 'right', 'right', 'right']}
      />

      <H2>20d Top 信号对比</H2>
      <BarChart
        categories={[
          '高位+K线确认',
          '20日高位强势',
          '新高+缩振+低风险',
          '高位+实体强+低风险',
          '接近20日新高',
        ]}
        valueSuffix="%"
        height={260}
        series={[
          { name: '20d 单笔均值', data: [2.296, 2.242, 2.223, 2.217, 2.091], tone: 'success' },
          { name: '随机基准', data: [1.348, 1.348, 1.348, 1.348, 1.348], tone: 'neutral' },
        ]}
      />

      <Table
        headers={['排序信号', '20d 单笔均值', '相对随机', '滚动净值', '最大回撤', '触及 +15%']}
        rows={top20Rows}
        rowTone={['success', 'info', 'success', 'success', 'info', undefined]}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>组合定义</H2>
      <Table
        headers={['组合', '分位评分组件']}
        rows={comboDefinitions}
        columnAlign={['left', 'left']}
      />

      <H2>怎么读</H2>
      <Text>
        组合不是为了追求更高的 +15% 触发率，而是把收益曲线变得更平滑。
        <Code>高位+K线确认</Code> 的 20d 触及 +15% 比 <Code>接近20日新高</Code> 低，
        但滚动净值更高、回撤明显更浅。
      </Text>
      <Text>
        下一步可以围绕这三个组合做参数网格：价格位置权重、K线确认权重、风险惩罚权重，以及 top3/top5/top10 对比。
      </Text>

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_bull_pool_ranker_lab/20260503_165924/summary.json</Code>.
        口径: AMV bull + LF2, 每日 top5, 持仓 5/10/20 天。
      </Text>
    </Stack>
  );
}
