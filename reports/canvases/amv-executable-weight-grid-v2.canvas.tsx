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

const focusRows = [
  ['P3/K0.5/R0', 'original Top3', '+160.14%', '+278.93%', '20.2%', '9.0%', '可执行口径首选候选'],
  ['P2/K0.5/R0', 'original Top3', '+152.21%', '+248.54%', '16.8%', '7.6%', '当前 reference'],
  ['P1/K0.5/M1', 'original Top3', '+32.87%', '+1054.55%', '73.5%', '28.0%', '动量污染重'],
  ['P2/K0.5/M0.5', 'original Top3', '+29.14%', '+768.74%', '59.7%', '26.4%', '动量污染重'],
  ['P3/K1/M2', 'original Top3', '+31.62%', '+1045.97%', '76.4%', '28.2%', '动量污染重'],
  ['single ret_5d', 'original Top3', '+264.07%', '+365.43%', '69.0%', '14.1%', '高收益但事件票属性强'],
];

const refillRows = [
  ['P3/K0.5/R0', '+151.90%', '+117.44%', '0.0%', '0.2%', 'q95=4 / max=258'],
  ['P2/K0.5/R0', '+114.12%', '+90.33%', '0.0%', '0.2%', 'q95=4 / max=258'],
  ['P1/K0.5/M1', '+90.64%', '+55.71%', '0.0%', '2.0%', 'q95=16 / max=140'],
  ['P2/K0.5/M0.5', '+66.44%', '+53.72%', '0.0%', '1.4%', 'q95=14 / max=244'],
  ['P3/K1/M2', '+57.72%', '+31.32%', '0.0%', '2.0%', 'q95=17 / max=170'],
  ['single ret_5d', '+165.93%', '+0.25%', '0.0%', '1.1%', 'q95=7 / max=21'],
];

const topOriginalRows = [
  ['single_ret_5d', '+264.07%', '-47.11%', '+365.43%', '69.0%', '14.1%'],
  ['P3/K0.5/R0', '+160.14%', '-5.58%', '+278.93%', '20.2%', '9.0%'],
  ['P2/K0.5/R0', '+152.21%', '-5.27%', '+248.54%', '16.8%', '7.6%'],
  ['P1/K0.5/R0', '+140.77%', '-5.07%', '+191.25%', '13.7%', '5.4%'],
  ['P3/K1/R0', '+138.84%', '-5.13%', '+200.67%', '15.2%', '6.5%'],
  ['P3/K0.5/R1.5', '+140.21%', '-8.42%', '+128.10%', '4.3%', '1.8%'],
  ['P2/K0.5/R1.5', '+136.11%', '-9.27%', '+112.98%', '3.2%', '1.1%'],
  ['P3/K1/R1', '+128.63%', '-7.09%', '+129.40%', '5.8%', '2.0%'],
];

const topRefillRows = [
  ['P3/K0.5/R0', '+151.90%', '-7.23%', '+117.44%', '0.0%', '0.2%'],
  ['single_ret_5d', '+165.93%', '-30.93%', '+0.25%', '0.0%', '1.1%'],
  ['P3/K0.5/R1.5', '+138.69%', '-10.38%', '+115.35%', '0.0%', '0.4%'],
  ['P2/K0.5/R1.5', '+133.21%', '-9.86%', '+104.09%', '0.0%', '0.2%'],
  ['P3/K1/M0.5', '+125.64%', '-6.89%', '+111.46%', '0.0%', '0.7%'],
  ['P1/K0.5/R0', '+121.80%', '-5.58%', '+101.77%', '0.0%', '0.2%'],
  ['P3/K1/R1', '+121.11%', '-6.62%', '+102.09%', '0.0%', '0.2%'],
  ['price_pos_20d', '+125.76%', '-13.58%', '+84.47%', '0.0%', '0.5%'],
];

export default function AmvExecutableWeightGridV2() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Executable Weight Grid v2</H1>
        <Text tone="secondary">
          重跑早期 AMV 多头宽池因子、组合与权重网格。信号仍由 <Code>D close</Code>{' '}
          排序生成，主评估改为 <Code>D+1 open -&gt; D+7 close</Code>，辅助保留{' '}
          <Code>D close -&gt; D+6 close</Code>。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+160.14%" label="P3/K0.5/R0 executable NAV" tone="success" />
        <Stat value="+152.21%" label="P2/K0.5/R0 executable NAV" tone="success" />
        <Stat value="+29% ~ +33%" label="P/K/M executable NAV" tone="warning" />
        <Stat value="90" label="rankers 全量扫描" tone="info" />
      </Grid>

      <Callout tone="success" title="核心结论">
        可执行口径重新支持早期 AMV 的“高位 + K 线确认、低/无动量”结构。
        <Code>P3/K0.5/R0</Code> 在标签侧 executable 指标略强于当前 reference{' '}
        <Code>P2/K0.5/R0</Code>，但仍需 Rust <Code>T+1 open / 6td / Top3 / no-stop</Code>{' '}
        真实回测确认。带动量 P/K/M 的 close-to-close NAV 仍然爆炸，但 executable NAV
        只有低几十个百分点，说明污染很重。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>主评估 vs 辅助诊断</H2>
          <BarChart
            categories={['P3/K0.5/R0', 'P2/K0.5/R0', 'P1/K0.5/M1', 'P2/K0.5/M0.5', 'P3/K1/M2']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Executable NAV', data: [160.14, 152.21, 32.87, 29.14, 31.62], tone: 'success' },
              { name: 'Close-to-close NAV', data: [278.93, 248.54, 1054.55, 768.74, 1045.97], tone: 'warning' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>artifacts/amv_executable_weight_grid/20260519_144938/compact.csv</Code>.
            Metrics are rolling NAV end values for 2021-04-20 to 2026-04-24.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>污染归因</H2>
          <BarChart
            categories={['P3/K0.5/R0', 'P2/K0.5/R0', 'P1/K0.5/M1', 'P2/K0.5/M0.5', 'P3/K1/M2']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'close 涨停覆盖天数', data: [20.2, 16.8, 73.5, 59.7, 76.4], tone: 'danger' },
              { name: 'T+1 高开覆盖天数', data: [9.0, 7.6, 28.0, 26.4, 28.2], tone: 'warning' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: same run. High open threshold is <Code>9.8%</Code>; price limits use 10% / 20% board rules.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>重点候选对照</H2>
      <Table
        headers={['候选', '场景', 'Executable NAV', 'CTC NAV', 'close 涨停天', 'T+1 高开天', '判断']}
        rows={focusRows}
        rowTone={['success', 'success', 'danger', 'danger', 'danger', 'warning']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right', 'left']}
      />

      <H2>跳过 close 涨停并补位后</H2>
      <Table
        headers={['候选', 'Executable NAV', 'CTC NAV', 'close 涨停天', 'T+1 高开天', '补位深度']}
        rows={refillRows}
        rowTone={['success', 'success', 'warning', 'warning', 'warning', 'warning']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
      />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Original Top3: executable tradeoff 前列</H2>
          <Table
            headers={['候选', 'Exec NAV', 'MaxDD', 'CTC NAV', 'close 涨停天', 'T+1 高开天']}
            rows={topOriginalRows}
            rowTone={['warning', 'success', 'success', 'success', 'success', 'success', 'success', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>Refill Top3: executable tradeoff 前列</H2>
          <Table
            headers={['候选', 'Exec NAV', 'MaxDD', 'CTC NAV', 'close 涨停天', 'T+1 高开天']}
            rows={topRefillRows}
            rowTone={['success', 'warning', 'success', 'success', 'success', 'success', 'success', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Callout tone="warning" title="执行约束">
        这个 canvas 仍是 Python 标签侧诊断，不含真实账户资金、手数、成本、no-repeat 与买卖失败路径。
        它的作用是过滤掉明显的不可执行优化方向。下一步应把 <Code>P3/K0.5/R0</Code>{' '}
        导出为静态 sleeve，接 Rust <Code>bt-amv-topn</Code> 做真实回测。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        Full run: <Code>artifacts/amv_executable_weight_grid/20260519_144938/summary.json</Code>.
        Script: <Code>scripts/amv_executable_weight_grid.py</Code>. Rankers reused from{' '}
        <Code>scripts/amv_yearly_weight_grid.py</Code>. Default: AMV bull LF2, Top3, horizon 6,
        <Code>mv_min = 100</Code>, <Code>amount_ma20_min = 5e7</Code>.
      </Text>
    </Stack>
  );
}
