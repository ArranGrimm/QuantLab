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

const originalRows = [
  ['B1/C0/R0', '+245.37%', '-23.29%', '+124.33%', '0.5%', '纯 bias pullback，最强'],
  ['B3/C1/R0', '+215.37%', '-20.29%', '+110.09%', '0.0%', '强 pullback 组合'],
  ['P1/B3/C1/R0', '+179.37%', '-18.70%', '+105.12%', '0.0%', '轻 P + pullback'],
  ['B2/C1/R0', '+176.51%', '-19.39%', '+90.94%', '0.0%', '纯 pullback'],
  ['P3/K0.5/R0', '+160.14%', '-5.58%', '+278.93%', '20.2%', '旧候选'],
  ['P2/K0.5/R0', '+152.21%', '-5.27%', '+248.54%', '16.8%', '当前 reference'],
  ['B1/C1/R0', '+163.08%', '-19.45%', '+77.06%', '0.0%', '均衡 pullback'],
  ['P2/C1/R0', '+145.52%', '-21.09%', '+613.86%', '28.7%', '污染偏高'],
];

const refillRows = [
  ['B1/C0/R0', '+242.58%', '-23.29%', '+124.83%', 'q95=3', '几乎不受补位影响'],
  ['B3/C1/R0', '+215.37%', '-20.29%', '+110.09%', 'q95=3', '稳定'],
  ['P1/B3/C1/R0', '+179.37%', '-18.70%', '+105.12%', 'q95=3', '稳定'],
  ['B2/C1/R0', '+176.51%', '-19.39%', '+90.94%', 'q95=3', '稳定'],
  ['P3/C1/R0', '+157.93%', '-11.05%', '+150.89%', 'q95=5', 'P/C 混合候选'],
  ['P2/C1/R0', '+155.87%', '-10.14%', '+149.99%', 'q95=5', 'P/C 混合候选'],
  ['P3/K0.5/R0', '+151.90%', '-7.23%', '+117.44%', 'q95=4', '旧候选补位'],
  ['P2/K0.5/C1/R0.5', '+138.29%', '-7.84%', '+139.22%', 'q95=5', '低回撤混合'],
];

export default function AmvExecutablePullbackGrid() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Pullback Combo Grid</H1>
        <Text tone="secondary">
          将全因子扫描发现的 <Code>ma_bias_20 / disp_bias_20 / KSFT / intraday_pos</Code>{' '}
          回调线索加入可解释组合网格。主评估为 <Code>D+1 open -&gt; D+7 close</Code>，
          辅助诊断为 <Code>D close -&gt; D+6 close</Code>。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+245.37%" label="最强 pullback exec NAV" tone="success" />
        <Stat value="0.5%" label="最强候选 close 涨停天" tone="success" />
        <Stat value="-23.29%" label="最强候选 MaxDD" tone="warning" />
        <Stat value="164" label="focused rankers" tone="info" />
      </Grid>

      <Callout tone="success" title="结论">
        pullback sleeve 成立，而且不是涨停污染产物。最强组合 <Code>B1/C0/R0</Code>{' '}
        基本是 <Code>ma_bias_20 + disp_bias_20</Code> 回调组合，收益高于现有 P/K reference，
        但回撤也明显更深。它更适合作为独立 sleeve 或互补 sleeve，而不是直接替换主基线。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Original Top3 对照</H2>
          <BarChart
            categories={['B1', 'B3/C1', 'P1/B3/C1', 'B2/C1', 'P3/K0.5', 'P2/K0.5']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Executable NAV', data: [245.37, 215.37, 179.37, 176.51, 160.14, 152.21], tone: 'success' },
              { name: 'MaxDD 绝对值', data: [23.29, 20.29, 18.7, 19.39, 5.58, 5.27], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>artifacts/amv_executable_pullback_grid/20260519_160017/compact.csv</Code>.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>污染对照</H2>
          <BarChart
            categories={['B1', 'B3/C1', 'P1/B3/C1', 'B2/C1', 'P3/K0.5', 'P2/K0.5']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'close 涨停覆盖天数', data: [0.5, 0.0, 0.0, 0.0, 20.2, 16.8], tone: 'danger' },
              { name: 'CTC NAV / 10', data: [12.43, 11.01, 10.51, 9.09, 27.89, 24.85], tone: 'warning' },
            ]}
          />
          <Text size="small" tone="tertiary">
            CTC NAV divided by 10 only for scale comparison.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>Original Top3 前列</H2>
      <Table
        headers={['候选', 'Exec NAV', 'MaxDD', 'CTC NAV', 'close 涨停天', '判断']}
        rows={originalRows}
        rowTone={['success', 'success', 'success', 'success', 'warning', 'success', 'success', 'warning']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
      />

      <H2>跳过 close 涨停并补位后</H2>
      <Table
        headers={['候选', 'Exec NAV', 'MaxDD', 'CTC NAV', '补位深度', '判断']}
        rows={refillRows}
        rowTone={['success', 'success', 'success', 'success', 'success', 'success', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right', 'left', 'left']}
      />

      <Callout tone="warning" title="下一步候选">
        建议优先导出 4 个静态 sleeve 接 Rust 回测：
        <Code>pullback_p0_k0_b1_c0_r0</Code>, <Code>pullback_p0_k0_b3_c1_r0</Code>,{' '}
        <Code>pullback_p2_k0_b0_c1_r0</Code>, <Code>pullback_p3_k0_b0_c1_r0</Code>。
        重点看真实成本、手数、no-repeat 和 T+1 open 后是否仍能保留优势。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        Full run: <Code>artifacts/amv_executable_pullback_grid/20260519_160017/summary.json</Code>.
        Script: <Code>scripts/amv_executable_pullback_grid.py</Code>. Focused grid: P/K/B/C/R, no momentum.
      </Text>
    </Stack>
  );
}
