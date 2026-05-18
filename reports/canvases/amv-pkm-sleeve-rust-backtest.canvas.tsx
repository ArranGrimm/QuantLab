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

const summaryRows = [
  ['manual P2/K0.5/R0', '+168.01%', '-14.97%', '273', '51.65%', '当前主基线'],
  ['P1/K0.5/M1', '+83.18%', '-49.27%', '268', '47.01%', '2025 强，但回撤深'],
  ['P3/K1/M2', '+92.79%', '-48.94%', '267', '46.82%', '2025 强，但未解决 2026'],
  ['P2/K0.5/M0.5', '+11.45%', '-61.87%', '270', '45.56%', '不成立'],
];

const annualRows = [
  ['manual P2/K0.5/R0', '+9.7%', '+38.4%', '+14.3%', '+48.9%', '+12.4%', '-8.7%'],
  ['P1/K0.5/M1', '+6.0%', '+21.0%', '-13.1%', '+23.4%', '+59.2%', '-16.4%'],
  ['P3/K1/M2', '+1.2%', '+24.6%', '-11.2%', '+36.7%', '+50.9%', '-16.5%'],
  ['P2/K0.5/M0.5', '+1.4%', '+8.6%', '-7.3%', '+55.1%', '-9.1%', '-22.6%'],
];

const signalRows = [
  ['P1/K0.5/M1', '2,428', '812', '188', '81'],
  ['P2/K0.5/M0.5', '2,431', '812', '175', '81'],
  ['P3/K1/M2', '2,431', '812', '191', '81'],
];

const horizonRows = [
  ['P1/K0.5/M1', '-79.91%', '-72.90%', '-39.99%', '-52.18%', '-19.32%', '+83.18%', '-17.70%', '6td'],
  ['P2/K0.5/M0.5', '-69.71%', '-74.68%', '-45.79%', '-64.87%', '-19.43%', '+11.45%', '-10.50%', '6td'],
  ['P3/K1/M2', '-79.99%', '-78.04%', '-29.48%', '-61.11%', '-15.06%', '+92.79%', '-34.44%', '6td'],
];

export default function AmvPkmSleeveRustBacktest() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV P/K/M 袖子 Rust 回测</H1>
        <Text tone="secondary">
          将年度权重网格里的动量增强候选接入 <Code>bt-amv-topn</Code>。
          口径为 <Code>T+1 open</Code> 买入、<Code>max_hold_trading_days = 6</Code>、
          Top3、no-stop、AMV bull entry gate。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+168.01%" label="主基线净收益" tone="success" />
        <Stat value="+92.79%" label="P/K/M 最好净收益" tone="warning" />
        <Stat value="-48.94%" label="P/K/M 最浅 MaxDD" tone="danger" />
        <Stat value="1-7td" label="扫描持有期" tone="info" />
      </Grid>

      <Callout tone="warning" title="结论">
        标签侧 P/K/M 动量增强没有兑现成更好的真实交易主基线。补扫 <Code>4td / 5td / 7td</Code>{' '}
        后，三组候选最佳仍是 <Code>6td</Code>。它们明显改善 2025，
        但总收益和回撤都弱于 <Code>manual_p2_k0p5_r0_6td</Code>，且 2026 亏损更深。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>净收益 vs 回撤</H2>
          <BarChart
            categories={['manual', 'P1/K0.5/M1', 'P3/K1/M2', 'P2/K0.5/M0.5']}
            valueSuffix="%"
            height={240}
            series={[
              { name: '净收益', data: [168.01, 83.18, 92.79, 11.45], tone: 'success' },
              { name: '最大回撤绝对值', data: [14.97, 49.27, 48.94, 61.87], tone: 'danger' },
            ]}
          />
        </Stack>
        <Stack gap={10}>
          <H2>2025/2026 对照</H2>
          <BarChart
            categories={['manual', 'P1/K0.5/M1', 'P3/K1/M2', 'P2/K0.5/M0.5']}
            valueSuffix="%"
            height={240}
            series={[
              { name: '2025', data: [12.4, 59.2, 50.9, -9.1], tone: 'success' },
              { name: '2026 YTD', data: [-8.7, -16.4, -16.5, -22.6], tone: 'warning' },
            ]}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>真实回测汇总</H2>
      <Table
        headers={['袖子', '净收益', 'MaxDD', '交易数', '胜率', '判断']}
        rows={summaryRows}
        rowTone={['success', 'warning', 'warning', 'danger']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
      />

      <H2>年度收益</H2>
      <Table
        headers={['袖子', '2021', '2022', '2023', '2024', '2025', '2026 YTD']}
        rows={annualRows}
        rowTone={['success', 'warning', 'warning', 'danger']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>信号与过滤</H2>
      <Table
        headers={['袖子', '执行信号行', '执行信号日', '涨停过滤', '执行日非 bull 阻断']}
        rows={signalRows}
        columnAlign={['left', 'right', 'right', 'right', 'right']}
      />

      <H2>持仓周期扫描</H2>
      <Table
        headers={['袖子', '1td', '2td', '3td', '4td', '5td', '6td', '7td', '最佳']}
        rows={horizonRows}
        rowTone={['warning', 'danger', 'warning']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'left']}
      />

      <Callout tone="info" title="下一步">
        P/K/M 和主基线一样，最佳持仓周期仍是 6td。补扫 4td / 5td / 7td 后，
        5td 相对短持仓有所修复但仍为负，7td 又明显回落。
        说明问题不是“动量袖子换个持仓期就能兑现”，而是 P/K/M 的真实执行质量不足。
        2026 的问题仍应回到入场后上冲空间不足、执行日收弱、短期市场宽度和赚钱效应确认。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_static_sleeve_signals/pkm_rust_6td_no_stop_summary_20260518_120000.json</Code>.
        持仓周期扫描: <Code>artifacts/amv_static_sleeve_signals/pkm_rust_horizon_scan_1td_to_7td_summary_20260518_121300.json</Code>.
        配置: <Code>backtest-engine/crates/amv-topn/config_6td_no_stop.toml</Code> 及临时 4td/5td/7td no-stop 派生配置.
      </Text>
    </Stack>
  );
}
