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
  ['manual P2/K0.5/R0', '+170.80%', '-15.30%', '274', '51.09%', '当前主基线'],
  ['P3/K1/M2', '+103.58%', '-48.08%', '268', '47.01%', '2025 强，但未解决 2026'],
  ['P1/K0.5/M1', '+90.34%', '-48.04%', '269', '47.21%', '2025 强，但回撤深'],
  ['P2/K0.5/M0.5', '+12.10%', '-62.15%', '270', '45.93%', '不成立'],
];

const annualRows = [
  ['manual P2/K0.5/R0', '+10.4%', '+39.7%', '+14.3%', '+49.7%', '+12.6%', '-8.8%'],
  ['P3/K1/M2', '+0.4%', '+28.1%', '-12.8%', '+37.2%', '+52.1%', '-12.9%'],
  ['P1/K0.5/M1', '+5.7%', '+23.9%', '-13.9%', '+23.2%', '+59.6%', '-14.1%'],
  ['P2/K0.5/M0.5', '+0.5%', '+10.2%', '-7.7%', '+56.9%', '-9.4%', '-22.8%'],
];

const signalRows = [
  ['P1/K0.5/M1', '2,428', '812', '188', '81'],
  ['P2/K0.5/M0.5', '2,431', '812', '175', '81'],
  ['P3/K1/M2', '2,431', '812', '191', '81'],
];

const horizonRows = [
  ['P1/K0.5/M1', '-79.91%', '-72.90%', '-39.99%', '-52.18%', '-19.32%', '+90.34%', '-17.70%', '6td'],
  ['P2/K0.5/M0.5', '-69.71%', '-74.68%', '-45.79%', '-64.87%', '-19.43%', '+12.10%', '-10.50%', '6td'],
  ['P3/K1/M2', '-79.99%', '-78.04%', '-29.48%', '-61.11%', '-15.06%', '+103.58%', '-34.44%', '6td'],
];

const rollingRows = [
  ['manual P2/K0.5/R0', '5td rolling18', '+11.90%', '-9.25%', '1,361', '46.66%', '-1.08%'],
  ['manual P2/K0.5/R0', '6td rolling21', '+23.61%', '-9.33%', '1,335', '46.82%', '-0.60%'],
  ['P1/K0.5/M1', '5td rolling18', '-13.99%', '-31.19%', '1,248', '45.75%', '+4.12%'],
  ['P1/K0.5/M1', '6td rolling21', '-7.29%', '-28.28%', '1,235', '44.53%', '-2.58%'],
  ['P2/K0.5/M0.5', '5td rolling18', '-6.23%', '-31.08%', '1,301', '45.89%', '+3.96%'],
  ['P2/K0.5/M0.5', '6td rolling21', '+0.21%', '-26.59%', '1,293', '45.09%', '-0.33%'],
  ['P3/K1/M2', '5td rolling18', '-14.10%', '-29.90%', '1,241', '46.41%', '+3.50%'],
  ['P3/K1/M2', '6td rolling21', '-8.57%', '-27.50%', '1,226', '44.54%', '-2.95%'],
];

const ctcDiagnosticRows = [
  ['manual P2/K0.5/R0', '6td rolling21', '+20.88%', '-11.08%', '1,311', '138', '0'],
  ['P1/K0.5/M1', '6td rolling21', '+5.87%', '-15.38%', '772', '812', '3'],
  ['P2/K0.5/M0.5', '6td rolling21', '+15.31%', '-11.24%', '971', '602', '3'],
  ['P3/K1/M2', '6td rolling21', '+8.83%', '-12.38%', '732', '856', '3'],
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
        <Stat value="+170.80%" label="主基线净收益" tone="success" />
        <Stat value="+103.58%" label="P/K/M 最好净收益" tone="warning" />
        <Stat value="-48.04%" label="P/K/M 最浅 MaxDD" tone="danger" />
        <Stat value="1-7td" label="扫描持有期" tone="info" />
      </Grid>

      <Callout tone="warning" title="结论">
        标签侧 P/K/M 动量增强没有兑现成更好的真实交易主基线。补扫 <Code>4td / 5td / 7td</Code>{' '}
        后，三组候选最佳仍是 <Code>6td</Code>。它们明显改善 2025，
        但总收益和回撤都弱于 <Code>manual_p2_k0p5_r0_6td</Code>，且 2026 亏损更深。
        rolling cohort 与 close-to-close 诊断口径下，P/K/M 也没有追上 manual。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>净收益 vs 回撤</H2>
          <BarChart
            categories={['manual', 'P3/K1/M2', 'P1/K0.5/M1', 'P2/K0.5/M0.5']}
            valueSuffix="%"
            height={240}
            series={[
              { name: '净收益', data: [170.8, 103.58, 90.34, 12.1], tone: 'success' },
              { name: '最大回撤绝对值', data: [15.3, 48.08, 48.04, 62.15], tone: 'danger' },
            ]}
          />
        </Stack>
        <Stack gap={10}>
          <H2>2025/2026 对照</H2>
          <BarChart
            categories={['manual', 'P3/K1/M2', 'P1/K0.5/M1', 'P2/K0.5/M0.5']}
            valueSuffix="%"
            height={240}
            series={[
              { name: '2025', data: [12.6, 52.1, 59.6, -9.4], tone: 'success' },
              { name: '2026 YTD', data: [-8.8, -12.9, -14.1, -22.8], tone: 'warning' },
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

      <H2>Rolling Cohort 真实组合</H2>
      <Table
        headers={['袖子', '配置', '净收益', 'MaxDD', '交易数', '胜率', '2026 YTD']}
        rows={rollingRows}
        rowTone={['warning', 'success', 'danger', 'danger', 'warning', 'warning', 'danger', 'danger']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>Close-to-close 诊断 B</H2>
      <Table
        headers={['袖子', '配置', '净收益', 'MaxDD', '交易数', 'close 涨停过滤', '高开过滤']}
        rows={ctcDiagnosticRows}
        rowTone={['success', 'warning', 'warning', 'warning']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right', 'right']}
      />

      <Callout tone="info" title="下一步">
        Python rolling NAV 的 P/K/M 高收益很大一部分来自信号日 close 已经涨停、现实不可买的票。
        close-to-close 诊断加入涨停不可买后，P/K/M 收益只剩低两位数；再切到 T+1 open 后进一步塌缩。
        下一步若继续，应做过滤归因表，而不是直接把 Python rolling NAV 当作可交易上限。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        数据来源: <Code>artifacts/amv_static_sleeve_signals/pkm_rust_6td_no_stop_new_lot_summary_20260519_085900.json</Code>.
        持仓周期扫描: <Code>artifacts/amv_static_sleeve_signals/pkm_rust_horizon_scan_1td_to_7td_summary_20260518_121300.json</Code>.
        表中 <Code>6td</Code> 已用新手数口径校准，其他持仓期仍待重扫。
        Rolling cohort: <Code>artifacts/amv_static_sleeve_signals/amv_rolling_cohort_new_lot_summary_20260519_090500.json</Code>.
        Close-to-close diagnostic: <Code>artifacts/amv_close_to_close_diagnostic_signals/amv_close_to_close_cohort_diagnostic_summary_20260519_140000.json</Code>.
        配置: <Code>backtest-engine/crates/amv-topn/config_6td_no_stop.toml</Code> 及临时 4td/5td/7td no-stop 派生配置.
      </Text>
    </Stack>
  );
}
