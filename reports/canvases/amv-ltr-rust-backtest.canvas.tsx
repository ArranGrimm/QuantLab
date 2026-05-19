// @ts-nocheck
import { Callout, Divider, Grid, H1, H2, H3, Stack, Stat, Table, Text } from 'cursor/canvas';

const backtestRows = [
  ['规则基线 6td no stop', '+144.02%', '+195.18%', '-14.71%', '264', '52.3%', '+72.01万'],
  ['LTR no_risk_old_state', '-0.01%', '+20.55%', '-52.93%', '165', '43.0%', '-45元'],
  ['LTR kbar_momentum_old_state', '-66.97%', '-55.36%', '-74.93%', '166', '43.4%', '-33.49万'],
];

const annualRows = [
  ['kbar_momentum_old_state', '2023', '-37.97%', '-47.89%'],
  ['kbar_momentum_old_state', '2024', '-35.95%', '-44.40%'],
  ['kbar_momentum_old_state', '2025', '-7.44%', '-29.98%'],
  ['kbar_momentum_old_state', '2026 YTD', '-9.62%', '-27.45%'],
  ['no_risk_old_state', '2023', '+1.02%', '-35.02%'],
  ['no_risk_old_state', '2024', '+47.51%', '-34.72%'],
  ['no_risk_old_state', '2025', '-27.52%', '-30.32%'],
  ['no_risk_old_state', '2026 YTD', '-7.63%', '-23.77%'],
];

const executableRows = [
  ['kbar', '1td', '+0.16%', '-0.38%', '47.2%', '+2.16%'],
  ['kbar', '6td', '+0.13%', '-1.57%', '43.7%', '-0.01%'],
  ['no_risk', '1td', '+0.23%', '-0.47%', '46.8%', '+2.22%'],
  ['no_risk', '6td', '-0.61%', '-3.05%', '39.1%', '-1.76%'],
];

export default function AmvLtrRustBacktest() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>AMV LTR Rust 回测首轮</H1>
        <Text tone="secondary">
          Source: artifacts/amv_ltr_signals · bt-amv-topn · config_6td_no_stop.toml · T+1 open buy / 6td close sell
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="-0.01%" label="no_risk 净收益" tone="warning" />
        <Stat value="-66.97%" label="kbar 净收益" tone="danger" />
        <Stat value="+144.02%" label="规则基线净收益" tone="success" />
        <Stat value="124 / 126" label="LTR 涨停过滤次数" tone="warning" />
      </Grid>

      <Callout tone="danger" title="结论">
        <Text>
          LTR 的 6d close-to-close 标签没有兑现到真实执行层。真实 T+1 开盘买入后, 6td edge 基本消失;
          no_risk 仅打平, kbar 大幅亏损, 都不能替代当前规则基线。
        </Text>
      </Callout>

      <H2>真实回测对比</H2>
      <Table
        headers={['策略', '净收益', 'Gross收益', '最大回撤', '交易数', '胜率', '净PnL']}
        rows={backtestRows}
        rowTone={['success', 'warning', 'danger']}
      />

      <Divider />

      <Grid columns={2} gap={18}>
        <Stack gap={10}>
          <H3>LTR 年度路径</H3>
          <Table headers={['Variant', '年份', '收益', '最大回撤']} rows={annualRows} />
        </Stack>
        <Stack gap={10}>
          <H3>可执行标签塌缩</H3>
          <Text tone="secondary">
            从执行日开盘买入到 h 日后收盘的等权样本收益, 已扣除涨停不可买和非 bull 执行日。
          </Text>
          <Table
            headers={['Variant', '持有', '全样本均值', '中位数', '胜率', '2026均值']}
            rows={executableRows}
          />
        </Stack>
      </Grid>

      <Divider />

      <Callout tone="info" title="下一步">
        <Text>
          不应继续把当前 6d LTR 直接接 Rust。下一步应改标签为执行口径:
          T+1 open 到 N 日 close 的收益, 并加入涨停不可买过滤; 或先探索 1td/2td 的执行口径标签。
        </Text>
      </Callout>
    </Stack>
  );
}
