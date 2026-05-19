import { Callout, Divider, Grid, H1, H2, H3, Stack, Stat, Table, Text } from 'cursor/canvas';

const contributionRows = [
  ['kbar_momentum_state', '+5.36%', '+3.67%', '56.6%', '61.6%', '+19.49%', '-7.68%', '43%', '88%'],
  ['no_risk', '+4.52%', '+0.76%', '53.5%', '57.6%', '+18.53%', '-7.99%', '46%', '103%'],
];

const overlapRows = [
  ['2023', '0.88 / 3', '29.2%', '20.2%'],
  ['2024', '1.42 / 3', '47.4%', '35.5%'],
  ['2025', '1.29 / 3', '43.1%', '31.0%'],
  ['2026', '1.42 / 3', '47.5%', '38.2%'],
];

const beforeAfterRows = [
  ['2023', '0.88 / 3', '+0.35pp', '+0.59pp', '54.0%'],
  ['2024', '1.17 / 3', '+0.09pp', '+0.71pp', '45.9%'],
  ['2025', '1.62 / 3', '+1.58pp', '+0.31pp', '49.4%'],
  ['2026', '1.18 / 3', '-2.89pp', '-1.64pp', '39.4%'],
];

const codeRows = [
  ['no_risk', 'sz.002931', '5', '+232.5pp', '约52%'],
  ['no_risk', 'sh.603629', '3', '+84.0pp', '约19%'],
  ['no_risk', 'sz.301396', '2', '+83.8pp', '约19%'],
  ['kbar_momentum_state', 'sz.002931', '6', '+239.5pp', '约45%'],
  ['kbar_momentum_state', 'sz.002155', '4', '+88.0pp', '约17%'],
  ['kbar_momentum_state', 'sh.603950', '2', '+69.2pp', '约13%'],
];

export default function AmvLtrSelectionAttribution() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>AMV LTR 选股归因</H1>
        <Text tone="secondary">
          Source: artifacts/amv_bull_pool_listwise_ranker/20260516_112444/selection_analysis_20260516_112604 · 2026 top3 选股归因 · 6d 标签
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="1.42 / 3" label="2026 两模型日均重合" />
        <Stat value="+5.36%" label="kbar+momentum 2026均值" tone="success" />
        <Stat value="+4.52%" label="no_risk 2026均值" tone="success" />
        <Stat value="45%-52%" label="sz.002931 净贡献占比" tone="warning" />
      </Grid>

      <Callout tone="warning" title="结论先行">
        <Text>
          LTR 候选确实有 alpha, 但 2026 的高收益仍明显依赖少数股票重复入选。
          接 Rust 前应先做去极值、去单票重复、或单票贡献上限敏感性测试。
        </Text>
      </Callout>

      <H2>2026 贡献与路径</H2>
      <Text tone="secondary">
        top10 占比按等权选股的 6d 未来收益累计估算; MFE/MAE 是入场后 6 日路径。
      </Text>
      <Table
        headers={['Variant', '均值', '中位数', '胜率', 'Hit15', '平均MFE', '平均MAE', 'Top10/正收益', 'Top10/净收益']}
        rows={contributionRows}
        rowTone={['success', 'success']}
      />

      <Divider />

      <Grid columns={2} gap={18}>
        <Stack gap={10}>
          <H3>no_risk vs kbar_momentum_state 重合度</H3>
          <Table headers={['年份', '日均重合', '重合比例', 'Jaccard']} rows={overlapRows} />
        </Stack>
        <Stack gap={10}>
          <H3>状态补齐前后 no_risk 对比</H3>
          <Table
            headers={['年份', '日均重合', '收益差', 'MFE差', '改善日占比']}
            rows={beforeAfterRows}
            rowTone={[undefined, undefined, 'success', 'warning']}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>2026 股票级贡献集中度</H2>
      <Table
        headers={['Variant', '股票', '入选次数', '累计收益贡献', '占该variant净收益']}
        rows={codeRows}
        rowTone={['warning', undefined, undefined, 'warning', undefined, undefined]}
      />

      <Callout tone="info" title="下一步">
        <Text>
          最值得做的是稳健性测试: 去掉每个 variant 的最大贡献股票、限制同一股票重复入选次数、
          对 2026 top/worst 票做入场后路径复盘。通过后再进入 Rust 回测更合理。
        </Text>
      </Callout>
    </Stack>
  );
}
