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
  ['ret_5d_desc', '5日动量强', '+264.07%', '-47.11%', '+365.43%', '69.0%', '14.1%', 'attack/event'],
  ['ma_bias_20_asc', '20日均线回调', '+231.03%', '-23.19%', '+132.19%', '0.7%', '0.2%', '低污染强候选'],
  ['disp_bias_20_asc', '成本线下回归', '+187.54%', '-22.68%', '+94.73%', '0.4%', '0.4%', '低污染强候选'],
  ['KSFT_asc', '收盘位置偏低', '+178.36%', '-24.50%', '+2.59%', '0.0%', '0.7%', '低污染强候选'],
  ['combo_high_pos_kmid2_lowrisk', '高位+实体强+低风险', '+157.02%', '-11.44%', '+152.69%', '10.3%', '1.4%', '稳健组合'],
  ['combo_high_pos_kbar_confirm', '高位+K线确认', '+140.77%', '-5.07%', '+191.25%', '13.7%', '5.4%', '现有主线结构'],
  ['far_from_high_20d', '远离20日高点', '+126.72%', '-26.88%', '+62.51%', '2.0%', '0.4%', '回调线索'],
  ['ret_20d_asc', '20日反转', '+114.83%', '-25.73%', '+73.80%', '1.4%', '0.7%', '反转线索'],
];

const refillRows = [
  ['ma_bias_20_asc', '+227.14%', '+131.40%', '0.0%', '0.2%', 'q95=3 / max=5'],
  ['disp_bias_20_asc', '+188.64%', '+95.87%', '0.0%', '0.4%', 'q95=3 / max=5'],
  ['KSFT_asc', '+178.36%', '+2.59%', '0.0%', '0.7%', 'q95=3 / max=3'],
  ['ret_5d_desc', '+165.93%', '+0.25%', '0.0%', '1.1%', 'q95=7 / max=21'],
  ['combo_high_pos_kmid2_lowrisk', '+138.53%', '+119.77%', '0.0%', '0.5%', 'q95=3 / max=48'],
  ['combo_high_pos_kbar_confirm', '+121.80%', '+101.77%', '0.0%', '0.2%', 'q95=4 / max=243'],
  ['price_pos_20d_desc', '+125.76%', '+84.47%', '0.0%', '0.5%', 'q95=4 / max=7'],
  ['intraday_pos_asc', '+114.96%', '+51.02%', '0.0%', '0.4%', 'q95=4 / max=24'],
];

const lowPollutionRows = [
  ['ma_bias_20_asc', '20日均线回调', '+231.03%', '-23.19%', '0.7%', '5 / 5'],
  ['disp_bias_20_asc', '成本线下回归', '+187.54%', '-22.68%', '0.4%', '5 / 5'],
  ['KSFT_asc', '收盘位置偏低', '+178.36%', '-24.50%', '0.0%', '5 / 5'],
  ['far_from_high_20d', '远离20日高点', '+126.72%', '-26.88%', '2.0%', '3 / 5'],
  ['ret_20d_asc', '20日反转', '+114.83%', '-25.73%', '1.4%', '4 / 5'],
  ['vol_price_corr_20d_desc', '量价同涨', '+103.53%', '-44.65%', '26.4%', '4 / 5'],
];

export default function AmvExecutableFactorScan() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Executable Factor Scan</H1>
        <Text tone="secondary">
          早期全因子扫描按 executable-aware 口径重跑。主评估为{' '}
          <Code>D+1 open -&gt; D+7 close</Code>，辅助诊断为 <Code>D close -&gt; D+6 close</Code>，
          并强制记录 close 涨停、T+1 涨停、高开与补位表现。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="47" label="早期 rankers" tone="info" />
        <Stat value="+231.03%" label="最佳低污染 exec NAV" tone="success" />
        <Stat value="+264.07%" label="最高 exec NAV" tone="warning" />
        <Stat value="0.0% ~ 0.7%" label="低污染候选 close 涨停天" tone="success" />
      </Grid>

      <Callout tone="success" title="新线索">
        executable 口径下，除了 <Code>ret_5d_desc</Code> 的高收益高回撤动量 archetype，
        低污染强候选集中在“20 日均线/成本线下回归 + 收盘位置偏低”的回调反弹结构。
        这条线索比继续增加 P/K/M 动量权重更值得进入下一轮组合网格。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Executable NAV 前列</H2>
          <BarChart
            categories={['ret5强', '均线回调', '成本线回归', '收盘偏低', '高位实体低风险', '高位K线']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'Executable NAV', data: [264.07, 231.03, 187.54, 178.36, 157.02, 140.77], tone: 'success' },
              { name: 'MaxDD 绝对值', data: [47.11, 23.19, 22.68, 24.5, 11.44, 5.07], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>artifacts/amv_executable_factor_scan/20260519_151529/compact.csv</Code>.
            Rolling NAV end and max drawdown, 2021-04-20 to 2026-04-24.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>污染对照</H2>
          <BarChart
            categories={['ret5强', '均线回调', '成本线回归', '收盘偏低', '高位实体低风险', '高位K线']}
            valueSuffix="%"
            height={260}
            series={[
              { name: 'close 涨停覆盖天数', data: [69.0, 0.7, 0.4, 0.0, 10.3, 13.7], tone: 'danger' },
              { name: 'T+1 高开覆盖天数', data: [14.1, 0.2, 0.4, 0.7, 1.4, 5.4], tone: 'warning' },
            ]}
          />
          <Text size="small" tone="tertiary">
            High open threshold is <Code>9.8%</Code>; price limit uses main board 10% and ChiNext/STAR 20%.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>Original Top3 前列</H2>
      <Table
        headers={['ID', '含义', 'Exec NAV', 'MaxDD', 'CTC NAV', 'close 涨停天', 'T+1 高开天', '判断']}
        rows={originalRows}
        rowTone={['warning', 'success', 'success', 'success', 'success', 'success', 'success', 'success']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right', 'right', 'left']}
      />

      <H2>跳过 close 涨停并补位后</H2>
      <Table
        headers={['ID', 'Exec NAV', 'CTC NAV', 'close 涨停天', 'T+1 高开天', '补位深度']}
        rows={refillRows}
        rowTone={['success', 'success', 'success', 'warning', 'success', 'success', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
      />

      <H2>低污染优先候选</H2>
      <Table
        headers={['ID', '含义', 'Exec NAV', 'MaxDD', 'close 涨停天', '稳定正 edge 年数']}
        rows={lowPollutionRows}
        rowTone={['success', 'success', 'success', 'warning', 'warning', 'warning']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right']}
      />

      <Callout tone="warning" title="下一步">
        不建议把 <Code>ret_5d_desc</Code> 直接当主线，它更像 attack/event sleeve。
        更合理的下一轮是构造 <Code>P/K + pullback</Code> 或 <Code>pullback + lowrisk</Code>{' '}
        网格，检验“高位强势”和“回调后反弹”是否能形成互补。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        Full run: <Code>artifacts/amv_executable_factor_scan/20260519_151529/summary.json</Code>.
        Script: <Code>scripts/amv_executable_factor_scan.py</Code>. Rankers: early{' '}
        <Code>RANKERS + COMBO_RANKERS</Code>, AMV bull LF2, Top3, horizon 6.
      </Text>
    </Stack>
  );
}
