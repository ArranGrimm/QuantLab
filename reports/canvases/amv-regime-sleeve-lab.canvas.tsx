// @ts-nocheck
import { Callout, Divider, Grid, H1, H2, H3, Stack, Stat, Table, Text } from 'cursor/canvas';

const overallRows = [
  ['1td', 'daily_oracle', '+5.14pp', '+5.40%', '95.8%'],
  ['1td', 'static_ret_5d', '+0.66pp', '+0.92%', '53.9%'],
  ['1td', 'state_classifier', '+0.54pp', '+0.80%', '53.9%'],
  ['2td', 'daily_oracle', '+6.14pp', '+6.53%', '96.1%'],
  ['2td', 'static_ret_20d', '+0.70pp', '+1.09%', '52.7%'],
  ['2td', 'static_ret_5d', '+0.69pp', '+1.08%', '48.7%'],
  ['2td', 'state_classifier', '+0.24pp', '+0.63%', '47.6%'],
  ['3td', 'daily_oracle', '+6.99pp', '+7.48%', '96.5%'],
  ['3td', 'static_ret_5d', '+0.66pp', '+1.15%', '49.6%'],
  ['3td', 'state_classifier', '+0.09pp', '+0.58%', '46.3%'],
];

const yearlyRows = [
  ['1td', 'oracle', '+4.44', '+4.90', '+5.37', '+5.84'],
  ['1td', 'state classifier', '+1.38', '-0.02', '+0.49', '+0.29'],
  ['1td', 'ret_5d', '+0.26', '+0.05', '+1.07', '+1.27'],
  ['2td', 'oracle', '+5.16', '+6.16', '+6.02', '+7.21'],
  ['2td', 'state classifier', '+0.42', '-0.78', '+0.64', '+0.69'],
  ['2td', 'ret_5d', '+0.40', '-0.46', '+1.00', '+1.81'],
  ['3td', 'oracle', '+5.82', '+7.26', '+7.39', '+7.50'],
  ['3td', 'state classifier', '+0.46', '-0.83', '+0.12', '+0.61'],
  ['3td', 'ret_5d', '+0.46', '-0.77', '+1.18', '+1.77'],
];

const choiceRows = [
  ['1td', 'state_classifier', 'ret_20d 361 days, ret_5d 13 days'],
  ['2td', 'state_classifier', 'ret_20d 200 days, ret_5d 173 days'],
  ['3td', 'state_classifier', 'ret_5d 322 days, ret_20d 50 days'],
  ['1td', 'train_best', 'ret_20d 374 days'],
  ['2td', 'train_best', 'ret_20d 373 days'],
  ['3td', 'train_best', 'ret_20d 235 days, kmid2 137 days'],
];

export default function AmvRegimeSleeveLab() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>AMV Regime Sleeve Lab</H1>
        <Text tone="secondary">
          Source: artifacts/amv_bull_pool_regime_sleeve/20260516_142833, 20260516_142854, 20260516_142915.
          Label: T+1 open to T+N close, limit-up entry excluded.
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="+6.99pp" label="3td oracle edge" tone="success" />
        <Stat value="+0.66pp" label="3td static ret_5d edge" tone="warning" />
        <Stat value="+0.09pp" label="3td state classifier edge" tone="danger" />
        <Stat value="ret_5d / ret_20d" label="Classifier almost only chooses" />
      </Grid>

      <Callout tone="warning" title="结论">
        <Text>
          袖子切换存在很高的事后上限, 但当前状态特征 classifier 没学到可交易切换。它没有超过静态动量,
          说明下一步应先研究 oracle 可预测性, 而不是直接把状态模型接回测。
        </Text>
      </Callout>

      <H2>Overall Edge</H2>
      <Table
        headers={['Horizon', 'Strategy', 'Avg edge', 'Avg top return', 'Positive edge days']}
        rows={overallRows}
      />

      <Divider />

      <Grid columns={2} gap={18}>
        <Stack gap={10}>
          <H3>Yearly Edge By Strategy</H3>
          <Table
            headers={['Horizon', 'Strategy', '2023', '2024', '2025', '2026']}
            rows={yearlyRows}
          />
        </Stack>
        <Stack gap={10}>
          <H3>Model Choice Distribution</H3>
          <Text tone="secondary">
            The state classifier mostly collapses to ret_5d / ret_20d and does not discover robust K-line or manual-combo switching.
          </Text>
          <Table headers={['Horizon', 'Strategy', 'Chosen sleeves']} rows={choiceRows} />
        </Stack>
      </Grid>
    </Stack>
  );
}
