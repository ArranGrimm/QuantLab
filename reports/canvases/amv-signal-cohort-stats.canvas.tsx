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

const categories = ['Reference P2', 'P3', 'PB3/CP1', 'Trend label top', 'Trend Rust top'];

const cohortRows = [
  ['Reference P2', '+98.55%', '+45.16%', '+89.98%', '+0.282%', '-3.330%', '55.1%', '+12.22%', '+96.23%', '+214.00%'],
  ['P3', '+102.93%', '+48.36%', '+105.19%', '+0.305%', '-3.182%', '55.3%', '+31.75%', '+86.03%', '+187.08%'],
  ['PB3/CP1', '+186.48%', '+109.64%', '+176.58%', '+0.512%', '-5.194%', '53.8%', '+119.61%', '+142.01%', '+420.40%'],
  ['Trend label top', '+229.26%', '+141.10%', '+227.85%', '+0.443%', '-2.376%', '57.4%', '+157.84%', '+198.40%', '+370.63%'],
  ['Trend Rust top', '+215.39%', '+130.90%', '+219.09%', '+0.441%', '-2.533%', '57.6%', '+145.90%', '+199.60%', '+306.67%'],
];

const accountRows = [
  ['Reference P2', 'static strict', '+170.80%', '主线旧 reference，static 真实账户强于 cohort refill 诊断'],
  ['P3', 'static strict', '+201.69%', '主基线替换候选，价值主要来自 static cadence 与低换手'],
  ['PB3/CP1', 'rolling21 refill', '+99.62%', 'rolling 真实账户承接较好，互补 sleeve 代表'],
  ['Trend label top', 'rolling21 refill', '+56.97%', 'cohort 信号质量最高之一，但 rolling 真实账户承接弱'],
  ['Trend Rust top', 'rolling21 refill / duplicate', '+60.17% / +106.50%', 'duplicate 证明 no-repeat 压制明显，但仍需解释集中度与成本'],
];

const interpretationRows = [
  ['Trend-only', 'event-time cohort 最强', '真实 rolling 受 no-repeat、资金暴露、成本压制', '作为进攻候选库，不替代主线'],
  ['PB3/CP1', 'cohort 次强，sleeve worst 仍高', 'Rust rolling 承接较好', '当前唯一 pullback 代表'],
  ['P3', 'cohort 不惊艳', 'Rust static strict 最强', '主基线替换候选'],
  ['Reference P2', 'cohort 和 P3 接近但略弱', 'Rust static 已被 P3 超过', '保留为 reference'],
];

export default function AmvSignalCohortStats() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Signal Cohort Diagnostics</H1>
        <Text tone="secondary">
          对比主线、P3、PB3 pullback 与 trend-only top 的 Python rolling cohort 多统计。
          口径为 <Code>T+1 open -&gt; D+7 close</Code>、horizon 6、Top3 / refill Top10，起始日期 2021-01-01。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+229.26%" label="Trend label top cohort NAV" tone="success" />
        <Stat value="+186.48%" label="PB3/CP1 cohort NAV" tone="info" />
        <Stat value="+102.93%" label="P3 cohort NAV" tone="neutral" />
        <Stat value="+201.69%" label="P3 Rust static strict" tone="success" />
      </Grid>

      <Callout tone="warning" title="读法">
        这张 Canvas 展示的是信号质量诊断，不是账户收益替代品。trend-only 的 cohort 统计最强，
        但 Rust account NAV 仍要受重复持仓、资金暴露、手数和成本约束；P3 的 cohort 不突出，
        但 static 真实账户表现最好。
      </Callout>

      <Grid columns="1.15fr 0.85fr" gap={18}>
        <Stack gap={10}>
          <H2>Refill Top10: event-time cohort NAV vs cost-adjusted NAV</H2>
          <BarChart
            categories={categories}
            valueSuffix="%"
            height={285}
            series={[
              { name: 'Event-time cohort NAV', data: [98.55, 102.93, 186.48, 229.26, 215.39], tone: 'success' },
              { name: 'Cost-adjusted NAV, 0.35% round trip', data: [45.16, 48.36, 109.64, 141.1, 130.9], tone: 'warning' },
              { name: 'Dense calendar zero-return NAV', data: [89.98, 105.19, 176.58, 227.85, 219.09], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Axis: strategy sleeve on X, NAV in percent on Y. Source:
            <Code>reports/amv_signal_cohort_stats_main_pullback_trend.json</Code>, refill Top10 scenario.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Daily Top3 return distribution</H2>
          <BarChart
            categories={categories}
            valueSuffix="%"
            height={285}
            series={[
              { name: 'Median daily return', data: [0.282, 0.305, 0.512, 0.443, 0.441], tone: 'success' },
              { name: 'P10 daily return', data: [-3.33, -3.182, -5.194, -2.376, -2.533], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Axis: strategy sleeve on X, daily Top3 average return in percent on Y. P10 is the 10th percentile loss-side tail.
          </Text>
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Six cohort sleeves: worst / median / best path NAV</H2>
          <BarChart
            categories={categories}
            valueSuffix="%"
            height={300}
            series={[
              { name: 'Worst sleeve NAV', data: [12.22, 31.75, 119.61, 157.84, 145.9], tone: 'danger' },
              { name: 'Median sleeve NAV', data: [96.23, 86.03, 142.01, 198.4, 199.6], tone: 'info' },
              { name: 'Best sleeve NAV', data: [214.0, 187.08, 420.4, 370.63, 306.67], tone: 'success' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Axis: strategy sleeve on X, six event-time cohort sleeve NAV in percent on Y. Shows path sensitivity beyond the mean.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Cohort signal quality vs Rust account NAV</H2>
          <Table
            headers={['Sleeve', 'Representative Rust mode', 'Rust net', 'Interpretation']}
            rows={accountRows}
            rowTone={['neutral', 'success', 'info', 'warning', 'warning']}
            columnAlign={['left', 'left', 'right', 'left']}
          />
          <Text size="small" tone="tertiary">
            Rust values come from existing <Code>bt-amv-topn</Code> reports. Modes differ intentionally:
            P3 is evaluated as static, PB3 and trend-only as rolling diagnostics.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>Refill Top10 cohort table</H2>
      <Table
        headers={[
          'Sleeve',
          'Cohort NAV',
          'Cost adj NAV',
          'Dense NAV',
          'Median daily',
          'P10 daily',
          'Win rate',
          'Sleeve worst',
          'Sleeve median',
          'Sleeve best',
        ]}
        rows={cohortRows}
        rowTone={['neutral', 'neutral', 'info', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <H2>Interpretation Map</H2>
      <Table
        headers={['Family', 'Cohort diagnosis', 'Rust account diagnosis', 'Current role']}
        rows={interpretationRows}
        rowTone={['warning', 'info', 'success', 'neutral']}
        columnAlign={['left', 'left', 'left', 'left']}
      />

      <Callout tone="info" title="Current conclusion">
        trend-only 的 event-time cohort 统计最强，说明信号质量不是幻觉；但它不自动转化为账户收益。
        当前主线仍更适合 <Code>P3 static</Code>，互补 sleeve 仍更适合 <Code>PB3/CP1 rolling</Code>。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        Source: <Code>reports/amv_signal_cohort_stats_main_pullback_trend.json</Code>.
        Cost-adjusted cohort uses a simple 0.35% round-trip deduction per cohort cycle. Dense calendar fills non-signal dates with zero return.
      </Text>
    </Stack>
  );
}
