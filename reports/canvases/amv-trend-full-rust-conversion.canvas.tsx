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

const labelRows = [
  ['Trend P1/K0.5/PB1/RV1', '+256.94%', '-4.51%', '+207.10%', '3', '2025/2026 edge < 0'],
  ['Trend P1/K1/PB1/RV1', '+243.77%', '-4.38%', '+201.91%', '3', '2025/2026 edge < 0'],
  ['Trend P2/K0.5/PB2/RV1', '+216.01%', '-4.11%', '+171.89%', '3', '2025/2026 edge < 0'],
  ['PB1/CP0/RV0', '+242.58%', '-23.29%', '+124.83%', '3', '5/5 stable years'],
  ['PB3/CP1/RV0', '+215.37%', '-20.29%', '+110.09%', '3', '5/5 stable years'],
];

const rustRows = [
  ['Trend P1/K0.5/PB1/RV1', 'static strict', '+85.85%', '+126.15%', '-43.24%', '276'],
  ['Trend P1/K1/PB1/RV1', 'static strict', '+76.15%', '+116.26%', '-42.12%', '276'],
  ['Trend P2/K0.5/PB2/RV1', 'static strict', '+123.58%', '+171.28%', '-38.27%', '276'],
  ['PB1/CP0/RV0', 'static strict', '+190.28%', '+230.07%', '-43.43%', '276'],
  ['PB3/CP1/RV0', 'rolling21 refill', '+99.62%', '+130.78%', '-20.70%', '1,650'],
  ['Trend P1/K1/PB1/RV1', 'rolling21 refill', '+60.17%', '+95.13%', '-11.58%', '1,633'],
];

const yearlyRows = [
  ['Trend P1/K0.5/PB1/RV1 rolling', '+26.48%', '-4.47%', '+3.81%', '+9.94%', '+10.52%', '+3.00%'],
  ['Trend P1/K1/PB1/RV1 rolling', '+31.54%', '-5.78%', '+0.11%', '+14.35%', '+10.17%', '+2.48%'],
  ['PB3/CP1/RV0 rolling', '-0.41%', '+8.20%', '-3.37%', '+28.34%', '+29.73%', '+15.15%'],
  ['PB1/CP0/RV0 rolling', '-2.32%', '+5.39%', '+7.08%', '+26.31%', '+21.11%', '+12.32%'],
];

export default function AmvTrendFullRustConversion() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Trend Full Grid: Label 到 Rust 转换</H1>
        <Text tone="secondary">
          对比 Mac full grid 中 trend-only 新 Top 候选与普通 pullback sleeve 的 Python executable label、
          Rust 静态/rolling 回测和年度结构。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+256.94%" label="Trend full top label NAV" tone="success" />
        <Stat value="+123.58%" label="Trend full top best Rust static" tone="warning" />
        <Stat value="+60.17%" label="Trend full top best Rust rolling" tone="warning" />
        <Stat value="+99.62%" label="PB3 rolling Rust" tone="success" />
      </Grid>

      <Callout tone="warning" title="关键结论">
        trend-only full top 不是涨停污染问题。它的 Python label 上限更高，但 Rust 真实组合里的 gross edge
        本身就掉得更多；普通 pullback 的 label 年度结构更稳定，尤其 2025/2026 为正 edge，所以 rolling Rust
        更能保留优势。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Label NAV vs Rust Net</H2>
          <BarChart
            categories={['Trend P1/K0.5', 'Trend P1/K1', 'Trend P2/K0.5', 'PB1 static', 'PB3 rolling']}
            valueSuffix="%"
            height={270}
            series={[
              { name: 'Python executable label NAV', data: [256.94, 243.77, 216.01, 242.58, 215.37], tone: 'success' },
              { name: 'Rust net return', data: [85.85, 76.15, 123.58, 190.28, 99.62], tone: 'warning' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Source: trend full grid summary and Rust report summaries. Trend Rust uses static strict for static candidates;
            PB3 uses rolling21 refill representative.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Rust Gross vs Cost Drag</H2>
          <BarChart
            categories={['Trend P1/K0.5 rolling', 'Trend P1/K1 rolling', 'Trend P2/K0.5 rolling', 'PB3 rolling', 'PB1 rolling']}
            valueSuffix="%"
            height={270}
            series={[
              { name: 'Rust gross return', data: [90.55, 95.13, 91.57, 130.78, 118.09], tone: 'success' },
              { name: 'Cost drag', data: [33.58, 34.96, 34.84, 31.15, 28.68], tone: 'danger' },
            ]}
          />
          <Text size="small" tone="tertiary">
            Cost drag is gross return minus net return, percentage points of initial capital.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>Python Label 对照</H2>
      <Table
        headers={['Candidate', 'Label Exec NAV', 'Label MaxDD', 'Label CTC NAV', 'Rank q95', 'Year structure']}
        rows={labelRows}
        rowTone={['warning', 'warning', 'warning', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
      />

      <H2>Rust 真实组合对照</H2>
      <Table
        headers={['Candidate', 'Scenario', 'Net', 'Gross', 'MaxDD', 'Trades']}
        rows={rustRows}
        rowTone={['warning', 'warning', 'warning', 'success', 'success', 'warning']}
        columnAlign={['left', 'left', 'right', 'right', 'right', 'right']}
      />

      <H2>Rolling 年度收益</H2>
      <Table
        headers={['Candidate', '2021', '2022', '2023', '2024', '2025', '2026']}
        rows={yearlyRows}
        rowTone={['warning', 'warning', 'success', 'success']}
        columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
      />

      <Callout tone="info" title="怎么解释损耗">
        Python label 是每日 Top3 cohort 的重叠净值诊断；Rust static 是真实持仓占用后每 6td 换仓，
        Rust rolling 又加入 no-repeat、资金、手数和成本。trend-only 的 label 更依赖某些平滑 cohort
        路径，转成真实组合后 gross edge 掉到约 <Code>90%~95%</Code>；PB3 rolling gross 仍有
        <Code>+130.78%</Code>，所以最终更强。
      </Callout>

      <Divider />

      <Text size="small" tone="tertiary">
        Artifacts: <Code>reports/amv_trend_filter_full_top_rust_summary.json</Code>,{' '}
        <Code>reports/amv_executable_sleeve_rust_yearly.json</Code>,{' '}
        <Code>artifacts/amv_executable_trend_filter_grid/20260520_212625/summary.json</Code>.
      </Text>
    </Stack>
  );
}
