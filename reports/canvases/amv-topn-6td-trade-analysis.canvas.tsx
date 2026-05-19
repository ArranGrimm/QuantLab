import { BarChart, Callout, Divider, Grid, H1, H2, Stack, Stat, Table, Text } from "cursor/canvas";

const annualRows = [
  ["2021", "+0.48%", "-9.35%", "243"],
  ["2022", "+38.36%", "-5.10%", "242"],
  ["2023", "+14.18%", "-4.22%", "242"],
  ["2024", "+49.00%", "-14.43%", "242"],
  ["2025", "+12.53%", "-10.81%", "243"],
  ["2026 YTD", "-8.79%", "-9.78%", "80"],
];

const topProfitRows = [
  ["sh.600839", "2024-10-23", "2024-10-31", "+61.32%", "+17.54万"],
  ["sh.603667", "2025-01-16", "2025-01-24", "+21.83%", "+8.52万"],
  ["sh.600221", "2024-11-04", "2024-11-12", "+21.45%", "+7.98万"],
  ["sz.000559", "2024-10-31", "2024-11-08", "+21.18%", "+7.53万"],
  ["sh.601609", "2025-08-15", "2025-08-25", "+15.03%", "+6.28万"],
];

const topLossRows = [
  ["sz.000733", "2026-04-21", "2026-04-29", "-9.17%", "-3.95万"],
  ["sz.000516", "2024-11-13", "2024-11-21", "-8.93%", "-3.75万"],
  ["sh.688789", "2026-01-16", "2026-01-26", "-7.19%", "-3.23万"],
  ["sh.600755", "2025-08-27", "2025-09-04", "-6.50%", "-2.94万"],
  ["sz.002831", "2024-11-14", "2024-11-22", "-6.00%", "-2.44万"],
];

const missedStopRows = [
  ["sh.601607", "2022-03-14", "-6.26%", "+28.11%", "+34.36pp"],
  ["sz.002456", "2024-10-22", "-11.00%", "+14.66%", "+25.66pp"],
  ["sz.000876", "2022-06-28", "-7.66%", "+3.12%", "+10.79pp"],
  ["sz.002217", "2026-04-21", "-10.09%", "-0.35%", "+9.74pp"],
  ["sh.600219", "2021-07-23", "-8.72%", "-1.87%", "+6.85pp"],
];

export default function AmvTopn6tdTradeAnalysis() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV TopN 6td 交易归因</H1>
        <Text tone="secondary">
          基线为 top3 高位+K线确认 + 6td + 无固定止损。分析产物来自 analysis_20260510_125808。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+144.02%" label="净收益" tone="success" />
        <Stat value="-14.71%" label="最大回撤" tone="info" />
        <Stat value="264" label="交易数" />
        <Stat value="52.27%" label="胜率" />
      </Grid>

      <Callout tone="warning" title="当前判断">
        6td 无止损值得作为新基线，但还不是最终策略。Top10 盈利贡献了正收益的 40.45%，几乎覆盖净利润的
        97.74%，说明收益集中度高；2026 YTD 当前为 -8.79%，需要先确认近期回撤不是结构性失效。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>年度收益</H2>
          <BarChart
            categories={["2021", "2022", "2023", "2024", "2025", "2026"]}
            series={[{ name: "年度收益", data: [0.48, 38.36, 14.18, 49.0, 12.53, -8.79], tone: "success" }]}
            valueSuffix="%"
            height={240}
          />
        </Stack>

        <Stack gap={10}>
          <H2>单笔分布</H2>
          <Grid columns={2} gap={12}>
            <Stat value="+1.12%" label="单笔均值" />
            <Stat value="+0.19%" label="单笔中位数" />
            <Stat value="-4.74%" label="P10" tone="warning" />
            <Stat value="+6.33%" label="P90" tone="success" />
          </Grid>
          <Text tone="secondary">
            平均盈利约 1.26 万，平均亏损约 0.81 万，payoff ratio 为 1.56。不是靠极高胜率，而是靠盈亏比和少数大票拉动。
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <H2>年度表现</H2>
      <Table
        headers={["年份", "收益", "年内 MaxDD", "交易日"]}
        rows={annualRows}
        rowTone={[undefined, "success", undefined, "success", undefined, "warning"]}
        columnAlign={["left", "right", "right", "right"]}
      />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>最大盈利</H2>
          <Table
            headers={["代码", "买入", "卖出", "单笔收益", "PnL"]}
            rows={topProfitRows}
            rowTone={["success", undefined, undefined, undefined, undefined]}
            columnAlign={["left", "left", "left", "right", "right"]}
          />
        </Stack>

        <Stack gap={10}>
          <H2>最大亏损</H2>
          <Table
            headers={["代码", "买入", "卖出", "单笔收益", "PnL"]}
            rows={topLossRows}
            rowTone={["danger", undefined, undefined, undefined, undefined]}
            columnAlign={["left", "left", "left", "right", "right"]}
          />
        </Stack>
      </Grid>

      <H2>5% 止损错杀</H2>
      <Grid columns={4} gap={14}>
        <Stat value="29" label="匹配到止损交易" />
        <Stat value="18" label="后续反弹交易" tone="warning" />
        <Stat value="62.07%" label="错杀比例" tone="warning" />
        <Stat value="+18.56万" label="错杀 PnL 差额" tone="success" />
      </Grid>
      <Table
        headers={["代码", "买入", "止损结果", "无止损结果", "差值"]}
        rows={missedStopRows}
        rowTone={["danger", "danger", undefined, undefined, undefined]}
        columnAlign={["left", "left", "right", "right", "right"]}
      />

      <Callout tone="info" title="下一步">
        先不要继续调参。下一步更应该拆 2024 年大赢家和 2026 年亏损段：看它们的 AMV 位置、买入 gap、行业/主题集中度，
        再决定是否加入“过热过滤”或“弱市主动降仓”。
      </Callout>
    </Stack>
  );
}
