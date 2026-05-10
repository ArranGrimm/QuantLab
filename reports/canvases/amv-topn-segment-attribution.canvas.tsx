import { BarChart, Callout, Divider, Grid, H1, H2, Stack, Stat, Table, Text } from "cursor/canvas";

const segmentRows = [
  ["全样本", "264", "+72.01万", "52.27%", "+1.12%", "+5.11%", "-3.15%", "+0.12%"],
  ["非 2026", "246", "+83.77万", "53.66%", "+1.32%", "+5.35%", "-3.07%", "+0.21%"],
  ["2024 盈利", "25", "+60.72万", "100.00%", "+8.28%", "+14.72%", "-2.67%", "+1.48%"],
  ["2024 Top10", "10", "+51.13万", "100.00%", "+17.10%", "+26.81%", "-4.13%", "+1.06%"],
  ["2026 全部", "18", "-11.76万", "33.33%", "-1.52%", "+1.77%", "-4.20%", "-1.15%"],
  ["2026 亏损", "12", "-16.30万", "0.00%", "-3.13%", "+1.01%", "-5.41%", "-1.30%"],
];

const monthRows = [
  ["2024-10", "6", "+30.89万", "+17.09%", "83.33%", "+2.07%"],
  ["2024-09", "9", "+11.06万", "+4.77%", "66.67%", "+0.60%"],
  ["2025-01", "6", "+9.81万", "+4.16%", "66.67%", "+1.62%"],
  ["2026-01", "9", "-1.50万", "-0.36%", "55.56%", "+0.03%"],
  ["2026-04", "9", "-10.26万", "-2.67%", "11.11%", "+0.61%"],
];

const winnerRows = [
  ["sh.600839", "2024-10-23", "+61.32%", "+61.89%", "-2.95%", "+3.77%"],
  ["sh.600221", "2024-11-04", "+21.45%", "+52.50%", "-11.25%", "+5.26%"],
  ["sz.000559", "2024-10-31", "+21.18%", "+35.85%", "-0.34%", "+0.67%"],
  ["sz.002456", "2024-10-22", "+14.66%", "+15.06%", "-14.63%", "+7.98%"],
  ["sz.000591", "2024-09-25", "+15.64%", "+30.93%", "+0.00%", "+0.94%"],
];

const loserRows = [
  ["sz.000733", "2026-04-21", "-9.17%", "+0.84%", "-10.01%", "+0.48%"],
  ["sh.688789", "2026-01-16", "-7.19%", "+0.08%", "-9.71%", "+0.00%"],
  ["sz.002389", "2026-04-21", "-4.75%", "+0.27%", "-6.54%", "+0.00%"],
  ["sh.600801", "2026-04-10", "-4.27%", "+0.93%", "-5.11%", "+0.00%"],
  ["sz.000987", "2026-04-30", "-4.13%", "+0.22%", "-4.55%", "-0.75%"],
];

export default function AmvTopnSegmentAttribution() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV TopN 分段归因</H1>
        <Text tone="secondary">
          对 `6td + no stop` 的 trades.csv 重新拼接 signal.parquet，分析 2024 大赢家与 2026 亏损段。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+26.81%" label="2024 Top10 平均 MFE" tone="success" />
        <Stat value="+1.01%" label="2026 亏损平均 MFE" tone="warning" />
        <Stat value="-1.30%" label="2026 亏损入场日表现" tone="warning" />
        <Stat value="2026-04" label="最大亏损月份" />
      </Grid>

      <Callout tone="warning" title="核心结论">
        2024 大赢家来自行情窗口，而不是 rank/score 明显更强。2026 亏损段的 score 仍接近全样本，
        但持有期平均 MFE 只有 +1.01%，MAE 扩大到 -5.41%，说明入场后缺乏向上弹性。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>MFE / MAE 对照</H2>
          <BarChart
            categories={["全样本", "2024盈利", "2024 Top10", "2026全部", "2026亏损"]}
            series={[
              { name: "MFE", data: [5.11, 14.72, 26.81, 1.77, 1.01], tone: "success" },
              { name: "MAE", data: [-3.15, -2.67, -4.13, -4.2, -5.41], tone: "danger" },
            ]}
            valueSuffix="%"
            height={260}
          />
        </Stack>

        <Stack gap={10}>
          <H2>关键月份</H2>
          <BarChart
            categories={["2024-10", "2024-09", "2025-01", "2026-01", "2026-04"]}
            series={[{ name: "PnL", data: [30.89, 11.06, 9.81, -1.5, -10.26], tone: "info" }]}
            valueSuffix="万"
            height={260}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>分段摘要</H2>
      <Table
        headers={["分段", "交易", "PnL", "胜率", "单笔均值", "MFE", "MAE", "入场日"]}
        rows={segmentRows}
        rowTone={[undefined, undefined, "success", "success", "warning", "danger"]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right"]}
      />

      <H2>月份贡献</H2>
      <Table
        headers={["月份", "交易", "PnL", "单笔均值", "胜率", "开盘 gap"]}
        rows={monthRows}
        rowTone={["success", "success", undefined, "warning", "danger"]}
        columnAlign={["left", "right", "right", "right", "right", "right"]}
      />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>2024 大赢家样本</H2>
          <Table
            headers={["代码", "买入", "收益", "MFE", "MAE", "gap"]}
            rows={winnerRows}
            rowTone={["success", undefined, undefined, undefined, undefined]}
            columnAlign={["left", "left", "right", "right", "right", "right"]}
          />
        </Stack>

        <Stack gap={10}>
          <H2>2026 主要亏损样本</H2>
          <Table
            headers={["代码", "买入", "收益", "MFE", "MAE", "gap"]}
            rows={loserRows}
            rowTone={["danger", undefined, undefined, undefined, undefined]}
            columnAlign={["left", "left", "right", "right", "right", "right"]}
          />
        </Stack>
      </Grid>

      <Callout tone="info" title="策略含义">
        继续优化方向不是简单过滤开盘 gap：2026 的平均 gap 并不异常。更值得测试的是入场确认或环境确认，
        例如要求执行日不能收弱，或在 AMV bull 里增加短期市场宽度/赚钱效应确认。
      </Callout>
    </Stack>
  );
}
