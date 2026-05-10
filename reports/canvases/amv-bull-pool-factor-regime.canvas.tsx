import { BarChart, Callout, Divider, Grid, H1, H2, Stack, Stat, Table, Text } from "cursor/canvas";

const factorRows = [
  ["K线振幅收缩", "+1.16pp", "+0.0499", "51.53%", "+6.75pp"],
  ["实体占比偏强", "+1.06pp", "-0.0104", "51.53%", "+13.36pp"],
  ["接近20日新高", "+0.86pp", "+0.0107", "48.65%", "+14.56pp"],
  ["当前组合", "+0.82pp", "+0.0035", "51.89%", "+0.15pp"],
  ["5日动量", "+0.75pp", "-0.0223", "47.93%", "+27.42pp"],
  ["20日高位", "+0.35pp", "-0.0136", "49.73%", "+2.19pp"],
  ["换手放大", "-0.82pp", "-0.0143", "39.64%", "+9.76pp"],
];

const yearRows = [
  ["2024", "K线振幅收缩", "+2.44pp", "+0.0561"],
  ["2024", "当前组合", "+2.12pp", "-0.0267"],
  ["2025", "实体占比偏强", "+2.17pp", "-0.0334"],
  ["2025", "接近20日新高", "+2.03pp", "-0.0120"],
  ["2026", "5日动量", "+4.13pp", "-0.0060"],
  ["2026", "20日动量", "+4.00pp", "-0.0114"],
  ["2026", "当前组合", "+0.05pp", "-0.0002"],
];

const phaseRows = [
  ["early", "K线振幅收缩", "+1.98pp", "+0.0290"],
  ["early", "当前组合", "+1.68pp", "+0.0025"],
  ["middle", "K线振幅收缩", "+1.23pp", "+0.0533"],
  ["middle", "当前组合", "+0.10pp", "-0.0192"],
  ["late", "5日动量", "+2.24pp", "-0.0012"],
  ["late", "实体占比偏强", "+2.07pp", "-0.0184"],
  ["late", "当前组合", "+1.18pp", "+0.0358"],
];

const profitRows = [
  ["赚钱效应弱", "5日动量", "+1.12pp"],
  ["赚钱效应弱", "接近20日新高", "+0.92pp"],
  ["赚钱效应弱", "当前组合", "+0.81pp"],
  ["赚钱效应强", "K线振幅收缩", "+1.76pp"],
  ["赚钱效应强", "实体占比偏强", "+1.61pp"],
  ["赚钱效应强", "当前组合", "+0.82pp"],
];

export default function AmvBullPoolFactorRegime() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Bull Pool 因子标签分析</H1>
        <Text tone="secondary">
          Polars 版，6d 标签，覆盖 2021-04-20 到 2026-04-27；指标为日度 Spearman IC 和 top3 edge。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+1.16pp" label="KLEN 全样本 edge" tone="success" />
        <Stat value="+1.06pp" label="KMID2 全样本 edge" tone="success" />
        <Stat value="+0.05pp" label="当前组合 2026 edge" tone="warning" />
        <Stat value="+4.13pp" label="2026 5日动量 edge" tone="info" />
      </Grid>

      <Callout tone="warning" title="核心结论">
        当前组合不是最稳的单一信号。KLEN 和 KMID2 更像稳定核心；2026 里动量因子明显领先，
        说明固定的“高位+K线确认”权重在不同市场状态下会错配。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>全样本 top3 edge</H2>
          <BarChart
            categories={["KLEN", "KMID2", "新高", "组合", "5d动量", "高位", "换手"]}
            series={[{ name: "edge", data: [1.16, 1.06, 0.86, 0.82, 0.75, 0.35, -0.82], tone: "success" }]}
            valueSuffix="pp"
            height={240}
          />
        </Stack>

        <Stack gap={10}>
          <H2>2026 分化</H2>
          <BarChart
            categories={["5d动量", "20d动量", "卖压低", "KLEN", "KMID2", "组合", "新高"]}
            series={[{ name: "2026 edge", data: [4.13, 4.0, 2.89, 2.8, 1.56, 0.05, -0.53], tone: "info" }]}
            valueSuffix="pp"
            height={240}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>全样本因子</H2>
      <Table
        headers={["因子", "top3 edge", "IC", "正 edge 天数", "Hit15 edge"]}
        rows={factorRows}
        rowTone={["success", "success", undefined, undefined, undefined, undefined, "danger"]}
        columnAlign={["left", "right", "right", "right", "right"]}
      />

      <H2>年度重点</H2>
      <Table
        headers={["年份", "最有代表性因子", "top3 edge", "IC"]}
        rows={yearRows}
        rowTone={["success", undefined, "success", undefined, "info", "info", "warning"]}
        columnAlign={["left", "left", "right", "right"]}
      />

      <H2>AMV Bull 阶段</H2>
      <Table
        headers={["阶段", "因子", "top3 edge", "IC"]}
        rows={phaseRows}
        rowTone={["success", undefined, "success", "warning", "info", "success", undefined]}
        columnAlign={["left", "left", "right", "right"]}
      />

      <H2>赚钱效应强弱</H2>
      <Table
        headers={["分组", "因子", "top3 edge"]}
        rows={profitRows}
        rowTone={[undefined, undefined, undefined, "success", "success", undefined]}
        columnAlign={["left", "left", "right"]}
      />

      <Callout tone="info" title="对 LTR 的启发">
        这不是单个因子线性 IC 很强的问题，而是不同市场状态下 topN 极值选择的优势来源在变化。
        LTR 值得做，但必须加入市场状态特征，让模型学习何时偏 KLEN/KMID2、何时偏动量。
      </Callout>
    </Stack>
  );
}
