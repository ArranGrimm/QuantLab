import { BarChart, Callout, Divider, Grid, H1, H2, Stack, Stat, Table, Text } from "cursor/canvas";

const refRows = [
  ["2021", "+1.10%", "+0.40pp", "+16.55%", "-4.00%", "81"],
  ["2022", "+1.14%", "+0.77pp", "+22.56%", "-2.72%", "104"],
  ["2023", "+0.64%", "+1.25pp", "+14.73%", "-4.10%", "137"],
  ["2024", "+3.30%", "+2.12pp", "+78.90%", "-1.95%", "111"],
  ["2025", "+0.63%", "-0.75pp", "+7.22%", "-3.80%", "89"],
  ["2026", "+1.10%", "+0.05pp", "+7.05%", "-1.58%", "33"],
];

const componentRows = [
  ["当前组合", "+0.82pp", "5/6", "+2.12pp", "-0.75pp", "+0.05pp"],
  ["20日高位", "+0.35pp", "5/6", "+0.33pp", "+0.23pp", "-0.14pp"],
  ["接近20日新高", "+0.87pp", "5/6", "+0.09pp", "+2.03pp", "-0.53pp"],
  ["K线振幅收缩", "+1.16pp", "6/6", "+2.44pp", "+1.56pp", "+2.80pp"],
  ["实体占比偏强", "+1.06pp", "6/6", "+0.82pp", "+2.17pp", "+1.56pp"],
];

const gridRows = [
  ["P3/K0.5/R0", "+0.93pp", "5/6", "+0.41pp", "价格权重更高"],
  ["P2/K0.5/R0", "+0.82pp", "5/6", "+0.05pp", "当前基线"],
  ["P3/K1/R0", "+0.69pp", "5/6", "+0.25pp", "K线权重略高"],
  ["P1/K0.5/R0", "+0.63pp", "4/6", "+0.22pp", "价格权重偏低"],
  ["P2/K1/R0", "+0.63pp", "4/6", "+0.22pp", "K线权重偏高"],
];

export default function AmvBullPoolYearlyFactor() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Bull Pool 因子年度稳定性</H1>
        <Text tone="secondary">
          回到 Rust 回测前的排序评价框架，按年份拆解 top3 高位+K线确认组合；主视角使用 6d horizon。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+2.12pp" label="2024 edge" tone="success" />
        <Stat value="+0.05pp" label="2026 edge" tone="warning" />
        <Stat value="5/6" label="当前组合正 edge 年份" />
        <Stat value="2026-04-27" label="6d-only 数据截止" tone="info" />
      </Grid>

      <Callout tone="warning" title="结论">
        你的怀疑是对的：当前组合在 2024 显著更适配，6d 单笔均值 +3.30%，相对随机 +2.12pp。
        修正为 6d-only 后，2026 edge 降到 +0.05pp；权重网格里 P3/K0.5/R0 仍比当前 P2/K0.5/R0 略强。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>当前组合年度 edge</H2>
          <BarChart
            categories={["2021", "2022", "2023", "2024", "2025", "2026"]}
            series={[{ name: "edge", data: [0.4, 0.77, 1.25, 2.12, -0.75, 0.05], tone: "success" }]}
            valueSuffix="pp"
            height={240}
          />
        </Stack>

        <Stack gap={10}>
          <H2>组成因子 6d edge</H2>
          <BarChart
            categories={["组合", "高位", "新高", "缩振", "实体强"]}
            series={[{ name: "全样本 edge", data: [0.9, 0.36, 0.88, 1.09, 1.02], tone: "info" }]}
            valueSuffix="pp"
            height={240}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>当前组合分年份</H2>
      <Table
        headers={["年份", "单笔均值", "相对随机", "rolling NAV", "MaxDD", "样本日"]}
        rows={refRows}
        rowTone={[undefined, undefined, undefined, "success", "warning", "warning"]}
        columnAlign={["left", "right", "right", "right", "right", "right"]}
      />

      <H2>组成因子对照</H2>
      <Table
        headers={["ranker", "全样本 edge", "正 edge 年份", "2024", "2025", "2026*"]}
        rows={componentRows}
        rowTone={[undefined, undefined, undefined, "success", "success"]}
        columnAlign={["left", "right", "right", "right", "right", "right"]}
      />

      <H2>权重网格启发</H2>
      <Table
        headers={["组合", "全样本 edge", "正 edge 年份", "2026* edge", "解释"]}
        rows={gridRows}
        rowTone={["success", undefined, undefined, undefined, undefined]}
        columnAlign={["left", "right", "right", "right", "left"]}
      />

      <Callout tone="info" title="重要口径">
        多 horizon 版本因为全局要求 20d 前瞻收益有效，pool 被截到 2026-02-02；6d-only 版本可以覆盖到 2026-04-27。
        这说明 2026-04 不是 AMV 数据缺失，而是旧探索脚本的 20d 前瞻过滤造成了样本口径错位。
      </Callout>
    </Stack>
  );
}
