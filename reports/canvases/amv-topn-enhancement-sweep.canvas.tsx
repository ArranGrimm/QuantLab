import { BarChart, Callout, Divider, Grid, H1, H2, Stack, Stat, Table, Text } from "cursor/canvas";

const firstRoundRows = [
  ["stop off + hold 6td", "+144.02%", "-14.71%", "52.27%", "264", "当前最优"],
  ["stop off + hold 7td", "+99.68%", "-14.95%", "47.16%", "229", "次优"],
  ["trailing 8/4", "+78.91%", "-16.94%", "48.15%", "189", "无止损分支"],
  ["stop off + bear exit", "+56.69%", "-19.52%", "48.68%", "189", "主动转空退出"],
  ["trailing 10/5", "+55.48%", "-19.56%", "47.06%", "187", "移动止盈较弱"],
  ["stop off 10td", "+50.30%", "-18.85%", "47.54%", "183", "10 交易日无止损"],
  ["stop off + hold 5td", "+21.56%", "-28.46%", "44.22%", "303", "太短"],
];

const noStopRows = [
  ["hold 6td", "+144.02%", "-14.71%", "52.27%", "264", "当前最优"],
  ["hold 7td", "+99.68%", "-14.95%", "47.16%", "229", "次优"],
  ["trailing 8/4", "+78.91%", "-16.94%", "48.15%", "189", "牺牲收益"],
  ["bear exit", "+56.69%", "-19.52%", "48.68%", "189", "没有降回撤"],
  ["trailing 10/5", "+55.48%", "-19.56%", "47.06%", "187", "较弱"],
  ["hold 10td", "+50.30%", "-18.85%", "47.54%", "183", "基准无止损"],
  ["gap 5%", "+47.66%", "-15.77%", "47.54%", "183", "略降收益"],
  ["hold 15td", "+46.73%", "-23.63%", "49.63%", "135", "太长"],
  ["hold 5td", "+21.56%", "-28.46%", "44.22%", "303", "太短"],
];

export default function AmvTopnEnhancementSweep() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV TopN 增强消融</H1>
        <Text tone="secondary">
          使用最新信号 artifacts/amv_topn/20260510_115834，回测至 2026-05-08。当前口径已改为 max_hold_trading_days。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+144.02%" label="最佳净收益" tone="success" />
        <Stat value="-14.71%" label="最佳方案 MaxDD" tone="info" />
        <Stat value="+25.67%" label="10td 5%止损基线" />
        <Stat value="264" label="最佳方案交易数" />
      </Grid>

      <Callout tone="success" title="首要结论">
        自然日持有期结果已废弃。交易日口径下，固定 5% 止损仍是拖累，但幅度回归正常：
        10td 基线 +25.67%，10td 无止损 +50.30%，6td 无止损最佳为 +144.02%。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>第一轮规则对照</H2>
          <BarChart
            categories={["6td off", "7td off", "trail 8/4", "bear exit", "trail 10/5", "10td off", "5td off"]}
            series={[{ name: "净收益", data: [144.02, 99.68, 78.91, 56.69, 55.48, 50.3, 21.56], tone: "success" }]}
            valueSuffix="%"
            height={240}
          />
        </Stack>

        <Stack gap={10}>
          <H2>无固定止损二阶对照</H2>
          <BarChart
            categories={["6td", "7td", "trail 8/4", "bear exit", "trail 10/5", "10td", "15td", "5td"]}
            series={[{ name: "净收益", data: [144.02, 99.68, 78.91, 56.69, 55.48, 50.3, 46.73, 21.56], tone: "info" }]}
            valueSuffix="%"
            height={240}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>第一轮 Top 结果</H2>
      <Table
        headers={["变体", "净收益", "MaxDD", "胜率", "交易数", "解释"]}
        rows={firstRoundRows}
        rowTone={["success", undefined, undefined, undefined, undefined, undefined, "warning"]}
        columnAlign={["left", "right", "right", "right", "right", "left"]}
      />

      <H2>无固定止损分支</H2>
      <Table
        headers={["变体", "净收益", "MaxDD", "胜率", "交易数", "解释"]}
        rows={noStopRows}
        rowTone={["success", undefined, undefined, undefined, undefined, undefined, undefined, "warning", "warning"]}
        columnAlign={["left", "right", "right", "right", "right", "left"]}
      />

      <H2>下一步</H2>
      <Text>
        当前最值得确认的是 6td 无固定止损是否每年都稳。自然日持有期的异常强势不应丢掉，
        后续应以显式 max_hold_calendar_days 分支复刻，而不是恢复有歧义的旧 max_hold_days。
      </Text>
    </Stack>
  );
}
