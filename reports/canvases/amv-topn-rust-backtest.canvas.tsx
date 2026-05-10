import { BarChart, Callout, Divider, Grid, H1, H2, Stack, Stat, Table, Text } from "cursor/canvas";

const rows = [
  ["5d", "+29.88%", "-35.99%", "49.48%", "388", "0.301", "+82.93%", "26.53 万"],
  ["10d", "+65.54%", "-14.69%", "54.58%", "240", "0.186", "+103.68%", "19.07 万"],
  ["20d", "+10.26%", "-28.01%", "44.24%", "165", "0.128", "+29.74%", "9.74 万"],
];

export default function AmvTopnRustBacktest() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV TopN Rust 真实回测</H1>
        <Text tone="secondary">
          信号为 top3 高位+K线确认 P2/K0.5/R0，执行方式为 T 日收盘出信号、T+1 开盘买入、收盘退出评估。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+65.54%" label="10d 净收益最佳" tone="success" />
        <Stat value="-14.69%" label="10d 最大回撤" tone="info" />
        <Stat value="557" label="执行信号日" />
        <Stat value="78" label="执行日 AMV gate 屏蔽信号" tone="warning" />
      </Grid>

      <Callout tone="success" title="当前最优档位">
        10d 明显优于 5d 和 20d：净收益最高、回撤最低、胜率最高。5d 交易太频繁且成本吃掉大量毛收益；20d
        则出现更明显的回撤和信号兑现衰减。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>收益与回撤</H2>
          <BarChart
            categories={["5d", "10d", "20d"]}
            series={[
              { name: "净收益", data: [29.88, 65.54, 10.26], tone: "success" },
              { name: "最大回撤", data: [35.99, 14.69, 28.01], tone: "danger" },
            ]}
            valueSuffix="%"
            height={220}
          />
        </Stack>

        <Stack gap={10}>
          <H2>交易强度</H2>
          <BarChart
            categories={["5d", "10d", "20d"]}
            series={[{ name: "总交易数", data: [388, 240, 165], tone: "info" }]}
            height={220}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>三档持有期对照</H2>
      <Table
        headers={["持有期", "净收益", "最大回撤", "胜率", "交易数", "日均交易", "毛收益", "总成本"]}
        rows={rows}
        rowTone={[undefined, "success", undefined]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right"]}
      />

      <H2>读数</H2>
      <Text>
        本次结果说明研究阶段的 20d 单笔均值不能直接等同于组合真实收益。真实交易里，资金占用、T+1
        开盘执行、涨停过滤、交易成本和止损都会改变结果路径。
      </Text>
      <Text>
        后续优先围绕 10d 做细化：验证止损开关、AMV 转空是否主动清仓、以及是否用风险因子做过滤或仓位控制。
      </Text>
    </Stack>
  );
}
