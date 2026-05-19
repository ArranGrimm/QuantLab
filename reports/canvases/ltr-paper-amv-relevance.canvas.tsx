import {
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

const takeawayRows = [
  ['核心观点', '不要先预测收益再排序；应该直接优化“谁排前面”。'],
  ['最相关部分', 'Listwise / NDCG / Precision@K，与我们 top3/top5/top10 实验直接对应。'],
  ['对当前工作的意义', '我们现在的手工组合分数，本质上已经在做简化版 Learning-to-Rank。'],
  ['不应照搬部分', '论文主线是月频美股 long-short；我们是 A股日频 long-only + AMV gate。'],
];

const actionRows = [
  ['立刻可用', '把评估指标从单笔均值改成 Precision@K、topN NAV、MaxDD、年份拆分。'],
  ['下一步可试', '用 LightGBM ranker / LambdaRank 学习 AMV bull 宽池内 top3 排序。'],
  ['需要小心', '不要把风险因子直接混进主排序；它更适合做过滤、仓位或退出。'],
  ['验证优先级', '先做真实 Rust 回测，再决定是否上 Learning-to-Rank 训练。'],
];

export default function LTRPaperAMVRelevance() {
  return (
    <Stack gap={18}>
      <Stack gap={6}>
        <H1>Learning-to-Rank 论文对 AMV 研究的启发</H1>
        <Text tone="secondary">
          论文: <Code>Empirical Asset Pricing via Learning-to-Rank</Code>。
          重点不是照搬模型，而是借鉴它的任务定义：直接优化横截面排序，而不是预测绝对收益。
        </Text>
      </Stack>

      <Grid columns={3} gap={14}>
        <Stat value="高度相关" label="与当前 AMV topN 排序" tone="success" />
        <Stat value="Listwise" label="最值得借鉴的范式" tone="success" />
        <Stat value="先回测" label="当前优先级" tone="info" />
      </Grid>

      <Callout tone="success" title="一句话结论">
        这篇论文支持我们当前从“预测收益”转向“直接挑 topN 排序”的路线。
        但它不替代真实回测；它主要告诉我们下一阶段训练模型时应该考虑 rank objective。
      </Callout>

      <Divider />

      <H2>核心拆解</H2>
      <Table
        headers={['问题', '结论']}
        rows={takeawayRows}
        rowTone={['success', 'success', 'info', undefined]}
        columnAlign={['left', 'left']}
      />

      <H2>对我们的行动建议</H2>
      <Table
        headers={['层级', '怎么用']}
        rows={actionRows}
        rowTone={['success', 'info', 'info', 'success']}
        columnAlign={['left', 'left']}
      />

      <H2>和当前结果怎么连起来</H2>
      <Text>
        我们现在的 <Code>top3 高位+K线确认 P2/K0.5/R0</Code> 不是收益预测模型，
        而是一个手工排序器。论文的核心观点正是：投资组合真正关心的是排序前几名是否对，
        不是每只股票未来收益能不能预测得很准。
      </Text>
      <Text>
        所以下一步最自然的升级不是继续手工加因子，而是在 AMV bull 宽池里训练一个
        <Code>Learning-to-Rank</Code> 模型，让模型直接学习“哪些股票该排进 top3”。
      </Text>

      <Divider />

      <Text size="small" tone="tertiary">
        结论适用边界: 论文使用美股月频 long-short 组合；当前项目是 A股日频 long-only，
        且带 AMV 市场状态过滤。因此只能借鉴任务定义和评估方式，不能直接套用论文收益数字。
      </Text>
    </Stack>
  );
}
