// @ts-nocheck
import {
  BarChart,
  Callout,
  Code,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
} from 'cursor/canvas';

const p3Events = [
  ['Has limit-up 5d', '21', '+622.8K', '+8.99%', '57.1%'],
  ['Has limit-up 10d', '22', '+623.8K', '+8.60%', '59.1%'],
  ['Has limit-up 20d', '24', '+649.7K', '+8.19%', '62.5%'],
  ['Yesterday limit-up', '14', '+598.1K', '+12.81%', '71.4%'],
  ['Re-board after pullback', '16', '+644.9K', '+11.81%', '68.8%'],
  ['One-word LU in 10d', '10', '+336.5K', '+9.80%', '60.0%'],
];

const p3Buckets = [
  ['After LU 1-3d', '14', '+598.1K', '+12.81%', '71.4%'],
  ['After LU 4-10d', '3', '+47.7K', '+3.40%', '66.7%'],
  ['After LU 11-20d', '3', '+62.2K', '+7.63%', '100.0%'],
  ['After LU >20d', '217', '+284.8K', '+0.62%', '49.8%'],
  ['No prior LU', '37', '+15.7K', '+0.33%', '56.8%'],
];

const p3Streak = [
  ['Last first board', '204', '+937.1K', '+1.66%', '54.9%'],
  ['Last 2 boards', '29', '-122.2K', '-1.41%', '34.5%'],
  ['Last 3+ boards', '4', '+177.8K', '+12.26%', '25.0%'],
  ['No prior LU', '37', '+15.7K', '+0.33%', '56.8%'],
];

const pb3Events = [
  ['Has limit-up 20d', '229', '+164.5K', '+1.98%', '49.8%'],
  ['No limit-up 20d', '1421', '+333.6K', '+0.88%', '47.8%'],
  ['First-board pullback', '11', '+16.5K', '+4.42%', '54.5%'],
  ['Recent failed LU 5d', '22', '-2.3K', '-0.43%', '36.4%'],
  ['Yesterday limit-up', '2', '-3.6K', '-7.53%', '0.0%'],
];

const supportedRows = [
  ['near-N-day close limit-up', 'supported', 'raw OHLC + previous raw close approximation'],
  ['days since previous limit-up', 'supported', 'per-code last close limit-up index'],
  ['limit-up streak / first board', 'supported', 'consecutive close limit-up count'],
  ['failed limit-up', 'supported approximation', 'high touched limit, close did not close limit-up'],
  ['one-word limit-up', 'supported approximation', 'open/low/close all at limit-up'],
  ['first-board pullback / reclaim / re-board', 'supported approximation', 'daily-bar event logic'],
  ['seal amount / seal time / open count', 'not supported', 'needs minute/tick/Level2'],
  ['official ex-right limit reference', 'not supported yet', 'stock_daily has no official pre_close'],
];

const rustRows = [
  ['Reboard / reclaim', '-21.54%', '68.57%', '268', '144 / 118d', '-7.51%', 'reject'],
  ['Recent LU ranked', '+46.25%', '59.87%', '269', '144 / 118d', '-7.31%', 'too much DD'],
  ['First-board pullback', '+85.32%', '56.50%', '271', '3 / 3d', '+20.16%', 'best first pass'],
];

const firstBoardYearly = [
  ['2021', '+27.40%'],
  ['2022', '+45.02%'],
  ['2023', '-11.64%'],
  ['2024', '+1.42%'],
  ['2025', '-10.05%'],
  ['2026', '+20.16%'],
];

const holdRiskRows = [
  ['Base 3td', '+85.21%', '67.60%', '430', '+166.70%', 'too costly'],
  ['Base 5td', '+183.57%', '45.35%', '306', '+250.10%', 'best return'],
  ['Base 6td', '+85.32%', '56.50%', '271', '+137.89%', 'baseline'],
  ['Low-vol 5td', '+16.94%', '21.74%', '202', '+42.94%', 'too weak'],
  ['Quality 6td', '+53.36%', '21.71%', '107', '+67.67%', 'best defensive'],
];

const holdRiskYearly = [
  ['2021', '+48.36%', '+9.24%'],
  ['2022', '+8.28%', '+3.53%'],
  ['2023', '+3.73%', '-6.02%'],
  ['2024', '+17.65%', '+24.02%'],
  ['2025', '+26.48%', '+9.81%'],
  ['2026', '+10.44%', '+5.95%'],
];

const drawdownRows = [
  ['Base 5td', '2023-08-03', '2024-02-05', '2025-02-19', '45.35%', '43', '-431.9K', '25.6%'],
  ['Quality 6td', '2021-06-03', '2022-10-28', '2024-11-05', '21.71%', '29', '-99.5K', '31.0%'],
];

const drawdownWorstMonths = [
  ['2023-08', '6', '-98.6K', '-4.37%', '16.7%'],
  ['2023-09', '9', '-84.8K', '-2.73%', '22.2%'],
  ['2023-12', '9', '-85.0K', '-3.09%', '22.2%'],
  ['2024-01', '6', '-83.4K', '-4.99%', '16.7%'],
  ['2024-02', '1', '-54.7K', '-21.26%', '0.0%'],
];

const riskProfileRows = [
  ['Base all trades', '306', '0.813', '63.1%', '14.7%', '39.5%', '0.944'],
  ['Base drawdown trades', '43', '0.870', '74.4%', '20.9%', '60.5%', '0.904'],
  ['Base worst 12 trades', '12', '0.888', '91.7%', '16.7%', '33.3%', '0.921'],
  ['Quality all trades', '107', '0.537', '0.0%', '0.0%', '86.0%', '0.777'],
];

const drawdownWorstTrades = [
  ['sz.002738', '2024-01-26', '-54.7K', '-21.26%', '10', '0.849', '0.333'],
  ['sh.601127', '2023-08-10', '-42.7K', '-11.50%', '8', '0.940', '0.944'],
  ['sh.603348', '2023-08-02', '-41.9K', '-10.95%', '5', '0.943', '0.222'],
  ['sh.601595', '2024-01-10', '-37.8K', '-13.79%', '6', '0.959', '0.392'],
  ['sh.603918', '2023-09-07', '-36.4K', '-10.31%', '7', '0.956', '0.136'],
];

const medium128DiagnosticRows = [
  ['medium ok', '243', '+471.6K', '+0.81%', '50.2%', '26 / -302.8K'],
  ['medium weak', '63', '+446.3K', '+2.80%', '52.4%', '17 / -129.1K'],
  ['quality high', '129', '+142.6K', '+0.58%', '46.5%', '15 / -143.1K'],
  ['quality low', '30', '+105.3K', '+1.58%', '46.7%', '5 / -82.9K'],
];

const riskVariantRows = [
  ['Base 5td', '+183.57%', '45.35%', '50.65%', '+250.10%', 'reference'],
  ['ATR penalty 1.0', '+172.23%', '45.14%', '51.63%', '+239.83%', 'tiny DD help'],
  ['ATR penalty 2.0', '+218.89%', '45.55%', '51.96%', '+290.26%', 'best return'],
  ['medium128 penalty', '+151.50%', '46.99%', '48.04%', '+210.42%', 'reject'],
  ['stale quality penalty', '+170.09%', '45.36%', '48.69%', '+228.31%', 'no help'],
  ['risk mix', '+170.70%', '46.46%', '51.31%', '+237.35%', 'no help'],
  ['weak gate', '+158.02%', '34.14%', '52.73%', '+208.52%', 'DD cut, alpha cut'],
  ['weak top1', '+202.64%', '50.54%', '51.70%', '+259.57%', 'reject'],
  ['weak tier', '+226.38%', '51.30%', '51.88%', '+285.93%', 'reject'],
  ['weak score penalty', '+157.07%', '41.70%', '51.63%', '+216.33%', 'mild DD help'],
];

const weakWindowProfileRows = [
  ['AMV 5d slope', '+3.75%', '+0.24%', 'weaker'],
  ['AMV dd from high', '-2.30%', '-4.29%', 'deeper'],
  ['Candidate count', '15.8', '6.8', 'thin pool'],
  ['Top3 avg score', '8.46', '6.03', 'low quality'],
  ['Top3 avg ATR rank', '0.781', '0.872', 'high risk'],
  ['Top3 stale share', '39.4%', '62.8%', 'older LU'],
  ['Top3 reclaim share', '43.3%', '25.6%', 'less confirmation'],
];

const weakWindowRuleRows = [
  ['drawdown-like pool', '35', '-132.1K', '+132.1K', '21', '5 / 5'],
  ['AMV flat + thin pool', '37', '-91.7K', '+91.7K', '21', '6 / 7'],
  ['LU count low + bad pool', '32', '-162.4K', '+162.4K', '21', '4 / 5'],
  ['stale + no reclaim', '68', '-66.9K', '+66.9K', '25', '10 / 8'],
];

const weakControlRows = [
  ['Base 5td', 'Top3 every signal day', '2,282', '306', '+183.57%', '45.35%'],
  ['Weak gate', 'Skip weak_score >= 3.0 days', '1,550', '256', '+158.02%', '34.14%'],
  ['Weak top1', 'Weak days buy rank1 only', '1,816', '294', '+202.64%', '50.54%'],
  ['Weak tier', 'Weak days Top1 / semi-weak Top2', '1,729', '293', '+226.38%', '51.30%'],
  ['Weak score penalty', 'Weak days penalize high ATR / stale / no reclaim', '2,282', '306', '+157.07%', '41.70%'],
];

export default function AmvLimitEcologyDiagnostic() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>AMV Limit-Up Ecology Diagnostic</H1>
        <Text tone="secondary">
          第三阶段第一版：使用当前可用的日线 raw OHLCV 构造涨停生态近似特征，并按 <Code>signal_date + code</Code>
          拼到 P3/PB3 已成交交易。结果是 trade-level diagnostic，不是 Rust 回测。
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="+649.7K" label="P3 PnL with LU in 20d" tone="success" />
        <Stat value="24" label="P3 LU-event trades" tone="warning" />
        <Stat value="+644.9K" label="P3 re-board PnL" tone="success" />
        <Stat value="raw OHLC" label="Limit logic basis" tone="info" />
      </Grid>

      <Callout tone="success" title="主要发现">
        P3 已成交交易中，近 20 日有涨停的样本只有 <Code>24</Code> 笔，却贡献 <Code>+649.7K</Code>；
        其中 <Code>re-board after pullback</Code> 的 <Code>16</Code> 笔贡献 <Code>+644.9K</Code>。
        这不适合直接当 P3 hard gate，因为无涨停样本本身仍有 <Code>+358.8K</Code>，但非常适合作为独立 event sleeve 种子。
      </Callout>

      <Callout tone="warning" title="Rust 首轮反差">
        独立 event sleeve 接 <Code>6td static strict top3</Code> 后，最强 trade-level 的 reboard/reclaim 没有兑现：
        T+1 open 涨停阻塞 <Code>144</Code> 条信号，净收益 <Code>-21.54%</Code>。
        当前最好是更温和的 first-board pullback，净收益 <Code>+85.32%</Code>，但 MaxDD 仍有 <Code>56.50%</Code>。
      </Callout>

      <Grid columns="1.15fr 0.85fr" gap={18}>
        <Stack gap={10}>
          <H2>Rust Static Strict Results</H2>
          <Table
            headers={['Sleeve', 'Total', 'MaxDD', 'Trades', 'LU blocked', '2026', 'Decision']}
            rows={rustRows}
            rowTone={['danger', 'warning', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'left']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_limit_ecology_event_sleeve_rust_summary.json</Code>; current Rust still executes on adjusted OHLC.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>First-Board Yearly</H2>
          <Table
            headers={['Year', 'Return']}
            rows={firstBoardYearly}
            rowTone={['success', 'success', 'danger', 'info', 'danger', 'success']}
            columnAlign={['left', 'right']}
          />
        </Stack>
      </Grid>

      <Callout tone="warning" title="Focused Scan 结论">
        持仓期扫描后，原始 first-board pullback 的 <Code>5td</Code> 最强，total <Code>+183.57%</Code>，
        且 2021-2026 年年为正，但 MaxDD 仍有 <Code>45.35%</Code>。过滤后最防守的是
        <Code>quality 6td</Code>，total <Code>+53.36%</Code>，MaxDD <Code>21.71%</Code>，但收益偏低且 2023 仍亏。
      </Callout>

      <Callout tone="danger" title="Drawdown Attribution 结论">
        <Code>base 5td</Code> 的大回撤不是涨停阻塞，也不是单笔黑天鹅，而是 2023-08 到 2024-02
        连续弱段中反复买入高 ATR 的“首板后回踩”。最大回撤区间 43 笔交易累计 <Code>-431.9K</Code>，
        胜率只有 <Code>25.6%</Code>；最差 12 笔里 <Code>91.7%</Code> 的 ATR rank 高于 0.8。
      </Callout>

      <Grid columns="1.15fr 0.85fr" gap={18}>
        <Stack gap={10}>
          <H2>Hold / Risk Focused Scan</H2>
          <Table
            headers={['Variant', 'Total', 'MaxDD', 'Trades', 'Gross', 'Decision']}
            rows={holdRiskRows}
            rowTone={['danger', 'warning', 'warning', 'info', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_limit_first_board_pullback_hold_risk_scan.json</Code>; 3td/5td/6td static strict scan.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Base 5td vs Quality 6td</H2>
          <Table
            headers={['Year', 'Base 5td', 'Quality 6td']}
            rows={holdRiskYearly}
            rowTone={['success', 'success', 'warning', 'success', 'success', 'success']}
            columnAlign={['left', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Grid columns="1.1fr 0.9fr" gap={18}>
        <Stack gap={10}>
          <H2>Drawdown Path Attribution</H2>
          <Table
            headers={['Variant', 'Peak', 'Trough', 'Recovered', 'MaxDD', 'DD trades', 'DD PnL', 'DD win']}
            rows={drawdownRows}
            rowTone={['danger', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_limit_first_board_pullback_drawdown_attribution.json</Code>; DD PnL is realized
            trade PnL during peak-to-trough exits.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Base 5td Drawdown Months</H2>
          <Table
            headers={['Month', 'Trades', 'PnL', 'Avg ret', 'Win']}
            rows={drawdownWorstMonths}
            rowTone={['danger', 'danger', 'danger', 'danger', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Risk Profile</H2>
          <Table
            headers={['Sample', 'Trades', 'Avg ATR rank', 'ATR > 0.8', 'Panic > 0.8', 'LU age >= 7d', 'Amount 5/20']}
            rows={riskProfileRows}
            rowTone={['warning', 'danger', 'danger', 'info']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>Base Drawdown Worst Trades</H2>
          <Table
            headers={['Code', 'Entry', 'PnL', 'Ret', 'LU age', 'ATR rank', 'Panic rank']}
            rows={drawdownWorstTrades}
            rowTone={['danger', 'danger', 'danger', 'danger', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Callout tone="warning" title="medium128 对首板回踩的判断">
        <Code>medium128</Code> 对 P3 主线有效，但对首板回踩 sleeve 不能直接照搬。全样本里
        <Code>medium weak</Code> 交易反而贡献 <Code>+446.3K</Code>，单独惩罚后 Rust total 从
        <Code>+183.57%</Code> 降到 <Code>+151.50%</Code>，MaxDD 还升到 <Code>46.99%</Code>。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>medium128 Trade Diagnostic</H2>
          <Table
            headers={['Bucket', 'Trades', 'PnL', 'Avg ret', 'Win', 'DD sample']}
            rows={medium128DiagnosticRows}
            rowTone={['info', 'success', 'info', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_limit_first_board_medium128_diagnostic.json</Code>; DD sample is
            2023-08-03 to 2024-02-05 by trade exit date.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>5td Risk Rerank Rust Scan</H2>
          <Table
            headers={['Variant', 'Total', 'MaxDD', 'Win', 'Gross', 'Decision']}
            rows={riskVariantRows}
            rowTone={['warning', 'warning', 'success', 'danger', 'warning', 'danger', 'success', 'danger', 'danger', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'left']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_limit_first_board_risk_variant_scan.json</Code>; all variants keep
            static strict Top3 and 5td except the defensive quality endpoint shown above.
          </Text>
        </Stack>
      </Grid>

      <Callout tone="info" title="弱窗口诊断">
        最大回撤期在开仓前确实有可观测特征，但不是普通市场情绪单项能解释。更贴近本 sleeve 的信号是
        <Code>候选池变薄 + Top3 分数下降 + Top3 高 ATR / 老涨停 / 缺少 reclaim</Code>。
        Rust 对照显示 <Code>top1/downshift</Code> 反而放大回撤，说明弱窗口里保留 rank1 并不稳定；
        真正能降回撤的是 <Code>weakgate</Code>，但收益也被砍掉。
      </Callout>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Weak Window Profile</H2>
          <Table
            headers={['Feature', 'Non-DD', 'DD window', 'Direction']}
            rows={weakWindowProfileRows}
            rowTone={['warning', 'warning', 'danger', 'danger', 'danger', 'danger', 'danger']}
            columnAlign={['left', 'right', 'right', 'left']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_limit_first_board_weak_window_diagnostic.json</Code>; DD window is
            2023-08-03 to 2024-02-05 by trade exit date.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Diagnostic Weak Rules</H2>
          <Table
            headers={['Rule', 'Flagged', 'Flagged PnL', 'Delta', 'DD hits', 'Big W/L']}
            rows={weakWindowRuleRows}
            rowTone={['success', 'warning', 'success', 'warning']}
            columnAlign={['left', 'right', 'right', 'right', 'right', 'right']}
          />
          <Text size="small" tone="tertiary">
            These are trade-level diagnostics only. Positive delta means the flagged trades lost money in historical
            trade PnL; Rust gate/downshift is still required.
          </Text>
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Weak Window Control Rust Scan</H2>
        <Table
          headers={['Variant', 'Mechanism', 'Signal rows', 'Trades', 'Total', 'MaxDD']}
          rows={weakControlRows}
          rowTone={['warning', 'success', 'danger', 'danger', 'warning']}
          columnAlign={['left', 'left', 'right', 'right', 'right', 'right']}
        />
        <Text size="small" tone="tertiary">
          Source: <Code>reports/amv_limit_first_board_risk_variant_scan.json</Code>; weak_score combines market breadth,
          AMV phase and same-day candidate pool health, all observed at signal_date.
        </Text>
      </Stack>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 Limit Ecology Events</H2>
          <Table
            headers={['Event', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
            rows={p3Events}
            rowTone={['success', 'success', 'success', 'success', 'success', 'success']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
          <Text size="small" tone="tertiary">
            Source: <Code>reports/amv_limit_ecology_diagnostic.json</Code>; P3 static strict trades, raw daily approximation.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>P3 Days Since Prior Limit-Up</H2>
          <BarChart
            categories={['1-3d', '4-10d', '11-20d', '>20d', 'None']}
            valueSuffix="K"
            height={260}
            series={[
              { name: 'Total PnL', data: [598.1, 47.7, 62.2, 284.8, 15.7], tone: 'info' },
            ]}
          />
          <Text size="small" tone="tertiary">
            P3 强收益主要集中在涨停后 1-3 个交易日，说明它已经偶然捕捉到一部分强势股事件。
          </Text>
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>P3 Prior Streak Context</H2>
          <Table
            headers={['Prior LU streak', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
            rows={p3Streak}
            rowTone={['success', 'danger', 'success', 'info']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
        </Stack>

        <Stack gap={10}>
          <H2>PB3 Contrast</H2>
          <Table
            headers={['Event', 'Trades', 'PnL', 'Avg ret', 'Win rate']}
            rows={pb3Events}
            rowTone={['success', 'info', 'success', 'warning', 'danger']}
            columnAlign={['left', 'right', 'right', 'right', 'right']}
          />
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Data Support Boundary</H2>
        <Table
          headers={['Factor', 'Status', 'Current basis / gap']}
          rows={supportedRows}
          rowTone={['success', 'success', 'success', 'warning', 'warning', 'warning', 'danger', 'danger']}
          columnAlign={['left', 'left', 'left']}
        />
      </Stack>

      <Callout tone="warning" title="下一步解释">
        第三阶段已有 alpha 线索，但还没有 allocation-ready sleeve。当前最清楚的选择不是 rolling/refill，
        而是继续打磨弱窗口定义：<Code>weakgate</Code> 明显降 MaxDD 但牺牲收益，<Code>weakscorepen</Code>
        只温和降回撤；<Code>weaktop1/weaktier</Code> 暂时否决。raw execution 修正后也需要复核。
      </Callout>
    </Stack>
  );
}
