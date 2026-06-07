# rpa_capture: 指南针活跃市值截图 RPA (Windows 端)

本目录代码运行在 **Windows 环境**, 唯一职责是 **截图 + 文件落地**, 不做任何 OCR / 解析 / 入库。

## 为什么需要 RPA

- `0AMV (活跃市值)` 是指南针专利指标, 客户端没有导出接口, 也没有公开 API。
- 项目里的 AMV bull pool / regime 研究依赖完整历史序列, 不能继续靠手工圈定的 `LOOSE_PERIODS`。
- 当前唯一可行路径: 在 **指南针全赢系统** 里打开 readout, 用方向键逐日前进, 截图后交给 `rpa_parse/` 做 OCR 和入库。
- Capture PoC 已通过 (`2026-04-21`); 当前重点是 **历史回填** 和 **收盘后增量截图**。

完整管道: `rpa_capture/` (Windows 截图) → 共享文件夹 → `rpa_parse/` (Mac OCR + DuckDB)。Parse 侧说明见 [`../rpa_parse/README.md`](../rpa_parse/README.md)。

## 部署形态

- **当前阶段**: 直接在物理 Windows 机器上跑通 PoC, 图片留本地即可
- **最终形态**: 主力机切换到 Mac 后, 本目录运行在 Mac 上的 PD Windows 虚拟机内, 图片通过共享文件夹同步给 Mac 端的解析模块

代码本身完全可移植, 切换部署形态时无需改动。

## 设计原则

- **依赖最少**: 只 `pywinauto` + `mss` 两个包, 不挑环境
- **职责单一**: 只截图, 不解析; 不被 OCR 模型升级影响
- **跟 Mac 端解耦**: 截图文件通过 PD 共享文件夹同步到 Mac, Mac 端独立做 OCR + 入库
- **可重复**: 同一目录可断点续抓 (`--start-seq`)

## 阶段拆分

```
[Windows 端]                           [Mac 端]
rpa_capture/                           rpa_parse/
   ↓                                       ↑
shots/                  本地或共享       shots/
  ├── seq_00000.png    ─────────────→     ├── seq_00000.png
  ├── seq_00001.png                       ├── seq_00001.png
  └── manifest.jsonl                      └── manifest.jsonl
                                              ↓
                                    active_market_value.duckdb
```

时间方向: `seq=0` 是 **最早起始日** (例如 2019-01-04), `seq=N` 是 **最新一天**。
按 → 方向键, 时间天然顺序前进, 入库后无需 reverse。

## 使用步骤

### 1. 安装依赖

```bash
cd D:\WorkSpace\Tinkering\QuantLab\rpa_capture
python -m pip install -r requirements.txt
```

### 2. 手动准备指南针客户端

1. 启动 **指南针全赢系统**
2. 打开 `[0AMV 活跃市值]` 指标
3. 顶部周期切到 **日**
4. 把指南针窗口 **最大化**
5. 把图表向左拖动/滚轮滚到 **希望的起始日期** (例如 2019-01-04 附近)
6. 鼠标在该 **起始日附近的 K 线** 上 **点一下**, 让顶部左侧 readout 出现该日的日期 + 开高低收数字
7. **确认这个 readout 的日期 = 你想开始的日期**, 否则按方向键微调

### 3. 跑 PoC: 先抓 1 张验证

```bash
python run_capture.py --no-arrow --output ./shots
```

只截当前一帧, 不按方向键. 跑完打开 `shots/seq_00000.png` 检查:

- [ ] 顶部左侧 readout 区域可见, 文字清晰
- [ ] 日期、开/高/低/收/幅 数字都能用肉眼看清
- [ ] 截图分辨率正常 (不被压缩)
- [ ] readout 显示的日期 = 你期望的起始日期

### 4. 跑 PoC: 抓 10 个交易日

```bash
python run_capture.py --days 10 --output ./shots
```

跑完检查:

- [ ] `seq_00000.png` 的日期 = 你设的起始日 (例如 2019-01-04)
- [ ] `seq_00009.png` 的日期 = 起始日 + 9 个交易日
- [ ] 中间没有跳过、没有重复的日期
- [ ] `manifest.jsonl` 里 10 行记录都正常

### 4.5. 标定 readout 区域 (一次性, 大幅减小图片体积)

默认全屏截图每张 2~3 MB, 1700 张 ≈ 4 GB. 标定后只截 readout 那一小块,
每张降到 ~30 KB, 1700 张 ≈ 50 MB, 同时 OCR 准确率更高.

```bash
python calibrate_region.py
```

操作:
1. 弹出全屏半透明覆盖 (指南针仍可见, 颜色变暗)
2. 用鼠标拖一个矩形框, 框住顶部左侧 readout 区域
   (日期 + 开/高/低/收/幅/量/额/振/涨, 留点边距没关系)
3. 松开鼠标即保存到 `region.json`

之后 `run_capture.py` 会自动读取这个区域.

也可以不用交互工具, 直接 CLI 传:
```bash
python run_capture.py --region 15,50,200,200 --no-arrow
```

> 标定时指南针窗口的位置/大小必须跟跑 RPA 时一致 (一般都最大化).

### 5. 历史回填 (从 2019 到今天 ≈ 1700 天, 大约 6~10 分钟)

```bash
python run_capture.py --days 1700 --output ./shots --sleep-ms 200
```

> 跑大批量前先把 PoC (步骤 4) 跑通, 否则会浪费时间.

> 跑完后用图片查看器抽查 `seq_00000` / `seq_00500` / `seq_01000` / `seq_01699`,
> 确认日期连续, 然后才把图片喂给后续解析阶段.

## CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--days` | 10 | 抓多少个交易日 (= 1 次截图 + N-1 次按 →) |
| `--output` | `./shots` | 输出目录 |
| `--sleep-ms` | 200 | 每次按键后等待毫秒 (太短可能截到旧帧) |
| `--start-seq` | 0 | 起始 seq, 续跑用 |
| `--overwrite` | False | 允许覆盖已存在截图; 默认禁止覆盖 |
| `--no-arrow` | False | 只截当前一帧, 用于第一次验证 |
| `--no-focus` | False | 完全跳过窗口激活, 假设指南针已在前台 (最稳妥, 不会漂) |
| `--precount` | 3 | 开始截图前的倒计时秒数 |
| `--region` | None | 只截区域 `left,top,width,height`, 不传则读 region.json |
| `--region-file` | `./region.json` | region 配置文件路径 |

## 关于 cursor 漂移问题

**症状**: 你手动选了 2019-01-02, 但脚本截到的图却是另一个日期 (例如 2018-12-12)。

**根因**: pywinauto 的 `Application().connect() + set_focus()` 在某些情况下会
合成一次"激活点击"在窗口中心, 这一击落在 K 线主图上 → 图表 cursor 跳到 click 位置。

**修复**: v2 已改用纯 Win32 `SetForegroundWindow` + 把鼠标移到屏幕左上角 `(2, 2)`,
不会产生任何点击。

**如果改完仍然漂**, 试试 `--no-focus`:

```bash
python run_capture.py --no-arrow --no-focus --output ./shots
```

`--no-focus` 完全不动指南针窗口, 但需要你保证运行时指南针已经在前台。
推荐操作顺序:

1. 在指南针里手动选好起始 K 线
2. **保持指南针在前台**, 用 `Win+R` 调出运行框, 输入 `cmd /c "cd /d D:\WorkSpace\Tinkering\QuantLab\rpa_capture && python run_capture.py --no-arrow --no-focus"` 然后回车
3. cmd 窗口会一闪而过, 但脚本会在指南针前台情况下截图

或者让另一个人按你的电脑回车键也行. 重点是: **脚本启动时不打断指南针的前台焦点**.

## 输出格式

```
shots/
├── seq_00000.png       # 第 0 张截图 (起始日, 例如 2019-01-04)
├── seq_00001.png       # 第 1 张截图 (按了 1 次 →, 起始日的下一个交易日)
├── ...
├── seq_01699.png       # 接近今天
└── manifest.jsonl      # 元数据
```

`manifest.jsonl` 每行 1 个 JSON:

```json
{"seq": 0, "filename": "seq_00000.png", "captured_at": "2026-04-21T10:00:00.123", "step": 0, "arrow_pressed": true}
```

注意: `manifest.jsonl` **不记录交易日期**, 因为本阶段不做 OCR. 真实日期由 Mac 端 OCR 阶段从图片里识别出来后再写入数据库.

## 日更 / 增量截图

如果 Mac 端当前最后一张是 `seq_01771.png`, 下一次在 Windows / PD 里从新日期开始抓时:

```bash
python run_capture.py --days 5 --start-seq 1772 --output ./shots
```

脚本默认会检查目标文件是否已存在; 如果 `seq_01772.png` 已存在会直接中止, 防止覆盖历史截图。
只有确认要重抓覆盖时才加:

```bash
python run_capture.py --days 5 --start-seq 1772 --output ./shots --overwrite
```

## 后续工作

- [ ] 历史回填: `--days 1700` 全量截图 (建议在 PD VM 或固定 Windows 环境一次性跑完)
- [ ] 日更: Windows 计划任务 + `--start-seq` 增量截图
- [ ] 与 `rpa_parse/` 联调: 新增截图后跑 `--incremental` 解析和 DuckDB upsert

长期可复用同一套框架抓取其他指南针 readout 指标; 每次只需重新标定 `region.json`。

## 已知限制 / TODO

- [ ] 目前没有"窗口找不到自动启动指南针"的容错; 需手动开
- [ ] 未标定 `region.json` 时全屏截图较大 (~2~3 MB/张); 标定 readout 区域后单张约 tens of KB
- [ ] 日期连续性校验在 `rpa_parse/` 阶段完成, 不在 capture 阶段报警
- [ ] 没有 Windows 计划任务的日更脚本
- [ ] 没有"光标走到右边缘后图表是否会自动向左滚动"的兜底; 跑大批量前先用 `--days 50` 验证

## 风险备忘

| 风险 | 缓解 |
|---|---|
| 指南针 UI 改版 | 重跑 `calibrate_region.py` |
| cursor 漂移 | 优先 `--no-focus`; 见上文 troubleshooting |
| 屏幕分辨率 / DPI 变化 | 固定最大化窗口; 重标定 region |
