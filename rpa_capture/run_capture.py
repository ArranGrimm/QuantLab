"""指南针活跃市值 - 截图 RPA (Windows VM 端)

第一阶段 PoC: 只做截图 + 文件落地, 不做 OCR / 不入库 / 不解析。

运行前你需要手动:
  1. 打开指南针客户端 (指南针全赢系统)
  2. 切到 [0AMV 活跃市值] 指标
  3. 顶部周期切到 "日"
  4. 把指南针窗口最大化
  5. 把图表向左拖/滚动到希望的起始日期 (例如 2019-01-04 附近)
  6. 鼠标在该起始日附近点一下, 让顶部 readout 出现日期数字

然后运行:
  python run_capture.py --days 10 --output D:/rpa_output

脚本会:
  - 找到指南针窗口并把焦点切过去
  - 循环 N 次: 全屏截图 → 写 PNG → 按 → → sleep
  - 输出 manifest.jsonl 记录每张图的元数据 (seq + 时间戳)

时间方向: seq=0 是最早 (起始点), seq=N 是最新 (向今天前进). 这样入库后
天然按时间顺序排列, 不需要 reverse.

第二阶段在 Mac 上读这些图做 OCR + 入库, 跟本脚本完全解耦.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mss
from pywinauto import findwindows
from pywinauto.keyboard import send_keys

WINDOW_TITLE_KEYWORDS = ["活跃市值", "指南针"]

# Win32 常量 / API 句柄
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_SW_RESTORE = 9

REGION_FILE_DEFAULT = Path(__file__).parent / "region.json"


def enable_dpi_awareness() -> None:
    """让 mss 截图坐标跟 region 配置坐标使用同一像素系统

    高 DPI 屏幕 (笔记本 200% 缩放等) 不开这个会导致坐标偏移.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def load_region(
    cli_region: str | None,
    region_file: Path,
) -> tuple[int, int, int, int] | None:
    """解析 region 配置, 优先级: CLI > region.json > None (=全屏)

    CLI 格式: "left,top,width,height" 例如 "15,50,200,200"
    """
    if cli_region:
        parts = [int(x.strip()) for x in cli_region.split(",")]
        if len(parts) != 4:
            raise ValueError(f"--region 必须是 'left,top,width,height' 格式, 收到: {cli_region}")
        return tuple(parts)  # type: ignore[return-value]
    if region_file.exists():
        cfg = json.loads(region_file.read_text(encoding="utf-8"))
        return (cfg["left"], cfg["top"], cfg["width"], cfg["height"])
    return None


def find_compass_window():
    """按窗口标题关键词找指南针主窗口, 找不到就报错并列出候选"""
    candidates = findwindows.find_elements(title_re=".*活跃市值.*")
    if not candidates:
        candidates = findwindows.find_elements(title_re=".*指南针.*")
    if not candidates:
        all_titles = [
            w.name for w in findwindows.find_elements()
            if w.name and len(w.name) > 0
        ]
        print("未找到指南针窗口. 当前桌面所有窗口标题前 30 个:", file=sys.stderr)
        for t in all_titles[:30]:
            print(f"  {t}", file=sys.stderr)
        raise RuntimeError("找不到指南针窗口, 请先手动打开它并切到活跃市值指标")
    chosen = candidates[0]
    print(f"找到窗口: handle={chosen.handle}  title='{chosen.name}'")
    return chosen


def bring_to_front(handle: int) -> None:
    """纯 Win32 把窗口拉到前台, 不点击、不调用 UIA, 不会触发图表 cursor 漂移

    pywinauto 的 Application().connect() + set_focus() 在某些情况下会产生
    "合成点击" 把图表 cursor 弹到窗口中心位置, 这是之前光标漂移的根本原因.
    """
    if not _user32.IsWindow(handle):
        raise RuntimeError(f"窗口 handle {handle} 已失效")
    if _user32.IsIconic(handle):
        _user32.ShowWindow(handle, _SW_RESTORE)
    if _user32.GetForegroundWindow() != handle:
        _user32.SetForegroundWindow(handle)


def park_mouse(x: int = 2, y: int = 2) -> None:
    """把鼠标停到屏幕角落 (默认左上角), 避免在 K 线主图区域 hover 干扰 cursor

    指南针窗口最大化后, (2, 2) 落在窗口最左上角 (顶部菜单区), 不会触碰主图.
    """
    _user32.SetCursorPos(x, y)


def take_screenshot(output_path: Path, region: tuple[int, int, int, int] | None = None) -> None:
    """全屏截图 (region=None) 或区域截图

    region: (left, top, width, height) 像素坐标; PoC 默认 None 即全屏
    """
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1]  # 主屏 (monitors[0] 是所有屏总和)
        else:
            left, top, width, height = region
            monitor = {"left": left, "top": top, "width": width, "height": height}
        img = sct.grab(monitor)
        mss.tools.to_png(img.rgb, img.size, output=str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="指南针活跃市值截图 RPA")
    parser.add_argument("--days", type=int, default=10, help="抓取多少个交易日 (按 → 次数 + 1)")
    parser.add_argument("--output", type=str, default="./shots", help="输出目录")
    parser.add_argument("--sleep-ms", type=int, default=200, help="按键后等待毫秒数")
    parser.add_argument("--start-seq", type=int, default=0, help="起始 seq (续跑时用)")
    parser.add_argument(
        "--no-arrow", action="store_true",
        help="只截当前一帧, 不按方向键; 用于先验证截图能拿到什么内容",
    )
    parser.add_argument(
        "--no-focus", action="store_true",
        help="完全跳过窗口激活, 假设指南针已在前台. 最不易触发 cursor 漂移, "
             "但需要你在 Alt+Tab 后再按 Enter 启动",
    )
    parser.add_argument(
        "--precount", type=int, default=3,
        help="开始截图前的倒计时秒数, 期间脚本会把鼠标移到角落",
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="只截区域 'left,top,width,height' (像素). 不传则读 region.json, "
             "都没有就全屏",
    )
    parser.add_argument(
        "--region-file", type=str, default=str(REGION_FILE_DEFAULT),
        help="region 配置文件路径, 默认 ./region.json",
    )
    args = parser.parse_args()

    enable_dpi_awareness()
    region = load_region(args.region, Path(args.region_file))
    if region is None:
        print("截图模式: 全屏 (未配置 region)")
    else:
        print(f"截图模式: 区域 left={region[0]} top={region[1]} "
              f"width={region[2]} height={region[3]}")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    print(f"输出目录: {output_dir}")

    if args.no_focus:
        print("[--no-focus] 跳过窗口激活, 假设指南针已在前台并已选好起始 K 线")
    else:
        win_elem = find_compass_window()
        bring_to_front(win_elem.handle)
        print(f"已用 Win32 SetForegroundWindow 拉前台 (无点击): handle={win_elem.handle}")

    park_mouse()
    print(f"鼠标已停到屏幕左上角 (2, 2), 避免触发图表 hover")

    for sec in range(args.precount, 0, -1):
        print(f"  {sec}s 后开始截图...")
        time.sleep(1)

    sleep_sec = args.sleep_ms / 1000.0

    # 续写模式: 续跑时不覆盖之前的 manifest
    manifest_mode = "a" if args.start_seq > 0 else "w"
    iterations = 1 if args.no_arrow else args.days

    with open(manifest_path, manifest_mode, encoding="utf-8") as f_manifest:
        for i in range(iterations):
            seq = args.start_seq + i
            png_path = output_dir / f"seq_{seq:05d}.png"
            captured_at = datetime.now().isoformat(timespec="milliseconds")

            try:
                take_screenshot(png_path, region=region)
            except Exception as exc:
                print(f"[seq={seq}] 截图失败: {exc}", file=sys.stderr)
                break

            record = {
                "seq": seq,
                "filename": png_path.name,
                "captured_at": captured_at,
                "step": i,
                "arrow_pressed": not args.no_arrow,
            }
            f_manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_manifest.flush()

            if (i + 1) % 10 == 0 or i == 0:
                print(f"[seq={seq}] 已截图 {png_path.name}")

            if not args.no_arrow and i < iterations - 1:
                send_keys("{RIGHT}")
                time.sleep(sleep_sec)

    print(f"\n完成. 共截图 {iterations} 张, manifest: {manifest_path}")
    print("\n下一步:")
    print("  1. 用图片查看器打开 seq_00000.png 检查质量")
    print("  2. 重点确认顶部左侧 readout 区域 (日期 + 开高低收) 文字是否清晰")
    print("  3. 比较 seq_00000 和 seq_00001, 确认日期是否真的前进了 1 天")
    print("  4. 当前在 Windows 直跑时, 图片留本地即可; 切到 Mac 主力时改用 PD 共享文件夹")
    return 0


if __name__ == "__main__":
    sys.exit(main())
