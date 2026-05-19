"""calibrate_region.py - 交互式标定 readout 区域

运行后弹出一个半透明全屏覆盖, 用鼠标拖动一个矩形框选指南针顶部
左侧的 readout 区域 (日期 + 开高低收 + ...), 松开鼠标自动保存到
region.json. 之后 run_capture.py 会自动读取这个区域, 只截这一小块.

依赖: 仅 stdlib (tkinter), 无需新依赖.

注意: 标定时指南针窗口的位置必须跟跑 RPA 时完全一致 (一般都最大化).
按 ESC 取消.
"""
from __future__ import annotations

import ctypes
import json
import sys
import tkinter as tk
from pathlib import Path


REGION_FILE = Path(__file__).parent / "region.json"


def enable_dpi_awareness() -> None:
    """让 tkinter 报告物理像素坐标, 跟 mss 截图坐标系一致

    高 DPI 屏幕上 (如笔记本 200%缩放), 不开这个会导致 tkinter 报的坐标
    跟实际屏幕像素不一致, mss 截图就会偏.
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


def main() -> int:
    enable_dpi_awareness()

    print("即将打开全屏标定窗口 (半透明覆盖, 指南针仍可见)")
    print("操作:")
    print("  1. 鼠标拖动一个矩形框选指南针的 readout 区域")
    print("     (顶部左侧那一小块: 日期 + 开/高/低/收/幅/量/额 ...)")
    print("  2. 松开鼠标即自动保存到 region.json")
    print("  3. ESC 取消, 不保存")
    print()
    print("提示: 框可以稍微留一点边距, 不必精确卡到像素")
    input("按回车继续...")

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.35)  # 35% 不透明度, 指南针仍可见
    root.attributes("-topmost", True)
    root.configure(bg="black")

    canvas = tk.Canvas(root, cursor="crosshair", bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    info = canvas.create_text(
        20, 20, anchor="nw", fill="yellow",
        font=("Segoe UI", 14, "bold"),
        text="拖动鼠标框选 readout 区域, ESC 取消",
    )
    coord_label = canvas.create_text(
        20, 50, anchor="nw", fill="lime",
        font=("Consolas", 12),
        text="",
    )

    state: dict = {"start": None, "rect_id": None, "result": None}

    def on_press(e: tk.Event) -> None:
        state["start"] = (e.x_root, e.y_root)
        if state["rect_id"] is not None:
            canvas.delete(state["rect_id"])
        state["rect_id"] = canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="red", width=2,
        )

    def on_drag(e: tk.Event) -> None:
        if state["start"] is None:
            return
        sx, sy = state["start"]
        canvas.coords(
            state["rect_id"],
            sx - root.winfo_rootx(),
            sy - root.winfo_rooty(),
            e.x_root - root.winfo_rootx(),
            e.y_root - root.winfo_rooty(),
        )
        left = min(sx, e.x_root)
        top = min(sy, e.y_root)
        w = abs(e.x_root - sx)
        h = abs(e.y_root - sy)
        canvas.itemconfig(
            coord_label,
            text=f"left={left}  top={top}  width={w}  height={h}",
        )

    def on_release(e: tk.Event) -> None:
        if state["start"] is None:
            return
        sx, sy = state["start"]
        ex, ey = e.x_root, e.y_root
        left = min(sx, ex)
        top = min(sy, ey)
        width = abs(ex - sx)
        height = abs(ey - sy)
        if width < 10 or height < 10:
            print("⚠ 框选区域太小 (<10px), 重新拖一次")
            state["start"] = None
            if state["rect_id"] is not None:
                canvas.delete(state["rect_id"])
                state["rect_id"] = None
            return
        state["result"] = {
            "left": int(left), "top": int(top),
            "width": int(width), "height": int(height),
        }
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda e: root.destroy())

    root.mainloop()

    if state["result"] is None:
        print("已取消, 未保存")
        return 1

    REGION_FILE.write_text(
        json.dumps(state["result"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n标定完成, 区域已保存:")
    print(json.dumps(state["result"], indent=2, ensure_ascii=False))
    print(f"\n配置文件: {REGION_FILE}")
    print("\n下一步: 直接跑 run_capture.py 即可, 它会自动读取 region.json")
    print("  python run_capture.py --no-arrow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
