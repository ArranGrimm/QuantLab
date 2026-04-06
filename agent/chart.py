"""
K线图生成模块 — 日K + 成交量 + WL/YL 知行线 + B1信号标记
导出高清 PNG 供 AI 多模态评审使用。

图表结构:
  Row 1 (75%): 蜡烛图 + WL 线(蓝) + YL 线(橙) + B1 三角标记
  Row 2 (25%): 成交量柱 (红涨绿跌)
"""
from pathlib import Path
from datetime import datetime

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_UP = "#F56C6C"
COLOR_DOWN = "#67C23A"
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E4E7ED"
COLOR_WL = "#409EFF"
COLOR_YL = "#E6A23C"
COLOR_SIGNAL = "#E74C3C"

DEFAULT_BARS = 90
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 800
DEFAULT_SCALE = 2


def make_chart(
    df: pl.DataFrame,
    code: str,
    target_date: str,
    bars: int = DEFAULT_BARS,
    title: str | None = None,
) -> go.Figure:
    """
    生成日K线图 (蜡烛图 + 成交量 + WL/YL + B1标记)。

    WL/YL 已在 calc_b1_factors_wmacd 中基于完整历史计算,
    此处只做截断显示, 保证均线准确性。

    Args:
        df: calc_b1_factors_wmacd 输出的 DataFrame (需含 WL/YL/b1_signal)
        code: 股票代码
        target_date: 信号日期 (YYYY-MM-DD), 显示范围截止到此日
        bars: 显示K线根数
        title: 自定义图表标题
    """
    if isinstance(target_date, str):
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        target_dt = target_date

    stock_df = (
        df.filter(
            (pl.col("code") == code)
            & (pl.col("date") <= target_dt)
        )
        .sort("date")
        .tail(bars)
    )

    if stock_df.is_empty():
        raise ValueError(f"No data for {code} up to {target_date}")

    # 日期转字符串 — 最简单的方式去除周末/节假日空白
    dates = stock_df["date"].cast(pl.String).to_list()
    opens = stock_df["open_adj"].to_list()
    highs = stock_df["high_adj"].to_list()
    lows = stock_df["low_adj"].to_list()
    closes = stock_df["close_adj"].to_list()
    volumes = stock_df["volume"].to_list()

    wl = stock_df["WL"].to_list() if "WL" in stock_df.columns else None
    yl = stock_df["YL"].to_list() if "YL" in stock_df.columns else None

    has_b1 = "b1_signal" in stock_df.columns
    b1_dates, b1_lows = [], []
    if has_b1:
        b1_df = stock_df.filter(pl.col("b1_signal"))
        b1_dates = b1_df["date"].cast(pl.String).to_list()
        b1_lows = b1_df["low_adj"].to_list()

    vol_colors = [
        COLOR_UP if c >= o else COLOR_DOWN
        for c, o in zip(closes, opens)
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens, high=highs, low=lows, close=closes,
            increasing_line_color=COLOR_UP,
            increasing_fillcolor=COLOR_UP,
            decreasing_line_color=COLOR_DOWN,
            decreasing_fillcolor=COLOR_DOWN,
            name="K线",
            showlegend=False,
            line=dict(width=1),
        ),
        row=1, col=1,
    )

    if wl:
        fig.add_trace(
            go.Scatter(
                x=dates, y=wl,
                mode="lines", name="WL(白)",
                line=dict(color=COLOR_WL, width=1.5),
            ),
            row=1, col=1,
        )
    if yl:
        fig.add_trace(
            go.Scatter(
                x=dates, y=yl,
                mode="lines", name="YL(黄)",
                line=dict(color=COLOR_YL, width=2),
            ),
            row=1, col=1,
        )

    if b1_dates:
        markers_y = [low * 0.98 for low in b1_lows]
        fig.add_trace(
            go.Scatter(
                x=b1_dates, y=markers_y,
                mode="markers", name="B1信号",
                marker=dict(symbol="triangle-up", size=12, color=COLOR_SIGNAL),
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Bar(
            x=dates, y=volumes,
            marker_color=vol_colors,
            name="成交量",
            showlegend=False,
        ),
        row=2, col=1,
    )

    chart_title = title or f"{code} 日线走势"
    fig.update_layout(
        title=dict(
            text=f"<b>{chart_title}</b>  (to {dates[-1]})",
            font=dict(size=16, color="#303133"),
        ),
        height=DEFAULT_HEIGHT,
        xaxis_rangeslider_visible=False,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=10),
        ),
        margin=dict(l=60, r=30, t=50, b=30),
    )

    for row_i in [1, 2]:
        fig.update_xaxes(
            type="category",
            showgrid=True,
            gridcolor=COLOR_GRID,
            row=row_i, col=1,
        )
    fig.update_xaxes(nticks=15, tickangle=-45, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    fig.update_yaxes(title="价格", showgrid=True, gridcolor=COLOR_GRID, row=1, col=1)
    fig.update_yaxes(title="", showgrid=False, row=2, col=1)

    return fig


def export_chart(
    df: pl.DataFrame,
    code: str,
    target_date: str,
    output_dir: str | Path,
    bars: int = DEFAULT_BARS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> Path:
    """
    生成并导出 K 线图为 PNG 文件。

    Returns:
        导出的文件路径
    """
    fig = make_chart(df, code, target_date, bars=bars)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_code = code.replace(".", "_")
    output_path = output_dir / f"{safe_code}.png"

    fig.write_image(
        str(output_path),
        format="png",
        width=width,
        height=height,
        scale=scale,
    )

    return output_path
