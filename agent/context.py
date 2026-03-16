"""
结构化指标提取模块 — 从 B1 信号行生成人类可读的指标文本。
这段文本将与 K 线图一起发送给 LLM，为其提供精确数值补充。
"""


def build_context(row: dict) -> str:
    """
    从一行 calc_b1_factors_wmacd 的输出 (dict) 中提取关键指标，
    构建结构化文本块供 LLM 阅读。

    Args:
        row: DataFrame.iter_rows(named=True) 产出的单行 dict
    """
    code = row.get("code", "?")
    date = row.get("date", "?")
    close = row.get("close_adj", 0)

    j_val = row.get("J", None)
    k_val = row.get("K", None)
    d_val = row.get("D", None)

    trigger_type = _get_trigger_type(row)
    shape_pct = abs(close - row.get("open_adj", close)) / close * 100 if close else 0
    max_yang_vol = row.get("max_yang_vol_28", 1)
    vol_ratio = row.get("volume", 0) / max_yang_vol * 100 if max_yang_vol else 0

    bias_wl = row.get("Bias_C_WL", None)
    bias_yl = row.get("Bias_C_YL", None)
    bias_wl_yl = row.get("Bias_WL_YL", None)

    rw_hist = row.get("rw_hist", None)
    rw_dif = row.get("rw_dif", None)
    rm_hist = row.get("rm_hist", None)
    weekly_wl_yl = row.get("WEEKLY_WL_YL_OK", None)

    yangyin_p1 = _safe_ratio(row.get("vol_yang_p1"), row.get("vol_yin_p1"))
    yangyin_p2 = _safe_ratio(row.get("vol_yang_p2"), row.get("vol_yin_p2"))

    lines = [
        f"股票: {code} | 信号日期: {date} | 收盘价: {close:.2f}",
        "",
        "--- 核心 ---",
        f"J值: {_fmt(j_val)} | K值: {_fmt(k_val)} | D值: {_fmt(d_val)}",
        f"触发类型: {trigger_type}",
        f"形态幅度(实体): {shape_pct:.1f}% | 量比窒息: {vol_ratio:.0f}% (当日量/28日最大阳量)",
        "",
        "--- 均线 ---",
        f"WL偏离(C-WL): {_fmt(bias_wl)}% | YL偏离(C-YL): {_fmt(bias_yl)}%",
        f"WL-YL间距: {_fmt(bias_wl_yl)}%",
        "",
        "--- 多周期 ---",
        f"周线MACD: {_macd_status(rw_dif, rw_hist)}",
        f"月线MACD: {_macd_status_monthly(rm_hist)}",
        f"周线WL>YL: {_bool_cn(weekly_wl_yl)}",
        "",
        "--- 量价 ---",
        f"红绿比(21d): {yangyin_p1} | 红绿比(14d): {yangyin_p2}",
    ]

    return "\n".join(lines)


def _get_trigger_type(row: dict) -> str:
    plry = row.get("PLRY_CNT", False)
    key_k = row.get("KEY_K_EXIST", False)
    if plry and key_k:
        return "倍量柱 + 关键K"
    if plry:
        return "倍量柱(28日内≥3次)"
    if key_k:
        return "关键K(28日内存在)"
    return "未知"


def _fmt(val, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _safe_ratio(yang, yin) -> str:
    if yang is None or yin is None or yin == 0:
        return "N/A"
    return f"{yang / yin:.2f}"


def _macd_status(dif, hist) -> str:
    if dif is None or hist is None:
        return "N/A"
    parts = []
    parts.append("DIF>0(水上)" if dif > 0 else "DIF<0(水下)")
    parts.append("红柱" if hist > 0 else "绿柱")
    return ", ".join(parts)


def _macd_status_monthly(hist) -> str:
    if hist is None:
        return "N/A"
    return "红柱" if hist > 0 else "绿柱"


def _bool_cn(val) -> str:
    if val is None:
        return "N/A"
    return "是" if val else "否"
