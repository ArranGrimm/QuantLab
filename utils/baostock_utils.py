"""Baostock 数据工具：ST 黑名单 + 行业分类。

不再依赖 AKShare。ST 列表由 Baostock query_stock_basic 获取，结果本地缓存。
行业分类使用 query_stock_industry（申万一级行业）。
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import polars as pl

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"
_ST_CACHE = _CACHE_DIR / "st_blacklist.json"
_INDUSTRY_CACHE = _CACHE_DIR / "industry_map.json"
_CACHE_MAX_AGE_DAYS = 1


# ── generic cache helpers ──────────────────────────────────────────────


def _load_cache(path: Path, max_age_days: int | None = _CACHE_MAX_AGE_DAYS) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_date = datetime.date.fromisoformat(data["date"])
        if max_age_days is None or (datetime.date.today() - cached_date).days <= max_age_days:
            return data
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _save_cache(path: Path, payload: dict):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload["date"] = datetime.date.today().isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False))


# ── ST 黑名单 ──────────────────────────────────────────────────────────


def get_st_blacklist_pl(date_str: str | None = None) -> list[str]:
    """获取 ST 黑名单 (sh.600000 格式)。

    使用 Baostock query_stock_basic 获取全量股票基本信息，
    从股票名称中筛选含 "ST" 的标的。结果缓存 1 天。
    网络失败时回退到缓存；缓存也没有时返回空列表。
    """
    cached = _load_cache(_ST_CACHE)
    if cached is not None:
        codes = cached.get("codes", [])
        print(f"[ST] 使用本地缓存，共 {len(codes)} 只")
        return codes

    codes = _fetch_st_baostock(date_str)
    if codes:
        return codes

    # 兜底: 允许使用过期缓存
    stale = _load_cache(_ST_CACHE, max_age_days=None)
    if stale is not None:
        codes = stale.get("codes", [])
        print(f"[ST] 实时源失败，使用过期本地缓存，共 {len(codes)} 只")
        return codes

    return []


def _fetch_st_baostock(date_str: str | None = None) -> list[str]:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[ST] Baostock 登录失败: {lg.error_msg}")
        return []

    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    rs = bs.query_stock_basic()
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    if not data_list:
        print("[ST] 未获取到 Baostock 数据，请检查日期是否为交易日")
        return []

    fields = rs.fields if rs.fields else ["code", "code_name", "ipoDate", "outDate", "type", "status"]
    df = pl.DataFrame(data_list, schema=fields, orient="row")

    # 筛选名称含 ST 的，且未退市
    codes = (
        df.lazy()
        .filter(
            pl.col("code_name").str.contains("ST")
            & (pl.col("status") == "1")
        )
        .select("code")
        .collect()
        .get_column("code")
        .to_list()
    )

    _save_cache(_ST_CACHE, {"codes": codes})
    print(f"[ST] Baostock 获取成功，共 {len(codes)} 只 (已缓存)")
    return codes


# ── 行业分类 ──────────────────────────────────────────────────────────


def get_stock_industry(date_str: str | None = None) -> pl.DataFrame:
    """获取全量股票行业分类 (申万一级行业)。

    通过 Baostock query_stock_industry 获取，结果缓存 1 天。

    Returns:
        DataFrame with columns: code, code_name, industry
    """
    cached = _load_cache(_INDUSTRY_CACHE)
    if cached is not None:
        rows = cached.get("rows", [])
        print(f"[行业] 使用本地缓存，共 {len(rows)} 只")
        return pl.DataFrame(rows)

    rows = _fetch_industry_baostock(date_str)
    if rows:
        return pl.DataFrame(rows)

    # 兜底: 过期缓存
    stale = _load_cache(_INDUSTRY_CACHE, max_age_days=None)
    if stale is not None:
        rows = stale.get("rows", [])
        print(f"[行业] 实时源失败，使用过期本地缓存，共 {len(rows)} 只")
        return pl.DataFrame(rows)

    return pl.DataFrame()


def _fetch_industry_baostock(date_str: str | None = None) -> list[dict]:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[行业] Baostock 登录失败: {lg.error_msg}")
        return []

    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    rs = bs.query_stock_industry()
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    if not data_list:
        return []

    fields = rs.fields if rs.fields else ["code", "code_name", "industry", "industryClassification"]
    result = []
    for row in data_list:
        d = dict(zip(fields, row))
        result.append({"code": d["code"], "code_name": d.get("code_name", ""), "industry": d.get("industry", "")})

    _save_cache(_INDUSTRY_CACHE, {"rows": result})
    print(f"[行业] Baostock 获取成功，共 {len(result)} 只 (已缓存)")
    return result


# ── CLI ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    st_list = get_st_blacklist_pl()
    print("ST 样例:", st_list[:5])

    industry_df = get_stock_industry()
    print(f"行业分类: {industry_df.height} 只, {industry_df['industry'].n_unique()} 个行业")
    print(industry_df.head(5))
