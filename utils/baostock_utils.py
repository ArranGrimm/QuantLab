import akshare as ak
import polars as pl
import datetime
import json
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"
_ST_CACHE = _CACHE_DIR / "st_blacklist.json"
_CACHE_MAX_AGE_DAYS = 1


def _code_to_baostock(code: str) -> str:
    """纯数字代码 → sh.600000 / sz.000001 格式"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def _load_cache(max_age_days: int | None = _CACHE_MAX_AGE_DAYS) -> list[str] | None:
    """读取本地缓存，过期则返回 None"""
    if not _ST_CACHE.exists():
        return None
    try:
        data = json.loads(_ST_CACHE.read_text())
        cached_date = datetime.date.fromisoformat(data["date"])
        if max_age_days is None or (datetime.date.today() - cached_date).days <= max_age_days:
            return data["codes"]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _save_cache(codes: list[str]):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _ST_CACHE.write_text(
        json.dumps({"date": datetime.date.today().isoformat(), "codes": codes},
                   ensure_ascii=False)
    )


def get_st_blacklist_pl(date_str=None) -> list[str]:
    """
    获取 ST 黑名单 (sh.600000 格式)。

    优先使用 AKShare stock_zh_a_st_em (东方财富，单次 HTTP)，
    结果缓存到 data/.cache/st_blacklist.json，1 天内复用。
    网络失败时回退到缓存；缓存也没有时回退到 Baostock。
    date_str 参数保留向后兼容，但 AKShare 接口不需要。
    """
    cached = _load_cache()
    if cached is not None:
        print(f"[ST] 使用本地缓存，共 {len(cached)} 只")
        return cached

    # ── 方案 A: AKShare (快，推荐) ──
    try:
        df = ak.stock_zh_a_st_em()
        codes = [_code_to_baostock(c) for c in df["代码"].tolist()]
        _save_cache(codes)
        print(f"[ST] AKShare 获取成功，共 {len(codes)} 只 (已缓存)")
        return codes
    except Exception as e:
        print(f"[ST] AKShare 失败: {e}，尝试 Baostock...")

    # ── 方案 B: Baostock 兜底 ──
    codes = _get_st_baostock(date_str)
    if codes:
        return codes

    # ── 方案 C: 网络不可用时允许使用过期缓存，避免回测 universe 漂移 ──
    stale_cached = _load_cache(max_age_days=None)
    if stale_cached is not None:
        print(f"[ST] 实时源失败，使用过期本地缓存，共 {len(stale_cached)} 只")
        return stale_cached

    return []


def _get_st_baostock(date_str=None) -> list[str]:
    """Baostock 兜底方案 (慢但稳)"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"Baostock 登录失败: {lg.error_msg}")
        return []

    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    rs = bs.query_all_stock(day=date_str)
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    if not data_list:
        print("未获取到 Baostock 数据，请检查日期是否为交易日")
        return []

    df = pl.DataFrame(data_list, schema=rs.fields, orient="row")
    codes = (
        df.lazy()
        .filter(pl.col("code_name").str.contains("ST"))
        .select("code")
        .collect()
        .get_column("code")
        .to_list()
    )

    _save_cache(codes)
    print(f"[ST] Baostock 获取成功，共 {len(codes)} 只 (已缓存)")
    return codes


if __name__ == "__main__":
    st_list = get_st_blacklist_pl()
    print("样例:", st_list[:5])
