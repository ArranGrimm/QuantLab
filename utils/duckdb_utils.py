import polars as pl

_A_SHARE_LOT_SIZE = 100.0

def get_adj_factor_frame(conn):
    """
    [辅助函数] 从数据库读取因子表，并计算所有股票的'前复权系数'
    逻辑：
    1. 前复权因子 = 当前累积dr / 最新累积dr
    2. 使用 lazy 模式处理，提高性能
    """
    # 1. 读取因子数据
    q_factor = pl.read_database(
        "SELECT code, date, dr FROM qmt_factors ORDER BY code, date", 
        conn
    ).lazy()

    # 2. 计算累积因子 (Cumulative Product)
    # 注意：如果没有因子记录的日子，后续 join_asof 会自动沿用最近的一次因子
    q_factor_processed = (
        q_factor
        .with_columns([
            pl.col("date").cast(pl.Date), # 确保是 Date 类型
            pl.col("dr").fill_null(1.0)   # 防御性编程，防止 null
        ])
        .sort(["code", "date"])
        .with_columns([
            # 按股票分组计算累积乘积
            pl.col("dr").cum_prod().over("code").alias("cum_dr")
        ])
    )
    
    # 3. 获取每个股票最新的累积因子 (用于归一化)
    # 这一步计算出每个股票当下的最新因子值
    q_last_factor = (
        q_factor_processed
        .group_by("code")
        .agg(pl.col("cum_dr").last().alias("latest_cum_dr"))
    )

    # 4. 合并并计算最终的前复权调节比例 (Adj Ratio)
    # ratio = cum_dr / latest_cum_dr
    # 这种方式保证了最新一天的价格与原始价格一致 (ratio=1)
    q_adj_ratio = (
        q_factor_processed
        .join(q_last_factor, on="code")
        .with_columns([
            (pl.col("cum_dr") / pl.col("latest_cum_dr")).alias("adj_ratio")
        ])
        .select(["code", "date", "adj_ratio"])
    )
    
    return q_adj_ratio

# ==============================================================================
# 方法 1: 加载日线数据 (包含复权计算 + 市值计算)
# ==============================================================================
def load_daily_data_full(conn, codes: list[str] = None, *, db_source: str = "qmt", tdx_db: str = ""):
    """
    功能：
    1. 读取行情数据（QMT: stock_daily 手动复权, TDX: v_stock_qfq 预计算）
    2. 计算前复权价格 (QMT) / 直接使用 (TDX)
    3. 关联股本计算市值
    4. 保持与旧代码一致的列结构

    Args:
        conn: DuckDB 连接 (QMT)
        codes: 股票代码列表 (QMT: sh.600000, TDX: sh600000)
        db_source: "qmt" | "tdx"
        tdx_db: TDX 数据库路径 (db_source="tdx" 时必填)
    """
    if db_source == "tdx":
        return _load_daily_from_tdx(tdx_db, codes, start_date="2019-01-01")

    code_filter = ""
    if codes:
        placeholders = ", ".join(f"'{c}'" for c in codes)
        code_filter = f" WHERE code IN ({placeholders})"

    # --- A. 读取基础数据 ---
    # 1. 日线行情 (Raw Data)
    q_daily = pl.read_database(
        f"SELECT code, date, open, high, low, close, volume, amount FROM stock_daily{code_filter}", 
        conn
    ).lazy().with_columns(pl.col("date").cast(pl.Date))

    # 2. 股本数据
    q_cap = pl.read_database(
        f"SELECT code, date, circulating_capital FROM finance_capital{code_filter} ORDER BY code, date", 
        conn
    ).lazy().with_columns(pl.col("date").cast(pl.Date))

    # 3. 复权因子 (调用辅助函数)
    q_factors = get_adj_factor_frame(conn)

    # --- B. 数据组装 ---
    
    q_full = (
        q_daily
        .sort(["code", "date"])
        
        # 1. 关联复权因子 (ASOF JOIN)
        # 找不到因子的日期(比如上市前或最近无分红)，会向前回溯找到最近的一个因子
        # 如果是上市初期没有任何因子，adj_ratio 可能会是 null，需要填充为 1.0 (基准)
        .join_asof(
            q_factors.sort(["code", "date"]), 
            on="date", 
            by="code", 
            strategy="backward"
        )
        .with_columns(pl.col("adj_ratio").fill_null(1.0)) # 填充无因子日期的默认值
        
        # 2. 计算前复权价格
        .with_columns([
            pl.col("open").alias("open_raw"),
            pl.col("high").alias("high_raw"),
            pl.col("low").alias("low_raw"),
            pl.col("close").alias("close_raw"),
            (pl.col("open") * pl.col("adj_ratio")).alias("open_adj"),
            (pl.col("high") * pl.col("adj_ratio")).alias("high_adj"),
            (pl.col("low") * pl.col("adj_ratio")).alias("low_adj"),
            (pl.col("close") * pl.col("adj_ratio")).alias("close_adj"),
            # A 股日线 volume 单位是“手”，需先还原成股数再计算价格。
            pl.when(pl.col("volume") > 0)
              .then(pl.col("amount") / (pl.col("volume") * _A_SHARE_LOT_SIZE))
              .otherwise(None)
              .alias("vwap_raw"),
        ])
        .with_columns([
            (pl.col("vwap_raw") * pl.col("adj_ratio")).alias("vwap_adj"),
        ])
        
        # 3. 关联股本数据 (ASOF JOIN)
        # 同样使用 backward 策略：用当日或当日之前最近的股本数据
        .join_asof(
            q_cap.sort(["code", "date"]),
            on="date",
            by="code",
            strategy="backward"
        )
        
        # 4. 计算市值 (保持旧逻辑: close_raw * circulating / 1e8)
        .with_columns([
            (pl.col("close_raw") * pl.col("circulating_capital") / 1e8).alias("market_cap_100m")
        ])
        
        # 5. 过滤停牌/无量数据 (保持旧逻辑)
        .filter(pl.col("volume") > 0)
        
        # 6. 前一日复权收盘价 (涨跌停判断、收益计算等通用需求)
        .with_columns(
            [
                pl.col("close_adj").shift(1).over("code").alias("pre_close_adj"),
                pl.col("close_raw").shift(1).over("code").alias("pre_close_raw"),
            ]
        )
        
        # 7. 选择并排序最终列
        .select([
            "code", "date", 
            "open_adj", "high_adj", "low_adj", "close_adj", "vwap_adj", "pre_close_adj",
            "open_raw", "high_raw", "low_raw", "close_raw", "pre_close_raw",
            "volume", "amount", 
            "vwap_raw", "market_cap_100m",
            "circulating_capital",
        ])
        .sort(["code", "date"])
    )
    
    return q_full


def _load_daily_from_tdx(db_path: str, codes: list[str] = None, *, start_date: str = "2019-01-01"):
    """从 TDX 数据库加载日线，输出列与 QMT load_daily_data_full 完全一致。"""
    import duckdb
    code_sql = ""
    if codes:
        tdx_codes = [c.replace(".", "") for c in codes]
        placeholders = ", ".join(f"'{c}'" for c in tdx_codes)
        code_sql = f" AND q.symbol IN ({placeholders})"

    tdx_conn = duckdb.connect(db_path, read_only=True)
    try:
        result = pl.read_database(
            f"""
            SELECT
                left(q.symbol, 2) || '.' || right(q.symbol, -2) AS code,
                q.date,
                q.open   AS open_adj,   q.high   AS high_adj,
                q.low    AS low_adj,    q.close  AS close_adj,
                q.preclose AS pre_close_adj,
                b.open   AS open_raw,   b.high   AS high_raw,
                b.low    AS low_raw,    b.close  AS close_raw,
                b.preclose AS pre_close_raw,
                CAST(b.volume AS DOUBLE) AS volume,
                b.amount,
                (q.open + q.high + q.low + q.close) / 4.0 AS vwap_adj,
                (b.open + b.high + b.low + b.close) / 4.0 AS vwap_raw,
                q.floatmv AS market_cap_100m,
                0.0       AS circulating_capital
            FROM v_stock_qfq q
            JOIN v_stock_bfq b USING(symbol, date)
            WHERE q.date >= '{start_date}'
              AND q.close > 0 AND b.close > 0
              {code_sql}
            ORDER BY code, q.date
            """,
            tdx_conn,
        )
        return result.lazy()
    finally:
        tdx_conn.close()


# ==============================================================================
# 方法 2: 加载 60分钟数据 (只计算前复权)
# ==============================================================================
def load_60m_data_adj(conn):
    """
    功能：
    1. 读取 stock_60m (Raw)
    2. 将 time 转换为 date 以便匹配因子
    3. 动态计算前复权
    """
    
    # --- A. 读取数据 ---
    # 1. 60分钟行情
    q_60m = pl.read_database(
        "SELECT code, time, open, high, low, close, volume, amount FROM stock_60m", 
        conn
    ).lazy()

    # 2. 复权因子 (复用)
    q_factors = get_adj_factor_frame(conn)

    # --- B. 计算逻辑 ---
    
    q_60m_adj = (
        q_60m
        .sort(["code", "time"])
        
        # 1. 生成辅助列 join_date 用于关联日线因子
        # 假设 time 是 Timestamp 类型，直接提取 Date
        .with_columns(pl.col("time").dt.date().alias("join_date"))
        
        # 2. 关联复权因子
        # 注意：这里用 join_date 去对齐因子的 date
        .join_asof(
            q_factors.sort(["code", "date"]),
            left_on="join_date",
            right_on="date",
            by="code",
            strategy="backward"
        )
        .with_columns(pl.col("adj_ratio").fill_null(1.0))
        
        # 3. 计算前复权价格
        .with_columns([
            (pl.col("open") * pl.col("adj_ratio")).alias("open_adj"),
            (pl.col("high") * pl.col("adj_ratio")).alias("high_adj"),
            (pl.col("low") * pl.col("adj_ratio")).alias("low_adj"),
            (pl.col("close") * pl.col("adj_ratio")).alias("close_adj"),
            pl.when(pl.col("volume") > 0)
              .then(pl.col("amount") / (pl.col("volume") * _A_SHARE_LOT_SIZE))
              .otherwise(None)
              .alias("vwap_raw"),
        ])
        .with_columns([
            (pl.col("vwap_raw") * pl.col("adj_ratio")).alias("vwap_adj"),
        ])
        
        # 4. 清理列
        .select([
            "code", "time", 
            "open_adj", "high_adj", "low_adj", "close_adj", "vwap_adj",
            "volume", "amount"
        ])
        .sort(["code", "time"])
    )
    
    return q_60m_adj # 返回 LazyFrame


# ==============================================================================
# 方法 3: 加载单支股票日线数据 (包含复权计算 + 市值计算)
# ==============================================================================
def load_daily_data_single(conn, code):
    """
    功能：读取单只股票数据，针对"无分红股票"做了容错处理
    """
    
    # --- A. 读取基础数据 (Lazy) ---
    q_daily = pl.read_database(
        f"SELECT code, date, open, high, low, close, volume, amount FROM stock_daily WHERE code = '{code}'", 
        conn
    ).lazy().with_columns(pl.col("date").cast(pl.Date))

    q_cap = pl.read_database(
        f"SELECT code, date, circulating_capital FROM finance_capital WHERE code = '{code}' ORDER BY code, date", 
        conn
    ).lazy().with_columns(pl.col("date").cast(pl.Date))

    # --- B. 处理复权因子 (核心修改点) ---
    
    # 1. 获取该股票的因子并立即执行 (collect)
    # 因为单只股票的因子数据量极小(几十行)，这里 collect 不会影响性能，反而能让我们检查 dataframe 是否为空
    q_factors_df = get_adj_factor_frame(conn).filter(pl.col("code") == code).collect()

    # 2. 分支逻辑
    if q_factors_df.height == 0:
        # 【分支 1】无除权记录：直接赋值 adj_ratio = 1.0
        q_daily_with_factor = q_daily.with_columns(pl.lit(1.0).alias("adj_ratio"))
    else:
        # 【分支 2】有除权记录：执行 join_asof
        # 注意：这里要把 q_factors_df 转回 lazy 模式参与计算
        q_daily_with_factor = (
            q_daily
            .sort(["code", "date"])
            .join_asof(
                q_factors_df.lazy().sort(["code", "date"]), 
                on="date", 
                by="code", 
                strategy="backward",
                check_sortedness=False
            )
            .with_columns(pl.col("adj_ratio").fill_null(1.0)) # 填充上市前的时间段
        )

    # --- C. 数据组装 (通用逻辑) ---
    
    q_single = (
        q_daily_with_factor
        
        # 1. 计算前复权价格
        .with_columns([
            (pl.col("open") * pl.col("adj_ratio")).alias("open_adj"),
            (pl.col("high") * pl.col("adj_ratio")).alias("high_adj"),
            (pl.col("low") * pl.col("adj_ratio")).alias("low_adj"),
            (pl.col("close") * pl.col("adj_ratio")).alias("close_adj"),
            pl.col("close").alias("close_raw") 
        ])
        
        # 2. 关联股本数据 (ASOF JOIN)
        .join_asof(
            q_cap.sort(["code", "date"]),
            on="date",
            by="code",
            strategy="backward",
            check_sortedness=False
        )
        
        # 3. 计算市值
        .with_columns([
            (pl.col("close_raw") * pl.col("circulating_capital") / 1e8).alias("market_cap_100m")
        ])
        
        # 4. 过滤与排序
        .filter(pl.col("volume") > 0)
        .select([
            "code", "date", 
            "open_adj", "high_adj", "low_adj", "close_adj", 
            "volume", "amount", 
            "close_raw", "market_cap_100m"
        ])
        .sort(["code", "date"])
    )
    
    return q_single.collect()


# ==============================================================================
# A股涨跌停标记 (与 Rust bt-core 保持一致的判定逻辑)
# ==============================================================================

_LIMIT_TOLERANCE = 0.001  # 0.1% 容差, 覆盖四舍五入精度误差

def add_price_limit_cols(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    为 DataFrame 添加 is_limit_up / is_limit_down 标记列.

    要求输入含 code, close_adj, pre_close_adj 列.
    规则与 Rust bt-core 完全一致:
      主板 (60/00) → ±10%, 创业板 (300/301) → ±20%, 科创板 (688/689) → ±20%

    不删除任何行, 仅打标记.
    """
    is_gem_star = (
        pl.col("code").str.starts_with("300")
        | pl.col("code").str.starts_with("301")
        | pl.col("code").str.starts_with("688")
        | pl.col("code").str.starts_with("689")
    )
    limit_pct = pl.when(is_gem_star).then(0.20).otherwise(0.10)
    daily_ret = pl.col("close_adj") / pl.col("pre_close_adj") - 1

    return df.with_columns([
        (daily_ret >= (limit_pct - _LIMIT_TOLERANCE)).alias("is_limit_up"),
        (daily_ret <= -(limit_pct - _LIMIT_TOLERANCE)).alias("is_limit_down"),
    ])
