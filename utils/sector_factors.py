import polars as pl

def get_sector_status(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算板块状态表 (粒度：日期 + 行业)
    返回列: [date, industry, SECTOR_OK, sector_breadth]
    """
    # 关联行业，并计算个股是否强势 (MA20)
    # 注意：这里我们只需要这几列来计算板块，不需要把所有列都拖进来
    base = df.select(["code", "date", "close_adj", "industry"]).with_columns(
        pl.col("close_adj").shift(1).over("code").alias("prev_close"),
    )
    
    # 计算个股强势状态 (站上MA20)
    stock_status = base.sort(["code", "date"]).with_columns([
        pl.col("close_adj").rolling_mean(20).over("code").alias("ma20"),
        (pl.col("close_adj") / pl.col("prev_close") - 1).alias("pct_change")
    ]).with_columns([
        (pl.col("close_adj") > pl.col("ma20")).alias("is_strong")
    ])
    
    # 3. 聚合计算板块指标 (Group By 在这里发生，粒度变粗)
    sector_stats = stock_status.group_by(["date", "industry"]).agg([
        # 宽度: 强势股占比
        pl.col("is_strong").mean().alias("breadth"),
        # 趋势: 等权涨跌幅
        pl.col("pct_change").mean().alias("idx_pct")
    ]).sort(["industry", "date"])
    
    # 4. 计算板块趋势 (Window Function over industry)
    result = sector_stats.with_columns([
        # 合成指数
        (1 + pl.col("idx_pct")).cum_prod().over("industry").alias("idx"),
    ]).with_columns([
        # 指数均线
        pl.col("idx").rolling_mean(20).over("industry").alias("idx_ma20")
    ]).with_columns([
        # 定义共振信号
        (
            (pl.col("breadth") > 0.4) &             # 宽度 > 40%
            (pl.col("idx") > pl.col("idx_ma20"))    # 板块指数多头
        ).alias("SECTOR_OK")
    ])
    
    # 只返回关键列，方便后续 Join
    return result.select(["date", "industry", "SECTOR_OK", "breadth"])