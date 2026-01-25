import baostock as bs
import polars as pl
import datetime
from tqdm import tqdm

def get_st_blacklist_pl(date_str=None):
    """
    使用 Baostock 获取全市场股票名称，筛选 ST 股，
    并利用 Polars 高效处理返回适配 QMT 格式的黑名单列表。
    """
    # 1. 登录 Baostock 系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f"Baostock 登录失败: {lg.error_msg}")
        return []

    # 2. 获取当前日期 (或者指定最近的一个交易日)
    # 注意：如果是周末或节假日，建议往前推几天，确保能取到数据
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    # 3. 获取全市场证券信息 (包含 code, code_name 等)
    rs = bs.query_all_stock(day=date_str)
    
    # 4. 收集数据 (Baostock 这一步必须循环，无法避免)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    bs.logout()

    if not data_list:
        print("未获取到 Baostock 数据，请检查日期是否为交易日")
        return []

    # ==============================================================================
    # 5. Polars 介入：高性能处理
    # ==============================================================================
    
    # 直接将 List 转为 Polars DataFrame
    # rs.fields 通常是: "code", "tradeStatus", "code_name"
    df = pl.DataFrame(data_list, schema=rs.fields, orient="row")
    
    # 逻辑处理链：
    # 1. 筛选 code_name 包含 "ST" 的行
    # 2. 转换 code 格式：Baostock 是 "sh.600000" -> QMT 是 "600000.SH"
    st_codes_series = (
        df
        .lazy() # 开启惰性执行优化
        .filter(pl.col("code_name").str.contains("ST")) # 同时覆盖 ST 和 *ST
        .select([
            # 字符串切片重组：
            # sh.600000 -> slice(3) 取后6位 + "." + slice(0,2) 取前2位并转大写
            (pl.col("code").str.slice(3) + "_" + pl.col("code").str.slice(0, 2).str.to_uppercase()).alias("qmt_code")
        ])
        .collect()
        .get_column("qmt_code")
    )
    
    result_list = st_codes_series.to_list()
    print(f"✅ 已获取 ST 黑名单，共 {len(result_list)} 只。")
    return result_list

def download_all_60m_data(start_date, end_date, save_path="a_share_60m.parquet"):
    # 1. 登录系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return

    print("正在获取全市场股票列表...")
    # 获取证券信息：股票、指数
    rs = bs.query_all_stock(day=end_date)
    
    # 转换为 Polars DataFrame 方便处理
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        print("未获取到股票列表，请检查日期或网络。")
        bs.logout()
        return

    df_stocks = pl.DataFrame(data_list, schema=["code", "tradeStatus", "code_name"], orient="row")
    
    # 过滤掉指数（BaoStock的指数没有分钟线数据，必须过滤，否则报错或空回）
    # 且只保留在交易状态的股票
    target_stocks = df_stocks.filter(
        (pl.col("code").str.starts_with("sh.6")) | 
        (pl.col("code").str.starts_with("sz."))
    )["code"].to_list()

    print(f"待下载股票数量: {len(target_stocks)} 只")

    # 2. 循环下载
    all_data = []
    
    # 使用 tqdm 显示进度条
    for code in tqdm(target_stocks, desc="Downloading", unit="stock"):
        # 频率：60=60分钟
        # 复权：2=前复权 (强烈建议策略回测用前复权)
        rs = bs.query_history_k_data_plus(
            code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date, 
            end_date=end_date,
            frequency="60", 
            adjustflag="3" 
        )
        
        if rs.error_code != '0':
            continue

        # 收集该股票的所有行
        stock_rows = []
        while rs.next():
            stock_rows.append(rs.get_row_data())
        
        if stock_rows:
            all_data.extend(stock_rows)

    bs.logout()

    print(f"下载完成，共获取 {len(all_data)} 条 K线数据。正在转换格式...")

    # 3. 数据清洗与类型转换 (这一步对 Parquet 极其重要)
    if not all_data:
        print("没有下载到任何数据。")
        return

    # 定义 Schema，防止全变成 String
    schema = {
        "date": pl.String, # 暂时存 String, 后面转 Date
        "time": pl.String, # 格式 YYYYMMDDHHMMSSssss
        "code": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "amount": pl.Float64,
        "adjustflag": pl.String
    }

    # 创建大表
    df_final = pl.DataFrame(all_data, schema=list(schema.keys()), orient="row")
    
    # 类型转换 (Cast)
    df_final = df_final.with_columns([
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("amount").cast(pl.Float64),
        # 将 time 字符串转换为时间戳，方便后续通过时间过滤
        pl.col("time").str.to_datetime("%Y%m%d%H%M%S%f").alias("datetime")
    ])

    # 4. 保存为 Parquet
    # 使用 zstd 压缩，体积极小且读取极快
    print(f"正在保存至 {save_path} ...")
    df_final.write_parquet(save_path, compression="zstd")
    print("保存成功！")

def create_adjustment_views(conn):
    print("🚀 正在创建前复权视图 (Views)...")

    # ================= 1. 日线前复权视图 (v_kline_daily_qfq) =================
    # 逻辑：K线价格 * 前复权因子 (fore_adjust_factor)
    # COALESCE(..., 1.0) 用于防止找不到因子时变成 NULL，默认不复权(1.0)
    conn.execute("""
        CREATE OR REPLACE VIEW v_kline_daily_qfq AS
        SELECT
            k.code,
            k.date,
            -- 价格字段复权
            k.open * COALESCE(f.fore_adjust_factor, 1.0) AS open,
            k.high * COALESCE(f.fore_adjust_factor, 1.0) AS high,
            k.low * COALESCE(f.fore_adjust_factor, 1.0) AS low,
            k.close * COALESCE(f.fore_adjust_factor, 1.0) AS close,
            k.preclose * COALESCE(f.fore_adjust_factor, 1.0) AS preclose,
            -- 其他字段保持原样
            k.volume,
            k.amount,
            k.turn,
            k.tradestatus,
            k.pctChg,
            k.isST,
            -- 保留因子方便核对
            f.fore_adjust_factor
        FROM kline_daily k
        ASOF LEFT JOIN adjust_factors f
            ON k.code = f.code AND k.date >= f.date
    """)
    print("✅ 日线前复权视图 [v_kline_daily_qfq] 创建完成")

    # ================= 2. 60分钟前复权视图 (v_kline_60m_qfq) =================
    # 逻辑：虽然是分钟线，但因子是按“日”生效的。
    # 所以依然是用 kline_60m.date 去匹配 adjust_factors.date
    conn.execute("""
        CREATE OR REPLACE VIEW v_kline_60m_qfq AS
        SELECT
            k.code,
            k.date,
            k.time,
            -- 价格字段复权
            k.open * COALESCE(f.fore_adjust_factor, 1.0) AS open,
            k.high * COALESCE(f.fore_adjust_factor, 1.0) AS high,
            k.low * COALESCE(f.fore_adjust_factor, 1.0) AS low,
            k.close * COALESCE(f.fore_adjust_factor, 1.0) AS close,
            -- 其它字段
            k.volume,
            k.amount,
            k.adjustflag,
            f.fore_adjust_factor
        FROM kline_60m k
        ASOF LEFT JOIN adjust_factors f
            ON k.code = f.code AND k.date >= f.date
    """)
    print("✅ 60分钟前复权视图 [v_kline_60m_qfq] 创建完成")



# --- 测试运行 ---
if __name__ == "__main__":
    st_list = get_st_blacklist_pl()
    # 打印前5个看看格式对不对
    print("样例:", st_list[:5])
    # --- 执行 ---
    # 获取过去一年的时间
    end = '2026-01-22'
    start = '2025-01-01'

    download_all_60m_data(start, end)
