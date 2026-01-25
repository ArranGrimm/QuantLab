import baostock as bs
import polars as pl
import duckdb
import datetime
import os
from tqdm import tqdm

# ================= 配置区域 =================
DB_PATH = "/Users/zhangyubo/Projects/QuantData/Ashare/baostock_data.duckdb"
INIT_START_DATE = "2018-01-01" 
UPDATE_LOOKBACK_DAYS = 15
BATCH_SIZE = 500  # 每500只股票落盘一次，平衡效率与断点保护
# ===========================================

def init_db(conn):
    """初始化数据库表结构 (三张表) - 使用优化的数据类型"""
    
    # 1. 60分钟 K线表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kline_60m (
            code VARCHAR,
            date DATE,
            time TIMESTAMP,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, amount DOUBLE,
            adjustflag TINYINT,
            PRIMARY KEY (code, time)
        )
    """)
    
    # 2. 日线 K线表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kline_daily (
            code VARCHAR,
            date DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            preclose DOUBLE,
            volume DOUBLE, amount DOUBLE,
            adjustflag TINYINT,
            turn DOUBLE,
            tradestatus TINYINT,
            pctChg DOUBLE,
            isST BOOLEAN,
            PRIMARY KEY (code, date)
        )
    """)
    
    # 3. 复权因子表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS adjust_factors (
            code VARCHAR,
            date DATE,
            fore_adjust_factor DOUBLE,
            back_adjust_factor DOUBLE,
            adjust_factor DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)

def get_last_date(conn, table: str, date_col: str = 'date') -> str | None:
    """
    获取增量同步的起始日期
    逻辑：取表中最大日期，作为增量起点
    个别落后的股票靠 ON CONFLICT 处理
    """
    try:
        result = conn.execute(f"""
            SELECT MAX({date_col}) FROM {table}
        """).fetchone()[0]
        if result is None:
            return None
        # 如果是 TIMESTAMP，只取日期部分
        return str(result)[:10]
    except Exception:
        return None

# --- 辅助函数：安全转换为 Float64（空字符串 -> None）---
def _safe_float(col_name: str):
    return pl.when(pl.col(col_name) == '').then(None).otherwise(pl.col(col_name)).cast(pl.Float64).alias(col_name)


# --- 辅助函数：落盘60分钟数据 ---
def _flush_60m(conn, buffer):
    if not buffer:
        return
    df = pl.DataFrame(
        buffer,
        schema=['code', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag'],
        orient='row'
    ).with_columns([
        pl.col('date').str.to_date('%Y-%m-%d'),
        pl.col('time').str.to_datetime('%Y%m%d%H%M%S%3f'),
        _safe_float('open'),
        _safe_float('high'),
        _safe_float('low'),
        _safe_float('close'),
        _safe_float('volume'),
        _safe_float('amount'),
        pl.when(pl.col('adjustflag') == '').then(None).otherwise(pl.col('adjustflag')).cast(pl.Int8).alias('adjustflag'),
    ])
    conn.execute("""
        INSERT INTO kline_60m SELECT * FROM df
        ON CONFLICT (code, time) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, 
            close=EXCLUDED.close, volume=EXCLUDED.volume, amount=EXCLUDED.amount
    """)


# --- 任务 A: 下载60分钟线 ---
def task_download_60m(conn, code_list, start_date, end_date):
    print(f"\n🚀 任务 A: 同步 60分钟 K线 ({start_date} ~ {end_date})...")
    
    buffer = []
    stock_count = 0
    pbar = tqdm(code_list, unit="stock", ncols=100)
    
    for code in pbar:
        pbar.set_description(f"📉 60m: {code}")
        
        rs = bs.query_history_k_data_plus(
            code,
            "code,date,time,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date, end_date=end_date,
            frequency="60", adjustflag="3"
        )
        
        if rs.error_code == '0':
            while rs.next():
                buffer.append(rs.get_row_data())
        
        stock_count += 1
        if stock_count % BATCH_SIZE == 0:
            pbar.set_description(f"💾 落盘 {len(buffer):,} 行...")
            _flush_60m(conn, buffer)
            buffer = []
    
    # 剩余数据落盘
    if buffer:
        print(f"💾 60分钟线最后落盘 ({len(buffer):,} 行)...")
        _flush_60m(conn, buffer)


# --- 辅助函数：落盘日线数据 ---
def _flush_daily(conn, buffer):
    if not buffer:
        return
    df = pl.DataFrame(
        buffer,
        schema=['code', 'date', 'open', 'high', 'low', 'close', 'preclose', 
                'volume', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'isST'],
        orient='row'
    ).with_columns([
        pl.col('date').str.to_date('%Y-%m-%d'),
        _safe_float('open'),
        _safe_float('high'),
        _safe_float('low'),
        _safe_float('close'),
        _safe_float('preclose'),
        _safe_float('volume'),
        _safe_float('amount'),
        pl.when(pl.col('adjustflag') == '').then(None).otherwise(pl.col('adjustflag')).cast(pl.Int8).alias('adjustflag'),
        _safe_float('turn'),
        pl.when(pl.col('tradestatus') == '').then(None).otherwise(pl.col('tradestatus')).cast(pl.Int8).alias('tradestatus'),
        _safe_float('pctChg'),
        (pl.col('isST') == '1').alias('isST'),
    ])
    conn.execute("""
        INSERT INTO kline_daily SELECT * FROM df
        ON CONFLICT (code, date) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, 
            close=EXCLUDED.close, preclose=EXCLUDED.preclose,
            volume=EXCLUDED.volume, amount=EXCLUDED.amount,
            turn=EXCLUDED.turn, pctChg=EXCLUDED.pctChg
    """)


# --- 任务 B: 下载日线 ---
def task_download_daily(conn, code_list, start_date, end_date):
    print(f"\n🚀 任务 B: 同步日线 K线 ({start_date} ~ {end_date})...")
    
    buffer = []
    stock_count = 0
    pbar = tqdm(code_list, unit="stock", ncols=100)
    
    for code in pbar:
        pbar.set_description(f"📈 日线: {code}")
        
        rs = bs.query_history_k_data_plus(
            code,
            "code,date,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3"
        )
        
        if rs.error_code == '0':
            while rs.next():
                buffer.append(rs.get_row_data())
        
        stock_count += 1
        if stock_count % BATCH_SIZE == 0:
            pbar.set_description(f"💾 落盘 {len(buffer):,} 行...")
            _flush_daily(conn, buffer)
            buffer = []
    
    # 剩余数据落盘
    if buffer:
        print(f"💾 日线最后落盘 ({len(buffer):,} 行)...")
        _flush_daily(conn, buffer)


# --- 辅助函数：落盘因子数据 ---
def _flush_factors(conn, buffer):
    if not buffer:
        return
    df = pl.DataFrame(
        buffer,
        schema=['code', 'date', 'fore_adjust_factor', 'back_adjust_factor', 'adjust_factor'],
        orient='row'
    ).with_columns([
        pl.col('date').str.to_date('%Y-%m-%d'),
        pl.col('fore_adjust_factor').cast(pl.Float64),
        pl.col('back_adjust_factor').cast(pl.Float64),
        pl.col('adjust_factor').cast(pl.Float64),
    ])
    conn.execute("""
        INSERT INTO adjust_factors SELECT * FROM df
        ON CONFLICT (code, date) DO UPDATE SET
            fore_adjust_factor=EXCLUDED.fore_adjust_factor,
            back_adjust_factor=EXCLUDED.back_adjust_factor,
            adjust_factor=EXCLUDED.adjust_factor
    """)


# --- 任务 C: 下载复权因子 ---
def task_download_factors(conn, code_list, start_date, end_date):
    print(f"\n🚀 任务 C: 同步复权因子 ({start_date} ~ {end_date})...")
    
    buffer = []
    stock_count = 0
    pbar = tqdm(code_list, unit="stock", ncols=100)
    
    for code in pbar:
        pbar.set_description(f"➗ 因子: {code}")
        
        rs = bs.query_adjust_factor(code=code, start_date=start_date, end_date=end_date)
        
        if rs.error_code == '0':
            while rs.next():
                buffer.append(rs.get_row_data())
        
        stock_count += 1
        if stock_count % BATCH_SIZE == 0:
            pbar.set_description(f"💾 落盘 {len(buffer):,} 行...")
            _flush_factors(conn, buffer)
            buffer = []
    
    # 剩余数据落盘
    if buffer:
        print(f"💾 因子最后落盘 ({len(buffer):,} 行)...")
        _flush_factors(conn, buffer)

def get_stock_list(date: str) -> list[str]:
    """获取 A 股股票列表 (排除指数/基金/债券)"""
    rs = bs.query_all_stock(day=date)
    code_list = []
    while rs.next():
        code = rs.get_row_data()[0]
        # 上海: sh.600xxx, sh.601xxx, sh.603xxx, sh.605xxx (主板), sh.688xxx (科创板)
        # 深圳: sz.000xxx (主板), sz.002xxx (中小板), sz.300xxx, sz.301xxx (创业板)
        if code.startswith('sh.60') or code.startswith('sh.68'):
            code_list.append(code)
        elif code.startswith('sz.00') or code.startswith('sz.30'):
            code_list.append(code)
    return code_list


def main(year: int = None, end_date: str = None):
    """
    主函数
    Args:
        year: 指定年份模式，如 2024。设置后只同步该年数据，方便测试。
              None 时使用自动模式（基于各表的最大日期增量同步）
    """
    mode_desc = f"指定年份 {year}" if year else "自动模式"
    print(f"🚀 BaoStock 同步脚本启动 [{mode_desc}]...")
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    init_db(conn)
    
    # 确定日期范围
    if year:
        # 指定年份模式：只同步该年
        start_60m = f"{year}-01-01"
        start_daily = f"{year}-01-01"
        end_date = f"{year}-12-31"
        mode = f"YEAR_{year}"
    else:
        # 自动模式：基于各表的最大日期增量同步
        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        mode = "AUTO"
        
        # 60分钟线：从最大日期开始（表空则从 INIT_START_DATE）
        last_60m = get_last_date(conn, 'kline_60m')
        start_60m = last_60m if last_60m else INIT_START_DATE
        
        # 日线：从最大日期开始
        last_daily = get_last_date(conn, 'kline_daily')
        start_daily = last_daily if last_daily else INIT_START_DATE
        
        # 复权因子：跟随日线的增量起点（表空则从 1990-12-19 全量）
        has_factor_data = conn.execute("SELECT COUNT(*) FROM adjust_factors").fetchone()[0] > 0
        start_factor = start_daily if has_factor_data else "1990-12-19"

    print(f"📌 模式: {mode}")
    print(f"📌 60分钟线: {start_60m} ~ {end_date}")
    print(f"📌 日线:     {start_daily} ~ {end_date}")
    print(f"📌 复权因子: {start_factor} ~ {end_date}")
    
    # 登录
    bs.login()
    
    # 获取标的
    print("📋 获取股票列表...")
    code_list = get_stock_list(end_date)
    print(f"📋 共 {len(code_list)} 只A股")
    
    # 执行三大任务（各自使用独立的起始日期）
    task_download_60m(conn, code_list, start_60m, end_date)
    task_download_daily(conn, code_list, start_daily, end_date)
    if year is None:
        task_download_factors(conn, code_list, start_factor, end_date)
        print("✅ 复权因子下载完成")
    else:
        print("✅ 复权因子不下载, 需要用户自己判断数据日期")
    
    # 统计
    k60_count = conn.execute("SELECT count(*) FROM kline_60m").fetchone()[0]
    kd_count = conn.execute("SELECT count(*) FROM kline_daily").fetchone()[0]
    f_count = conn.execute("SELECT count(*) FROM adjust_factors").fetchone()[0]
    
    bs.logout()
    conn.close()
    
    print("-" * 40)
    print(f"📊 60分钟行数: {k60_count:,}")
    print(f"📈 日线行数:   {kd_count:,}")
    print(f"➗ 因子行数:   {f_count:,}")
    print(f"📂 数据库: {DB_PATH}")
    print("-" * 40)


if __name__ == "__main__":
    # 示例用法:
    main(end_date='2026-01-23')        # 自动模式 (INIT 或 UPDATE)
    # main(2024)    # 只同步 2024 年数据 (测试用)
    # main(2025)