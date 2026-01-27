import os
import math
import datetime
import pandas as pd
import duckdb
from xtquant import xtdata

# ==================== 配置区域 ====================
# Windows VM 内的路径 (映射到 Mac 的 Z 盘)
MAC_DATA_ROOT = r"Z:\Ashare"
DB_PATH = os.path.join(MAC_DATA_ROOT, "qmt_data.duckdb")

TARGET_SECTOR = '沪深A股'
INIT_START_DATE = '20150101'  # 首次初始化的起始日期
BATCH_SIZE = 50
# =================================================

# --- 财务数据配置 (核心逻辑) ---
# 定义 QMT 表名 -> DuckDB 表名 -> 字段映射
# 这样以后要加字段，只需要改这个字典，不用动代码
FINANCE_CONFIG = {
    'Capital': {
        'target_table': 'finance_capital',
        'col_map': {
            'm_timetag': 'date',                # 变动日
            'm_anntime': 'pub_date',            # 公告日
            'total_capital': 'total_capital',   # 总股本
            'circulating_capital': 'circulating_capital' # 流通股本
        }
    },
    'Balance': {
        'target_table': 'finance_balance',
        'col_map': {
            'm_timetag': 'date',
            'm_anntime': 'pub_date',
            # 归属于母公司股东权益 -> 净资产 (用于算 PB)
            'tot_shrhldr_eqy_excl_min_int': 'net_assets'
        }
    },
    'Income': {
        'target_table': 'finance_income',
        'col_map': {
            'm_timetag': 'date',
            'm_anntime': 'pub_date',
            # 归属于母公司所有者的净利润 -> 净利润 (用于算 PE)
            'net_profit_excl_min_int_inc': 'net_profit',
            # 营业收入 (用于算 PS)
            'revenue': 'revenue'
        }
    }
}

def fmt_code(qmt_code):
    """格式转换: 600570.SH -> sh.600570"""
    try:
        code, suffix = qmt_code.split('.')
        return f"{suffix.lower()}.{code}"
    except ValueError:
        return qmt_code

def get_start_date(conn, table: str) -> str | None:
    """
    获取增量同步的起始日期
    逻辑：如果表有数据，返回近三个月前的日期；否则返回 None
    返回格式：YYYYMMDD（QMT 格式）
    """
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count == 0:
            return None
        # 有数据：返回近三个月前的日期
        three_months_ago = datetime.date.today() - datetime.timedelta(days=90)
        return three_months_ago.strftime('%Y%m%d')
    except Exception:
        return None


def init_db(conn):
    """初始化所有必要的表"""
    
    # 1. 日线行情 (Raw)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_daily (
            code VARCHAR, date DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, amount DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    
    # 2. 60分钟行情 (Raw)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_60m (
            code VARCHAR, 
            time TIMESTAMP,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, amount DOUBLE,
            PRIMARY KEY (code, time)
        )
    """)
    
    # 3. 复权因子
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qmt_factors (
            code VARCHAR, date DATE,
            interest DOUBLE, stock_bonus DOUBLE, stock_gift DOUBLE, dr DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)

    # 3. 三大财务表 (根据配置自动生成建表语句)
    # Capital
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finance_capital (
            code VARCHAR, date DATE, pub_date DATE,
            total_capital DOUBLE, circulating_capital DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    # Balance
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finance_balance (
            code VARCHAR, date DATE, pub_date DATE,
            net_assets DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    # Income
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finance_income (
            code VARCHAR, date DATE, pub_date DATE,
            net_profit DOUBLE, revenue DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)

def on_progress(data):
    finished = data.get('finished', 0)
    total = data.get('total', 1)
    print(f"\r⏳ QMT下载中: {finished}/{total} ({finished/total:.1%})", end="")

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# --- 模块1: 日线行情同步 ---
def task_sync_daily(conn, stock_list, start_date: str):
    print(f"\n[1/4] 正在同步日线行情 ({start_date} ~ now)...")
    # QMT 自身是增量下载，start_time 用 INIT_START_DATE 确保本地缓存完整
    xtdata.download_history_data2(stock_list, period='1d', incrementally=True, callback=on_progress)
    
    fields = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']
    total_batches = math.ceil(len(stock_list) / BATCH_SIZE)

    for i, batch_codes in enumerate(chunks(stock_list, BATCH_SIZE)):
        print(f"\r🚀 日线批次 {i+1}/{total_batches} ...", end="")
        
        data_raw = xtdata.get_market_data_ex(field_list=fields, stock_list=batch_codes, period='1d', start_time=start_date, dividend_type='none')

        df_list = []
        for code, df in data_raw.items():
            if not df.empty:
                if 'time' not in df.columns: 
                    df = df.reset_index()
                df = df.rename(columns={'index': 'time'})
                # QMT 返回的是北京时间的毫秒时间戳
                df['date'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.date
                df['code'] = fmt_code(code)
                df_list.append(df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']])
        
        if df_list:
            big_df = pd.concat(df_list, ignore_index=True)
            conn.execute("""
                INSERT INTO stock_daily SELECT * FROM big_df
                ON CONFLICT (code, date) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume, amount = EXCLUDED.amount
            """)
    print("\n✅ 日线行情入库完毕")


# --- 模块2: 60分钟行情同步 ---
def task_sync_60m(conn, stock_list, start_date: str):
    print(f"\n[2/4] 正在同步60分钟行情 ({start_date} ~ now)...")
    # QMT 自身是增量下载，start_time 用 INIT_START_DATE 确保本地缓存完整
    xtdata.download_history_data2(stock_list, period='5m', incrementally=True, callback=on_progress)
    
    fields = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']
    total_batches = math.ceil(len(stock_list) / BATCH_SIZE)

    for i, batch_codes in enumerate(chunks(stock_list, BATCH_SIZE)):
        print(f"\r🚀 60分钟批次 {i+1}/{total_batches} ...", end="")
        
        data_raw = xtdata.get_market_data_ex(field_list=fields, stock_list=batch_codes, period='1h', start_time=start_date, dividend_type='none')

        df_list = []
        for code, df in data_raw.items():
            if not df.empty:
                if 'time' not in df.columns: 
                    df = df.reset_index()
                df = df.rename(columns={'index': 'time'})
                # QMT 返回的是北京时间的毫秒时间戳
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
                df['code'] = fmt_code(code)
                df_list.append(df[['code', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']])
        
        if df_list:
            big_df = pd.concat(df_list, ignore_index=True)
            conn.execute("""
                INSERT INTO stock_60m SELECT * FROM big_df
                ON CONFLICT (code, time) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume, amount = EXCLUDED.amount
            """)
    print("\n✅ 60分钟行情入库完毕")

# --- 辅助函数：落盘分红因子 ---
def _flush_dividend(conn, df_list):
    if not df_list:
        return
    big_df = pd.concat(df_list, ignore_index=True).rename(
        columns={'stockBonus': 'stock_bonus', 'stockGift': 'stock_gift'}
    )
    conn.execute("""
        INSERT INTO qmt_factors SELECT * FROM big_df
        ON CONFLICT (code, date) DO UPDATE SET
            interest=EXCLUDED.interest, stock_bonus=EXCLUDED.stock_bonus,
            stock_gift=EXCLUDED.stock_gift, dr=EXCLUDED.dr
    """)


# --- 模块3: 分红数据同步 ---
def task_sync_dividend(conn, stock_list, start_date: str = '1990-12-19'):
    print(f"\n[3/4] 正在同步分红因子({start_date} ~ now)...")
    df_list = []
    
    for idx, code in enumerate(stock_list):
        if idx % 100 == 0: 
            print(f"\r💾 处理中: {idx}/{len(stock_list)}", end="")
        
        # 使用 start_time 参数进行增量获取
        df = xtdata.get_divid_factors(code, start_time=start_date)
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns={'index': 'date'})
            df['code'] = fmt_code(code)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce').dt.date
            
            # 补全缺失列
            for c in ['interest', 'stockBonus', 'stockGift', 'dr']:
                if c not in df.columns: 
                    df[c] = 0.0 if c != 'dr' else 1.0
            
            df_list.append(df[['code', 'date', 'interest', 'stockBonus', 'stockGift', 'dr']])
            
        if len(df_list) >= 500:
            _flush_dividend(conn, df_list)
            df_list = []

    _flush_dividend(conn, df_list)
    print("\n✅ 分红数据同步完毕")

# --- 模块4: 财务数据同步 ---
def task_sync_finance(conn, stock_list):
    # 需要下载的表列表
    tables = list(FINANCE_CONFIG.keys()) # ['Capital', 'Balance', 'Income']
    
    print(f"\n[4/4] 正在同步财务数据 {tables}...")
    
    # 1. 批量下载 (一次性下载所有需要的表)
    xtdata.download_financial_data2(stock_list, table_list=tables, callback=on_progress)
    
    # 2. 批量处理并入库
    total_batches = math.ceil(len(stock_list) / BATCH_SIZE)
    
    for i, batch_codes in enumerate(chunks(stock_list, BATCH_SIZE)):
        print(f"\r🚀 财务批次 {i+1}/{total_batches} ...", end="")
        
        # 批量获取数据
        data_batch = xtdata.get_financial_data(batch_codes, table_list=tables)
        
        # 遍历每一种财务表 (Capital, Balance, Income)
        for table_name, config in FINANCE_CONFIG.items():
            target_table = config['target_table']
            col_map = config['col_map']
            
            df_list = []
            
            for code, table_dict in data_batch.items():
                df = table_dict.get(table_name)
                
                if df is not None and not df.empty:
                    # 标准化处理
                    df = df.copy()  # 避免 SettingWithCopyWarning
                    df['code'] = fmt_code(code)
                    
                    # 字段重命名与提取
                    temp_data = {'code': df['code']}
                    
                    for src_col, target_col in col_map.items():
                        if src_col in df.columns:
                            if 'time' in src_col or 'date' in src_col:
                                temp_data[target_col] = pd.to_datetime(df[src_col], format='%Y%m%d', errors='coerce').dt.date
                            else:
                                temp_data[target_col] = pd.to_numeric(df[src_col], errors='coerce')
                        else:
                            # 缺失列填 None
                            temp_data[target_col] = None
                    
                    sub_df = pd.DataFrame(temp_data)
                    df_list.append(sub_df)

            if df_list:
                big_df = pd.concat(df_list, ignore_index=True)
                
                # 动态生成 Upsert 语句
                # 构造 SET a=EXCLUDED.a, b=EXCLUDED.b ...
                update_cols = [f"{col}=EXCLUDED.{col}" for col in col_map.values() if col not in ['code', 'date']]
                update_str = ", ".join(update_cols)
                
                sql = f"""
                    INSERT INTO {target_table} BY NAME SELECT * FROM big_df
                    ON CONFLICT (code, date) DO UPDATE SET {update_str}
                """
                conn.execute(sql)

    print("\n✅ 所有财务数据同步完毕")

def main():
    print("🚀 QMT -> DuckDB 同步脚本启动...")
    
    try:
        xtdata.connect(port=58610)
    except Exception as e:
        print(f"❌ QMT 连接失败: {e}")
        return

    print(f"📂 数据库: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)
    
    try:
        init_db(conn)
        all_stocks = xtdata.get_stock_list_in_sector(TARGET_SECTOR)
        print(f"📋 标的: {len(all_stocks)} 只")
        start_daily = get_start_date(conn, 'stock_daily') or INIT_START_DATE
        # start_60m = get_start_date(conn, 'stock_60m') or INIT_START_DATE
        start_60m = '2025-01-01'

        print(f"📌 日线起始:     {start_daily}")
        print(f"📌 60分钟起始:   {start_60m}")
        print("📌 分红因子起始: 1990-12-19")
        
        # 执行同步任务
        task_sync_daily(conn, all_stocks, start_daily)
        task_sync_60m(conn, all_stocks, start_60m)
        task_sync_dividend(conn, all_stocks)
        # 财务数据始终全量同步（数据量不大，且有 ON CONFLICT 保护）
        task_sync_finance(conn, all_stocks)
        
        # 统计
        daily_count = conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
        m60_count = conn.execute("SELECT COUNT(*) FROM stock_60m").fetchone()[0]
        factor_count = conn.execute("SELECT COUNT(*) FROM qmt_factors").fetchone()[0]
        
        print("-" * 40)
        print(f"📈 日线行数:     {daily_count:,}")
        print(f"📉 60分钟行数:   {m60_count:,}")
        print(f"➗ 分红因子行数: {factor_count:,}")
        print("-" * 40)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 发生错误: {e}")
    finally:
        conn.close()
        print("\n🎉 任务完成!")


if __name__ == "__main__":
    main()