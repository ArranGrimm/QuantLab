import baostock as bs
import polars as pl
import datetime

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

# --- 测试运行 ---
if __name__ == "__main__":
    st_list = get_st_blacklist_pl()
    # 打印前5个看看格式对不对
    print("样例:", st_list[:5])