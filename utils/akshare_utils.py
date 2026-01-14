import akshare as ak
import pandas as pd

def generate_sector_file():
    print("🚀 开始下载申万一级行业数据 (基于 Akshare)...")
    
    # 1. 获取行业列表
    df_index = ak.index_classify_sw(level="L1")
    
    all_data = []
    
    # 2. 遍历每个行业获取成分股
    for idx, row in df_index.iterrows():
        sector_name = row['index_name']
        sector_code = row['index_code']
        print(f"正在获取: {sector_name} ({sector_code})...")
        
        try:
            # 获取成分股
            df_member = ak.index_component_sw(index_code=sector_code)
            # 统一列名格式，方便 Polars 读取
            # Akshare返回的 code 通常是 "000001" 这种，你需要根据你的系统转成 "000001_SZ" 或者保持原样
            for stock_code in df_member['stock_code']:
                # 简单的数据清洗
                all_data.append({
                    "code": stock_code,
                    "industry": sector_name
                })
        except Exception as e:
            print(f"⚠️ 获取 {sector_name} 失败: {e}")

    # 3. 保存
    df_final = pd.DataFrame(all_data)
    
    # 这里做一步简单的后缀处理，为了匹配你 Rust 里的 code 格式 (如 600000_SH)
    # 如果你的系统里 code 主要是带后缀的，可以用简单的规则加一下
    def format_code(c):
        if c.startswith('6'): return f"{c}_SH"  # noqa: E701
        if c.startswith('0') or c.startswith('3'): return f"{c}_SZ"  # noqa: E701
        if c.startswith('8') or c.startswith('4'): return f"{c}_BJ"  # noqa: E701
        return c

    df_final['code'] = df_final['code'].apply(format_code)
    
    output_path = "data/sector_map.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ 行业分类表已保存至: {output_path}，共 {len(df_final)} 条数据")

if __name__ == "__main__":
    generate_sector_file()