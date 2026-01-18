import akshare as ak
import pandas as pd
import time

def generate_sector_file():
    print("🚀 开始下载东方财富行业板块数据...")
    
    # 1. 获取所有行业板块列表
    # 对应函数：stock_board_industry_name_em
    try:
        df_boards = ak.stock_board_industry_name_em()
        print(f"✅ 获取到 {len(df_boards)} 个行业板块")
    except Exception as e:
        print(f"❌ 获取板块列表失败: {e}")
        return

    # 准备一个列表存储所有结果
    all_data = []
    
    # 2. 遍历每个板块，获取成分股
    # 对应函数：stock_board_industry_cons_em
    total = len(df_boards)
    
    for i, row in df_boards.iterrows():
        board_name = row['板块名称']
        board_code = row['板块代码']
        
        print(f"[{i+1}/{total}] 正在获取: {board_name} ...", end="\r")
        
        try:
            # 获取该板块下的成分股
            df_cons = ak.stock_board_industry_cons_em(symbol=board_name)
            
            # 提取我们需要的数据
            # df_cons 通常包含 '代码', '名称' 等字段
            for _, stock_row in df_cons.iterrows():
                all_data.append({
                    "code": stock_row['代码'],      # 比如 000001
                    "name": stock_row['名称'],      # 比如 平安银行
                    "industry": board_name,        # 比如 银行
                    "industry_code": board_code    # 比如 BK0475
                })
            
            # ⚠️ 关键：稍微停顿一下，防止请求太快被东方财富封IP
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"\n⚠️ 跳过 {board_name}: {e}")

    print(f"\n🎉 数据下载完成，共收集 {len(all_data)} 条记录")
    
    # 3. 数据清洗与保存
    if not all_data:
        print("未获取到数据，请检查网络。")
        return

    df_final = pd.DataFrame(all_data)
    
    # 4. 格式化代码 (适配你的 Rust/Polars 系统)
    # 东方财富返回的是纯数字代码 (000001)，你的系统可能需要后缀 (000001_SZ)
    def format_code(c):
        c = str(c)
        if c.startswith('6'): return f"{c}_SH"
        if c.startswith('0') or c.startswith('3'): return f"{c}_SZ"
        if c.startswith('8') or c.startswith('4'): return f"{c}_BJ" # 北交所
        return c

    df_final['code'] = df_final['code'].apply(format_code)
    
    output_path = "data/sector_map_em.csv"
    df_final.to_csv(output_path, index=False)
    print(f"💾 文件已保存至: {output_path}")
    print("包含列: code, name, industry, industry_code")

if __name__ == "__main__":
    generate_sector_file()