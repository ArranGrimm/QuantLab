import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import polars as pl
    return (duckdb,)


@app.cell
def _(duckdb):
    DB_PATH = "/Users/zhangyubo/Projects/QuantData/Ashare/qmt_data.duckdb"

    conn = duckdb.connect(DB_PATH)
    return (conn,)


@app.cell
def _():
    test_code = 'sh.600036'
    return (test_code,)


@app.cell
def _(conn, test_code):
    df3 = conn.execute(f"""
        SELECT 
            date,
            interest,
            stock_bonus,
            stock_gift,
            dr
        FROM qmt_factors 
        WHERE code = '{test_code}' 
        ORDER BY date 
    
    """).pl()
    return (df3,)


@app.cell
def _(df3):
    df3
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
