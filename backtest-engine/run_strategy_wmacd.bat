@echo off

:: 执行命令
cargo run -p bt-b1 --release -- --config crates/b1/config_wmacd.toml --data ../data/signals/market_data_wmacd.parquet

pause