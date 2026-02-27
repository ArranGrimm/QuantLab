@echo off

:: 执行命令
cargo run --release -- --config config_wmacd.toml --data ../data/signals/market_data_wmacd.parquet

pause