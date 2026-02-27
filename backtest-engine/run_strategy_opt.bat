@echo off

:: 执行命令
cargo run --release -- --config config.toml --data ../data/signals/market_data_opt.parquet

pause