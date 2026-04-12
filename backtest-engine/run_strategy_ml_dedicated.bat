@echo off

cargo run -p bt-b1 --release -- --config crates/b1/config_ml.toml --data ../data/signals/market_data_b1ml_dedicated.parquet

pause