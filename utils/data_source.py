"""Market data access layer — single switch for QMT vs TDX daily OHLC."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Literal, Protocol

import duckdb
import polars as pl

from utils.duckdb_utils import duckdb_query_lazy, load_daily_data_full

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QMT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"
DEFAULT_TDX_DB = ROOT.parent / "QuantData" / "Ashare" / "tdx.db"

Provider = Literal["qmt", "tdx"]


@dataclass(frozen=True)
class DataSourceSettings:
    provider: Provider = "tdx"
    qmt_db: Path = DEFAULT_QMT_DB
    tdx_db: Path = DEFAULT_TDX_DB
    start_date: str = "2019-01-01"


def resolve_data_source(
    *,
    provider: str | None = None,
    qmt_db: Path | str | None = None,
    tdx_db: Path | str | None = None,
) -> DataSourceSettings:
    """Resolve active data source. Priority: explicit args > env > defaults."""
    env_provider = os.environ.get("QLAB_DATA_SOURCE", "").strip().lower()
    resolved_provider: Provider
    if provider:
        resolved_provider = _normalize_provider(provider)
    elif env_provider in ("qmt", "tdx"):
        resolved_provider = env_provider  # type: ignore[assignment]
    else:
        resolved_provider = "tdx"

    settings = DataSourceSettings(provider=resolved_provider)

    env_qmt = os.environ.get("QLAB_QMT_DB", "").strip()
    env_tdx = os.environ.get("QLAB_TDX_DB", "").strip()
    qmt_path = Path(qmt_db) if qmt_db else (Path(env_qmt) if env_qmt else settings.qmt_db)
    tdx_path = Path(tdx_db) if tdx_db else (Path(env_tdx) if env_tdx else settings.tdx_db)

    return replace(settings, qmt_db=qmt_path, tdx_db=tdx_path)


def _normalize_provider(value: str) -> Provider:
    normalized = value.strip().lower()
    if normalized not in ("qmt", "tdx"):
        raise ValueError(f"Unsupported data source: {value!r} (expected 'qmt' or 'tdx')")
    return normalized  # type: ignore[return-value]


class DailyMarketReader(Protocol):
    settings: DataSourceSettings

    def load_daily_full(self, codes: list[str] | None = None) -> pl.LazyFrame: ...

    def load_raw_ohlc(self, start_date: str, end_date: str) -> pl.DataFrame: ...

    def load_daily_qfq_simple(self, start_date: str) -> pl.DataFrame: ...

    def resolve_end_date(self) -> str: ...

    def close(self) -> None: ...


class QmtDailyReader:
    def __init__(self, settings: DataSourceSettings) -> None:
        self.settings = settings
        self._conn = duckdb.connect(str(settings.qmt_db), read_only=True)

    def load_daily_full(self, codes: list[str] | None = None) -> pl.LazyFrame:
        return load_daily_data_full(self._conn, codes)

    def load_raw_ohlc(self, start_date: str, end_date: str) -> pl.DataFrame:
        return self._conn.execute(
            """
            SELECT code, date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE date >= ? AND date <= ?
            ORDER BY code, date
            """,
            [start_date, end_date],
        ).pl()

    def load_daily_qfq_simple(self, start_date: str) -> pl.DataFrame:
        return self._conn.execute(
            """
            SELECT code, date, open, high, low, close, volume, amount
            FROM v_stock_daily_qfq_qmt
            WHERE date >= ?
            ORDER BY code, date
            """,
            [start_date],
        ).pl()

    def resolve_end_date(self) -> str:
        max_date = self._conn.execute("SELECT MAX(date) FROM v_stock_daily_qfq_qmt").fetchone()[0]
        return str(max_date)

    def close(self) -> None:
        self._conn.close()


class TdxDailyReader:
    def __init__(self, settings: DataSourceSettings) -> None:
        self.settings = settings
        self._conn = duckdb.connect(str(settings.tdx_db), read_only=True)

    def load_daily_full(self, codes: list[str] | None = None) -> pl.LazyFrame:
        code_sql = ""
        if codes:
            tdx_codes = [c.replace(".", "") for c in codes]
            placeholders = ", ".join(f"'{c}'" for c in tdx_codes)
            code_sql = f" AND q.symbol IN ({placeholders})"

        return duckdb_query_lazy(
            self._conn,
            f"""
            SELECT
                left(q.symbol, 2) || '.' || right(q.symbol, -2) AS code,
                q.date,
                q.open   AS open_adj,   q.high   AS high_adj,
                q.low    AS low_adj,    q.close  AS close_adj,
                q.preclose AS pre_close_adj,
                b.open   AS open_raw,   b.high   AS high_raw,
                b.low    AS low_raw,    b.close  AS close_raw,
                b.preclose AS pre_close_raw,
                CAST(b.volume AS DOUBLE) / 100.0 AS volume,
                b.amount,
                (q.open + q.high + q.low + q.close) / 4.0 AS vwap_adj,
                (b.open + b.high + b.low + b.close) / 4.0 AS vwap_raw,
                q.floatmv / 1e8 AS market_cap_100m,
                q.turnover AS turnover,
                CASE
                    WHEN q.turnover > 0 THEN CAST(b.volume AS DOUBLE) / q.turnover
                    ELSE 0.0
                END AS circulating_capital
            FROM v_stock_qfq q
            JOIN v_stock_bfq b USING(symbol, date)
            WHERE q.date >= '{self.settings.start_date}'
              AND q.close > 0 AND b.close > 0
              AND b.volume > 0
              AND q.symbol NOT LIKE 'bj%'
              AND b.symbol NOT LIKE 'bj%'
              {code_sql}
            ORDER BY code, q.date
            """,
        )

    def load_raw_ohlc(self, start_date: str, end_date: str) -> pl.DataFrame:
        return self._conn.execute(
            """
            SELECT
                left(symbol, 2) || '.' || right(symbol, -2) AS code,
                date, open, high, low, close,
                CAST(volume AS DOUBLE) AS volume,
                amount
            FROM v_stock_bfq
            WHERE date >= ? AND date <= ?
              AND close > 0
              AND symbol NOT LIKE 'bj%'
            ORDER BY code, date
            """,
            [start_date, end_date],
        ).pl()

    def load_daily_qfq_simple(self, start_date: str) -> pl.DataFrame:
        return self._conn.execute(
            """
            SELECT
                left(symbol, 2) || '.' || right(symbol, -2) AS code,
                date, open, high, low, close,
                CAST(volume AS DOUBLE) AS volume,
                amount
            FROM v_stock_qfq
            WHERE date >= ?
              AND close > 0
              AND symbol NOT LIKE 'bj%'
            ORDER BY code, date
            """,
            [start_date],
        ).pl()

    def resolve_end_date(self) -> str:
        max_date = self._conn.execute("SELECT MAX(date) FROM v_stock_qfq").fetchone()[0]
        return str(max_date)

    def close(self) -> None:
        self._conn.close()


def open_daily_reader(settings: DataSourceSettings) -> DailyMarketReader:
    if settings.provider == "tdx":
        return TdxDailyReader(settings)
    return QmtDailyReader(settings)


@contextmanager
def daily_reader(settings: DataSourceSettings) -> Iterator[DailyMarketReader]:
    reader = open_daily_reader(settings)
    try:
        yield reader
    finally:
        reader.close()
