"""儲存：DuckDB。

- transactions：清理後的成交資料（處理層 clean_record 的輸出 + loaded_at）
- fetch_log：data/raw/fetch_log.jsonl 的內容（每次打 API 的紀錄，含 404）

重新載入以原始檔為單位「先刪後插」，同一個檔案載入幾次都不會重複。
"""

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


def _parsed(name: str, sql_type: str) -> list[tuple[str, str]]:
    """數值欄位的三欄：原字串、數值、status。"""
    return [(f"{name}_raw", "VARCHAR"), (name, sql_type), (f"{name}_status", "VARCHAR")]


# 欄位順序和 processing.clean.clean_record 的輸出一致（有測試檢查），最後加上 loaded_at
TRANSACTION_COLUMNS: list[tuple[str, str]] = [
    ("source", "VARCHAR"),
    ("city", "VARCHAR"),
    ("year", "INTEGER"),
    ("quarter", "INTEGER"),
    ("price_classification", "VARCHAR"),
    ("raw_file", "VARCHAR"),
    ("row_index", "INTEGER"),
    ("price_category", "VARCHAR"),
    ("property_type", "VARCHAR"),
    ("region", "VARCHAR"),
    ("prefecture", "VARCHAR"),
    ("municipality", "VARCHAR"),
    ("municipality_code", "VARCHAR"),
    ("district_name", "VARCHAR"),
    ("district_code", "VARCHAR"),
    ("land_shape", "VARCHAR"),
    ("use", "VARCHAR"),
    ("purpose", "VARCHAR"),
    ("road_direction", "VARCHAR"),
    ("road_classification", "VARCHAR"),
    ("city_planning", "VARCHAR"),
    ("period", "VARCHAR"),
    ("renovation", "VARCHAR"),
    ("remarks", "VARCHAR"),
    ("floor_plan_raw", "VARCHAR"),
    ("floor_plan", "VARCHAR"),
    ("structure_raw", "VARCHAR"),
    ("structure", "VARCHAR"),
    *_parsed("trade_price", "BIGINT"),
    *_parsed("area_m2", "DOUBLE"),
    *_parsed("total_floor_area_m2", "DOUBLE"),
    *_parsed("frontage_m", "DOUBLE"),
    *_parsed("road_breadth_m", "DOUBLE"),
    *_parsed("land_unit_price", "BIGINT"),
    *_parsed("price_per_tsubo", "BIGINT"),
    *_parsed("coverage_ratio", "INTEGER"),
    *_parsed("floor_area_ratio", "INTEGER"),
    *_parsed("building_year", "INTEGER"),
    ("unit_price_per_m2", "DOUBLE"),
    ("unit_price_per_m2_status", "VARCHAR"),
    ("loaded_at", "TIMESTAMP"),
]

_FETCH_LOG_COLUMNS = [
    ("fetched_at", "TIMESTAMPTZ"),
    ("city", "VARCHAR"),
    ("year", "INTEGER"),
    ("quarter", "INTEGER"),
    ("price_classification", "VARCHAR"),
    ("http_status", "INTEGER"),
    ("result", "VARCHAR"),
    ("file", "VARCHAR"),
]


def _ddl(table: str, columns: list[tuple[str, str]], extra: str = "") -> str:
    # 欄位名一律加雙引號（use 之類的名字可能撞到 SQL 保留字）
    cols = ",\n  ".join(f'"{name}" {sql_type}' for name, sql_type in columns)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n  {cols}{extra}\n)"


class DuckDBStore:
    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(_ddl("transactions", TRANSACTION_COLUMNS, ',\n  PRIMARY KEY ("raw_file", "row_index")'))
        self.con.execute(_ddl("fetch_log", _FETCH_LOG_COLUMNS))

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def replace_files(
        self,
        rows_by_file: Mapping[str, Iterable[Mapping[str, Any]]],
        loaded_at: datetime,
        prune_others: bool = False,
    ) -> int:
        """每個原始檔：先刪掉它舊的資料，再插入新的。全部在同一個 transaction 裡，失敗就整批還原。

        prune_others=True：rows_by_file 裡沒有的原始檔，資料也一併刪掉
        （原始檔被刪了，資料庫也要跟著消失；用在「整個 data/raw 重新載入」的時候）。
        """
        names = [name for name, _ in TRANSACTION_COLUMNS]
        placeholders = ", ".join("?" for _ in names)
        column_list = ", ".join(f'"{n}"' for n in names)
        insert = f"INSERT INTO transactions ({column_list}) VALUES ({placeholders})"
        inserted = 0
        self.con.begin()
        try:
            if prune_others:
                keep = list(rows_by_file)
                if keep:
                    self.con.execute("DELETE FROM transactions WHERE NOT list_contains(?, raw_file)", [keep])
                else:
                    self.con.execute("DELETE FROM transactions")
            for raw_file, rows in rows_by_file.items():
                self.con.execute("DELETE FROM transactions WHERE raw_file = ?", [raw_file])
                values = []
                for row in rows:
                    # 刪除用的是 key，寫入用的是 row 裡的 raw_file，兩者不一致就會刪不乾淨
                    if row["raw_file"] != raw_file:
                        raise ValueError(f"raw_file 不一致：{row['raw_file']!r} 放在 {raw_file!r} 底下")
                    values.append([row[n] for n in names[:-1]] + [loaded_at])
                if values:
                    self.con.executemany(insert, values)
                inserted += len(values)
            self.con.commit()
        except BaseException:
            self.con.rollback()
            raise
        return inserted

    def replace_fetch_log(self, log_path: Path) -> int:
        """fetch_log.jsonl 是唯一的真實來源，表格每次整個重建。"""
        entries = []
        if log_path.is_file():
            entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        names = [name for name, _ in _FETCH_LOG_COLUMNS]
        self.con.begin()
        try:
            self.con.execute("DELETE FROM fetch_log")
            if entries:
                self.con.executemany(
                    f"INSERT INTO fetch_log VALUES ({', '.join('?' for _ in names)})",
                    [[e[n] for n in names] for e in entries],
                )
            self.con.commit()
        except BaseException:
            self.con.rollback()
            raise
        return len(entries)
