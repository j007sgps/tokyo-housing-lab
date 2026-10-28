"""儲存：DuckDB。

- transactions：清理後的成交資料（處理層 clean_record 的輸出 + loaded_at）
- fetch_log：data/raw/fetch_log.jsonl 的內容（每次打 API 的紀錄，含 404）

重新載入以原始檔為單位「先刪後插」，同一個檔案載入幾次都不會重複。
"""

import json
import logging
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, NamedTuple

import duckdb

logger = logging.getLogger(__name__)


def _parsed(name: str, sql_type: str) -> list[tuple[str, str]]:
    """數值欄位的三欄：原字串、數值、status。"""
    return [(f"{name}_raw", "VARCHAR"), (name, sql_type), (f"{name}_status", "VARCHAR")]


# 欄位順序和 processing.clean.clean_record 的輸出一致（有測試檢查），最後加上 loaded_at
TRANSACTION_COLUMNS: list[tuple[str, str]] = [
    ("source", "VARCHAR"),
    ("city", "VARCHAR"),
    ("region", "VARCHAR"),
    ("year", "INTEGER"),
    ("quarter", "INTEGER"),
    ("price_classification", "VARCHAR"),
    ("raw_file", "VARCHAR"),
    ("row_index", "INTEGER"),
    ("price_category", "VARCHAR"),
    ("property_type", "VARCHAR"),
    ("land_area_type", "VARCHAR"),
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
    ("region", "VARCHAR"),
    ("year", "INTEGER"),
    ("quarter", "INTEGER"),
    ("price_classification", "VARCHAR"),
    ("http_status", "INTEGER"),
    ("result", "VARCHAR"),
    ("file", "VARCHAR"),
]


# 時點修正係數（tokyo-housing-lab adjust 寫入）：某區某季的成交 × factor ≈ target 期間的平均水準
TIME_ADJUSTMENT_COLUMNS = [
    ("region", "VARCHAR"),
    ("city", "VARCHAR"),
    ("year", "INTEGER"),
    ("quarter", "INTEGER"),
    ("target", "VARCHAR"),  # 例：2025Q2〜2026Q1
    ("factor", "DOUBLE"),  # NULL = 無法修正（資料以前的季、全東京也沒有資料）
    ("factor_low", "DOUBLE"),  # 95% 區間
    ("factor_high", "DOUBLE"),
    ("basis", "VARCHAR"),  # city = 該區自己的指數；tokyo = 樣本不足，用全東京的
    ("n", "INTEGER"),  # 該區在那一期（4 季）的筆數
    ("area_elasticity", "DOUBLE"),  # factor 是 55㎡ 的；面積 A 的物件再乘 (A / 55)^area_elasticity
    ("computed_at", "TIMESTAMP"),
]


# 車站（load 時從 data/raw/tiles/ 建）：XKT015 的月台依 S12_001g 合併成一站
STATION_COLUMNS = [
    ("station_id", "VARCHAR"),  # S12_001g；對不到車站的成交點是 "pt:經度,緯度"
    ("name", "VARCHAR"),
    ("lines", "VARCHAR[]"),  # 「營運公司／路線」
    ("lon", "DOUBLE"),
    ("lat", "DOUBLE"),
    ("passengers", "BIGINT"),  # 最近一年各路線的乘降人數合計
    ("passengers_year", "INTEGER"),
]
STATION_LINE_COLUMNS = [
    ("operator", "VARCHAR"),
    ("line", "VARCHAR"),
    ("station_id", "VARCHAR"),
    ("seq", "INTEGER"),  # 路線上的順序（從位置推算，不是官方資料）
]
# XPT001 的成交點（東京都內）：成交的鍵 ＋ 最寄駅
STATION_POINT_COLUMNS = [
    ("year", "INTEGER"),
    ("quarter", "INTEGER"),
    ("price_classification", "VARCHAR"),
    ("city", "VARCHAR"),
    ("district_code", "VARCHAR"),
    ("trade_price", "BIGINT"),
    ("area", "DOUBLE"),
    ("building_year", "INTEGER"),
    ("floor_plan", "VARCHAR"),
    ("structure", "VARCHAR"),
    ("station_id", "VARCHAR"),
]


# 成交點 → XIT001 的成交（INSERT 時跳過已經對上的）。keys 的欄位名是 station_point 的
_MATCH_SQL = """
    INSERT INTO transaction_station
    SELECT t.raw_file, t.row_index, k.station_id
    FROM transactions t
    JOIN (
        SELECT {keys}, any_value(station_id) AS station_id
        FROM station_point
        WHERE trade_price IS NOT NULL AND area IS NOT NULL
        GROUP BY {keys}
        HAVING count(DISTINCT station_id) = 1
    ) k ON {on}
    WHERE NOT EXISTS (
        SELECT 1 FROM transaction_station s WHERE s.raw_file = t.raw_file AND s.row_index = t.row_index
    )
"""
# station_point 的欄位 → transactions 的欄位（同名的不用列）
_TRANSACTION_KEY = {
    "area": "t.area_m2",
    "building_year": "CASE WHEN t.building_year_status = 'ok' THEN t.building_year END",
    "floor_plan": "coalesce(t.floor_plan, '')",
    "structure": "coalesce(t.structure, '')",
}


class QuarterlyMedian(NamedTuple):
    year: int
    quarter: int
    price_classification: str
    median_unit_price: float | None
    n: int
    n_excluded: int  # 單價算不出來（面積是代碼、空值…）而沒算進中位數的筆數


def _ddl(table: str, columns: list[tuple[str, str]], extra: str = "") -> str:
    # 欄位名一律加雙引號（use 之類的名字可能撞到 SQL 保留字）
    cols = ",\n  ".join(f'"{name}" {sql_type}' for name, sql_type in columns)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n  {cols}{extra}\n)"


class DuckDBStore:
    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute("SET enable_progress_bar = false")  # 長查詢的進度條會混進 CLI 的輸出
        self._create("transactions", TRANSACTION_COLUMNS, ',\n  PRIMARY KEY ("raw_file", "row_index")')
        self._create("fetch_log", _FETCH_LOG_COLUMNS)
        self._create("time_adjustment", TIME_ADJUSTMENT_COLUMNS, ',\n  PRIMARY KEY ("city", "year", "quarter")')
        self._create("station", STATION_COLUMNS, ',\n  PRIMARY KEY ("station_id")')
        self._create("station_line", STATION_LINE_COLUMNS)
        self._create("station_point", STATION_POINT_COLUMNS)
        self._create(
            "transaction_station",
            [("raw_file", "VARCHAR"), ("row_index", "INTEGER"), ("station_id", "VARCHAR")],
            ',\n  PRIMARY KEY ("raw_file", "row_index")',
        )

    def _create(self, table: str, columns: list[tuple[str, str]], extra: str = "") -> None:
        """表格不存在就建立；欄位和程式不一致（舊版本建的）就砍掉重建。

        資料庫的內容全部可以由 data/raw 重新 load，所以重建不會遺失任何東西。
        """
        existing = self.con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        if existing and [c for (c,) in existing] != [name for name, _ in columns]:
            logger.warning("%s 的欄位和目前版本不同，重建（請重新執行 load）", table)
            self.con.execute(f"DROP TABLE {table}")
        self.con.execute(_ddl(table, columns, extra))

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def replace_files(
        self,
        rows_by_file: Mapping[str, Iterable[Mapping[str, Any]]] | Iterable[tuple[str, Iterable[Mapping[str, Any]]]],
        loaded_at: datetime,
        prune_others: bool = False,
    ) -> int:
        """每個原始檔：先刪掉它舊的資料，再插入新的。全部在同一個 transaction 裡，失敗就整批還原。

        rows_by_file 可以是 dict，也可以是 (原始檔名, rows) 的串流（generator），
        東京都全域幾十萬筆時不用一次全部放在記憶體裡。
        prune_others=True：這次沒出現的原始檔，資料也一併刪掉
        （原始檔被刪了，資料庫也要跟著消失；用在「整個 data/raw 重新載入」的時候）。

        executemany 實測只有每秒約 300 筆，所以先把 rows 寫成暫存 NDJSON，
        再用 DuckDB 的 read_json 一次 INSERT（欄位型別照 TRANSACTION_COLUMNS 指定）。
        """
        items = rows_by_file.items() if isinstance(rows_by_file, Mapping) else rows_by_file
        names = [name for name, _ in TRANSACTION_COLUMNS]
        stamp = loaded_at.isoformat(sep=" ")
        files: list[str] = []
        inserted = 0
        with tempfile.TemporaryDirectory() as tmp:
            ndjson = Path(tmp) / "rows.ndjson"
            with ndjson.open("w", encoding="utf-8") as out:
                for raw_file, rows in items:
                    files.append(raw_file)
                    for row in rows:
                        # 刪除用的是 key，寫入用的是 row 裡的 raw_file，兩者不一致就會刪不乾淨
                        if row["raw_file"] != raw_file:
                            raise ValueError(f"raw_file 不一致：{row['raw_file']!r} 放在 {raw_file!r} 底下")
                        record = {n: row[n] for n in names[:-1]}
                        record["loaded_at"] = stamp
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        inserted += 1

            columns = "{" + ", ".join(f"'{n}': '{t}'" for n, t in TRANSACTION_COLUMNS) + "}"
            column_list = ", ".join(f'"{n}"' for n in names)
            self.con.begin()
            try:
                if prune_others:
                    self.con.execute("DELETE FROM transactions WHERE NOT list_contains(?, raw_file)", [files])
                self.con.execute("DELETE FROM transactions WHERE list_contains(?, raw_file)", [files])
                if inserted:
                    self.con.execute(
                        f"INSERT INTO transactions ({column_list}) SELECT {column_list} FROM "
                        f"read_json(?, format = 'newline_delimited', columns = {columns})",
                        [str(ndjson)],
                    )
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
                    # M1 時期的紀錄用 city 而不是 region
                    [[e.get(n, e.get("city")) if n == "region" else e[n] for n in names] for e in entries],
                )
            self.con.commit()
        except BaseException:
            self.con.rollback()
            raise
        return len(entries)

    def replace_time_adjustments(self, rows: Iterable[Mapping[str, Any]], computed_at: datetime) -> int:
        """時點修正係數整張表重寫（係數彼此相關，一次全部換掉才一致）。"""
        names = [name for name, _ in TIME_ADJUSTMENT_COLUMNS]
        values = [[computed_at if n == "computed_at" else r[n] for n in names] for r in rows]
        self.con.begin()
        try:
            self.con.execute("DELETE FROM time_adjustment")
            if values:
                self.con.executemany(f"INSERT INTO time_adjustment VALUES ({', '.join('?' for _ in names)})", values)
            self.con.commit()
        except BaseException:
            self.con.rollback()
            raise
        return len(values)

    def replace_stations(
        self,
        stations: Iterable[Mapping[str, Any]],
        lines: Iterable[Mapping[str, Any]],
        points: Iterable[Mapping[str, Any]],
    ) -> int:
        """車站、路線、成交點整批換掉，再重建「成交 → 車站」的對應。回傳對上的成交筆數。

        對應方式：XPT001 的成交點和 XIT001 的成交，兩段比對——
        1. (年, 季, 價格種類, 町丁目, 總價, 面積, 建築年, 房型, 構造)
        2. 第 1 段對不上的，用 (年, 季, 價格種類, 町丁目, 總價, 面積)
        同一個鍵有好幾筆、而且最寄駅不同時不對（分不出哪筆是哪筆）；最寄駅相同就沒差。
        只用第 2 段的鍵時，同季同町丁目同價同面積、最寄駅不同的物件（相鄰車站之間的大樓）有 2% 對不上。
        """
        with tempfile.TemporaryDirectory() as tmp:
            self.con.begin()
            try:
                for table, columns, rows in (
                    ("station", STATION_COLUMNS, stations),
                    ("station_line", STATION_LINE_COLUMNS, lines),
                    ("station_point", STATION_POINT_COLUMNS, points),
                ):
                    self.con.execute(f"DELETE FROM {table}")
                    self._insert_ndjson(table, columns, rows, Path(tmp) / f"{table}.ndjson")
                self.con.execute("DELETE FROM transaction_station")
                base = ["year", "quarter", "price_classification", "district_code", "trade_price", "area"]
                for keys in (base + ["building_year", "floor_plan", "structure"], base):
                    self.con.execute(_MATCH_SQL.format(
                        keys=", ".join(keys),
                        on=" AND ".join(f"{_TRANSACTION_KEY.get(k, 't.' + k)} IS NOT DISTINCT FROM k.{k}" for k in keys),
                    ))  # fmt: skip
                matched = self.con.execute("SELECT count(*) FROM transaction_station").fetchone()[0]
                self.con.commit()
            except BaseException:
                self.con.rollback()
                raise
        return matched

    def _insert_ndjson(
        self, table: str, columns: list[tuple[str, str]], rows: Iterable[Mapping[str, Any]], path: Path
    ) -> None:
        # executemany 太慢（每秒約 300 筆），成交點有幾十萬筆：寫成 NDJSON 再 read_json 一次插入
        names = [n for n, _ in columns]
        count = 0
        with path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps({n: row[n] for n in names}, ensure_ascii=False) + "\n")
                count += 1
        if count:
            types = "{" + ", ".join(f"'{n}': '{t}'" for n, t in columns) + "}"
            cols = ", ".join(f'"{n}"' for n in names)
            self.con.execute(
                f"INSERT INTO {table} ({cols}) SELECT {cols} FROM "
                f"read_json(?, format = 'newline_delimited', columns = {types})",
                [str(path)],
            )

    def query(self, name: str, **params: Any) -> list[dict[str, Any]]:
        """執行 queries/<name>.sql，每一列回傳成 {欄位名: 值}。"""
        sql = files("tokyo_housing_lab.queries").joinpath(f"{name}.sql").read_text(encoding="utf-8")
        cursor = self.con.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def query_columns(self, name: str, **params: Any) -> dict[str, Any]:
        """執行 queries/<name>.sql，回傳 {欄位名: numpy 陣列}（幾十萬列的樣本用；可為 NULL 的數值欄位是 masked array）。"""
        sql = files("tokyo_housing_lab.queries").joinpath(f"{name}.sql").read_text(encoding="utf-8")
        return self.con.execute(sql, params).fetchnumpy()

    def quarterly_median(self, city: str) -> list[QuarterlyMedian]:
        """M1 完成條件的查詢（SQL 在 queries/quarterly_median.sql）。"""
        return [QuarterlyMedian(**row) for row in self.query("quarterly_median", city=city)]
