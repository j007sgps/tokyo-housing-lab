"""把各層串起來：原始檔 →（讀取層）SourceRecord →（處理層）篩選、清理 →（儲存）DuckDB。"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tokyo_housing_lab.processing import layers as ly
from tokyo_housing_lab.processing import stations as st
from tokyo_housing_lab.processing.clean import clean_record
from tokyo_housing_lab.processing.filter import is_used_condo
from tokyo_housing_lab.processing.raster import rle_encode
from tokyo_housing_lab.sources.raw_store import FETCH_LOG_NAME, read_raw_bytes
from tokyo_housing_lab.sources.reinfolib_hazards import parse_flood, parse_school, parse_surge
from tokyo_housing_lab.sources.reinfolib_points import parse_xkt015, parse_xpt001
from tokyo_housing_lab.sources.reinfolib_records import query_from_filename, read_raw_file
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_DB_PATH = Path("data/tokyo_housing_lab.duckdb")


@dataclass(frozen=True)
class LoadSummary:
    files: int  # 實際載入的原始檔數
    records_read: int
    rows_loaded: int  # 篩選後（中古マンション等）寫進 transactions 的筆數
    fetch_log_entries: int
    skipped_files: tuple[str, ...] = ()  # 內容已被其他檔案涵蓋而跳過的原始檔


def select_raw_files(raw_dir: Path) -> tuple[list[Path], list[Path]]:
    """決定要載入哪些原始檔，回傳 (要載入的, 跳過的)。

    - 某一季、某種價格已經有都道府縣（area，例如 13）的檔案時，同一季的市區町村（city）檔案
      內容已經包含在裡面，跳過，避免同一筆成交算兩次（M1 的立川市檔案就是這種情況）
    - 同一個查詢同時有 .json 和 .json.gz 時，用 .json.gz
    """
    paths = sorted(p for p in raw_dir.glob("xit001_*") if p.name.endswith((".json", ".json.gz")))
    by_query = {}
    for path in paths:
        q = query_from_filename(path.name)
        if q not in by_query or path.name.endswith(".gz"):
            by_query[q] = path
    areas = {(q.region, q.year, q.quarter, q.price_classification) for q in by_query if len(q.region) == 2}
    selected, skipped = [], [p for p in paths if p not in by_query.values()]
    for q, path in sorted(by_query.items(), key=lambda kv: kv[1].name):
        if len(q.region) == 5 and (q.region[:2], q.year, q.quarter, q.price_classification) in areas:
            skipped.append(path)
        else:
            selected.append(path)
    for path in skipped:
        logger.warning("%s 的內容已包含在其他原始檔裡，跳過", path.name)
    return selected, sorted(skipped)


def load_raw_dir(raw_dir: Path, store: DuckDBStore) -> LoadSummary:
    """data/raw 底下的全部原始檔重新載入。只讀檔案，不打 API。

    任何一個檔案有問題（格式不對、Period 對不上…）就整批不寫入，資料庫維持原狀。
    一次處理一個檔案、邊讀邊寫暫存檔，東京都全域的資料量也不會吃光記憶體。
    """
    paths, skipped = select_raw_files(raw_dir)
    records_read = 0

    def rows_by_file():
        nonlocal records_read
        for path in paths:
            records = read_raw_file(path)
            records_read += len(records)
            rows = [clean_record(r) for r in records if is_used_condo(r)]
            logger.info("%s：%d 筆，中古マンション等 %d 筆", path.name, len(records), len(rows))
            yield path.name, rows

    loaded_at = datetime.now(UTC).replace(tzinfo=None)  # DuckDB TIMESTAMP 存 UTC
    rows_loaded = store.replace_files(rows_by_file(), loaded_at, prune_others=True)
    log_entries = store.replace_fetch_log(raw_dir / FETCH_LOG_NAME)
    return LoadSummary(len(paths), records_read, rows_loaded, log_entries, tuple(p.name for p in skipped))


@dataclass(frozen=True)
class StationLoadSummary:
    tiles: int  # 讀了幾個圖磚檔
    stations: int  # 有名字的車站
    unknown_stations: int  # 對不到車站的點（名稱不明）
    points: int  # 東京都內的成交點
    matched: int  # 對上 XIT001 的成交筆數
    candidates: int  # 同期間（成交點有資料的年）東京都中古マンション的成交筆數


def load_station_tiles(raw_dir: Path, store: DuckDBStore, region: str = "13") -> StationLoadSummary | None:
    """data/raw/tiles/ → 車站、路線順序、成交點、成交 → 車站。沒有這個資料夾就什麼都不做。"""
    from tokyo_housing_lab.sources.reinfolib_tiles import TILE_DIR

    tile_dir = raw_dir / TILE_DIR
    station_files = sorted(tile_dir.glob("xkt015_*.json.gz"))
    point_files = sorted(tile_dir.glob("xpt001_*.json.gz"))
    if not station_files:
        return None
    segments = [seg for path in station_files for seg in parse_xkt015(read_raw_bytes(path))]
    points = []
    for path in point_files:
        found = parse_xpt001(read_raw_bytes(path), region)
        logger.info("%s：東京都內的成交點 %d 筆", path.name, len(found))
        points += found
    stations = st.build_stations(segments)
    ids, unknown = st.assign_points(points, segments)
    station_rows = [
        {
            "station_id": s.station_id, "name": s.name, "lines": [f"{o}／{ln}" for o, ln in s.lines],
            "lon": s.lon, "lat": s.lat, "passengers": s.passengers, "passengers_year": s.passengers_year,
        }
        for s in stations + unknown
    ]  # fmt: skip
    line_rows = [
        {"operator": o, "line": ln, "station_id": sid, "seq": seq} for o, ln, sid, seq in st.station_lines(segments)
    ]
    point_rows = (
        {
            "year": p.year, "quarter": p.quarter, "price_classification": p.price_classification, "city": p.city,
            "district_code": p.district_code, "trade_price": p.trade_price, "area": p.area,
            "building_year": p.building_year, "floor_plan": p.floor_plan, "structure": p.structure, "station_id": sid,
        }
        for p, sid in zip(points, ids, strict=True)
    )  # fmt: skip
    matched = store.replace_stations(station_rows, line_rows, point_rows)
    years = sorted({p.year for p in points})
    candidates = (
        store.query("condo_count", region=region, first_year=years[0], last_year=years[-1])[0]["n"] if years else 0
    )
    return StationLoadSummary(
        len(station_files) + len(point_files), len(stations), len(unknown), len(points), matched, candidates
    )


@dataclass(frozen=True)
class LayerLoadSummary:
    tiles: dict[str, int]  # 圖層 → 圖磚檔數
    cells: dict[str, int]  # 淹水：有浸水深的格子數（東京都本島內）
    schools: dict[str, int]  # 學區：學校數


_TILE_NAME = re.compile(r"_z(\d+)_(\d+)_(\d+)\.json\.gz$")
_LAYER_FILES = {"flood": "xkt026", "surge": "xkt027", "elementary": "xkt004", "junior": "xkt005"}


def _rounded(points: np.ndarray) -> list:
    return np.round(points, 6).tolist()


def load_layer_tiles(raw_dir: Path, store: DuckDBStore) -> LayerLoadSummary | None:
    """data/raw/tiles/ 的圖層 → hazard_grid、school_district。沒有任何圖層的檔案就什麼都不做。"""
    from tokyo_housing_lab.sources.reinfolib_tiles import TILE_DIR

    tile_dir = raw_dir / TILE_DIR
    paths = {kind: sorted(tile_dir.glob(f"{prefix}_*.json.gz")) for kind, prefix in _LAYER_FILES.items()}
    paths = {kind: p for kind, p in paths.items() if p}
    if not paths:
        return None
    tg = ly.tokyo_grid()
    grid_rows, school_rows, cells, schools = [], [], {}, {}
    for kind in ("flood", "surge"):
        if kind not in paths:
            continue
        parse = parse_flood if kind == "flood" else parse_surge
        grid = ly.hazard_grid(tg, (parse(read_raw_bytes(path)) for path in paths[kind]))
        cells[kind] = int((grid > 0).sum())
        grid_rows += [
            {"kind": kind, "y": r + tg.grid.y0, "x0": a + tg.grid.x0, "x1": b + tg.grid.x0, "level": v}
            for r, a, b, v in rle_encode(grid)
        ]
        logger.info("%s：%d 個圖磚 → %d 格", kind, len(paths[kind]), cells[kind])
    for kind, api in (("elementary", "XKT004"), ("junior", "XKT005")):
        if kind not in paths:
            continue
        pieces = []
        for path in paths[kind]:
            z, x, y = map(int, _TILE_NAME.search(path.name).groups())
            pieces += parse_school(read_raw_bytes(path), api, x, y, z)
        merged = ly.merge_schools(pieces)
        schools[kind] = len(merged)
        school_rows += [
            {
                "kind": kind, "code": s.code, "name": s.name, "city": s.city, "owner": s.owner, "address": s.address,
                "polygons": json.dumps([[_rounded(r) for r in poly] for poly in s.polygons], separators=(",", ":")),
                "outlines": json.dumps([_rounded(line) for line in s.outlines], separators=(",", ":")),
            }
            for s in merged
        ]  # fmt: skip
    store.replace_layers(list(paths), grid_rows, school_rows)
    return LayerLoadSummary({k: len(v) for k, v in paths.items()}, cells, schools)
