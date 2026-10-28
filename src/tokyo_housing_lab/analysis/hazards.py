"""分析層：車站周邊與各區的淹水比例、車站所在的學區（M3 後半）。

- 淹水比例 = 範圍內（東京都本島的格子）浸水深達到門檻的格子比例；門檻 THRESHOLDS（0.5m＝床上、3m＝2 樓）
- 車站周邊：車站座標半徑 STATION_RADIUS_M 公尺內的格子。成交只知道最寄駅、不知道確切位置，
  所以這是「車站周邊」的指標，不是每筆物件的
- 車站在都界附近時，圓的一部分在都外（沒有資料）：coverage 是圓內有資料的比例，太低（< MIN_COVERAGE）的不算
- 學區：車站座標落在哪個小學校區、中學校區（資料有重疊時可能不只一個）
"""

import json
import math
from dataclasses import dataclass, field

import numpy as np

from tokyo_housing_lab.processing.layers import TokyoGrid, tokyo_grid
from tokyo_housing_lab.processing.raster import (
    GRID_LEVEL,
    cell_meters,
    point_in_polygon,
    project,
    rle_decode,
    unproject,
)
from tokyo_housing_lab.processing.regions import municipality_geojson
from tokyo_housing_lab.sources.reinfolib_hazards import FLOOD_DEPTHS, SURGE_DEPTHS, depth_band
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

KINDS = ("flood", "surge")
THRESHOLDS = (0.5, 3.0)  # 公尺：0.5m 以上＝床上浸水，3m 以上＝淹到 2 樓
STATION_RADIUS_M = 500.0
MIN_COVERAGE = 0.5
SCHOOL_KINDS = ("elementary", "junior")


@dataclass(frozen=True)
class Shares:
    """kind → 門檻 → 比例。"""

    values: dict[str, dict[float, float]]

    def __getitem__(self, key: tuple[str, float]) -> float:
        kind, threshold = key
        return self.values[kind][threshold]


@dataclass
class StationHazard:
    station_id: str
    name: str
    lon: float
    lat: float
    coverage: float  # 半徑內在東京都本島（有資料）的比例
    shares: Shares | None  # coverage 不足時是 None
    city: str | None = None  # 車站座標所在的市區町村（都外是 None）
    schools: dict[str, list[str]] = field(default_factory=dict)  # elementary / junior → 校名


@dataclass(frozen=True)
class CityHazard:
    city: str
    name: str
    area_km2: float  # 地圖行政區界（簡化版）的面積
    shares: Shares
    schools: dict[str, int]  # elementary / junior → 學區數（0 = 國土數值情報沒有這個市區町村的學區）


@dataclass
class HazardGrids:
    tg: TokyoGrid
    levels: dict[str, np.ndarray]  # kind → 每格的浸水深等級
    _masks: dict[tuple[str, float], np.ndarray] = field(default_factory=dict, repr=False)

    def at_least(self, kind: str, threshold: float) -> np.ndarray:
        """浸水深的下限 ≥ threshold 的格子（算一次就留著：每個車站都要用）。"""
        key = (kind, threshold)
        if key not in self._masks:
            depths = FLOOD_DEPTHS if kind == "flood" else dict(enumerate(SURGE_DEPTHS, 1))
            ok = [level for level in depths if depth_band(kind, level)[0] >= threshold]
            self._masks[key] = np.isin(self.levels[kind], ok)
        return self._masks[key]


def load_grids(store: DuckDBStore) -> HazardGrids:
    tg = tokyo_grid()
    g = tg.grid
    levels = {}
    for kind in KINDS:
        rows = store.con.execute(
            "SELECT y - ?, x0 - ?, x1 - ?, level FROM hazard_grid WHERE kind = ?", [g.y0, g.x0, g.x0, kind]
        ).fetchall()
        if not rows:
            raise ValueError(f"沒有{'洪水' if kind == 'flood' else '高潮'}的資料，請先執行 fetch-hazards 和 load")
        levels[kind] = rle_decode(rows, g.shape)
    return HazardGrids(tg, levels)


def _shares(grids: HazardGrids, where: np.ndarray | tuple) -> Shares:
    n = grids.tg.mask[where].sum()
    return Shares({
        kind: {t: float(grids.at_least(kind, t)[where].sum() / n) if n else math.nan for t in THRESHOLDS}
        for kind in KINDS
    })  # fmt: skip


def city_hazards(grids: HazardGrids, schools: dict[str, dict[str, int]] | None = None) -> list[CityHazard]:
    names = {f["properties"]["code"]: f["properties"]["name"] for f in municipality_geojson()["features"]}
    tg = grids.tg
    out = []
    for i, code in enumerate(tg.codes, 1):
        inside = tg.cities == i
        lat = _row_lat(tg, inside)
        area = float(inside.sum()) * cell_meters(lat) ** 2 / 1e6
        counts = (schools or {}).get(code, {})
        out.append(
            CityHazard(code, names[code], area, _shares(grids, inside), {k: counts.get(k, 0) for k in SCHOOL_KINDS})
        )
    return out


def _row_lat(tg: TokyoGrid, inside: np.ndarray) -> float:
    """區的平均緯度（換算面積用）。"""
    rows = np.nonzero(inside.any(axis=1))[0]
    return float(unproject(0, tg.grid.y0 + rows.mean() + 0.5, GRID_LEVEL)[1])


def station_hazards(
    grids: HazardGrids, stations: list[dict], radius_m: float = STATION_RADIUS_M
) -> list[StationHazard]:
    """stations：station 表的列（station_id、name、lon、lat）。"""
    tg = grids.tg
    g = tg.grid
    out = []
    for s in stations:
        x, y = project(s["lon"], s["lat"], GRID_LEVEL)
        cx, cy = float(x) - g.x0, float(y) - g.y0
        r = radius_m / cell_meters(s["lat"])
        r0, r1 = max(int(cy - r), 0), min(int(cy + r) + 2, g.height)
        c0, c1 = max(int(cx - r), 0), min(int(cx + r) + 2, g.width)
        if r0 >= r1 or c0 >= c1:
            out.append(StationHazard(s["station_id"], s["name"], s["lon"], s["lat"], 0.0, None))
            continue
        here = tg.cities[int(cy), int(cx)] if 0 <= cy < g.height and 0 <= cx < g.width else 0
        city = tg.codes[here - 1] if here else None
        rows, cols = np.mgrid[r0:r1, c0:c1]
        disk = (rows + 0.5 - cy) ** 2 + (cols + 0.5 - cx) ** 2 <= r * r
        total = math.pi * r * r
        where = (rows[disk], cols[disk])
        coverage = float(tg.mask[where].sum() / total)
        shares = _shares(grids, where) if coverage >= MIN_COVERAGE else None
        out.append(StationHazard(s["station_id"], s["name"], s["lon"], s["lat"], min(coverage, 1.0), shares, city))
    return out


def assign_schools(stations: list[StationHazard], schools: list[dict]) -> None:
    """schools：school_district 表的列（kind、name、polygons）。車站座標落在哪個學區。"""
    lon = np.array([s.lon for s in stations])
    lat = np.array([s.lat for s in stations])
    for s in stations:
        s.schools = {k: [] for k in SCHOOL_KINDS}
    for row in schools:
        for poly in json.loads(row["polygons"]):
            rings = [np.asarray(r, dtype=float) for r in poly]
            outer = rings[0]
            near = np.nonzero(
                (lon >= outer[:, 0].min()) & (lon <= outer[:, 0].max())
                & (lat >= outer[:, 1].min()) & (lat <= outer[:, 1].max())
            )[0]  # fmt: skip
            if not len(near):
                continue
            for i in near[point_in_polygon(lon[near], lat[near], rings)]:
                names = stations[i].schools[row["kind"]]
                if row["name"] not in names:
                    names.append(row["name"])


@dataclass(frozen=True)
class HazardAnalysis:
    grids: HazardGrids
    cities: list[CityHazard]
    stations: list[StationHazard]
    schools: list[dict]  # school_district 表的列（地圖畫學區用）


def build_hazards(store: DuckDBStore) -> HazardAnalysis:
    """要先跑過 fetch-hazards、load（車站的部分還要 fetch-stations）。"""
    grids = load_grids(store)
    school_rows = store.query("school_district")
    counts: dict[str, dict[str, int]] = {}
    for r in school_rows:
        counts.setdefault(r["city"], {}).setdefault(r["kind"], 0)
        counts[r["city"]][r["kind"]] += 1
    stations = station_hazards(grids, store.query("station"))
    assign_schools(stations, school_rows)
    return HazardAnalysis(grids, city_hazards(grids, counts), stations, school_rows)
