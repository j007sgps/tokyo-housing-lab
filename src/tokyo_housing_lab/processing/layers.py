"""處理層：圖層（淹水格網、學區）。

- 淹水：每個圖磚的多邊形畫到同一張格網（GRID_LEVEL，約 31m），每格取最大的浸水深等級。
  洪水是每條河川各自的想定區域，同一個地點常有好幾條河川的多邊形重疊，取最大值就是「最壞的情況」
- 格網只留東京都本島的市區町村內（regions 的行政區界）；圖磚會超出都界，不遮掉的話都界外面只有一半的資料
- 學區：同一校在不同圖磚的部分（已切齊圖磚）用學校代碼拼成一筆
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import numpy as np

from tokyo_housing_lab.processing.raster import GRID_LEVEL, Grid, project_polygon
from tokyo_housing_lab.processing.regions import municipality_polygons
from tokyo_housing_lab.sources.reinfolib_hazards import HazardArea, SchoolPiece


@dataclass(frozen=True)
class TokyoGrid:
    grid: Grid
    cities: np.ndarray  # 每格的市區町村（codes 的索引 + 1；0 = 都外）
    codes: list[str]

    @property
    def mask(self) -> np.ndarray:
        return self.cities > 0


@cache
def tokyo_grid() -> TokyoGrid:
    pairs = municipality_polygons()
    codes = sorted({code for code, _ in pairs})
    polygons = [project_polygon(p, GRID_LEVEL) for _, p in pairs]
    grid = Grid.covering(polygons)
    cities = grid.labels(polygons, [codes.index(code) + 1 for code, _ in pairs])
    return TokyoGrid(grid, cities, codes)


def add_hazard_areas(tg: TokyoGrid, areas: list[HazardArea], out: np.ndarray) -> None:
    """一個圖磚的多邊形加進格網（每格取最大等級）。"""
    polygons = [project_polygon(poly, GRID_LEVEL) for a in areas for poly in a.polygons]
    levels = [a.level for a in areas for _ in a.polygons]
    if polygons:
        tg.grid.max_level(polygons, levels, out=out)


def hazard_grid(tg: TokyoGrid, tiles: Iterable[list[HazardArea]]) -> np.ndarray:
    out = np.zeros(tg.grid.shape, dtype=np.uint8)
    for areas in tiles:
        add_hazard_areas(tg, areas, out)
    out[~tg.mask] = 0
    return out


@dataclass
class School:
    code: str
    name: str
    city: str
    owner: str
    address: str
    polygons: list  # 多邊形 → 環 → (N, 2) 經緯度
    outlines: list  # 邊界線 → (N, 2) 經緯度


def merge_schools(pieces: Iterable[SchoolPiece]) -> list[School]:
    """同一校的各個圖磚的部分 → 一筆。校名等欄位每個圖磚都一樣（實測），用第一個。"""
    by_code: dict[str, list[SchoolPiece]] = defaultdict(list)
    for p in pieces:
        by_code[p.code].append(p)
    out = []
    for code, parts in sorted(by_code.items()):
        first = parts[0]
        for p in parts[1:]:
            if (p.name, p.city) != (first.name, first.city):
                raise ValueError(f"學校代碼 {code} 在不同圖磚的校名或市區町村不同：{first.name}／{p.name}")
        out.append(
            School(
                code, first.name, first.city, first.owner, first.address,
                [poly for p in parts for poly in p.polygons], [line for p in parts for line in p.outlines],
            )
        )  # fmt: skip
    return out
