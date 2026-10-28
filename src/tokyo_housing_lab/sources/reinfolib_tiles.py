"""讀取層：圖磚型 API（XYZ 圖磚，GeoJSON）。

- XPT001 不動産価格（取引価格・成約価格）情報のポイント：每筆成交是一個點，點的位置是「最寄駅」
  （實測：立川附近 266 筆只有 6 種座標，各自離某個車站 3〜52m）。XIT001 沒有車站欄位，靠這個知道最寄駅。
- XKT015 駅別乗降客数（國土數值情報 S12）：車站（每條路線一段月台線段）、站名、路線、乘降人數。

z = 11 是 API 允許的最小縮放（z = 10 回 400）。一個 z = 11 圖磚一年約 1.4 萬筆、20MB，所以一年一個請求。
原始回應原樣 gzip 存到 data/raw/tiles/。
"""

import gzip
import logging
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from tokyo_housing_lab.sources.reinfolib_api import API_ROOT, ReinfolibClient

logger = logging.getLogger(__name__)

ZOOM = 11
# 東京都本島（不含伊豆・小笠原）往外多抓約 5km：最寄駅在神奈川、埼玉的東京物件（町田、練馬…）也要抓到
TOKYO_BBOX = (138.89, 35.45, 139.97, 35.95)  # 西、南、東、北
LAND_TYPE_CONDO = "07"  # 中古マンション等
TILE_DIR = "tiles"


def tile_of(lon: float, lat: float, z: int = ZOOM) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tiles_covering(bbox: tuple[float, float, float, float] = TOKYO_BBOX, z: int = ZOOM) -> list[tuple[int, int]]:
    west, south, east, north = bbox
    x0, y0 = tile_of(west, north, z)  # 圖磚的 y 往南增加
    x1, y1 = tile_of(east, south, z)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


@dataclass(frozen=True)
class TileQuery:
    api: Literal["XPT001", "XKT015"]
    x: int
    y: int
    year: int | None = None  # XPT001 才有：一年（第 1〜4 季）
    z: int = ZOOM

    @property
    def label(self) -> str:
        return f"{self.api} z{self.z}/{self.x}/{self.y}" + (f" {self.year}" if self.year else "")

    @property
    def filename(self) -> str:
        # 例：xpt001_z11_1818_806_2024.json.gz、xkt015_z11_1818_806.json.gz
        stem = f"{self.api.lower()}_z{self.z}_{self.x}_{self.y}"
        return f"{stem}_{self.year}.json.gz" if self.year else f"{stem}.json.gz"

    def params(self) -> dict[str, str]:
        p = {"response_format": "geojson", "z": str(self.z), "x": str(self.x), "y": str(self.y)}
        if self.api == "XPT001":
            p |= {"from": f"{self.year}1", "to": f"{self.year}4", "landTypeCode": LAND_TYPE_CONDO}
        return p


def _save(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    os.replace(tmp, path)


def fetch_station_tiles(
    client: ReinfolibClient,
    raw_dir: Path,
    start_year: int,
    today: date,
    tiles: list[tuple[int, int]] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """每個圖磚：車站（XKT015）一次、成交點（XPT001）每年一次。

    已存在的檔案跳過；今年和去年每次都重抓（官方還會補資料）。404 不存檔。
    回傳 {"saved": …, "skipped": …, "no_data": …}。
    """
    root = raw_dir / TILE_DIR
    tiles = tiles if tiles is not None else tiles_covering()
    refresh = {today.year, today.year - 1}
    counts = {"saved": 0, "skipped": 0, "no_data": 0}
    queries = [TileQuery("XKT015", x, y) for x, y in tiles]
    queries += [TileQuery("XPT001", x, y, year) for year in range(start_year, today.year + 1) for x, y in tiles]
    for i, query in enumerate(queries, 1):
        path = root / query.filename
        if path.is_file() and not force and query.year not in refresh:
            counts["skipped"] += 1
            continue
        status, body = client.get(API_ROOT + query.api, query.params(), query.label)
        if body is None:
            counts["no_data"] += 1
        else:
            _save(path, body)
            counts["saved"] += 1
        logger.info("（%d／%d）%s -> %s", i, len(queries), query.label, status)
    return counts
