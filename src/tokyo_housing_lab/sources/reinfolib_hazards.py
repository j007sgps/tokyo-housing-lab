"""讀取層：解析圖層 API（洪水、高潮、學區）的 GeoJSON 圖磚。

- XKT026 洪水浸水想定区域（想定最大規模）：A31a_205 浸水深ランク（整數代碼）→ FLOOD_DEPTHS
- XKT027 高潮浸水想定区域：A49_003 浸水深是文字（「0.3m以上0.5m未満」）→ SURGE_DEPTHS 的順序當等級
- XKT004 小学校区（A27_*）、XKT005 中学校区（A32_*）：一校一個 MultiPolygon

圖磚型 API 的多邊形會超出圖磚約 40m（相鄰圖磚互相重疊）。淹水是取最大值，重疊沒差；
學區要拼回一校一筆，所以切齊圖磚的邊界（clip_ring），再記下哪些邊是圖磚的邊界（畫線時不畫）。
"""

import json
import math
import re
from dataclasses import dataclass

import numpy as np

from tokyo_housing_lab.processing.raster import (
    clip_ring,
    geojson_polygons,
    project_polygon,
    unproject,
)


class HazardFormatError(ValueError):
    """欄位的格式和預期不同（不猜，直接報錯）。"""


# 洪水的浸水深ランク → (下限, 上限) 公尺。官方的代碼表：国土数値情報 A31a「浸水深ランクコード」
# （https://nlftp.mlit.go.jp/ksj/gml/codelist/water_depth_code.xlsx）：1 = 0〜0.5m未満（床下）、2 = 0.5〜3.0m未満
# （床上〜1 樓）、3 = 3.0〜5.0m未満（2 樓部分）、4 = 5.0〜10.0m未満（2 樓淹沒）、5 = 10.0〜20.0m未満、6 = 20.0m以上
FLOOD_DEPTHS: dict[int, tuple[float, float]] = {
    1: (0.0, 0.5),
    2: (0.5, 3.0),
    3: (3.0, 5.0),
    4: (5.0, 10.0),
    5: (10.0, 20.0),
    6: (20.0, math.inf),
}
# 高潮：文字的分級（實測東京灣有 0.3m未満〜5m以上10m未満）。等級 = 順序 + 1
SURGE_DEPTHS: list[tuple[float, float]] = [
    (0.0, 0.3),
    (0.3, 0.5),
    (0.5, 1.0),
    (1.0, 3.0),
    (3.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, math.inf),
]
_DEPTH_TEXT = re.compile(r"(?:(\d+(?:\.\d+)?)m以上)?(?:(\d+(?:\.\d+)?)m未満)?")
SCHOOL_FIELDS = {"XKT004": "A27", "XKT005": "A32"}


@dataclass(frozen=True)
class HazardArea:
    level: int
    polygons: list  # GeoJSON 的多邊形（環的 list）


@dataclass
class SchoolPiece:
    """一校的學區在一個圖磚裡的部分（已切齊圖磚）。座標是經緯度。"""

    code: str
    name: str
    city: str
    owner: str  # 設置者（〇〇区立）
    address: str
    polygons: list[list[np.ndarray]]  # 多邊形 → 環 → (N, 2) 經緯度
    outlines: list[np.ndarray]  # 學區的邊界線（去掉圖磚邊界的部分）


def depth_text_band(text: str) -> tuple[float, float]:
    """「0.3m以上0.5m未満」→ (0.3, 0.5)、「0.3m未満」→ (0, 0.3)、「20m以上」→ (20, inf)。"""
    m = _DEPTH_TEXT.fullmatch(text.strip())
    if not m or not (m.group(1) or m.group(2)):
        raise HazardFormatError(f"浸水深的格式不明：{text!r}")
    return float(m.group(1) or 0), float(m.group(2)) if m.group(2) else math.inf


def surge_level(text: str) -> int:
    band = depth_text_band(text)
    if band not in SURGE_DEPTHS:
        raise HazardFormatError(f"高潮的浸水深不在已知的分級裡：{text!r}")
    return SURGE_DEPTHS.index(band) + 1


def depth_band(kind: str, level: int) -> tuple[float, float]:
    return FLOOD_DEPTHS[level] if kind == "flood" else SURGE_DEPTHS[level - 1]


def _features(body: bytes) -> list[dict]:
    data = json.loads(body)
    if not isinstance(data.get("features"), list):
        raise HazardFormatError("不是 GeoJSON FeatureCollection")
    return data["features"]


def parse_flood(body: bytes) -> list[HazardArea]:
    out = []
    for f in _features(body):
        level = f["properties"].get("A31a_205")
        if not isinstance(level, int) or level not in FLOOD_DEPTHS:
            raise HazardFormatError(f"洪水的浸水深ランク不明：{level!r}")
        out.append(HazardArea(level, geojson_polygons(f["geometry"])))
    return out


def parse_surge(body: bytes) -> list[HazardArea]:
    out = []
    for f in _features(body):
        text = f["properties"].get("A49_003")
        if not isinstance(text, str):
            raise HazardFormatError(f"高潮的浸水深不明：{text!r}")
        out.append(HazardArea(surge_level(text), geojson_polygons(f["geometry"])))
    return out


def _on_tile_edge(a: np.ndarray, b: np.ndarray, x: int, y: int) -> bool:
    return any(a[i] == b[i] == bound for i, bounds in ((0, (x, x + 1)), (1, (y, y + 1))) for bound in bounds)


def _clip_to_tile(polygon: list, x: int, y: int, z: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """回傳 (切好的環（圖磚座標）, 邊界線（圖磚座標，不含圖磚邊界上的邊）)。"""
    rings, lines = [], []
    for ring in project_polygon(polygon, z):
        clipped = clip_ring(ring, x, y, x + 1, y + 1)
        # 去掉連續重複的點（GeoJSON 的環首尾相同、切的時候也可能產生）
        clipped = clipped[np.any(clipped != np.roll(clipped, 1, axis=0), axis=1)] if len(clipped) else clipped
        if len(clipped) < 3:
            continue
        rings.append(clipped)
        nxt = np.roll(clipped, -1, axis=0)
        border = [_on_tile_edge(a, b, x, y) for a, b in zip(clipped, nxt, strict=True)]
        if not any(border):
            lines.append(np.vstack([clipped, clipped[:1]]))
            continue
        start = border.index(True) + 1  # 從圖磚邊界之後開始走，線才不會在起點被切成兩段
        current: list[np.ndarray] = []
        for k in range(start, start + len(clipped)):
            i = k % len(clipped)
            if border[i]:
                if len(current) > 1:
                    lines.append(np.array(current))
                current = []
                continue
            current = current or [clipped[i]]
            current.append(nxt[i])
        if len(current) > 1:
            lines.append(np.array(current))
    return rings, lines


def _to_lonlat(a: np.ndarray, z: int) -> np.ndarray:
    lon, lat = unproject(a[:, 0], a[:, 1], z)
    return np.column_stack([lon, lat])


def parse_school(body: bytes, api: str, x: int, y: int, z: int, city_prefix: str = "13") -> list[SchoolPiece]:
    """只留 city_prefix 開頭的市區町村（圖磚裡也有鄰縣的學區）。"""
    prefix = SCHOOL_FIELDS[api]
    out = []
    for f in _features(body):
        p = f["properties"]
        try:
            code, name, city = (
                p[f"{prefix}_003"],
                p[f"{prefix}_004_ja"],
                p[f"{prefix}_001"],
            )
            owner, address = p.get(f"{prefix}_002") or "", p.get(f"{prefix}_005") or ""
        except KeyError as exc:
            raise HazardFormatError(f"{api} 缺少欄位 {exc}") from exc
        if not (isinstance(code, str) and isinstance(name, str) and isinstance(city, str)):
            raise HazardFormatError(f"{api} 的學校代碼、校名或市區町村不是字串：{code!r} {name!r} {city!r}")
        if not city.startswith(city_prefix):
            continue
        if f["geometry"]["type"] in (
            "LineString",
            "MultiLineString",
            "Point",
            "MultiPoint",
        ):
            continue  # 圖磚切剩的細縫（實測鄰縣有幾筆）：沒有面積
        polygons, outlines = [], []
        for polygon in geojson_polygons(f["geometry"]):
            rings, lines = _clip_to_tile(polygon, x, y, z)
            if rings:
                polygons.append([_to_lonlat(r, z) for r in rings])
            outlines += [_to_lonlat(line, z) for line in lines]
        if polygons:
            out.append(SchoolPiece(code, name, city, owner, address, polygons, outlines))
    return out
