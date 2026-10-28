"""處理層：多邊形的幾何運算（numpy；不用 shapely，因為環境裝不了新套件）。

座標系統是 Web Mercator 的圖磚座標：縮放 z 時整個世界是 2^z × 2^z，1 單位 = 一個 z 的圖磚。
- 淹水的格網是 GRID_LEVEL = 20 的「圖磚」：一格約 38m × cos(緯度)，東京約 31m。
  地圖本身也是 Mercator，所以格網在畫面上是整齊的矩形
- 多邊形的環（ring）是 (N, 2) 的陣列；多邊形 = 環的 list（第一個是外圈，其餘是洞）
"""

import math
from collections.abc import Iterable, Sequence

import numpy as np

GRID_LEVEL = 20
EARTH_CIRCUMFERENCE_M = 40075016.686

Ring = np.ndarray
Polygon = list[np.ndarray]


def project(lon: np.ndarray | float, lat: np.ndarray | float, z: int) -> tuple[np.ndarray, np.ndarray]:
    """經緯度 → 縮放 z 的圖磚座標（浮點數；取整數部分就是圖磚編號）。"""
    n = 2.0**z
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def unproject(x: np.ndarray | float, y: np.ndarray | float, z: int) -> tuple[np.ndarray, np.ndarray]:
    n = 2.0**z
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lon = x / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def cell_meters(lat: float, z: int = GRID_LEVEL) -> float:
    """縮放 z 的一格在緯度 lat 的邊長（公尺）。"""
    return EARTH_CIRCUMFERENCE_M / 2.0**z * math.cos(math.radians(lat))


def project_polygon(polygon: Iterable[Sequence[Sequence[float]]], z: int) -> Polygon:
    """GeoJSON 的 [[經度, 緯度], …] 環 → 圖磚座標的陣列。不到 3 個點的環沒有面積，略過（實測洪水有空的環 []）。"""
    out = []
    for ring in polygon:
        if len(ring) < 3:
            continue
        a = np.asarray(ring, dtype=float)
        x, y = project(a[:, 0], a[:, 1], z)
        out.append(np.column_stack([x, y]))
    return out


def geojson_polygons(geometry: dict) -> list[list]:
    """Polygon / MultiPolygon → 多邊形的 list（每個是 GeoJSON 的環的 list）。"""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return list(geometry["coordinates"])
    raise ValueError(f"不支援的幾何類型：{geometry['type']!r}")


def polygon_spans(
    polygons: Sequence[Polygon],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """掃描線填色（偶奇規則，所以洞會挖掉）：格子 (列 r, 行 c) 的中心是 (c + 0.5, r + 0.5)。

    回傳 (多邊形編號, 列, 起行, 迄行)，表示該列的 [起行, 迄行) 在多邊形內。全部向量化，不逐列跑迴圈。
    """
    pid, x1, y1, x2, y2 = [], [], [], [], []
    for i, polygon in enumerate(polygons):
        for ring in polygon:
            if len(ring) < 3:
                continue
            a, b = (
                ring,
                np.roll(ring, -1, axis=0),
            )  # 環沒有首尾相同也可以（重複的那條邊長度 0，不影響）
            pid.append(np.full(len(a), i))
            (
                x1.append(a[:, 0]),
                y1.append(a[:, 1]),
                x2.append(b[:, 0]),
                y2.append(b[:, 1]),
            )
    empty = np.zeros(0, dtype=np.int64)
    if not pid:
        return empty, empty, empty, empty
    pid, x1, y1, x2, y2 = (np.concatenate(v) for v in (pid, x1, y1, x2, y2))
    # 邊依 y 排好：ya < yb；水平的邊不跟任何掃描線相交
    swap = y1 > y2
    xa, ya = np.where(swap, x2, x1), np.where(swap, y2, y1)
    xb, yb = np.where(swap, x1, x2), np.where(swap, y1, y2)
    # 掃描線 y = r + 0.5，取 ya <= y < yb（半開區間：頂點不會被算兩次）
    r0 = np.ceil(ya - 0.5).astype(np.int64)
    r1 = np.ceil(yb - 0.5).astype(np.int64)
    count = np.maximum(r1 - r0, 0)
    keep = count > 0
    pid, xa, ya, xb, yb, r0, count = (
        pid[keep],
        xa[keep],
        ya[keep],
        xb[keep],
        yb[keep],
        r0[keep],
        count[keep],
    )
    edge = np.repeat(np.arange(len(count)), count)
    offset = np.arange(len(edge)) - np.repeat(np.cumsum(count) - count, count)
    row = r0[edge] + offset
    yc = row + 0.5
    x = xa[edge] + (yc - ya[edge]) * (xb[edge] - xa[edge]) / (yb[edge] - ya[edge])
    p = pid[edge]
    order = np.lexsort((x, row, p))
    p, row, x = p[order], row[order], x[order]
    # 同一個多邊形、同一列的交點一定是偶數個：兩兩配對
    p, row, xl, xr = p[0::2], row[0::2], x[0::2], x[1::2]
    c0 = np.ceil(xl - 0.5).astype(np.int64)
    c1 = np.ceil(xr - 0.5).astype(np.int64)
    ok = c1 > c0
    return p[ok], row[ok], c0[ok], c1[ok]


class Grid:
    """格網：左上角 (x0, y0)、寬 width、高 height（縮放 level 的格子單位）。"""

    def __init__(self, x0: int, y0: int, width: int, height: int, level: int = GRID_LEVEL) -> None:
        self.x0, self.y0, self.width, self.height, self.level = (
            x0,
            y0,
            width,
            height,
            level,
        )

    @classmethod
    def covering(cls, polygons: Sequence[Polygon], level: int = GRID_LEVEL) -> "Grid":
        pts = np.concatenate([ring for polygon in polygons for ring in polygon])
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
        return cls(int(x0), int(y0), int(x1 - x0), int(y1 - y0), level)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        """(西, 南, 東, 北)。"""
        west, north = unproject(self.x0, self.y0, self.level)
        east, south = unproject(self.x0 + self.width, self.y0 + self.height, self.level)
        return float(west), float(south), float(east), float(north)

    def coverage(self, polygons: Sequence[Polygon]) -> np.ndarray:
        """每格被幾個多邊形蓋到（多邊形的座標要是這個格網的 level）。"""
        diff = np.zeros((self.height, self.width + 1), dtype=np.int32)
        _, row, c0, c1 = polygon_spans([[ring - (self.x0, self.y0) for ring in poly] for poly in polygons])
        row, c0, c1 = self._clip(row, c0, c1)
        np.add.at(diff, (row, c0), 1)
        np.add.at(diff, (row, c1), -1)
        return np.cumsum(diff, axis=1)[:, :-1]

    def max_level(
        self,
        polygons: Sequence[Polygon],
        levels: Sequence[int],
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """每格取蓋到它的多邊形的最大等級（0 = 沒有）。out 可以傳入累加（一個圖磚一個圖磚加進去）。"""
        out = np.zeros(self.shape, dtype=np.uint8) if out is None else out
        levels = np.asarray(levels)
        for level in sorted(set(levels.tolist())):
            selected = [p for p, lv in zip(polygons, levels, strict=True) if lv == level]
            np.maximum(
                out,
                np.where(self.coverage(selected) > 0, level, 0).astype(np.uint8),
                out=out,
            )
        return out

    def labels(self, polygons: Sequence[Polygon], ids: Sequence[int]) -> np.ndarray:
        """每格屬於哪個多邊形（ids；0 = 都不屬於；重疊時取後面的）。"""
        out = np.zeros(self.shape, dtype=np.int32)
        p, row, c0, c1 = polygon_spans([[ring - (self.x0, self.y0) for ring in poly] for poly in polygons])
        keep = (row >= 0) & (row < self.height)
        ids = np.asarray(ids)
        for i, r, a, b in zip(
            p[keep],
            row[keep],
            np.clip(c0[keep], 0, self.width),
            np.clip(c1[keep], 0, self.width),
            strict=True,
        ):
            out[r, a:b] = ids[i]
        return out

    def _clip(self, row: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = (row >= 0) & (row < self.height)
        row, c0, c1 = (
            row[keep],
            np.clip(c0[keep], 0, self.width),
            np.clip(c1[keep], 0, self.width),
        )
        ok = c1 > c0
        return row[ok], c0[ok], c1[ok]


def tiles_intersecting(polygons: Sequence[Polygon], z: int, sub: int = 8) -> list[tuple[int, int]]:
    """和多邊形相交的縮放 z 圖磚（多邊形的座標是縮放 z 的）。

    把每個圖磚切成 sub × sub 格做掃描線，再加上頂點所在的圖磚：比一格還細的突出部分也一定有頂點，不會漏。
    """
    scaled = [[ring * sub for ring in poly] for poly in polygons]
    _, row, c0, c1 = polygon_spans(scaled)
    tiles = set()
    for r, a, b in zip(
        (row // sub).tolist(),
        (c0 // sub).tolist(),
        ((c1 - 1) // sub).tolist(),
        strict=True,
    ):
        tiles.update((x, r) for x in range(a, b + 1))
    for poly in polygons:
        for ring in poly:
            tiles.update(map(tuple, np.floor(ring).astype(int).tolist()))
    return sorted(tiles)


def rle_encode(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """非 0 的連續格子 → (列, 起行, 迄行, 值)，[起行, 迄行)。"""
    h, w = grid.shape
    padded = np.zeros((h, w + 2), dtype=np.int64)
    padded[:, 1:-1] = grid
    rows, cols = np.nonzero(np.diff(padded, axis=1))  # (r, j)：第 j 格和前一格不同；j 可以是 w（列尾）
    same_row = rows[:-1] == rows[1:]
    r, a, b = rows[:-1][same_row], cols[:-1][same_row], cols[1:][same_row]
    inside = a < w
    r, a, b = r[inside], a[inside], b[inside]
    value = grid[r, a]
    nz = value != 0
    return list(
        zip(
            r[nz].tolist(),
            a[nz].tolist(),
            b[nz].tolist(),
            value[nz].tolist(),
            strict=True,
        )
    )


def rle_decode(runs: Iterable[tuple[int, int, int, int]], shape: tuple[int, int], dtype=np.uint8) -> np.ndarray:
    out = np.zeros(shape, dtype=dtype)
    for r, a, b, value in runs:
        out[r, a:b] = value
    return out


def point_in_polygon(x: np.ndarray, y: np.ndarray, polygon: Polygon) -> np.ndarray:
    """偶奇規則（洞在外圈裡面就挖掉）。x、y 是同樣長度的陣列。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    for ring in polygon:
        a, b = ring, np.roll(ring, -1, axis=0)
        ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        cross = (ay[None, :] > y[:, None]) != (by[None, :] > y[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = ax[None, :] + (y[:, None] - ay[None, :]) * (bx - ax)[None, :] / (by - ay)[None, :]
        inside ^= (np.sum(cross & (x[:, None] < xi), axis=1) % 2).astype(bool)
    return inside


def clip_ring(ring: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray:
    """Sutherland–Hodgman：把環切到矩形裡（圖磚型 API 的多邊形會超出圖磚一點，切齊才能拼起來）。"""
    pts = ring
    for axis, bound, keep_low in (
        (0, xmin, False),
        (0, xmax, True),
        (1, ymin, False),
        (1, ymax, True),
    ):
        if not len(pts):
            break
        inside = pts[:, axis] <= bound if keep_low else pts[:, axis] >= bound
        nxt = np.roll(pts, -1, axis=0)
        nxt_inside = np.roll(inside, -1)
        out = []
        for p, q, pin, qin in zip(pts, nxt, inside, nxt_inside, strict=True):
            if pin:
                out.append(p)
            if pin != qin:
                t = (bound - p[axis]) / (q[axis] - p[axis])
                cut = p + t * (q - p)
                cut[axis] = bound  # 剛好落在邊界上（之後要靠它判斷哪些邊是圖磚的邊界）
                out.append(cut)
        pts = np.array(out).reshape(-1, 2)
    return pts


def simplify(ring: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas–Peucker（閉合的環：先從離起點最遠的點切成兩段）。點太少就原樣回傳。"""
    if len(ring) <= 4:
        return ring
    far = int(np.argmax(np.sum((ring - ring[0]) ** 2, axis=1)))
    keep = np.zeros(len(ring), dtype=bool)
    keep[[0, far, len(ring) - 1]] = True
    stack = [(0, far), (far, len(ring) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = ring[j] - ring[i]
        rel = ring[i + 1 : j] - ring[i]
        length = math.hypot(*seg)
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length if length else np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmax(d))
        if d[k] > tolerance:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return ring[keep]
