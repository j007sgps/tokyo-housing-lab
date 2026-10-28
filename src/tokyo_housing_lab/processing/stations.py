"""處理層：車站。

- 車站：XKT015 的月台線段依 S12_001g 合併（新宿的 JR、私鉄、地下鉄是同一站）
- 成交點 → 車站：XPT001 的點就是最寄駅的位置，對到 MAX_MATCH_METERS 以內最近的月台線段
- 路線上的車站順序：沒有現成資料，用位置推算（最小生成樹的最長路徑當主幹，其他站插到最近的位置）
"""

import math
from collections import defaultdict
from itertools import pairwise
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial import cKDTree

from tokyo_housing_lab.sources.reinfolib_points import StationPoint, StationSegment

MAX_MATCH_METERS = 200.0  # 實測成交點離月台中心 3〜52m
UNKNOWN_PREFIX = "pt:"  # 對不到車站的點：用座標當代號（新開的車站可能還沒收錄）
_LAT0 = 35.7  # 東京附近：經緯度 → 公尺的換算


def to_meters(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.column_stack([lon * 111320 * math.cos(math.radians(_LAT0)), lat * 110940])


@dataclass(frozen=True)
class Station:
    station_id: str  # S12_001g；對不到的點是 "pt:經度,緯度"
    name: str
    lines: tuple[tuple[str, str], ...]  # (營運公司, 路線)
    lon: float
    lat: float
    passengers: int | None  # 最近一年各路線合計
    passengers_year: int | None


def unique_segments(segments: list[StationSegment]) -> list[StationSegment]:
    """跨在圖磚邊界上的月台會出現在兩個圖磚裡：依代碼去重。"""
    seen: dict[str, StationSegment] = {}
    for s in segments:
        seen.setdefault(s.code, s)
    return list(seen.values())


def build_stations(segments: list[StationSegment]) -> list[Station]:
    groups: dict[str, list[StationSegment]] = defaultdict(list)
    for s in unique_segments(segments):
        groups[s.group].append(s)
    out = []
    for gid, segs in sorted(groups.items()):
        years = [y for s in segs for y in s.passengers]
        year = max(years) if years else None
        passengers = sum(s.passengers.get(year, 0) for s in segs) if year else None
        out.append(
            Station(
                gid, segs[0].name, tuple(sorted({(s.operator, s.line) for s in segs})),
                float(np.mean([s.lon for s in segs])), float(np.mean([s.lat for s in segs])), passengers, year,
            )
        )  # fmt: skip
    return out


def assign_points(points: list[StationPoint], segments: list[StationSegment]) -> tuple[list[str], list[Station]]:
    """每個成交點 → 車站代號。回傳 (代號, 對不到時新增的「名稱不明」車站)。"""
    segs = unique_segments(segments)
    if not points:
        return [], []
    plon = np.array([p.lon for p in points])
    plat = np.array([p.lat for p in points])
    ids = np.full(len(points), "", dtype=object)
    if segs:
        tree = cKDTree(to_meters(np.array([s.lon for s in segs]), np.array([s.lat for s in segs])))
        dist, idx = tree.query(to_meters(plon, plat))
        ok = dist <= MAX_MATCH_METERS
        ids[ok] = [segs[i].group for i in idx[ok]]
    unknown: dict[str, Station] = {}
    for i in np.nonzero(ids == "")[0].tolist():
        sid = f"{UNKNOWN_PREFIX}{plon[i]:.4f},{plat[i]:.4f}"
        ids[i] = sid
        unknown.setdefault(sid, Station(sid, "", (), float(plon[i]), float(plat[i]), None, None))
    return ids.tolist(), list(unknown.values())


def order_line(ids: list[str], xy: np.ndarray) -> list[str]:
    """一條路線上的車站排成一列：找總長度最短的開放路徑（近似）。

    起點有好幾種（最小生成樹的最長路徑＋插入、從每一站出發的最近鄰串接），各自用 2-opt 和「搬一站」改善，
    取最短的。只做 2-opt 會卡在局部最佳（大江戸線的 6 字形曾排出從六本木開始、多繞 6% 的順序）；
    加上「搬一站」後單一起點大多就夠，多起點在 120 條路線裡只再改善了支線多的鶴見線（3%），全部也只要幾秒。
    環狀線會在某一處斷開，排成一列。
    """
    n = len(ids)
    if n <= 2:
        return list(ids)
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    starts = [_mst_insertion(dist)] + [_nearest_neighbor(dist, k) for k in range(n)]
    best = min((_improve(p, dist) for p in starts), key=lambda p: (_length(p, dist), p))
    return [ids[i] for i in best]


def _length(path: list[int], dist: np.ndarray) -> float:
    return float(sum(dist[a, b] for a, b in pairwise(path)))


def _mst_insertion(dist: np.ndarray) -> list[int]:
    """最小生成樹的最長路徑當主幹，其他站插到使總長度增加最少的位置。"""
    n = len(dist)
    tree = minimum_spanning_tree(csr_matrix(dist))
    tree = tree + tree.T
    a = int(np.argmax(shortest_path(tree, directed=False, indices=0)))
    da, pred = shortest_path(tree, directed=False, indices=a, return_predecessors=True)
    path = [int(np.argmax(da))]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    for k in sorted(set(range(n)) - set(path), key=lambda k: float(np.min(dist[k, path]))):
        path.insert(_best_insertion(path, k, dist)[1], k)
    return path


def _nearest_neighbor(dist: np.ndarray, start: int) -> list[int]:
    path, left = [start], set(range(len(dist))) - {start}
    while left:
        nxt = min(left, key=lambda k: (dist[path[-1], k], k))
        path.append(nxt)
        left.remove(nxt)
    return path


def _best_insertion(path: list[int], k: int, dist: np.ndarray) -> tuple[float, int]:
    """把 k 插進 path：(增加的長度, 位置)。可以插在兩端。"""
    best = (math.inf, 0)
    for i in range(len(path) + 1):
        left = dist[path[i - 1], k] if i > 0 else 0.0
        right = dist[k, path[i]] if i < len(path) else 0.0
        removed = dist[path[i - 1], path[i]] if 0 < i < len(path) else 0.0
        best = min(best, (left + right - removed, i))
    return best


def _improve(path: list[int], dist: np.ndarray) -> list[int]:
    """2-opt（把一段翻過來）＋搬一站，直到總長度不能再縮短。"""
    path = list(path)
    n = len(path)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # 翻轉 path[i..j]：換掉 (i−1, i)、(j, j+1) 兩條邊（開放路徑的兩端沒有邊）
                before = (dist[path[i - 1], path[i]] if i > 0 else 0.0) + (dist[path[j], path[j + 1]] if j < n - 1 else 0.0)
                after = (dist[path[i - 1], path[j]] if i > 0 else 0.0) + (dist[path[i], path[j + 1]] if j < n - 1 else 0.0)
                if after < before - 1e-6:
                    path[i : j + 1] = path[i : j + 1][::-1]
                    improved = True
        for idx in range(n):
            k = path[idx]
            rest = path[:idx] + path[idx + 1 :]
            current = _length(path, dist) - _length(rest, dist)
            gain, pos = _best_insertion(rest, k, dist)
            if gain < current - 1e-6:
                rest.insert(pos, k)
                path = rest
                improved = True
    return path


def station_lines(segments: list[StationSegment]) -> list[tuple[str, str, str, int]]:
    """(營運公司, 路線, 車站代號, 順序)。同一路線的位置用該路線那一段月台的中心。"""
    by_line: dict[tuple[str, str], dict[str, StationSegment]] = defaultdict(dict)
    for s in unique_segments(segments):
        by_line[(s.operator, s.line)].setdefault(s.group, s)
    out = []
    for (operator, line), segs in sorted(by_line.items()):
        ids = sorted(segs)
        xy = to_meters(np.array([segs[i].lon for i in ids]), np.array([segs[i].lat for i in ids]))
        out += [(operator, line, sid, seq) for seq, sid in enumerate(order_line(ids, xy))]
    return out
