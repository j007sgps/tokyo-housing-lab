import gzip
import json
import math
import struct
import zlib

import httpx
import numpy as np
import pytest
from conftest import FAKE_KEY, FakeClock

from tokyo_housing_lab.config import ApiKey
from tokyo_housing_lab.pipeline import load_layer_tiles
from tokyo_housing_lab.processing.layers import hazard_grid, merge_schools, tokyo_grid
from tokyo_housing_lab.processing.png import palette_png
from tokyo_housing_lab.processing.raster import (
    GRID_LEVEL,
    Grid,
    cell_meters,
    clip_ring,
    point_in_polygon,
    polygon_spans,
    project,
    project_polygon,
    rle_decode,
    rle_encode,
    simplify,
    tiles_intersecting,
    unproject,
)
from tokyo_housing_lab.sources.reinfolib_api import ReinfolibClient
from tokyo_housing_lab.sources.reinfolib_hazards import (
    HazardArea,
    HazardFormatError,
    SchoolPiece,
    depth_band,
    depth_text_band,
    parse_flood,
    parse_school,
    parse_surge,
    surge_level,
)
from tokyo_housing_lab.sources.reinfolib_tiles import TileQuery, fetch_hazard_tiles, hazard_queries, tile_of
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

SQUARE = [np.array([[1, 1], [5, 1], [5, 4], [1, 4]], dtype=float)]  # 4 × 3 格
DONUT = [
    np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
    np.array([[3, 3], [7, 3], [7, 7], [3, 7]], dtype=float),  # 洞
]


# --- 座標、柵格化 ---


def test_project_roundtrip_and_tiles():
    x, y = project(139.767, 35.681, 11)
    assert (int(x), int(y)) == tile_of(139.767, 35.681, 11)
    lon, lat = unproject(x, y, 11)
    assert float(lon) == pytest.approx(139.767) and float(lat) == pytest.approx(35.681)
    assert cell_meters(35.68) == pytest.approx(31.0, abs=0.2)  # 縮放 20 的一格


def test_project_polygon_skips_empty_rings():
    rings = project_polygon([[], [[139.8, 35.6], [139.9, 35.6], [139.9, 35.7]], [[139.8, 35.6]]], 14)
    assert len(rings) == 1 and rings[0].shape == (3, 2)


def test_spans_square_and_hole():
    g = Grid(0, 0, 10, 10)
    cov = g.coverage([SQUARE])
    assert cov.sum() == 12 and cov[1:4, 1:5].all()
    cov = g.coverage([DONUT])
    assert cov.sum() == 100 - 16 and not cov[3:7, 3:7].any()


def test_spans_ignore_orientation_and_closing_point():
    closed = [np.vstack([SQUARE[0][::-1], SQUARE[0][-1:]])]  # 反方向、首尾重複
    assert (Grid(0, 0, 10, 10).coverage([closed]) == Grid(0, 0, 10, 10).coverage([SQUARE])).all()


def test_spans_are_vectorized_per_polygon():
    p, row, c0, c1 = polygon_spans([SQUARE, [r + 20 for r in SQUARE]])
    assert set(p.tolist()) == {0, 1} and len(row) == 6
    assert (c1 - c0 == 4).all()


def test_max_level_takes_deepest_overlap_and_accumulates():
    g = Grid(0, 0, 10, 10)
    big = [np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float)]
    out = g.max_level([big, SQUARE], [1, 3])
    assert out[2, 2] == 3 and out[5, 5] == 1 and out[8, 8] == 0
    g.max_level([big], [2], out=out)  # 下一個圖磚：只會往上加
    assert out[2, 2] == 3 and out[5, 5] == 2


def test_grid_clips_polygons_outside():
    g = Grid(2, 2, 4, 4)
    cov = g.coverage([[np.array([[-5, -5], [20, -5], [20, 20], [-5, 20]], dtype=float)]])
    assert cov.shape == (4, 4) and cov.all()


def test_labels():
    g = Grid(0, 0, 10, 10)
    lab = g.labels([SQUARE, [np.array([[6, 6], [9, 6], [9, 9], [6, 9]], dtype=float)]], [7, 9])
    assert lab[2, 2] == 7 and lab[7, 7] == 9 and lab[0, 0] == 0


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for shape in [(1, 1), (5, 7), (30, 40)]:
        g = rng.integers(0, 4, shape).astype(np.uint8)
        runs = rle_encode(g)
        assert (rle_decode(runs, shape) == g).all()
        assert all(v > 0 and b > a for _, a, b, v in runs)
    assert rle_encode(np.zeros((3, 3), dtype=np.uint8)) == []
    assert rle_encode(np.array([[2, 2, 0, 1]], dtype=np.uint8)) == [(0, 0, 2, 2), (0, 3, 4, 1)]


def test_point_in_polygon_with_hole():
    inside = point_in_polygon(np.array([1.0, 5.0, 11.0, 8.5]), np.array([1.0, 5.0, 5.0, 8.5]), DONUT)
    assert inside.tolist() == [True, False, False, True]


def test_clip_ring_lands_exactly_on_bounds():
    ring = np.array([[-1, 0.5], [2, 0.5], [2, 0.7], [-1, 0.7]], dtype=float)
    clipped = clip_ring(ring, 0, 0, 1, 1)
    assert clipped[:, 0].min() == 0.0 and clipped[:, 0].max() == 1.0
    assert len(clip_ring(np.array([[2, 2], [3, 2], [3, 3]], dtype=float), 0, 0, 1, 1)) == 0


def test_simplify_keeps_shape():
    t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)]) * 100
    s = simplify(circle, 0.5)
    assert 10 < len(s) < 100
    # 簡化後的點都在原本的圓上，最遠的偏差不超過容許值
    assert np.abs(np.hypot(s[:, 0], s[:, 1]) - 100).max() < 1e-9
    zigzag = np.array([[0, 0], [1, 0.01], [2, 0], [3, 0.01], [4, 0], [4, 4], [0, 4]], dtype=float)
    assert len(simplify(zigzag, 0.1)) == 4


def test_tiles_intersecting():
    small = [np.array([[3.2, 5.2], [3.4, 5.2], [3.4, 5.4]], dtype=float)]
    assert tiles_intersecting([small], 0) == [(3, 5)]
    wide = [np.array([[0.5, 0.5], [3.5, 0.5], [3.5, 1.5], [0.5, 1.5]], dtype=float)]
    assert tiles_intersecting([wide], 0) == [(x, y) for x in range(4) for y in (0, 1)]
    # 細長的突出（比切分的格子還細）：靠頂點也會抓到
    spike = [np.array([[0.5, 0.5], [0.6, 0.5], [2.55, 0.51]], dtype=float)]
    assert (2, 0) in tiles_intersecting([spike], 0)


def test_palette_png_is_valid():
    img = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    png = palette_png(img, [(0, 0, 0, 0), (255, 0, 0, 200), (0, 0, 255, 255)])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (2, 2)
    idat = png.index(b"IDAT")
    size = struct.unpack(">I", png[idat - 4 : idat])[0]
    raw = zlib.decompress(png[idat + 4 : idat + 4 + size])
    assert raw == b"\x00\x00\x01\x00\x02\x01"
    with pytest.raises(ValueError, match="調色盤"):
        palette_png(np.array([[3]], dtype=np.uint8), [(0, 0, 0, 0)])


# --- 解析 ---


def collection(*features):
    return json.dumps({"type": "FeatureCollection", "features": list(features)}).encode()


def polygon_feature(props, ring, kind="Polygon"):
    coords = [ring] if kind == "Polygon" else [[ring]]
    return {"type": "Feature", "properties": props, "geometry": {"type": kind, "coordinates": coords}}


SMALL = [[139.80, 35.68], [139.81, 35.68], [139.81, 35.69], [139.80, 35.69], [139.80, 35.68]]


def test_parse_flood():
    areas = parse_flood(collection(polygon_feature({"A31a_202": "荒川", "A31a_205": 4}, SMALL)))
    assert areas[0].level == 4 and len(areas[0].polygons) == 1
    assert depth_band("flood", 2) == (0.5, 3.0)
    for bad in (7, "4", None):
        with pytest.raises(HazardFormatError):
            parse_flood(collection(polygon_feature({"A31a_205": bad}, SMALL)))


@pytest.mark.parametrize(
    ("text", "band", "level"),
    [
        ("0.3m未満", (0.0, 0.3), 1),
        ("0.3m以上0.5m未満", (0.3, 0.5), 2),
        ("1m以上3m未満", (1.0, 3.0), 4),
        ("5m以上10m未満", (5.0, 10.0), 6),
        ("20m以上", (20.0, math.inf), 8),
    ],
)
def test_surge_levels(text, band, level):
    assert depth_text_band(text) == band and surge_level(text) == level and depth_band("surge", level) == band


@pytest.mark.parametrize("text", ["", "深い", "0.4m未満", "1m以上2m未満"])
def test_surge_unknown_text_raises(text):
    with pytest.raises(HazardFormatError):
        surge_level(text)


def test_parse_surge():
    areas = parse_surge(collection(polygon_feature({"A49_003": "3m以上5m未満"}, SMALL)))
    assert areas[0].level == 5


def school_props(code="B1", name="第一小学校", city="13108"):
    return {"A27_001": city, "A27_002": "江東区立", "A27_003": code, "A27_004_ja": name, "A27_005": "江東区1-1"}


def test_parse_school_clips_to_tile_and_drops_tile_edges():
    z = 11
    x, y = tile_of(139.80, 35.68, z)
    edge_lon = float(unproject(x + 1, y, z)[0])  # 圖磚的東邊界
    ring = [[edge_lon - 0.01, 35.68], [edge_lon + 0.0004, 35.68], [edge_lon + 0.0004, 35.69], [edge_lon - 0.01, 35.69]]
    body = collection(
        polygon_feature(school_props(), [*ring, ring[0]], "MultiPolygon"),
        polygon_feature(school_props("B2", city="12203"), SMALL),  # 鄰縣
        {"type": "Feature", "properties": school_props("B3"), "geometry": {"type": "LineString", "coordinates": SMALL[:2]}},
    )
    pieces = parse_school(body, "XKT004", x, y, z)
    assert [p.code for p in pieces] == ["B1"]
    lon = pieces[0].polygons[0][0][:, 0]
    assert lon.max() == pytest.approx(edge_lon, abs=1e-9)  # 超出的 0.0004° 切掉了
    # 邊界線：北、西、南三邊（連成一條），東邊（圖磚邊界）不畫
    assert len(pieces[0].outlines) == 1
    line = pieces[0].outlines[0]
    assert len(line) == 4 and not np.isclose(line[:, 0], edge_lon).all()
    with pytest.raises(HazardFormatError, match="缺少欄位"):
        parse_school(collection(polygon_feature({"A27_001": "13108"}, SMALL)), "XKT004", x, y, z)


def test_parse_school_inside_tile_keeps_closed_outline():
    z = 11
    x, y = tile_of(139.805, 35.685, z)
    pieces = parse_school(collection(polygon_feature(school_props(), SMALL)), "XKT004", x, y, z)
    (line,) = pieces[0].outlines
    assert np.allclose(line[0], line[-1]) and len(line) == 5  # 4 個角 ＋ 回到起點


def test_merge_schools():
    def piece(code, name="甲小学校", city="13108"):
        return SchoolPiece(code, name, city, "江東区立", "", [[np.zeros((3, 2))]], [np.zeros((2, 2))])

    merged = merge_schools([piece("B2"), piece("B1"), piece("B2")])
    assert [s.code for s in merged] == ["B1", "B2"]
    assert len(merged[1].polygons) == 2 and len(merged[1].outlines) == 2
    with pytest.raises(ValueError, match="B1"):
        merge_schools([piece("B1"), piece("B1", name="乙小学校")])


# --- 圖磚、載入 ---


def test_hazard_queries_cover_tokyo():
    queries = hazard_queries(["flood", "elementary"])
    flood = {(q.x, q.y) for q in queries if q.api == "XKT026"}
    assert all(q.z == 14 for q in queries if q.api == "XKT026")
    assert tile_of(139.80, 35.68, 14) in flood and tile_of(139.33, 35.66, 14) in flood  # 江東、八王子
    assert tile_of(139.60, 35.30, 14) not in flood  # 橫須賀
    assert TileQuery("XKT026", 14552, 6451, z=14).filename == "xkt026_z14_14552_6451.json.gz"
    assert TileQuery("XKT004", 1818, 806).params() == {"response_format": "geojson", "z": "11", "x": "1818", "y": "806"}


def test_fetch_hazard_tiles_skips_existing(tmp_path, monkeypatch):
    import tokyo_housing_lab.sources.reinfolib_tiles as rt

    monkeypatch.setattr(rt, "hazard_queries", lambda layers: [TileQuery("XKT026", 1, 2, z=14), TileQuery("XKT004", 3, 4)])
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path.rsplit("/", 1)[1])
        assert request.headers["Ocp-Apim-Subscription-Key"] == FAKE_KEY
        return httpx.Response(200, content=collection())

    clock = FakeClock()
    client = ReinfolibClient(ApiKey(FAKE_KEY), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep)
    assert fetch_hazard_tiles(client, tmp_path) == {"saved": 2, "skipped": 0, "no_data": 0}
    assert json.loads(gzip.decompress((tmp_path / "tiles" / "xkt026_z14_1_2.json.gz").read_bytes()))["features"] == []
    assert fetch_hazard_tiles(client, tmp_path) == {"saved": 0, "skipped": 2, "no_data": 0}
    assert fetch_hazard_tiles(client, tmp_path, force=True)["saved"] == 2
    assert calls == ["XKT026", "XKT004"] * 2


def test_hazard_grid_masks_outside_tokyo():
    tg = tokyo_grid()
    assert len(tg.codes) == 53
    koto = [[139.80, 35.67], [139.82, 35.67], [139.82, 35.68], [139.80, 35.68], [139.80, 35.67]]
    sea = [[139.85, 35.55], [139.87, 35.55], [139.87, 35.56], [139.85, 35.56], [139.85, 35.55]]  # 東京灣
    grid = hazard_grid(tg, [[HazardArea(3, [[koto]]), HazardArea(5, [[sea]])]])
    assert set(np.unique(grid).tolist()) == {0, 3}
    w = cell_meters(35.675)
    area = (grid == 3).sum() * w * w / 1e6  # 約 1.8km × 1.1km
    assert area == pytest.approx(0.02 * 111.32 * math.cos(math.radians(35.675)) * 0.01 * 110.95, rel=0.05)


def write_tile(root, name, body):
    path = root / "tiles" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body))


def test_load_layer_tiles(tmp_path):
    koto = [[139.80, 35.67], [139.82, 35.67], [139.82, 35.68], [139.80, 35.68], [139.80, 35.67]]
    fx, fy = tile_of(139.81, 35.675, 14)
    write_tile(tmp_path, f"xkt026_z14_{fx}_{fy}.json.gz", collection(polygon_feature({"A31a_205": 2}, koto)))
    write_tile(tmp_path, f"xkt026_z14_{fx + 1}_{fy}.json.gz", collection(polygon_feature({"A31a_205": 3}, koto)))
    sx, sy = tile_of(139.805, 35.685, 11)
    write_tile(tmp_path, f"xkt004_z11_{sx}_{sy}.json.gz", collection(polygon_feature(school_props(), SMALL)))
    with DuckDBStore(tmp_path / "t.duckdb") as store:
        summary = load_layer_tiles(tmp_path, store)
        assert summary.tiles == {"flood": 2, "elementary": 1}
        assert summary.schools == {"elementary": 1} and summary.cells["flood"] > 1000
        levels = store.con.execute("SELECT DISTINCT kind, level FROM hazard_grid").fetchall()
        assert levels == [("flood", 3)]  # 重疊的地方取最大的等級
        name, polygons, outlines = store.con.execute("SELECT name, polygons, outlines FROM school_district").fetchone()
        ring = np.array(json.loads(polygons)[0][0])
        assert name == "第一小学校" and ring.min(axis=0) == pytest.approx([139.80, 35.68])
        assert len(json.loads(outlines)) == 1
        # 再載入一次：只換掉這次有的圖層
        (tmp_path / "tiles" / f"xkt004_z11_{sx}_{sy}.json.gz").unlink()
        load_layer_tiles(tmp_path, store)
        assert store.con.execute("SELECT count(*) FROM school_district").fetchone()[0] == 1
        assert store.con.execute("SELECT DISTINCT y >= 0 FROM hazard_grid").fetchall() == [(True,)]
    assert GRID_LEVEL == 20
