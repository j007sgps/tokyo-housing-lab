import gzip
import json
import math
from datetime import date, datetime
from itertools import pairwise

import httpx
import numpy as np
import pytest

from conftest import FAKE_KEY, FakeClock
from test_municipalities import rows_for
from tokyo_housing_lab.config import ApiKey
from tokyo_housing_lab.processing.stations import (
    MAX_MATCH_METERS,
    assign_points,
    build_stations,
    order_line,
    station_lines,
    to_meters,
)
from tokyo_housing_lab.sources.reinfolib_api import ReinfolibClient
from tokyo_housing_lab.sources.reinfolib_points import (
    PointFormatError,
    StationPoint,
    parse_area,
    parse_building_year,
    parse_period,
    parse_price,
    parse_xkt015,
    parse_xpt001,
)
from tokyo_housing_lab.sources.reinfolib_tiles import TileQuery, fetch_station_tiles, tile_of, tiles_covering
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

# --- 圖磚 ---


def test_tile_of_known_values():
    # z=0 只有一張；東京駅（139.767, 35.681）在 z=11 的 (1819, 806)
    assert tile_of(139.767, 35.681, 0) == (0, 0)
    assert tile_of(139.767, 35.681, 11) == (1819, 806)


def test_tiles_covering_tokyo():
    tiles = tiles_covering()
    xs, ys = {x for x, _ in tiles}, {y for _, y in tiles}
    assert len(tiles) == len(xs) * len(ys)  # 長方形
    assert (1819, 806) in tiles and tile_of(139.414, 35.698) in tiles  # 東京駅、立川
    assert tile_of(138.95, 35.80) in tiles  # 奥多摩


def test_tile_query():
    q = TileQuery("XPT001", 1818, 806, 2024)
    assert q.filename == "xpt001_z11_1818_806_2024.json.gz"
    assert q.params() == {
        "response_format": "geojson", "z": "11", "x": "1818", "y": "806",
        "from": "20241", "to": "20244", "landTypeCode": "07",
    }  # fmt: skip
    s = TileQuery("XKT015", 1818, 806)
    assert s.filename == "xkt015_z11_1818_806.json.gz" and "from" not in s.params()


def test_fetch_station_tiles_skips_and_refreshes(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path.rsplit("/", 1)[1], request.url.params.get("from")))
        assert request.headers["Ocp-Apim-Subscription-Key"] == FAKE_KEY
        if request.url.params.get("from") == "20231":
            return httpx.Response(404)
        return httpx.Response(200, json={"type": "FeatureCollection", "features": []})

    clock = FakeClock()
    client = ReinfolibClient(ApiKey(FAKE_KEY), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep)
    tiles = [(1818, 806)]
    counts = fetch_station_tiles(client, tmp_path, 2022, date(2025, 3, 1), tiles=tiles)
    assert counts == {"saved": 4, "skipped": 0, "no_data": 1}  # XKT015 ＋ 2022、2024、2025；2023 是 404
    saved = tmp_path / "tiles" / "xpt001_z11_1818_806_2022.json.gz"
    assert json.loads(gzip.decompress(saved.read_bytes())) == {"type": "FeatureCollection", "features": []}
    assert not (tmp_path / "tiles" / "xpt001_z11_1818_806_2023.json.gz").exists()
    # 第二次：已存在的跳過，但今年和去年（2025、2024）重抓；404 的那年再問一次
    calls.clear()
    counts = fetch_station_tiles(client, tmp_path, 2022, date(2025, 3, 1), tiles=tiles)
    assert counts == {"saved": 2, "skipped": 2, "no_data": 1}
    assert sorted(c[1] for c in calls) == ["20231", "20241", "20251"]
    assert all(s >= 1.0 for s in clock.sleeps)  # 節流照樣有效


# --- 解析 ---


@pytest.mark.parametrize(
    ("text", "yen"),
    [("4,800万円", 48_000_000), ("1億2,000万円", 120_000_000), ("2億円", 200_000_000), ("４８０万円", 4_800_000), ("", None), ("非公開", None)],
)
def test_parse_price(text, yen):
    assert parse_price(text) == yen


@pytest.mark.parametrize(("text", "area"), [("60㎡", 60.0), ("1,200㎡", 1200.0), ("2,000㎡以上", None), ("", None)])
def test_parse_area(text, area):
    assert parse_area(text) == area


@pytest.mark.parametrize(("text", "year"), [("2016年", 2016), ("２０１６年", 2016), ("戦前", None), ("", None)])
def test_parse_building_year(text, year):
    assert parse_building_year(text) == year


def test_parse_period():
    assert parse_period("2024年第3四半期") == (2024, 3)
    with pytest.raises(PointFormatError):
        parse_period("2024年")


def point_feature(lon, lat, city="13202", district="132020070", period="2024年第1四半期", category="不動産取引価格情報",
                  price="4,800万円", area="60㎡"):  # fmt: skip
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "point_in_time_name_ja": period, "city_code": city, "district_code": district,
            "price_information_category_name_ja": category, "u_transaction_price_total_ja": price, "u_area_ja": area,
        },
    }  # fmt: skip


def segment_feature(group, code, name, operator, line, coords, passengers=None):
    props = {"S12_001g": group, "S12_001c": code, "S12_001_ja": name, "S12_002_ja": operator, "S12_003_ja": line}
    for year, n in (passengers or {}).items():
        props[f"{year}_passenger_count"] = n
    props["2011_passenger_count"] = 0  # 沒資料的年是 0
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": props}


def collection(*features):
    return json.dumps({"type": "FeatureCollection", "features": list(features)}, ensure_ascii=False).encode()


def test_parse_xpt001_keeps_tokyo_only():
    body = collection(
        point_feature(139.41, 35.69),
        point_feature(139.40, 35.60, city="14150", district="141500010"),  # 神奈川
        point_feature(139.41, 35.69, category="成約価格情報", price="1億2,000万円", area="2,000㎡以上"),
    )
    pts = parse_xpt001(body)
    assert len(pts) == 2
    assert pts[0] == StationPoint(139.41, 35.69, "13202", "132020070", 2024, 1, "01", 48_000_000, 60.0)
    assert (pts[1].price_classification, pts[1].trade_price, pts[1].area) == ("02", 120_000_000, None)


def test_parse_xpt001_rejects_unknown_category():
    with pytest.raises(PointFormatError, match="價格種類"):
        parse_xpt001(collection(point_feature(139.41, 35.69, category="なにか")))


def test_parse_xkt015():
    body = collection(
        segment_feature("003634", "c1", "立川", "東日本旅客鉄道", "中央線", [[139.41, 35.69], [139.42, 35.70]], {2023: 300_000}),
    )
    (s,) = parse_xkt015(body)
    assert (s.group, s.code, s.name, s.line) == ("003634", "c1", "立川", "中央線")
    assert (s.lon, s.lat) == pytest.approx((139.415, 35.695))
    assert s.passengers == {2023: 300_000}  # 0 的年不算


# --- 車站 ---


def segments_example():
    return parse_xkt015(
        collection(
            segment_feature("G1", "a", "立川", "JR", "中央線", [[139.4130, 35.6980], [139.4146, 35.6980]], {2023: 300}),
            segment_feature("G1", "b", "立川", "JR", "南武線", [[139.4140, 35.6975], [139.4150, 35.6975]], {2023: 0, 2019: 50}),
            segment_feature("G1", "a", "立川", "JR", "中央線", [[139.4130, 35.6980], [139.4146, 35.6980]], {2023: 300}),  # 隔壁圖磚的重複
            segment_feature("G2", "c", "西国立", "JR", "南武線", [[139.4240, 35.6935], [139.4246, 35.6935]], {2023: 20}),
        )
    )


def test_build_stations_merges_lines_and_dedupes():
    stations = {s.station_id: s for s in build_stations(segments_example())}
    tachikawa = stations["G1"]
    assert tachikawa.lines == (("JR", "中央線"), ("JR", "南武線"))
    assert (tachikawa.passengers, tachikawa.passengers_year) == (300, 2023)  # 重複的那段不重複算
    assert stations["G2"].passengers == 20


def test_assign_points_nearest_within_threshold():
    segs = segments_example()
    near = StationPoint(139.4138, 35.6980, "13202", "d", 2024, 1, "01", 1, 1.0)  # 立川中央線的中心
    nishi = StationPoint(139.4243, 35.6936, "13202", "d", 2024, 1, "01", 1, 1.0)
    far = StationPoint(139.50, 35.75, "13202", "d", 2024, 1, "01", 1, 1.0)  # 幾公里外：對不到
    ids, unknown = assign_points([near, nishi, far, far], segs)
    assert ids[:2] == ["G1", "G2"]
    assert ids[2] == ids[3] == "pt:139.5000,35.7500"
    assert [u.station_id for u in unknown] == ["pt:139.5000,35.7500"] and unknown[0].name == ""
    # 門檻：剛好在 MAX_MATCH_METERS 內外
    step = MAX_MATCH_METERS / 110940  # 緯度方向
    inside = StationPoint(139.4138, 35.6980 + 0.95 * step, "13202", "d", 2024, 1, "01", 1, 1.0)
    outside = StationPoint(139.4138, 35.6980 + 1.05 * step, "13202", "d", 2024, 1, "01", 1, 1.0)
    ids, _ = assign_points([inside, outside], segs[:1])
    assert ids[0] == "G1" and ids[1].startswith("pt:")


def test_order_line_straight_and_shuffled():
    rng = np.random.default_rng(0)
    xs = np.arange(10) * 1000.0 + rng.normal(0, 50, 10)  # 一條彎彎的直線
    xy = np.column_stack([xs, np.sin(xs / 3000) * 800])
    order = rng.permutation(10)
    ids = [f"s{i}" for i in order]
    got = order_line(ids, xy[order])
    expected = [f"s{i}" for i in range(10)]
    assert got in (expected, expected[::-1])


def test_order_line_loop_follows_circle():
    # 環狀線（12 站的圓）：排成一列時，相鄰的站在圓上也相鄰（只在一處斷開）
    angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    xy = np.column_stack([np.cos(angles), np.sin(angles)]) * 5000
    rng = np.random.default_rng(1)
    order = rng.permutation(12)
    got = [int(s[1:]) for s in order_line([f"s{i}" for i in order], xy[order])]
    steps = [(b - a) % 12 for a, b in pairwise(got)]
    assert set(steps) <= {1, 11}  # 每一步都是往同一個方向的鄰居
    assert len(set(steps)) == 1


def test_station_lines_order_per_line():
    rows = station_lines(segments_example())
    by_line = {}
    for operator, line, sid, seq in rows:
        by_line.setdefault((operator, line), []).append((seq, sid))
    assert sorted(by_line[("JR", "南武線")]) in ([(0, "G1"), (1, "G2")], [(0, "G2"), (1, "G1")])
    assert by_line[("JR", "中央線")] == [(0, "G1")]


def test_to_meters_scale():
    a, b = to_meters(np.array([139.0, 139.0]), np.array([35.7, 35.7 + 1 / 110.94]))
    assert np.hypot(*(b - a)) == pytest.approx(1000, rel=1e-3)


# --- 存進 DuckDB、對回 XIT001 ---


def test_replace_stations_matches_transactions(tmp_path):
    with DuckDBStore(tmp_path / "s.duckdb") as store:
        rows = rows_for("13202", "立川市", 2024, 1, "01", [800_000, 900_000, 1_000_000], district_code="132020070")
        rows += rows_for("13202", "立川市", 2024, 1, "01", [700_000, 700_000], district_code="132020080")
        for i, r in enumerate(rows):
            r["row_index"] = i
        store.replace_files({rows[0]["raw_file"]: rows}, datetime(2026, 1, 1))

        def pt(district, unit, station, building_year=None):
            return {"year": 2024, "quarter": 1, "price_classification": "01", "city": "13202",
                    "district_code": district, "trade_price": unit * 50, "area": 50.0,
                    "building_year": building_year, "floor_plan": "", "structure": "", "station_id": station}  # fmt: skip

        stations = [{"station_id": s, "name": s, "lines": ["JR／中央線"], "lon": 139.4, "lat": 35.7,
                     "passengers": None, "passengers_year": None} for s in ("A", "B")]  # fmt: skip
        points = [
            pt("132020070", 800_000, "A"),
            pt("132020070", 900_000, "B"),
            pt("132020070", 1_000_000, "A") | {"trade_price": None},  # 價格格式不明：不對
            pt("132020080", 700_000, "A"),  # 同一個鍵兩筆、最寄駅相同 → 兩筆都對
            pt("132020080", 700_000, "A"),
        ]
        matched = store.replace_stations(stations, [{"operator": "JR", "line": "中央線", "station_id": "A", "seq": 0}], points)
        assert matched == 4
        got = store.con.execute(
            "SELECT t.trade_price, s.station_id FROM transactions t JOIN transaction_station s USING (raw_file, row_index) ORDER BY 1"
        ).fetchall()
        assert got == [(35_000_000, "A"), (35_000_000, "A"), (40_000_000, "A"), (45_000_000, "B")]

        # 同一個鍵、最寄駅不同 → 分不出來，都不對；重跑會整批換掉
        points[4] = pt("132020080", 700_000, "B")
        assert store.replace_stations(stations, [], points) == 2
        assert store.con.execute("SELECT count(*) FROM station_line").fetchone()[0] == 0
        assert store.con.execute("SELECT lines FROM station WHERE station_id = 'A'").fetchone()[0] == ["JR／中央線"]


# 大江戸線（實際的站序與 XKT015 的月台中心）：一段直線接一個圈的「6 字形」
OEDO = [
    ("光が丘", 139.62924, 35.75848), ("練馬春日町", 139.64076, 35.75135), ("豊島園", 139.64921, 35.7416),
    ("練馬", 139.65456, 35.73732), ("新江古田", 139.67056, 35.73247), ("落合南長崎", 139.68351, 35.72335),
    ("中井", 139.68631, 35.71387), ("東中野", 139.68324, 35.70737), ("中野坂上", 139.68297, 35.69787),
    ("西新宿五丁目", 139.68468, 35.68991), ("都庁前", 139.69262, 35.69059), ("新宿西口", 139.69917, 35.69317),
    ("東新宿", 139.70768, 35.69791), ("若松河田", 139.71818, 35.69916), ("牛込柳町", 139.72519, 35.69946),
    ("牛込神楽坂", 139.73605, 35.70092), ("飯田橋", 139.74566, 35.70308), ("春日", 139.75217, 35.7085),
    ("本郷三丁目", 139.7606, 35.70756), ("上野御徒町", 139.77307, 35.70796), ("新御徒町", 139.78191, 35.707),
    ("蔵前", 139.79225, 35.70551), ("両国", 139.79744, 35.69678), ("森下", 139.79821, 35.68799),
    ("清澄白河", 139.79867, 35.68294), ("門前仲町", 139.7951, 35.67276), ("月島", 139.7841, 35.66389),
    ("勝どき", 139.77714, 35.65895), ("築地市場", 139.76674, 35.66497), ("汐留", 139.76134, 35.66499),
    ("大門", 139.75571, 35.65669), ("赤羽橋", 139.74421, 35.65498), ("麻布十番", 139.73612, 35.65648),
    ("六本木", 139.73129, 35.66406), ("青山一丁目", 139.72397, 35.67292), ("国立競技場", 139.71487, 35.67985),
    ("代々木", 139.7015, 35.68378), ("新宿", 139.69936, 35.68763),
]


def test_order_line_six_shape_like_oedo():
    # 只靠插入會排出「都庁前→代々木→六本木」這種跳躍；排出來的總長度不能比實際的站序長太多
    names = [n for n, _, _ in OEDO]
    xy = to_meters(np.array([x for _, x, _ in OEDO]), np.array([y for _, _, y in OEDO]))
    pos = {n: i for i, n in enumerate(names)}

    def length(seq):
        return float(sum(np.hypot(*(xy[pos[b]] - xy[pos[a]])) for a, b in pairwise(seq)))

    order = np.random.default_rng(3).permutation(len(names))
    got = order_line([names[i] for i in order], xy[order])
    assert length(got) <= 1.02 * length(names)
    assert got[0] == "光が丘" or got[-1] == "光が丘"  # 直線那一端是起點



def test_replace_stations_second_key_separates_neighbors(tmp_path):
    # 同季、同町丁目、同價、同面積的兩間（不同大樓），最寄駅不同：只用基本的鍵分不出來，
    # 加上建築年（房型、構造）就分得出來
    with DuckDBStore(tmp_path / "s.duckdb") as store:
        rows = rows_for("13102", "中央区", 2024, 1, "01", [700_000], building_year="2000年", district_code="131020010")
        rows += rows_for("13102", "中央区", 2024, 1, "01", [700_000], building_year="2015年", district_code="131020010")
        rows += rows_for("13102", "中央区", 2024, 1, "01", [800_000], building_year="戦前", district_code="131020010")
        for i, r in enumerate(rows):
            r["row_index"] = i
        store.replace_files({rows[0]["raw_file"]: rows}, datetime(2026, 1, 1))

        def pt(unit, station, building_year):
            return {"year": 2024, "quarter": 1, "price_classification": "01", "city": "13102",
                    "district_code": "131020010", "trade_price": unit * 50, "area": 50.0,
                    "building_year": building_year, "floor_plan": "", "structure": "", "station_id": station}  # fmt: skip

        stations = [{"station_id": s, "name": s, "lines": [], "lon": 139.8, "lat": 35.7,
                     "passengers": None, "passengers_year": None} for s in ("A", "B", "C")]  # fmt: skip
        # 建築年「戦前」兩邊都是 NULL：一樣要對得上
        points = [pt(700_000, "A", 2000), pt(700_000, "B", 2015), pt(800_000, "C", None)]
        assert store.replace_stations(stations, [], points) == 3
        got = store.con.execute(
            "SELECT t.building_year_raw, s.station_id FROM transactions t JOIN transaction_station s USING (raw_file, row_index) ORDER BY 1"
        ).fetchall()
        assert got == [("2000年", "A"), ("2015年", "B"), ("戦前", "C")]
