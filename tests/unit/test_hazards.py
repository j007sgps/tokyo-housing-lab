import gzip

import pytest
from test_layers import collection, polygon_feature, school_props

from tokyo_housing_lab import cli
from tokyo_housing_lab.analysis.hazards import (
    MIN_COVERAGE,
    assign_schools,
    city_hazards,
    load_grids,
    station_hazards,
)
from tokyo_housing_lab.pipeline import load_layer_tiles
from tokyo_housing_lab.sources.reinfolib_tiles import tile_of
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore


def square(lon, lat, half):
    return [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half],
            [lon - half, lat - half]]  # fmt: skip


KAMEIDO = (139.826, 35.697)  # 江東區：周邊 2km 見方都是洪水等級 3（3〜5m）、高潮等級 3（0.5〜1m）
SETAGAYA = (139.65, 35.64)  # 世田谷區：沒有淹水
KAWAGUCHI = (139.72, 35.80)  # 埼玉縣：都外
STATIONS = [
    {"station_id": "K", "name": "亀戸", "lon": KAMEIDO[0], "lat": KAMEIDO[1]},
    {"station_id": "S", "name": "世田谷", "lon": SETAGAYA[0], "lat": SETAGAYA[1]},
    {"station_id": "W", "name": "川口", "lon": KAWAGUCHI[0], "lat": KAWAGUCHI[1]},
]


def write_tile(root, name, body):
    path = root / "tiles" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body))


def build_hazard_db(root):
    """亀戸周邊 2km 見方淹水、亀戸站在水神小學校區；車站：亀戸、世田谷、川口（都外）。"""
    x, y = tile_of(*KAMEIDO, 14)
    big = square(*KAMEIDO, 0.01)
    write_tile(root, f"xkt026_z14_{x}_{y}.json.gz", collection(polygon_feature({"A31a_205": 3}, big)))
    x, y = tile_of(*KAMEIDO, 13)
    write_tile(root, f"xkt027_z13_{x}_{y}.json.gz", collection(polygon_feature({"A49_003": "0.5m以上1m未満"}, big)))
    x, y = tile_of(*KAMEIDO, 11)
    write_tile(root, f"xkt004_z11_{x}_{y}.json.gz",
               collection(polygon_feature(school_props("B1", "水神小学校", "13108"), square(*KAMEIDO, 0.003))))  # fmt: skip
    path = root / "hz.duckdb"
    with DuckDBStore(path) as store:
        load_layer_tiles(root, store)
        for s in STATIONS:
            store.con.execute("INSERT INTO station VALUES (?, ?, [], ?, ?, NULL, NULL)",
                              [s["station_id"], s["name"], s["lon"], s["lat"]])  # fmt: skip
    return path


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_hazard_db(tmp_path_factory.mktemp("hz"))


def test_station_shares_and_thresholds(db):
    with DuckDBStore(db) as store:
        grids = load_grids(store)
    k, s, w = station_hazards(grids, STATIONS)
    assert k.city == "13108" and k.coverage == pytest.approx(1.0, abs=0.02)
    # 洪水等級 3（3〜5m）：≥0.5m、≥3m 都算；高潮等級 3（0.5〜1m）：只有 ≥0.5m 算
    assert k.shares["flood", 0.5] == k.shares["flood", 3.0] == pytest.approx(1.0)
    assert k.shares["surge", 0.5] == pytest.approx(1.0) and k.shares["surge", 3.0] == 0
    assert s.city == "13112" and s.shares["flood", 0.5] == 0
    assert w.city is None and w.coverage < MIN_COVERAGE and w.shares is None


def test_station_radius(db):
    with DuckDBStore(db) as store:
        grids = load_grids(store)
    # 車站在淹水區的東邊界上：半徑內大約一半淹水
    (edge,) = station_hazards(grids, [{"station_id": "E", "name": "E", "lon": KAMEIDO[0] + 0.01, "lat": KAMEIDO[1]}])
    assert edge.shares["flood", 0.5] == pytest.approx(0.5, abs=0.05)
    (small,) = station_hazards(grids, [STATIONS[0] | {"lon": KAMEIDO[0] + 0.0115}], radius_m=100)
    assert small.shares["flood", 0.5] == 0  # 離邊界約 140m：100m 的圓不碰到


def test_city_hazards_and_schools(db):
    with DuckDBStore(db) as store:
        grids = load_grids(store)
        schools = store.query("school_district")
    cities = {c.city: c for c in city_hazards(grids, {"13108": {"elementary": 1}})}
    assert len(cities) == 53
    koto = cities["13108"]
    assert koto.name == "江東区" and 35 < koto.area_km2 < 50
    # 2km 見方大約 3.6km²，一部分在墨田區
    assert 0.03 < koto.shares["flood", 0.5] < 0.1 and koto.shares["surge", 3.0] == 0
    assert koto.schools == {"elementary": 1, "junior": 0} and cities["13112"].shares["flood", 0.5] == 0
    stations = station_hazards(grids, STATIONS)
    assign_schools(stations, schools)
    assert stations[0].schools == {"elementary": ["水神小学校"], "junior": []}
    assert stations[1].schools == {"elementary": [], "junior": []}


def test_cli_hazards(db, capsys):
    assert cli.main(["hazards", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert out.index("江東区") < out.index("世田谷区") and "沒有學區資料" in out
    assert cli.main(["hazards", "--by", "station", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "亀戸" in out and "水神小学校" in out and "川口" not in out
    assert "100%／100%" in out  # 亀戸的洪水
    assert cli.main(["hazards", "--by", "station", "--city", "13112", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "世田谷区的車站" in out and "亀戸" not in out and "（沒有學區資料）" in out


def test_cli_hazards_needs_data(tmp_path, capsys):
    with DuckDBStore(tmp_path / "e.duckdb"):
        pass
    assert cli.main(["hazards", "--db", str(tmp_path / "e.duckdb")]) == 1
    assert "fetch-hazards" in capsys.readouterr().err
