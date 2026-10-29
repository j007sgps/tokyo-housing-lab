import json
import math
import re
from datetime import date
from importlib.resources import files

import pytest

from test_municipalities import build_synthetic_db
from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.analysis.municipalities import COMBINED, build_analysis
from tokyo_housing_lab.map import build
from tokyo_housing_lab.map.build import BOUNDARY_CREDIT, NOT_ADVICE, STATION_CREDIT, build_map_data, render_html
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

GENERATED = date(2026, 10, 28)
PAYLOAD = re.compile(r'<script id="map-data" type="application/json">(.*?)</script>', re.S)


@pytest.fixture
def analysis(tmp_path):
    with DuckDBStore(build_synthetic_db(tmp_path / "t.duckdb")) as s:
        return build_analysis(s)


def payload_of(html):
    found = PAYLOAD.findall(html)
    assert len(found) == 1
    return json.loads(found[0])


# --- 邊界資料 ---


def boundaries():
    return json.loads(files("tokyo_housing_lab.map").joinpath("tokyo_municipalities.geojson").read_text(encoding="utf-8"))


def test_boundaries_are_tokyo_mainland_only():
    feats = boundaries()["features"]
    codes = [f["properties"]["code"] for f in feats]
    assert len(feats) == 53  # 23 區 + 26 市 + 西多摩郡 4 町村
    assert len(set(codes)) == len(codes)
    assert all(re.fullmatch(r"13\d{3}", c) for c in codes)
    assert all(int(c) < 13361 for c in codes)  # 沒有伊豆・小笠原的離島
    assert {"13101", "13104", "13202", "13308"} <= set(codes)
    assert all(f["geometry"]["type"] in ("Polygon", "MultiPolygon") for f in feats)


# --- 地圖資料 ---


def test_map_data_structure(analysis):
    d = build_map_data(analysis, GENERATED)
    json.dumps(d)  # 可以序列化
    assert d["generated"] == "2026-10-28"
    assert d["priceTypes"]["01"]["completeYears"] == [2006]
    assert d["priceTypes"]["01"]["latestPeriod"] == "2006Q2〜2007Q1"
    assert d["priceTypes"]["02"]["completeYears"] == []
    assert d["thresholds"]["minN"] == 30
    assert d["text"] == {
        "disclaimer": DISCLAIMER, "boundaryCredit": BOUNDARY_CREDIT, "stationCredit": STATION_CREDIT, "notAdvice": NOT_ADVICE,
    }  # fmt: skip
    assert d["stations"] is None  # 沒給車站分析
    chuo = d["municipalities"]["13102"]
    assert chuo["name"] == "中央区"
    assert chuo["latest"]["01"]["status"] == "ok" and chuo["latest"]["01"]["n"] == 161
    assert chuo["yearly"]["01"][0][:2] == [2006, 162]
    assert d["municipalities"]["13999"]["latest"]["01"]["status"] == "insufficient"
    # 綜合：年份和最新 4 季的期間跟著取引；投資模式可以選綜合或取引
    assert list(d["priceTypes"]) == ["combined", "01", "02"]
    assert d["priceTypes"]["combined"] == {"label": "綜合", "completeYears": [2006], "latestPeriod": "2006Q2〜2007Q1"}
    assert d["investPriceTypes"] == ["combined", "01"]
    # 成約沒有完整年 → 無法換算，綜合就是取引
    assert chuo["offset"] == {"status": "insufficient", "years": 0}
    assert chuo["latest"]["combined"]["median"] == chuo["latest"]["01"]["median"]
    assert chuo["yearly"]["combined"] == [[2006, 162, chuo["yearly"]["01"][0][2]]]
    assert set(chuo["invest"]) == {"01", "combined"}


def test_growth_pairs_match_analysis(tmp_path):
    # 用兩個完整年的合成資料：漲幅組合必須和 analysis.growth 算出來的一模一樣
    from test_municipalities import rows_for

    files_: dict = {}
    for y, base in ((2010, 400_000), (2011, 440_000), (2012, 500_000)):
        for q in (1, 2, 3, 4):
            rows = rows_for("13104", "新宿区", y, q, "01", [base + i * 100 for i in range(20)])
            rows += rows_for("13104", "新宿区", y, q, "02", [base * 9 // 10 + i * 90 for i in range(20)])  # 剛好便宜一成
            rows += rows_for("13999", "小村", y, q, "01", [base, base + 1])  # 每年只有 8 筆
            if y == 2011:
                rows += rows_for("13998", "半村", y, q, "01", [base + i for i in range(10)])  # 只有 2011 年夠
            for r in rows:
                same = files_.setdefault(r["raw_file"], [])
                same.append(r | {"row_index": len(same)})
    with DuckDBStore(tmp_path / "g.duckdb") as s:
        from datetime import datetime

        s.replace_files(files_, datetime(2026, 1, 1))
        a = build_analysis(s)
    d = build_map_data(a, GENERATED)
    pairs = d["municipalities"]["13104"]["growth"]["01"]
    assert set(pairs) == {"2010-2011", "2010-2012", "2011-2012"}
    g = a.growth("13104", "01", 2010, 2012)
    assert pairs["2010-2012"] == [round(g.total, 4), round(g.ci_low, 4), round(g.ci_high, 4), int(g.status == "uncertain")]
    # 任何一年樣本不足的組合不放（地圖上就是灰色）
    assert d["municipalities"]["13999"]["growth"]["01"] == {}
    assert d["municipalities"]["13998"]["growth"]["01"] == {}
    # 綜合也一樣和 analysis 一致；3 個重疊年 → 可以換算，落差剛好 1/0.9
    gc = a.growth("13104", COMBINED, 2010, 2012)
    combined = d["municipalities"]["13104"]["growth"]["combined"]
    assert combined["2010-2012"] == [round(gc.total, 4), round(gc.ci_low, 4), round(gc.ci_high, 4), int(gc.status == "uncertain")]
    assert combined["2010-2012"][0] == pytest.approx(pairs["2010-2012"][0], abs=1e-3)
    off = d["municipalities"]["13104"]["offset"]
    assert off["status"] == "ok" and off["years"] == 3 and off["unstable"] is False
    assert off["pct"] == pytest.approx(1 / 0.9 - 1, abs=1e-3)
    assert off["low"] < off["pct"] < off["high"]
    assert [y for y, _ in off["gaps"]] == [2010, 2011, 2012]


def test_insufficient_growth_pairs_are_omitted(analysis):
    d = build_map_data(analysis, GENERATED)
    # 只有一個完整年，沒有任何起訖組合
    assert d["municipalities"]["13102"]["growth"] == {"combined": {}, "01": {}, "02": {}}


# --- HTML ---


def test_standalone_html(analysis):
    html = render_html(analysis, GENERATED)
    assert html.startswith("<!doctype html>")
    assert "<title>東京中古マンション地図</title>" in html
    p = payload_of(html)
    assert p["data"]["municipalities"]["13102"]["name"] == "中央区"
    assert len(p["boundaries"]["features"]) == 53
    assert "/*__MAP_DATA__*/" not in html
    # 免責聲明、出處、非投資建議都在（由資料帶入頁尾）
    assert p["data"]["text"]["disclaimer"] == DISCLAIMER
    assert "不是投資建議" in p["data"]["text"]["notAdvice"]
    assert "国土数値情報" in p["data"]["text"]["boundaryCredit"]
    assert 'id="note-disclaimer"' in html and 'id="note-credit"' in html


def test_fragment_has_no_document_wrapper(analysis):
    html = render_html(analysis, GENERATED, standalone=False)
    assert not html.lstrip().lower().startswith("<!doctype")
    assert "<html" not in html and "<body" not in html
    assert html.startswith("<title>")
    payload_of(html)


def test_script_tag_in_data_cannot_break_out(analysis):
    analysis.municipalities["13102"].name = '</script><script>alert(1)</script>'
    html = render_html(analysis, GENERATED)
    assert "alert(1)</script>" not in html
    assert payload_of(html)["data"]["municipalities"]["13102"]["name"] == '</script><script>alert(1)</script>'


def test_d3_is_pinned(analysis):
    html = render_html(analysis, GENERATED)
    assert re.search(r'src="https://cdnjs\.cloudflare\.com/ajax/libs/d3/\d+\.\d+\.\d+/d3\.min\.js"', html)


def test_hedonic_in_map_data(tmp_path):
    from test_municipalities import build_hedonic_db

    from tokyo_housing_lab.analysis.municipalities import HEDONIC

    with DuckDBStore(build_hedonic_db(tmp_path / "h.duckdb")) as s:
        a = build_analysis(s)
    d = build_map_data(a, GENERATED)
    assert list(d["priceTypes"]) == ["hedonic", "combined", "01", "02"]
    assert d["priceTypes"]["hedonic"]["label"] == "品質調整"
    assert d["priceTypes"]["hedonic"]["completeYears"] == [2010, 2011, 2012, 2013]
    assert d["investPriceTypes"] == ["hedonic", "combined", "01"]
    assert d["hedonic"]["standard"].startswith("55㎡") and d["hedonic"]["r2"] > 0.9
    m = d["municipalities"]["13104"]
    g = a.growth("13104", HEDONIC, 2010, 2013)
    assert m["growth"]["hedonic"]["2010-2013"] == [round(g.total, 4), round(g.ci_low, 4), round(g.ci_high, 4), int(g.status == "uncertain")]
    assert m["latest"]["hedonic"]["status"] == "ok"
    assert [r[0] for r in m["yearly"]["hedonic"]] == [2010, 2011, 2012, 2013]
    assert set(m["invest"]) == {"01", "combined", "hedonic"}


def test_map_without_hedonic(analysis):
    # 估不出品質調整（合成資料面積全是 50㎡）：沒有這個價格種類，其他照常
    d = build_map_data(analysis, GENERATED)
    assert "hedonic" not in d["priceTypes"] and d["hedonic"] is None
    assert d["investPriceTypes"] == ["combined", "01"]


# --- 車站 ---


@pytest.fixture(scope="module")
def station_analyses(tmp_path_factory):
    from test_station_analysis import build_station_db

    from tokyo_housing_lab.analysis.stations import build_station_analysis

    with DuckDBStore(build_station_db(tmp_path_factory.mktemp("st") / "st.duckdb")) as s:
        return build_analysis(s), build_station_analysis(s)


def test_station_map_data(station_analyses):
    a, sa = station_analyses
    st = build_map_data(a, GENERATED, sa)["stations"]
    assert st["years"] == list(range(2016, 2026)) and st["investWindow"] == [2016, 2025]
    assert st["latestPeriod"] == "2023Q2〜2026Q1" and st["target"] == "2025Q2〜2026Q1"
    assert st["lines"] == []  # 路線上有成交的車站不到 LINE_MIN_STATIONS 站：不放進選單
    assert set(st["list"]) == {"A", "B"}  # 只放有成交的車站
    s = st["list"]["A"]
    assert s["name"] == "西新宿" and s["lines"] == ["東京メトロ丸ノ内線"] and s["pax"] == 100000
    assert s["city"] == "13104" and s["total"] == 410 and s["cities"] == {"13104": 410}
    g = sa.growth("A", 2016, 2025)
    assert s["growth"]["2016-2025"] == [round(g.total, 4), round(g.ci_low, 4), round(g.ci_high, 4), int(g.status == "uncertain")]
    assert s["latest"]["median"] == round(sa.stations["A"].latest.median) and s["invest"]["status"] == "ok"
    assert [r[0] for r in s["yearly"]] == st["years"]
    # 所在區的比較：同方法
    assert st["cityLatest"]["13104"]["n"] == 360 and "2016-2025" in st["cityGrowth"]["13104"]
    assert st["cityInvest"]["13104"]["status"] == "ok" and len(st["cityYearly"]["13104"]) == 10


def test_station_lines_in_map_data(station_analyses, monkeypatch):
    a, sa = station_analyses
    monkeypatch.setattr(build, "LINE_MIN_STATIONS", 1)
    st = build_map_data(a, GENERATED, sa)["stations"]
    assert st["lines"] == [{"key": "JR東日本／中央線", "label": "JR東日本中央線", "group": "JR東日本", "ids": ["C", "B"]}]
    # 路線上沒有成交的車站：只有名稱和位置（路線圖、地圖上的路線要用）
    assert st["list"]["C"] == {"name": "新宿", "lines": ["JR東日本中央線"], "lon": 139.7, "lat": 35.69, "pax": 3000000, "paxYear": 2023}


def test_render_html_with_stations(station_analyses):
    a, sa = station_analyses
    p = payload_of(render_html(a, GENERATED, stations=sa))
    assert set(p["data"]["stations"]["list"]) == {"A", "B"}
    assert "駅別乗降客数" in p["data"]["text"]["stationCredit"]
    assert payload_of(render_html(a, GENERATED))["data"]["stations"] is None


# --- 淹水、學區圖層 ---


@pytest.fixture(scope="module")
def hazard_analysis(tmp_path_factory):
    from test_hazards import build_hazard_db

    from tokyo_housing_lab.analysis.hazards import build_hazards

    with DuckDBStore(build_hazard_db(tmp_path_factory.mktemp("hz"))) as s:
        return build_hazards(s)


def png_pixels(png):
    import struct
    import zlib

    w, h = struct.unpack(">II", png[16:24])
    i = png.index(b"IDAT")
    size = struct.unpack(">I", png[i - 4 : i])[0]
    raw = zlib.decompress(png[i + 4 : i + 4 + size])
    return [list(raw[r * (w + 1) + 1 : (r + 1) * (w + 1)]) for r in range(h)]


def test_depth_png_uses_standard_classes():
    import numpy as np

    flood = build.depth_png("flood", np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.uint8))
    assert png_pixels(flood) == [[0, 1, 2, 3, 4, 5, 6]]
    # 高潮的 8 級併進同一套 6 色：0.3m未満、0.3〜0.5m → 0.5m未満；0.5〜1m、1〜3m → 0.5〜3m …
    surge = build.depth_png("surge", np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint8))
    assert png_pixels(surge) == [[1, 1, 2, 2, 3, 4, 5, 6]]


def test_encode_line_roundtrip():
    import numpy as np

    pts = np.array([[139.70001, 35.68002], [139.70001, 35.68002], [139.70105, 35.67990]])
    enc = build.encode_line(pts)
    assert enc == [13970001, 3568002, 104, -12]  # 重複的點去掉
    x = y = 0
    out = []
    for i in range(0, len(enc), 2):
        x, y = x + enc[i], y + enc[i + 1]
        out.append([x / 1e5, y / 1e5])
    assert out == [[139.70001, 35.68002], [139.70105, 35.6799]]


def test_layer_map_data(analysis, hazard_analysis):
    lay = build_map_data(analysis, GENERATED, layers=hazard_analysis)["layers"]
    west, south, east, north = lay["bounds"]
    assert west < 139.0 and east > 139.9 and south < 35.51 and north > 35.89
    assert lay["thresholds"] == [0.5, 3.0] and lay["radius"] == 500
    assert lay["depthClasses"][0]["label"] == "0.5m未満" and len(lay["depthClasses"]) == 6
    assert all(v.startswith("data:image/png;base64,") for v in lay["images"].values())
    assert lay["cities"]["13108"]["flood"][0] > 0 and lay["cities"]["13108"]["schools"] == {"elementary": 1, "junior": 0}
    assert set(lay["stations"]) == {"K", "S"}  # 都外的川口沒有資料
    assert lay["stations"]["K"] == {"flood": [1.0, 1.0], "surge": [1.0, 0.0], "elementary": ["水神小学校"], "junior": []}
    (school,) = lay["schools"]["elementary"]
    assert school["name"] == "水神小学校" and school["city"] == "13108"
    assert len(school["outlines"]) == 1 and len(school["polygons"]) == 1
    assert "小学校区" in lay["credits"]["elementary"] and "洪水浸水想定区域" in lay["credits"]["flood"]
    assert build_map_data(analysis, GENERATED)["layers"] is None


def test_render_html_with_layers(analysis, hazard_analysis):
    html = render_html(analysis, GENERATED, layers=hazard_analysis)
    assert payload_of(html)["data"]["layers"]["schools"]["junior"] == []
    assert 'id="layer-select"' in html


# --- 估價表單 ---


def js_like_log(e, city, district, area, built, plan, plus_s, structure, renovated, seiyaku):
    """照 template.html 的 estimateLog 算（地圖的 JS 也是這個公式）。"""
    c = e["cities"][city]
    age = max(0, e["year"] - built)
    la = math.log(area)
    out = c["now"] + c["districts"][district]
    out += e["area"][0] * la + sum(coef * max(0, la - knot) for knot, coef in e["area"][1])
    out += e["age"][0] * age + sum(coef * max(0, age - knot) for knot, coef in e["age"][1])
    out += e["areaNow"] * math.log(area / e["standardArea"])
    out += e["oldSeismic"] * (built <= e["oldSeismicLastYear"])
    out += e["plans"][plan] + e["plusS"] * plus_s + e["structures"][structure] + e["renovated"] * renovated
    return out + (c["seiyaku"] if seiyaku else 0)


@pytest.fixture(scope="module")
def estimator_and_window(tmp_path_factory):
    from test_estimate import build_estimate_db

    from tokyo_housing_lab.analysis.estimate import build_estimator
    from tokyo_housing_lab.analysis.window import fit_window, load_window_sample

    with DuckDBStore(build_estimate_db(tmp_path_factory.mktemp("est") / "e.duckdb")) as s:
        ws = load_window_sample(s)
    return build_estimator(ws, fit_window(ws)), ws


@pytest.mark.parametrize(
    ("city", "district", "area", "built", "plan", "structure", "renovation"),
    [
        ("13104", "西新宿", 60, 2005, "2LDK", "RC", "未改装"),
        ("13104", "大久保", 22, 1978, "1LDK+S", "RC", "未改装"),
        ("13104", "小町", 95, 2024, "3LDK", "RC", "未改装"),
        ("13202", "富士見町", 70, 1990, "3LDK", "RC", "未改装"),
    ],
)
def test_estimate_data_formula_matches_table(estimator_and_window, city, district, area, built, plan, structure, renovation):
    from tokyo_housing_lab.analysis import hedonic as hd
    from tokyo_housing_lab.analysis.estimate import Subject

    est, _ = estimator_and_window
    e = build.build_estimate_data(est)
    s = Subject(city, district, area, built, plan, structure, renovation)
    group, plus_s = hd.floor_plan_group(plan)
    for pc in ("01", "02"):
        if pc == "02" and e["cities"][city]["seiyaku"] is None:
            continue
        got = js_like_log(e, city, district, area, built, group, plus_s, structure, renovation == "改装済み", pc == "02")
        assert got == pytest.approx(est.table.log_price(s, pc), abs=1e-4)


def test_estimate_data_structure(estimator_and_window, analysis):
    est, _ = estimator_and_window
    e = build_map_data(analysis, GENERATED, estimator=est)["estimate"]
    assert e["period"] == "2025Q1〜2025Q4" and e["year"] == 2025
    assert set(e["cities"]) == {"13104", "13202"} and e["cities"]["13202"]["seiyaku"] is None
    assert set(e["cities"]["13104"]["districts"]) == {"西新宿", "大久保", "小町"}
    assert e["cities"]["13104"]["top"] in ("西新宿", "大久保")  # 成交最多的町名（小町只有幾筆）
    assert e["levels"] == [0.5, 0.8] and len(e["quantiles"]) == len(e["groups"]) == 10
    lo, hi = e["quantiles"][0][1]
    assert lo < 0 < hi
    assert set(e["plans"]) == {"2LDK", "1LDK", "3LDK"} and e["plans"]["2LDK"] == 0
    assert e["structures"] == {"RC": 0.0} and "全部" in e["coverage"]
    assert build_map_data(analysis, GENERATED)["estimate"] is None
