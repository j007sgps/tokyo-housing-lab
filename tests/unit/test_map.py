import json
import re
from datetime import date
from importlib.resources import files

import pytest

from test_municipalities import build_synthetic_db
from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.analysis.municipalities import build_analysis
from tokyo_housing_lab.map.build import BOUNDARY_CREDIT, NOT_ADVICE, build_map_data, render_html
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
    assert d["text"] == {"disclaimer": DISCLAIMER, "boundaryCredit": BOUNDARY_CREDIT, "notAdvice": NOT_ADVICE}
    chuo = d["municipalities"]["13102"]
    assert chuo["name"] == "中央区"
    assert chuo["latest"]["01"]["status"] == "ok" and chuo["latest"]["01"]["n"] == 161
    assert chuo["yearly"]["01"][0][:2] == [2006, 162]
    assert d["municipalities"]["13999"]["latest"]["01"]["status"] == "insufficient"


def test_growth_pairs_match_analysis(tmp_path):
    # 用兩個完整年的合成資料：漲幅組合必須和 analysis.growth 算出來的一模一樣
    from test_municipalities import rows_for

    files_: dict = {}
    for y, base in ((2010, 400_000), (2011, 440_000), (2012, 500_000)):
        for q in (1, 2, 3, 4):
            rows = rows_for("13104", "新宿区", y, q, "01", [base + i * 100 for i in range(20)])
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


def test_insufficient_growth_pairs_are_omitted(analysis):
    d = build_map_data(analysis, GENERATED)
    # 只有一個完整年，沒有任何起訖組合
    assert d["municipalities"]["13102"]["growth"] == {"01": {}, "02": {}}


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
