"""整條管線：抓取 → 清理 → 存入 DuckDB → 查詢。

- test_pipeline_with_fixtures：API 用 fixture 模擬，平常就會跑
- test_pipeline_real_api：真的打 API（需要金鑰與網路），標成 manual：uv run pytest -m manual
"""

import gzip
import json
import statistics
from datetime import date

import httpx
import pytest

from conftest import FAKE_KEY, FIXTURES, FakeClock
from tokyo_housing_lab.config import ApiKey, load_api_key
from tokyo_housing_lab.pipeline import load_raw_dir
from tokyo_housing_lab.sources.raw_store import RawStore, read_raw_bytes
from tokyo_housing_lab.sources.reinfolib_api import ReinfolibClient, fetch_range
from tokyo_housing_lab.sources.reinfolib_records import query_from_filename
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

TACHIKAWA = "13202"


def expected_medians(raw_dir):
    """不經過我們的清理程式，直接從原始 JSON 算：(年, 季, 種類) → (中位數, 筆數)。"""
    result = {}
    for path in sorted(raw_dir.glob("xit001_*")):
        q = query_from_filename(path.name)
        rows = [
            r
            for r in json.loads(read_raw_bytes(path))["data"]
            if r["Type"] == "中古マンション等" and r["MunicipalityCode"] == TACHIKAWA
        ]
        prices = [int(r["TradePrice"]) / float(r["Area"]) for r in rows]
        if prices:
            result[(q.year, q.quarter, q.price_classification)] = (statistics.median(prices), len(prices))
    return result


def run_pipeline(client, raw_dir, db_path, start, today):
    fetch_range(client, RawStore(raw_dir), TACHIKAWA, start, today)
    with DuckDBStore(db_path) as store:
        load_raw_dir(raw_dir, store)
        return store.quarterly_median(TACHIKAWA)


def test_pipeline_with_fixtures(tmp_path):
    samples = {pc: (FIXTURES / f"xit001_13202_2024Q1_p{pc}_sample.json").read_bytes() for pc in ("01", "02")}

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        if (p["year"], p["quarter"]) != ("2024", "1"):
            return httpx.Response(404)  # 2024Q2 還沒公開
        body = samples[p["priceClassification"]]
        return httpx.Response(200, content=gzip.compress(body), headers={"Content-Encoding": "gzip"})

    clock = FakeClock()
    client = ReinfolibClient(ApiKey(FAKE_KEY), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep)
    raw_dir = tmp_path / "raw"
    with client:
        rows = run_pipeline(client, raw_dir, tmp_path / "t.duckdb", (2024, 1), date(2024, 4, 1))

    assert [(r.year, r.quarter, r.price_classification) for r in rows] == [(2024, 1, "01"), (2024, 1, "02")]
    expected = expected_medians(raw_dir)
    for r in rows:
        median, n = expected[(r.year, r.quarter, r.price_classification)]
        assert r.median_unit_price == pytest.approx(median)
        assert (r.n, r.n_excluded) == (n, 0)


@pytest.mark.manual
def test_pipeline_real_api(tmp_path):
    """立川市 2024Q1 一季，真的打 API（4 次請求左右）。"""
    raw_dir = tmp_path / "raw"
    with ReinfolibClient(load_api_key()) as client:
        rows = run_pipeline(client, raw_dir, tmp_path / "t.duckdb", (2024, 1), date(2024, 1, 15))

    assert {r.price_classification for r in rows} == {"01", "02"}
    assert all((r.year, r.quarter) == (2024, 1) for r in rows)
    assert all(r.n > 0 and r.median_unit_price > 0 for r in rows)
    expected = expected_medians(raw_dir)
    for r in rows:
        median, n = expected[(r.year, r.quarter, r.price_classification)]
        assert r.median_unit_price == pytest.approx(median)
        assert r.n + r.n_excluded == n


def test_area_pipeline_to_map(tmp_path):
    """東京都全域（area=13）：mock 抓取 → 清理 → DuckDB → 分析 → 地圖 HTML。"""
    from tokyo_housing_lab.analysis.municipalities import build_analysis
    from tokyo_housing_lab.map.build import render_html

    samples = {pc: (FIXTURES / f"xit001_13202_2024Q1_p{pc}_sample.json").read_bytes() for pc in ("01", "02")}
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        requested.append(dict(p))
        if (p["year"], p["quarter"]) != ("2024", "1"):
            return httpx.Response(404)
        return httpx.Response(200, content=gzip.compress(samples[p["priceClassification"]]), headers={"Content-Encoding": "gzip"})

    clock = FakeClock()
    raw_dir = tmp_path / "raw"
    with ReinfolibClient(ApiKey(FAKE_KEY), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep) as client:
        fetch_range(client, RawStore(raw_dir), "13", (2024, 1), date(2024, 4, 1))
    assert all(r["area"] == "13" and "city" not in r for r in requested)

    with DuckDBStore(tmp_path / "t.duckdb") as store:
        load_raw_dir(raw_dir, store)
        analysis = build_analysis(store)
    html = render_html(analysis, date(2026, 10, 28))
    payload = json.loads(html.split('<script id="map-data" type="application/json">', 1)[1].split("</script>", 1)[0])
    tachikawa = payload["data"]["municipalities"][TACHIKAWA]
    assert tachikawa["name"] == "立川市"
    assert tachikawa["latest"]["01"]["n"] == 3  # 樣本裡 3 筆中古マンション
    assert tachikawa["latest"]["01"]["status"] == "insufficient"  # 遠少於 30 筆
    # 「最新 4 季」是以最新一季往回推的固定期間；這份資料剛好只有其中一季有成交
    assert payload["data"]["priceTypes"]["01"]["latestPeriod"] == "2023Q2〜2024Q1"
