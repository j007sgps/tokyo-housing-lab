import math
from datetime import datetime

import numpy as np
import pytest
from test_municipalities import rows_for

from tokyo_housing_lab.analysis import hedonic as hd
from tokyo_housing_lab.analysis.estimate import (
    COMPARABLE_AREA_RATIO,
    COMPARABLE_YEARS,
    COMPARABLES_MAX,
    Subject,
    build_estimator,
    coef_table,
    comparables,
    estimate,
    group_label,
    interval_group,
)
from tokyo_housing_lab.analysis.window import fit_window, load_window_sample
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

# 町名 → (地點效果, 主要的用途地域, 容積率)；「小町」只有幾筆，會併進該區的「其他」
DISTRICTS = {
    "13104": {
        "西新宿": (0.3, "商業地域", "600"),
        "大久保": (0.0, "第１種住居地域", "300"),
        "小町": (0.1, "商業地域", "400"),
    },
    "13202": {"曙町": (-0.6, "商業地域", "400"), "富士見町": (-0.8, "第１種中高層住居専用地域", "200")},
}
NAMES = {"13104": "新宿区", "13202": "立川市"}
PLANS = {"2LDK": 0.0, "1LDK": 0.05, "3LDK": -0.04}
SIGMA = 0.15
QUARTERS = [(y, q) for y in range(2021, 2026) for q in (1, 2, 3, 4)]


def true_log(city, district, year, quarter, age, area, plan):
    trend = 0.06 * (year - 2021 + (quarter - 1) / 4)
    return 13.2 + DISTRICTS[city][district][0] + trend - 0.012 * age + 0.08 * math.log(area / 55) + PLANS[plan]


def draw(rng, city):
    names = list(DISTRICTS[city])
    p = np.array([0.02 if n == "小町" else 1.0 for n in names])
    d = str(rng.choice(names, p=p / p.sum()))
    return (
        d,
        int(rng.integers(0, 50)),
        int(rng.choice([20, 25, 30, 40, 50, 55, 60, 70, 80, 95])),
        str(rng.choice(list(PLANS))),
    )


def build_estimate_db(path):
    rng = np.random.default_rng(3)
    files: dict[str, list] = {}
    for year, q in QUARTERS:
        for city in DISTRICTS:
            for pc, k in (("01", 90), ("02", 60 if city == "13104" else 0)):
                for _ in range(k):
                    d, age, area, plan = draw(rng, city)
                    log_p = true_log(city, d, year, q, age, area, plan) + rng.normal(0, SIGMA)
                    log_p += math.log(0.93) if pc == "02" else 0.0
                    _, cp, far = DISTRICTS[city][d]
                    # 同一町名裡也有別的用途地域、容積率（實際資料也是）；兩者各自獨立，才不會和町名共線
                    cp = "近隣商業地域" if rng.random() < 0.3 else cp
                    far = str(rng.choice(["300", "500"])) if rng.random() < 0.3 else far
                    for r in rows_for(city, NAMES[city], year, q, pc, [math.exp(log_p)], building_year=f"{year - age}年",
                                      area=area, district_name=d, floor_plan=plan, structure="RC",
                                      city_planning=cp, floor_area_ratio=far):  # fmt: skip
                        same = files.setdefault(r["raw_file"], [])
                        same.append(r | {"row_index": len(same)})
    with DuckDBStore(path) as s:
        s.replace_files(files, datetime(2026, 1, 1))
    return path


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    with DuckDBStore(build_estimate_db(tmp_path_factory.mktemp("est") / "e.duckdb")) as s:
        ws = load_window_sample(s)
    return ws, fit_window(ws)


@pytest.fixture(scope="module")
def estimator(window):
    ws, model = window
    return build_estimator(ws, model)


def predict_like_estimate(model, ws, s: Subject, pc: str) -> float:
    """用 design_matrix／predict：該町名每一筆成交的地點（用途地域、容積率）× 第 0 期的 4 季，換上這間的建物，取平均。"""
    ms = model.sample
    rows = np.nonzero((ms.city == s.city) & (ms.district == s.district))[0]
    n = len(rows) * 4

    def rep(a):
        return np.repeat(a[rows], 4)

    sample = hd.Sample(
        city=rep(ms.city), district=rep(ms.district), year=np.full(n, ws.last_year), quarter=np.full(n, 1),
        price_classification=np.full(n, pc), log_price=np.zeros(n), area=np.full(n, s.area),
        building_year=np.full(n, s.building_year), floor_plan=np.full(n, s.floor_plan), structure=np.full(n, s.structure),
        renovation=np.full(n, s.renovation), city_planning=rep(ms.city_planning), floor_area_ratio=rep(ms.floor_area_ratio),
    )  # fmt: skip
    pred, keep = hd.predict(model, sample, np.zeros(n, dtype=int), np.tile(np.arange(4), len(rows)))
    assert keep.all()
    return float(pred.mean())


@pytest.mark.parametrize(
    "s",
    [
        Subject("13104", "西新宿", 60, 2005),
        Subject("13104", "大久保", 25, 1978, "1LDK"),  # 舊耐震
        Subject("13104", "小町", 80, 2020, "3LDK"),  # 小町名：「其他」的係數＋自己的用途地域
        Subject("13202", "富士見町", 70, 1995),
    ],
)
def test_coef_table_matches_predict(window, s):
    ws, model = window
    table = coef_table(model, ws.last_year)
    assert table.log_price(s, "01") == pytest.approx(predict_like_estimate(model, ws, s, "01"), abs=1e-9)
    if s.city == "13104":
        assert table.log_price(s, "02") == pytest.approx(predict_like_estimate(model, ws, s, "02"), abs=1e-9)


def test_table_recovers_truth(window):
    ws, model = window
    table = coef_table(model, ws.last_year)
    s = Subject("13104", "西新宿", 55, ws.last_year - 17)
    # 第 0 期 = 2025 年 4 季的平均；西新宿是商業地域、容積率 600
    truth = np.mean([true_log("13104", "西新宿", 2025, q, 17, 55, "2LDK") for q in (1, 2, 3, 4)])
    assert table.log_price(s, "01") == pytest.approx(truth, abs=0.05)
    assert table.log_price(s, "02") - table.log_price(s, "01") == pytest.approx(math.log(0.93), abs=0.03)
    assert table.plans[0] == "2LDK" and {"1LDK", "3LDK"} <= set(table.plans)


def test_errors_and_trade_only_city(estimator):
    with pytest.raises(ValueError, match="町名"):
        estimate(estimator, Subject("13104", "歌舞伎町", 60, 2005))
    with pytest.raises(ValueError, match="13999"):
        estimate(estimator, Subject("13999", "某町", 60, 2005))
    e = estimate(estimator, Subject("13202", "曙町", 60, 2005))
    assert set(e.ranges) == {"01"}  # 立川沒有成約


def test_ranges_are_nested_and_total_is_unit_times_area(estimator):
    e = estimate(estimator, Subject("13104", "西新宿", 62.5, 2005))
    for r50, r80 in e.ranges.values():
        assert (r50.level, r80.level) == (0.5, 0.8)
        assert r80.unit_low < r50.unit_low < r50.unit_high < r80.unit_high
        assert r50.total_low == pytest.approx(r50.unit_low * 62.5) and r80.total_high == pytest.approx(
            r80.unit_high * 62.5
        )
    assert e.ranges["02"][0].unit_high < e.ranges["01"][0].unit_high


def test_coverage_on_fresh_draws(estimator, window):
    """同樣的資料產生方式再抽 3000 筆（模型沒看過）：落在區間裡的比例要接近名目值。"""
    ws, _ = window
    rng = np.random.default_rng(99)
    hits = {0.5: [], 0.8: []}
    for _ in range(3000):
        city = str(rng.choice(list(DISTRICTS)))
        d, age, area, plan = draw(rng, city)
        q = int(rng.integers(1, 5))
        actual = math.exp(true_log(city, d, 2025, q, age, area, plan) + rng.normal(0, SIGMA))
        e = estimate(estimator, Subject(city, d, area, ws.last_year - age, plan))
        for r in e.ranges["01"]:
            hits[r.level].append(r.unit_low <= actual <= r.unit_high)
    assert np.mean(hits[0.5]) == pytest.approx(0.5, abs=0.07)
    assert np.mean(hits[0.8]) == pytest.approx(0.8, abs=0.06)


def test_calibration_report(estimator):
    assert estimator.period_label == "2025Q1〜2025Q4"
    assert set(estimator.quantiles) == set(range(10))
    cov = estimator.coverage["全部"]
    assert cov[0.5] == pytest.approx(0.5, abs=0.12) and cov[0.8] == pytest.approx(0.8, abs=0.1)
    assert estimator.coverage_n["全部"] > 50 and "成約" in estimator.coverage  # 第 0 期 960 筆的 10%


def test_interval_group():
    assert interval_group("13104", 2020, 2025) == 0 and interval_group("13202", 2020, 2025) == 1
    assert interval_group("13104", 1975, 2025) == 8 and interval_group("13104", 2026, 2025) == 0
    assert group_label(0) == "23 區・屋齡 0〜10 年" and group_label(9) == "多摩・屋齡 40 年以上"


def test_comparables(estimator):
    s = Subject("13104", "西新宿", 60, 2005)
    comps = comparables(estimator, s)
    assert 0 < len(comps) <= COMPARABLES_MAX
    for c in comps:
        assert c.year * 4 + c.quarter - 1 > estimator.last - 12  # 最近 3 年
        assert abs(math.log(c.area / 60)) <= math.log(COMPARABLE_AREA_RATIO) + 1e-9
        assert abs(c.building_year - 2005) <= COMPARABLE_YEARS
    # 換算成這間 = 這間的預測 × 那筆的殘差：殘差 = 原始 ÷ 那筆自己的預測
    model = estimator.model
    ms = model.sample
    for c in comps:
        i = np.nonzero((ms.city == "13104") & (ms.district == "西新宿") & (ms.year == c.year) & (ms.quarter == c.quarter)
                       & (ms.area == c.area) & (np.isclose(np.exp(ms.log_price), c.unit_price)))[0][0]  # fmt: skip
        resid = model.fit.resid[i]
        assert c.converted == pytest.approx(math.exp(estimator.table.log_price(s, c.price_classification) + resid))
    # 時點修正：資料每年 +6%，越舊的成交修正越多
    old = [c for c in comps if c.year <= 2023]
    for c in old:
        assert c.time_adjusted / c.unit_price == pytest.approx(
            math.exp(0.06 * (2025.375 - c.year - (c.quarter - 1) / 4)), rel=0.05
        )
    assert comparables(estimator, Subject("13104", "西新宿", 200, 2005)) == []  # 面積差太多


def test_comparables_need_model(estimator):
    from dataclasses import replace

    with pytest.raises(ValueError, match="模型"):
        comparables(replace(estimator, model=None), Subject("13104", "西新宿", 60, 2005))
