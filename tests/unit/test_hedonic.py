import math

import numpy as np
import pytest
import scipy.sparse as sp

from tokyo_housing_lab.analysis.hedonic import (
    MIN_DISTRICT_N,
    OTHER_DISTRICT,
    Sample,
    Spec,
    city_planning_group,
    design_matrix,
    far_group,
    fit_hedonic,
    fit_ols,
    floor_plan_group,
    hedonic_growth,
    hedonic_price,
    hedonic_trend,
    index_series,
    predict,
    renovation_group,
    structure_group,
    window_period,
    yearly_period,
)
from tokyo_housing_lab.analysis.trend import MIN_N

# --- 特性的分類 ---


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        ("2LDK", ("2LDK", False)),
        ("2LDK+S", ("2LDK", True)),
        ("1K", ("1R/1K", False)),
        ("1R", ("1R/1K", False)),
        ("1DK", ("1DK", False)),
        ("1LDK+S", ("1LDK", True)),
        ("2K", ("2DK", False)),
        ("3DK", ("3DK", False)),
        ("4LDK", ("4+", False)),
        ("5DK", ("4+", False)),
        ("オープンフロア", ("其他", False)),
        ("", ("不明", False)),
    ],
)
def test_floor_plan_group(plan, expected):
    assert floor_plan_group(plan) == expected


def test_category_groups():
    assert structure_group("RC") is None and structure_group("SRC") == "SRC"
    assert structure_group("鉄骨造") == "其他" and structure_group("") == "不明"
    assert renovation_group("未改装") is None and renovation_group("改装済み") == "改装済み" and renovation_group("") == "不明"
    assert city_planning_group("商業地域") is None and city_planning_group("準工業地域") == "準工業地域"
    assert city_planning_group("工業専用地域") == "其他" and city_planning_group("") == "不明"
    assert far_group(200) is None and far_group(300) == "≤300" and far_group(800) == ">600"
    assert far_group(float("nan")) == "不明"


def test_window_period():
    s = make_sample(np.random.default_rng(0), n_per=1)
    years = np.array([2025, 2025, 2026, 2025, 2024])
    quarters = np.array([1, 2, 1, 4, 2])
    sub = s.subset(np.arange(5))
    sub.year, sub.quarter = years, quarters
    # 最新一季 2026Q1：2025Q2〜2026Q1 = 0、2024Q2〜2025Q1 = −1
    assert window_period(sub, 2026, 1).tolist() == [-1, 0, 0, 0, -1]


# --- 合成資料 ---

CITIES = {"13101": ["甲町", "乙町", "丙町"], "13102": ["丁町", "戊町", "己町"]}
SMALL = {"13101": "小町", "13102": "微町"}  # 筆數少於 MIN_DISTRICT_N，會併進「其他」
YEARS = [2010, 2011, 2012, 2013]
TRUE = {
    "district": {"甲町": 13.0, "乙町": 13.3, "丙町": 12.8, "丁町": 13.6, "戊町": 13.1, "己町": 12.9, "小町": 13.5, "微町": 12.6},
    "time": {"13101": {2010: 0.0, 2011: 0.05, 2012: 0.12, 2013: 0.10}, "13102": {2010: 0.0, 2011: -0.03, 2012: 0.02, 2013: 0.20}},
    "area": (-0.10, 0.05, 0.0, 0.08, 0.0, -0.05, 0.0),  # ln 面積 的斜率、各節點（25/40/55/70/90）之後斜率的變化
    "age": (-0.012, -0.004, 0.0, 0.002, 0.0, 0.003, 0.0),
    "old_seismic": -0.08,
    "plan": {"2LDK": 0.0, "1R/1K": 0.07, "1LDK": 0.02, "3LDK": -0.01, "4+": -0.03},
    "plus_s": -0.02,
    "structure": {"RC": 0.0, "SRC": 0.04, "鉄骨造": -0.06, "": -0.01},
    "renovation": {"未改装": 0.0, "改装済み": 0.05, "": 0.01},
    "city_planning": {"商業地域": 0.0, "準工業地域": -0.04, "第１種低層住居専用地域": 0.03},
    "far": {200: 0.0, 300: 0.02, 500: 0.05, 800: 0.09},
    "seiyaku_city": {"13101": -0.10, "13102": -0.15},
    "seiyaku_time": {2012: 0.0, 2013: 0.03},
}
PLANS = ["2LDK", "1K", "1LDK", "3LDK", "4LDK", "2LDK+S"]
FARS = [200, 300, 500, 800, float("nan")]


def true_building(area, age):
    la = math.log(area)
    a = TRUE["area"]
    out = a[0] * la + sum(b * max(0.0, la - math.log(k)) for b, k in zip(a[1:6], (25, 40, 55, 70, 90), strict=True))
    g = TRUE["age"]
    return out + g[0] * age + sum(b * max(0.0, age - k) for b, k in zip(g[1:6], (5, 10, 20, 30, 40), strict=True))


def true_far(x):
    return 0.03 if math.isnan(x) else TRUE["far"][x]


def make_sample(rng, n_per=60, noise=0.0, district_weights=None, new_share=None):
    """各區、各年各 n_per 筆。district_weights(city, year) → 各町名的機率；new_share(year) → 新物件的比例。"""
    cols = {k: [] for k in ("city", "district", "year", "quarter", "price_classification", "log_price", "area",
                             "building_year", "floor_plan", "structure", "renovation", "city_planning", "floor_area_ratio")}
    for city, districts in CITIES.items():
        for year in YEARS:
            w = district_weights(city, year) if district_weights else None
            rows = [(rng.choice(districts, p=w)) for _ in range(n_per)]
            if year == 2011:
                rows += [SMALL[city]] * 3  # 小町名每區只有 3 筆
            for d in rows:
                area = float(rng.choice([20, 30, 45, 55, 65, 80, 100]))
                newer = new_share(year) if new_share else 0.5
                age = int(rng.integers(0, 10) if rng.random() < newer else rng.integers(10, 50))
                by = year - age
                plan = str(rng.choice(PLANS))
                st = str(rng.choice(list(TRUE["structure"])))
                ren = str(rng.choice(list(TRUE["renovation"])))
                cp = str(rng.choice(list(TRUE["city_planning"])))
                far = float(rng.choice(FARS))
                pc = "02" if year >= 2012 and rng.random() < 0.4 else "01"
                group = floor_plan_group(plan)
                y = TRUE["district"][d] + TRUE["time"][city][year] + true_building(area, age)
                y += TRUE["old_seismic"] if by <= 1981 else 0.0
                y += TRUE["plan"][group[0]] + (TRUE["plus_s"] if group[1] else 0.0)
                y += TRUE["structure"][st] + TRUE["renovation"][ren] + TRUE["city_planning"][cp] + true_far(far)
                if pc == "02":
                    y += TRUE["seiyaku_city"][city] + TRUE["seiyaku_time"][year]
                y += rng.normal(0, noise) if noise else 0.0
                for k, v in zip(cols, (city, d, year, int(rng.integers(1, 5)), pc, y, area, by, plan, st, ren, cp, far), strict=True):
                    cols[k].append(v)
    arrays = {k: np.array(v, dtype=object if isinstance(v[0], str) else None) for k, v in cols.items()}
    return Sample.from_columns(arrays)


@pytest.fixture(scope="module")
def exact_model():
    s = make_sample(np.random.default_rng(1))
    return s, fit_hedonic(s, yearly_period(s))


def test_exact_data_recovers_coefficients(exact_model):
    _, m = exact_model
    assert m.outliers == 0
    assert m.fit.r2 == pytest.approx(1.0)
    assert m.coef("舊耐震")[0] == pytest.approx(TRUE["old_seismic"], abs=1e-9)
    assert m.coef("房型:1R/1K")[0] == pytest.approx(0.07, abs=1e-9)
    assert m.coef("房型:+S")[0] == pytest.approx(-0.02, abs=1e-9)
    assert m.coef("構造:其他")[0] == pytest.approx(-0.06, abs=1e-9)
    assert m.coef("改裝:不明")[0] == pytest.approx(0.01, abs=1e-9)
    assert m.coef("容積率:>600")[0] == pytest.approx(0.09, abs=1e-9)
    assert m.coef("成約:13102")[0] == pytest.approx(-0.15, abs=1e-9)
    assert m.coef("成約×時間:2013")[0] == pytest.approx(0.03, abs=1e-9)
    assert m.coef("屋齡>20")[0] == pytest.approx(0.002, abs=1e-9)
    # 基準期間的係數 = 0，其他期間 = 和基準的差
    assert m.coef("時間:13102:2013")[0] == pytest.approx(0.20, abs=1e-9)
    assert "時間:13101:2010" not in m.design.index


def test_small_districts_are_merged(exact_model):
    _, m = exact_model
    assert f"町名:13101 {OTHER_DISTRICT}" in m.design.index
    assert not any("小町" in nm for nm in m.design.names)
    assert MIN_DISTRICT_N > 3


def test_index_levels_and_growth_are_exact(exact_model):
    s, m = exact_model
    series = index_series(m)
    for city in CITIES:
        rows = s.city == city
        # 地點組成：町名（含基準期間吸收進來的部分＝0）、用途地域、容積率，依該區筆數平均
        loc = np.mean([TRUE["district"][d] + TRUE["city_planning"][cp] + true_far(f)
                       for d, cp, f in zip(s.district[rows], s.city_planning[rows], s.floor_area_ratio[rows], strict=True)])
        standard = true_building(55, 17)
        for year in YEARS:
            assert series[city].levels[year] == pytest.approx(TRUE["time"][city][year] + loc + standard, abs=1e-9)
        g = hedonic_growth(series[city], 2010, 2013)
        assert g.total == pytest.approx(math.exp(TRUE["time"][city][2013]) - 1, abs=1e-9)
    assert series["13101"].n[2011] == 63  # 小町名的 3 筆也算在該區


def test_composition_shift_does_not_move_index():
    # 真實價格完全沒變，但後面幾年賣的多是貴的町名、新的物件 → 中位數上漲，指數不動
    flat = {c: dict.fromkeys(YEARS, 0.0) for c in CITIES}
    saved = TRUE["time"]
    TRUE["time"] = flat
    try:
        s = make_sample(
            np.random.default_rng(2),
            n_per=200,
            noise=0.05,
            district_weights=lambda c, y: [0.1 + 0.2 * (y - 2010), 0.45 - 0.1 * (y - 2010), 0.45 - 0.1 * (y - 2010)],
            new_share=lambda y: 0.2 + 0.2 * (y - 2010),
        )
    finally:
        TRUE["time"] = saved
    m = fit_hedonic(s, yearly_period(s))
    series = index_series(m)
    for city in CITIES:
        rows = (s.city == city) & (s.price_classification == "01")
        med = {y: np.median(s.log_price[rows & (s.year == y)]) for y in (2010, 2013)}
        assert med[2013] - med[2010] > 0.10  # 中位數漲了 10% 以上
        g = hedonic_growth(series[city], 2010, 2013)
        assert abs(g.total) < 0.03  # 指數幾乎沒動
        assert g.ci_low < 0 < g.ci_high


def test_outliers_are_removed_and_reported():
    s = make_sample(np.random.default_rng(3), noise=0.05)
    s.log_price[5] += 3.0  # 20 倍
    m = fit_hedonic(s, yearly_period(s))
    assert m.outliers >= 1
    assert len(m.sample) == len(s) - m.outliers
    assert m.coef("舊耐震")[0] == pytest.approx(TRUE["old_seismic"], abs=0.03)


def test_cluster_robust_cov_matches_hand_computation():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 3))
    y = rng.normal(size=12)
    clusters = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3])
    f = fit_ols(sp.csr_matrix(X), y, clusters)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[clusters == g].T @ e[clusters == g], X[clusters == g].T @ e[clusters == g]) for g in range(4))
    cov = bread @ meat @ bread * (4 / 3) * (11 / 9)
    assert f.beta == pytest.approx(beta)
    assert f.cov == pytest.approx(cov)
    assert f.clusters == 4 and f.n == 12


def test_singular_design_is_reported():
    X = sp.csr_matrix(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    with pytest.raises(ValueError, match="滿秩"):
        fit_ols(X, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]))


def test_growth_se_uses_time_covariance(exact_model):
    s = make_sample(np.random.default_rng(5), noise=0.1)
    m = fit_hedonic(s, yearly_period(s))
    series = index_series(m)["13101"]
    g = hedonic_growth(series, 2011, 2013)
    i, j = m.design.index["時間:13101:2011"], m.design.index["時間:13101:2013"]
    var = m.fit.cov[i, i] + m.fit.cov[j, j] - 2 * m.fit.cov[i, j]
    d = series.levels[2013] - series.levels[2011]
    assert g.total == pytest.approx(math.exp(d) - 1)
    assert g.ci_high == pytest.approx(math.exp(d + 1.959964 * math.sqrt(var)) - 1)
    # 從基準期間開始的漲幅：只有迄年那個係數的變異
    g0 = hedonic_growth(series, 2010, 2013)
    assert g0.ci_high == pytest.approx(math.exp(series.levels[2013] - series.levels[2010] + 1.959964 * math.sqrt(m.fit.cov[j, j])) - 1)


def test_growth_trend_price_thresholds(exact_model):
    _, m = exact_model
    series = index_series(m)["13101"]
    with pytest.raises(ValueError, match="迄年"):
        hedonic_growth(series, 2013, 2011)
    small = type(series)(series.city, series.levels, {**series.n, 2011: MIN_N - 1}, series.level_se, series.time_cov)
    assert hedonic_growth(small, 2011, 2013).status == "insufficient"
    assert hedonic_price(small, 2011).status == "insufficient"
    assert hedonic_price(None, 2011).status == "insufficient"
    p = hedonic_price(series, 2013)
    assert p.status == "ok" and p.median == pytest.approx(math.exp(series.levels[2013]))
    t = hedonic_trend(series, 2013)
    assert t.years_used == 4 and t.status == "insufficient"  # 10 年裡只有 4 年，不到 6 年


def test_predict_with_template(exact_model):
    s, m = exact_model
    pred, keep = predict(m, s, yearly_period(s))
    assert keep.all()
    assert pred == pytest.approx(s.log_price, abs=1e-9)
    # 沒見過的期間（早於基準）和沒見過的類別不能預測
    other = s.subset(np.arange(4))
    other.year = np.array([2009, 2011, 2011, 2011])
    other.building_year = np.array([2000, 2000, 2000, 2000])
    other.structure = np.array(["RC", "RC", "RC", "RC"], dtype=object)
    other.floor_plan = np.array(["2LDK", "2LDK", "9LDK", "2LDK"], dtype=object)
    _, keep2 = predict(m, other, yearly_period(other))
    assert keep2.tolist() == [False, True, True, True]


def test_design_rejects_unseen_category(exact_model):
    s, m = exact_model
    other = s.subset(np.arange(2))
    other.city_planning = np.array(["工業地域", "商業地域"], dtype=object)  # 訓練資料沒有「工業地域」
    _, keep = design_matrix(other, yearly_period(other), Spec(), template=m.design)
    assert keep.tolist() == [False, True]


def test_seiyaku_only_kept_for_cities_with_both_sources(exact_model):
    # 奥多摩町那種情況：只有 2 筆成約、沒有取引 → 成約效果和町名效果分不開，那 2 筆不放
    s, _ = exact_model
    extra = s.subset(np.arange(2))
    extra.city = np.array(["13308", "13308"], dtype=object)
    extra.district = np.array(["氷川", "氷川"], dtype=object)
    extra.price_classification = np.array(["02", "02"], dtype=object)
    combined = Sample(**{k: np.concatenate([getattr(s, k), getattr(extra, k)]) for k in s.__dataclass_fields__})
    d, keep = design_matrix(combined, yearly_period(combined), Spec())
    assert keep[-2:].tolist() == [False, False] and keep[:-2].all()
    assert "13308" not in d.seiyaku_cities and "成約:13308" not in d.index
    assert {"13101", "13102"} <= d.seiyaku_cities


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-6])
def test_collinear_columns_are_named(eps):
    # 第二欄（幾乎）等於第一欄。eps = 1e-6 時 Cholesky 不會失敗，但會算出互相抵銷的荒謬係數 → 也要擋下來
    a = np.arange(1.0, 7.0)
    X = sp.csr_matrix(np.column_stack([np.ones(6), a, a * (1 + eps) + eps * a**2]))
    with pytest.raises(ValueError, match="丙"):
        fit_ols(X, a * 2, np.arange(6), names=["甲", "乙", "丙"])


def test_empty_column_is_named():
    X = sp.csr_matrix(np.column_stack([np.ones(4), np.zeros(4)]))
    with pytest.raises(ValueError, match="乙"):
        fit_ols(X, np.ones(4), np.arange(4), names=["甲", "乙"])
