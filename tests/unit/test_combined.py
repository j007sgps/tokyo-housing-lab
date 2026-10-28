import math

import pytest

from tokyo_housing_lab.analysis.combined import (
    OFFSET_MIN_YEARS,
    combined_growth,
    combined_latest,
    combined_series,
    combined_trend,
    fit_offset,
)
from tokyo_housing_lab.analysis.trend import MIN_N, YearStat, growth_between, median_log_se


def stat(median, n=100, spread=0.4):
    return YearStat(n, median, spread)


def series(rate, years, base=1_000_000, n=100, factor=1.0):
    return {y: stat(base * factor * (1 + rate) ** (y - years[0]), n=n) for y in years}


OLD = list(range(2016, 2026))
OVERLAP = list(range(2021, 2026))


# --- 落差 ---


def test_offset_exact_ratio():
    # 成約剛好是取引 × 0.9：取引比成約高 1/0.9 − 1 ≈ 11.1%，每年一樣 → τ² = 0
    y01 = series(0.05, OLD)
    y02 = {y: stat(y01[y].median * 0.9) for y in OVERLAP}
    off, weights = fit_offset(y01, y02)
    assert off.status == "ok"
    assert off.log == pytest.approx(math.log(1 / 0.9))
    assert off.pct == pytest.approx(1 / 0.9 - 1)
    assert off.tau2 == 0
    assert off.years == 5
    assert sum(weights.values()) == pytest.approx(1)


def test_offset_needs_min_overlap_years():
    y01 = series(0.05, OLD)
    y02 = {y: stat(y01[y].median * 0.9) for y in OVERLAP[: OFFSET_MIN_YEARS - 1]}
    off, weights = fit_offset(y01, y02)
    assert off.status == "insufficient" and off.log is None and weights == {}


def test_offset_ignores_small_years():
    y01 = series(0.05, OLD)
    y02 = {y: stat(y01[y].median * 0.9) for y in OVERLAP}
    y02[2023] = stat(1, n=MIN_N - 1)  # 離譜但筆數不足：不用
    off, weights = fit_offset(y01, y02)
    assert off.years == 4 and 2023 not in weights
    assert off.log == pytest.approx(math.log(1 / 0.9))


def test_offset_drifting_gap_has_tau2():
    # 落差每年漂移（+20% → −10%），遠超過抽樣誤差 → τ² > 0，θ̂ 的誤差也跟著變大
    y01 = series(0.05, OLD, n=2000)
    gaps = [1.2, 1.1, 1.0, 0.95, 0.9]
    y02 = {y: stat(y01[y].median / g, n=2000) for y, g in zip(OVERLAP, gaps, strict=True)}
    off, _ = fit_offset(y01, y02)
    steady, _ = fit_offset(y01, {y: stat(y01[y].median / 1.03, n=2000) for y in OVERLAP})
    assert off.tau2 > 0.005
    assert off.se > 3 * steady.se


def test_offset_matches_hand_computed_dl():
    # 3 個重疊年，手算 DerSimonian–Laird
    y01 = {2021: stat(100, n=100), 2022: stat(110, n=400), 2023: stat(120, n=100)}
    y02 = {2021: stat(90, n=100), 2022: stat(90, n=400), 2023: stat(110, n=100)}
    d = [math.log(100 / 90), math.log(110 / 90), math.log(120 / 110)]
    v = [2 * median_log_se(y01[y]) ** 2 for y in (2021, 2022, 2023)]
    w = [1 / x for x in v]
    fixed = sum(a * b for a, b in zip(w, d, strict=True)) / sum(w)
    q = sum(a * (b - fixed) ** 2 for a, b in zip(w, d, strict=True))
    tau2 = max(0, (q - 2) / (sum(w) - sum(a * a for a in w) / sum(w)))
    ws = [1 / (x + tau2) for x in v]
    off, _ = fit_offset(y01, y02)
    assert off.tau2 == pytest.approx(tau2)
    assert off.log == pytest.approx(sum(a * b for a, b in zip(ws, d, strict=True)) / sum(ws))
    assert off.se == pytest.approx(math.sqrt(1 / sum(ws)))


# --- 綜合序列 ---


def exact_pair(rate=0.05):
    y01 = series(rate, OLD)
    return y01, {y: stat(y01[y].median * 0.9) for y in OVERLAP}


def test_combined_equals_toriihiki_when_ratio_is_exact():
    y01, y02 = exact_pair()
    s = combined_series(y01, y02)
    for y in OLD:
        assert s.levels[y].median == pytest.approx(y01[y].median)
    assert s.levels[2020].n == 100 and s.levels[2021].n == 200


def test_combined_growth_same_value_narrower_ci():
    y01, y02 = exact_pair()
    s = combined_series(y01, y02)
    g = combined_growth(s, 2021, 2025)
    g01 = growth_between(y01, 2021, 2025)
    assert g.total == pytest.approx(g01.total)
    assert (g.ci_high - g.ci_low) < (g01.ci_high - g01.ci_low)  # 多了成約的資料，區間變窄


def test_combined_growth_across_2021_includes_offset_error():
    # 只用取引的 2016 → 換算後的成約參與的 2025：θ̂ 的誤差要算進去，但仍不會比只用取引寬太多
    y01, y02 = exact_pair()
    s = combined_series(y01, y02)
    g = combined_growth(s, 2016, 2025)
    assert g.total == pytest.approx(growth_between(y01, 2016, 2025).total)
    assert g.status == "ok"


def test_combined_growth_variance_matches_hand_computation():
    # 3 個重疊年、τ² = 0，2021→2023 的變異數照線性組合手算
    y01 = {2021: stat(100, n=100), 2022: stat(105, n=100), 2023: stat(110, n=100)}
    y02 = {y: stat(y01[y].median * 0.9, n=400) for y in y01}
    s = combined_series(y01, y02)
    off = s.offset
    v1, v2 = median_log_se(y01[2021]) ** 2, median_log_se(y02[2021]) ** 2
    a = {y: 1 / 3 for y in y01}  # 每年的落差誤差一樣 → 權重平均
    assert s.weights == pytest.approx(a)
    w1, w2 = 1 / v1, 1 / (v2 + off.se**2)
    p, q = w1 / (w1 + w2), w2 / (w1 + w2)

    def terms(y):
        t = {("01", t_): q * a[t_] for t_ in a} | {("02", t_): -q * a[t_] for t_ in a}
        t[("01", y)] += p
        t[("02", y)] += q
        return t

    t21, t23 = terms(2021), terms(2023)
    var = sum((t23[k] - t21[k]) ** 2 * (v1 if k[0] == "01" else v2) for k in t21)
    g = combined_growth(s, 2021, 2023)
    se = (math.log(1 + g.ci_high) - math.log(1 + g.total)) / 1.959964
    assert se == pytest.approx(math.sqrt(var))
    # θ̂ 在兩年都一樣，差分時抵銷：等於 p²(2v1) + q²(2v2)
    assert var == pytest.approx(p * p * 2 * v1 + q * q * 2 * v2)


def test_growth_from_toriihiki_only_year_includes_offset_error():
    # 2020 只有取引、2021〜2023 有兩邊：θ̂ 不會抵銷，它的誤差要算進漲幅的區間
    y01 = {2020: stat(95, n=100), 2021: stat(100, n=100), 2022: stat(105, n=100), 2023: stat(110, n=100)}
    y02 = {y: stat(y01[y].median * 0.9, n=400) for y in (2021, 2022, 2023)}
    s = combined_series(y01, y02)
    v1, v2 = median_log_se(y01[2021]) ** 2, median_log_se(y02[2021]) ** 2
    w1, w2 = 1 / v1, 1 / (v2 + s.offset.se**2)
    p, q = w1 / (w1 + w2), w2 / (w1 + w2)
    # 2023 = p·x01_23 + q·(x02_23 + Σ⅓(x01_t − x02_t))；2020 = x01_20
    c01 = {t: q / 3 + (p if t == 2023 else 0) for t in (2021, 2022, 2023)}
    c02 = {t: -q / 3 + (q if t == 2023 else 0) for t in (2021, 2022, 2023)}
    var = v1 + sum(c * c * v1 for c in c01.values()) + sum(c * c * v2 for c in c02.values())
    g = combined_growth(s, 2020, 2023)
    se = (math.log(1 + g.ci_high) - math.log(1 + g.total)) / 1.959964
    assert se == pytest.approx(math.sqrt(var))
    assert se > math.sqrt(v1 + p * p * v1 + q * q * v2)  # 比「忘了 θ̂ 的誤差」寬


def test_drifting_gap_lowers_seiyaku_weight():
    y01 = series(0.05, OLD, n=2000)
    drift = combined_series(y01, {y: stat(y01[y].median / g, n=2000) for y, g in zip(OVERLAP, [1.2, 1.1, 1.0, 0.95, 0.9], strict=True)})
    steady = combined_series(y01, {y: stat(y01[y].median / 1.03, n=2000) for y in OVERLAP})
    assert drift.levels[2023].terms[("02", 2023)] < steady.levels[2023].terms[("02", 2023)] / 2
    assert drift.var[("02", 2023)] == pytest.approx(median_log_se(stat(1, n=2000)) ** 2 + drift.offset.tau2)


def test_only_one_side_usable_in_a_year():
    y01, y02 = exact_pair()
    y01[2024] = stat(1, n=MIN_N - 1)  # 取引這年筆數不足 → 只用換算後的成約
    s = combined_series(y01, y02)
    assert s.levels[2024].median == pytest.approx(y02[2024].median / 0.9)
    assert s.levels[2024].n == 100


def test_insufficient_offset_falls_back_to_toriihiki():
    y01 = series(0.05, OLD)
    y02 = {2025: stat(500_000)}  # 只有 1 個重疊年
    s = combined_series(y01, y02)
    assert s.offset.status == "insufficient"
    assert s.levels[2025].median == pytest.approx(y01[2025].median)
    g, g01 = combined_growth(s, 2016, 2025), growth_between(y01, 2016, 2025)
    assert (g.total, g.ci_low, g.ci_high) == pytest.approx((g01.total, g01.ci_low, g01.ci_high))


def test_combined_growth_insufficient_and_bad_order():
    y01, y02 = exact_pair()
    y01[2018] = stat(1, n=MIN_N - 1)
    s = combined_series(y01, y02)
    assert combined_growth(s, 2018, 2025).status == "insufficient"
    with pytest.raises(ValueError, match="迄年"):
        combined_growth(s, 2025, 2021)


def test_combined_trend_recovers_rate():
    y01, y02 = exact_pair(0.04)
    t = combined_trend(combined_series(y01, y02), 2025)
    assert t.annual == pytest.approx(0.04)
    assert t.years_used == 10


# --- 最新 4 季 ---


def test_combined_latest_in_toriihiki_level():
    y01, y02 = exact_pair()
    s = combined_series(y01, y02)
    p = combined_latest(s, stat(2_000_000), stat(1_800_000))
    assert p.status == "ok"
    assert p.median == pytest.approx(2_000_000)
    assert p.n == 200
    only01 = combined_latest(s, stat(2_000_000), None)
    assert (p.ci_high - p.ci_low) < (only01.ci_high - only01.ci_low)


def test_combined_latest_insufficient():
    y01, y02 = exact_pair()
    s = combined_series(y01, y02)
    p = combined_latest(s, stat(1, n=5), None)
    assert p.status == "insufficient" and p.median is None and p.n == 5
