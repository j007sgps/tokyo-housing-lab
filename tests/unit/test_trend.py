import math

import pytest

from tokyo_housing_lab.analysis.trend import (
    INVEST_MAX_CI_WIDTH,
    INVEST_MIN_YEARS,
    MIN_N,
    YearStat,
    growth_between,
    latest_price,
    long_term_trend,
    median_log_se,
)


def stat(median, n=100, spread=0.4):
    return YearStat(n, median, spread)


# --- 中位數的誤差 ---


def test_median_se_shrinks_with_sqrt_n():
    assert median_log_se(stat(1, n=400)) == pytest.approx(median_log_se(stat(1, n=100)) / 2)
    # √(π/2) × 0.4 / √100
    assert median_log_se(stat(1, n=100, spread=0.4)) == pytest.approx(math.sqrt(math.pi / 2) * 0.04)


# --- 購買模式 ---


def test_latest_price_ci_is_symmetric_in_log():
    p = latest_price(stat(1_000_000, n=MIN_N))
    assert p.status == "ok"
    assert p.ci_low < 1_000_000 < p.ci_high
    assert math.log(p.ci_high / p.median) == pytest.approx(math.log(p.median / p.ci_low))


@pytest.mark.parametrize("value", [None, stat(1_000_000, n=MIN_N - 1), stat(1_000_000, n=0)])
def test_latest_price_insufficient(value):
    p = latest_price(value)
    assert p.status == "insufficient"
    assert p.ci_low is None and p.ci_high is None


def test_latest_price_boundary_n_equals_min_is_ok():
    assert latest_price(stat(1, n=MIN_N)).status == "ok"


# --- 趨勢模式 ---


def test_growth_basic_math():
    g = growth_between({2015: stat(500_000, n=10_000), 2025: stat(1_000_000, n=10_000)}, 2015, 2025)
    assert g.total == pytest.approx(1.0)  # 漲一倍
    assert g.annual == pytest.approx(2 ** (1 / 10) - 1)  # 每年約 +7.2%
    assert g.ci_low < 1.0 < g.ci_high
    assert g.status == "ok"


def test_growth_ci_contains_zero_is_uncertain():
    g = growth_between({2024: stat(1_000_000, n=MIN_N), 2025: stat(1_030_000, n=MIN_N)}, 2024, 2025)
    assert g.ci_low < 0 < g.ci_high
    assert g.status == "uncertain"
    assert g.total == pytest.approx(0.03)  # 數字照樣給，只是標成看不出漲跌


def test_growth_can_be_negative():
    g = growth_between({2007: stat(1_000_000, n=5000), 2012: stat(800_000, n=5000)}, 2007, 2012)
    assert g.total == pytest.approx(-0.2)
    assert g.ci_high < 0
    assert g.status == "ok"


@pytest.mark.parametrize(
    "series",
    [
        {2025: stat(1)},  # 起年沒資料
        {2015: stat(1)},  # 迄年沒資料
        {2015: stat(1, n=MIN_N - 1), 2025: stat(2)},
        {2015: stat(1), 2025: stat(2, n=MIN_N - 1)},
    ],
)
def test_growth_insufficient(series):
    g = growth_between(series, 2015, 2025)
    assert g.status == "insufficient"
    assert g.total is None


@pytest.mark.parametrize(("start", "end"), [(2025, 2025), (2025, 2015)])
def test_growth_rejects_bad_order(start, end):
    with pytest.raises(ValueError, match="迄年"):
        growth_between({}, start, end)


# --- 投資模式 ---


def exact_series(rate, years=range(2016, 2026), n=100, base=500_000):
    return {y: stat(base * (1 + rate) ** (y - 2016), n=n) for y in years}


def test_long_trend_recovers_exact_growth_rate():
    t = long_term_trend(exact_series(0.05), 2025)
    assert (t.first_year, t.last_year, t.years_used) == (2016, 2025, 10)
    assert t.annual == pytest.approx(0.05)
    assert t.ci_low == pytest.approx(0.05) and t.ci_high == pytest.approx(0.05)  # 完全吻合時區間寬度 0
    assert t.status == "ok"


def test_long_trend_negative_rate():
    assert long_term_trend(exact_series(-0.02), 2025).annual == pytest.approx(-0.02)


def test_long_trend_known_ols_with_noise():
    # 6 個有雜訊的點：和手算的最小平方法（自由度 4，t = 2.776）比對
    logs = [0.0, 0.1, 0.1, 0.3, 0.3 + 0.09 * 1, 0.3 + 0.09 * 2]
    series = {2020 + i: stat(math.exp(v)) for i, v in enumerate(logs)}
    t = long_term_trend(series, 2025)
    xs = list(range(6))
    mx, my = sum(xs) / 6, sum(logs) / 6
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, logs, strict=True)) / sxx
    a = my - b * mx
    s2 = sum((y - a - b * x) ** 2 for x, y in zip(xs, logs, strict=True)) / 4
    half = 2.776 * math.sqrt(s2 / sxx)  # t(4)
    assert t.annual == pytest.approx(math.exp(b) - 1)
    assert t.ci_low == pytest.approx(math.exp(b - half) - 1, rel=1e-3)
    assert t.ci_high == pytest.approx(math.exp(b + half) - 1, rel=1e-3)


def test_long_trend_only_uses_last_10_years():
    series = exact_series(0.05) | {2010: stat(1)}  # 很久以前的離譜值不影響
    assert long_term_trend(series, 2025).annual == pytest.approx(0.05)


def test_long_trend_skips_small_years():
    series = exact_series(0.05)
    series[2020] = stat(1, n=MIN_N - 1)  # 這年樣本不足：不算，不會被離譜值拉走
    t = long_term_trend(series, 2025)
    assert t.years_used == 9
    assert t.annual == pytest.approx(0.05)


def test_long_trend_insufficient_years():
    series = exact_series(0.05, years=range(2025 - INVEST_MIN_YEARS + 2, 2026))
    t = long_term_trend(series, 2025)
    assert t.years_used == INVEST_MIN_YEARS - 1
    assert t.status == "insufficient" and t.annual is None


def test_long_trend_min_years_boundary_is_computed():
    series = exact_series(0.05, years=range(2025 - INVEST_MIN_YEARS + 1, 2026))
    assert long_term_trend(series, 2025).status == "ok"


def test_long_trend_wide_ci_is_uncertain():
    # 一年高一年低的鋸齒：平均有成長，但區間很寬
    series = {y: stat(500_000 * 1.05 ** (y - 2016) * (1.3 if y % 2 else 0.75)) for y in range(2016, 2026)}
    t = long_term_trend(series, 2025)
    assert t.ci_high - t.ci_low > INVEST_MAX_CI_WIDTH
    assert t.status == "uncertain"
    assert t.annual is not None  # 數字照樣給
