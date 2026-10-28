"""分析：三種地圖模式的數字怎麼算。全部是純函式，不碰資料庫。

- 購買模式：最新 4 季的 ㎡ 單價中位數（latest_price）
- 趨勢模式：起年 → 迄年中位數的漲幅與 95% 區間（growth_between）
- 投資模式：最近 10 個完整年，年度中位數取 log 做最小平方法 → 年成長率與 95% 區間（long_term_trend）

為什麼要區間：實測一年只有 20 筆時，中位數本身的 95% 誤差就有 ±18%（江東区 2024 年 bootstrap），
「漲了 15%」可能只是抽樣的運氣。所以每個數字都附區間，樣本太少的不上色。

注意：這些都只是過去資料的整理，不是預測，也不是投資建議。中位數還會受「那年賣出的物件組成」
（新舊、大小）影響，尚未做時點修正或品質調整。
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

# 門檻（步驟 2 依實際資料的筆數分布提出，等使用者確認）
MIN_N = 30  # 一年（或最新 4 季）少於這個筆數就算「樣本不足」：30 筆時中位數誤差約 ±13%
INVEST_WINDOW_YEARS = 10  # 投資模式看最近幾個完整年
INVEST_MIN_YEARS = 6  # 10 年裡至少要有幾年達到 MIN_N 才計算
INVEST_MAX_CI_WIDTH = 0.06  # 年成長率的 95% 區間寬度超過 6 個百分點 → 「不確定」

# 中位數的標準誤 ≈ √(π/2) × σ / √n（大樣本、近似常態時）
_MEDIAN_SE_FACTOR = math.sqrt(math.pi / 2)
_Z95 = 1.959964
# t 分布 97.5% 分位數（自由度 1〜30），算最小平方法斜率的區間用；不想為了這張表裝 scipy
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}  # fmt: skip

Status = Literal["ok", "uncertain", "insufficient"]


@dataclass(frozen=True)
class YearStat:
    """某區某一年（或某段期間）的統計。spread_log = 1.4826 × MAD(ln 單價)。"""

    n: int
    median: float
    spread_log: float


def median_log_se(stat: YearStat) -> float:
    """ln(中位數) 的標準誤。"""
    return _MEDIAN_SE_FACTOR * stat.spread_log / math.sqrt(stat.n)


@dataclass(frozen=True)
class Price:
    """購買模式：中位數與 95% 區間。"""

    median: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    status: Status  # ok／insufficient


def latest_price(stat: YearStat | None) -> Price:
    if stat is None or stat.n < MIN_N:
        return Price(stat.median if stat else None, None, None, stat.n if stat else 0, "insufficient")
    se = median_log_se(stat)
    return Price(stat.median, stat.median * math.exp(-_Z95 * se), stat.median * math.exp(_Z95 * se), stat.n, "ok")


@dataclass(frozen=True)
class Growth:
    """趨勢模式：start_year → end_year 的漲幅。total = 0.25 代表漲了 25%。"""

    start_year: int
    end_year: int
    total: float | None
    annual: float | None  # 換算成每年（複利）
    ci_low: float | None  # total 的 95% 區間
    ci_high: float | None
    status: Status  # ok／uncertain（區間包含 0，看不出漲跌）／insufficient


def growth_between(series: Mapping[int, YearStat], start_year: int, end_year: int) -> Growth:
    if end_year <= start_year:
        raise ValueError(f"迄年必須晚於起年：{start_year} → {end_year}")
    a, b = series.get(start_year), series.get(end_year)
    if a is None or b is None or a.n < MIN_N or b.n < MIN_N:
        return Growth(start_year, end_year, None, None, None, None, "insufficient")
    d = math.log(b.median) - math.log(a.median)
    se = math.hypot(median_log_se(a), median_log_se(b))
    low, high = math.exp(d - _Z95 * se) - 1, math.exp(d + _Z95 * se) - 1
    status: Status = "uncertain" if low <= 0 <= high else "ok"
    annual = math.exp(d / (end_year - start_year)) - 1
    return Growth(start_year, end_year, math.exp(d) - 1, annual, low, high, status)


@dataclass(frozen=True)
class LongTrend:
    """投資模式：first_year〜last_year 的年成長率（最小平方法）。annual = 0.05 代表每年 +5%。"""

    first_year: int
    last_year: int
    years_used: int  # 達到 MIN_N、實際拿來算的年數
    annual: float | None
    ci_low: float | None
    ci_high: float | None
    status: Status  # ok／uncertain（區間太寬）／insufficient（有效年數不足）


def long_term_trend(series: Mapping[int, YearStat], last_year: int) -> LongTrend:
    first_year = last_year - INVEST_WINDOW_YEARS + 1
    points = [
        (year, math.log(series[year].median))
        for year in range(first_year, last_year + 1)
        if year in series and series[year].n >= MIN_N
    ]
    k = len(points)
    if k < INVEST_MIN_YEARS:
        return LongTrend(first_year, last_year, k, None, None, None, "insufficient")

    mean_x = sum(x for x, _ in points) / k
    mean_y = sum(y for _, y in points) / k
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    intercept = mean_y - slope * mean_x
    residual_var = sum((y - intercept - slope * x) ** 2 for x, y in points) / (k - 2)
    half = _T95[k - 2] * math.sqrt(residual_var / sxx)
    low, high = math.exp(slope - half) - 1, math.exp(slope + half) - 1
    status: Status = "uncertain" if high - low > INVEST_MAX_CI_WIDTH else "ok"
    return LongTrend(first_year, last_year, k, math.exp(slope) - 1, low, high, status)
