"""綜合指標：把成約價格換算成取引價格的水準，和取引價格依誤差加權平均，得到一條 2006 年起的序列。

為什麼不直接把兩邊的成交混在一起：同一區、同一年，取引的中位數通常比成約高（實測各區 −3%〜+53%），
2021 年成約加入時中位數會突然往下掉；同一筆成交也可能兩邊都有（API 沒有交易 ID，無法去重複）。

做法（全部以 ln(中位數) 計）：
1. 每區的落差 θ：重疊年（兩邊都達到 MIN_N）的 ln(取引) − ln(成約)，用 random-effects（DerSimonian–Laird）
   估計。τ² 是落差本身每年的變動（扣掉抽樣誤差後）；實測有些區的落差逐年漂移（中央区 +9% → −9%），
   當成固定值區間會太窄。重疊年不到 OFFSET_MIN_YEARS 年的區不換算，綜合就只用取引。
2. 成約換算成取引水準：ln(成約) + θ̂，誤差 = 抽樣誤差 + τ² + θ̂ 的誤差。
3. 每年兩邊依誤差的倒數加權平均；只有一邊達到 MIN_N 的年份就只用那一邊。
4. 誤差：每個綜合值都寫成「各觀測值的線性組合」並保留係數。各觀測值（每年每種價格的 ln 中位數，
   成約那邊連同 τ²）視為互相獨立，任兩年的差的變異數就能精確算出——θ̂ 用的是同一批資料，
   兩者的相關會自動算進去。

已知限制：同一筆成交可能同時在取引和成約裡，兩邊的誤差其實正相關，算出來的區間會偏窄。
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from tokyo_housing_lab.analysis.trend import (
    INVEST_WINDOW_YEARS,
    MIN_N,
    Growth,
    LongTrend,
    Price,
    YearStat,
    growth_from_log,
    median_log_se,
    price_from_log,
    trend_from_points,
)

OFFSET_MIN_YEARS = 3  # 至少要幾個重疊年才換算成約

Key = tuple[str, int | str]  # 觀測值：(價格種類, 年 或 "latest")
LATEST = "latest"


@dataclass(frozen=True)
class Offset:
    """取引比成約高多少（ln）。status=insufficient 時不換算，綜合只用取引。"""

    log: float | None
    se: float | None  # θ̂ 的標準誤（已含 τ²）
    tau2: float | None
    years: int  # 用到的重疊年數
    status: Literal["ok", "insufficient"]

    @property
    def pct(self) -> float | None:
        """成約比取引便宜多少：取引 ÷ 成約 − 1。"""
        return None if self.log is None else math.exp(self.log) - 1


@dataclass(frozen=True)
class Level:
    """某一年的綜合值：ln 價格 = Σ 係數 × 觀測值。"""

    log: float
    n: int  # 兩邊實際用到的筆數合計（可能有重複的成交）
    terms: dict[Key, float]

    @property
    def median(self) -> float:
        return math.exp(self.log)


@dataclass
class CombinedSeries:
    offset: Offset
    weights: dict[int, float]  # θ̂ = Σ weights[y] × (ln 取引_y − ln 成約_y)
    levels: dict[int, Level]
    var: dict[Key, float]  # 各觀測值的變異數

    def se(self, terms: Mapping[Key, float]) -> float:
        return math.sqrt(sum(c * c * self.var[k] for k, c in terms.items()))


def _usable(stat: YearStat | None) -> bool:
    return stat is not None and stat.n >= MIN_N


def fit_offset(y01: Mapping[int, YearStat], y02: Mapping[int, YearStat]) -> tuple[Offset, dict[int, float]]:
    """回傳 (Offset, 各重疊年在 θ̂ 裡的權重 a_y)；θ̂ = Σ a_y (ln 取引_y − ln 成約_y)。"""
    years = sorted(y for y in y01 if _usable(y01[y]) and _usable(y02.get(y)))
    k = len(years)
    if k < OFFSET_MIN_YEARS:
        return Offset(None, None, None, k, "insufficient"), {}
    d = {y: math.log(y01[y].median) - math.log(y02[y].median) for y in years}
    v = {y: median_log_se(y01[y]) ** 2 + median_log_se(y02[y]) ** 2 for y in years}
    # DerSimonian–Laird
    w = {y: 1 / v[y] for y in years}
    sw = sum(w.values())
    fixed = sum(w[y] * d[y] for y in years) / sw
    q = sum(w[y] * (d[y] - fixed) ** 2 for y in years)
    c = sw - sum(x * x for x in w.values()) / sw
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = {y: 1 / (v[y] + tau2) for y in years}
    sws = sum(ws.values())
    a = {y: ws[y] / sws for y in years}
    theta = sum(a[y] * d[y] for y in years)
    return Offset(theta, math.sqrt(1 / sws), tau2, k, "ok"), a


def _combine(
    key: int | str,
    s01: YearStat | None,
    s02: YearStat | None,
    offset: Offset,
    weights: Mapping[int, float],
    var: dict[Key, float],
) -> Level | None:
    """一個期間（某年或最新 4 季）的綜合值；順便把用到的觀測值變異數登記到 var。"""
    parts: list[tuple[float, float, int, dict[Key, float]]] = []  # (ln 值, 近似變異數, 筆數, 係數)
    if _usable(s01):
        var[("01", key)] = median_log_se(s01) ** 2
        parts.append((math.log(s01.median), var[("01", key)], s01.n, {("01", key): 1.0}))
    if _usable(s02) and offset.status == "ok":
        var[("02", key)] = median_log_se(s02) ** 2 + offset.tau2
        terms: dict[Key, float] = {("02", key): 1.0}
        for y, a in weights.items():  # + θ̂
            terms[("01", y)] = terms.get(("01", y), 0.0) + a
            terms[("02", y)] = terms.get(("02", y), 0.0) - a
        parts.append((math.log(s02.median) + offset.log, var[("02", key)] + offset.se**2, s02.n, terms))
    if not parts:
        return None
    # 權重用近似變異數（忽略 θ̂ 和這年的相關）：權重怎麼取都不偏，真正的誤差下面用係數精確算
    inv = [1 / p[1] for p in parts]
    total = sum(inv)
    terms_out: dict[Key, float] = {}
    for (_, _, _, t), wi in zip(parts, inv, strict=True):
        for k, c in t.items():
            terms_out[k] = terms_out.get(k, 0.0) + c * wi / total
    log = sum(p[0] * wi for p, wi in zip(parts, inv, strict=True)) / total
    return Level(log, sum(p[2] for p in parts), terms_out)


def combined_series(y01: Mapping[int, YearStat], y02: Mapping[int, YearStat]) -> CombinedSeries:
    offset, weights = fit_offset(y01, y02)
    var: dict[Key, float] = {}
    # θ̂ 用到的年份，兩邊的變異數都要登記
    for y in weights:
        var[("01", y)] = median_log_se(y01[y]) ** 2
        var[("02", y)] = median_log_se(y02[y]) ** 2 + offset.tau2
    levels = {}
    for y in sorted(set(y01) | set(y02)):
        level = _combine(y, y01.get(y), y02.get(y), offset, weights, var)
        if level is not None:
            levels[y] = level
    return CombinedSeries(offset, weights, levels, var)


def combined_growth(series: CombinedSeries, start_year: int, end_year: int) -> Growth:
    if end_year <= start_year:
        raise ValueError(f"迄年必須晚於起年：{start_year} → {end_year}")
    a, b = series.levels.get(start_year), series.levels.get(end_year)
    if a is None or b is None:
        return Growth(start_year, end_year, None, None, None, None, "insufficient")
    diff = dict(b.terms)
    for k, c in a.terms.items():
        diff[k] = diff.get(k, 0.0) - c
    return growth_from_log(start_year, end_year, b.log - a.log, series.se(diff))


def combined_trend(series: CombinedSeries, last_year: int) -> LongTrend:
    """投資模式：綜合序列最近 10 年的最小平方法。區間來自殘差，和取引版本同一個算法。"""
    first_year = last_year - INVEST_WINDOW_YEARS + 1
    points = [(y, series.levels[y].log) for y in range(first_year, last_year + 1) if y in series.levels]
    return trend_from_points(first_year, last_year, points)


def combined_latest(series: CombinedSeries, s01: YearStat | None, s02: YearStat | None) -> Price:
    """最新 4 季的綜合價格（取引水準）。最新 4 季和最後一個完整年的成交有重疊，這部分的相關沒算進去。"""
    level = _combine(LATEST, s01, s02, series.offset, series.weights, series.var)
    if level is None:
        n = sum(s.n for s in (s01, s02) if s is not None)
        return Price(None, None, None, n, "insufficient")
    return price_from_log(level.log, series.se(level.terms), level.n)
