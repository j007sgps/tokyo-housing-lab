"""分析層：從 DuckDB 撈出各市區町村的數字，算好三種模式的結果。"""

from collections import defaultdict
from dataclasses import dataclass, field

from tokyo_housing_lab.analysis.combined import (
    CombinedSeries,
    combined_growth,
    combined_latest,
    combined_series,
    combined_trend,
)
from tokyo_housing_lab.analysis.trend import (
    Growth,
    LongTrend,
    Price,
    YearStat,
    growth_between,
    latest_price,
    long_term_trend,
)
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

COMBINED = "combined"  # 綜合指標（取引＋換算後的成約），當成第三種「價格種類」用
INVEST_PRICE_CLASSIFICATION = "01"  # 投資模式只用取引價格：成約價格 2021 年才開始，不夠 10 年
AGE_BREAKDOWN_YEARS = 5  # 詳細面板的建築年代別中位數：最新完整年往前看幾年


@dataclass(frozen=True)
class AgeBucket:
    decade: int  # 1980 = 1980〜1989 年建
    n: int
    median: float


@dataclass
class Combined:
    series: CombinedSeries
    latest: Price
    invest: LongTrend


@dataclass
class Municipality:
    city: str
    name: str
    yearly: dict[str, dict[int, YearStat]] = field(default_factory=dict)  # 價格種類 → 年 → 統計
    latest: dict[str, Price] = field(default_factory=dict)  # 價格種類 → 最新 4 季
    invest: LongTrend | None = None
    by_age: list[AgeBucket] = field(default_factory=list)
    combined: Combined | None = None


@dataclass
class Analysis:
    region: str
    complete_years: dict[str, list[int]]  # 價格種類 → 4 季都有資料的年
    latest_period: dict[str, tuple[int, int, int, int]]  # 價格種類 → (起年, 起季, 迄年, 迄季)
    municipalities: dict[str, Municipality]

    def growth(self, city: str, price_classification: str, start_year: int, end_year: int) -> Growth:
        m = self.municipalities[city]
        if price_classification == COMBINED:
            return combined_growth(m.combined.series, start_year, end_year)
        series = m.yearly.get(price_classification, {})
        return growth_between(series, start_year, end_year)


def _quarter(qi: int) -> tuple[int, int]:
    return qi // 4, qi % 4 + 1


def build_analysis(store: DuckDBStore, region: str = "13") -> Analysis:
    # 有資料的價格種類一定有 key（沒有完整年就是空 list），地圖靠這個決定能不能選
    complete: dict[str, list[int]] = defaultdict(list)
    for row in store.query("complete_years"):
        complete[row["price_classification"]].append(row["year"])

    munis: dict[str, Municipality] = {}

    def muni(row: dict) -> Municipality:
        return munis.setdefault(row["city"], Municipality(row["city"], row["name"]))

    # 年度表只放完整年：只有 1、2 季的年（例如 2005、最新一年）中位數會偏，不拿來比
    for row in store.query("municipality_yearly", region=region):
        if row["year"] in complete[row["price_classification"]]:  # 也順便替這個價格種類建立 key
            m = muni(row)
            m.yearly.setdefault(row["price_classification"], {})[row["year"]] = YearStat(
                row["n"], row["median_unit_price"], row["spread_log"]
            )

    latest_period = {}
    latest_stats: dict[tuple[str, str], YearStat] = {}
    for row in store.query("municipality_latest", region=region):
        m = muni(row)
        stat = YearStat(row["n"], row["median_unit_price"], row["spread_log"])
        latest_stats[(row["city"], row["price_classification"])] = stat
        m.latest[row["price_classification"]] = latest_price(stat)
        latest_period[row["price_classification"]] = (*_quarter(row["first_qi"]), *_quarter(row["last_qi"]))

    invest_years = complete.get(INVEST_PRICE_CLASSIFICATION, [])
    if invest_years:
        last = max(invest_years)
        for m in munis.values():
            m.invest = long_term_trend(m.yearly.get(INVEST_PRICE_CLASSIFICATION, {}), last)
        for row in store.query("municipality_by_age", region=region, last_year=last, years=AGE_BREAKDOWN_YEARS):
            if row["city"] in munis:
                munis[row["city"]].by_age.append(AgeBucket(row["decade"], row["n"], row["median_unit_price"]))

    # 最新 4 季沒有成交的區，購買模式也要有一筆「樣本不足」
    for m in munis.values():
        for pc in latest_period:
            m.latest.setdefault(pc, latest_price(None))

    # 綜合指標：以取引為主軸（年份、投資模式的 10 年都跟取引），沒有取引的完整年就不算
    if invest_years:
        invest_last = max(invest_years)
        for m in munis.values():
            series = combined_series(m.yearly.get("01", {}), m.yearly.get("02", {}))
            latest = combined_latest(series, latest_stats.get((m.city, "01")), latest_stats.get((m.city, "02")))
            m.combined = Combined(series, latest, combined_trend(series, invest_last))

    return Analysis(region, dict(complete), latest_period, dict(sorted(munis.items())))
