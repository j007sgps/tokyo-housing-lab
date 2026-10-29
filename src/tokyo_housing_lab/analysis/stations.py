"""分析層：車站層級的價格與走勢（M4 後半）。

每筆成交先用品質調整模型換算成典型物件（hedonic.standardize），再依最寄駅分組，三種模式沿用 trend.py：
- 購買：最近 BUY_QUARTERS 季的成交，時點修正到最新 4 季的水準後取中位數（只用 4 季的話，大部分車站筆數不夠）
- 趨勢：每年標準化價格的中位數 → growth_between
- 投資：同上 → long_term_trend（最近 10 個完整年）
門檻和區一樣是 MIN_N 筆。

- 標準化保留地點（町名、用途地域、容積率）：那是車站周邊的特性，不該扣掉
- 取引、成約一起用：成約的效果已經扣掉，都是取引的水準（區的投資模式只用取引，因為中位數沒辦法扣）
- 模型和時點修正一樣用「以最新一季為終點、每 4 季一期」＋季內走勢，只估最近 HEDONIC_LATEST_WINDOWS 期
  （車站資料從 2016 年開始，更早的不需要）。面積的效果隨期間變動（Spec.area_by_period）：車站之間小坪數的比例差很多
- 所在區的比較：同一個區、同樣方法（不限有對上車站的成交）算出的購買模式價格

已知限制：XPT001 只給最寄駅、沒有到車站的距離，所以同一站的物件不分遠近；中位數的區間沒有算進
標準化係數和時點修正係數本身的誤差（筆數夠的車站，主要的誤差來自抽樣）。
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from tokyo_housing_lab.analysis import hedonic as hd
from tokyo_housing_lab.analysis.trend import (
    Growth,
    LongTrend,
    Price,
    YearStat,
    growth_between,
    latest_price,
    long_term_trend,
)
from tokyo_housing_lab.analysis.window import WindowSample, fit_window, load_window_sample
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

BUY_QUARTERS = 12  # 購買模式：最近 3 年
# 路線的顯示名稱：國土數值情報的營運公司是正式名稱、地下鐵有「N号線」的前綴
OPERATOR_SHORT = {"東日本旅客鉄道": "JR", "東海旅客鉄道": "JR東海", "東京地下鉄": "東京メトロ", "東京都": "都営", "京浜急行電鉄": "京急", "相模鉄道": "相鉄"}
LINE_LABELS = {  # 正式名稱和一般叫法差很多的
    "ゆりかもめ／東京臨海新交通臨海線": "ゆりかもめ",
    "東京臨海高速鉄道／臨海副都心線": "りんかい線",
    "首都圏新都市鉄道／常磐新線": "つくばエクスプレス",
    "多摩都市モノレール／多摩都市モノレール線": "多摩モノレール",
    "東京モノレール／東京モノレール羽田空港線": "東京モノレール",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StationInfo:
    station_id: str
    name: str
    lines: tuple[str, ...]  # 「營運公司／路線」
    lon: float
    lat: float
    passengers: int | None
    passengers_year: int | None


@dataclass
class StationResult:
    info: StationInfo
    cities: dict[str, int] = field(default_factory=dict)  # 區 → 筆數（最寄駅相同，物件可能在不同區）
    yearly: dict[int, YearStat] = field(default_factory=dict)  # 完整年 → 標準化價格的統計
    latest: Price = field(default_factory=lambda: latest_price(None))
    invest: LongTrend | None = None

    @property
    def city(self) -> str | None:
        """成交最多的區（同數時取代碼小的）。"""
        return min(self.cities, key=lambda c: (-self.cities[c], c)) if self.cities else None


@dataclass
class StationAnalysis:
    years: list[int]  # 有車站資料的完整年
    latest_period: tuple[int, int, int, int]  # 購買模式的期間 (起年, 起季, 迄年, 迄季)
    target: str  # 時點修正的目標期（例：2025Q3〜2026Q2）
    stations: dict[str, StationResult]
    lines: dict[str, list[str]]  # 「營運公司／路線」→ 依順序的車站代號
    city_latest: dict[str, Price]  # 區 → 同方法的購買模式價格
    city_yearly: dict[str, dict[int, YearStat]]  # 區 → 同方法的每年統計（車站面板和區比較用）
    standard: str
    n: int  # 標準化、對上車站的成交筆數
    outliers: int  # 模型去掉的離群值
    unadjusted: int  # 購買模式期間內沒有時點修正係數、沒算進去的筆數

    def growth(self, station_id: str, start_year: int, end_year: int) -> Growth:
        return growth_between(self.stations[station_id].yearly, start_year, end_year)


def operator_label(key: str) -> str:
    """「東日本旅客鉄道／中央線」→「JR」、「京王電鉄／京王線」→「京王」（路線選單的分組）。"""
    if key in LINE_LABELS:
        return LINE_LABELS[key]
    operator = key.split("／", 1)[0]
    return OPERATOR_SHORT.get(operator) or re.sub(r"(電鉄|鉄道|鐵道)$", "", operator)


def line_label(key: str) -> str:
    """「東京都／12号線大江戸線」→「都営大江戸線」、「京王電鉄／京王線」→「京王線」。"""
    if key in LINE_LABELS:
        return LINE_LABELS[key]
    line = re.sub(r"^\d+号線(?=.)", "", key.split("／", 1)[1])
    short = operator_label(key)
    return line if line.startswith(short) else f"{short}{line}"


def _stat(values: np.ndarray) -> YearStat:
    med = float(np.median(values))
    return YearStat(len(values), math.exp(med), 1.4826 * float(np.median(np.abs(values - med))))


def _quarter_label(qi: int) -> str:
    return f"{qi // 4}Q{qi % 4 + 1}"


def _load_stations(store: DuckDBStore) -> tuple[dict[str, StationResult], dict[str, list[str]]]:
    stations = {
        r["station_id"]: StationResult(
            StationInfo(r["station_id"], r["name"], tuple(r["lines"]), r["lon"], r["lat"], r["passengers"], r["passengers_year"])
        )
        for r in store.query("station")
    }
    lines: dict[str, list[str]] = defaultdict(list)
    for r in store.query("station_line"):
        lines[f"{r['operator']}／{r['line']}"].append(r["station_id"])
    return stations, dict(lines)


def build_station_analysis(
    store: DuckDBStore, region: str = "13", window: tuple[WindowSample, hd.Model] | None = None
) -> StationAnalysis:
    """要先跑過 fetch-stations 和 load。估一次品質調整模型（約 10 秒）；window 可以傳入已經估好的（估價也用同一個）。"""
    ws, model = window if window is not None else (None, None)
    if ws is None:
        ws = load_window_sample(store, region)
    if not (ws.station_ids != "").any():
        raise ValueError("沒有成交對上車站，請先執行 fetch-stations 和 load")
    if model is None:
        model = fit_window(ws)
    sample, period, subperiod, station_ids, qi = ws.sample, ws.period, ws.subperiod, ws.station_ids, ws.qi
    last, ly, lq = ws.last, ws.last_year, ws.last_quarter
    std, ok = hd.standardize(model, sample, period, subperiod)

    # 購買模式：時點修正到最新 4 季（區 × 季的係數，和 time_adjustment 表同一個方法）
    city = sample.city.astype(str)
    buy = ok & (qi > last - BUY_QUARTERS)
    quarters = [(i // 4, i % 4 + 1) for i in range(last - BUY_QUARTERS + 1, last + 1)]
    factors = {
        (a.city, a.year * 4 + a.quarter - 1): a.log_factor
        for a in hd.time_adjustments(model, sorted(set(city[buy].tolist())), quarters, ly, lq)
    }
    log_factor = [factors.get(k) for k in zip(city[buy].tolist(), qi[buy].tolist(), strict=True)]
    adjusted_log = np.full(len(sample), np.nan)  # 時點修正後的標準化 ln 單價；購買模式期間以外、無法修正 → nan
    adjusted_log[buy] = std[buy] + np.array([math.nan if f is None else f for f in log_factor])
    adjusted = ~np.isnan(adjusted_log)

    stations, lines = _load_stations(store)
    on_station = ok & (station_ids != "")
    years_present = Counter((int(y), int(q)) for y, q in zip(sample.year[on_station], sample.quarter[on_station], strict=True))
    years = sorted({y for y, _ in years_present if all((y, q) in years_present for q in range(1, 5))})

    order = np.argsort(station_ids, kind="stable")
    sid_sorted = station_ids[order]
    bounds = np.flatnonzero(np.r_[True, sid_sorted[1:] != sid_sorted[:-1], True])
    for start, end in pairwise(bounds):
        sid = str(sid_sorted[start])
        rows = order[start:end]
        rows = rows[ok[rows]]
        if not sid or not len(rows):
            continue
        if sid not in stations:  # 名稱不明的點（pt:經度,緯度）不在 station 表裡
            logger.warning("成交對上的車站 %s 不在車站表裡", sid)
            continue
        r = stations[sid]
        r.cities = dict(Counter(city[rows].tolist()).most_common())
        year = sample.year[rows].astype(int)
        for y in years:
            values = std[rows][year == y]
            if len(values):
                r.yearly[y] = _stat(values)
        values = adjusted_log[rows]
        values = values[~np.isnan(values)]
        r.latest = latest_price(_stat(values) if len(values) else None)
    if years:
        for r in stations.values():
            r.invest = long_term_trend(r.yearly, max(years))

    city_latest, city_yearly = {}, {}
    year_all = sample.year.astype(int)
    for c in sorted(set(city.tolist())):
        values = adjusted_log[(city == c) & adjusted]
        city_latest[c] = latest_price(_stat(values) if len(values) else None)
        in_city = ok & (city == c)
        city_yearly[c] = {y: _stat(std[in_city & (year_all == y)]) for y in years if (in_city & (year_all == y)).any()}

    first = last - BUY_QUARTERS + 1
    return StationAnalysis(
        years, (first // 4, first % 4 + 1, ly, lq), f"{_quarter_label(last - 3)}〜{_quarter_label(last)}",
        stations, lines, city_latest, city_yearly, model.spec.standard_label(), int(on_station.sum()), model.outliers,
        int((buy & ~adjusted).sum()),
    )  # fmt: skip
