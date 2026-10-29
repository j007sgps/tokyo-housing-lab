"""地圖：把分析結果和邊界資料嵌進 HTML 範本，產生一個不依賴其他檔案的 HTML。

所有數字（含趨勢模式每一組起訖年的漲幅）都在 Python 這邊算好，HTML 裡的 JavaScript 只負責查表、畫圖，
計算邏輯只有 analysis/trend.py 一份。
"""

import base64
import json
import math
from collections import Counter
from datetime import date
from importlib.resources import files
from itertools import combinations
from typing import Any

import numpy as np

from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.analysis.estimate import AGE_BANDS, KU_MAX, PLAN_GROUPS, Estimator, group_label
from tokyo_housing_lab.analysis.hazards import STATION_RADIUS_M, THRESHOLDS, HazardAnalysis
from tokyo_housing_lab.analysis.municipalities import COMBINED, HEDONIC, INVEST_PRICE_CLASSIFICATION, Analysis
from tokyo_housing_lab.analysis.stations import StationAnalysis, line_label, operator_label
from tokyo_housing_lab.processing.png import palette_png
from tokyo_housing_lab.processing.raster import simplify
from tokyo_housing_lab.sources.reinfolib_hazards import FLOOD_DEPTHS, SURGE_DEPTHS
from tokyo_housing_lab.analysis.trend import (
    INVEST_MAX_CI_WIDTH,
    growth_between,
    long_term_trend,
    INVEST_MIN_YEARS,
    INVEST_WINDOW_YEARS,
    MIN_N,
    Z95,
    LongTrend,
    Price,
)

PRICE_LABELS = {HEDONIC: "品質調整", COMBINED: "綜合", "01": "取引價格", "02": "成約價格"}
INVEST_PRICE_TYPES = [HEDONIC, COMBINED, INVEST_PRICE_CLASSIFICATION]  # 成約價格不到 10 年，投資模式不能選
BOUNDARY_CREDIT = (
    "「国土数値情報（行政区域データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/）を加工して作成"
    "（スマートニュース メディア研究所 japan-topography の簡略化データを使用）"
)
STATION_CREDIT = "「国土数値情報（駅別乗降客数データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/）を加工して作成"
LAYER_CREDITS = {
    "flood": "「国土数値情報（洪水浸水想定区域データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/）を加工して作成",
    "surge": "「国土数値情報（高潮浸水想定区域データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/）を加工して作成",
    "elementary": "「国土数値情報（小学校区データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/）を加工して作成",
    "junior": "「国土数値情報（中学校区データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/）を加工して作成",
}
# 浸水深的顏色：「水害ハザードマップ作成の手引き」的標準色（下限 → 顏色）。高潮的細分級併進同一套
DEPTH_CLASSES = [
    (0.0, "0.5m未満", (247, 245, 169)),
    (0.5, "0.5〜3m", (255, 216, 192)),
    (3.0, "3〜5m", (255, 183, 183)),
    (5.0, "5〜10m", (255, 145, 145)),
    (10.0, "10〜20m", (242, 133, 201)),
    (20.0, "20m以上", (220, 122, 220)),
]
DEPTH_ALPHA = 200
SCHOOL_TOLERANCE_DEG = 1.5e-4  # 學區的簡化（約 15m）：地圖上看不出差別，檔案小很多
NOT_ADVICE = "只是過去成交資料的整理，不是價格預測，也不是投資建議。"
LINE_MIN_STATIONS = 3  # 路線圖：至少幾站有成交，才放進選單
LINE_GROUP_ORDER = ["JR", "東京メトロ", "都営"]  # 路線選單的分組順序；其他公司依名稱
TREND_DEFAULT_SHARE = 0.8  # 車站趨勢模式的預設起年：能比較的車站數達到最多時的這個比例、最早的一年
DATA_PLACEHOLDER = "/*__MAP_DATA__*/null"


def _r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None else round(x, digits)


def _period(p: tuple[int, int, int, int]) -> str:
    return f"{p[0]}Q{p[1]}〜{p[2]}Q{p[3]}"


def _price(p: Price) -> dict[str, Any]:
    return {
        "median": None if p.median is None else round(p.median),
        "low": None if p.ci_low is None else round(p.ci_low),
        "high": None if p.ci_high is None else round(p.ci_high),
        "n": p.n,
        "status": p.status,
    }


def _invest(t: LongTrend | None) -> dict[str, Any] | None:
    if t is None:
        return None
    return {"annual": _r(t.annual), "low": _r(t.ci_low), "high": _r(t.ci_high), "years": t.years_used, "status": t.status}


def _yearly(stats: dict) -> list[list[int]]:
    return [[y, s.n, round(s.median)] for y, s in sorted(stats.items())]


def _growth_pairs(series: dict, years: list[int]) -> dict[str, list]:
    pairs = {}
    for start, end in combinations(years, 2):
        g = growth_between(series, start, end)
        if g.status != "insufficient":
            pairs[f"{start}-{end}"] = [_r(g.total), _r(g.ci_low), _r(g.ci_high), int(g.status == "uncertain")]
    return pairs


def build_station_data(sa: StationAnalysis) -> dict[str, Any]:
    """車站圖層和路線圖的數字。只放有成交的車站，和選單裡的路線上的車站（沒成交的只有名稱和位置）。"""
    with_data = {sid for sid, r in sa.stations.items() if r.cities}
    lines = []
    for key, ids in sa.lines.items():
        if sum(i in with_data for i in ids) >= LINE_MIN_STATIONS:
            lines.append({"key": key, "label": line_label(key), "group": operator_label(key), "ids": ids})

    def order(ln: dict) -> tuple:
        g = ln["group"]
        rank = LINE_GROUP_ORDER.index(g) if g in LINE_GROUP_ORDER else len(LINE_GROUP_ORDER)
        return (rank, g, -sum(i in with_data for i in ln["ids"]), ln["label"])

    lines.sort(key=order)
    needed = with_data | {i for ln in lines for i in ln["ids"]}
    stations = {}
    for sid in sorted(needed):
        r = sa.stations[sid]
        i = r.info
        s: dict[str, Any] = {
            "name": i.name,
            "lines": [line_label(k) for k in i.lines],
            "lon": round(i.lon, 5),
            "lat": round(i.lat, 5),
            "pax": i.passengers,
            "paxYear": i.passengers_year,
        }
        if r.cities:
            s |= {
                "city": r.city,
                "cities": r.cities,
                "total": sum(r.cities.values()),
                "yearly": _yearly(r.yearly),
                "latest": _price(r.latest),
                "growth": _growth_pairs(r.yearly, sa.years),
                "invest": _invest(r.invest),
            }
        stations[sid] = s
    last = max(sa.years) if sa.years else None
    # 2020 年以前只有取引（成約 2021 年起），起年太早的話大部分車站筆數不夠
    counts = {y: sum(f"{y}-{last}" in s.get("growth", {}) for s in stations.values()) for y in sa.years[:-1]}
    best = max(counts.values(), default=0)
    start = min((y for y, c in counts.items() if c >= TREND_DEFAULT_SHARE * best), default=None)
    return {
        "years": sa.years,
        "investWindow": None if last is None else [last - INVEST_WINDOW_YEARS + 1, last],
        "trendDefault": None if start is None else [start, last],
        "latestPeriod": _period(sa.latest_period),
        "target": sa.target,
        "standard": sa.standard,
        "n": sa.n,
        "lines": lines,
        "cityLatest": {c: _price(p) for c, p in sa.city_latest.items()},
        # 所在區的比較：同方法（標準化價格的中位數），不是地圖上區的品質調整指數
        "cityYearly": {c: _yearly(ys) for c, ys in sa.city_yearly.items()},
        "cityGrowth": {c: _growth_pairs(ys, sa.years) for c, ys in sa.city_yearly.items()},
        "cityInvest": {c: _invest(long_term_trend(ys, last)) if last else None for c, ys in sa.city_yearly.items()},
        "list": stations,
    }


def build_map_data(
    analysis: Analysis,
    generated: date,
    stations: StationAnalysis | None = None,
    layers: HazardAnalysis | None = None,
    estimator: Estimator | None = None,
) -> dict[str, Any]:
    """地圖要的全部數字。漲幅只放「兩年都達到樣本門檻」的組合，沒放的就是樣本不足。

    價格種類除了 "01"／"02"，還有 "combined"（綜合：取引＋換算成取引水準的成約，以取引為主軸）
    和 "hedonic"（品質調整：典型物件的價格，取引水準）。兩者的年份、最新 4 季都跟著取引。
    """
    years_of = dict(sorted(analysis.complete_years.items()))
    derived = []
    if analysis.hedonic is not None:
        derived.append(HEDONIC)
    if any(m.combined for m in analysis.municipalities.values()):
        derived.append(COMBINED)
    if derived:
        years_of = {pc: years_of[INVEST_PRICE_CLASSIFICATION] for pc in derived} | years_of
    latest_of = dict(analysis.latest_period)
    if INVEST_PRICE_CLASSIFICATION in latest_of:
        latest_of |= {pc: latest_of[INVEST_PRICE_CLASSIFICATION] for pc in derived}
    price_types = {
        pc: {
            "label": PRICE_LABELS[pc],
            "completeYears": years,
            "latestPeriod": _period(latest_of[pc]) if pc in latest_of else None,
        }
        for pc, years in years_of.items()
    }
    invest_window = None
    municipalities = {}
    for code, m in analysis.municipalities.items():
        growth: dict[str, dict[str, list]] = {}
        for pc, years in years_of.items():
            pairs = {}
            for start, end in combinations(years, 2):
                g = analysis.growth(code, pc, start, end)
                if g.status != "insufficient":
                    pairs[f"{start}-{end}"] = [_r(g.total), _r(g.ci_low), _r(g.ci_high), int(g.status == "uncertain")]
            growth[pc] = pairs
        if m.invest:
            invest_window = (m.invest.first_year, m.invest.last_year)

        yearly = {pc: [[y, s.n, round(s.median)] for y, s in sorted(ys.items())] for pc, ys in m.yearly.items()}
        latest = {pc: _price(p) for pc, p in m.latest.items()}
        invest = {INVEST_PRICE_CLASSIFICATION: _invest(m.invest)}
        if m.hedonic:
            h = m.hedonic
            yearly[HEDONIC] = [[y, h.series.n[y], round(h.series.median(y))] for y in sorted(h.series.levels)]
            latest[HEDONIC] = _price(h.latest)
            invest[HEDONIC] = _invest(h.invest)
        elif HEDONIC in years_of:
            latest[HEDONIC] = _price(Price(None, None, None, 0, "insufficient"))
        offset = None
        if m.combined:
            c = m.combined
            yearly[COMBINED] = [[y, lv.n, round(lv.median)] for y, lv in sorted(c.series.levels.items())]
            latest[COMBINED] = _price(c.latest)
            invest[COMBINED] = _invest(c.invest)
            o = c.series.offset
            offset = {"status": o.status, "years": o.years}
            if o.status == "ok":
                offset |= {
                    "pct": _r(o.pct),
                    "low": _r(math.exp(o.log - Z95 * o.se) - 1),
                    "high": _r(math.exp(o.log + Z95 * o.se) - 1),
                    "unstable": o.unstable,
                    "gaps": [[y, _r(math.exp(d) - 1)] for y, d in sorted(o.gaps.items())],
                }

        municipalities[code] = {
            "name": m.name,
            "yearly": yearly,
            "latest": latest,
            "growth": growth,
            "invest": invest,
            "offset": offset,
            "byAge": [[b.decade, b.n, round(b.median)] for b in m.by_age],
        }

    return {
        "generated": generated.isoformat(),
        "priceTypes": price_types,
        "investPriceTypes": [pc for pc in INVEST_PRICE_TYPES if pc in years_of],
        "hedonic": None if analysis.hedonic is None else {
            "standard": analysis.hedonic.standard,
            "n": analysis.hedonic.n,
            "outliers": analysis.hedonic.outliers,
            "r2": round(analysis.hedonic.r2, 3),
        },
        "investWindow": invest_window,
        "thresholds": {
            "minN": MIN_N,
            "investWindowYears": INVEST_WINDOW_YEARS,
            "investMinYears": INVEST_MIN_YEARS,
            "investMaxCiWidth": INVEST_MAX_CI_WIDTH,
        },
        "text": {
            "disclaimer": DISCLAIMER,
            "boundaryCredit": BOUNDARY_CREDIT,
            "stationCredit": STATION_CREDIT,
            "notAdvice": NOT_ADVICE,
        },
        "municipalities": municipalities,
        "stations": None if stations is None else build_station_data(stations),
        "layers": None if layers is None else build_layer_data(layers),
        "estimate": None if estimator is None else build_estimate_data(estimator),
    }


def _depth_class(lower: float) -> int:
    """浸水深的下限 → DEPTH_CLASSES 的編號（1 起算）。"""
    return max(i for i, (low, _, _) in enumerate(DEPTH_CLASSES, 1) if lower >= low)


def depth_png(kind: str, levels: np.ndarray) -> bytes:
    depths = FLOOD_DEPTHS if kind == "flood" else dict(enumerate(SURGE_DEPTHS, 1))
    lut = np.zeros(256, dtype=np.uint8)
    for level, (low, _) in depths.items():
        lut[level] = _depth_class(low)
    palette = [(0, 0, 0, 0)] + [(*rgb, DEPTH_ALPHA) for _, _, rgb in DEPTH_CLASSES]
    return palette_png(lut[levels], palette)


def encode_line(points: np.ndarray) -> list[int]:
    """經緯度 → 1e-5 度的整數，第一點是絕對值、之後是和前一點的差（JSON 小很多）。重複的點去掉。"""
    q = np.round(np.asarray(points) * 1e5).astype(np.int64)
    d = np.vstack([q[:1], np.diff(q, axis=0)])
    keep = np.r_[True, np.any(d[1:] != 0, axis=1)]
    return d[keep].ravel().tolist()


def build_layer_data(hz: HazardAnalysis) -> dict[str, Any]:
    g = hz.grids.tg.grid
    west, south, east, north = g.bounds_lonlat()
    low, high = THRESHOLDS

    def shares(sh) -> dict[str, list]:
        return {k: [_r(sh[k, low], 3), _r(sh[k, high], 3)] for k in ("flood", "surge")}

    schools: dict[str, list] = {"elementary": [], "junior": []}
    for r in hz.schools:
        schools[r["kind"]].append({
            "name": r["name"], "city": r["city"], "address": r["address"],
            "outlines": [encode_line(simplify(np.array(line), SCHOOL_TOLERANCE_DEG)) for line in json.loads(r["outlines"])],
            "polygons": [[encode_line(simplify(np.array(ring), SCHOOL_TOLERANCE_DEG)) for ring in poly]
                         for poly in json.loads(r["polygons"])],  # 滑鼠移上去用（切在圖磚邊界的各塊）
        })  # fmt: skip
    return {
        "bounds": [west, south, east, north],
        "thresholds": [low, high],
        "radius": STATION_RADIUS_M,
        "depthClasses": [{"label": label, "color": "#" + bytes(rgb).hex()} for _, label, rgb in DEPTH_CLASSES],
        "images": {
            kind: "data:image/png;base64," + base64.b64encode(depth_png(kind, hz.grids.levels[kind])).decode()
            for kind in ("flood", "surge")
        },
        "cities": {c.city: {**shares(c.shares), "schools": c.schools} for c in hz.cities},
        "stations": {
            s.station_id: {**shares(s.shares), "elementary": s.schools.get("elementary", []), "junior": s.schools.get("junior", [])}
            for s in hz.stations
            if s.shares is not None
        },
        "schools": schools,
        "credits": LAYER_CREDITS,
    }


def build_estimate_data(est: Estimator) -> dict[str, Any]:
    """估價表單用：係數表攤成 JS 好算的形式（公式見 template.html 的 estimateLog；測試用 Python 照算一遍）。"""
    t = est.table
    b, spec = t.building, t.spec
    counts = Counter(zip(est.model.sample.city.tolist(), est.model.sample.district.tolist(), strict=True)) if est.model else Counter()
    return {
        "year": t.year,
        "period": est.period_label,
        "standardArea": spec.standard_area,
        "oldSeismicLastYear": spec.old_seismic_last_year,
        "area": [b.get("ln面積", 0.0), [[math.log(k), b.get(f"ln面積>{k:g}", 0.0)] for k in spec.area_knots]],
        "age": [b.get("屋齡", 0.0), [[k, b.get(f"屋齡>{k:g}", 0.0)] for k in spec.age_knots]],
        "areaNow": b.get("面積×時間:0", 0.0),
        "oldSeismic": b.get("舊耐震", 0.0),
        "plans": {g: b.get(f"房型:{g}", 0.0) for g in PLAN_GROUPS if g == "2LDK" or g in t.plans},
        "plusS": b.get("房型:+S", 0.0),
        "structures": {st: b.get(f"構造:{st}", 0.0) for st in t.structures},
        "renovated": b.get("改裝:改装済み", 0.0),
        "cities": {
            c: {
                "now": _r(now, 6),
                "seiyaku": _r(t.seiyaku.get(c), 6),
                "districts": {d: _r(v, 6) for d, v in sorted(t.district_loc[c].items())},
                "top": max(sorted(t.district_loc[c]), key=lambda d: counts[(c, d)]),  # 表單預設：成交最多的町名
            }
            for c, now in t.city_now.items()
        },
        "ageBands": list(AGE_BANDS),
        "kuMax": KU_MAX,
        "groups": [group_label(g) for g in sorted(est.quantiles)],
        "quantiles": [[[_r(lo, 6), _r(hi, 6)] for _, (lo, hi) in sorted(est.quantiles[g].items())] for g in sorted(est.quantiles)],
        "levels": sorted(next(iter(est.quantiles.values()))),
        "coverage": {k: [_r(v, 3) for _, v in sorted(c.items())] for k, c in est.coverage.items()},
        "coverageN": est.coverage_n,
    }


def _boundaries() -> dict[str, Any]:
    return json.loads(files("tokyo_housing_lab.map").joinpath("tokyo_municipalities.geojson").read_text(encoding="utf-8"))


def _embed(obj: Any) -> str:
    # 放進 <script type="application/json">：把 < 跳脫，資料裡就算出現 </script> 也不會提早結束標籤
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


_STANDALONE_HEAD = (
    '<!doctype html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>body{margin:0}</style>\n</head>\n<body>\n"
)


def render_html(
    analysis: Analysis,
    generated: date,
    standalone: bool = True,
    stations: StationAnalysis | None = None,
    layers: HazardAnalysis | None = None,
    estimator: Estimator | None = None,
) -> str:
    """standalone=True：完整的 HTML 文件，瀏覽器直接打開。
    False：只有內容片段（給會自己包上 <html>／<head> 的地方用，例如 claude.ai 的 Artifact）。
    """
    template = files("tokyo_housing_lab.map").joinpath("template.html").read_text(encoding="utf-8")
    if template.count(DATA_PLACEHOLDER) != 1:
        raise RuntimeError("template.html 裡的資料佔位字串必須剛好一個")
    payload = {"data": build_map_data(analysis, generated, stations, layers, estimator), "boundaries": _boundaries()}
    html = template.replace(DATA_PLACEHOLDER, _embed(payload))
    return _STANDALONE_HEAD + html + "\n</body>\n</html>\n" if standalone else html
