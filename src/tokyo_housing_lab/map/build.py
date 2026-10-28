"""地圖：把分析結果和邊界資料嵌進 HTML 範本，產生一個不依賴其他檔案的 HTML。

所有數字（含趨勢模式每一組起訖年的漲幅）都在 Python 這邊算好，HTML 裡的 JavaScript 只負責查表、畫圖，
計算邏輯只有 analysis/trend.py 一份。
"""

import json
import math
from datetime import date
from importlib.resources import files
from itertools import combinations
from typing import Any

from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.analysis.municipalities import COMBINED, INVEST_PRICE_CLASSIFICATION, Analysis
from tokyo_housing_lab.analysis.trend import (
    INVEST_MAX_CI_WIDTH,
    INVEST_MIN_YEARS,
    INVEST_WINDOW_YEARS,
    MIN_N,
    Z95,
    LongTrend,
    Price,
)

PRICE_LABELS = {COMBINED: "綜合", "01": "取引價格", "02": "成約價格"}
INVEST_PRICE_TYPES = [COMBINED, INVEST_PRICE_CLASSIFICATION]  # 成約價格不到 10 年，投資模式不能選
BOUNDARY_CREDIT = (
    "「国土数値情報（行政区域データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/）を加工して作成"
    "（スマートニュース メディア研究所 japan-topography の簡略化データを使用）"
)
NOT_ADVICE = "只是過去成交資料的整理，不是價格預測，也不是投資建議。"
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


def build_map_data(analysis: Analysis, generated: date) -> dict[str, Any]:
    """地圖要的全部數字。漲幅只放「兩年都達到樣本門檻」的組合，沒放的就是樣本不足。

    價格種類除了 "01"／"02"，還有 "combined"（綜合：取引＋換算成取引水準的成約，以取引為主軸）。
    """
    years_of = dict(sorted(analysis.complete_years.items()))
    has_combined = any(m.combined for m in analysis.municipalities.values())
    if has_combined:
        years_of = {COMBINED: years_of[INVEST_PRICE_CLASSIFICATION]} | years_of
    latest_of = dict(analysis.latest_period)
    if has_combined and INVEST_PRICE_CLASSIFICATION in latest_of:
        latest_of[COMBINED] = latest_of[INVEST_PRICE_CLASSIFICATION]
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
        "investWindow": invest_window,
        "thresholds": {
            "minN": MIN_N,
            "investWindowYears": INVEST_WINDOW_YEARS,
            "investMinYears": INVEST_MIN_YEARS,
            "investMaxCiWidth": INVEST_MAX_CI_WIDTH,
        },
        "text": {"disclaimer": DISCLAIMER, "boundaryCredit": BOUNDARY_CREDIT, "notAdvice": NOT_ADVICE},
        "municipalities": municipalities,
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


def render_html(analysis: Analysis, generated: date, standalone: bool = True) -> str:
    """standalone=True：完整的 HTML 文件，瀏覽器直接打開。
    False：只有內容片段（給會自己包上 <html>／<head> 的地方用，例如 claude.ai 的 Artifact）。
    """
    template = files("tokyo_housing_lab.map").joinpath("template.html").read_text(encoding="utf-8")
    if template.count(DATA_PLACEHOLDER) != 1:
        raise RuntimeError("template.html 裡的資料佔位字串必須剛好一個")
    payload = {"data": build_map_data(analysis, generated), "boundaries": _boundaries()}
    html = template.replace(DATA_PLACEHOLDER, _embed(payload))
    return _STANDALONE_HEAD + html + "\n</body>\n</html>\n" if standalone else html
