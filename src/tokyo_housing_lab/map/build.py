"""地圖：把分析結果和邊界資料嵌進 HTML 範本，產生一個不依賴其他檔案的 HTML。

所有數字（含趨勢模式每一組起訖年的漲幅）都在 Python 這邊算好，HTML 裡的 JavaScript 只負責查表、畫圖，
計算邏輯只有 analysis/trend.py 一份。
"""

import json
from datetime import date
from importlib.resources import files
from itertools import combinations
from typing import Any

from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.analysis.municipalities import INVEST_PRICE_CLASSIFICATION, Analysis
from tokyo_housing_lab.analysis.trend import (
    INVEST_MAX_CI_WIDTH,
    INVEST_MIN_YEARS,
    INVEST_WINDOW_YEARS,
    MIN_N,
    growth_between,
)

PRICE_LABELS = {"01": "取引價格", "02": "成約價格"}
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


def build_map_data(analysis: Analysis, generated: date) -> dict[str, Any]:
    """地圖要的全部數字。漲幅只放「兩年都達到樣本門檻」的組合，沒放的就是樣本不足。"""
    price_types = {
        pc: {
            "label": PRICE_LABELS[pc],
            "completeYears": years,
            "latestPeriod": _period(analysis.latest_period[pc]) if pc in analysis.latest_period else None,
        }
        for pc, years in sorted(analysis.complete_years.items())
    }
    invest_window = None
    municipalities = {}
    for code, m in analysis.municipalities.items():
        growth: dict[str, dict[str, list]] = {}
        for pc, years in analysis.complete_years.items():
            series = m.yearly.get(pc, {})
            pairs = {}
            for start, end in combinations(years, 2):
                g = growth_between(series, start, end)
                if g.status != "insufficient":
                    pairs[f"{start}-{end}"] = [_r(g.total), _r(g.ci_low), _r(g.ci_high), int(g.status == "uncertain")]
            growth[pc] = pairs

        invest = None
        if m.invest:
            invest_window = (m.invest.first_year, m.invest.last_year)
            invest = {
                "annual": _r(m.invest.annual),
                "low": _r(m.invest.ci_low),
                "high": _r(m.invest.ci_high),
                "years": m.invest.years_used,
                "status": m.invest.status,
            }

        municipalities[code] = {
            "name": m.name,
            "yearly": {pc: [[y, s.n, round(s.median)] for y, s in sorted(ys.items())] for pc, ys in m.yearly.items()},
            "latest": {
                pc: {
                    "median": None if p.median is None else round(p.median),
                    "low": None if p.ci_low is None else round(p.ci_low),
                    "high": None if p.ci_high is None else round(p.ci_high),
                    "n": p.n,
                    "status": p.status,
                }
                for pc, p in m.latest.items()
            },
            "growth": growth,
            "invest": invest,
            "byAge": [[b.decade, b.n, round(b.median)] for b in m.by_age],
        }

    return {
        "generated": generated.isoformat(),
        "priceTypes": price_types,
        "investPriceType": INVEST_PRICE_CLASSIFICATION,
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
