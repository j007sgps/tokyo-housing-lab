"""命令列入口。

  fetch：打 API，原始檔存到 data/raw/（預設東京都全域 2005Q3〜；已存在的跳過，最新 4 季重抓）
  fetch-stations：圖磚型 API（成交的最寄駅、車站資料），原始檔存到 data/raw/tiles/
  fetch-hazards：圖磚型 API（洪水、高潮、小學校區、中學校區），原始檔存到 data/raw/tiles/
  load ：只讀 data/raw/，清理後寫進 DuckDB（不打 API）
  query：每季 ㎡ 單價中位數與筆數（只讀 DuckDB）
  analyze：各市區町村的排行（購買／趨勢／投資模式，只讀 DuckDB）
  hedonic：品質調整指數的模型診斷（係數、配適度、驗證；只讀 DuckDB）
  adjust：時點修正係數（舊成交 → 最新 4 季的水準），寫進 DuckDB 的 time_adjustment 表
  hazards：各區、各車站周邊的淹水比例（洪水、高潮）與車站所在的學區（只讀 DuckDB）
  map  ：產生地圖 HTML（只讀 DuckDB）
"""

import argparse
import logging
import math
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np

from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.analysis import hedonic as hd
from tokyo_housing_lab.analysis.hazards import STATION_RADIUS_M, THRESHOLDS, build_hazards
from tokyo_housing_lab.analysis.municipalities import COMBINED, HEDONIC, INVEST_PRICE_CLASSIFICATION, build_analysis
from tokyo_housing_lab.analysis.stations import StationAnalysis, build_station_analysis, line_label
from tokyo_housing_lab.analysis.time_adjust import basis_counts, compute_time_adjustments
from tokyo_housing_lab.analysis.trend import MIN_N
from tokyo_housing_lab.config import MissingApiKeyError, load_api_key
from tokyo_housing_lab.pipeline import (
    DEFAULT_DB_PATH,
    DEFAULT_RAW_DIR,
    load_layer_tiles,
    load_raw_dir,
    load_station_tiles,
)
from tokyo_housing_lab.sources.raw_store import RawStore
from tokyo_housing_lab.sources.reinfolib_api import (
    REFRESH_RECENT_QUARTERS,
    ReinfolibAPIError,
    ReinfolibClient,
    fetch_range,
)
from tokyo_housing_lab.sources.reinfolib_records import RawFileError
from tokyo_housing_lab.sources.reinfolib_tiles import HAZARD_LAYERS, fetch_hazard_tiles, fetch_station_tiles
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

TACHIKAWA = "13202"
TOKYO = "13"


def _quarter(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})Q([1-4])", text)
    if not m:
        raise argparse.ArgumentTypeError(f"格式要像 2021Q1：{text!r}")
    return int(m.group(1)), int(m.group(2))


def _make_client(timeout: float = 30.0) -> ReinfolibClient:
    # 測試時換成 MockTransport 版本
    return ReinfolibClient(load_api_key(), timeout=timeout)


def _fetch_stations(args: argparse.Namespace) -> None:
    # z = 11 的圖磚一年約 20MB，回應可能要 20 秒以上
    with _make_client(timeout=180.0) as client:
        counts = fetch_station_tiles(client, args.raw_dir, args.start, date.today(), force=args.force)
    print(f"抓取完成：新存檔 {counts['saved']}、無資料(404) {counts['no_data']}、已有跳過 {counts['skipped']}")


def _fetch_hazards(args: argparse.Namespace) -> None:
    # 洪水 z14 的圖磚在低地一個可以到 10MB 以上
    with _make_client(timeout=180.0) as client:
        counts = fetch_hazard_tiles(client, args.raw_dir, args.only, force=args.force)
    print(f"抓取完成：新存檔 {counts['saved']}、無資料(404) {counts['no_data']}、已有跳過 {counts['skipped']}")


def _fetch(args: argparse.Namespace) -> None:
    with _make_client() as client:
        outcomes = fetch_range(
            client,
            RawStore(args.raw_dir),
            args.region,
            args.start,
            date.today(),
            force=args.force,
            refresh_recent=args.refresh_recent,
        )
    counts = Counter(outcome for _, outcome in outcomes)
    print(f"抓取完成：新存檔 {counts['saved']}、無資料(404) {counts['no_data']}、已有跳過 {counts['skipped']}")


def _load(args: argparse.Namespace) -> None:
    with DuckDBStore(args.db) as store:
        summary = load_raw_dir(args.raw_dir, store)
    print(
        f"載入完成：{summary.files} 個原始檔、{summary.records_read} 筆 → "
        f"中古マンション等 {summary.rows_loaded} 筆寫入 {args.db}（fetch_log {summary.fetch_log_entries} 筆）"
    )
    if summary.skipped_files:
        print(f"跳過 {len(summary.skipped_files)} 個原始檔（內容已包含在東京都全域的檔案裡）")
    with DuckDBStore(args.db) as store:
        stations = load_station_tiles(args.raw_dir, store)
    if stations:
        rate = stations.matched / stations.candidates if stations.candidates else 0
        print(
            f"車站：{stations.tiles} 個圖磚檔 → 車站 {stations.stations} 站（另有名稱不明 {stations.unknown_stations} 處）、"
            f"成交點 {stations.points} 筆 → 對上成交 {stations.matched} 筆（同期間的 {rate:.1%}）"
        )
    with DuckDBStore(args.db) as store:
        layers = load_layer_tiles(args.raw_dir, store)
    if layers:
        parts = [f"{LAYER_LABELS[k]} {n} 個圖磚" for k, n in layers.tiles.items()]
        parts += [f"{LAYER_LABELS[k]}的浸水格 {n:,}" for k, n in layers.cells.items()]
        parts += [f"{LAYER_LABELS[k]} {n} 校" for k, n in layers.schools.items()]
        print("圖層：" + "、".join(parts))


PRICE_LABELS = {"01": "取引", "02": "成約"}
LAYER_LABELS = {"flood": "洪水", "surge": "高潮", "elementary": "小學校區", "junior": "中學校區"}


def _query(args: argparse.Namespace) -> None:
    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    if args.adjusted:
        _query_adjusted(args)
        return
    with DuckDBStore(args.db) as store:
        rows = store.quarterly_median(args.city)
    print(f"市區町村 {args.city}　中古マンション等　㎡ 單價中位數（日圓／㎡）")
    print(f"{'期間':<8}{'種類':<4}{'中位數':>12}{'筆數':>6}{'無法計算':>8}")
    for r in rows:
        median = f"{r.median_unit_price:,.0f}" if r.median_unit_price is not None else "-"
        print(
            f"{r.year}Q{r.quarter:<3}{PRICE_LABELS.get(r.price_classification, r.price_classification):<4}"
            f"{median:>14}{r.n:>8}{r.n_excluded:>10}"
        )
    if not rows:
        print("（沒有資料）")
    print()
    print(DISCLAIMER)


def _query_adjusted(args: argparse.Namespace) -> None:
    with DuckDBStore(args.db) as store:
        rows = store.query("quarterly_median_adjusted", city=args.city)
    target = next((r["target"] for r in rows if r["target"]), None)
    if rows and target is None:
        raise ValueError("還沒有時點修正係數，請先執行 adjust")
    print(f"市區町村 {args.city}　中古マンション等　㎡ 單價中位數（日圓／㎡）")
    print(f"修正後 = 每筆成交 × 該季的時點修正係數 ×（面積／55㎡）^面積彈性，換算成 {target} 的平均水準")
    print(f"{'期間':<8}{'種類':<4}{'中位數':>12}{'修正後':>12}{'筆數':>6}")
    for r in rows:
        median = f"{r['median_unit_price']:,.0f}"
        adjusted = "-" if r["median_adjusted"] is None else f"{r['median_adjusted']:,.0f}"
        mark = "＊" if r["basis"] == "tokyo" else ""
        print(
            f"{r['year']}Q{r['quarter']:<3}{PRICE_LABELS.get(r['price_classification'], r['price_classification']):<4}"
            f"{median:>14}{adjusted:>14}{r['n']:>8}{mark}"
        )
    if not rows:
        print("（沒有資料）")
    else:
        print("＊ 這區那一期樣本不足，用全東京的係數")
    print()
    print(NOT_ADVICE)
    print(DISCLAIMER)


def _adjust(args: argparse.Namespace) -> None:
    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    with DuckDBStore(args.db) as store:
        result = compute_time_adjustments(store, args.region)
        written = store.replace_time_adjustments(result.rows(args.region), datetime.now())
        names = {r["city"]: r["name"] for r in store.query("municipality_names", region=args.region)}
    counts = basis_counts(result)
    print(f"時點修正：換算成 {result.target} 的平均水準（{result.n:,} 筆成交估計，極端值 {result.outliers:,} 筆不用）")
    print(
        f"已寫入 time_adjustment：{written:,} 列（該區自己的指數 {counts['city']:,}、"
        f"樣本不足用全東京 {counts['tokyo']:,}、無法修正 {counts[None]:,}）"
    )
    ly = result.last[0]
    cols = [(ly - 10, 1), (ly - 5, 1), (ly - 1, 1)]
    by_key = {(a.city, a.year, a.quarter): a for a in result.adjustments}
    print("55㎡ 的成交價 × 係數 ≈ 現在的水準（＊ = 樣本不足，用全東京的係數）")
    print(f"  {'市區町村':<10}" + "".join(f"{y}Q{q:>1}".rjust(11) for y, q in cols))
    for city in sorted({a.city for a in result.adjustments}):
        cells = []
        for y, q in cols:
            a = by_key.get((city, y, q))
            cells.append("—" if a is None or a.factor is None else f"×{a.factor:.2f}{'＊' if a.basis == 'tokyo' else ''}")
        print(f"  {names.get(city, city):<10}" + "".join(c.rjust(11) for c in cells))
    # 面積彈性全東京共用：每季一個數字，舉 25㎡、80㎡ 的例子
    sample = next((by_key[(c, *yq)] for yq in cols for c in sorted({a.city for a in result.adjustments})
                   if (c, *yq) in by_key and by_key[(c, *yq)].factor is not None), None)  # fmt: skip
    if sample and sample.area_elasticity:
        e = sample.area_elasticity
        print(
            f"面積不是 55㎡ 的再乘（面積／55）^面積彈性（每季一個值，全東京共用）。例：{sample.year}Q{sample.quarter} 的彈性 {e:+.2f}"
            f" → 25㎡ 再 ×{(25 / 55) ** e:.2f}、80㎡ 再 ×{(80 / 55) ** e:.2f}"
        )
    print()
    print(NOT_ADVICE)
    print(DISCLAIMER)


NOT_ADVICE = "※ 只是過去成交資料的整理，不是價格預測，也不是投資建議。中位數會受當年成交物件的組成（新舊、大小）影響。"


def _man(yen: float) -> str:
    return f"{yen / 10000:.1f}"


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _find_line(sa: StationAnalysis, text: str) -> str:
    """--line：顯示名稱（JR中央線）或正式名稱（東日本旅客鉄道／中央線）完全相同的優先，否則部分相符要只有一條。"""
    exact = [k for k in sa.lines if text in (k, line_label(k))]
    found = exact or [k for k in sa.lines if text in k or text in line_label(k)]
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(f"找不到路線「{text}」")
    raise ValueError(f"「{text}」符合好幾條路線，請說清楚：{'、'.join(sorted(line_label(k) for k in found))}")


def _analyze_stations(args: argparse.Namespace) -> None:
    if args.price is not None:
        raise ValueError("車站一律用品質調整（典型物件），不能指定 --price")
    with DuckDBStore(args.db) as store:
        sa = build_station_analysis(store, args.region)
        names = {r["city"]: r["name"] for r in store.query("municipality_names", region=args.region)}
    line = _find_line(sa, args.line) if args.line else None
    ids = sa.lines[line] if line else [sid for sid, r in sa.stations.items() if r.cities]
    label = f"品質調整（典型物件 {sa.standard}，取引水準）"
    where = f"{line_label(line)} 各站（依路線上的順序）" if line else "各車站"

    def row(sid: str, value: str, detail: str) -> str:
        r = sa.stations[sid]
        return f"  {r.info.name:<10}{names.get(r.city, '') if r.city else '':<8}{value}  （{detail}）"

    rows: list[tuple[str, float | None, str]] = []  # (代號, 排序用的值, 輸出)
    if args.mode == "buy":
        y1, q1, y2, q2 = sa.latest_period
        print(f"購買模式：{label}，{where}，{y1}Q{q1}〜{y2}Q{q2} 換算成 {sa.target} 水準的 ㎡ 單價中位數（萬日圓／㎡）")
        for sid in ids:
            p = sa.stations[sid].latest
            ok = p.status == "ok"
            rows.append((sid, p.median if ok else None, row(sid, f"{_man(p.median):>7}", f"95% {_man(p.ci_low)}〜{_man(p.ci_high)}，{p.n} 筆") if ok else ""))
    elif args.mode == "trend":
        start = args.start or sa.years[0]
        end = args.end or sa.years[-1]
        for y in (start, end):
            if y not in sa.years:
                raise ValueError(f"車站沒有 {y} 年的完整資料（可選：{sa.years[0]}〜{sa.years[-1]}）")
        print(f"趨勢模式：{label}，{where}，{start} → {end} 年的漲幅")
        for sid in ids:
            g = sa.growth(sid, start, end)
            mark = "  ？區間包含 0，看不出漲跌" if g.status == "uncertain" else ""
            ok = g.status != "insufficient"
            rows.append((sid, g.total if ok else None,
                         row(sid, f"{_pct(g.total):>8}", f"每年 {_pct(g.annual)}，95% {_pct(g.ci_low)}〜{_pct(g.ci_high)}") + mark if ok else ""))  # fmt: skip
    else:
        print(f"投資模式：{label}，{where}，{sa.years[-1] - 9}〜{sa.years[-1]} 年的長期趨勢（年成長率）")
        for sid in ids:
            t = sa.stations[sid].invest
            ok = t is not None and t.status != "insufficient"
            mark = "  ？區間太寬，不確定" if ok and t.status == "uncertain" else ""
            rows.append((sid, t.annual if ok else None,
                         row(sid, f"{_pct(t.annual):>7}/年", f"95% {_pct(t.ci_low)}〜{_pct(t.ci_high)}，{t.years_used} 年") + mark if ok else ""))  # fmt: skip
    if not line:
        rows.sort(key=lambda r: -(r[1] if r[1] is not None else -math.inf))
    insufficient = []
    for sid, v, text in rows:
        if v is None:
            if line:
                print(f"  {sa.stations[sid].info.name:<10}{'':<8}樣本不足")
            else:
                insufficient.append(sa.stations[sid].info.name)
        else:
            print(text)
    if insufficient:
        print(f"  樣本不足（未滿 {MIN_N} 筆）：{len(insufficient)} 站")
    print("※ 依「最寄駅」分組，沒有到車站的距離；路線上的順序是依位置推算的。")
    print()
    print(NOT_ADVICE)
    print(DISCLAIMER)


def _analyze(args: argparse.Namespace) -> None:
    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    if args.by == "station":
        _analyze_stations(args)
        return
    if args.line:
        raise ValueError("--line 只能和 --by station 一起用")
    pc = args.price or "01"
    with DuckDBStore(args.db) as store:
        a = build_analysis(store, args.region, with_hedonic=pc == HEDONIC)  # 品質調整要估模型（十幾秒），用到才估
    if pc == HEDONIC and a.hedonic is None:
        raise ValueError("這個範圍的資料估不出品質調整指數（資料太少或欄位共線，詳見 warning）")
    label = {
        COMBINED: "綜合（取引＋換算後的成約，取引水準）",
        HEDONIC: f"品質調整（典型物件 {a.hedonic.standard}，取引水準）" if a.hedonic else "",
    }.get(pc) or f"{PRICE_LABELS[pc]}價格"
    # 綜合、品質調整的年份、最新 4 季期間都跟著取引
    base_pc = INVEST_PRICE_CLASSIFICATION if pc in (COMBINED, HEDONIC) else pc
    latest_of = {COMBINED: lambda m: m.combined and m.combined.latest, HEDONIC: lambda m: m.hedonic and m.hedonic.latest}
    munis = list(a.municipalities.values())
    insufficient: list[str] = []

    if args.mode == "buy":
        y1, q1, y2, q2 = a.latest_period[base_pc]
        print(f"購買模式：{label}，{y1}Q{q1}〜{y2}Q{q2} 的 ㎡ 單價中位數（萬日圓／㎡）")
        rows = [(m, latest_of[pc](m) if pc in latest_of else m.latest[pc]) for m in munis]
        rows = [(m, p) for m, p in rows if p]
        for m, p in sorted(rows, key=lambda r: -(r[1].median or 0)):
            if p.status == "insufficient":
                insufficient.append(f"{m.name}({p.n})")
                continue
            print(f"  {m.name:<10}{_man(p.median):>7}  （95% {_man(p.ci_low)}〜{_man(p.ci_high)}，{p.n} 筆）")
    elif args.mode == "trend":
        years = a.complete_years.get(base_pc, [])
        start = args.start or (years[-1] - 10 if len(years) > 10 else years[0])
        end = args.end or years[-1]
        for y in (start, end):
            if y not in years:
                raise ValueError(f"{label}沒有 {y} 年的完整資料（可選：{years[0]}〜{years[-1]}）")
        print(f"趨勢模式：{label}，{start} → {end} 年 ㎡ 單價中位數的漲幅")
        rows = [(m, a.growth(m.city, pc, start, end)) for m in munis]
        for m, g in sorted(rows, key=lambda r: -(r[1].total if r[1].total is not None else -9)):
            if g.status == "insufficient":
                insufficient.append(m.name)
                continue
            mark = "  ？區間包含 0，看不出漲跌" if g.status == "uncertain" else ""
            print(
                f"  {m.name:<10}{_pct(g.total):>8}  （每年 {_pct(g.annual)}，95% {_pct(g.ci_low)}〜{_pct(g.ci_high)}）{mark}"
            )
    else:
        if pc == "02":
            raise ValueError("投資模式只能用取引價格（01）、綜合（combined）或品質調整（hedonic）：成約價格不到 10 年")
        trend_of = {
            COMBINED: lambda m: m.combined and m.combined.invest,
            HEDONIC: lambda m: m.hedonic and m.hedonic.invest,
        }.get(pc, lambda m: m.invest)
        rows = [(m, trend_of(m)) for m in munis if trend_of(m)]
        t0 = rows[0][1]
        print(f"投資模式：{label}，{t0.first_year}〜{t0.last_year} 年的長期趨勢（年成長率）")
        for m, t in sorted(rows, key=lambda r: -(r[1].annual if r[1].annual is not None else -9)):
            if t.status == "insufficient":
                insufficient.append(f"{m.name}({t.years_used}年)")
                continue
            mark = "  ？區間太寬，不確定" if t.status == "uncertain" else ""
            print(
                f"  {m.name:<10}{_pct(t.annual):>7}/年  （95% {_pct(t.ci_low)}〜{_pct(t.ci_high)}，"
                f"{t.years_used} 年）{mark}"
            )
    if insufficient:
        print(f"  樣本不足（每年未滿 {MIN_N} 筆）：{'、'.join(insufficient)}")
    print()
    print(NOT_ADVICE)
    print(DISCLAIMER)


def _hedonic(args: argparse.Namespace) -> None:
    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    with DuckDBStore(args.db) as store:
        years = [r["year"] for r in store.query("complete_years") if r["price_classification"] == INVEST_PRICE_CLASSIFICATION]
        sample = hd.Sample.from_columns(store.query_columns("hedonic_sample", region=args.region))
    if not years:
        raise ValueError("沒有取引價格的完整年，無法估計")
    sample = sample.subset(np.isin(sample.year, years))
    period = hd.yearly_period(sample)
    m = hd.fit_hedonic(sample, period)
    f = m.fit
    print(f"品質調整指數（hedonic 迴歸）：{years[0]}〜{years[-1]} 年，{f.n:,} 筆成交（極端值 {m.outliers:,} 筆不用）")
    print(f"  R² {f.r2:.3f}、殘差 σ {f.sigma:.3f}、參數 {len(m.design.names):,} 個、町名 cluster {f.clusters:,} 個")
    if m.design.pooled_cities:
        print(f"  町名和期間分不開、町名合併成一個的區：{'、'.join(sorted(m.design.pooled_cities))}")
    print(f"典型物件：{m.spec.standard_label()}")
    base = hd.building_effect(m, m.spec.standard_area, m.spec.standard_age)
    ages = "、".join(f"{a} 年 {_pct(math.exp(hd.building_effect(m, m.spec.standard_area, a) - base) - 1)}" for a in (0, 5, 10, 20, 30, 40, 50))
    areas = "、".join(f"{a}㎡ {_pct(math.exp(hd.building_effect(m, a, m.spec.standard_age) - base) - 1)}" for a in (20, 25, 40, 70, 90, 120))
    print(f"  屋齡（和 {m.spec.standard_age:g} 年比）：{ages}")
    print(f"  面積（和 {m.spec.standard_area:g}㎡ 比，每㎡）：{areas}")
    print("係數（和基準類別比；95% 區間）：")
    for name in m.design.names:
        if name.startswith(("舊耐震", "房型:", "構造:", "改裝:", "成約×時間:")):
            b, se = m.coef(name)
            print(f"  {name:<14}{_pct(math.exp(b) - 1):>8}  （{_pct(math.exp(b - 1.96 * se) - 1)}〜{_pct(math.exp(b + 1.96 * se) - 1)}）")
    gaps = sorted(math.exp(m.coef(n)[0]) - 1 for n in m.design.names if n.startswith("成約:"))
    if gaps:
        print(f"  成約（和取引比，各區）：中位數 {_pct(gaps[len(gaps) // 2])}（{_pct(gaps[0])}〜{_pct(gaps[-1])}，{len(gaps)} 區）")
    if args.validate:
        v = hd.validate_holdout(sample, period)
        print(f"驗證（隨機留 10% 不參加估計：{v.n_test:,} 筆，無法預測 {v.unpredictable} 筆）：")
        for label, med, within in (("品質調整模型", v.model_median_error, v.model_within),
                                   ("同町名同年的中位數", v.baseline_median_error, v.baseline_within)):  # fmt: skip
            print(f"  {label:<10} 誤差中位數 {med:.1%}、±10% 以內 {within[0.1]:.1%}、±20% 以內 {within[0.2]:.1%}")
    print()
    print(NOT_ADVICE)
    print(DISCLAIMER)


DEFAULT_MAP_PATH = Path("data/tokyo_map.html")


def _map(args: argparse.Namespace) -> None:
    from tokyo_housing_lab.map.build import render_html

    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    with DuckDBStore(args.db) as store:
        a = build_analysis(store, args.region)
        try:
            stations = build_station_analysis(store, args.region)
        except ValueError as e:  # 沒有車站資料：只畫行政區
            print(f"（不畫車站：{e}）")
            stations = None
        try:
            layers = build_hazards(store)
        except ValueError as e:  # 沒有圖層資料
            print(f"（不畫淹水、學區圖層：{e}）")
            layers = None
    html = render_html(a, date.today(), standalone=not args.fragment, stations=stations, layers=layers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    n_st = "" if stations is None else f"、{sum(1 for r in stations.stations.values() if r.cities)} 個車站"
    print(f"地圖已產生：{args.out}（{len(html.encode()) // 1024} KB，{len(a.municipalities)} 個市區町村{n_st}）。用瀏覽器打開即可。")


def _hazards(args: argparse.Namespace) -> None:
    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    with DuckDBStore(args.db) as store:
        hz = build_hazards(store)
    cities, stations = hz.cities, hz.stations
    names = {c.city: c.name for c in cities}
    low, high = THRESHOLDS

    def rank(shares) -> tuple[float, float]:  # 0.5m 以上的比例，同分再看 3m 以上
        return -shares[args.sort, low], -shares[args.sort, high]

    cols = f"洪水 ≥{low:g}m／≥{high:g}m　高潮 ≥{low:g}m／≥{high:g}m"

    def pct(shares) -> str:
        return "　".join(f"{shares[k, low]:>5.0%}／{shares[k, high]:>4.0%}" for k in ("flood", "surge"))

    if args.by == "city":
        print(f"各市區町村：面積中浸水深達到門檻的比例（{cols}）")
        for c in sorted(cities, key=lambda c: rank(c.shares)):
            school = "" if c.schools["elementary"] else "  ※ 沒有學區資料"
            print(f"  {c.name:<8}{pct(c.shares)}  {c.area_km2:6.1f}km²{school}")
    else:
        rows = [s for s in stations if s.shares is not None and (args.city is None or s.city == args.city)]
        if args.city is not None and not rows:
            raise ValueError(f"找不到 {args.city} 的車站（市區町村代碼，例：13108）")
        where = names[args.city] if args.city else "東京都本島"
        print(f"{where}的車站：半徑 {STATION_RADIUS_M:g}m 內浸水深達到門檻的比例（{cols}）、車站所在的學區")
        for s in sorted(rows, key=lambda s: rank(s.shares)):
            school = "、".join(s.schools["elementary"] + s.schools["junior"]) or "（沒有學區資料）"
            print(f"  {s.name:<10}{names.get(s.city, ''):<6}{pct(s.shares)}  {school}")
        print("※ 成交只知道最寄駅，這是車站周邊的指標，不是每筆物件的；學區是車站座標所在的學區")
    print("※ 洪水是想定最大規模（不含內水氾濫）；學區是國土數值情報令和5年度（品川區、荒川區沒有資料）")
    print()
    print(DISCLAIMER)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokyo-housing-lab", epilog=DISCLAIMER)
    sub = parser.add_subparsers(dest="command")

    fetch = sub.add_parser("fetch", help="從 API 抓原始資料到 data/raw/", epilog=DISCLAIMER)
    fetch.add_argument(
        "--region",
        default=TOKYO,
        help="2 碼＝都道府縣、5 碼＝市區町村（預設 13 東京都全域；立川市是 13202）",
    )
    fetch.add_argument("--start", type=_quarter, default=(2005, 3), help="從哪一季開始（預設 2005Q3，官方最早的一季）")
    fetch.add_argument("--force", action="store_true", help="已有原始檔的季也全部重抓")
    fetch.add_argument(
        "--refresh-recent",
        type=int,
        default=REFRESH_RECENT_QUARTERS,
        help=f"有資料的最新幾季每次重抓（預設 {REFRESH_RECENT_QUARTERS}；0 = 不重抓）",
    )
    fetch.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fetch.set_defaults(handler=_fetch)

    fst = sub.add_parser(
        "fetch-stations", help="成交的最寄駅（XPT001）與車站資料（XKT015）→ data/raw/tiles/", epilog=DISCLAIMER
    )
    fst.add_argument("--start", type=int, default=2016, help="成交點從哪一年開始（預設 2016）")
    fst.add_argument("--force", action="store_true", help="已有原始檔的也全部重抓（預設只重抓今年和去年）")
    fst.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fst.set_defaults(handler=_fetch_stations)

    fhz = sub.add_parser(
        "fetch-hazards",
        help="洪水（XKT026）、高潮（XKT027）、小學校區（XKT004）、中學校區（XKT005）→ data/raw/tiles/",
        epilog=DISCLAIMER,
    )
    fhz.add_argument(
        "--only", nargs="+", choices=list(HAZARD_LAYERS), help="只抓這些圖層（預設全部；洪水約 550 個圖磚、要 30 分鐘以上）"
    )
    fhz.add_argument("--force", action="store_true", help="已有原始檔的也全部重抓（官方一年更新一次）")
    fhz.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fhz.set_defaults(handler=_fetch_hazards)

    load = sub.add_parser("load", help="data/raw/ → 清理 → DuckDB（不打 API）", epilog=DISCLAIMER)
    load.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    load.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    load.set_defaults(handler=_load)

    query = sub.add_parser("query", help="每季 ㎡ 單價中位數與筆數", epilog=DISCLAIMER)
    query.add_argument("--city", default=TACHIKAWA, help="市區町村代碼（預設 13202 立川市）")
    query.add_argument("--adjusted", action="store_true", help="多一欄時點修正後的中位數（要先執行 adjust）")
    query.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    query.set_defaults(handler=_query)

    analyze = sub.add_parser("analyze", help="各市區町村的排行：購買／趨勢／投資模式", epilog=DISCLAIMER)
    analyze.add_argument("mode", choices=["buy", "trend", "invest"], help="buy＝最新價格、trend＝起訖年漲幅、invest＝長期趨勢")
    analyze.add_argument(
        "--price",
        choices=["01", "02", COMBINED, HEDONIC],
        default=None,
        help="01＝取引價格（預設）、02＝成約價格、combined＝綜合（取引＋換算成取引水準的成約）、"
        "hedonic＝品質調整（典型物件的價格，要估模型，十幾秒）。--by station 時不能指定（一律品質調整）",
    )
    analyze.add_argument(
        "--by", choices=["city", "station"], default="city", help="city＝市區町村（預設）、station＝車站（要先 fetch-stations、load）"
    )
    analyze.add_argument("--line", help="--by station 時只看一條路線，依站的順序（例：JR中央線、大江戸線）")
    analyze.add_argument("--from", dest="start", type=int, help="趨勢模式的起年（預設：最新完整年的 10 年前）")
    analyze.add_argument("--to", dest="end", type=int, help="趨勢模式的迄年（預設：最新的完整年）")
    analyze.add_argument("--region", default=TOKYO)
    analyze.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    analyze.set_defaults(handler=_analyze)

    hed = sub.add_parser("hedonic", help="品質調整指數的模型診斷：係數、配適度、驗證", epilog=DISCLAIMER)
    hed.add_argument("--validate", action="store_true", help="另外隨機留 10%% 驗證預測誤差（再估一次模型）")
    hed.add_argument("--region", default=TOKYO)
    hed.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    hed.set_defaults(handler=_hedonic)

    adj = sub.add_parser("adjust", help="時點修正係數：舊成交 → 最新 4 季的水準，寫進 time_adjustment 表", epilog=DISCLAIMER)
    adj.add_argument("--region", default=TOKYO)
    adj.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    adj.set_defaults(handler=_adjust)

    hz = sub.add_parser("hazards", help="各區、各車站周邊的淹水比例（洪水、高潮）與學區", epilog=DISCLAIMER)
    hz.add_argument("--by", choices=["city", "station"], default="city", help="city＝市區町村（預設）、station＝車站")
    hz.add_argument("--sort", choices=["flood", "surge"], default="flood", help="依哪一種的 0.5m 以上比例排序（預設洪水）")
    hz.add_argument("--city", help="--by station 時只看這個市區町村（代碼，例：13108 江東區）")
    hz.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    hz.set_defaults(handler=_hazards)

    map_ = sub.add_parser("map", help="產生地圖 HTML（趨勢／購買／投資模式）", epilog=DISCLAIMER)
    map_.add_argument("--out", type=Path, default=DEFAULT_MAP_PATH, help=f"輸出位置（預設 {DEFAULT_MAP_PATH}）")
    map_.add_argument("--fragment", action="store_true", help="只輸出內容片段，不含 <html>／<head>")
    map_.add_argument("--region", default=TOKYO)
    map_.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    map_.set_defaults(handler=_map)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 每個請求我們自己會記一行，不用重複
    try:
        args.handler(args)
    except (MissingApiKeyError, ReinfolibAPIError, RawFileError, FileNotFoundError, ValueError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    return 0
