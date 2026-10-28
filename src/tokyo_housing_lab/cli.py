"""命令列入口。

  fetch：打 API，原始檔存到 data/raw/（預設東京都全域 2005Q3〜；已存在的跳過，最新 4 季重抓）
  load ：只讀 data/raw/，清理後寫進 DuckDB（不打 API）
  query：每季 ㎡ 單價中位數與筆數（只讀 DuckDB）
  analyze：各市區町村的排行（購買／趨勢／投資模式，只讀 DuckDB）
  map  ：產生地圖 HTML（只讀 DuckDB）
"""

import argparse
import logging
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from tokyo_housing_lab import DISCLAIMER
from tokyo_housing_lab.analysis.municipalities import COMBINED, INVEST_PRICE_CLASSIFICATION, build_analysis
from tokyo_housing_lab.analysis.trend import MIN_N
from tokyo_housing_lab.config import MissingApiKeyError, load_api_key
from tokyo_housing_lab.pipeline import DEFAULT_DB_PATH, DEFAULT_RAW_DIR, load_raw_dir
from tokyo_housing_lab.sources.raw_store import RawStore
from tokyo_housing_lab.sources.reinfolib_api import (
    REFRESH_RECENT_QUARTERS,
    ReinfolibAPIError,
    ReinfolibClient,
    fetch_range,
)
from tokyo_housing_lab.sources.reinfolib_records import RawFileError
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

TACHIKAWA = "13202"
TOKYO = "13"


def _quarter(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})Q([1-4])", text)
    if not m:
        raise argparse.ArgumentTypeError(f"格式要像 2021Q1：{text!r}")
    return int(m.group(1)), int(m.group(2))


def _make_client() -> ReinfolibClient:
    # 測試時換成 MockTransport 版本
    return ReinfolibClient(load_api_key())


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


PRICE_LABELS = {"01": "取引", "02": "成約"}


def _query(args: argparse.Namespace) -> None:
    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
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


NOT_ADVICE = "※ 只是過去成交資料的整理，不是價格預測，也不是投資建議。中位數會受當年成交物件的組成（新舊、大小）影響。"


def _man(yen: float) -> str:
    return f"{yen / 10000:.1f}"


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _analyze(args: argparse.Namespace) -> None:
    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    with DuckDBStore(args.db) as store:
        a = build_analysis(store, args.region)
    pc = args.price
    label = "綜合（取引＋換算後的成約，取引水準）" if pc == COMBINED else f"{PRICE_LABELS[pc]}價格"
    # 綜合的年份、最新 4 季期間都跟著取引
    base_pc = INVEST_PRICE_CLASSIFICATION if pc == COMBINED else pc
    munis = list(a.municipalities.values())
    insufficient: list[str] = []

    if args.mode == "buy":
        y1, q1, y2, q2 = a.latest_period[base_pc]
        print(f"購買模式：{label}，{y1}Q{q1}〜{y2}Q{q2} 的 ㎡ 單價中位數（萬日圓／㎡）")
        rows = [(m, m.combined.latest if pc == COMBINED else m.latest[pc]) for m in munis if pc != COMBINED or m.combined]
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
        if pc not in (INVEST_PRICE_CLASSIFICATION, COMBINED):
            raise ValueError("投資模式只能用取引價格（01）或綜合（combined）：成約價格不到 10 年")
        trend_of = (lambda m: m.combined and m.combined.invest) if pc == COMBINED else (lambda m: m.invest)
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


DEFAULT_MAP_PATH = Path("data/tokyo_map.html")


def _map(args: argparse.Namespace) -> None:
    from tokyo_housing_lab.map.build import render_html

    if not args.db.is_file():
        raise FileNotFoundError(f"找不到資料庫 {args.db}，請先執行 load")
    with DuckDBStore(args.db) as store:
        a = build_analysis(store, args.region)
    html = render_html(a, date.today(), standalone=not args.fragment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"地圖已產生：{args.out}（{len(html.encode()) // 1024} KB，{len(a.municipalities)} 個市區町村）。用瀏覽器打開即可。")


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

    load = sub.add_parser("load", help="data/raw/ → 清理 → DuckDB（不打 API）", epilog=DISCLAIMER)
    load.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    load.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    load.set_defaults(handler=_load)

    query = sub.add_parser("query", help="每季 ㎡ 單價中位數與筆數", epilog=DISCLAIMER)
    query.add_argument("--city", default=TACHIKAWA, help="市區町村代碼（預設 13202 立川市）")
    query.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    query.set_defaults(handler=_query)

    analyze = sub.add_parser("analyze", help="各市區町村的排行：購買／趨勢／投資模式", epilog=DISCLAIMER)
    analyze.add_argument("mode", choices=["buy", "trend", "invest"], help="buy＝最新價格、trend＝起訖年漲幅、invest＝長期趨勢")
    analyze.add_argument(
        "--price",
        choices=["01", "02", COMBINED],
        default="01",
        help="01＝取引價格（預設）、02＝成約價格、combined＝綜合（取引＋換算成取引水準的成約）",
    )
    analyze.add_argument("--from", dest="start", type=int, help="趨勢模式的起年（預設：最新完整年的 10 年前）")
    analyze.add_argument("--to", dest="end", type=int, help="趨勢模式的迄年（預設：最新的完整年）")
    analyze.add_argument("--region", default=TOKYO)
    analyze.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    analyze.set_defaults(handler=_analyze)

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
