"""命令列入口。

  fetch：打 API，原始檔存到 data/raw/（預設東京都全域 2005Q3〜；已存在的跳過，最新 4 季重抓）
  load ：只讀 data/raw/，清理後寫進 DuckDB（不打 API）
  query：每季 ㎡ 單價中位數與筆數（只讀 DuckDB）
"""

import argparse
import logging
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from tokyo_housing_lab import DISCLAIMER
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
    except (MissingApiKeyError, ReinfolibAPIError, RawFileError, FileNotFoundError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    return 0
