"""把各層串起來：原始檔 →（讀取層）SourceRecord →（處理層）篩選、清理 →（儲存）DuckDB。"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tokyo_housing_lab.processing.clean import clean_record
from tokyo_housing_lab.processing.filter import is_used_condo
from tokyo_housing_lab.sources.raw_store import FETCH_LOG_NAME
from tokyo_housing_lab.sources.reinfolib_records import query_from_filename, read_raw_file
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_DB_PATH = Path("data/tokyo_housing_lab.duckdb")


@dataclass(frozen=True)
class LoadSummary:
    files: int  # 實際載入的原始檔數
    records_read: int
    rows_loaded: int  # 篩選後（中古マンション等）寫進 transactions 的筆數
    fetch_log_entries: int
    skipped_files: tuple[str, ...] = ()  # 內容已被其他檔案涵蓋而跳過的原始檔


def select_raw_files(raw_dir: Path) -> tuple[list[Path], list[Path]]:
    """決定要載入哪些原始檔，回傳 (要載入的, 跳過的)。

    - 某一季、某種價格已經有都道府縣（area，例如 13）的檔案時，同一季的市區町村（city）檔案
      內容已經包含在裡面，跳過，避免同一筆成交算兩次（M1 的立川市檔案就是這種情況）
    - 同一個查詢同時有 .json 和 .json.gz 時，用 .json.gz
    """
    paths = sorted(p for p in raw_dir.glob("xit001_*") if p.name.endswith((".json", ".json.gz")))
    by_query = {}
    for path in paths:
        q = query_from_filename(path.name)
        if q not in by_query or path.name.endswith(".gz"):
            by_query[q] = path
    areas = {(q.region, q.year, q.quarter, q.price_classification) for q in by_query if len(q.region) == 2}
    selected, skipped = [], [p for p in paths if p not in by_query.values()]
    for q, path in sorted(by_query.items(), key=lambda kv: kv[1].name):
        if len(q.region) == 5 and (q.region[:2], q.year, q.quarter, q.price_classification) in areas:
            skipped.append(path)
        else:
            selected.append(path)
    for path in skipped:
        logger.warning("%s 的內容已包含在其他原始檔裡，跳過", path.name)
    return selected, sorted(skipped)


def load_raw_dir(raw_dir: Path, store: DuckDBStore) -> LoadSummary:
    """data/raw 底下的全部原始檔重新載入。只讀檔案，不打 API。

    任何一個檔案有問題（格式不對、Period 對不上…）就整批不寫入，資料庫維持原狀。
    一次處理一個檔案、邊讀邊寫暫存檔，東京都全域的資料量也不會吃光記憶體。
    """
    paths, skipped = select_raw_files(raw_dir)
    records_read = 0

    def rows_by_file():
        nonlocal records_read
        for path in paths:
            records = read_raw_file(path)
            records_read += len(records)
            rows = [clean_record(r) for r in records if is_used_condo(r)]
            logger.info("%s：%d 筆，中古マンション等 %d 筆", path.name, len(records), len(rows))
            yield path.name, rows

    loaded_at = datetime.now(UTC).replace(tzinfo=None)  # DuckDB TIMESTAMP 存 UTC
    rows_loaded = store.replace_files(rows_by_file(), loaded_at, prune_others=True)
    log_entries = store.replace_fetch_log(raw_dir / FETCH_LOG_NAME)
    return LoadSummary(len(paths), records_read, rows_loaded, log_entries, tuple(p.name for p in skipped))
