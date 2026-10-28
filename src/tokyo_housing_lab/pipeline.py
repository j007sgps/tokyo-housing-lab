"""把各層串起來：原始檔 →（讀取層）SourceRecord →（處理層）篩選、清理 →（儲存）DuckDB。"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tokyo_housing_lab.processing.clean import clean_record
from tokyo_housing_lab.processing.filter import is_used_condo
from tokyo_housing_lab.sources.raw_store import FETCH_LOG_NAME
from tokyo_housing_lab.sources.reinfolib_records import read_raw_file
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_DB_PATH = Path("data/tokyo_housing_lab.duckdb")


@dataclass(frozen=True)
class LoadSummary:
    files: int
    records_read: int
    rows_loaded: int  # 篩選後（中古マンション等）寫進 transactions 的筆數
    fetch_log_entries: int


def load_raw_dir(raw_dir: Path, store: DuckDBStore) -> LoadSummary:
    """data/raw 底下的全部原始檔重新載入。只讀檔案，不打 API。

    任何一個檔案有問題（格式不對、Period 對不上…）就整批不寫入，資料庫維持原狀。
    """
    paths = sorted(raw_dir.glob("xit001_*.json"))
    rows_by_file = {}
    records_read = 0
    for path in paths:
        records = read_raw_file(path)
        records_read += len(records)
        rows_by_file[path.name] = [clean_record(r) for r in records if is_used_condo(r)]
        logger.info("%s：%d 筆，中古マンション等 %d 筆", path.name, len(records), len(rows_by_file[path.name]))

    loaded_at = datetime.now(UTC).replace(tzinfo=None)  # DuckDB TIMESTAMP 存 UTC
    rows_loaded = store.replace_files(rows_by_file, loaded_at, prune_others=True)
    log_entries = store.replace_fetch_log(raw_dir / FETCH_LOG_NAME)
    return LoadSummary(len(paths), records_read, rows_loaded, log_entries)
