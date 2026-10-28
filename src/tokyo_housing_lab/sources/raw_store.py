"""原始資料保存：API 回傳的 JSON 原文存到 data/raw/，加上一份抓取紀錄。

存成 gzip（.json.gz）：東京都一季的 JSON 約 7MB，壓縮後約 0.2MB。
gzip 是無損的，解壓後和 API 回傳的 bytes 完全相同。舊版存的 .json 也照樣讀得到。
"""

import gzip
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from tokyo_housing_lab.sources.base import QuarterQuery

FETCH_LOG_NAME = "fetch_log.jsonl"


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, query: QuarterQuery) -> Path:
        # 例：xit001_13_2024Q1_p01.json.gz
        return self.root / f"{_stem(query)}.json.gz"

    def _legacy_path(self, query: QuarterQuery) -> Path:
        # M1 時期未壓縮的檔名
        return self.root / f"{_stem(query)}.json"

    def exists(self, query: QuarterQuery) -> bool:
        return self.path_for(query).is_file() or self._legacy_path(query).is_file()

    def save(self, query: QuarterQuery, body: bytes) -> Path:
        """把 bytes 原封不動（不重新序列化）gzip 壓縮後寫入。

        先寫到暫存檔再改名，抓到一半中斷時不會留下被當成「已抓過」的殘缺檔。
        同一個查詢的舊版 .json 會刪掉，避免同一份資料有兩個檔案。
        """
        path = self.path_for(query)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(gzip.compress(body, mtime=0))
        os.replace(tmp, path)
        self._legacy_path(query).unlink(missing_ok=True)
        return path

    def append_log(self, query: QuarterQuery, http_status: int, result: str, file: Path | None) -> None:
        """每次真的打了 API 就記一行。404（該期無資料）也會記，之後能分辨「沒資料」和「沒抓」。"""
        entry = {
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "region": query.region,
            "year": query.year,
            "quarter": query.quarter,
            "price_classification": query.price_classification,
            "http_status": http_status,
            "result": result,
            "file": file.name if file else None,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / FETCH_LOG_NAME).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _stem(query: QuarterQuery) -> str:
    return f"xit001_{query.region}_{query.year}Q{query.quarter}_p{query.price_classification}"


def read_raw_bytes(path: Path) -> bytes:
    """原始檔的內容（API 回傳的 bytes）。.json.gz 解壓、.json 直接讀。"""
    data = path.read_bytes()
    return gzip.decompress(data) if path.name.endswith(".gz") else data
