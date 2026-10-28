"""原始資料保存：API 回傳的 JSON 原文存到 data/raw/，加上一份抓取紀錄。"""

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
        # 例：xit001_13202_2024Q1_p01.json
        name = f"xit001_{query.city}_{query.year}Q{query.quarter}_p{query.price_classification}.json"
        return self.root / name

    def exists(self, query: QuarterQuery) -> bool:
        return self.path_for(query).is_file()

    def save(self, query: QuarterQuery, body: bytes) -> Path:
        """原封不動寫入 bytes（不重新序列化）。

        先寫到暫存檔再改名，抓到一半中斷時不會留下被當成「已抓過」的殘缺檔。
        """
        path = self.path_for(query)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)
        return path

    def append_log(self, query: QuarterQuery, http_status: int, result: str, file: Path | None) -> None:
        """每次真的打了 API 就記一行。404（該期無資料）也會記，之後能分辨「沒資料」和「沒抓」。"""
        entry = {
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "city": query.city,
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
