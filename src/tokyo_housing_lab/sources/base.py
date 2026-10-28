"""讀取層共用的型別。

統一格式 SourceRecord 要等步驟 2 看過真實欄位後才定義，這裡先只放「查詢條件」。
"""

import re
from dataclasses import dataclass
from datetime import date

PRICE_CLASSIFICATIONS = ("01", "02")  # 01＝取引價格、02＝成約價格


@dataclass(frozen=True)
class QuarterQuery:
    """對 API 的一次查詢：某城市、某年某季、某一種價格。

    年／季與價格種類都以「請求參數」為準記錄下來，不靠回應內容推回去，
    這樣每一筆資料一定帶得到期間與種類。
    """

    city: str
    year: int
    quarter: int
    price_classification: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{5}", self.city):
            raise ValueError(f"city 必須是 5 碼數字：{self.city!r}")
        if not 1 <= self.quarter <= 4:
            raise ValueError(f"quarter 必須是 1〜4：{self.quarter!r}")
        if self.price_classification not in PRICE_CLASSIFICATIONS:
            raise ValueError(f"price_classification 必須是 01 或 02：{self.price_classification!r}")

    @property
    def label(self) -> str:
        return f"{self.city} {self.year}Q{self.quarter} p{self.price_classification}"


def quarter_of(day: date) -> tuple[int, int]:
    return day.year, (day.month - 1) // 3 + 1


def quarters_between(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """start 到 end（兩端都含）的每一季，例如 (2021, 1) 〜 (2021, 3)。"""
    year, quarter = start
    result = []
    while (year, quarter) <= end:
        result.append((year, quarter))
        year, quarter = (year + 1, 1) if quarter == 4 else (year, quarter + 1)
    return result
