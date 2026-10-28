"""處理層：欄位清理。規則依 docs/xit001_fields.md 的實際觀察。

數值欄位一律產生三欄：<col>_raw（原字串）、<col>（數值或 None）、<col>_status。
無法轉換的值保留原字串並標記，不丟掉。

status 的意思：
- ok：正常轉換
- empty：空字串（資料本來就沒有）
- sentinel：9999 這類看起來是「上限值」的代碼，數值當成 None
- unparsable：轉不了，例如 BuildingYear 的「戦前」
"""

import re
import unicodedata
from typing import Any, Literal, NamedTuple

from tokyo_housing_lab.sources.base import SourceRecord

ParseStatus = Literal["ok", "empty", "sentinel", "unparsable"]

# 步驟 2 在 Area、TotalFloorArea 看到 "9999"、在 Frontage 看到 "9999.9"
SENTINELS = frozenset({"9999", "9999.9"})

_INTEGER = re.compile(r"\d+")
_DECIMAL = re.compile(r"\d+(\.\d+)?")
_YEAR = re.compile(r"(\d{4})年")


class Parsed(NamedTuple):
    value: Any
    status: ParseStatus


def _normalize(raw: str) -> str:
    # NFKC：全形數字、全形英文轉半形（"１２３" → "123"、"ＲＣ" → "RC"），再去掉前後空白
    return unicodedata.normalize("NFKC", raw).strip()


def parse_int(raw: str, sentinels: frozenset[str] = frozenset()) -> Parsed:
    s = _normalize(raw)
    if s == "":
        return Parsed(None, "empty")
    if s in sentinels:
        return Parsed(None, "sentinel")
    if _INTEGER.fullmatch(s):
        return Parsed(int(s), "ok")
    return Parsed(None, "unparsable")


def parse_decimal(raw: str, sentinels: frozenset[str] = frozenset()) -> Parsed:
    s = _normalize(raw)
    if s == "":
        return Parsed(None, "empty")
    if s in sentinels:
        return Parsed(None, "sentinel")
    if _DECIMAL.fullmatch(s):
        return Parsed(float(s), "ok")
    return Parsed(None, "unparsable")


def parse_building_year(raw: str) -> Parsed:
    """實際格式是 "2022年"。"戦前" 等其他文字一律 unparsable，原字串另外保留。"""
    s = _normalize(raw)
    if s == "":
        return Parsed(None, "empty")
    m = _YEAR.fullmatch(s)
    if m:
        return Parsed(int(m.group(1)), "ok")
    return Parsed(None, "unparsable")


def normalize_text(raw: str) -> str | None:
    """文字欄位：轉半形，空字串變成 None。"""
    s = _normalize(raw)
    return s or None


def unit_price_per_m2(trade_price: Parsed, area: Parsed) -> Parsed:
    """TradePrice ÷ Area（API 的中古マンション沒有提供單價）。

    兩個都 ok 才計算；否則沿用比較嚴重的那個 status（unparsable > sentinel > empty）。
    """
    if trade_price.status == "ok" and area.status == "ok":
        if area.value <= 0:
            return Parsed(None, "unparsable")
        return Parsed(trade_price.value / area.value, "ok")
    for status in ("unparsable", "sentinel", "empty"):
        if status in (trade_price.status, area.status):
            return Parsed(None, status)
    raise AssertionError("unreachable")


# 照原樣保留的文字欄位（不轉半形，原字串就是值）
_TEXT_FIELDS = (
    "price_category",
    "property_type",
    "region",
    "prefecture",
    "municipality",
    "municipality_code",
    "district_name",
    "district_code",
    "land_shape",
    "use",
    "purpose",
    "road_direction",
    "road_classification",
    "city_planning",
    "period",
    "renovation",
    "remarks",
)

# 數值欄位：輸出欄名 → (SourceRecord 欄位名, 轉換函式)
_NUMERIC_FIELDS = {
    "trade_price": ("trade_price", parse_int),
    "area_m2": ("area", lambda s: parse_decimal(s, SENTINELS)),
    "total_floor_area_m2": ("total_floor_area", lambda s: parse_decimal(s, SENTINELS)),
    "frontage_m": ("frontage", lambda s: parse_decimal(s, SENTINELS)),
    "road_breadth_m": ("road_breadth", parse_decimal),
    "land_unit_price": ("land_unit_price", parse_int),
    "price_per_tsubo": ("price_per_tsubo", parse_int),
    "coverage_ratio": ("coverage_ratio", parse_int),
    "floor_area_ratio": ("floor_area_ratio", parse_int),
    "building_year": ("building_year", parse_building_year),
}


def clean_record(record: SourceRecord) -> dict[str, Any]:
    """一筆 SourceRecord → 一列清理後的資料（之後直接寫進 DuckDB）。"""
    f = record.fields
    row: dict[str, Any] = {
        "source": record.source,
        "city": record.city,
        "year": record.year,
        "quarter": record.quarter,
        "price_classification": record.price_classification,
        "raw_file": record.raw_file,
        "row_index": record.row_index,
    }
    for name in _TEXT_FIELDS:
        row[name] = f[name] or None

    # 全形的分類文字：保留原字串，另外存一份轉半形的（"３ＬＤＫ" → "3LDK"）
    for name in ("floor_plan", "structure"):
        row[f"{name}_raw"] = f[name]
        row[name] = normalize_text(f[name])

    parsed: dict[str, Parsed] = {}
    for out, (src, parse) in _NUMERIC_FIELDS.items():
        parsed[out] = p = parse(f[src])
        row[f"{out}_raw"] = f[src]
        row[out] = p.value
        row[f"{out}_status"] = p.status

    unit = unit_price_per_m2(parsed["trade_price"], parsed["area_m2"])
    row["unit_price_per_m2"] = unit.value
    row["unit_price_per_m2_status"] = unit.status
    return row
