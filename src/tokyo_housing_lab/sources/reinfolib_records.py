"""讀取層：把 data/raw/ 的 XIT001 原始檔轉成 SourceRecord。

API 的英文欄位名在這裡換成內部欄位名，處理層就不用知道 API 的命名。
"""

import json
import logging
import re
from pathlib import Path

from tokyo_housing_lab.sources.base import QuarterQuery, SourceRecord

logger = logging.getLogger(__name__)

# API 欄位名 → 內部欄位名（29 個，依步驟 2 實際看到的全部欄位）
FIELD_MAP = {
    "PriceCategory": "price_category",
    "Type": "property_type",
    "Region": "region",
    "MunicipalityCode": "municipality_code",
    "Prefecture": "prefecture",
    "Municipality": "municipality",
    "DistrictName": "district_name",
    "DistrictCode": "district_code",
    "TradePrice": "trade_price",
    "PricePerUnit": "price_per_tsubo",  # 土地才有：每坪單價
    "FloorPlan": "floor_plan",
    "Area": "area",
    "UnitPrice": "land_unit_price",  # 土地才有：每 ㎡ 單價
    "LandShape": "land_shape",
    "Frontage": "frontage",
    "TotalFloorArea": "total_floor_area",
    "BuildingYear": "building_year",
    "Structure": "structure",
    "Use": "use",
    "Purpose": "purpose",
    "Direction": "road_direction",
    "Classification": "road_classification",
    "Breadth": "road_breadth",
    "CityPlanning": "city_planning",
    "CoverageRatio": "coverage_ratio",
    "FloorAreaRatio": "floor_area_ratio",
    "Period": "period",
    "Renovation": "renovation",
    "Remarks": "remarks",
}

_FILENAME = re.compile(r"xit001_(\d{5})_(\d{4})Q([1-4])_p(\d{2})\.json")


class RawFileError(ValueError):
    pass


def query_from_filename(name: str) -> QuarterQuery:
    """xit001_13202_2024Q1_p01.json → 請求時的查詢條件（RawStore.path_for 的反向）。"""
    m = _FILENAME.fullmatch(name)
    if not m:
        raise RawFileError(f"不是 XIT001 原始檔的檔名：{name}")
    city, year, quarter, pc = m.groups()
    return QuarterQuery(city, int(year), int(quarter), pc)


def read_raw_file(path: Path) -> list[SourceRecord]:
    """讀一個原始檔。年／季、價格種類以檔名（＝請求參數）為準，並和回應內容交叉檢查。"""
    query = query_from_filename(path.name)
    doc = json.loads(path.read_bytes())
    if doc.get("status") != "OK":
        raise RawFileError(f"{path.name}：status 不是 OK（{doc.get('status')!r}）")

    expected_period = f"{query.year}年第{query.quarter}四半期"
    records = []
    for i, row in enumerate(doc["data"]):
        missing = FIELD_MAP.keys() - row.keys()
        if missing:
            raise RawFileError(f"{path.name} 第 {i} 筆缺少欄位：{sorted(missing)}")
        unknown = row.keys() - FIELD_MAP.keys()
        if unknown:
            # API 新增了欄位：不中斷，但要讓人知道有東西沒被收進來
            logger.warning("%s 第 %d 筆有未知欄位（未收錄）：%s", path.name, i, sorted(unknown))
        if row["Period"] != expected_period:
            raise RawFileError(
                f"{path.name} 第 {i} 筆的 Period={row['Period']!r}，和請求的 {expected_period} 不一致"
            )
        records.append(
            SourceRecord(
                source="api",
                city=query.city,
                year=query.year,
                quarter=query.quarter,
                price_classification=query.price_classification,
                raw_file=path.name,
                row_index=i,
                fields={ours: row[api] for api, ours in FIELD_MAP.items()},
            )
        )
    return records
