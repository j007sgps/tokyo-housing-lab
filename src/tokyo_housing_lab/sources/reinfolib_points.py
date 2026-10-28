"""讀取層：解析圖磚型 API 的 GeoJSON。

- XPT001（成交點）→ StationPoint：成交的鍵（季、價格種類、町丁目、總價、面積、建築年、房型、構造）＋最寄駅的座標
  欄位是給人看的字串（「4,800万円」「60㎡」「2024年第1四半期」），這裡轉成數值，才能和 XIT001 的資料對上
- XKT015（駅別乗降客数）→ StationSegment：每條路線的一段月台
"""

import json
import re
import unicodedata
from dataclasses import dataclass

PRICE_CATEGORY = {"不動産取引価格情報": "01", "成約価格情報": "02"}
_PERIOD = re.compile(r"(\d{4})年第([1-4])四半期")
_PRICE = re.compile(r"(?:(\d+)億)?(?:(\d+)万)?円")


class PointFormatError(ValueError):
    """欄位的格式和預期不同（不猜，直接報錯）。"""


@dataclass(frozen=True)
class StationPoint:
    lon: float
    lat: float
    city: str
    district_code: str
    year: int
    quarter: int
    price_classification: str
    trade_price: int | None  # 日圓；格式不明 → None（對不上 XIT001，不影響其他筆）
    area: float | None  # ㎡
    building_year: int | None = None  # 「2016年」→ 2016；空白、「戦前」→ None
    floor_plan: str = ""  # NFKC 正規化（「２ＬＤＫ」→「2LDK」），和 XIT001 清理後的格式相同
    structure: str = ""


@dataclass(frozen=True)
class StationSegment:
    group: str  # S12_001g：同一車站（可跨路線）共用的代碼
    code: str  # S12_001c：這一段（車站 × 路線）的代碼
    name: str
    operator: str
    line: str
    lon: float  # 月台線段的中心
    lat: float
    passengers: dict[int, int]  # 年 → 乘降人數（同一車站只有其中一段有數字，其他段是 0）


def parse_price(text: str) -> int | None:
    """「4,800万円」「1億2,000万円」→ 日圓。"""
    m = _PRICE.fullmatch(unicodedata.normalize("NFKC", text).replace(",", ""))
    if not m or not (m.group(1) or m.group(2)):
        return None
    return int(m.group(1) or 0) * 10**8 + int(m.group(2) or 0) * 10**4


def parse_area(text: str) -> float | None:
    """「60㎡」→ 60.0（NFKC 會把 ㎡ 變成 m2）；「2,000㎡以上」之類的不是確切數字 → None。"""
    m = re.fullmatch(r"([\d.]+)m2", unicodedata.normalize("NFKC", text).replace(",", ""))
    return float(m.group(1)) if m else None


def parse_building_year(text: str) -> int | None:
    m = re.fullmatch(r"(\d{4})年", unicodedata.normalize("NFKC", text))
    return int(m.group(1)) if m else None


def parse_period(text: str) -> tuple[int, int]:
    m = _PERIOD.fullmatch(text)
    if not m:
        raise PointFormatError(f"期間的格式不明：{text!r}")
    return int(m.group(1)), int(m.group(2))


def parse_xpt001(body: bytes, city_prefix: str = "13") -> list[StationPoint]:
    """只留 city_prefix 開頭的市區町村（圖磚裡也有鄰縣的成交）。"""
    out = []
    for f in json.loads(body)["features"]:
        p = f["properties"]
        if not p["city_code"].startswith(city_prefix):
            continue
        if f["geometry"]["type"] != "Point":
            raise PointFormatError(f"成交點的幾何不是 Point：{f['geometry']['type']}")
        category = p["price_information_category_name_ja"]
        if category not in PRICE_CATEGORY:
            raise PointFormatError(f"價格種類不明：{category!r}")
        lon, lat = f["geometry"]["coordinates"][:2]
        year, quarter = parse_period(p["point_in_time_name_ja"])
        out.append(
            StationPoint(
                float(lon), float(lat), p["city_code"], p["district_code"], year, quarter,
                PRICE_CATEGORY[category], parse_price(p["u_transaction_price_total_ja"]), parse_area(p["u_area_ja"]),
                parse_building_year(p.get("u_construction_year_ja", "")),
                unicodedata.normalize("NFKC", p.get("floor_plan_name_ja", "")),
                unicodedata.normalize("NFKC", p.get("building_structure_name_ja", "")),
            )
        )  # fmt: skip
    return out


def parse_xkt015(body: bytes) -> list[StationSegment]:
    out = []
    for f in json.loads(body)["features"]:
        p = f["properties"]
        coords = f["geometry"]["coordinates"]
        if f["geometry"]["type"] == "MultiLineString":
            coords = [c for part in coords for c in part]
        elif f["geometry"]["type"] != "LineString":
            raise PointFormatError(f"車站的幾何不是 LineString：{f['geometry']['type']}")
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        passengers = {
            int(k[:4]): int(v) for k, v in p.items() if re.fullmatch(r"\d{4}_passenger_count", k) and v
        }
        out.append(
            StationSegment(
                str(p["S12_001g"]), str(p["S12_001c"]), p["S12_001_ja"], p["S12_002_ja"], p["S12_003_ja"],
                lon, lat, passengers,
            )
        )  # fmt: skip
    return out
