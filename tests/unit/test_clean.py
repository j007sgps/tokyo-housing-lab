import pytest

from tokyo_housing_lab.processing.clean import (
    SENTINELS,
    Parsed,
    clean_record,
    normalize_text,
    parse_building_year,
    parse_decimal,
    parse_int,
    unit_price_per_m2,
)
from tokyo_housing_lab.sources.base import SourceRecord
from tokyo_housing_lab.sources.reinfolib_records import FIELD_MAP


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 正常值
        ("11000000", Parsed(11000000, "ok")),
        ("80", Parsed(80, "ok")),
        # 邊界值
        ("0", Parsed(0, "ok")),
        ("2500000000", Parsed(2500000000, "ok")),  # 實際資料最大的成交價
        ("99999999999999", Parsed(99999999999999, "ok")),
        ("", Parsed(None, "empty")),
        ("  ", Parsed(None, "empty")),
        (" 80 ", Parsed(80, "ok")),
        ("１２３", Parsed(123, "ok")),  # 全形數字
        # 異常值
        ("12.5", Parsed(None, "unparsable")),
        ("-5", Parsed(None, "unparsable")),
        ("1,000", Parsed(None, "unparsable")),
        ("2000以上", Parsed(None, "unparsable")),
        ("戦前", Parsed(None, "unparsable")),
        ("abc", Parsed(None, "unparsable")),
        ("�", Parsed(None, "unparsable")),  # 亂碼
    ],
)
def test_parse_int(raw, expected):
    assert parse_int(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("65", Parsed(65.0, "ok")),
        ("3.6", Parsed(3.6, "ok")),
        ("0.8", Parsed(0.8, "ok")),  # 實際資料最小的道路寬度
        ("0", Parsed(0.0, "ok")),
        ("", Parsed(None, "empty")),
        ("１．５", Parsed(1.5, "ok")),  # 全形
        ("9999", Parsed(None, "sentinel")),
        ("9999.9", Parsed(None, "sentinel")),
        ("9998", Parsed(9998.0, "ok")),  # 只有指定的代碼才算 sentinel
        ("1.", Parsed(None, "unparsable")),
        (".5", Parsed(None, "unparsable")),
        ("1.2.3", Parsed(None, "unparsable")),
        ("2000㎡以上", Parsed(None, "unparsable")),
        ("50〜60", Parsed(None, "unparsable")),
    ],
)
def test_parse_decimal_with_sentinels(raw, expected):
    assert parse_decimal(raw, SENTINELS) == expected


def test_sentinel_only_applies_when_asked():
    assert parse_decimal("9999") == Parsed(9999.0, "ok")
    assert parse_int("9999") == Parsed(9999, "ok")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2022年", Parsed(2022, "ok")),
        ("1966年", Parsed(1966, "ok")),
        ("２０２２年", Parsed(2022, "ok")),
        ("", Parsed(None, "empty")),
        ("戦前", Parsed(None, "unparsable")),
        ("2022", Parsed(None, "unparsable")),  # 實際格式一定有「年」，沒有就當成沒看過的格式
        ("平成5年", Parsed(None, "unparsable")),
        ("22年", Parsed(None, "unparsable")),
        ("2022年頃", Parsed(None, "unparsable")),
    ],
)
def test_parse_building_year(raw, expected):
    assert parse_building_year(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("３ＬＤＫ", "3LDK"),
        ("２ＬＤＫ＋Ｓ", "2LDK+S"),
        ("ＲＣ、鉄骨造", "RC、鉄骨造"),
        ("オープンフロア", "オープンフロア"),
        ("", None),
        (" ", None),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    ("price", "area", "expected"),
    [
        (Parsed(11000000, "ok"), Parsed(40.0, "ok"), Parsed(275000.0, "ok")),
        (Parsed(0, "ok"), Parsed(40.0, "ok"), Parsed(0.0, "ok")),
        (Parsed(11000000, "ok"), Parsed(0.0, "ok"), Parsed(None, "unparsable")),
        (Parsed(11000000, "ok"), Parsed(None, "empty"), Parsed(None, "empty")),
        (Parsed(11000000, "ok"), Parsed(None, "sentinel"), Parsed(None, "sentinel")),
        (Parsed(None, "unparsable"), Parsed(None, "sentinel"), Parsed(None, "unparsable")),
        (Parsed(None, "empty"), Parsed(None, "sentinel"), Parsed(None, "sentinel")),
    ],
)
def test_unit_price_per_m2(price, area, expected):
    assert unit_price_per_m2(price, area) == expected


def make_record(**overrides) -> SourceRecord:
    fields = dict.fromkeys(FIELD_MAP.values(), "")
    fields.update(
        price_category="不動産取引価格情報",
        property_type="中古マンション等",
        trade_price="11000000",
        area="40",
        building_year="2022年",
        floor_plan="２ＤＫ",
        structure="ＲＣ",
        period="2024年第1四半期",
    )
    fields.update(overrides)
    return SourceRecord("api", "13202", 2024, 1, "01", "xit001_13202_2024Q1_p01.json", 7, fields)


def test_clean_record_normal():
    row = clean_record(make_record())
    assert row["year"] == 2024 and row["quarter"] == 1 and row["price_classification"] == "01"
    assert row["raw_file"] == "xit001_13202_2024Q1_p01.json" and row["row_index"] == 7
    assert (row["trade_price"], row["trade_price_status"], row["trade_price_raw"]) == (11000000, "ok", "11000000")
    assert (row["area_m2"], row["area_m2_status"]) == (40.0, "ok")
    assert (row["unit_price_per_m2"], row["unit_price_per_m2_status"]) == (275000.0, "ok")
    assert (row["building_year"], row["building_year_status"]) == (2022, "ok")
    assert (row["floor_plan"], row["floor_plan_raw"]) == ("2DK", "２ＤＫ")
    assert (row["structure"], row["structure_raw"]) == ("RC", "ＲＣ")
    assert row["purpose"] is None  # 空字串 → None
    assert row["property_type"] == "中古マンション等"


def test_clean_record_keeps_raw_when_unparsable():
    row = clean_record(make_record(building_year="戦前", area="9999"))
    assert (row["building_year"], row["building_year_status"], row["building_year_raw"]) == (None, "unparsable", "戦前")
    assert (row["area_m2"], row["area_m2_status"], row["area_m2_raw"]) == (None, "sentinel", "9999")
    assert (row["unit_price_per_m2"], row["unit_price_per_m2_status"]) == (None, "sentinel")
    assert row["trade_price"] == 11000000  # 其他欄位不受影響


def test_clean_record_every_numeric_has_three_columns():
    row = clean_record(make_record())
    numeric = [k[: -len("_status")] for k in row if k.endswith("_status")]
    assert "trade_price" in numeric and "building_year" in numeric
    for name in numeric:
        if name != "unit_price_per_m2":  # 計算出來的欄位沒有原字串
            assert f"{name}_raw" in row
        assert name in row
