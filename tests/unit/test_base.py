from datetime import date

import pytest

from tokyo_housing_lab.sources.base import QuarterQuery, quarter_of, quarters_between


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 1, 1), (2026, 1)),
        (date(2026, 3, 31), (2026, 1)),
        (date(2026, 4, 1), (2026, 2)),
        (date(2026, 10, 28), (2026, 4)),
        (date(2026, 12, 31), (2026, 4)),
    ],
)
def test_quarter_of(day, expected):
    assert quarter_of(day) == expected


def test_quarters_between_crosses_year():
    assert quarters_between((2021, 3), (2022, 2)) == [(2021, 3), (2021, 4), (2022, 1), (2022, 2)]


def test_quarters_between_single():
    assert quarters_between((2021, 1), (2021, 1)) == [(2021, 1)]


def test_quarters_between_end_before_start_is_empty():
    assert quarters_between((2021, 2), (2021, 1)) == []


def test_m1_range_count():
    # 2021Q1〜2026Q4 = 6 年 × 4 季
    assert len(quarters_between((2021, 1), quarter_of(date(2026, 10, 28)))) == 24


def test_query_label():
    assert QuarterQuery("13202", 2024, 1, "01").label == "13202 2024Q1 p01"


@pytest.mark.parametrize(
    ("city", "quarter", "pc"),
    [
        ("1320", 1, "01"),  # 4 碼
        ("13202a", 1, "01"),
        ("13202", 0, "01"),
        ("13202", 5, "01"),
        ("13202", 1, "1"),
        ("13202", 1, "03"),
        ("13202", 1, ""),
    ],
)
def test_query_rejects_invalid(city, quarter, pc):
    with pytest.raises(ValueError):
        QuarterQuery(city, 2024, quarter, pc)
