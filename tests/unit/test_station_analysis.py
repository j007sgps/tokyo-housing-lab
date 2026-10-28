import math
from datetime import datetime

import numpy as np
import pytest
from test_municipalities import rows_for

from tokyo_housing_lab.analysis.stations import build_station_analysis, line_label, operator_label
from tokyo_housing_lab.analysis.trend import MIN_N
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore

# 新宿区：A 站在西新宿（貴）、B 站在大久保（便宜）；早稲田的成交沒對上車站
DISTRICT = {"西新宿": 0.3, "大久保": -0.2, "早稲田": 0.0}
STATION = {"西新宿": "A", "大久保": "B"}
TREND = 0.05  # 每年 +5%（ln）
QUARTERS = [(y, q) for y in range(2016, 2026) for q in (1, 2, 3, 4)] + [(2026, 1)]


def true_log(district, year, age):
    return 13.5 + DISTRICT[district] + TREND * (year - 2016) - 0.01 * age


def build_station_db(path):
    """A 站賣出的物件越來越舊（屋齡 5 → 30 年）：中位數漲得比真的少，標準化後要還原。"""
    rng = np.random.default_rng(11)
    files: dict[str, list] = {}
    for year, q in QUARTERS:
        for district in DISTRICT:
            for _ in range(10):
                if district == "西新宿":
                    age = int(5 + 25 * (year - 2016) / 10 + rng.integers(0, 4))
                else:
                    age = int(rng.integers(0, 40))
                area = int(rng.integers(20, 100))
                price = math.exp(true_log(district, year, age) + rng.normal(0, 0.02))
                for r in rows_for("13104", "新宿区", year, q, "01", [price], building_year=f"{year - age}年", area=area,
                                  district_name=district, floor_plan="2LDK", structure="RC"):  # fmt: skip
                    same = files.setdefault(r["raw_file"], [])
                    same.append(r | {"row_index": len(same)})
    with DuckDBStore(path) as s:
        s.replace_files(files, datetime(2026, 4, 1))
        s.con.execute("""
            INSERT INTO station VALUES
                ('A', '西新宿', ['東京メトロ／丸ノ内線'], 139.69, 35.69, 100000, 2023),
                ('B', '大久保', ['JR東日本／中央線'], 139.70, 35.70, NULL, NULL),
                ('C', '新宿', ['JR東日本／中央線'], 139.70, 35.69, 3000000, 2023)
        """)
        s.con.execute("INSERT INTO station_line VALUES ('JR東日本', '中央線', 'C', 0), ('JR東日本', '中央線', 'B', 1)")
        s.con.execute("""
            INSERT INTO transaction_station
            SELECT raw_file, row_index, CASE district_name WHEN '西新宿' THEN 'A' ELSE 'B' END
            FROM transactions WHERE district_name IN ('西新宿', '大久保')
        """)
    return path


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    with DuckDBStore(build_station_db(tmp_path_factory.mktemp("st") / "st.duckdb")) as s:
        return build_station_analysis(s)


def test_periods_and_tables(analysis):
    assert analysis.years == list(range(2016, 2026))  # 2026 只有 1 季：不是完整年
    assert analysis.latest_period == (2023, 2, 2026, 1)  # 最近 12 季
    assert analysis.target == "2025Q2〜2026Q1"
    assert analysis.lines == {"JR東日本／中央線": ["C", "B"]}
    assert set(analysis.stations) == {"A", "B", "C"}
    assert analysis.n == 2 * 10 * len(QUARTERS)  # 只算對上車站的
    assert analysis.unadjusted == 0


def test_station_info_and_city(analysis):
    a = analysis.stations["A"]
    assert a.info.name == "西新宿" and a.info.lines == ("東京メトロ／丸ノ内線",)
    assert a.city == "13104" and a.cities == {"13104": 10 * len(QUARTERS)}
    c = analysis.stations["C"]  # 路線上有、但沒有成交的車站
    assert c.yearly == {} and c.latest.status == "insufficient" and c.city is None


def test_standardized_series_removes_aging(analysis):
    a = analysis.stations["A"]
    # 典型物件（屋齡 17 年）的水準
    for year in (2016, 2020, 2025):
        assert a.yearly[year].n == 40
        assert math.log(a.yearly[year].median) == pytest.approx(true_log("西新宿", year, 17), abs=0.02)
    g = analysis.growth("A", 2016, 2025)
    assert g.status == "ok"
    assert math.log1p(g.total) == pytest.approx(9 * TREND, abs=0.02)
    assert a.invest.status == "ok" and math.log1p(a.invest.annual) == pytest.approx(TREND, abs=0.005)


def test_latest_is_time_adjusted_median_of_twelve_quarters(analysis):
    a, b = analysis.stations["A"], analysis.stations["B"]
    assert a.latest.n == b.latest.n == 120
    # 目標期 2025Q2〜2026Q1：3 季在 2025、1 季在 2026
    target = TREND * (2025 - 2016) + TREND / 4
    assert math.log(a.latest.median) == pytest.approx(true_log("西新宿", 2016, 17) + target, abs=0.02)
    assert a.latest.median / b.latest.median == pytest.approx(math.exp(0.5), rel=0.03)
    assert a.latest.ci_low < a.latest.median < a.latest.ci_high
    # 區的比較：同方法、包含沒對上車站的成交（早稲田）
    city = analysis.city_latest["13104"]
    assert city.n == 360
    assert math.log(city.median) == pytest.approx(true_log("早稲田", 2016, 17) + target, abs=0.03)


def test_needs_station_data(tmp_path):
    with DuckDBStore(tmp_path / "e.duckdb") as s:
        rows = rows_for("13104", "新宿区", 2024, 1, "01", [1_000_000] * MIN_N)
        s.replace_files({rows[0]["raw_file"]: rows}, datetime(2026, 1, 1))
        with pytest.raises(ValueError, match="fetch-stations"):
            build_station_analysis(s)


@pytest.mark.parametrize(
    ("key", "label", "group"),
    [
        ("東京都／12号線大江戸線", "都営大江戸線", "都営"),
        ("東日本旅客鉄道／中央線", "JR中央線", "JR"),
        ("東京地下鉄／4号線丸ノ内線", "東京メトロ丸ノ内線", "東京メトロ"),
        ("京王電鉄／京王線", "京王線", "京王"),
        ("東急電鉄／東急多摩川線", "東急多摩川線", "東急"),
        ("西武鉄道／新宿線", "西武新宿線", "西武"),
        ("ゆりかもめ／東京臨海新交通臨海線", "ゆりかもめ", "ゆりかもめ"),
        ("首都圏新都市鉄道／常磐新線", "つくばエクスプレス", "つくばエクスプレス"),
    ],
)
def test_line_label(key, label, group):
    assert line_label(key) == label and operator_label(key) == group
