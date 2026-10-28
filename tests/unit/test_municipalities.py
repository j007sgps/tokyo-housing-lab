import statistics
from datetime import datetime

import pytest

from tokyo_housing_lab.analysis.municipalities import COMBINED, build_analysis
from tokyo_housing_lab.analysis.trend import MIN_N
from tokyo_housing_lab.processing.clean import clean_record
from tokyo_housing_lab.sources.base import SourceRecord
from tokyo_housing_lab.sources.reinfolib_records import FIELD_MAP
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore


def rows_for(city, name, year, quarter, pc, prices, building_year="2000年"):
    raw_file = f"xit001_13_{year}Q{quarter}_p{pc}.json.gz"
    out = []
    for i, price in enumerate(prices):
        fields = dict.fromkeys(FIELD_MAP.values(), "") | {
            "property_type": "中古マンション等",
            "municipality": name,
            "municipality_code": city,
            "trade_price": str(price * 50),
            "area": "50",
            "building_year": building_year,
            "period": f"{year}年第{quarter}四半期",
        }
        out.append(clean_record(SourceRecord("api", "13", city, year, quarter, pc, raw_file, i, fields)))
    return out


def build_synthetic_db(path):
    """合成資料（㎡ 單價）：
    - 2005：只有 Q3、Q4（不完整年）
    - 2006：4 季都有（完整年）
    - 2007：只有 Q1（最新一季，不完整年）
    中央区每季 40 筆；小村每季 2 筆（樣本不足）。成約價格（02）只有 2006Q4、2007Q1。
    """
    files: dict[str, list] = {}

    def add(rows):
        for r in rows:
            same_file = files.setdefault(r["raw_file"], [])
            same_file.append(r | {"row_index": len(same_file)})  # 同一個檔案裡的筆數號碼不能重複

    quarters = [(2005, 3), (2005, 4), (2006, 1), (2006, 2), (2006, 3), (2006, 4), (2007, 1)]
    for k, (y, q) in enumerate(quarters):
        add(rows_for("13102", "中央区", y, q, "01", [100_000 * (k + 1) + i * 1000 for i in range(40)]))
        add(rows_for("13999", "小村", y, q, "01", [10_000, 20_000]))
    add(rows_for("13102", "中央区", 2006, 4, "02", [500_000 + i for i in range(40)]))
    add(rows_for("13102", "中央区", 2007, 1, "02", [600_000 + i for i in range(40)]))
    add(rows_for("13102", "中央区", 2006, 1, "01", [1_000], building_year="1985年"))
    add(rows_for("13102", "中央区", 2006, 2, "01", [2_000], building_year="戦前"))
    with DuckDBStore(path) as s:
        s.replace_files(files, datetime(2026, 1, 1))
    return path


@pytest.fixture
def store(tmp_path):
    with DuckDBStore(build_synthetic_db(tmp_path / "t.duckdb")) as s:
        yield s


def test_complete_years(store):
    a = build_analysis(store)
    # 2005、2007 不完整；成約價格有資料但沒有任何完整年（地圖據此停用成約的趨勢模式）
    assert a.complete_years == {"01": [2006], "02": []}


def test_yearly_only_complete_years_and_independent_median(store):
    a = build_analysis(store)
    m = a.municipalities["13102"]
    assert set(m.yearly["01"]) == {2006}
    prices = [p for k in range(2, 6) for p in [100_000 * (k + 1) + i * 1000 for i in range(40)]] + [1_000, 2_000]
    st = m.yearly["01"][2006]
    assert st.n == len(prices)
    assert st.median == pytest.approx(statistics.median(prices))


def test_latest_is_last_four_quarters_per_price_type(store):
    a = build_analysis(store)
    assert a.latest_period["01"] == (2006, 2, 2007, 1)
    assert a.latest_period["02"] == (2006, 2, 2007, 1)
    p = a.municipalities["13102"].latest["01"]
    # 2006Q2〜2007Q1 共 4 季 × 40 筆，加上 2006Q2 那 1 筆戦前
    assert p.n == 161
    assert p.status == "ok"
    assert a.municipalities["13102"].latest["02"].n == 80


def test_small_municipality_is_insufficient(store):
    a = build_analysis(store)
    m = a.municipalities["13999"]
    assert m.latest["01"].status == "insufficient"
    assert m.latest["01"].n == 8 < MIN_N
    assert m.latest["02"].status == "insufficient"  # 沒有任何成約資料也有一筆
    assert m.latest["02"].n == 0
    assert m.invest.status == "insufficient"


def test_by_age_excludes_unknown_year(store):
    a = build_analysis(store)
    decades = {b.decade: b.n for b in a.municipalities["13102"].by_age}
    # 最新完整年（2006）往前 5 年內：2005Q3〜2006Q4 共 6 季 × 40 筆；戦前那筆沒有建築年，不列入
    assert decades == {1980: 1, 2000: 240}


def test_region_filter(store):
    a = build_analysis(store, region="13999")
    assert list(a.municipalities) == ["13999"]


def test_growth_lookup(store):
    a = build_analysis(store)
    with pytest.raises(ValueError):
        a.growth("13102", "01", 2006, 2006)
    assert a.growth("13102", "01", 2005, 2006).status == "insufficient"  # 2005 不是完整年


def test_combined_falls_back_to_toriihiki_without_overlap_years(store):
    # 成約沒有任何完整年 → 無法換算，綜合就是取引
    a = build_analysis(store)
    m = a.municipalities["13102"]
    c = m.combined
    assert c.series.offset.status == "insufficient"
    assert c.series.levels[2006].median == pytest.approx(m.yearly["01"][2006].median)
    assert c.latest.median == pytest.approx(m.latest["01"].median)
    assert c.latest.n == m.latest["01"].n  # 成約沒換算，不算進筆數
    assert c.invest.status == "insufficient"
    assert a.municipalities["13999"].combined.latest.status == "insufficient"
    with pytest.raises(ValueError):
        a.growth("13102", COMBINED, 2006, 2006)
    assert a.growth("13102", COMBINED, 2005, 2006).status == "insufficient"
