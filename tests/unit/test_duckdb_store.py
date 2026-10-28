import gzip
import json
import shutil
from datetime import datetime

import duckdb
import pytest

from conftest import FIXTURES
from tokyo_housing_lab.pipeline import load_raw_dir
from tokyo_housing_lab.processing.clean import clean_record
from tokyo_housing_lab.sources.reinfolib_records import RawFileError
from tokyo_housing_lab.storage.duckdb_store import TRANSACTION_COLUMNS, DuckDBStore
from test_clean import make_record

T0 = datetime(2026, 10, 28, 0, 0, 0)


@pytest.fixture
def raw_dir(tmp_path):
    """fixture 複製成正式檔名的 data/raw。01 樣本：中古マンション 3 筆＋土地 2 筆；02：中古マンション 2 筆＋土地 1 筆。"""
    d = tmp_path / "raw"
    d.mkdir()
    for pc in ("01", "02"):
        shutil.copy(FIXTURES / f"xit001_13202_2024Q1_p{pc}_sample.json", d / f"xit001_13202_2024Q1_p{pc}.json")
    return d


@pytest.fixture
def store(tmp_path):
    with DuckDBStore(tmp_path / "test.duckdb") as s:
        yield s


def count(store, where="TRUE"):
    return store.con.execute(f"SELECT count(*) FROM transactions WHERE {where}").fetchone()[0]


def test_schema_matches_clean_record():
    # 處理層加了欄位卻忘了改資料表，這個測試會先失敗
    assert [name for name, _ in TRANSACTION_COLUMNS] == [*clean_record(make_record()), "loaded_at"]


def test_load_only_used_condos(raw_dir, store):
    summary = load_raw_dir(raw_dir, store)
    assert (summary.files, summary.records_read, summary.rows_loaded) == (2, 8, 5)
    assert count(store) == 5
    assert count(store, "property_type <> '中古マンション等'") == 0
    assert count(store, "price_classification = '01'") == 3
    assert count(store, "price_classification = '02'") == 2


def test_values_and_types_survive_round_trip(raw_dir, store):
    load_raw_dir(raw_dir, store)
    row = store.con.execute(
        "SELECT year, quarter, trade_price, area_m2, unit_price_per_m2, building_year_status, floor_plan, loaded_at "
        "FROM transactions WHERE raw_file = 'xit001_13202_2024Q1_p01.json' AND row_index = 0"
    ).fetchone()
    sample = json.loads((raw_dir / "xit001_13202_2024Q1_p01.json").read_text())["data"][0]
    year, quarter, price, area, unit, by_status, floor_plan, loaded_at = row
    assert (year, quarter) == (2024, 1)
    assert price == int(sample["TradePrice"]) and isinstance(price, int)
    assert area == float(sample["Area"])
    assert unit == pytest.approx(price / area)
    assert by_status == "ok"
    assert floor_plan.isascii()  # 已轉半形
    assert isinstance(loaded_at, datetime)


def test_reload_does_not_duplicate(raw_dir, store):
    load_raw_dir(raw_dir, store)
    load_raw_dir(raw_dir, store)
    assert count(store) == 5


def test_reload_replaces_changed_file(raw_dir, store):
    load_raw_dir(raw_dir, store)
    path = raw_dir / "xit001_13202_2024Q1_p01.json"
    doc = json.loads(path.read_text())
    doc["data"] = doc["data"][:1]  # 官方修正後只剩 1 筆
    path.write_text(json.dumps(doc, ensure_ascii=False))
    load_raw_dir(raw_dir, store)
    assert count(store, "price_classification = '01'") == 1
    assert count(store, "price_classification = '02'") == 2


def test_deleted_raw_file_is_pruned(raw_dir, store):
    load_raw_dir(raw_dir, store)
    (raw_dir / "xit001_13202_2024Q1_p02.json").unlink()
    load_raw_dir(raw_dir, store)
    assert count(store, "price_classification = '02'") == 0
    assert count(store) == 3


def test_bad_file_rolls_back_everything(raw_dir, store):
    load_raw_dir(raw_dir, store)
    path = raw_dir / "xit001_13202_2024Q1_p02.json"
    doc = json.loads(path.read_text())
    doc["data"][0]["Period"] = "2024年第2四半期"
    path.write_text(json.dumps(doc, ensure_ascii=False))
    (raw_dir / "xit001_13202_2024Q1_p01.json").unlink()  # 這個變更也不能生效
    with pytest.raises(RawFileError):
        load_raw_dir(raw_dir, store)
    assert count(store) == 5  # 維持上一次成功載入的狀態


def row_for(raw_file, **overrides):
    return dict(clean_record(make_record()), raw_file=raw_file, **overrides)


def test_db_failure_mid_load_rolls_back(store):
    store.replace_files({"a.json": [row_for("a.json")]}, T0)
    broken = row_for("a.json", trade_price="not a number")  # 型別錯誤，INSERT 會失敗
    with pytest.raises(Exception):
        store.replace_files({"b.json": [row_for("b.json")], "a.json": [broken]}, T0)
    rows = store.con.execute("SELECT raw_file, trade_price FROM transactions").fetchall()
    assert rows == [("a.json", 11000000)]  # b.json 沒寫進去，a.json 也沒被刪


def test_row_raw_file_must_match_key(store):
    with pytest.raises(ValueError, match="raw_file"):
        store.replace_files({"a.json": [row_for("other.json")]}, T0)
    assert count(store) == 0


def test_empty_raw_dir_clears_transactions(raw_dir, store, tmp_path):
    load_raw_dir(raw_dir, store)
    empty = tmp_path / "empty"
    empty.mkdir()
    summary = load_raw_dir(empty, store)
    assert summary.files == 0
    assert count(store) == 0


def test_fetch_log_loaded_and_replaced(raw_dir, store):
    log = raw_dir / "fetch_log.jsonl"
    entry = {
        "fetched_at": "2026-10-28T00:31:15+00:00",
        "city": "13202",
        "year": 2026,
        "quarter": 2,
        "price_classification": "01",
        "http_status": 404,
        "result": "no_data",
        "file": None,
    }
    log.write_text(json.dumps(entry) + "\n")
    assert load_raw_dir(raw_dir, store).fetch_log_entries == 1
    log.write_text(json.dumps(entry) + "\n" + json.dumps(entry | {"quarter": 3}) + "\n")
    load_raw_dir(raw_dir, store)
    rows = store.con.execute("SELECT quarter, http_status, result, file FROM fetch_log ORDER BY quarter").fetchall()
    assert rows == [(2, 404, "no_data", None), (3, 404, "no_data", None)]


def test_reserved_word_column_works(raw_dir, store):
    load_raw_dir(raw_dir, store)
    assert store.con.execute('SELECT count("use") FROM transactions').fetchone()[0] >= 0


def test_fetch_log_new_region_key(raw_dir, store):
    entry = {
        "fetched_at": "2026-10-28T02:00:00+00:00",
        "region": "13",
        "year": 2026,
        "quarter": 2,
        "price_classification": "02",
        "http_status": 404,
        "result": "no_data",
        "file": None,
    }
    (raw_dir / "fetch_log.jsonl").write_text(json.dumps(entry) + "\n")
    load_raw_dir(raw_dir, store)
    assert store.con.execute("SELECT region FROM fetch_log").fetchall() == [("13",)]


def test_rows_can_be_streamed(store):
    def stream():
        for name in ("a.json", "b.json"):
            yield name, (row_for(name, row_index=i) for i in range(3))

    assert store.replace_files(stream(), T0, prune_others=True) == 6
    assert count(store) == 6


def test_old_schema_is_rebuilt(tmp_path):
    db = tmp_path / "old.duckdb"
    with duckdb.connect(str(db)) as con:  # M1 時期的表格：沒有 region 欄
        con.execute("CREATE TABLE transactions (source VARCHAR, city VARCHAR, raw_file VARCHAR, row_index INTEGER)")
        con.execute("INSERT INTO transactions VALUES ('api', '13202', 'x.json', 0)")
    with DuckDBStore(db) as s:
        cols = [c for (c,) in s.con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'transactions' ORDER BY ordinal_position").fetchall()]
        assert cols == [name for name, _ in TRANSACTION_COLUMNS]
        assert count(s) == 0  # 重建後是空的，重新 load 即可


# --- 東京都全域（area）檔案與 M1 的市區町村（city）檔案並存 ---


def gz_copy(src, dst, municipality=None):
    doc = json.loads(src.read_text())
    if municipality:
        for row in doc["data"]:
            row["MunicipalityCode"] = municipality
    dst.write_bytes(gzip.compress(json.dumps(doc, ensure_ascii=False).encode()))


def test_city_file_covered_by_area_file_is_skipped(raw_dir, store):
    # raw_dir 已有立川市 2024Q1 的 01、02（未壓縮的 .json）；加上東京都全域 2024Q1 的 01
    gz_copy(FIXTURES / "xit001_13202_2024Q1_p01_sample.json", raw_dir / "xit001_13_2024Q1_p01.json.gz")
    summary = load_raw_dir(raw_dir, store)
    assert summary.skipped_files == ("xit001_13202_2024Q1_p01.json",)
    assert summary.files == 2
    assert count(store, "price_classification = '01'") == 3  # 沒有算兩次
    assert count(store, "raw_file = 'xit001_13202_2024Q1_p01.json'") == 0
    assert count(store, "region = '13' AND city = '13202'") == 3


def test_city_file_from_other_quarter_is_kept(raw_dir, store):
    gz_copy(FIXTURES / "xit001_13202_2024Q1_p01_sample.json", raw_dir / "xit001_13_2024Q2_p01.json.gz")
    path = raw_dir / "xit001_13_2024Q2_p01.json.gz"
    doc = json.loads(gzip.decompress(path.read_bytes()))
    for row in doc["data"]:
        row["Period"] = "2024年第2四半期"
    path.write_bytes(gzip.compress(json.dumps(doc, ensure_ascii=False).encode()))
    summary = load_raw_dir(raw_dir, store)
    assert summary.skipped_files == ()
    assert summary.files == 3


def test_gz_preferred_over_legacy_json_of_same_query(raw_dir, store):
    gz_copy(FIXTURES / "xit001_13202_2024Q1_p01_sample.json", raw_dir / "xit001_13202_2024Q1_p01.json.gz")
    summary = load_raw_dir(raw_dir, store)
    assert "xit001_13202_2024Q1_p01.json" in summary.skipped_files
    assert count(store, "price_classification = '01'") == 3
