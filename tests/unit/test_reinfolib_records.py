import json
import logging
import shutil

import pytest

from conftest import FIXTURES
from tokyo_housing_lab.processing.filter import is_used_condo
from tokyo_housing_lab.sources.reinfolib_records import (
    FIELD_MAP,
    RawFileError,
    query_from_filename,
    read_raw_file,
)

SAMPLE_01 = FIXTURES / "xit001_13202_2024Q1_p01_sample.json"
SAMPLE_02 = FIXTURES / "xit001_13202_2024Q1_p02_sample.json"


def as_raw_file(tmp_path, fixture, name="xit001_13202_2024Q1_p01.json"):
    # fixture 的檔名帶 _sample，複製成正式原始檔的檔名
    path = tmp_path / name
    shutil.copy(fixture, path)
    return path


def write_raw(tmp_path, doc, name="xit001_13202_2024Q1_p01.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return path


def sample_row():
    return json.loads(SAMPLE_01.read_text())["data"][0]


def test_query_from_filename():
    q = query_from_filename("xit001_13202_2024Q1_p02.json")
    assert (q.city, q.year, q.quarter, q.price_classification) == ("13202", 2024, 1, "02")


@pytest.mark.parametrize(
    "name", ["xit001_13202_2024Q5_p01.json", "xit001_13202_2024Q1_p01.json.tmp", "fetch_log.jsonl", "foo.json"]
)
def test_query_from_filename_rejects_other_files(name):
    with pytest.raises(RawFileError):
        query_from_filename(name)


def test_read_real_sample(tmp_path):
    records = read_raw_file(as_raw_file(tmp_path, SAMPLE_01))
    assert len(records) == 5
    r = records[0]
    assert (r.source, r.city, r.year, r.quarter, r.price_classification) == ("api", "13202", 2024, 1, "01")
    assert r.raw_file == "xit001_13202_2024Q1_p01.json"
    assert [rec.row_index for rec in records] == [0, 1, 2, 3, 4]
    assert set(r.fields) == set(FIELD_MAP.values())  # 29 個欄位都換成內部欄位名
    assert r.fields["property_type"] == "中古マンション等"
    assert r.fields["trade_price"] == sample_row()["TradePrice"]  # 值仍是原字串


def test_read_sample_02_and_filter(tmp_path):
    records = read_raw_file(as_raw_file(tmp_path, SAMPLE_02, "xit001_13202_2024Q1_p02.json"))
    assert all(r.price_classification == "02" for r in records)
    assert [is_used_condo(r) for r in records] == [True, True, False]


def test_status_not_ok_raises(tmp_path):
    with pytest.raises(RawFileError, match="status"):
        read_raw_file(write_raw(tmp_path, {"status": "NG", "data": []}))


def test_period_mismatch_raises(tmp_path):
    row = sample_row() | {"Period": "2024年第2四半期"}
    with pytest.raises(RawFileError, match="Period"):
        read_raw_file(write_raw(tmp_path, {"status": "OK", "data": [row]}))


def test_missing_field_raises(tmp_path):
    row = sample_row()
    del row["Area"]
    with pytest.raises(RawFileError, match="Area"):
        read_raw_file(write_raw(tmp_path, {"status": "OK", "data": [row]}))


def test_unknown_field_warns_but_continues(tmp_path, caplog):
    row = sample_row() | {"NewField": "x"}
    records = read_raw_file(write_raw(tmp_path, {"status": "OK", "data": [row]}))
    assert len(records) == 1
    assert "NewField" in caplog.text
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_empty_data(tmp_path):
    assert read_raw_file(write_raw(tmp_path, {"status": "OK", "data": []})) == []
