import json

from tokyo_housing_lab.sources.base import QuarterQuery
from tokyo_housing_lab.sources.raw_store import FETCH_LOG_NAME, RawStore

Q = QuarterQuery("13202", 2024, 1, "01")


def test_filename_has_city_year_quarter_and_price_kind(tmp_path):
    store = RawStore(tmp_path)
    assert store.path_for(Q).name == "xit001_13202_2024Q1_p01.json"
    assert store.path_for(QuarterQuery("13202", 2024, 1, "02")).name == "xit001_13202_2024Q1_p02.json"


def test_save_keeps_bytes_exactly(tmp_path):
    # 空白、跳脫字元、key 順序都要原封不動，證明沒有被重新序列化
    body = '{"b":1,  "a" : "\\u6771京",\n"c":[ ]}'.encode()
    store = RawStore(tmp_path / "raw")
    path = store.save(Q, body)
    assert path.read_bytes() == body
    assert store.exists(Q)


def test_save_leaves_no_temp_file(tmp_path):
    store = RawStore(tmp_path)
    store.save(Q, b"{}")
    assert [p.name for p in tmp_path.iterdir()] == ["xit001_13202_2024Q1_p01.json"]


def test_save_overwrites(tmp_path):
    store = RawStore(tmp_path)
    store.save(Q, b'{"v":1}')
    store.save(Q, b'{"v":2}')
    assert store.path_for(Q).read_bytes() == b'{"v":2}'


def test_append_log(tmp_path):
    store = RawStore(tmp_path)
    store.append_log(Q, 200, "saved", store.path_for(Q))
    store.append_log(QuarterQuery("13202", 2024, 2, "02"), 404, "no_data", None)
    lines = [json.loads(line) for line in (tmp_path / FETCH_LOG_NAME).read_text().splitlines()]
    assert [(e["year"], e["quarter"], e["price_classification"], e["http_status"], e["result"], e["file"]) for e in lines] == [
        (2024, 1, "01", 200, "saved", "xit001_13202_2024Q1_p01.json"),
        (2024, 2, "02", 404, "no_data", None),
    ]
    assert all(e["fetched_at"] for e in lines)
