import gzip
import json

from tokyo_housing_lab.sources.base import QuarterQuery
from tokyo_housing_lab.sources.raw_store import FETCH_LOG_NAME, RawStore, read_raw_bytes

Q = QuarterQuery("13", 2024, 1, "01")


def test_filename_has_region_year_quarter_and_price_kind(tmp_path):
    store = RawStore(tmp_path)
    assert store.path_for(Q).name == "xit001_13_2024Q1_p01.json.gz"
    assert store.path_for(QuarterQuery("13202", 2024, 1, "02")).name == "xit001_13202_2024Q1_p02.json.gz"


def test_save_keeps_bytes_exactly(tmp_path):
    # 空白、跳脫字元、key 順序都要原封不動，證明沒有被重新序列化；gzip 解壓後完全相同
    body = '{"b":1,  "a" : "\\u6771京",\n"c":[ ]}'.encode()
    store = RawStore(tmp_path / "raw")
    path = store.save(Q, body)
    assert read_raw_bytes(path) == body
    assert gzip.decompress(path.read_bytes()) == body  # 一般的 gzip 工具也解得開
    assert store.exists(Q)


def test_saved_file_is_compressed(tmp_path):
    body = json.dumps({"data": [{"Type": "中古マンション等", "TradePrice": "11000000"}] * 2000}).encode()
    path = RawStore(tmp_path).save(Q, body)
    assert path.stat().st_size < len(body) / 10


def test_save_is_deterministic(tmp_path):
    # 同樣的內容存兩次，檔案 bytes 一樣（gzip 標頭不放時間）
    store = RawStore(tmp_path)
    first = store.save(Q, b'{"v":1}').read_bytes()
    assert store.save(Q, b'{"v":1}').read_bytes() == first


def test_save_leaves_no_temp_file(tmp_path):
    store = RawStore(tmp_path)
    store.save(Q, b"{}")
    assert [p.name for p in tmp_path.iterdir()] == ["xit001_13_2024Q1_p01.json.gz"]


def test_save_overwrites(tmp_path):
    store = RawStore(tmp_path)
    store.save(Q, b'{"v":1}')
    store.save(Q, b'{"v":2}')
    assert read_raw_bytes(store.path_for(Q)) == b'{"v":2}'


def test_legacy_uncompressed_file_counts_as_existing_and_is_replaced(tmp_path):
    # M1 時期存的是未壓縮的 .json
    q = QuarterQuery("13202", 2021, 1, "01")
    legacy = tmp_path / "xit001_13202_2021Q1_p01.json"
    legacy.write_bytes(b'{"old":1}')
    store = RawStore(tmp_path)
    assert store.exists(q)
    assert read_raw_bytes(legacy) == b'{"old":1}'
    store.save(q, b'{"new":1}')
    assert not legacy.exists()  # 不留兩份
    assert [p.name for p in tmp_path.iterdir()] == ["xit001_13202_2021Q1_p01.json.gz"]


def test_append_log(tmp_path):
    store = RawStore(tmp_path)
    store.append_log(Q, 200, "saved", store.path_for(Q))
    store.append_log(QuarterQuery("13202", 2024, 2, "02"), 404, "no_data", None)
    lines = [json.loads(line) for line in (tmp_path / FETCH_LOG_NAME).read_text().splitlines()]
    assert [(e["region"], e["year"], e["quarter"], e["price_classification"], e["http_status"], e["result"], e["file"]) for e in lines] == [
        ("13", 2024, 1, "01", 200, "saved", "xit001_13_2024Q1_p01.json.gz"),
        ("13202", 2024, 2, "02", 404, "no_data", None),
    ]
    assert all(e["fetched_at"] for e in lines)
