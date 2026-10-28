import gzip
import logging
from datetime import date

import httpx
import pytest

from conftest import FAKE_KEY
from tokyo_housing_lab.config import ApiKey
from tokyo_housing_lab.sources.base import QuarterQuery
from tokyo_housing_lab.sources.raw_store import RawStore
from tokyo_housing_lab.sources.reinfolib_api import (
    KEY_HEADER,
    ReinfolibAPIError,
    ReinfolibClient,
    fetch_range,
)

Q = QuarterQuery("13202", 2024, 1, "01")


def make_client(clock, responses):
    """responses：依序回應的清單，元素是 httpx.Response 或要丟出的例外。"""
    queue = list(responses)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = ReinfolibClient(
        ApiKey(FAKE_KEY),
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleep=clock.sleep,
    )
    return client, requests


def gzip_json(body: bytes) -> httpx.Response:
    # 真實 API 回 Content-Encoding: gzip
    return httpx.Response(
        200,
        content=gzip.compress(body),
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
    )


def test_200_returns_decompressed_body_and_sends_key(clock, sample_body):
    client, requests = make_client(clock, [gzip_json(sample_body)])
    result = client.fetch(Q)
    assert result.http_status == 200
    assert result.has_data
    assert result.body == sample_body
    req = requests[0]
    assert req.headers[KEY_HEADER] == FAKE_KEY
    assert dict(req.url.params) == {
        "year": "2024",
        "quarter": "1",
        "city": "13202",
        "priceClassification": "01",
    }
    assert clock.sleeps == []


def test_404_means_no_data(clock):
    client, requests = make_client(clock, [httpx.Response(404)])
    result = client.fetch(Q)
    assert result.http_status == 404
    assert not result.has_data
    assert result.body is None
    assert len(requests) == 1


def test_429_then_200_uses_backoff(clock, sample_body):
    client, requests = make_client(clock, [httpx.Response(429), gzip_json(sample_body)])
    assert client.fetch(Q).body == sample_body
    assert len(requests) == 2
    assert clock.sleeps == [2.0]


def test_retry_after_header_is_respected(clock, sample_body):
    client, _ = make_client(
        clock, [httpx.Response(429, headers={"Retry-After": "7"}), gzip_json(sample_body)]
    )
    client.fetch(Q)
    assert clock.sleeps == [7.0]


def test_unparsable_retry_after_falls_back_to_backoff(clock, sample_body):
    client, _ = make_client(
        clock,
        [httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), gzip_json(sample_body)],
    )
    client.fetch(Q)
    assert clock.sleeps == [2.0]


def test_5xx_retries_exhausted_raises(clock):
    client, requests = make_client(clock, [httpx.Response(500)] * 4 + [httpx.Response(503)])
    with pytest.raises(ReinfolibAPIError) as exc_info:
        client.fetch(Q)
    assert exc_info.value.status_code == 503
    assert len(requests) == 5  # 1 次正式請求 + 4 次重試
    assert clock.sleeps == [2.0, 4.0, 8.0, 16.0]


def test_connection_error_is_retried(clock, sample_body):
    client, requests = make_client(clock, [httpx.ConnectError("boom"), gzip_json(sample_body)])
    assert client.fetch(Q).has_data
    assert len(requests) == 2
    assert clock.sleeps == [2.0]


def test_connection_error_exhausted_has_no_status(clock):
    client, _ = make_client(clock, [httpx.ReadTimeout("slow")] * 5)
    with pytest.raises(ReinfolibAPIError) as exc_info:
        client.fetch(Q)
    assert exc_info.value.status_code is None


@pytest.mark.parametrize("status", [400, 401, 403])
def test_other_4xx_raise_without_retry(clock, status):
    client, requests = make_client(clock, [httpx.Response(status)])
    with pytest.raises(ReinfolibAPIError) as exc_info:
        client.fetch(Q)
    assert exc_info.value.status_code == status
    assert len(requests) == 1
    assert clock.sleeps == []


def test_200_with_non_json_body_raises(clock):
    client, _ = make_client(clock, [httpx.Response(200, content=b"<html>maintenance</html>")])
    with pytest.raises(ReinfolibAPIError, match="不是 JSON"):
        client.fetch(Q)


def test_requests_are_at_least_one_second_apart(clock, sample_body):
    client, _ = make_client(clock, [gzip_json(sample_body)] * 3)
    client.fetch(Q)
    client.fetch(Q)  # 緊接著打：要等滿 1 秒
    clock.now += 0.3  # 中間自己花了 0.3 秒：只要再等 0.7 秒
    client.fetch(Q)
    assert clock.sleeps == pytest.approx([1.0, 0.7])


def test_no_wait_when_interval_already_passed(clock, sample_body):
    client, _ = make_client(clock, [gzip_json(sample_body)] * 2)
    client.fetch(Q)
    clock.now += 5
    client.fetch(Q)
    assert clock.sleeps == []


def test_key_never_appears_in_logs_or_errors(clock, caplog, sample_body):
    caplog.set_level(logging.DEBUG)
    client, _ = make_client(
        clock,
        [
            httpx.Response(429),
            httpx.ConnectError("boom"),
            gzip_json(sample_body),
            httpx.Response(404),
            httpx.Response(401),
        ],
    )
    client.fetch(Q)
    client.fetch(Q)
    with pytest.raises(ReinfolibAPIError) as exc_info:
        client.fetch(Q)
    assert caplog.records  # 確認真的有 log 被檢查到
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in str(exc_info.value)
    assert FAKE_KEY not in repr(exc_info.value)


# --- fetch_range：逐季抓取 + 存檔 ---


def test_fetch_range_saves_skips_and_records_no_data(clock, tmp_path, sample_body):
    store = RawStore(tmp_path)
    store.save(QuarterQuery("13202", 2021, 1, "01"), b'{"already": "here"}')
    # 2021-06-01 → 抓到 2021Q2。2021Q1 p01 已存在，不會發請求
    client, requests = make_client(
        clock,
        [
            gzip_json(sample_body),  # 2021Q1 p02
            gzip_json(sample_body),  # 2021Q2 p01
            httpx.Response(404),  # 2021Q2 p02
        ],
    )
    outcomes = fetch_range(client, store, "13202", (2021, 1), date(2021, 6, 1))

    assert [(q.year, q.quarter, q.price_classification, o) for q, o in outcomes] == [
        (2021, 1, "01", "skipped"),
        (2021, 1, "02", "saved"),
        (2021, 2, "01", "saved"),
        (2021, 2, "02", "no_data"),
    ]
    assert [(r.url.params["year"], r.url.params["quarter"], r.url.params["priceClassification"]) for r in requests] == [
        ("2021", "1", "02"),
        ("2021", "2", "01"),
        ("2021", "2", "02"),
    ]
    assert store.path_for(QuarterQuery("13202", 2021, 1, "01")).read_bytes() == b'{"already": "here"}'
    assert store.path_for(QuarterQuery("13202", 2021, 2, "01")).read_bytes() == sample_body
    assert not store.exists(QuarterQuery("13202", 2021, 2, "02"))  # 404 不存檔
    log_lines = (tmp_path / "fetch_log.jsonl").read_text().splitlines()
    assert len(log_lines) == 3  # 跳過的不算，有打 API 的才記


def test_fetch_range_force_refetches(clock, tmp_path, sample_body):
    store = RawStore(tmp_path)
    q = QuarterQuery("13202", 2021, 1, "01")
    store.save(q, b'{"old": true}')
    client, requests = make_client(clock, [gzip_json(sample_body)])
    fetch_range(client, store, "13202", (2021, 1), date(2021, 1, 15), price_classifications=["01"], force=True)
    assert len(requests) == 1
    assert store.path_for(q).read_bytes() == sample_body


def test_fetch_range_error_stops_and_keeps_earlier_files(clock, tmp_path, sample_body):
    store = RawStore(tmp_path)
    client, _ = make_client(clock, [gzip_json(sample_body), httpx.Response(401)])
    with pytest.raises(ReinfolibAPIError):
        fetch_range(client, store, "13202", (2021, 1), date(2021, 1, 15))
    # 先抓到的那一份已經存好，下次重跑會跳過它
    assert store.exists(QuarterQuery("13202", 2021, 1, "01"))
