"""讀取層：國交省 XIT001（不動産取引価格情報）API client。

- 請求之間至少間隔 min_interval 秒
- 429 / 5xx / 連線錯誤：退避重試（有 Retry-After 就照它的秒數）
- 404：該期沒資料，不是錯誤
- 其他 4xx（401、403…）：直接報錯，不重試
"""

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import httpx

from tokyo_housing_lab.config import ApiKey
from tokyo_housing_lab.sources.base import (
    PRICE_CLASSIFICATION_START,
    PRICE_CLASSIFICATIONS,
    QuarterQuery,
    quarter_of,
    quarters_between,
)
from tokyo_housing_lab.sources.raw_store import RawStore

logger = logging.getLogger(__name__)

BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
KEY_HEADER = "Ocp-Apim-Subscription-Key"
DEFAULT_BACKOFF_SECONDS = (2.0, 4.0, 8.0, 16.0)
# 最新幾季官方可能還會補資料，所以「有資料的最新 N 季」每次都重抓。N 是估計值，官方沒有公布修正期間
REFRESH_RECENT_QUARTERS = 4


class ReinfolibAPIError(RuntimeError):
    """API 呼叫失敗。訊息只含查詢條件與狀態碼，不含任何 header。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class FetchResult:
    query: QuarterQuery
    http_status: int
    body: bytes | None  # 解壓後的 JSON 原文；None 表示該期沒資料（404）

    @property
    def has_data(self) -> bool:
        return self.body is not None


class Throttle:
    """確保兩次請求的開始時間至少相隔 min_interval 秒。

    clock / sleep 可以換掉，測試時就不用真的等。
    """

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


def _retry_after_seconds(response: httpx.Response) -> float | None:
    # 只處理秒數格式；HTTP 日期格式很少見，遇到就退回一般的退避秒數
    value = response.headers.get("Retry-After", "").strip()
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


class ReinfolibClient:
    def __init__(
        self,
        api_key: ApiKey,
        *,
        base_url: str = BASE_URL,
        min_interval: float = 1.0,
        backoff_seconds: Sequence[float] = DEFAULT_BACKOFF_SECONDS,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url
        self._backoff_seconds = tuple(backoff_seconds)
        self._sleep = sleep
        self._throttle = Throttle(min_interval, clock=clock, sleep=sleep)
        self._http = httpx.Client(
            headers={KEY_HEADER: api_key.reveal()},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ReinfolibClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def fetch(self, query: QuarterQuery) -> FetchResult:
        params = {
            "year": str(query.year),
            "quarter": str(query.quarter),
            query.api_param: query.region,
            "priceClassification": query.price_classification,
        }
        # 第 0 次是正式請求，之後每一次重試前先等 backoff_seconds[i]
        for attempt in range(len(self._backoff_seconds) + 1):
            self._throttle.wait()
            try:
                response = self._http.get(self._base_url, params=params)
            except httpx.TransportError as exc:
                status = None
                reason = f"連線錯誤（{type(exc).__name__}）"
                retry_after = None
            else:
                status = response.status_code
                if status == 200:
                    return self._ok(query, response)
                if status == 404:
                    logger.info("%s -> 404，該期沒有資料", query.label)
                    return FetchResult(query, 404, None)
                if status != 429 and status < 500:
                    raise ReinfolibAPIError(f"{query.label}：HTTP {status}（不重試）", status)
                reason = f"HTTP {status}"
                retry_after = _retry_after_seconds(response)

            if attempt == len(self._backoff_seconds):
                raise ReinfolibAPIError(f"{query.label}：重試 {attempt} 次仍失敗，最後一次是 {reason}", status)
            wait = retry_after if retry_after is not None else self._backoff_seconds[attempt]
            logger.warning("%s -> %s，%.0f 秒後重試（第 %d 次）", query.label, reason, wait, attempt + 1)
            self._sleep(wait)
        raise AssertionError("unreachable")

    def _ok(self, query: QuarterQuery, response: httpx.Response) -> FetchResult:
        # response.content 已經由 httpx 依 Content-Encoding: gzip 解壓好了
        body = response.content
        try:
            json.loads(body)
        except ValueError as exc:
            raise ReinfolibAPIError(f"{query.label}：HTTP 200 但內容不是 JSON", 200) from exc
        logger.info("%s -> 200（%d bytes）", query.label, len(body))
        return FetchResult(query, 200, body)


FetchOutcome = Literal["saved", "no_data", "skipped"]


def fetch_range(
    client: ReinfolibClient,
    store: RawStore,
    region: str,
    start: tuple[int, int],
    today: date,
    price_classifications: Sequence[str] = PRICE_CLASSIFICATIONS,
    force: bool = False,
    refresh_recent: int = REFRESH_RECENT_QUARTERS,
) -> list[tuple[QuarterQuery, FetchOutcome]]:
    """從 start 那一季一路抓到 today 所在的那一季，01、02 分開請求。

    官方沒有「最新季」的查詢方式，所以抓到今天所在的季為止，還沒公開的季會回 404。
    各價格種類在 PRICE_CLASSIFICATION_START 之前的季不請求（官方沒有那麼早的資料）。
    已經存在的原始檔跳過，但「已有原始檔的季」裡最新的 refresh_recent 季照樣重抓；
    force=True 則全部重抓。404 不存檔，下次會再問一次，
    因為「現在還沒公開」的季之後可能就有了。
    """
    quarters = quarters_between(start, quarter_of(today))
    saved = [
        (y, q)
        for y, q in quarters
        if any(store.exists(QuarterQuery(region, y, q, pc)) for pc in price_classifications)
    ]
    recent = set(saved[-refresh_recent:]) if refresh_recent > 0 else set()

    outcomes: list[tuple[QuarterQuery, FetchOutcome]] = []
    for year, quarter in quarters:
        for pc in price_classifications:
            if (year, quarter) < PRICE_CLASSIFICATION_START[pc]:
                continue
            query = QuarterQuery(region, year, quarter, pc)
            if not force and (year, quarter) not in recent and store.exists(query):
                logger.info("%s 已有原始檔，跳過", query.label)
                outcomes.append((query, "skipped"))
                continue
            result = client.fetch(query)
            if result.body is not None:
                path = store.save(query, result.body)
                store.append_log(query, result.http_status, "saved", path)
                outcomes.append((query, "saved"))
            else:
                store.append_log(query, result.http_status, "no_data", None)
                outcomes.append((query, "no_data"))
    return outcomes
