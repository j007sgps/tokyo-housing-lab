from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# 測試專用的假金鑰，用來檢查它有沒有漏進 log 或錯誤訊息
FAKE_KEY = "fake-key-for-tests-0123456789abcdef"


class FakeClock:
    """假的 monotonic 時鐘：sleep() 只把時間往前推，並記下每次睡了多久。"""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def placeholder_body() -> bytes:
    return (FIXTURES / "xit001_placeholder.json").read_bytes()
