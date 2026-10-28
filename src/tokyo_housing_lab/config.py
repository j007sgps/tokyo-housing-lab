"""設定讀取。金鑰只從環境變數或 .env 來，而且包成不會被印出來的物件。"""

import os
from pathlib import Path

from dotenv import dotenv_values

API_KEY_NAME = "REINFOLIB_API_KEY"


class MissingApiKeyError(RuntimeError):
    pass


class ApiKey:
    """包住金鑰字串。print、log、repr、f-string 都只會看到 ***。

    真的要用（放進 HTTP header）時才呼叫 reveal()。
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "ApiKey(***)"

    def __str__(self) -> str:
        return "***"


def load_api_key(env_file: Path | None = None) -> ApiKey:
    """環境變數優先，沒有的話再看 .env（預設為目前目錄的 .env）。

    不用 load_dotenv()，避免把 .env 的內容灌進 os.environ 影響到其他東西。
    """
    value = os.environ.get(API_KEY_NAME)
    if not value:
        path = env_file if env_file is not None else Path.cwd() / ".env"
        if path.is_file():
            value = dotenv_values(path).get(API_KEY_NAME)
    if not value:
        # 訊息只寫變數名稱，不寫任何值
        raise MissingApiKeyError(
            f"找不到 {API_KEY_NAME}：請設定環境變數，或複製 .env.example 成 .env 後填入金鑰"
        )
    return ApiKey(value.strip())
