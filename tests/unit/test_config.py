import os

import pytest

from tokyo_housing_lab.config import API_KEY_NAME, ApiKey, MissingApiKeyError, load_api_key


@pytest.fixture(autouse=True)
def no_key_in_env(monkeypatch):
    monkeypatch.delenv(API_KEY_NAME, raising=False)


def test_env_var_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv(API_KEY_NAME, "from-env")
    assert load_api_key(tmp_path / ".env").reveal() == "from-env"


def test_env_var_wins_over_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_NAME}=from-dotenv\n")
    monkeypatch.setenv(API_KEY_NAME, "from-env")
    assert load_api_key(env_file).reveal() == "from-env"


def test_dotenv_is_used_when_env_var_missing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"# comment\n{API_KEY_NAME}=from-dotenv\n")
    assert load_api_key(env_file).reveal() == "from-dotenv"


def test_empty_env_var_falls_back_to_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_NAME}=from-dotenv\n")
    monkeypatch.setenv(API_KEY_NAME, "")
    assert load_api_key(env_file).reveal() == "from-dotenv"


def test_dotenv_does_not_leak_into_environ(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_NAME}=from-dotenv\n")
    load_api_key(env_file)
    assert API_KEY_NAME not in os.environ


def test_missing_key_raises(tmp_path):
    with pytest.raises(MissingApiKeyError, match=API_KEY_NAME):
        load_api_key(tmp_path / ".env")


def test_whitespace_around_key_is_stripped(monkeypatch, tmp_path):
    monkeypatch.setenv(API_KEY_NAME, "  abc \n")
    assert load_api_key(tmp_path / ".env").reveal() == "abc"


def test_key_is_masked_everywhere():
    key = ApiKey("super-secret")
    for shown in (repr(key), str(key), f"{key}", f"{key!r}", repr([key]), repr({"k": key})):
        assert "super-secret" not in shown
