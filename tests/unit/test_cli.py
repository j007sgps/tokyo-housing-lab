import gzip
import shutil

import duckdb
import httpx
import pytest

from conftest import FAKE_KEY, FIXTURES, FakeClock
from tokyo_housing_lab import DISCLAIMER, cli
from tokyo_housing_lab.config import API_KEY_NAME, ApiKey
from tokyo_housing_lab.sources.reinfolib_api import ReinfolibClient


def test_help_has_disclaimer(capsys):
    assert cli.main([]) == 0
    assert DISCLAIMER in capsys.readouterr().out


@pytest.mark.parametrize("command", ["fetch", "load"])
def test_subcommand_help_has_disclaimer(command, capsys):
    with pytest.raises(SystemExit):
        cli.main([command, "--help"])
    assert DISCLAIMER in capsys.readouterr().out.replace("\n", "")


def test_bad_start_quarter_is_rejected(capsys):
    with pytest.raises(SystemExit):
        cli.main(["fetch", "--start", "2021-1"])
    assert "2021Q1" in capsys.readouterr().err


def test_fetch_without_key_fails_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(API_KEY_NAME, raising=False)
    monkeypatch.chdir(tmp_path)  # 確保不會讀到專案目錄的 .env
    assert cli.main(["fetch", "--raw-dir", str(tmp_path / "raw")]) == 1
    assert API_KEY_NAME in capsys.readouterr().err


def test_fetch_and_load(monkeypatch, tmp_path, capsys):
    body = (FIXTURES / "xit001_13202_2024Q1_p01_sample.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["year"] == "2021" and request.url.params["quarter"] == "1":
            return httpx.Response(
                200,
                content=gzip.compress(body.replace(b"2024\xe5\xb9\xb4\xe7\xac\xac1", "2021年第1".encode())),
                headers={"Content-Encoding": "gzip"},
            )
        return httpx.Response(404)

    clock = FakeClock()
    monkeypatch.setattr(
        cli,
        "_make_client",
        lambda: ReinfolibClient(
            ApiKey(FAKE_KEY), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep
        ),
    )
    raw = tmp_path / "raw"
    db = tmp_path / "t.duckdb"

    assert cli.main(["fetch", "--raw-dir", str(raw), "--start", "2021Q1"]) == 0
    assert "新存檔 2" in capsys.readouterr().out
    assert sorted(p.name for p in raw.glob("xit001_*")) == [  # 預設抓東京都全域（area=13）
        "xit001_13_2021Q1_p01.json.gz",
        "xit001_13_2021Q1_p02.json.gz",
    ]

    assert cli.main(["load", "--raw-dir", str(raw), "--db", str(db)]) == 0
    assert "中古マンション等 6 筆" in capsys.readouterr().out  # 樣本有 3 筆 × 01、02 兩份
    with duckdb.connect(str(db)) as con:
        assert con.execute("SELECT count(*) FROM transactions").fetchone()[0] == 6
        assert con.execute("SELECT count(*) FROM fetch_log WHERE result = 'no_data'").fetchone()[0] > 0


def test_load_bad_file_returns_error(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIXTURES / "xit001_13202_2024Q1_p01_sample.json", raw / "xit001_13202_2023Q4_p01.json")
    assert cli.main(["load", "--raw-dir", str(raw), "--db", str(tmp_path / "t.duckdb")]) == 1
    assert "Period" in capsys.readouterr().err


def test_fake_key_not_printed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli,
        "_make_client",
        lambda: ReinfolibClient(
            ApiKey(FAKE_KEY), transport=httpx.MockTransport(lambda r: httpx.Response(401)), sleep=lambda s: None
        ),
    )
    assert cli.main(["fetch", "--raw-dir", str(tmp_path)]) == 1
    out = capsys.readouterr()
    assert "401" in out.err
    assert FAKE_KEY not in out.out + out.err


def test_query_without_db_fails_cleanly(tmp_path, capsys):
    assert cli.main(["query", "--db", str(tmp_path / "none.duckdb")]) == 1
    assert "load" in capsys.readouterr().err
    assert not (tmp_path / "none.duckdb").exists()


def test_query_prints_table_and_disclaimer(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    for pc in ("01", "02"):
        shutil.copy(FIXTURES / f"xit001_13202_2024Q1_p{pc}_sample.json", raw / f"xit001_13202_2024Q1_p{pc}.json")
    db = tmp_path / "t.duckdb"
    assert cli.main(["load", "--raw-dir", str(raw), "--db", str(db)]) == 0
    capsys.readouterr()
    assert cli.main(["query", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "2024Q1" in out and "取引" in out and "成約" in out
    assert out.rstrip().endswith(DISCLAIMER)


def test_query_other_city_has_no_rows(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIXTURES / "xit001_13202_2024Q1_p01_sample.json", raw / "xit001_13202_2024Q1_p01.json")
    db = tmp_path / "t.duckdb"
    cli.main(["load", "--raw-dir", str(raw), "--db", str(db)])
    capsys.readouterr()
    assert cli.main(["query", "--db", str(db), "--city", "13101"]) == 0
    assert "沒有資料" in capsys.readouterr().out
