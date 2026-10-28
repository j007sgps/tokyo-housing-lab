import gzip
import math
import shutil

import duckdb
import httpx
import pytest

from conftest import FAKE_KEY, FIXTURES, FakeClock
from test_municipalities import build_synthetic_db
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


# --- analyze ---


@pytest.fixture
def analysis_db(tmp_path):
    """test_municipalities 的合成資料：中央区（樣本夠）＋小村（樣本不足）。"""
    return build_synthetic_db(tmp_path / "t.duckdb")


def test_analyze_buy(analysis_db, capsys):
    assert cli.main(["analyze", "buy", "--db", str(analysis_db)]) == 0
    out = capsys.readouterr().out
    assert "2006Q2〜2007Q1" in out
    assert "中央区" in out
    assert "樣本不足" in out and "小村" in out
    assert "不是投資建議" in out
    assert out.rstrip().endswith(DISCLAIMER)


def test_analyze_trend_rejects_year_without_complete_data(analysis_db, capsys):
    assert cli.main(["analyze", "trend", "--from", "2005", "--to", "2006", "--db", str(analysis_db)]) == 1
    assert "2005" in capsys.readouterr().err


def test_analyze_invest_all_insufficient(analysis_db, capsys):
    assert cli.main(["analyze", "invest", "--db", str(analysis_db)]) == 0
    assert "樣本不足" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["buy", "invest"])
def test_analyze_combined(analysis_db, capsys, mode):
    assert cli.main(["analyze", mode, "--price", "combined", "--db", str(analysis_db)]) == 0
    out = capsys.readouterr().out
    assert "綜合" in out
    assert out.rstrip().endswith(DISCLAIMER)


def test_analyze_invest_rejects_seiyaku(analysis_db, capsys):
    assert cli.main(["analyze", "invest", "--price", "02", "--db", str(analysis_db)]) == 1
    assert "10 年" in capsys.readouterr().err


def test_analyze_without_db(tmp_path, capsys):
    assert cli.main(["analyze", "buy", "--db", str(tmp_path / "none.duckdb")]) == 1
    assert "load" in capsys.readouterr().err


def test_map_command(analysis_db, tmp_path, capsys):
    out = tmp_path / "m" / "map.html"
    assert cli.main(["map", "--db", str(analysis_db), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "地圖已產生" in capsys.readouterr().out


def test_map_fragment(analysis_db, tmp_path):
    out = tmp_path / "frag.html"
    assert cli.main(["map", "--db", str(analysis_db), "--out", str(out), "--fragment"]) == 0
    assert out.read_text(encoding="utf-8").startswith("<title>")


def test_map_without_db(tmp_path, capsys):
    assert cli.main(["map", "--db", str(tmp_path / "none.duckdb"), "--out", str(tmp_path / "x.html")]) == 1
    assert not (tmp_path / "x.html").exists()


# --- 品質調整 ---


@pytest.fixture(scope="module")
def hedonic_db(tmp_path_factory):
    from test_municipalities import build_hedonic_db

    return build_hedonic_db(tmp_path_factory.mktemp("h") / "h.duckdb")


@pytest.mark.parametrize("mode", ["buy", "trend", "invest"])
def test_analyze_hedonic(hedonic_db, capsys, mode):
    assert cli.main(["analyze", mode, "--price", "hedonic", "--db", str(hedonic_db)]) == 0
    out = capsys.readouterr().out
    assert "品質調整（典型物件 55㎡" in out
    assert "新宿区" in out
    assert out.rstrip().endswith(DISCLAIMER)


def test_analyze_hedonic_not_estimable(analysis_db, capsys):
    # 合成資料面積全是 50㎡：估不出來 → 錯誤訊息，不是 traceback
    assert cli.main(["analyze", "trend", "--price", "hedonic", "--db", str(analysis_db)]) == 1
    assert "估不出品質調整指數" in capsys.readouterr().err


def test_hedonic_command(hedonic_db, capsys):
    assert cli.main(["hedonic", "--validate", "--db", str(hedonic_db)]) == 0
    out = capsys.readouterr().out
    assert "2010〜2013 年" in out and "R²" in out
    assert "屋齡（和 17 年比）" in out and "成約（和取引比" in out
    assert "品質調整模型" in out and "同町名同年的中位數" in out
    assert "不是投資建議" in out and out.rstrip().endswith(DISCLAIMER)


# --- 時點修正 ---


def test_query_adjusted_requires_adjust(analysis_db, capsys):
    assert cli.main(["query", "--adjusted", "--city", "13102", "--db", str(analysis_db)]) == 1
    assert "請先執行 adjust" in capsys.readouterr().err


def test_adjust_then_query_adjusted(hedonic_db, capsys):
    assert cli.main(["adjust", "--db", str(hedonic_db)]) == 0
    out = capsys.readouterr().out
    assert "換算成 2013Q1〜2013Q4 的平均水準" in out
    assert "已寫入 time_adjustment：32 列" in out  # 2 區 × 16 季
    assert "新宿区" in out and out.rstrip().endswith(DISCLAIMER)
    assert "面積彈性" in out and "25㎡ 再 ×" in out
    with duckdb.connect(str(hedonic_db), read_only=True) as con:
        # 合成資料的面積效果沒有隨時間變：彈性幾乎是 0，全東京共用（每季一個值）
        e = con.execute("SELECT max(abs(area_elasticity)), count(DISTINCT (year, quarter, area_elasticity)) = count(DISTINCT (year, quarter)) FROM time_adjustment").fetchone()
        assert e[0] < 0.05 and e[1]
        rows = con.execute("SELECT year, quarter, factor, factor_low, factor_high, basis FROM time_adjustment WHERE city = '13104' ORDER BY year, quarter").fetchall()
    assert len(rows) == 16 and all(r[5] == "city" for r in rows)
    assert all(lo < f < hi for _, _, f, lo, hi, _ in rows)
    # 目標期（2013 年）4 季的係數，幾何平均 = 1
    assert math.prod(r[2] for r in rows if r[0] == 2013) == pytest.approx(1.0, abs=1e-9)
    # 新宿区真實的走勢：2010 → 2013 漲了 e^0.15
    assert math.prod(r[2] for r in rows if r[0] == 2010) ** 0.25 == pytest.approx(math.exp(0.15), rel=0.03)

    assert cli.main(["query", "--adjusted", "--city", "13104", "--db", str(hedonic_db)]) == 0
    out = capsys.readouterr().out
    assert "修正後" in out and "2013Q1〜2013Q4" in out and "2010Q1" in out
    assert out.rstrip().endswith(DISCLAIMER)


# --- 車站 ---


def test_fetch_stations_then_load(monkeypatch, tmp_path, capsys):
    from test_stations import collection, point_feature, segment_feature

    from tokyo_housing_lab.sources.reinfolib_tiles import tile_of

    xit = (FIXTURES / "xit001_13202_2024Q1_p01_sample.json").read_bytes()
    tachikawa = tile_of(139.414, 35.698)
    stations = collection(
        segment_feature("G1", "a", "立川", "JR", "中央線", [[139.4130, 35.6980], [139.4146, 35.6980]], {2023: 300}),
        segment_feature("G2", "c", "西国立", "JR", "南武線", [[139.4240, 35.6935], [139.4246, 35.6935]], {2023: 20}),
    )
    points = collection(
        point_feature(139.4138, 35.6980, district="132020070", price="4,800万円", area="60㎡"),
        point_feature(139.4243, 35.6935, district="132020140", price="1,100万円", area="40㎡"),
        point_feature(139.4243, 35.6935, district="132020010", price="9,300万円", area="70㎡", category="成約価格情報"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        api = request.url.path.rsplit("/", 1)[1]
        p = request.url.params
        if api == "XIT001":
            return httpx.Response(200, content=xit) if p["year"] == "2024" and p["quarter"] == "1" else httpx.Response(404)
        here = (int(p["x"]), int(p["y"])) == tachikawa
        if api == "XKT015":
            return httpx.Response(200, content=stations if here else collection())
        return httpx.Response(200, content=points if here and p["from"] == "20241" else collection())

    clock = FakeClock()
    monkeypatch.setattr(
        cli,
        "_make_client",
        lambda timeout=30.0: ReinfolibClient(
            ApiKey(FAKE_KEY), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep
        ),
    )
    raw, db = tmp_path / "raw", tmp_path / "t.duckdb"
    assert cli.main(["fetch", "--raw-dir", str(raw), "--start", "2024Q1"]) == 0
    assert cli.main(["fetch-stations", "--raw-dir", str(raw), "--start", "2024"]) == 0
    assert "抓取完成" in capsys.readouterr().out
    assert (raw / "tiles" / f"xkt015_z11_{tachikawa[0]}_{tachikawa[1]}.json.gz").is_file()

    assert cli.main(["load", "--raw-dir", str(raw), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "車站 2 站" in out and "成交點 3 筆" in out
    # 01 的 3 筆裡對上 2 筆（栄町→立川、富士見町→西国立）；02 的曙町對上 1 筆
    assert "對上成交 3 筆" in out
    with duckdb.connect(str(db)) as con:
        got = con.execute(
            "SELECT t.district_name, t.price_classification, s.name FROM transactions t "
            "JOIN transaction_station ts USING (raw_file, row_index) JOIN station s USING (station_id) ORDER BY 1, 2"
        ).fetchall()
    assert got == [("富士見町", "01", "西国立"), ("曙町", "02", "西国立"), ("栄町", "01", "立川")]


# --- 車站分析 ---


@pytest.fixture(scope="module")
def station_db(tmp_path_factory):
    from test_station_analysis import build_station_db

    return build_station_db(tmp_path_factory.mktemp("st") / "st.duckdb")


def test_analyze_by_station(station_db, capsys):
    assert cli.main(["analyze", "buy", "--by", "station", "--db", str(station_db)]) == 0
    out = capsys.readouterr().out
    assert "購買模式：品質調整" in out and "各車站" in out and "2023Q2〜2026Q1" in out
    assert out.index("西新宿") < out.index("大久保")  # 貴的在前面
    assert "沒有到車站的距離" in out and out.rstrip().endswith(DISCLAIMER)


def test_analyze_by_station_line(station_db, capsys):
    assert cli.main(["analyze", "trend", "--by", "station", "--line", "中央線", "--db", str(station_db)]) == 0
    out = capsys.readouterr().out
    assert "JR東日本中央線 各站（依路線上的順序）" in out and "2016 → 2025 年" in out
    lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
    assert lines[0].split()[:2] == ["新宿", "樣本不足"] and lines[1].split()[0] == "大久保"  # 依路線順序，沒成交的也列


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["analyze", "buy", "--by", "station", "--price", "01"], "不能指定 --price"),
        (["analyze", "buy", "--line", "中央線"], "只能和 --by station"),
        (["analyze", "buy", "--by", "station", "--line", "山手線"], "找不到路線"),
        (["analyze", "trend", "--by", "station", "--from", "2010"], "沒有 2010 年"),
    ],
)
def test_analyze_by_station_errors(station_db, capsys, argv, message):
    assert cli.main([*argv, "--db", str(station_db)]) == 1
    assert message in capsys.readouterr().err


def test_map_with_stations(station_db, tmp_path, capsys):
    out_file = tmp_path / "m.html"
    assert cli.main(["map", "--db", str(station_db), "--out", str(out_file)]) == 0
    assert "2 個車站" in capsys.readouterr().out
    assert '"stations":{"years":[2016' in out_file.read_text(encoding="utf-8")


# --- 圖層 ---


def test_fetch_hazards_then_load(tmp_path, monkeypatch, capsys):
    import tokyo_housing_lab.sources.reinfolib_tiles as rt
    from test_layers import SMALL, collection, polygon_feature, school_props

    from tokyo_housing_lab.sources.reinfolib_tiles import TileQuery, tile_of

    fx, fy = tile_of(139.805, 35.685, 14)
    sx, sy = tile_of(139.805, 35.685, 11)
    monkeypatch.setattr(
        rt, "hazard_queries", lambda layers: [TileQuery("XKT026", fx, fy, z=14), TileQuery("XKT004", sx, sy)]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        api = request.url.path.rsplit("/", 1)[1]
        if api == "XKT026":
            return httpx.Response(200, content=collection(polygon_feature({"A31a_205": 2}, SMALL)))
        return httpx.Response(200, content=collection(polygon_feature(school_props(), SMALL)))

    clock = FakeClock()
    monkeypatch.setattr(
        cli,
        "_make_client",
        lambda timeout=30.0: ReinfolibClient(
            ApiKey(FAKE_KEY), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep
        ),
    )
    raw, db = tmp_path / "raw", tmp_path / "t.duckdb"
    assert cli.main(["fetch-hazards", "--raw-dir", str(raw)]) == 0
    assert "新存檔 2" in capsys.readouterr().out
    assert cli.main(["load", "--raw-dir", str(raw), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "圖層：洪水 1 個圖磚、小學校區 1 個圖磚、洪水的浸水格" in out and "小學校區 1 校" in out
    assert FAKE_KEY not in out
