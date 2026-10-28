"""產生地圖用的東京都市區町村邊界：src/tokyo_housing_lab/map/tokyo_municipalities.geojson

來源：スマートニュース メディア研究所「japan-topography」（簡化 1%），
原始資料為國土交通省「國土數值情報（行政區域データ）」。使用時須依國交省規定註明出處。

處理：
- 去掉伊豆・小笠原的離島（代碼 13361 以後）和「所属未定地」（沒有代碼）
- 座標四捨五入到小數 4 位（約 10 公尺），刪掉因此變成重複的點
- properties 只留 code（5 碼）和 name

用法：uv run python scripts/prepare_boundaries.py [下載好的 N03-21_13_210101.json]
（不給路徑就直接從 GitHub 下載）
"""

import json
import sys
from pathlib import Path

import httpx

SOURCE_URL = (
    "https://raw.githubusercontent.com/smartnews-smri/japan-topography/main/"
    "data/municipality/geojson/s0010/N03-21_13_210101.json"
)
OUT = Path(__file__).resolve().parents[1] / "src/tokyo_housing_lab/map/tokyo_municipalities.geojson"
FIRST_ISLAND_CODE = 13361  # 大島町。之後都是伊豆・小笠原的島


def _ring(coords: list) -> list:
    out = []
    for x, y in coords:
        p = [round(x, 4), round(y, 4)]
        if not out or out[-1] != p:
            out.append(p)
    return out


def _geometry(g: dict) -> dict:
    if g["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [_ring(r) for r in g["coordinates"]]}
    return {"type": "MultiPolygon", "coordinates": [[_ring(r) for r in poly] for poly in g["coordinates"]]}


def main() -> None:
    if len(sys.argv) > 1:
        src = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        src = httpx.get(SOURCE_URL, timeout=60).raise_for_status().json()
    features = []
    for f in src["features"]:
        p = f["properties"]
        code = p.get("N03_007")
        if not code or int(code) >= FIRST_ISLAND_CODE:
            continue
        features.append({"type": "Feature", "properties": {"code": code, "name": p["N03_004"]}, "geometry": _geometry(f["geometry"])})
    features.sort(key=lambda f: f["properties"]["code"])
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(features)} 個市區町村 → {OUT}（{OUT.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    main()
