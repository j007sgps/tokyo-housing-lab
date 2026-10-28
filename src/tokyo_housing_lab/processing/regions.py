"""處理層：東京都本島 53 個市區町村的行政區界（地圖用的簡化版，map/tokyo_municipalities.geojson）。

圖層的圖磚範圍、淹水格網的遮罩、各區的面積比例都用這份。簡化過，所以邊界附近有約 100m 的誤差。
"""

import json
from functools import cache
from importlib.resources import files


@cache
def municipality_geojson() -> dict:
    return json.loads(
        files("tokyo_housing_lab.map").joinpath("tokyo_municipalities.geojson").read_text(encoding="utf-8")
    )


def municipality_polygons() -> list[tuple[str, list]]:
    """(市區町村代碼, GeoJSON 多邊形的環的 list)；MultiPolygon 拆成多筆。"""
    from tokyo_housing_lab.processing.raster import geojson_polygons

    return [
        (f["properties"]["code"], p)
        for f in municipality_geojson()["features"]
        for p in geojson_polygons(f["geometry"])
    ]
