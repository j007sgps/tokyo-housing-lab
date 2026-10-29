"""分析層：最近幾期的視窗模型（車站分析、估價共用）。

以最新一季為終點、每 4 季一期，加上期間內的季內走勢；只估最近 HEDONIC_LATEST_WINDOWS 期（約 11 年）。
第 0 期 = 最新 4 季，也就是購買模式、時點修正、估價的「現在」。
"""

from dataclasses import dataclass

import numpy as np

from tokyo_housing_lab.analysis import hedonic as hd
from tokyo_housing_lab.analysis.municipalities import HEDONIC_LATEST_WINDOWS
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore


@dataclass
class WindowSample:
    sample: hd.Sample  # 最近幾期的全部成交（還沒去掉離群值）
    period: np.ndarray  # 0 = 最新 4 季，−1 = 再前 4 季…
    subperiod: np.ndarray  # 期間內第幾季（0 = 最早）
    qi: np.ndarray  # 年 × 4 + 季 − 1
    station_ids: np.ndarray  # 最寄駅（沒對上的是空字串）
    last: int  # 最新一季的 qi

    @property
    def last_year(self) -> int:
        return self.last // 4

    @property
    def last_quarter(self) -> int:
        return self.last % 4 + 1

    def subset(self, mask: np.ndarray) -> "WindowSample":
        return WindowSample(
            self.sample.subset(mask),
            self.period[mask],
            self.subperiod[mask],
            self.qi[mask],
            self.station_ids[mask],
            self.last,
        )


def load_window_sample(store: DuckDBStore, region: str = "13", windows: int = HEDONIC_LATEST_WINDOWS) -> WindowSample:
    cols = store.query_columns("hedonic_sample", region=region)
    sample = hd.Sample.from_columns(cols)
    if not len(sample):
        raise ValueError("沒有成交資料，請先執行 fetch 和 load")
    station_ids = np.asarray(cols["station_id"]).astype(str)
    qi = sample.year.astype(int) * 4 + sample.quarter.astype(int) - 1
    last = int(qi.max())
    period = hd.window_period(sample, last // 4, last % 4 + 1)
    subperiod = hd.window_subperiod(sample, last // 4, last % 4 + 1)
    ws = WindowSample(sample, period, subperiod, qi, station_ids, last)
    return ws.subset(period > -windows)


def fit_window(ws: WindowSample) -> hd.Model:
    return hd.fit_hedonic(ws.sample, ws.period, subperiod=ws.subperiod)
