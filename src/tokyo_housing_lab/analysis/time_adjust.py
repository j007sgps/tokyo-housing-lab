"""時點修正：把舊的成交價換算成「最新 4 季的平均水準」。

用品質調整模型（hedonic.py），期間設成「以最新一季為終點、每 4 季一期」，再加上全東京共用的季內走勢，
得到每個「區 × 季」的係數。係數存進 DuckDB 的 time_adjustment 表，SQL join 就能用（M5 估價也讀這張表）。
"""

from collections import Counter
from dataclasses import dataclass

from tokyo_housing_lab.analysis import hedonic as hd
from tokyo_housing_lab.storage.duckdb_store import DuckDBStore


@dataclass(frozen=True)
class TimeAdjustmentResult:
    target: str  # 例：2025Q2〜2026Q1
    last: tuple[int, int]  # 最新一季
    adjustments: list[hd.Adjustment]
    n: int  # 估計用的成交筆數
    outliers: int

    def rows(self, region: str) -> list[dict]:
        """time_adjustment 表的列（不含 computed_at）。"""
        return [
            {
                "region": region,
                "city": a.city,
                "year": a.year,
                "quarter": a.quarter,
                "target": self.target,
                "factor": a.factor,
                "factor_low": a.low,
                "factor_high": a.high,
                "basis": a.basis,
                "n": a.n,
            }
            for a in self.adjustments
        ]


def quarter_label(qi: int) -> str:
    return f"{qi // 4}Q{qi % 4 + 1}"


def compute_time_adjustments(store: DuckDBStore, region: str = "13") -> TimeAdjustmentResult:
    """全部歷史資料估一次（約 15 秒），算出每個區、從第一季到最新一季的係數。"""
    sample = hd.Sample.from_columns(store.query_columns("hedonic_sample", region=region))
    if len(sample) == 0:
        raise ValueError("沒有可用的成交資料，請先執行 load")
    qi = sample.year.astype(int) * 4 + sample.quarter.astype(int) - 1
    first, last = int(qi.min()), int(qi.max())
    ly, lq = last // 4, last % 4 + 1
    model = hd.fit_hedonic(sample, hd.window_period(sample, ly, lq), subperiod=hd.window_subperiod(sample, ly, lq))
    cities = sorted(set(sample.city.astype(str).tolist()))
    quarters = [(i // 4, i % 4 + 1) for i in range(first, last + 1)]
    adjustments = hd.time_adjustments(model, cities, quarters, ly, lq)
    target = f"{quarter_label(last - 3)}〜{quarter_label(last)}"
    return TimeAdjustmentResult(target, (ly, lq), adjustments, model.fit.n, model.outliers)


def basis_counts(result: TimeAdjustmentResult) -> Counter:
    """city／tokyo／None（無法修正）各幾列。"""
    return Counter(a.basis for a in result.adjustments)
