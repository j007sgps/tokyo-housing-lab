"""分析層：估價（M5）。只給區間，不給單一數字。

ln 單價 = 區的現在水準 + 町名的地點效果 + 建物效果（+ 成約效果）
- 模型：車站分析共用的視窗模型（analysis/window.py）。「現在」= 第 0 期（最新 4 季）4 季的平均
- 區的現在水準：時間:區:0 + 4 季的季內效果平均
- 町名的地點效果：町名係數 + 用途地域、容積率的效果，依該町名的成交平均（使用者不會知道用途地域、容積率）
- 建物效果：面積、屋齡（分段線性）、面積×第 0 期、舊耐震、房型、構造、改裝
- 成約效果：成約:區 + 成約×時間:0（取引、成約都夠多的區才有）

全部攤成一張係數表（CoefTable）：Python 和地圖上的 JS 都用它算，測試保證和 design_matrix／predict 的結果相同。

區間：split conformal。第 0 期的成交隨機留 CALIBRATION_SHARE 不參加估計，用係數表預測（和實際估價一樣：
町名平均的地點、4 季平均），實際 − 預測的分位數就是區間（非對稱）。另外留 TEST_SHARE 檢驗覆蓋率。
分位數依「屋齡 × 23 區／多摩」分組（interval_group）：全部共用一組時，屋齡 40 年以上的 80% 區間只蓋到 74%、
10 年以內的蓋到 86%、多摩 75%；分組後每一群都在名目值 ±4 個百分點內（2026-10 實測）。
最後的模型用全部資料重估、區間沿用校正的分位數（訓練資料比較多，區間稍微保守）。

區間是「同樣條件的物件的成交價」會落在的範圍；模型沒有樓層、方位、離車站的距離、管理狀態，所以很寬。
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from tokyo_housing_lab.analysis import hedonic as hd
from tokyo_housing_lab.analysis.window import WindowSample, fit_window

LEVELS = (0.5, 0.8)  # 區間的信心水準
CALIBRATION_SHARE = 0.2  # 分 10 組，每組要有幾百筆
AGE_BANDS = (10, 20, 30, 40)  # 區間分組的屋齡界線（年）
KU_MAX = "13123"  # 23 區的代碼是 13101〜13123
TEST_SHARE = 0.1
MIN_GROUP_N = 100  # 校正筆數少於這個的組，用全部的分位數
SEED = 0
PRICE_LEVELS = {"01": "取引", "02": "成約"}
# 房型的類別（hedonic.floor_plan_group 的輸出；+S 另外算）。地圖的表單直接選類別
PLAN_GROUPS = ("1R/1K", "1DK", "1LDK", "2DK", "2LDK", "3DK", "3LDK", "4+")


@dataclass(frozen=True)
class Subject:
    city: str  # 市區町村代碼
    district: str  # 町名（XIT001 的 DistrictName）
    area: float  # ㎡
    building_year: int
    floor_plan: str = "2LDK"
    structure: str = "RC"
    renovation: str = "未改装"  # 未改装／改装済み


@dataclass
class CoefTable:
    """估價用的係數（全部是 ln 單價的加項）。"""

    spec: hd.Spec
    year: int  # 屋齡 = year − 建築年（最新一季的年）
    city_now: dict[str, float]  # 區 → 現在水準
    district_loc: dict[str, dict[str, float]]  # 區 → 町名 → 地點效果
    building: dict[str, float]  # 欄位名 → 係數（ln面積、屋齡、舊耐震、房型:…、構造:…、改裝:…、面積×時間:0）
    seiyaku: dict[str, float]  # 區 → 成約效果（沒有 = 這區只給取引）
    plans: list[str]  # 可以選的房型類別（2LDK 是基準）
    structures: list[str]  # 可以選的構造（RC、SRC…；資料裡有的）
    renovations: list[str]

    def building_effect(self, area: float, age: float, plan_group: str, plus_s: bool, structure: str, renovation: str,
                        old_seismic: bool) -> float:  # fmt: skip
        b = self.building
        terms = hd.building_terms(np.array([float(area)]), np.array([float(age)]), self.spec)
        out = sum(b.get(name, 0.0) * float(v[0]) for name, v in terms.items())
        out += b.get("面積×時間:0", 0.0) * math.log(area / self.spec.standard_area)
        out += b.get("舊耐震", 0.0) * old_seismic
        out += b.get(f"房型:{plan_group}", 0.0) + b.get("房型:+S", 0.0) * plus_s
        out += b.get(f"構造:{hd.structure_group(structure)}", 0.0) + b.get(
            f"改裝:{hd.renovation_group(renovation)}", 0.0
        )
        return out

    def log_price(self, s: Subject, price_classification: str = "01") -> float:
        """ln ㎡ 單價（點估計；只用來算區間，不對外顯示）。"""
        if s.city not in self.city_now:
            raise ValueError(f"{s.city} 最新 4 季沒有成交，估不出來")
        locs = self.district_loc[s.city]
        if s.district not in locs:
            raise ValueError(f"找不到町名「{s.district}」")
        group, plus_s = hd.floor_plan_group(s.floor_plan)
        age = max(0, self.year - s.building_year)
        old = s.building_year <= self.spec.old_seismic_last_year
        out = self.city_now[s.city] + locs[s.district]
        out += self.building_effect(s.area, age, group, plus_s, s.structure, s.renovation, old)
        if price_classification == hd.SEIYAKU:
            if s.city not in self.seiyaku:
                raise ValueError(f"{s.city} 的成約價格不夠多，沒有成約的估價")
            out += self.seiyaku[s.city]
        return out


def _coef(model: hd.Model, name: str) -> float:
    return model.coef(name)[0]


def coef_table(model: hd.Model, last_year: int) -> CoefTable:
    d, beta = model.design, model.fit.beta
    names = d.names
    # 區的現在水準：第 0 期有資料（或第 0 期就是基準）的區
    have_now = {c for c, p in zip(d.city.tolist(), d.period.tolist(), strict=True) if p == 0}
    quarter = np.mean([_coef(model, f"季內:0:{k}") for k in range(4)])
    city_now = {c: _coef(model, f"時間:{c}:0") + quarter for c in sorted(have_now)}
    # 町名的地點效果：該町名每一筆成交的地點欄位（町名、用途地域、容積率）× 係數，取平均
    loc_cols = np.array([any(nm.startswith(p) for p in hd.LOCATION_PREFIXES) for nm in names])
    loc_beta = np.where(loc_cols, beta, 0.0)
    row_loc = d.X @ loc_beta
    sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for c, dist, v in zip(model.sample.city.tolist(), model.sample.district.tolist(), row_loc.tolist(), strict=True):
        acc = sums[(c, dist)]
        acc[0] += v
        acc[1] += 1
    district_loc: dict[str, dict[str, float]] = defaultdict(dict)
    for (c, dist), (total, n) in sorted(sums.items()):
        if c in city_now:
            district_loc[c][dist] = total / n
    building = {
        nm: float(beta[i])
        for nm, i in d.index.items()
        if nm.split(":", 1)[0] in ("房型", "構造", "改裝") or nm in ("舊耐震", "面積×時間:0")
        or nm.startswith(("ln面積", "屋齡"))
    }  # fmt: skip
    seiyaku = {
        c: _coef(model, f"成約:{c}") + _coef(model, "成約×時間:0") for c in sorted(d.seiyaku_cities) if c in city_now
    }

    def options(prefix: str, base: str) -> list[str]:
        return [base] + sorted(nm.split(":", 1)[1] for nm in d.index if nm.startswith(prefix + ":") and nm != "房型:+S"
                               and nm.split(":", 1)[1] not in ("不明", "其他"))  # fmt: skip

    return CoefTable(
        model.spec, last_year, city_now, dict(district_loc), building, seiyaku,
        options("房型", "2LDK"), options("構造", "RC"), options("改裝", "未改装"),
    )  # fmt: skip


def interval_group(city: str, building_year: int, year: int) -> int:
    """區間的分組：屋齡帶（0〜4）× 2 + 多摩（1）。"""
    band = int(np.digitize(max(0, year - building_year), AGE_BANDS))
    return band * 2 + int(city > KU_MAX)


def group_label(g: int) -> str:
    bounds = (0, *AGE_BANDS)
    band = g // 2
    age = f"屋齡 {bounds[band]}〜{AGE_BANDS[band]} 年" if band < len(AGE_BANDS) else f"屋齡 {AGE_BANDS[-1]} 年以上"
    return f"{'多摩' if g % 2 else '23 區'}・{age}"


@dataclass
class Estimator:
    table: CoefTable
    quantiles: dict[int, dict[float, tuple[float, float]]]  # 分組 → 信心水準 → (ln 下限偏差, ln 上限偏差)
    calibration_n: dict[int, int]  # 分組 → 校正筆數
    coverage: dict[str, dict[float, float]] = field(default_factory=dict)  # 群組 → 信心水準 → 檢驗資料的覆蓋率
    coverage_n: dict[str, int] = field(default_factory=dict)
    period_label: str = ""  # 「現在」= 2025Q3〜2026Q2
    model: hd.Model | None = field(default=None, repr=False)  # 類似成交要用（地圖只用係數表）
    last: int = 0  # 最新一季（年 × 4 + 季 − 1）

    def quantiles_for(self, s: Subject) -> dict[float, tuple[float, float]]:
        return self.quantiles[interval_group(s.city, s.building_year, self.table.year)]


@dataclass(frozen=True)
class Range:
    level: float
    unit_low: float  # 日圓／㎡
    unit_high: float
    total_low: float  # 日圓
    total_high: float


@dataclass(frozen=True)
class Estimate:
    subject: Subject
    ranges: dict[str, list[Range]]  # 價格種類（01／02）→ 各信心水準的區間；成約不夠的區只有 01
    group: int  # 區間用的分組（interval_group）


def estimate(est: Estimator, s: Subject) -> Estimate:
    ranges = {}
    q = est.quantiles_for(s)
    for pc in PRICE_LEVELS:
        if pc == hd.SEIYAKU and s.city not in est.table.seiyaku:
            continue
        mu = est.table.log_price(s, pc)
        ranges[pc] = [
            Range(lv, math.exp(mu + lo), math.exp(mu + hi), math.exp(mu + lo) * s.area, math.exp(mu + hi) * s.area)
            for lv, (lo, hi) in sorted(q.items())
        ]
    return Estimate(s, ranges, interval_group(s.city, s.building_year, est.table.year))


def subject_of(sample: hd.Sample, i: int) -> Subject:
    return Subject(
        str(sample.city[i]), str(sample.district[i]), float(sample.area[i]), int(sample.building_year[i]),
        str(sample.floor_plan[i]), str(sample.structure[i]), str(sample.renovation[i]),
    )  # fmt: skip


def table_predictions(table: CoefTable, sample: hd.Sample) -> np.ndarray:
    """每一筆成交用係數表預測（和估價一樣：町名平均的地點、4 季平均）；估不出來的是 nan。"""
    out = np.full(len(sample), np.nan)
    pcs = sample.price_classification.astype(str)
    for i in range(len(sample)):
        try:
            out[i] = table.log_price(subject_of(sample, i), pcs[i])
        except ValueError:
            pass
    return out


def _groups(sample: hd.Sample, year: int) -> dict[str, np.ndarray]:
    age = year - sample.building_year.astype(int)
    area = sample.area.astype(float)
    ku = np.array([c <= "13123" for c in sample.city.astype(str).tolist()])
    pc = sample.price_classification.astype(str)
    return {
        "全部": np.ones(len(sample), dtype=bool),
        "取引": pc == "01", "成約": pc == "02",
        "23 區": ku, "多摩": ~ku,
        "屋齡 <10 年": age < 10, "屋齡 10〜20 年": (age >= 10) & (age < 20), "屋齡 20〜30 年": (age >= 20) & (age < 30),
        "屋齡 30〜40 年": (age >= 30) & (age < 40), "屋齡 ≥40 年": age >= 40,
        "<30㎡": area < 30, "30〜50㎡": (area >= 30) & (area < 50), "50〜70㎡": (area >= 50) & (area < 70), "≥70㎡": area >= 70,
    }  # fmt: skip


def calibrate(ws: WindowSample, seed: int = SEED) -> tuple[dict, dict[int, int], dict, dict]:
    """回傳 (分位數, 校正筆數, 覆蓋率, 檢驗筆數)。要多估一次模型（不含校正、檢驗資料）。"""
    u = np.random.default_rng(seed).random(len(ws.sample))
    latest = ws.period == 0
    calib = latest & (u < CALIBRATION_SHARE)
    test = latest & (u >= CALIBRATION_SHARE) & (u < CALIBRATION_SHARE + TEST_SHARE)
    model = fit_window(ws.subset(~(calib | test)))
    table = coef_table(model, ws.last_year)

    def residuals(mask: np.ndarray) -> tuple[np.ndarray, hd.Sample]:
        s = ws.sample.subset(mask)
        e = s.log_price - table_predictions(table, s)
        ok = ~np.isnan(e)
        return e[ok], s.subset(ok)

    def keys(s: hd.Sample) -> np.ndarray:
        return np.array([interval_group(c, int(b), ws.last_year)
                         for c, b in zip(s.city.astype(str).tolist(), s.building_year.tolist(), strict=True)])  # fmt: skip

    e_cal, s_cal = residuals(calib)
    k_cal = keys(s_cal)
    quantiles, n_cal = {}, {}
    for g in range(2 * (len(AGE_BANDS) + 1)):
        e = e_cal[k_cal == g] if (k_cal == g).sum() >= MIN_GROUP_N else e_cal  # 太少的組用全部的
        quantiles[g] = {lv: (float(np.quantile(e, (1 - lv) / 2)), float(np.quantile(e, (1 + lv) / 2))) for lv in LEVELS}
        n_cal[g] = len(e)
    e_test, s_test = residuals(test)
    k_test = keys(s_test)
    coverage, counts = {}, {}
    for name, mask in _groups(s_test, ws.last_year).items():
        if mask.sum():
            cov = {}
            for lv in LEVELS:
                lo = np.array([quantiles[g][lv][0] for g in k_test[mask]])
                hi = np.array([quantiles[g][lv][1] for g in k_test[mask]])
                cov[lv] = float(np.mean((e_test[mask] >= lo) & (e_test[mask] <= hi)))
            coverage[name], counts[name] = cov, int(mask.sum())
    return quantiles, n_cal, coverage, counts


def build_estimator(ws: WindowSample, model: hd.Model | None = None, seed: int = SEED) -> Estimator:
    """model：用全部資料估好的視窗模型（車站分析已經估過的話傳進來）。"""
    quantiles, n_cal, coverage, counts = calibrate(ws, seed)
    model = model if model is not None else fit_window(ws)
    first = ws.last - 3
    label = f"{first // 4}Q{first % 4 + 1}〜{ws.last_year}Q{ws.last_quarter}"
    return Estimator(coef_table(model, ws.last_year), quantiles, n_cal, coverage, counts, label, model, ws.last)


# ---- 類似成交 ----

COMPARABLE_QUARTERS = 12  # 最近 3 年
COMPARABLE_AREA_RATIO = 1.3  # 面積 ±30%
COMPARABLE_YEARS = 10  # 建築年 ±10 年
COMPARABLES_MAX = 8


@dataclass(frozen=True)
class Comparable:
    year: int
    quarter: int
    price_classification: str
    area: float
    building_year: int
    floor_plan: str
    unit_price: float  # 日圓／㎡（原始）
    time_adjusted: float  # 時點修正到現在（建物條件不變）
    converted: float  # 換算成這間的條件（時點＋建物特性的差；同一種價格水準）


def _time_effect(model: hd.Model, city: str, period: int, subperiod: int, area: float, seiyaku: bool) -> float:
    """某區某期某季的時間效果（含面積×期間、成約×期間）。"""
    c = model.coef
    out = c(f"時間:{city}:{period}")[0] + c(f"季內:{period}:{subperiod}")[0]
    out += c(f"面積×時間:{period}")[0] * math.log(area / model.spec.standard_area)
    return out + (c(f"成約×時間:{period}")[0] if seiyaku else 0.0)


def comparables(est: Estimator, s: Subject) -> list[Comparable]:
    """同町名、最近 3 年、面積和建築年相近的成交（模型實際用到的，離群值不算），依相似度排序。

    換算成這間 = 這間的係數表預測 × exp(那筆的殘差)：時點、面積、屋齡、房型等的差都用模型的係數換算，
    剩下的（樓層、方位…）就是那筆自己的殘差，原樣保留。
    """
    model = est.model
    if model is None:
        raise ValueError("沒有模型，無法找類似成交")
    ms, d = model.sample, model.design
    qi = ms.year.astype(int) * 4 + ms.quarter.astype(int) - 1
    area = ms.area.astype(float)
    by = ms.building_year.astype(int)
    pick = (
        (ms.city.astype(str) == s.city) & (ms.district.astype(str) == s.district) & (qi > est.last - COMPARABLE_QUARTERS)
        & (np.abs(np.log(area / s.area)) <= math.log(COMPARABLE_AREA_RATIO)) & (np.abs(by - s.building_year) <= COMPARABLE_YEARS)
    )  # fmt: skip
    rows = np.nonzero(pick)[0]
    if not len(rows):
        return []
    score = (
        np.abs(np.log(area[rows] / s.area)) / math.log(COMPARABLE_AREA_RATIO)
        + np.abs(by[rows] - s.building_year) / COMPARABLE_YEARS
    )
    rows = rows[np.lexsort((-qi[rows], score))][:COMPARABLES_MAX]
    out = []
    for i in rows.tolist():
        pc = str(ms.price_classification[i])
        seiyaku = pc == hd.SEIYAKU
        then = _time_effect(model, s.city, int(d.period[i]), int(d.subperiod[i]), area[i], seiyaku)
        now = np.mean([_time_effect(model, s.city, 0, k, area[i], seiyaku) for k in range(4)])
        log_p = float(ms.log_price[i])
        converted = est.table.log_price(s, pc) + float(model.fit.resid[i])
        out.append(Comparable(
            int(ms.year[i]), int(ms.quarter[i]), pc, float(area[i]), int(by[i]), str(ms.floor_plan[i]),
            math.exp(log_p), math.exp(log_p + now - then), math.exp(converted),
        ))  # fmt: skip
    return out
