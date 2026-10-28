"""品質調整指數（hedonic 迴歸）：同樣條件的物件，價格怎麼變。

中位數會受「那年賣了什麼物件」影響（新舊、大小、地段）。這裡用 OLS 把這些因素拆開：

    ln(㎡ 單價) = 町名 + 市區町村×期間 + 面積 + 面積×期間 + 屋齡 + 舊耐震 + 房型 + 構造 + 改裝 + 用途地域 + 容積率
                  + 成約×市區町村 + 成約×期間 + 誤差

- 「市區町村×期間」就是指數本身（每區以第一個期間為基準 = 0，避免和町名共線）
- 成約的兩項吸收取引與成約的落差（各區不同、也可逐年變動），指數一律以取引的水準表示
- 某區某期間的水準 L = 期間係數 + 該區的地點組成（町名、用途地域、容積率的效果，依該區成交筆數平均）
  + 典型物件（55㎡、屋齡 17 年、RC、2LDK、未改裝）的特性
- 誤差：以町名為 cluster 的 cluster-robust 共變異矩陣（同一町名的成交彼此相關，iid 假設會讓區間太窄）
- 離群值：先估一次，殘差超過 5σ（σ 用 MAD 估）的去掉再估一次

已知限制：API 沒有車站距離和樓層；特性的係數全東京共用；同一筆成交可能同時在取引和成約裡（無法去重複）。
"""

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from tokyo_housing_lab.analysis.trend import (
    INVEST_WINDOW_YEARS,
    MIN_N,
    Z95,
    Growth,
    LongTrend,
    Price,
    growth_from_log,
    price_from_log,
    trend_from_points,
)

MIN_DISTRICT_N = 10  # 筆數少於這個的町名，併進該區的「其他」
OUTLIER_SIGMAS = 5.0
OTHER_DISTRICT = "（其他）"
ALL_DISTRICTS = "（全部）"
SEIYAKU = "02"

# 用途地域：常見的分開，其餘併成「其他」；基準是商業地域
CITY_PLANNING_LEVELS = (
    "準工業地域",
    "第１種中高層住居専用地域",
    "近隣商業地域",
    "第１種住居地域",
    "第１種低層住居専用地域",
    "第２種住居地域",
    "第２種中高層住居専用地域",
    "工業地域",
    "準住居地域",
)
CITY_PLANNING_BASE = "商業地域"
# 容積率（%）的分段上限；基準是 ≤200
FAR_BINS = ((200, "≤200"), (300, "≤300"), (400, "≤400"), (600, "≤600"), (math.inf, ">600"))

# 地點組成：這些欄位的效果算進「該區的平均地點」，不算進典型物件
LOCATION_PREFIXES = ("町名:", "用途地域:", "容積率:")


@dataclass(frozen=True)
class Spec:
    area_knots: tuple[float, ...] = (25, 40, 55, 70, 90)  # ㎡；ln(面積) 的分段線性
    age_knots: tuple[float, ...] = (5, 10, 20, 30, 40)  # 年；屋齡的分段線性
    old_seismic_last_year: int = 1981  # 建築年 ≤ 這年 → 舊耐震
    standard_area: float = 55  # 典型物件：東京的中位數條件
    standard_age: float = 17
    # 面積的效果隨期間變動：加上 ln(面積 / standard_area) × 期間（全東京共用，第一個期間為基準）。
    # 實測 2016→2025 小坪數（<30㎡）和大坪數（≥70㎡）的相對價格變了約 15%，全期共用的面積係數會讓
    # 小坪數多的地方（車站）前期高估、後期低估。以典型物件的面積為中心，典型物件的水準不受影響
    area_by_period: bool = True

    def standard_label(self) -> str:
        return f"{self.standard_area:g}㎡・屋齡 {self.standard_age:g} 年・RC・2LDK・未改裝"


# ---- 樣本 ----


@dataclass
class Sample:
    """一列一筆成交（欄位對應 queries/hedonic_sample.sql）。全部是等長的 numpy 陣列。"""

    city: np.ndarray
    district: np.ndarray
    year: np.ndarray
    quarter: np.ndarray
    price_classification: np.ndarray
    log_price: np.ndarray
    area: np.ndarray
    building_year: np.ndarray
    floor_plan: np.ndarray
    structure: np.ndarray
    renovation: np.ndarray
    city_planning: np.ndarray
    floor_area_ratio: np.ndarray  # 不明 = nan

    @classmethod
    def from_columns(cls, cols: Mapping[str, np.ndarray]) -> "Sample":
        far = cols["floor_area_ratio"]
        far = np.ma.filled(np.ma.asarray(far).astype(float), np.nan)
        out = {f.name: np.asarray(cols[f.name]) for f in fields(cls) if f.name != "floor_area_ratio"}
        return cls(**out, floor_area_ratio=far)

    def __len__(self) -> int:
        return len(self.log_price)

    def subset(self, mask: np.ndarray) -> "Sample":
        return Sample(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})


def yearly_period(sample: Sample) -> np.ndarray:
    return sample.year.astype(int)


def window_period(sample: Sample, last_year: int, last_quarter: int) -> np.ndarray:
    """以最新一季為終點、每 4 季一個期間：0 = 最新 4 季，−1 = 再前 4 季…（購買模式用，和中位數的最新 4 季對齊）。"""
    qi = sample.year.astype(int) * 4 + sample.quarter.astype(int) - 1
    return (qi - (last_year * 4 + last_quarter - 1) + 3) // 4


def window_subperiod(sample: Sample, last_year: int, last_quarter: int) -> np.ndarray:
    """在 window_period 的 4 季裡是第幾季（0〜3，0 = 最早的那一季）。時點修正用。"""
    qi = sample.year.astype(int) * 4 + sample.quarter.astype(int) - 1
    return (qi - (last_year * 4 + last_quarter - 1) + 3) % 4


# ---- 特性的分類 ----

_PLAN = re.compile(r"(\d+)([A-Z]*)((?:\+[A-Z]+)*)")


def floor_plan_group(plan: str) -> tuple[str, bool]:
    """房型 → (類別, 有沒有 +S)。基準是 2LDK。"""
    if not plan:
        return "不明", False
    m = _PLAN.fullmatch(plan)
    if not m:
        return "其他", False
    rooms, kind, extra = int(m.group(1)), m.group(2), m.group(3)
    plus_s = "+S" in extra
    if rooms >= 4:
        return "4+", plus_s
    if rooms == 0:
        return "其他", plus_s
    if "L" in kind:
        return f"{rooms}LDK", plus_s
    if rooms == 1 and kind in ("R", "K", ""):
        return "1R/1K", plus_s
    return f"{rooms}DK", plus_s


def structure_group(s: str) -> str | None:
    return None if s == "RC" else "SRC" if s == "SRC" else "不明" if not s else "其他"


def renovation_group(s: str) -> str | None:
    return None if s == "未改装" else "改装済み" if s == "改装済み" else "不明"


def city_planning_group(s: str) -> str | None:
    if s == CITY_PLANNING_BASE:
        return None
    return s if s in CITY_PLANNING_LEVELS else "不明" if not s else "其他"


def far_group(x: float) -> str | None:
    if math.isnan(x):
        return "不明"
    for upper, label in FAR_BINS:
        if x <= upper:
            return None if label == FAR_BINS[0][1] else label
    raise AssertionError("unreachable")


def _hinges(x: np.ndarray, knots: tuple[float, ...]) -> list[np.ndarray]:
    return [x] + [np.maximum(0.0, x - k) for k in knots]


def building_terms(area: np.ndarray, age: np.ndarray, spec: Spec) -> dict[str, np.ndarray]:
    """面積、屋齡的連續項（分段線性）。"""
    la = np.log(area)
    out = dict(zip(["ln面積"] + [f"ln面積>{k:g}" for k in spec.area_knots], _hinges(la, tuple(math.log(k) for k in spec.area_knots)), strict=True))
    out |= dict(zip(["屋齡"] + [f"屋齡>{k:g}" for k in spec.age_knots], _hinges(age, spec.age_knots), strict=True))
    return out


# ---- 設計矩陣 ----


@dataclass
class Design:
    X: sp.csr_matrix
    names: list[str]
    index: dict[str, int]
    clusters: np.ndarray  # 每列的町名 cluster 編號
    city: np.ndarray
    period: np.ndarray
    seiyaku: np.ndarray  # bool
    base_period: dict[str, int]  # 各區的基準期間（係數 = 0）
    seiyaku_base_period: int | None
    seiyaku_cities: set[str]  # 取引、成約都夠多，才放成約的區
    big_districts: set[str]
    absorbed: set[str]  # 當成基準的欄位（分類變數在資料裡沒有原本的基準類別時，最常見的那一類）
    pooled_cities: set[str]  # 町名全部併成一個的區（町名和期間分不開時）
    subperiod: np.ndarray | None  # 期間內第幾季（有給才有「季內」欄位）
    subperiod_base: dict[int, int]  # 各期間的基準季：該期有資料的第一季
    area_base_period: int | None = None  # 面積 × 期間的基準期間（spec.area_by_period 才有）


@dataclass
class _Block:
    """一組欄位：每個「水準」一個欄位名（None = 基準，不放），codes 是每列的水準編號（−1 = 這列沒有）。"""

    level_names: list[str | None]
    codes: np.ndarray
    values: np.ndarray | None = None  # None = 全部是 1


def _categorical(prefix: str, column: np.ndarray, group) -> _Block:
    """分類只對不重複的值做一次，再用編號對回每一列（40 萬列逐列處理太慢）。"""
    uniques, inv = np.unique(column, return_inverse=True)
    names = [None if (g := group(u)) is None else f"{prefix}:{g}" for u in uniques.tolist()]
    return _Block(names, inv)


def _continuous(name: str, values: np.ndarray) -> _Block:
    # 值是 0 的列不放：整欄都是 0 的欄位（例如資料裡沒有超過 90㎡ 的物件）就不會出現，也不會被當成共線
    return _Block([name], np.where(values != 0, 0, -1), values)


def _disconnected_cities(district: np.ndarray, cp: np.ndarray, city: np.ndarray, n_district: int, n_cp: int) -> set[int]:
    """町名和「區×期間」當成圖的點、每筆成交連一條線。某區的圖斷成好幾塊時（例如資料少的區，
    每個町名只出現在某一期），各塊之間的時間變化分不開 → 回傳這些區，町名要全部併成一個。"""
    edges = np.unique(np.stack([district, n_district + cp]), axis=1)
    graph = sp.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n_district + n_cp,) * 2)
    _, labels = connected_components(graph, directed=False)
    pairs = np.unique(np.stack([city, labels[district]]), axis=1)
    counts = np.bincount(pairs[0])
    return set(np.nonzero(counts > 1)[0].tolist())


def design_matrix(
    sample: Sample,
    period: np.ndarray,
    spec: Spec,
    template: Design | None = None,
    subperiod: np.ndarray | None = None,
) -> tuple[Design, np.ndarray]:
    """回傳 (Design, 用到的列)。

    subperiod：期間內第幾季。有給時加上「季內」欄位（全東京共用，每期 3 個）：區×期間抓長期走勢，
    季內抓一年之內的變化（小區的季資料太少，不分區估）。

    沒給 template：依資料決定欄位與基準。不放的列（keep = False）是：取引或成約不到 MIN_N 筆的區的成約資料
    （例如奥多摩町只有 2 筆成約、沒有取引：成約效果和町名效果分不開）。
    給 template：沿用它的欄位與基準（驗證用），欄位不在 template 裡的列不放。
    """
    city = sample.city.astype(str)
    seiyaku = sample.price_classification.astype(str) == SEIYAKU
    period = np.asarray(period, dtype=int)
    cities, city_inv = np.unique(city, return_inverse=True)

    keys = np.char.add(np.char.add(city, " "), sample.district.astype(str))
    ukeys, key_inv, key_counts = np.unique(keys, return_inverse=True, return_counts=True)
    if template is None:
        big = {k for k, c in zip(ukeys.tolist(), key_counts.tolist(), strict=True) if c >= MIN_DISTRICT_N}
        n01 = np.bincount(city_inv[~seiyaku], minlength=len(cities))
        n02 = np.bincount(city_inv[seiyaku], minlength=len(cities))
        seiyaku_cities = {c for c, a, b in zip(cities.tolist(), n01, n02, strict=True) if a >= MIN_N and b >= MIN_N}
        usable = ~seiyaku | np.isin(cities, list(seiyaku_cities))[city_inv]
        first = np.full(len(cities), np.iinfo(np.int64).max)
        np.minimum.at(first, city_inv[usable], period[usable])
        base_period = {c: int(f) for c, f in zip(cities.tolist(), first.tolist(), strict=True) if f != np.iinfo(np.int64).max}
        sp_periods = period[seiyaku & usable]
        seiyaku_base = int(sp_periods.min()) if len(sp_periods) else None
        area_base = int(period[usable].min()) if spec.area_by_period and usable.any() else None
    else:
        big, base_period = template.big_districts, template.base_period
        seiyaku_cities, seiyaku_base = template.seiyaku_cities, template.seiyaku_base_period
        area_base = template.area_base_period
        usable = ~seiyaku | np.isin(cities, list(seiyaku_cities))[city_inv]
    def district_codes(pooled: set[str]) -> tuple[list[str], np.ndarray]:
        """町名 → 整數編號（小町名併進該區的「其他」）；後面都用編號，不對 40 萬個字串排序。"""
        merged = []
        for k in ukeys.tolist():
            c = k.split(" ", 1)[0]
            merged.append(f"{c} {ALL_DISTRICTS}" if c in pooled else k if k in big else f"{c} {OTHER_DISTRICT}")
        levels = sorted(set(merged))
        level_of = {nm: i for i, nm in enumerate(levels)}
        return levels, np.array([level_of[nm] for nm in merged])[key_inv]

    pooled = set() if template is None else template.pooled_cities
    district_levels, district = district_codes(pooled)

    # 時間：區 × 期間（各區的基準期間不放）
    pmin = int(period.min())
    span = int(period.max()) - pmin + 1
    cp_codes = city_inv * span + (period - pmin)
    ucp, cp_inv = np.unique(cp_codes, return_inverse=True)
    if template is None:
        pooled = _disconnected_cities(district[usable], cp_inv[usable], city_inv[usable], len(district_levels), len(ucp))
        pooled = {cities[i] for i in pooled}
        if pooled:
            district_levels, district = district_codes(pooled)
    cp_names = []
    for code in ucp.tolist():
        c, p = cities[code // span], code % span + pmin
        cp_names.append(None if base_period.get(c) == p else f"時間:{c}:{p}")
    # 成約 × 期間（全東京共用，第一個期間為基準）
    up, p_inv = np.unique(period, return_inverse=True)
    sp_names = [None if p == seiyaku_base else f"成約×時間:{p}" for p in up.tolist()]

    plans = sample.floor_plan.astype(str)
    uplans, plan_inv = np.unique(plans, return_inverse=True)
    plan_groups = [floor_plan_group(u) for u in uplans.tolist()]
    age = np.maximum(0, sample.year.astype(int) - sample.building_year.astype(int)).astype(float)

    # 建物特性的係數全期間共用。試過每 5 年一組係數：各期的面積斜率和該期的時間係數幾乎共線，
    # 時期交界處會跳（2006→2025 比「相鄰兩年分別估再串起來」的鏈結指數多 +12%）；全期共用的版本和鏈結指數只差 2%
    blocks = [_Block([f"町名:{d}" for d in district_levels], district), _Block(cp_names, cp_inv)]
    blocks += [_continuous(nm, v) for nm, v in building_terms(sample.area.astype(float), age, spec).items()]
    blocks.append(_Block([None, "舊耐震"], (sample.building_year.astype(int) <= spec.old_seismic_last_year).astype(int)))
    blocks.append(_Block([None if g == "2LDK" else f"房型:{g}" for g, _ in plan_groups], plan_inv))
    blocks.append(_Block([None, "房型:+S"], np.array([plus for _, plus in plan_groups], dtype=int)[plan_inv]))
    blocks.append(_categorical("構造", sample.structure.astype(str), structure_group))
    blocks.append(_categorical("改裝", sample.renovation.astype(str), renovation_group))
    blocks.append(_categorical("用途地域", sample.city_planning.astype(str), city_planning_group))
    blocks.append(_categorical("容積率", sample.floor_area_ratio.astype(float), far_group))
    blocks.append(_Block([f"成約:{c}" for c in cities.tolist()], np.where(seiyaku, city_inv, -1)))
    blocks.append(_Block(sp_names, np.where(seiyaku, p_inv, -1)))
    if spec.area_by_period:
        area_names = [None if p == area_base else f"面積×時間:{p}" for p in up.tolist()]
        blocks.append(_Block(area_names, p_inv, np.log(sample.area.astype(float) / spec.standard_area)))
    sub_base: dict[int, int] = {}
    if subperiod is not None:
        subperiod = np.asarray(subperiod, dtype=int)
        if template is None:
            # 基準季 = 該期有資料的第一季（最早一期可能缺前面幾季；拿沒資料的季當基準會和區×期間共線）
            first = np.full(span, 4)
            np.minimum.at(first, period[usable] - pmin, subperiod[usable])
            sub_base = {p + pmin: int(k) for p, k in enumerate(first.tolist()) if k < 4}
        else:
            sub_base = template.subperiod_base
        usub, sub_inv = np.unique((period - pmin) * 4 + subperiod, return_inverse=True)
        sub_names = []
        for code in usub.tolist():
            p, k = code // 4 + pmin, code % 4
            sub_names.append(None if sub_base.get(p) == k else f"季內:{p}:{k}")
        blocks.append(_Block(sub_names, sub_inv))

    if template is None:
        names: list[str] = []
        seen: set[str] = set()
        absorbed: set[str] = set()
        for j, b in enumerate(blocks):
            counts = np.bincount(b.codes[usable & (b.codes >= 0)], minlength=len(b.level_names))
            used = np.nonzero(counts)[0]
            # 每列都有值的分類變數（町名除外，它本來就沒有基準）卻沒有任何一列是基準類別：
            # 整組加起來 = 常數，和町名共線 → 最常見的那一類改當基準（例如某區的改裝欄全部空白）
            covers_all = j > 0 and b.values is None and (b.codes[usable] >= 0).all()
            if covers_all and all(b.level_names[i] is not None for i in used.tolist()) and len(used):
                absorbed.add(b.level_names[int(np.argmax(counts))])
            for nm in (b.level_names[i] for i in used.tolist()):
                if nm is not None and nm not in seen and nm not in absorbed:
                    seen.add(nm)
                    names.append(nm)
        index = {nm: i for i, nm in enumerate(names)}
    else:
        names, index, absorbed = template.names, template.index, template.absorbed

    keep = usable.copy()
    if template is not None:
        # 基準期間以前的期間：無法預測
        base = np.array([base_period.get(c, 10**9) for c in cities.tolist()])[city_inv]
        keep &= period >= base
    # 每個區塊每列最多一個值 → 先排成「列 × 區塊」的欄位編號與值，攤平就是 CSR（不用排序，40 萬列也很快）
    # 水準 → 欄位：−2 = 基準（不放）、−1 = template 裡沒有（這列不能用）；codes == −1（這列沒有這組欄位）→ 最後一個 −2
    level_cols = [
        np.array([-2 if nm is None or nm in absorbed else index.get(nm, -1) for nm in b.level_names] + [-2], dtype=np.int32)
        for b in blocks
    ]
    for b, lc in zip(blocks, level_cols, strict=True):
        keep &= lc[b.codes] != -1
    cols = np.stack([lc[b.codes[keep]] for b, lc in zip(blocks, level_cols, strict=True)], axis=1)
    vals = np.stack([np.ones(int(keep.sum())) if b.values is None else b.values[keep] for b in blocks], axis=1)
    present = cols >= 0
    indptr = np.concatenate([[0], np.cumsum(present.sum(axis=1))]).astype(np.int32)
    X = sp.csr_matrix((vals[present], cols[present], indptr), shape=(int(keep.sum()), len(names)), copy=False)
    # cluster 編號：有用到的町名重新編成 0, 1, 2…
    used = np.zeros(len(district_levels), dtype=bool)
    used[district[keep]] = True
    clusters = (np.cumsum(used) - 1)[district[keep]]
    design = Design(
        X, names, index, clusters, city[keep], period[keep], seiyaku[keep],
        base_period, seiyaku_base, seiyaku_cities, big, absorbed, pooled,
        None if subperiod is None else subperiod[keep], sub_base, area_base,
    )  # fmt: skip
    return design, keep


# ---- 估計 ----

_COLLINEAR_TOL = 1e-9


@dataclass
class Fit:
    beta: np.ndarray
    cov: np.ndarray  # cluster-robust
    resid: np.ndarray
    sigma: float  # 殘差標準差（1.4826 × MAD）
    r2: float
    n: int
    clusters: int


def fit_ols(X: sp.csr_matrix, y: np.ndarray, clusters: np.ndarray, names: list[str] | None = None) -> Fit:
    n, p = X.shape
    xtx = (X.T @ X).toarray()

    def singular(cols: list[int]) -> ValueError:
        label = "、".join(names[i] if names else str(i) for i in cols[:5])
        return ValueError(f"設計矩陣不是滿秩（有欄位共線或沒有資料）：{label}")

    try:
        factor = scipy.linalg.cho_factor(xtx)
    except np.linalg.LinAlgError as e:
        raise singular([_first_failing_column(xtx)]) from e
    # Cholesky 的對角線² ÷ 原本的對角線 = 這個欄位不能被前面欄位解釋的比例；幾乎是 0 就是共線
    # （數值上不一定會分解失敗，但會算出 ±10 這種互相抵銷的荒謬係數）
    explained = np.diag(factor[0]) ** 2 / np.diag(xtx)
    bad = np.nonzero(explained < _COLLINEAR_TOL)[0]
    if len(bad):
        raise singular(bad.tolist())
    beta = scipy.linalg.cho_solve(factor, X.T @ y)
    resid = y - X @ beta
    bread = scipy.linalg.cho_solve(factor, np.eye(p))
    g = int(clusters.max()) + 1
    member = sp.csr_matrix((np.ones(n), (clusters, np.arange(n))), shape=(g, n))
    scores = member @ X.multiply(resid[:, None]).tocsr()  # 各 cluster 的 Σ x·e
    meat = (scores.T @ scores).toarray()
    correction = g / (g - 1) * (n - 1) / (n - p) if g > 1 and n > p else 1.0
    cov = bread @ meat @ bread * correction
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / sst if sst > 0 else 1.0
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    return Fit(beta, cov, resid, sigma, r2, n, g)


def _first_failing_column(xtx: np.ndarray) -> int:
    """Cholesky 失敗時，二分搜尋第一個讓左上角子矩陣不再正定的欄位（只在出錯時跑）。"""
    lo, hi = 1, len(xtx)  # 前 lo 欄可以分解、前 hi 欄不行
    if not np.all(np.diag(xtx) > 0):
        return int(np.argmin(np.diag(xtx)))  # 整欄都是 0：沒有資料
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            scipy.linalg.cho_factor(xtx[:mid, :mid])
            lo = mid
        except np.linalg.LinAlgError:
            hi = mid
    return hi - 1


@dataclass
class Model:
    spec: Spec
    design: Design
    fit: Fit
    sample: Sample  # 去掉離群值後實際用到的樣本
    outliers: int
    inlier_range: tuple[float, float] = (-math.inf, math.inf)  # 第一次估計的殘差在這個範圍內才不算離群值

    def coef(self, name: str) -> tuple[float, float]:
        """(係數, 標準誤)；基準類別 → (0, 0)。"""
        i = self.design.index.get(name)
        return (0.0, 0.0) if i is None else (float(self.fit.beta[i]), math.sqrt(self.fit.cov[i, i]))


def fit_hedonic(
    sample: Sample, period: np.ndarray, spec: Spec = Spec(), subperiod: np.ndarray | None = None
) -> Model:
    design, keep = design_matrix(sample, period, spec, subperiod=subperiod)
    sample, period = sample.subset(keep), period[keep]
    subperiod = None if subperiod is None else subperiod[keep]
    fit = fit_ols(design.X, sample.log_price, design.clusters, design.names)
    center = float(np.median(fit.resid))
    inlier_range = (center - OUTLIER_SIGMAS * fit.sigma, center + OUTLIER_SIGMAS * fit.sigma)
    inlier = (fit.resid >= inlier_range[0]) & (fit.resid <= inlier_range[1])
    outliers = int((~inlier).sum())
    if outliers:
        sample, period = sample.subset(inlier), period[inlier]
        subperiod = None if subperiod is None else subperiod[inlier]
        design, keep = design_matrix(sample, period, spec, subperiod=subperiod)
        sample = sample.subset(keep)
        fit = fit_ols(design.X, sample.log_price, design.clusters, design.names)
    return Model(spec, design, fit, sample, outliers, inlier_range)


def predict(
    model: Model, sample: Sample, period: np.ndarray, subperiod: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (預測的 ln 單價, 能預測的列)。"""
    design, keep = design_matrix(sample, period, model.spec, template=model.design, subperiod=subperiod)
    return design.X @ model.fit.beta, keep


# 標準化時保留的欄位：地點、時間（其餘是建物特性和成約，換成典型物件的）
_KEPT_PREFIXES = LOCATION_PREFIXES + ("時間:", "季內:")


def standardize(
    model: Model, sample: Sample, period: np.ndarray, subperiod: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """每筆成交換算成典型物件、取引水準的 ln 單價：實際 − 建物特性與成約的效果 ＋ 典型物件的效果。

    地點（町名、用途地域、容積率）、時間和殘差都保留，所以同一站的成交還是有高有低。
    回傳 (標準化的 ln 單價, 能算的列)；不能算的列是 nan：模型沒見過的類別或期間，以及離群值（和估計時同一個範圍）。
    """
    design, keep = design_matrix(sample, period, model.spec, template=model.design, subperiod=subperiod)
    beta = model.fit.beta
    y = sample.log_price[keep]
    resid = y - design.X @ beta
    inlier = (resid >= model.inlier_range[0]) & (resid <= model.inlier_range[1])
    removed = np.array([not nm.startswith(_KEPT_PREFIXES) for nm in design.names])
    std = y - design.X[:, removed] @ beta[removed] + float(standard_vector(model) @ beta)
    rows = np.nonzero(keep)[0][inlier]
    out = np.full(len(sample), np.nan)
    out[rows] = std[inlier]
    ok = np.zeros(len(sample), dtype=bool)
    ok[rows] = True
    return out, ok


# ---- 指數 ----


@dataclass
class HedonicSeries:
    """某區各期間的品質調整水準（ln 日圓／㎡，典型物件、取引水準）。"""

    city: str
    levels: dict[int, float]
    n: dict[int, int]
    level_se: dict[int, float]
    time_cov: dict[tuple[int, int], float]  # 期間係數之間的共變異（基準期間 = 0）

    def median(self, period: int) -> float:
        return math.exp(self.levels[period])


def standard_vector(model: Model) -> np.ndarray:
    """典型物件的特性在係數向量上的權重（類別都是基準 → 0）。"""
    spec, index = model.spec, model.design.index
    s = np.zeros(len(index))
    terms = building_terms(np.array([spec.standard_area]), np.array([spec.standard_age]), spec)
    for name, v in terms.items():
        if name in index:
            s[index[name]] = v[0]
    # 典型物件的建築年 = 成交年 − 17：在任何一年都不是舊耐震（指數期間 2006 年以後）
    return s


def location_vectors(model: Model) -> dict[str, np.ndarray]:
    """各區的地點組成：町名、用途地域、容積率欄位，依該區成交筆數平均。"""
    d = model.design
    loc = np.array([any(nm.startswith(p) for p in LOCATION_PREFIXES) for nm in d.names])
    cities, inv = np.unique(d.city, return_inverse=True)
    counts = np.bincount(inv)
    member = sp.csr_matrix((1 / counts[inv], (inv, np.arange(len(inv)))), shape=(len(cities), len(inv)))
    mix = (member @ d.X).toarray()
    mix[:, ~loc] = 0
    return {c: mix[i] for i, c in enumerate(cities)}


def _quad(a: np.ndarray, b: np.ndarray, cov: np.ndarray) -> float:
    """a' V b，只取兩個向量有值的欄位（向量大多是 0）。"""
    ia, ib = np.nonzero(a)[0], np.nonzero(b)[0]
    return float(a[ia] @ cov[np.ix_(ia, ib)] @ b[ib])


def index_series(model: Model) -> dict[str, HedonicSeries]:
    """各區各期間的水準 L = u_p + 地點組成，u_p = 期間係數 + 典型物件。
    漲幅的變異 = Var(u_b − u_a)：地點組成和典型物件兩期一樣，相減後只剩期間係數。"""
    d, fit = model.design, model.fit
    standard = standard_vector(model)
    locs = location_vectors(model)
    counts = Counter(zip(d.city.tolist(), d.period.tolist(), strict=True))
    out = {}
    for city, loc in locs.items():
        periods = sorted(p for c, p in counts if c == city)
        u = {}
        for p in periods:
            u[p] = standard.copy()
            if (i := d.index.get(f"時間:{city}:{p}")) is not None:  # 基準期間沒有這欄（= 0）
                u[p][i] += 1
        levels = {p: float((u[p] + loc) @ fit.beta) for p in periods}
        se = {p: math.sqrt(max(0.0, _quad(u[p] + loc, u[p] + loc, fit.cov))) for p in periods}
        cov = {(p1, p2): _quad(u[p1], u[p2], fit.cov) for p1 in periods for p2 in periods}
        out[city] = HedonicSeries(city, levels, {p: counts[(city, p)] for p in periods}, se, cov)
    return out


def hedonic_growth(series: HedonicSeries, start: int, end: int) -> Growth:
    if end <= start:
        raise ValueError(f"迄年必須晚於起年：{start} → {end}")
    if any(series.n.get(p, 0) < MIN_N for p in (start, end)):
        return Growth(start, end, None, None, None, None, "insufficient")
    c = series.time_cov
    var = c[(start, start)] + c[(end, end)] - 2 * c[(start, end)]
    return growth_from_log(start, end, series.levels[end] - series.levels[start], math.sqrt(max(0.0, var)))


def hedonic_trend(series: HedonicSeries, last_year: int) -> LongTrend:
    first = last_year - INVEST_WINDOW_YEARS + 1
    points = [(p, series.levels[p]) for p in range(first, last_year + 1) if series.n.get(p, 0) >= MIN_N]
    return trend_from_points(first, last_year, points)


def hedonic_price(series: HedonicSeries | None, period: int) -> Price:
    n = 0 if series is None else series.n.get(period, 0)
    if n < MIN_N:
        return Price(None, None, None, n, "insufficient")
    return price_from_log(series.levels[period], series.level_se[period], n)


# ---- 時點修正 ----


@dataclass(frozen=True)
class Adjustment:
    """某區某季的成交，換算成目標期（最新 4 季）平均水準要乘的係數。basis = None：無法修正。"""

    city: str
    year: int
    quarter: int
    log_factor: float | None
    se: float | None
    basis: Literal["city", "tokyo"] | None  # city = 用該區自己的指數；tokyo = 樣本不足，用全東京的
    n: int  # 該區在那一期（4 季）的筆數
    # 係數是典型物件（55㎡）的；面積 A 的物件再乘 (A / 55)^area_elasticity（面積的效果隨期間變動，全東京共用）
    area_elasticity: float = 0.0

    @property
    def factor(self) -> float | None:
        return None if self.log_factor is None else math.exp(self.log_factor)

    @property
    def low(self) -> float | None:
        return None if self.log_factor is None else math.exp(self.log_factor - Z95 * self.se)

    @property
    def high(self) -> float | None:
        return None if self.log_factor is None else math.exp(self.log_factor + Z95 * self.se)

    def log_factor_for(self, area: float, standard_area: float = Spec.standard_area) -> float | None:
        if self.log_factor is None:
            return None
        return self.log_factor + self.area_elasticity * math.log(area / standard_area)


def time_adjustments(
    model: Model,
    cities: Sequence[str],
    quarters: Sequence[tuple[int, int]],
    last_year: int,
    last_quarter: int,
) -> list[Adjustment]:
    """model 要用 window_period ＋ window_subperiod 估（同一個 last_year、last_quarter）。

    ln(係數) = [區×期間₀ ＋ 期間₀ 內各季的平均] − [區×期間_w ＋ 期間_w 內第 k 季]
    町名、典型物件、成約等其他項目兩邊一樣，相減抵銷；誤差用對比向量 × cluster-robust 共變異。
    該區在那一期或目標期不到 MIN_N 筆（或模型裡沒有這區）→ 用全東京的係數：各區的係數依目標期筆數加權平均。
    這時不知道這區自己的走勢，區間另外加上「各區係數圍繞全東京平均的離散」（當成隨機抽一區），
    不然只反映全東京平均本身的誤差，會窄得不合理。
    係數是典型物件（55㎡）的；有面積 × 期間項時另外給 area_elasticity = 目標期 − 那一期的面積斜率
    （區間只算 55㎡ 的部分）。
    """
    d, beta, cov = model.design, model.fit.beta, model.fit.cov
    if d.subperiod is None:
        raise ValueError("時點修正要用加了 subperiod（期間內第幾季）的模型")
    size = len(d.names)
    counts = Counter(zip(d.city.tolist(), d.period.tolist(), strict=True))
    present = {(int(w), int(k)) for w, k in zip(d.period.tolist(), d.subperiod.tolist(), strict=True)}

    def unit(name: str) -> np.ndarray:
        v = np.zeros(size)
        v[d.index[name]] = 1.0
        return v

    def quarter_vec(w: int, k: int) -> np.ndarray | None:
        if (w, k) not in present:
            return None
        return np.zeros(size) if d.subperiod_base.get(w) == k else unit(f"季內:{w}:{k}")

    def city_vec(c: str, w: int) -> np.ndarray | None:
        if f"時間:{c}:{w}" in d.index:
            return unit(f"時間:{c}:{w}")
        return np.zeros(size) if d.base_period.get(c) == w else None

    def area_slope(w: int) -> float:
        """面積 × 期間的係數（基準期間、或模型沒有這一項 → 0）。"""
        i = d.index.get(f"面積×時間:{w}")
        return 0.0 if i is None else float(beta[i])

    target_q = [v for k in range(4) if (v := quarter_vec(0, k)) is not None]
    target_q_mean = np.mean(target_q, axis=0)
    model_cities = sorted(set(d.city.tolist()))

    def eligible(c: str, w: int) -> bool:
        return counts[(c, w)] >= MIN_N and counts[(c, 0)] >= MIN_N and city_vec(c, w) is not None and city_vec(c, 0) is not None

    tokyo_cache: dict[int, tuple[np.ndarray, float] | None] = {}

    def tokyo_city_part(w: int) -> tuple[np.ndarray, float] | None:
        """(Σ a_c (區×期間₀ − 區×期間_w), 各區圍繞它的加權變異)，a_c ∝ 目標期筆數。"""
        if w not in tokyo_cache:
            cs = [c for c in model_cities if eligible(c, w)]
            if not cs:
                tokyo_cache[w] = None
            else:
                weights = np.array([counts[(c, 0)] for c in cs], dtype=float)
                weights /= weights.sum()
                parts = [city_vec(c, 0) - city_vec(c, w) for c in cs]
                mean = sum(a * v for a, v in zip(weights, parts, strict=True))
                values = np.array([v @ beta for v in parts])
                spread = float(weights @ (values - mean @ beta) ** 2)
                tokyo_cache[w] = (mean, spread)
        return tokyo_cache[w]

    last_qi = last_year * 4 + last_quarter - 1
    out = []
    for c in cities:
        for y, q in quarters:
            r = y * 4 + q - 1 - last_qi + 3
            w, k = r // 4, r % 4
            n = counts[(c, w)]
            qv = quarter_vec(w, k)
            if qv is None:
                out.append(Adjustment(c, y, q, None, None, None, n))
                continue
            extra = 0.0
            if eligible(c, w):
                diff, basis = city_vec(c, 0) - city_vec(c, w), "city"
            elif (part := tokyo_city_part(w)) is not None:
                (diff, extra), basis = part, "tokyo"
            else:
                out.append(Adjustment(c, y, q, None, None, None, n))
                continue
            diff = diff + target_q_mean - qv
            se = math.sqrt(max(0.0, _quad(diff, diff, cov)) + extra)
            out.append(Adjustment(c, y, q, float(diff @ beta), se, basis, n, area_slope(0) - area_slope(w)))
    return out


# ---- 報告用 ----


def building_effect(model: Model, area: float, age: float) -> float:
    """面積、屋齡的效果（ln）；和典型物件比：building_effect(m, a, g) − building_effect(m, 55, 17)。"""
    terms = building_terms(np.array([float(area)]), np.array([float(age)]), model.spec)
    return sum(model.coef(name)[0] * v[0] for name, v in terms.items())


@dataclass(frozen=True)
class Validation:
    """隨機留一部分不參加估計，拿來預測；和「同町名、同期間的中位數」比。誤差都是 |ln 預測 − ln 實際|。"""

    n_train: int
    n_test: int
    unpredictable: int  # 驗證資料裡模型沒見過的町名、類別、期間
    model_median_error: float  # exp(誤差中位數) − 1
    model_within: dict[float, float]  # 誤差在 ±x 以內的比例
    baseline_median_error: float
    baseline_within: dict[float, float]


def validate_holdout(sample: Sample, period: np.ndarray, frac: float = 0.1, seed: int = 0, spec: Spec = Spec()) -> Validation:
    test = np.random.default_rng(seed).random(len(sample)) < frac
    train, held = sample.subset(~test), sample.subset(test)
    model = fit_hedonic(train, period[~test], spec)
    pred, ok = predict(model, held, period[test])
    actual = held.log_price[ok]
    # 基準：訓練資料裡同町名、同期間的中位數（沒有就用同區、同期間）
    tp, hp = period[~test], period[test][ok]
    med: dict[tuple, float] = {}
    for keys in (lambda s, p: zip(s.city.tolist(), s.district.tolist(), p.tolist(), strict=True),
                 lambda s, p: zip(s.city.tolist(), p.tolist(), strict=True)):  # fmt: skip
        groups: dict[tuple, list[float]] = {}
        for k, v in zip(keys(train, tp), train.log_price.tolist(), strict=True):
            groups.setdefault(k, []).append(v)
        med |= {k: float(np.median(v)) for k, v in groups.items()}
    h = held.subset(ok)
    base = np.array([
        med.get((c, d, p), med.get((c, p), np.nan))
        for c, d, p in zip(h.city.tolist(), h.district.tolist(), hp.tolist(), strict=True)
    ])  # fmt: skip

    def summary(err: np.ndarray) -> tuple[float, dict[float, float]]:
        err = np.abs(err[~np.isnan(err)])
        return math.exp(float(np.median(err))) - 1, {x: float(np.mean(err < math.log(1 + x))) for x in (0.1, 0.2)}

    m_med, m_within = summary(pred - actual)
    b_med, b_within = summary(base - actual)
    return Validation(int((~test).sum()), int(ok.sum()), int((~ok).sum()), m_med, m_within, b_med, b_within)
