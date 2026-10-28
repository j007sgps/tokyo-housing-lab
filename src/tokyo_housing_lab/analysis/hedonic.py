"""品質調整指數（hedonic 迴歸）：同樣條件的物件，價格怎麼變。

中位數會受「那年賣了什麼物件」影響（新舊、大小、地段）。這裡用 OLS 把這些因素拆開：

    ln(㎡ 單價) = 町名 + 市區町村×期間 + 面積 + 屋齡 + 舊耐震 + 房型 + 構造 + 改裝 + 用途地域 + 容積率
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
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from tokyo_housing_lab.analysis.trend import (
    INVEST_WINDOW_YEARS,
    MIN_N,
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


def design_matrix(
    sample: Sample, period: np.ndarray, spec: Spec, template: Design | None = None
) -> tuple[Design, np.ndarray]:
    """回傳 (Design, 用到的列)。

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
    else:
        big, base_period = template.big_districts, template.base_period
        seiyaku_cities, seiyaku_base = template.seiyaku_cities, template.seiyaku_base_period
        usable = ~seiyaku | np.isin(cities, list(seiyaku_cities))[city_inv]
    # 町名 → 整數編號（小町名併進該區的「其他」）；後面都用編號，不對 40 萬個字串排序
    merged = [k if k in big else f"{k.split(' ', 1)[0]} {OTHER_DISTRICT}" for k in ukeys.tolist()]
    district_levels = sorted(set(merged))
    level_of = {nm: i for i, nm in enumerate(district_levels)}
    district = np.array([level_of[nm] for nm in merged])[key_inv]

    # 時間：區 × 期間（各區的基準期間不放）
    pmin = int(period.min())
    span = int(period.max()) - pmin + 1
    cp_codes = city_inv * span + (period - pmin)
    ucp, cp_inv = np.unique(cp_codes, return_inverse=True)
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
        base_period, seiyaku_base, seiyaku_cities, big, absorbed,
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

    def coef(self, name: str) -> tuple[float, float]:
        """(係數, 標準誤)；基準類別 → (0, 0)。"""
        i = self.design.index.get(name)
        return (0.0, 0.0) if i is None else (float(self.fit.beta[i]), math.sqrt(self.fit.cov[i, i]))


def fit_hedonic(sample: Sample, period: np.ndarray, spec: Spec = Spec()) -> Model:
    design, keep = design_matrix(sample, period, spec)
    sample, period = sample.subset(keep), period[keep]
    fit = fit_ols(design.X, sample.log_price, design.clusters, design.names)
    inlier = np.abs(fit.resid - np.median(fit.resid)) <= OUTLIER_SIGMAS * fit.sigma
    outliers = int((~inlier).sum())
    if outliers:
        sample, period = sample.subset(inlier), period[inlier]
        design, keep = design_matrix(sample, period, spec)
        sample = sample.subset(keep)
        fit = fit_ols(design.X, sample.log_price, design.clusters, design.names)
    return Model(spec, design, fit, sample, outliers)


def predict(model: Model, sample: Sample, period: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (預測的 ln 單價, 能預測的列)。"""
    design, keep = design_matrix(sample, period, model.spec, template=model.design)
    return design.X @ model.fit.beta, keep


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
