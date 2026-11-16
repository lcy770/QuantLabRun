"""按时间边界加载面板，限制信号缓存大小，并用逐日统计量计算岭回归组合 IC。"""
# 真实行情评价与组合试算，负责时间切分、单因子指标、方向锁定和 Ridge 组合。
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..data import PanelData
from .._vendor.alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from .io import digest, save_json, cache_write_lock


def normalize(values, mask):
    # 按日做截面标准化，返回标准化信号、有效值掩码和具有截面变化的日期标记。
    """按日对有效截面标准化，返回标准化信号、有效值掩码及有截面变化的日期标记。

    输入按日期与股票排列，先缩放再中心化以降低数值溢出，样本不足或近常量截面置零。
    """
    observed = mask & np.isfinite(values)
    count = observed.sum(1)
    safe = np.where(observed, values, 0.0)
    scale = np.max(abs(safe), axis=1)
    scaled = safe / np.where(scale > 0, scale, 1)[:, None]
    centered = np.where(observed, scaled - scaled.sum(1)[:, None] / np.maximum(count, 1)[:, None], 0.0)
    variance = np.sum(centered ** 2, axis=1) / np.maximum(count, 1)
    variable = (count >= 3) & (variance > 1e-24)
    z = centered / np.sqrt(np.where(variable, variance, 1))[:, None]
    z[~variable] = 0
    return z, observed, variable


@dataclass
class Slice:
    panel: PanelData
    horizon: int
    target_values: np.ndarray | None = None

    def __post_init__(self):
        """建立固定评分网格的收益目标、有效掩码、截面权重及标签成熟日期。

        校验外部目标的形状，并要求至少存在一个有效且有截面变化的评分日。
        """
        self.target = (self.panel.target(self.horizon).numpy().astype(np.float64) if self.target_values is None
                       else np.array(self.target_values, dtype=np.float64, copy=True))
        if self.target.shape != (self.panel.n_days, self.panel.n_stocks):
            raise ValueError("target sidecar does not match scoring grid")
        self.mask = np.isfinite(self.target)
        self.y, _, self.active = normalize(self.target, self.mask)
        self.count = np.maximum(self.mask.sum(1), 1)
        self.weights = self.mask / self.count[:, None]
        if not self.active.any():
            raise ValueError("no valid cross-sectional target days")
        self.fit_weights = self.weights * self.active[:, None] / self.active.sum()
        back = self.panel.max_backtrack_days
        self.dates = self.panel.dates[back:back + self.panel.n_days].values.astype("datetime64[D]")
        self.ends = self.panel.dates[back + self.horizon:back + self.horizon + self.panel.n_days].values.astype("datetime64[D]")


def load_slice(spec, horizon, lookback, future):
    """从缓存面板截取指定日期范围，保留回看和未来缓冲，返回评分 Slice。

    存在外部收益标签时校验其预测期限，评分区间至少包含三个日期。
    """
    cache = Path(spec["cache"])
    panel = PanelData.from_memmap(cache, max_backtrack_days=lookback, max_future_days=future)
    dates = panel.dates[lookback:lookback + panel.n_days]
    selected = np.flatnonzero((dates >= spec.get("start", str(dates[0].date()))) &
                              (dates <= spec.get("end", str(dates[-1].date()))))
    if len(selected) < 3:
        raise ValueError("each scoring slice needs at least three dates")
    first, last = int(selected[0]), int(selected[-1])
    sub = PanelData({name: values[first:last + lookback + future + 1] for name, values in panel._arrays.items()},
                    panel.dates[first:last + lookback + future + 1], panel.stock_ids, lookback, future)
    metadata = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    target = None
    if metadata.get("label_file"):
        if metadata["label_horizon"] != horizon:
            raise ValueError("target sidecar horizon differs from requested horizon")
        target = np.load(cache / metadata["label_file"], mmap_mode="r")[first + lookback:last + lookback + 1]
    return Slice(sub, horizon, target)


def load_task(task, horizon, lookback, future, with_future=False):
    # 拟合、搜索和可选后续评分区间分别加载，并核验收益标签实际成熟时间。
    """加载任务的拟合、搜索及可选后续评分区间，返回按区间名称索引的字典。

    检查股票身份与顺序一致，并按标签实际成熟日期排除区间交叠。
    """
    fit = load_slice(task["fit"], horizon, lookback, future)
    search = load_slice(task["search"], horizon, lookback, future)
    if fit.panel.stock_ids != search.panel.stock_ids:
        raise ValueError("fit/search stock identity and ordering differ")
    if fit.ends.max() >= search.dates.min():
        raise ValueError("fit labels overlap search scoring dates; purge immature labels")
    result = {"fit": fit, "search": search}
    if with_future:
        final = load_slice(task["future"], horizon, lookback, future)
        if search.ends.max() >= final.dates.min() or final.panel.stock_ids != fit.panel.stock_ids:
            raise ValueError("search/future labels overlap or stock identities differ")
        result["future"] = final
    return result


@dataclass
class Signal:
    fit: np.ndarray
    search: np.ndarray
    metrics: dict


class Evaluator:
    # 原方案下层奖励读取 fit，候选发现与组合比较另外使用 search。
    # 后续实验的成熟历史终点评分由训练模块显式调用 segment 完成。
    def __init__(self, grammar, fit, search, *, min_coverage=0.8, ridge=1e-3,
                 cache_size=64, score_weights=(0.5, 0.3, 0.2), signal_store=None, persistent_ids=None):
        """绑定文法和拟合、搜索区间，校验覆盖率、Ridge 与下层评分权重。

        初始化指标、信号、下层奖励缓存和计时计数器，可按配置创建磁盘信号缓存目录。
        """
        self.grammar, self.fit, self.search = grammar, fit, search
        self.min_coverage, self.ridge, self.cache_size = min_coverage, ridge, cache_size
        if not 0 < min_coverage <= 1 or ridge <= 0 or cache_size < 2:
            raise ValueError("invalid evaluator configuration")
        self.score_weights = np.asarray(score_weights, dtype=float)
        if (self.score_weights.shape != (3,) or not np.isfinite(self.score_weights).all() or
                (self.score_weights < 0).any() or self.score_weights.sum() <= 0):
            raise ValueError("invalid lower reward weights")
        self.signals, self.metrics, self.trees = OrderedDict(), {}, {}
        self.public_signals = {}
        self.lower_rewards = OrderedDict()
        self.fit_results = OrderedDict()
        self.calls, self.requests, self.hits, self.seconds = 0, 0, 0, 0.0
        self.combination_evaluations, self.combination_seconds = 0, 0.0
        self.signal_store = Path(signal_store) if signal_store else None
        if self.signal_store:
            self.signal_store.mkdir(parents=True, exist_ok=True)
        self.disk_hits = 0
        self.persistent_ids = None if persistent_ids is None else frozenset(persistent_ids)

    def for_episode(self, initial):
        """共享只读的公共起点信号，生成候选及其结果归本次搜索回合管理。"""
        return Evaluator(self.grammar, self.fit, self.search, min_coverage=self.min_coverage,
                         ridge=self.ridge, cache_size=self.cache_size, score_weights=self.score_weights,
                         signal_store=self.signal_store, persistent_ids=[tree.id for tree in initial])

    def lower_score(self, metrics):
        # 原方案下层的连续单因子终点奖励，默认权重为 0.5、0.3、0.2。
        # 数值可靠时加权绝对原始 IC、绝对 RankIC 和 tanh(绝对 ICIR)。
        """用可靠的单区间指标计算下层连续质量分，不可靠时返回零。

        对绝对原始 IC、绝对 RankIC 和 tanh 绝对 ICIR 加权并归一化，默认权重为 0.5、0.3、0.2。
        """
        if not metrics["reliable"]:
            return 0.0
        return float(np.dot(self.score_weights, [abs(metrics["raw_ic"]), abs(metrics["rank_ic"]),
                         np.tanh(abs(metrics["icir"]))]) / self.score_weights.sum())

    def remember(self, key, result):
        """缓存信号及指标，公共起点保存在独立映射，生成候选使用限容量的回合内缓存。"""
        self.metrics[key] = result.metrics
        if self.persistent_ids is not None and key in self.persistent_ids:
        # 公共起点的只读内存映射句柄在本回合保持可用。
        # 生成候选使用下方的回合内最近最少使用缓存，并限制容量。
            self.public_signals[key] = result
        else:
            self.signals[key] = result
            if len(self.signals) > self.cache_size:
                self.signals.popitem(last=False)

    def segment(self, tree, scoring):
        # 将公式放到指定行情区间求值，评价使用固定股票、日期和收益目标网格。
        """在指定评分区间求值公式，返回标准化信号与 IC、RankIC、ICIR、覆盖率等指标。

        使用固定日期和股票网格，可靠性结合覆盖率、有效截面变化比例及无穷值检查。
        """
        values = self.grammar.compile(tree).evaluate(scoring.panel).detach().cpu().numpy().astype(float)
        if values.shape != scoring.target.shape:
            raise ValueError("formula output shape differs from fixed scoring grid")
        z, observed, variable = normalize(values, scoring.mask)
        coverage = float(observed[scoring.mask & scoring.active[:, None]].mean())
        daily = (z * scoring.y).sum(1) / scoring.count
        ic = float(np.clip(daily[scoring.active], -1, 1).mean())
        # 原单因子 PPO 指标在信号与收益同时有限的样本上计算 IC、RankIC 和 ICIR。
        v = torch.tensor(values, dtype=torch.float32)
        y = torch.tensor(scoring.target, dtype=torch.float32)
        v[~torch.isfinite(v)] = float("nan")
        y[~torch.isfinite(y)] = float("nan")
        pearson = batch_pearsonr(v, y).numpy()
        rank = batch_spearmanr(v, y).numpy()
        raw_ic = float(np.nanmean(pearson)) if np.isfinite(pearson).any() else 0.0
        rank_ic = float(np.nanmean(rank)) if np.isfinite(rank).any() else 0.0
        deviation = float(np.nanstd(pearson)) if np.isfinite(pearson).any() else 0.0
        icir = raw_ic / max(deviation, 1e-6)
        # 可靠性同时考虑覆盖率、有效截面变化日期比例和无穷值。
        reliable = coverage >= self.min_coverage and float(variable[scoring.active].mean()) >= 0.2 and not np.isinf(values).any()
        return z, {"ic": ic, "raw_ic": raw_ic, "rank_ic": rank_ic, "icir": icir,
                   "coverage": coverage, "reliable": bool(reliable), "days": int(scoring.active.sum())}

    def get(self, tree):
        """获取或计算公式的拟合期与搜索期信号及指标，返回 Signal 并更新缓存统计。

        方向由拟合期 IC 确定并锁定，再计算搜索质量、分段质量和等价信号哈希。
        可处理的求值失败记录为不可靠结果，磁盘缓存通过写锁和原子替换提交。
        """
        started = perf_counter()
        self.requests += 1
        key = tree.id
        if key in self.public_signals:
            self.hits += 1
            return self.public_signals[key]
        if key in self.signals:
            self.hits += 1
            self.signals.move_to_end(key)
            return self.signals[key]
        persistent = self.persistent_ids is None or key in self.persistent_ids
        prefix = self.signal_store / key if self.signal_store and persistent else None
        if prefix and prefix.with_suffix(".json").exists():
            payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
            result = Signal(np.load(prefix.with_suffix(".fit.npy"), mmap_mode="r"),
                            np.load(prefix.with_suffix(".search.npy"), mmap_mode="r"), payload)
            self.disk_hits += 1
            self.hits += 1
            self.trees[key] = tree
            self.remember(key, result)
            self.seconds += perf_counter() - started
            return result
        self.calls += 1
        self.trees[key] = tree
        try:
            if key in self.fit_results:
                fit, fm = self.fit_results[key]
            else:
                fit, fm = self.segment(tree, self.fit)
            search, sm = self.segment(tree, self.search)
            # 使用方向只由拟合期确定，随后按同一方向计算搜索期质量。
            direction = 1 if fm["ic"] >= 0 else -1
            lower = self.lower_score(fm)
            payload = {"fit": fm, "search": sm, "direction": direction,
                       "quality": direction * sm["ic"], "lower_reward": lower,
                       "reliable": fm["reliable"] and sm["reliable"]}
            daily = direction * (search * self.search.y).sum(1) / self.search.count
            payload["quality_fold_ics"] = [float(daily[index].mean()) for index in
                np.array_split(np.flatnonzero(self.search.active), min(3, int(self.search.active.sum())))]
            signed_fit, signed_search = direction * fit, direction * search
        # 固定标准化后，用信号哈希识别符号或仿射变换等价的信号。
            nonzero = signed_fit.flat[np.flatnonzero(signed_fit)[:1]]
            alias_sign = -1 if len(nonzero) and nonzero[0] < 0 else 1
            signal_hash = hashlib.sha256((alias_sign * signed_fit).tobytes() + (alias_sign * signed_search).tobytes()).hexdigest()
            payload["signal_hash"] = signal_hash
            result = Signal(signed_fit, signed_search, payload)
        except (ValueError, RuntimeError, FloatingPointError, IndexError, KeyError) as exc:
            payload = {"reliable": False, "quality": 0.0, "lower_reward": 0.0,
                       "error": f"{type(exc).__name__}: {exc}"}
            result = Signal(np.zeros_like(self.fit.y), np.zeros_like(self.search.y), payload)
        if prefix:
            with cache_write_lock(prefix.with_suffix(".lock")):
                if not prefix.with_suffix(".json").exists():
                    for suffix, values in ((".fit.npy", result.fit), (".search.npy", result.search)):
                        destination = prefix.with_suffix(suffix)
                        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
                        with temporary.open("wb") as handle:
                            np.save(handle, values)
                        os.replace(temporary, destination)
                    save_json(prefix.with_suffix(".json"), payload)
            payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
            result = Signal(np.load(prefix.with_suffix(".fit.npy"), mmap_mode="r"),
                            np.load(prefix.with_suffix(".search.npy"), mmap_mode="r"), payload)
        self.remember(key, result)
        self.seconds += perf_counter() - started
        return result

    def lower(self, tree):
        # 原方案下层评分入口只读取拟合期指标，标量缓存用于重复终点评分。
        # 共同奖励实验走训练模块的显式重评分分支，二者由配置区分。
        """计算并缓存原方案的拟合期单因子终点奖励，返回标量质量分。

        优先复用已有拟合指标或奖励缓存，可处理的求值失败返回零。
        """
        started = perf_counter()
        self.requests += 1
        key = tree.id
        if key in self.lower_rewards:
            self.hits += 1
            self.lower_rewards.move_to_end(key)
            if key in self.fit_results:
                self.fit_results.move_to_end(key)
            return self.lower_rewards[key]
        if "fit" in self.metrics.get(key, {}):
            self.hits += 1
            reward = self.lower_score(self.metrics[key]["fit"])
        else:
            self.calls += 1
            try:
                fitted, metrics = self.segment(tree, self.fit)
                self.fit_results[key] = (fitted, metrics)
                if len(self.fit_results) > self.cache_size:
                    self.fit_results.popitem(last=False)
                reward = self.lower_score(metrics)
            except (ValueError, RuntimeError, FloatingPointError, IndexError, KeyError):
                reward = 0.0
        self.lower_rewards[key] = reward
        if len(self.lower_rewards) > self.cache_size:
            self.lower_rewards.popitem(last=False)
        self.seconds += perf_counter() - started
        return reward


@dataclass(frozen=True)
class PoolScore:
    ids: tuple[str, ...]
    weights: tuple[float, ...]
    ic: float


class PoolStats:
    # 缓存拟合期线性方程统计量，以及搜索期逐日协方差，供加入和替换复用。
    """为当前工作组合和候选缓存统计量，以逐日矩精确计算组合 IC。"""
    def __init__(self, evaluator, trees=()):
        """初始化组合的拟合期充分统计量和搜索期逐日矩，并依次加入给定成员。"""
        self.evaluator = evaluator
        self.trees = ()
        self.g = np.zeros((0, 0))
        self.b = np.zeros(0)
        days = len(evaluator.search.dates)
        self.h = np.zeros((days, 0, 0))
        self.c = np.zeros((days, 0))
        self.means = np.zeros((days, 0))
        self.groups, self.signs = (), ()
        self.signals = ()
        for tree in trees:
            self.__dict__.update(self.extend(tree).__dict__)

    def extend(self, tree):
        """增量计算加入一个可靠因子后的组合统计量，返回新 PoolStats。

        复用已有信号并更新交叉矩，原对象保持可复用，结构重复时直接返回原对象。
        """
        if tree.id in [t.id for t in self.trees]:
            return self
        signal = self.evaluator.get(tree)
        if not signal.metrics["reliable"]:
            raise ValueError("unreliable signal cannot enter combination")
        new = object.__new__(PoolStats)
        new.__dict__ = self.__dict__.copy()
        new.trees = self.trees + (tree,)
        new.signals = self.signals + (signal,)
        new.g = np.pad(self.g, ((0, 1), (0, 1)))
        new.b = np.pad(self.b, (0, 1))
        new.h = np.pad(self.h, ((0, 0), (0, 1), (0, 1)))
        new.c = np.pad(self.c, ((0, 0), (0, 1)))
        new.means = np.pad(self.means, ((0, 0), (0, 1)))
        e, j = self.evaluator, len(self.trees)
        fw, sw = e.fit.fit_weights, e.search.weights
        new.b[j] = np.sum(fw * signal.fit * e.fit.y)
        new.means[:, j] = np.sum(sw * signal.search, axis=1)
        new.c[:, j] = np.sum(sw * signal.search * e.search.y, axis=1) - new.means[:, j] * np.sum(sw * e.search.y, axis=1)
        group, sign = tree.id, 1.0
        for i, previous in enumerate(new.signals):
            new.g[i, j] = new.g[j, i] = np.sum(fw * previous.fit * signal.fit)
            covariance = np.sum(sw * previous.search * signal.search, axis=1) - new.means[:, i] * new.means[:, j]
            new.h[:, i, j] = new.h[:, j, i] = covariance
            if i < j and group == tree.id:
                if np.array_equal(signal.fit, previous.fit):
                    group, sign = self.groups[i], self.signs[i]
                elif np.array_equal(signal.fit, -previous.fit):
                    group, sign = self.groups[i], -self.signs[i]
        new.groups = self.groups + (group,)
        new.signs = self.signs + (sign,)
        return new

    def subset(self, indices):
        """按成员索引切出独立的组合统计对象，保留给定顺序并复制对应矩阵。"""
        new = object.__new__(PoolStats)
        new.__dict__ = self.__dict__.copy()
        indices = list(indices)
        new.trees = tuple(self.trees[i] for i in indices)
        new.signals = tuple(self.signals[i] for i in indices)
        new.groups = tuple(self.groups[i] for i in indices)
        new.signs = tuple(self.signs[i] for i in indices)
        new.g = self.g[np.ix_(indices, indices)].copy()
        new.b = self.b[indices].copy()
        new.h = self.h[:, indices][:, :, indices].copy()
        new.c, new.means = self.c[:, indices].copy(), self.means[:, indices].copy()
        return new

    def score(self):
        """计算并返回组合评分，同时累计评价次数及耗时。"""
        started = perf_counter()
        self.evaluator.combination_evaluations += 1
        try:
            return self._score()
        finally:
            self.evaluator.combination_seconds += perf_counter() - started

    def _score(self):
        # 在拟合期解 Ridge 权重，再将锁定权重用于搜索期组合 IC 评价。
        """在拟合期求解 Ridge 权重，再锁定权重计算搜索期有向组合 IC。

        合并拟合信号等价组，数值抵消时用实际组合信号复核，返回成员、权重和 IC。
        """
        ids = tuple(t.id for t in self.trees)
        if not ids:
            return PoolScore((), (), 0.0)
        groups = sorted(set(self.groups))
        transform = np.zeros((len(ids), len(groups)))
        for col, group in enumerate(groups):
            members = [i for i, g in enumerate(self.groups) if g == group]
            for i in members:
                transform[i, col] = self.signs[i] / len(members)
        gram, rhs = transform.T @ self.g @ transform, transform.T @ self.b
        coefficients = np.linalg.solve(gram + self.evaluator.ridge * np.eye(len(groups)), rhs)
        w = transform @ coefficients
        variance = np.einsum("i,dij,j->d", w, self.h, w)
        covariance = self.c @ w
        scale = np.einsum("i,dij,j->d", abs(w), abs(self.h), abs(w))
        unstable = variance <= 1e-12 * np.maximum(scale, 1e-300)
        ic = np.divide(covariance, np.sqrt(np.maximum(variance, 0)), out=np.zeros_like(covariance), where=variance > 0)
        if unstable.any():
            # 出现数值抵消或近常量组合时，回到实际组合信号做稳定的精确评分。
            prediction = sum(weight * signal.search for weight, signal in zip(w, self.signals))
            z, _, _ = normalize(prediction, self.evaluator.search.mask)
            exact = (z * self.evaluator.search.y).sum(1) / self.evaluator.search.count
            ic[unstable] = exact[unstable]
        score = float(np.clip(ic[self.evaluator.search.active], -1, 1).mean())
        if not np.isfinite(w).all() or not np.isfinite(score):
            raise FloatingPointError("invalid combination fit/score")
        return PoolScore(ids, tuple(map(float, w)), score)


class WorkingPool:
    # 维护容量受限的当前组合，候选发现档案由搜索模块另外管理。
    def __init__(self, evaluator, capacity, initial=()):
        """校验组合容量并建立初始组合统计与评分，初始化组合试算计数。"""
        if capacity < 1 or len(initial) > capacity:
            raise ValueError("invalid working pool capacity")
        self.capacity = capacity
        self.stats = PoolStats(evaluator, initial)
        self.score = self.stats.score()
        self.evaluations, self.seconds = 1, 0.0

    def consider(self, tree):
        # 组合未满则试加入，已满则枚举替换一个成员，并与原组合比较。
        """试算加入候选或逐个替换现有成员，只保留超过数值容差的搜索期组合改善。

        更新工作组合及评价统计，返回相对调用前的组合 IC 增量，平分时保留当前组合。
        """
        started = perf_counter()
        old = self.score.ic
        extended = self.stats.extend(tree)
        if extended is self.stats:
            return 0.0
        n = len(self.stats.trees)
        options = [extended] if n < self.capacity else [extended.subset([j for j in range(n + 1) if j != i]) for i in range(n)]
        for option in options:
            score = option.score()
            self.evaluations += 1
            # 只接受超过数值容差的搜索期改善，平分时保留当前组合。
            if score.ic > self.score.ic + 1e-12:
                self.stats, self.score = option, score
        self.seconds += perf_counter() - started
        return self.score.ic - old
