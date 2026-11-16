"""候选生成与集合选择共用的有向单因子、等权组合和成员贡献评价。"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
from time import perf_counter

import numpy as np
import torch

from .evaluation import load_slice, normalize
from .io import digest
from .._vendor.alphagen.utils.correlation import batch_spearmanr


def daily_correlation(first, second, mask=None):
    """在每日共同有限样本上计算 Pearson 相关，无有效截面时返回 NaN。"""
    valid = np.isfinite(first) & np.isfinite(second)
    if mask is not None:
        valid &= mask
    x, _, x_active = normalize(first, valid)
    y, _, y_active = normalize(second, valid)
    count = valid.sum(1)
    result = np.sum(x * y, axis=1) / np.maximum(count, 1)
    result[~(x_active & y_active)] = np.nan
    return np.clip(result, -1.0, 1.0)


def block_deviation(daily, block_days=10):
    """将每日 IC 的不可计算值记零，计算连续交易日块均值的总体标准差。"""
    values = np.nan_to_num(daily, nan=0.0)
    blocks = [values[start:start + block_days].mean() for start in range(0, len(values), block_days)]
    return float(np.std(blocks, ddof=0)) if blocks else 0.0


def quality_score(metrics, thresholds):
    """严格计算 Word 中的连续质量裕量与覆盖扣分，不可计算时 Q=-1。"""
    values = np.array([metrics[name] for name in ("ic", "rank_ic", "icir", "coverage")], dtype=float)
    if not np.isfinite(values).all() or metrics.get("invalid_values", False):
        return -1.0
    limits = np.asarray(thresholds, dtype=float)
    margins = np.clip((values[:3] - limits[:3]) / limits[:3], -1.0, 1.0)
    return float(margins.mean() - max(0.0, (limits[3] - values[3]) / limits[3]))


def member_deficiency(metrics, thresholds):
    """计算每个成员四项门槛欠缺度的平均值，不可计算项记一。"""
    values = np.array([metrics[name] for name in ("ic", "rank_ic", "icir", "coverage")], dtype=float)
    limits = np.asarray(thresholds, dtype=float)
    deficiencies = np.ones(4)
    finite = np.isfinite(values)
    deficiencies[finite] = np.clip((limits[finite] - values[finite]) / limits[finite], 0.0, 1.0)
    return float(deficiencies.mean())


def load_selection_task(task, config, with_feedback=False, with_future=False):
    """加载 F/G/A 及显式请求的 R/验收段，按标签成熟日验证相邻数据边界。"""
    names = ["fit", "search", "validation"]
    if with_feedback:
        names.append("feedback")
    panels = {name: load_slice(task[name], config["target_horizon"], config["lookback"],
                               config["future_days"]) for name in names}
    for name, panel in panels.items():
        if not panel.panel.dates.is_monotonic_increasing or not panel.panel.dates.is_unique:
            raise ValueError(f"{task['name']}: {name} trading dates must be strictly increasing")
    for previous, current in zip(names, names[1:]):
        first, second = panels[previous], panels[current]
        if first.panel.stock_ids != second.panel.stock_ids:
            raise ValueError(f"{task['name']}: stock identity/order differs between {previous} and {current}")
        if first.ends.max() >= second.dates.min():
            raise ValueError(f"{task['name']}: immature {previous} labels overlap {current}")
    if with_future:
        final = load_slice(task["future"], config["target_horizon"], config["lookback"], config["future_days"])
        previous = panels[names[-1]]
        if previous.ends.max() >= final.dates.min() or previous.panel.stock_ids != final.panel.stock_ids:
            raise ValueError(f"{task['name']}: future labels/stock identity violate the locked boundary")
        panels["future"] = final
    return panels


@dataclass(frozen=True)
class Candidate:
    """已按 G 段准入的候选及其生成来源，方向由 F 段唯一确定。"""
    tree: object
    direction: int
    quality: float
    search_metrics: dict
    signal_hash: str
    proposal_id: str


class SelectionEvaluator:
    """缓存原始单因子评价，向策略仅提供指定的历史可见特征。"""

    def __init__(self, grammar, panels, config):
        """绑定任务数据、门槛、奖励权重与有界信号缓存。"""
        self.grammar, self.panels, self.config = grammar, panels, config
        self.thresholds = tuple(config["reward"]["thresholds"])
        self.cache = OrderedDict()
        self.metrics = {}
        self.directions = {}
        self.errors = {}
        self.assessments = {}
        self.calls = 0
        self.combination_seconds = 0.0

    def segment(self, tree, name):
        """在填零前统计单因子指标，并独立返回用于组合的标准化信号。"""
        key = (tree.id, name)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        panel = self.panels[name]
        self.calls += 1
        try:
            self.grammar.validate(tree)
            raw = self.grammar.compile(tree).evaluate(panel.panel).detach().cpu().numpy().astype(np.float64)
            if raw.shape != panel.target.shape:
                raise ValueError("formula output differs from the scoring grid")
            valid = np.isfinite(raw) & panel.mask
            daily = daily_correlation(raw, panel.target)
            finite = np.isfinite(daily)
            ic = float(daily[finite].mean()) if finite.any() else float("nan")
            deviation = float(daily[finite].std(ddof=0)) if finite.any() else 0.0
            x = torch.tensor(np.where(valid, raw, np.nan), dtype=torch.float64)
            y = torch.tensor(np.where(valid, panel.target, np.nan), dtype=torch.float64)
            ranks = batch_spearmanr(x, y).numpy()
            ranks[~finite] = np.nan
            rank_ic = float(np.nanmean(ranks)) if np.isfinite(ranks).any() else float("nan")
            metrics = {"ic": ic, "rank_ic": rank_ic,
                       "icir": ic / deviation if deviation > 0.0 else float("nan"),
                       "coverage": float(valid.sum() / max(panel.mask.sum(), 1)),
                       "days": int(finite.sum()), "invalid_day_fraction": float((~finite).mean()),
                       "block_std": block_deviation(daily, self.config["reward"]["block_days"]),
                       "invalid_values": bool(np.isinf(raw).any())}
            # 标准化使用当日因子有效股票，避免收益标签的缺失改变可部署的因子信号。
            z, _, _ = normalize(raw, np.isfinite(raw))
            result = (raw, z, daily, metrics)
        except (ValueError, RuntimeError, FloatingPointError, IndexError, KeyError) as error:
            self.errors[key] = f"{type(error).__name__}: {error}"
            metrics = {"ic": float("nan"), "rank_ic": float("nan"), "icir": float("nan"),
                       "coverage": 0.0, "days": 0, "invalid_day_fraction": 1.0,
                       "block_std": 0.0, "invalid_values": True}
            result = (np.full_like(panel.target, np.nan), np.zeros_like(panel.target),
                      np.full(len(panel.dates), np.nan), metrics)
        self.metrics[key] = result[3]
        self.cache[key] = result
        while len(self.cache) > self.config["signal_cache_size"]:
            self.cache.popitem(last=False)
        return result

    def direction(self, tree):
        """仅使用 F 段 IC 确定方向，F 不可计算时返回零并阻止候选准入。"""
        if tree.id not in self.directions:
            ic = self.segment(tree, "fit")[3]["ic"]
            self.directions[tree.id] = (1 if ic >= 0 else -1) if np.isfinite(ic) else 0
        return self.directions[tree.id]

    def directed_metrics(self, tree, name, direction=None):
        """将 F 固定方向应用于指定段的 IC、RankIC 和 ICIR，覆盖率保持原义。"""
        key = (tree.id, name)
        metrics = dict(self.metrics[key] if key in self.metrics else self.segment(tree, name)[3])
        direction = self.direction(tree) if direction is None else direction
        if key in self.errors:
            metrics["error"] = self.errors[key]
        elif not direction:
            metrics["error"] = self.errors.get((tree.id, "fit"), "fit IC is not computable")
        for metric in ("ic", "rank_ic", "icir"):
            metrics[metric] = metrics[metric] * direction if direction else float("nan")
        return metrics

    def assess(self, tree, proposal_id):
        """计算 G 段质量、四项准入和精确信号身份，不读取 A/R。"""
        if tree.id in self.assessments:
            candidate, eligible = self.assessments[tree.id]
            return replace(candidate, proposal_id=proposal_id), eligible
        metrics = self.directed_metrics(tree, "search")
        quality = quality_score(metrics, self.thresholds)
        values = np.array([metrics[name] for name in ("ic", "rank_ic", "icir", "coverage")])
        eligible = bool(np.isfinite(values).all() and (values >= self.thresholds).all()
                        and not metrics["invalid_values"] and self.direction(tree))
        z = self.segment(tree, "search")[1] * self.direction(tree)
        # 数值相同的有向标准化信号共用身份，公式规范化身份由 tree.id 提供。
        signal_hash = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
        candidate = Candidate(tree, self.direction(tree), quality, metrics, signal_hash, proposal_id)
        self.assessments[tree.id] = (candidate, eligible)
        return candidate, eligible

    def observation(self, candidates, maximum):
        """只根据 A 段建立成员特征与保留符号的共同有效样本相关矩阵。"""
        from .selection_models import SelectionObservation
        features, raw_signals = [], []
        for candidate in candidates:
            metrics = self.directed_metrics(candidate.tree, "validation", candidate.direction)
            values = [metrics["ic"], metrics["rank_ic"], np.tanh(metrics["icir"]), metrics["coverage"],
                      metrics["block_std"], metrics["invalid_day_fraction"]]
            features.append(tuple(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)))
            raw_signals.append(self.segment(candidate.tree, "validation")[0] * candidate.direction)
        correlations = np.eye(len(candidates))
        for i, first in enumerate(raw_signals):
            for j in range(i):
                daily = daily_correlation(first, raw_signals[j])
                rho = float(np.nanmean(daily)) if np.isfinite(daily).any() else 0.0
                correlations[i, j] = correlations[j, i] = rho
        identity = digest([(candidate.tree.id, candidate.direction, candidate.proposal_id) for candidate in candidates])
        return SelectionObservation(tuple(c.tree for c in candidates), tuple(features),
                                    tuple(map(tuple, correlations)), (), maximum, identity)

    def utility(self, candidates, indices, segment="feedback", weights=None):
        """以同一有向等权组合计算三项上层奖励，空组合奖励为零。"""
        started = perf_counter()
        indices = tuple(indices)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(candidates) for i in indices):
            raise ValueError("selected members must be distinct valid candidate indices")
        if not indices:
            return {"utility": 0.0, "ic": 0.0, "block_std": 0.0, "deficiency": 0.0,
                    "prediction_score": 0.0, "stability_penalty": 0.0, "quality_penalty": 0.0,
                    "invalid_day_fraction": 0.0, "daily_ic": [], "members": []}
        members = [candidates[i] for i in indices]
        weights = np.full(len(members), 1 / len(members)) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != (len(members),) or not np.isfinite(weights).all():
            raise ValueError("one finite locked weight per selected member is required")
        prediction = sum(self.segment(c.tree, segment)[1] * c.direction * weight for c, weight in zip(members, weights))
        daily = daily_correlation(prediction, self.panels[segment].target)
        invalid = ~np.isfinite(daily)
        daily = np.nan_to_num(daily, nan=0.0)
        mean = float(daily.mean())
        stability = block_deviation(daily, self.config["reward"]["block_days"])
        deficiency_segment = self.config["reward"]["member_segment"]
        # 独立验收时，反馈型 H 同样在验收段评价，方向与成员均已锁定。
        if deficiency_segment == "feedback":
            deficiency_segment = segment
        metrics = [self.directed_metrics(c.tree, deficiency_segment, c.direction) for c in members]
        deficiency = float(np.mean([member_deficiency(m, self.thresholds) for m in metrics]))
        weights = self.config["reward"]
        prediction_score = weights["alpha"] * mean
        stability_penalty = weights["beta"] * stability
        quality_penalty = weights["gamma"] * deficiency
        self.combination_seconds += perf_counter() - started
        return {"utility": prediction_score - stability_penalty - quality_penalty,
                "ic": mean, "block_std": stability, "deficiency": deficiency,
                "prediction_score": prediction_score, "stability_penalty": stability_penalty,
                "quality_penalty": quality_penalty, "invalid_day_fraction": float(invalid.mean()),
                "daily_ic": daily.tolist(), "members": metrics}

    def contributions(self, candidates, indices, segment="feedback"):
        """固定已选成员并逐个移除，重新等权，返回 D 与两侧效用证据。"""
        whole = self.utility(candidates, indices, segment)
        result = {}
        for index in indices:
            reduced = self.utility(candidates, [i for i in indices if i != index], segment)
            result[candidates[index].proposal_id] = {"value": whole["utility"] - reduced["utility"],
                "whole_utility": whole["utility"], "without_utility": reduced["utility"],
                "factor_id": candidates[index].tree.id, "remaining_count": len(indices) - 1}
        return whole, result
