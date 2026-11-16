# 完整搜索回合，串起上层选起点、下层编辑、真实评价、去重和组合更新。
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import random
from time import perf_counter

import numpy as np
import torch

from .ast import walk
from .evaluation import PoolStats, WorkingPool
from .io import append_json, check_budget
from .models import LowerTask, Observation
from .ppo import complete_returns, continuation_returns


def family(tree):
    """递归提取公式结构家族，将所有叶节点统一标记，用于语料划分和短名单多样性。"""
    return (tree.symbol if tree.children else "leaf", tuple(family(c) for c in tree.children))


@dataclass
class Entry:
    tree: object
    quality: float
    known: bool = False
    attempts: int = 0
    reward_sum: float = 0.0
    qualified: int = 0
    duplicates: int = 0
    exposures: int = 0
    seconds: float = 0.0
    contributions: int = 0

    def statistics(self):
        # 依次提供质量、扩展次数、平均奖励、发现率、重复率、初始已知及未访问标记。
        """返回起点质量、尝试数、平均奖励、发现率、重复率及已知和未访问标记。

        次数取对数，比例用至少为一的分母计算，供上层作为固定维度统计输入。
        """
        denominator = max(self.attempts, 1)
        return (self.quality, math.log1p(self.attempts), self.reward_sum / denominator,
                self.qualified / denominator, self.duplicates / denominator,
                float(self.known), float(self.attempts == 0))


class Archive:
    # entries 存可扩展起点，seen 存已见公式，accepted 存本回合合格新发现。
    # 工作组合单独维护，公式进入起点档案与成为有效新发现采用不同条件。
    def __init__(self, evaluator, initial, config):
        """用公共起点建立本回合档案、去重索引和工作组合。

        初始组合按拟合期质量排序并排除等价信号，记录初始搜索期组合 IC 作为增量基线。
        """
        self.evaluator, self.config = evaluator, config
        self.entries = {}
        self.seen = set()
        self.starting_points = set()
        self.signal_aliases = set()
        self.accepted = []
        self.known_comparators = []
        self.pair_cache = {}
        self.new_attempts, self.new_reward_sum = 0, 0.0
        self.new_qualified, self.new_duplicates = 0, 0
        self.new_seconds, self.new_contributions = 0.0, 0
        self.recent_discoveries = deque(maxlen=2)
        self.discoveries = []
        self.shortlist_seconds = self.novelty_seconds = 0.0
        indices = np.flatnonzero(evaluator.fit.active)
        self.novelty_indices = indices[np.argsort((np.arange(len(indices)) * 37) % len(indices))]
        self.novelty_prefixes = []
        self.novelty_matrix = None
        initial_reliable = []
        for tree in initial:
            signal = evaluator.get(tree)
            metrics = signal.metrics
            self.entries[tree.id] = Entry(tree, metrics["quality"], known=True)
            self.seen.add(tree.id)
            self.starting_points.add(tree.id)
            if metrics["reliable"]:
                self.signal_aliases.add(metrics["signal_hash"])
                initial_reliable.append(tree)
                self.known_comparators.append(tree)
                self.index_signal(signal)
        # 初始组合按拟合期成绩和信号等价性选择，搜索期用于后续成果评价。
        ranked = sorted(initial_reliable, key=lambda t: (-abs(evaluator.metrics[t.id]["fit"]["ic"]), t.id))
        selected = []
        for tree in ranked:
            candidate = evaluator.get(tree).fit
            if any(np.array_equal(candidate, evaluator.get(old).fit) or
                   np.array_equal(candidate, -evaluator.get(old).fit) for old in selected):
                continue
            selected.append(tree)
            if len(selected) >= config["pool_capacity"]:
                break
        self.pool = WorkingPool(evaluator, config["pool_capacity"], selected)
        self.initial_ic = self.pool.score.ic

    def shortlist(self, limit):
        # 从质量、历史产出、低访问次数和结构多样性四类排序轮换取公共短名单。
        """从最近发现、质量、历史产出、低访问次数和结构多样性中选取公共短名单。

        最多返回 limit 个起点，同时更新入选次数与短名单构造耗时。
        """
        started = perf_counter()
        values = list(self.entries.values())
        quality = sorted(values, key=lambda x: (-x.quality, x.exposures, x.tree.id))
        productivity = sorted(values, key=lambda x: (-x.reward_sum / max(x.attempts, 1), x.exposures, x.tree.id))
        underused = sorted(values, key=lambda x: (x.attempts, x.exposures, x.tree.id))
        families = {}
        for entry in underused:
            families.setdefault(family(entry.tree), []).append(entry)
        diverse = []
        while families:
            for key in list(families):
                diverse.append(families[key].pop(0))
                if not families[key]:
                    del families[key]
        groups, result, selected = [quality, productivity, underused, diverse], [], set()
        for key in reversed(self.recent_discoveries):
            if len(result) < limit:
                result.append(self.entries[key])
                selected.add(key)
        cursors = [0] * len(groups)
        while len(result) < min(limit, len(values)):
            for i, group in enumerate(groups):
                while cursors[i] < len(group) and group[cursors[i]].tree.id in selected:
                    cursors[i] += 1
                if cursors[i] < len(group) and len(result) < limit:
                    entry = group[cursors[i]]
                    result.append(entry)
                    selected.add(entry.tree.id)
                    cursors[i] += 1
        for entry in result:
            entry.exposures += 1
        self.shortlist_seconds += perf_counter() - started
        return result

    def observe(self, remaining_fraction=1.0, include_archive=False, cost_context=False):
        # 构造上层状态，包含候选统计、当前组合、剩余预算及可选档案背景。
        """构造当前上层观测，包含短名单统计、工作组合、NEW 统计和剩余预算。

        开关控制是否附加已见档案、相对平均扩展耗时与组合贡献率，返回 Observation。
        """
        entries = self.shortlist(self.config["shortlist_size"])
        denominator = max(self.new_attempts, 1)
        new_stats = (0., math.log1p(self.new_attempts), self.new_reward_sum / denominator,
                     self.new_qualified / denominator, self.new_duplicates / denominator,
                     0., float(self.new_attempts == 0))
        statistics = [entry.statistics() for entry in entries]
        if cost_context:
            # 成本观测实验额外加入相对平均扩展耗时和可见组合贡献率。
            total = self.new_attempts + sum(entry.attempts for entry in self.entries.values())
            seconds = self.new_seconds + sum(entry.seconds for entry in self.entries.values())
            mean = seconds / max(total, 1)
            def extra(attempts, elapsed, contributions):
                """根据历史尝试数计算相对平均耗时的对数和贡献率，供成本观测分支追加两维特征。"""
                relative = elapsed / attempts / mean if attempts and mean > 0 else 1.0
                return (math.log1p(relative), contributions / max(attempts, 1))
            statistics = [stats + extra(entry.attempts, entry.seconds, entry.contributions)
                          for stats, entry in zip(statistics, entries)]
            new_stats += extra(self.new_attempts, self.new_seconds, self.new_contributions)
        return Observation(tuple(e.tree.id for e in entries), tuple(e.tree for e in entries),
                           tuple(statistics), self.pool.stats.trees,
                           self.pool.score.weights, self.pool.score.ic, remaining_fraction, new_stats,
                           tuple(self.evaluator.trees[key] for key in sorted(self.seen)) if include_archive else ())

    def index_signal(self, signal):
        # 用前八个评分日计算平均绝对相关性的上界，提前排除明显低相关候选。
        # 无法由上界排除的候选继续使用完整评分日期核验。
        """将信号的前八个选定拟合日加入相关性索引，并使批量矩阵缓存失效。"""
        self.novelty_prefixes.append(signal.fit[self.novelty_indices[:8]])
        self.novelty_matrix = None

    def prefix_correlations(self, signal):
        """计算候选与索引信号在前八个选定日的加权截面相关性，返回信号数乘日期数的矩阵。"""
        if not self.novelty_prefixes:
            return np.empty((0, min(8, len(self.novelty_indices))))
        if self.novelty_matrix is None:
            self.novelty_matrix = np.stack(self.novelty_prefixes)
        indices = self.novelty_indices[:8]
        x, weights = signal.fit[indices], self.evaluator.fit.weights[indices]
        covariance = np.einsum("rds,ds->rd", self.novelty_matrix, weights * x)
        old_norm = np.einsum("rds,rds,ds->rd", self.novelty_matrix, self.novelty_matrix, weights)
        new_norm = np.sum(weights * x ** 2, axis=1)
        denominator = np.sqrt(old_norm * new_norm)
        return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-24)

    def novel(self, tree):
        # 高相关候选还会做双因子增量检查，具有额外信息时可以通过去冗余。
        """检查候选是否具有新增信息，返回通过标记及相关性、冗余性诊断。

        信号哈希检查后用相关性上界剪枝，高相关信号还须通过与相近信号的双因子增量检查。
        过程中复用成对检查缓存，并累计去冗余耗时。
        """
        started = perf_counter()
        signal = self.evaluator.get(tree)
        if signal.metrics["signal_hash"] in self.signal_aliases:
            self.novelty_seconds += perf_counter() - started
            return False, {"reason": "signal_equivalent", "rho": 1.0}
        comparisons = self.known_comparators + self.accepted
        prefixes = np.clip(abs(self.prefix_correlations(signal)), 0, 1).sum(1)
        indices = self.novelty_indices
        examined_prefix = min(8, len(indices))
        closest, highest = None, -1.0
        for number, previous in enumerate(comparisons):
            key = tuple(sorted((tree.id, previous.id)))
            if key not in self.pair_cache:
                total = float(prefixes[number])
                upper_bound = (total + len(indices) - examined_prefix) / len(indices)
                if upper_bound + 1e-12 < self.config["correlation_threshold"]:
                    self.pair_cache[key] = {"rho_upper_bound": upper_bound, "safely_pruned": True}
                    continue
                old = self.evaluator.get(previous).fit
                weights = self.evaluator.fit.weights
                pruned = False
                for start in range(examined_prefix, len(indices), 8):
                    selected = indices[start:start + 8]
                    x, y, w = signal.fit[selected], old[selected], weights[selected]
                    covariance = np.sum(w * x * y, axis=1)
                    denominator = np.sqrt(np.sum(w * x ** 2, axis=1) * np.sum(w * y ** 2, axis=1))
                    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-24)
                    total += float(np.clip(abs(correlation), 0, 1).sum())
                    examined = min(start + 8, len(indices))
                    upper_bound = (total + len(indices) - examined) / len(indices)
                    if upper_bound + 1e-12 < self.config["correlation_threshold"]:
                        self.pair_cache[key] = {"rho_upper_bound": upper_bound, "safely_pruned": True}
                        pruned = True
                        break
                if not pruned:
                    self.pair_cache[key] = {"rho": total / len(indices)}
            if self.pair_cache[key].get("safely_pruned"):
                continue
            rho = self.pair_cache[key]["rho"]
            if rho > highest:
                closest, highest = previous, rho
        result = {"rho": highest, "closest": closest.id if closest else None}
        if closest is not None and highest >= self.config["correlation_threshold"]:
            near = [(self.pair_cache[tuple(sorted((tree.id, old.id)))].get("rho", -1), old)
                    for old in comparisons]
            near = sorted((item for item in near if item[0] >= self.config["correlation_threshold"]),
                          key=lambda item: (-item[0], item[1].id))
            accepted, result["pair_gain"], result["neighbors_checked"] = True, float("inf"), 0
            for _, previous in near:
                key = tuple(sorted((tree.id, previous.id)))
                if "gain" not in self.pair_cache[key]:
                    pair = PoolStats(self.evaluator, [tree, previous])
                    joint = pair.score().ic
                    best = max(pair.subset([0]).score().ic, pair.subset([1]).score().ic)
                    self.pair_cache[key]["gain"] = joint - best
                result["neighbors_checked"] += 1
                result["pair_gain"] = min(result["pair_gain"], self.pair_cache[key]["gain"])
                if self.pair_cache[key]["gain"] < self.config["pair_gain_threshold"]:
                    accepted, result["redundant_to"] = False, previous.id
                    break
            result["reason"] = "near_signal_with_gain" if accepted else "redundant_signal"
        else:
            accepted, result["reason"] = True, "distinct_signal"
        self.novelty_seconds += perf_counter() - started
        return accepted, result

    def assess(self, tree):
        """试算候选，发现档案和工作组合的正式提交由后续步骤负责。"""
        # 这里只试算候选。主候选被计入预算后，才由 commit 更新真实档案。
        metrics = self.evaluator.get(tree).metrics
        result = {"tree": tree, "metrics": metrics, "new": False, "duplicate": tree.id in self.seen,
                  "gain": 0.0, "reward": 0.0, "pool": self.pool}
        # 重复、不可靠或未过质量门槛时，proposal 的 new=False、reward=0。
        if result["duplicate"] or not metrics["reliable"]:
            return result
        # 对质量低于门槛的公式也执行精确信号等价检查。
        if metrics["signal_hash"] in self.signal_aliases:
            result["duplicate"] = True
            return result
        if metrics["quality"] < self.config["quality_threshold"]:
            return result
        if metrics["direction"] * metrics["search"]["rank_ic"] < self.config.get("min_search_rank_ic", -1.):
            return result
        for name, metric in (("min_train_ic", "raw_ic"), ("min_icir", "icir")):
            threshold = self.config.get(name)
            if threshold is not None and abs(metrics["fit"][metric]) < threshold:
                return result
        folds = metrics.get("quality_fold_ics", [])
        required = self.config.get("min_positive_fold_fraction", 0.0)
        if folds and sum(value > 0 for value in folds) / len(folds) < required:
            return result
        novel, detail = self.novel(tree)
        result["novelty"] = detail
        result["duplicate"] = not novel
        if not novel:
            return result
        proposed = object.__new__(WorkingPool)
        proposed.__dict__ = self.pool.__dict__.copy()
        gain = proposed.consider(tree)
        # 原方案上层扩展奖励为新增发现 1 分，加搜索期组合 IC 增量的加权分。
        # 后续共同奖励实验会在训练模块中对已收集轨迹重算目标。
        result.update(new=True, gain=gain, reward=1.0 + self.config["combination_weight"] * gain, pool=proposed)
        return result

    def commit(self, proposal, source, seconds=0.0):
        # 主候选才更新档案、工作组合和起点统计。续采样分支只用于估计动作价值。
        """正式提交一个主候选，更新已见集合、可扩展起点、发现记录和工作组合。

        按来源累计奖励、发现、重复、耗时及贡献次数，供后续调度观测使用。
        """
        tree, metrics = proposal["tree"], proposal["metrics"]
        self.seen.add(tree.id)
        if metrics["reliable"]:
            self.signal_aliases.add(metrics["signal_hash"])
            self.entries.setdefault(tree.id, Entry(tree, metrics["quality"]))
        if proposal["new"]:
            self.accepted.append(tree)
            self.recent_discoveries.append(tree.id)
            self.index_signal(self.evaluator.get(tree))
            self.discoveries.append({"ast": tree.to_dict(), "expression": tree.text(), **metrics})
            self.pool = proposal["pool"]
        if source in self.entries:
            entry = self.entries[source]
            entry.attempts += 1
            entry.reward_sum += proposal["reward"]
            entry.qualified += int(proposal["new"])
            entry.duplicates += int(proposal["duplicate"])
            entry.seconds += seconds
            entry.contributions += int(proposal["new"] and proposal["gain"] > 1e-12)
        else:
            self.new_attempts += 1
            self.new_reward_sum += proposal["reward"]
            self.new_qualified += int(proposal["new"])
            self.new_duplicates += int(proposal["duplicate"])
            self.new_seconds += seconds
            self.new_contributions += int(proposal["new"] and proposal["gain"] > 1e-12)


def select_baseline(method, observation, archive, rng):
    # 随机、质量、产出和 UCB 对照共用同一候选短名单及 NEW 动作。
    """在与 PPO 相同的短名单和 NEW 动作上执行指定对照调度，返回动作索引。

    支持随机、质量优先、平均产出和 UCB，随机选择及平局处理使用传入的随机数生成器。
    """
    size = len(observation.ids)
    if method == "random":
        return rng.randrange(size + 1)
    # NEW 作为尚未访问的探索动作参与比较，其成绩由实际扩展结果累计。
    entries = [archive.entries[key] for key in observation.ids]
    if method == "quality":
        return max(range(size), key=lambda i: entries[i].quality) if size else size
    if method == "productivity":
        if archive.new_attempts == 0:
            return size
        unseen = [i for i, e in enumerate(entries) if e.attempts == 0]
        if unseen:
            return rng.choice(unseen)
        scores = [e.reward_sum / e.attempts for e in entries] + [archive.new_reward_sum / archive.new_attempts]
    elif method == "ucb":
        # 平均扩展奖励加探索项，被尝试较少的起点获得更高探索机会。
        total = 1 + archive.new_attempts + sum(e.attempts for e in archive.entries.values())
        scores = [e.reward_sum / e.attempts + math.sqrt(2 * math.log(total + 1) / e.attempts)
                  if e.attempts else float("inf") for e in entries]
        scores.append(archive.new_reward_sum / archive.new_attempts + math.sqrt(2 * math.log(total + 1) / archive.new_attempts)
                      if archive.new_attempts else float("inf"))
    else:
        raise ValueError(f"unknown scheduler: {method}")
    best = max(scores)
    return rng.choice([i for i, score in enumerate(scores) if score == best])


def episode(lower, upper, evaluator, initial, config, rng, *, scheduler="ppo", output=None, tag="search", record_lower=True,
            lower_training=None):
    # 一次调用对应一个完整搜索回合，可按墙钟时间或主候选尝试数截止。
    """执行完整搜索回合，返回上层轨迹、下层轨迹、下层主终点分数和搜索报告。

    按时间或尝试数限制运行起点选择、正向扰动、下层编辑、候选评价与正式提交。
    可选续采样使用动作前档案，时间截止后完成的候选仅记录，发现统计以主候选提交为准。
    """
    check_budget()
    started = perf_counter()
    deadline = started + config["seconds"]
    time_limited = config.get("budget_mode") != "attempts"
    archive = Archive(evaluator, initial, config)
    rows, trajectories, main_scores = [], [], []
    attempted = late = new_failures = new_draws = duplicates = completed = 0
    curve = []
    continuation_states, branches = set(), 0
    if lower_training is not None:
        # 在本批非终止编辑状态中预选续采样位置，选取过程使用独立随机数对象。
        if not record_lower:
            raise ValueError("lower training requires recorded lower trajectories")
        branch_rng = random.Random()
        branch_rng.setstate(rng.getstate())
        horizon = config.get("diffusion_steps", lower.noise_steps)
        choices = [(i, j) for i in range(1, config["max_attempts"] + 1) for j in range(horizon - 1)]
        continuation_states = set(branch_rng.sample(choices, min(lower_training["continuation_states"], len(choices))))
    while attempted < config["max_attempts"] and (not time_limited or perf_counter() < deadline):
        check_budget()
        attempted += 1
        time_left = max(0.0, (deadline - perf_counter()) / config["seconds"])
        attempts_left = (config["max_attempts"] - attempted + 1) / config["max_attempts"]
        remaining = attempts_left if config.get("budget_mode") == "attempts" else time_left
        expansion_started = perf_counter()
        observation = archive.observe(remaining, include_archive=getattr(upper, "use_archive", False),
                                      cost_context=getattr(upper, "cost_context", False))
        # 上层只选择起点，实际修改位置和替换内容由下层策略决定。
        if scheduler == "ppo":
            row = upper.sample(observation)
            index = row["index"]
        else:
            index = select_baseline(scheduler, observation, archive, rng)
            row = {"observation": observation, "index": index}
        source = observation.ids[index] if index < len(observation.ids) else "NEW"
        try:
            seed = (archive.entries[source].tree if source != "NEW" else
                    lower.grammar.sample(rng, archive.seen | archive.starting_points))
            archive.starting_points.add(seed.id)
            horizon = config.get("diffusion_steps", lower.noise_steps)
            noisy, _ = lower.corrupt(seed, config.get("perturbation_steps", horizon), rng)
            condition = {}
            if getattr(lower, "use_task", False):
                # 后续条件接口实验让原起点和当前组合背景持续传入每一步编辑。
                directed_weights = tuple(weight * evaluator.metrics[tree.id]["direction"]
                                         for tree, weight in zip(observation.pool_trees, observation.pool_weights))
                condition["task"] = LowerTask(seed, observation.pool_trees,
                                               directed_weights, observation.pool_ic,
                                               tuple(archive.accepted))
            final, trajectory = lower.sample(noisy, remaining=horizon, record=record_lower, **condition)
            # 先进行可见的搜索期检查，后续历史标签重评分由训练模块单独处理。
            proposal = archive.assess(final)
        except RuntimeError as error:
            if source != "NEW" or "grammar prior exhausted" not in str(error):
                raise
            new_failures += 1
            archive.new_attempts += 1
            archive.new_seconds += perf_counter() - expansion_started
            new_draws += lower.grammar.last_prior_attempts
            row.update(reward=0.0, credited=False)
            rows.append(row)
            if output:
                append_json(output, {"tag": tag, "attempt": attempted, "source": source, "error": str(error), "elapsed": perf_counter() - started})
            continue
        if source == "NEW":
            new_draws += lower.grammar.last_prior_attempts
        record = {"tag": tag, "attempt": attempted, "source": source,
                  "upper_old_logp": row.get("logp"), "upper_value": row.get("value"),
                  "remaining_fraction": observation.remaining_fraction,
                  "seed_ast": seed.to_dict(), "noise_ast": noisy.to_dict(), "ast": final.to_dict(),
                  "expression": final.text(), "metrics": proposal["metrics"],
                  "duplicate": proposal["duplicate"], "novelty": proposal.get("novelty"),
                  "proposed_reward": proposal["reward"], "elapsed": perf_counter() - started,
                  "perturbation_node_count_before": sum(1 for _ in walk(seed)),
                  "perturbation_node_count_after": sum(1 for _ in walk(noisy)),
                  "perturbation_changed_positions": sum(dict(walk(seed)).get(p) != dict(walk(noisy)).get(p)
                                                        for p in set(dict(walk(seed))) | set(dict(walk(noisy))))}
        if output:
            append_json(output, record)
        credited = not time_limited or perf_counter() <= deadline
        expansion_seconds = perf_counter() - expansion_started
        row.update(candidate=final.to_dict(), credited=credited)
        if getattr(upper, "cost_context", False):
            row["expansion_seconds"] = expansion_seconds
        if credited:
            if lower_training is not None:
                # 原方案下层用拟合期单因子奖励，共同完成反馈实验改用档案完成评分。
                main_reward = proposal["reward"] if lower_training.get("completion_reward") else evaluator.lower(final)
                evaluate = (lambda tree: archive.assess(tree)["reward"]) if lower_training.get("completion_reward") else evaluator.lower
                selected = {(0, j) for j in range(len(trajectory)) if (attempted, j) in continuation_states}
                # 所有辅助分支共用动作前档案，并恢复主候选的 PyTorch 随机流。
                device = next(lower.parameters()).device
                devices = [device.index] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    branches += continuation_returns(lower, [trajectory], [main_reward], evaluate,
                        samples=lower_training["continuation_samples"], selected=selected)
            archive.commit(proposal, source, expansion_seconds)
            completed += 1
            duplicates += int(proposal["duplicate"])
            row["reward"] = proposal["reward"]
            if record_lower:
                row["lower_trajectory_index"] = len(trajectories)
                trajectories.append(trajectory)
                main_scores.append(main_reward if lower_training is not None else evaluator.lower(final))
        else:
            # 截止后完成的候选记录耗时，实际发现与组合保持在截止前的状态。
            late += 1
            row["reward"] = 0.0
        if output:
            append_json(output, {"event": "credit", "tag": tag, "attempt": attempted,
                                "reward": row["reward"], "credited": credited,
                                "excluded_as_late": not credited,
                                "elapsed": perf_counter() - started})
        rows.append(row)
        curve.append({"seconds": perf_counter() - started, "new_factors": len(archive.accepted), "pool_ic": archive.pool.score.ic})
    total = complete_returns(rows)
    if not time_limited and (attempted != config["max_attempts"] or late):
        raise AssertionError("fixed-attempt episode returned an incomplete budget")
    expected = len(archive.accepted) + config["combination_weight"] * (archive.pool.score.ic - archive.initial_ic)
    if abs(total - expected) > 1e-8:
        raise AssertionError("discovery-count/combination reward telescoping invariant failed")
    elapsed = perf_counter() - started
    report = {"tag": tag, "scheduler": scheduler, "new_factors": len(archive.accepted),
              "lower_continuation_branches": branches,
              "attempts": attempted, "late_results_excluded": late, "return": total,
              "completed_candidates": completed, "duplicate_count": duplicates,
              "duplicate_rate": duplicates / max(completed, 1),
              "best_single_ic": max((d["quality"] for d in archive.discoveries), default=None),
              "best_single_rank_ic": max((d["direction"] * d["search"]["rank_ic"] for d in archive.discoveries), default=None),
              "best_single_icir": max((d["direction"] * d["search"]["icir"] for d in archive.discoveries), default=None),
              "initial_pool_ic": archive.initial_ic, "pool_ic": archive.pool.score.ic,
              "pool_ids": archive.pool.score.ids, "pool_weights": archive.pool.score.weights,
              "pool_asts": [t.to_dict() for t in archive.pool.stats.trees],
              "discoveries": archive.discoveries, "curve": curve,
              "budget_seconds": config["seconds"], "wall_seconds": elapsed,
              "budget_mode": "wall_seconds" if time_limited else "attempts",
              "new_per_budget_second": len(archive.accepted) / config["seconds"],
              "new_per_actual_second": len(archive.accepted) / max(elapsed, 1e-9),
              "evaluator_calls": evaluator.calls, "evaluation_requests": evaluator.requests,
              "evaluation_cache_hits": evaluator.hits, "evaluation_seconds": evaluator.seconds,
              "combination_evaluations": evaluator.combination_evaluations,
              "combination_seconds": evaluator.combination_seconds,
              "shortlist_seconds": archive.shortlist_seconds, "novelty_seconds": archive.novelty_seconds,
              "new_grammar_calls": new_draws, "new_grammar_exhaustions": new_failures,
              "initialization_included_in_budget": True}
    return rows, trajectories, main_scores, report
