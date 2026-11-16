"""冻结起点批采样、候选准入以及上层成员选择的真实执行路径。"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import random
from time import perf_counter

import numpy as np
import torch

from .io import check_budget, append_json
from .models import Observation
from .ppo import complete_returns, gae_targets
from .selection_evaluation import SelectionEvaluator, load_selection_task
from .selection_models import GeneratorPolicy


class StartBank:
    """每个任务独立维护的合法起点与 G 段扩展统计，批末更新。"""

    def __init__(self, initial):
        """登记初始公式，后续质量只根据该任务的 F/G 重新计算。"""
        self.entries = {tree.id: {"tree": tree, "attempts": 0, "reward_sum": 0.0,
                                  "qualified": 0, "duplicates": 0, "known": True} for tree in initial}
        self.new = {"attempts": 0, "reward_sum": 0.0, "qualified": 0, "duplicates": 0}

    def observe(self, evaluator, limit, remaining=1.0):
        """从质量与低访问起点轮换构造固定短名单，末动作由网络表示 NEW。"""
        entries = list(self.entries.values())
        qualities = {entry["tree"].id: evaluator.assess(entry["tree"], "start")[0].quality for entry in entries}
        groups = [sorted(entries, key=lambda e: (-qualities[e["tree"].id], e["tree"].id)),
                  sorted(entries, key=lambda e: (e["attempts"], e["tree"].id))]
        chosen, seen = [], set()
        for pair in zip(*groups):
            for entry in pair:
                if entry["tree"].id not in seen and len(chosen) < limit:
                    chosen.append(entry)
                    seen.add(entry["tree"].id)
            if len(chosen) >= min(limit, len(entries)):
                break

        def statistics(entry, quality, known):
            """将 G 段质量和历史扩展统计映射到复用起点网络的七维输入。"""
            count = max(entry["attempts"], 1)
            return (quality, np.log1p(entry["attempts"]), entry["reward_sum"] / count,
                    entry["qualified"] / count, entry["duplicates"] / count,
                    float(known), float(entry["attempts"] == 0))

        return Observation(tuple(e["tree"].id for e in chosen), tuple(e["tree"] for e in chosen),
                           tuple(statistics(e, qualities[e["tree"].id], e["known"]) for e in chosen),
                           (), (), 0.0, remaining, statistics(self.new, 0.0, False))

    def commit(self, proposals):
        """批末一次性提交 G 段质量成本奖励统计与合格新起点，保持 R 信息不可见。"""
        for proposal in proposals:
            entry = self.entries.get(proposal["source_id"], self.new)
            entry["attempts"] += 1
            # 统计只存 Q-κC，贡献 D 的 R 反馈不进入下一次起点观测。
            entry["reward_sum"] += proposal["base_reward"]
            entry["qualified"] += int(proposal["eligible"])
            entry["duplicates"] += int(proposal.get("duplicate", False))
            if proposal["eligible"]:
                candidate = proposal["candidate"]
                self.entries.setdefault(candidate.tree.id, {"tree": candidate.tree, "attempts": 0,
                    "reward_sum": 0.0, "qualified": 0, "duplicates": 0, "known": False})


class CandidatePool:
    """按 G 段准入、规范式/精确信号去重并以 Q 保留容量内候选。"""

    def __init__(self, capacity, initial=()):
        """恢复候选背景及完整已见身份，保留原始提案来源用于 D 归因。"""
        self.capacity = capacity
        self.candidates = list(initial)
        self.ids = {candidate.tree.id for candidate in initial}
        self.hashes = {candidate.signal_hash for candidate in initial}
        self.discoveries = list(initial)

    def add(self, proposal):
        """先判断完全重复，再接纳四项达标候选，容量淘汰与准入分别记账。"""
        candidate = proposal["candidate"]
        duplicate = candidate is not None and (candidate.tree.id in self.ids or candidate.signal_hash in self.hashes)
        proposal["duplicate"] = duplicate
        proposal["accepted"] = bool(proposal["eligible"] and not duplicate)
        if proposal["accepted"]:
            self.ids.add(candidate.tree.id)
            self.hashes.add(candidate.signal_hash)
            self.discoveries.append(candidate)
            self.candidates.append(candidate)
            self.candidates.sort(key=lambda c: (-c.quality, c.tree.id))
            del self.candidates[self.capacity:]


def collect_proposal(generator, evaluator, observation, settings, seed, proposal_id, record=True):
    """用独立随机流生成一个主候选，并测量选起点、扩散与真实评价的完整耗时。"""
    device = getattr(generator, "device", None)
    if device is None:
        device = next(generator.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        started = perf_counter()
        final, rows, source = generator.generate(observation, random.Random(seed), settings["diffusion_steps"],
                                                settings["perturbation_steps"], record)
        candidate, eligible = evaluator.assess(final, proposal_id) if final is not None else (None, False)
        finished = perf_counter()
        elapsed = finished - started
    source_id = "NEW"
    if rows and rows[0].get("index", len(observation.ids)) < len(observation.ids):
        source_id = observation.ids[rows[0]["index"]]
    elif not record and source is not None and source.id in observation.ids:
        source_id = source.id
    source_id = getattr(generator, "last_source_id", None) or source_id
    return {"candidate": candidate, "eligible": eligible, "seconds": elapsed, "rows": rows,
            "source_id": source_id, "seed": seed, "proposal_id": proposal_id, "finished_at": finished}


def initialize_worker(config):
    """在 CPU 采集进程初始化网络和有界的任务评价器缓存。"""
    global _config, _generator, _evaluators
    from .ast import Grammar
    _config = config
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    grammar = Grammar(**config["grammar"])
    _generator = GeneratorPolicy(grammar, config["width"], config["corpus"]["noise_steps"])
    _evaluators = {}


def collect_worker(job):
    """同步载入冻结下层版本，仅加载 F/G/A 并采集固定起点支持上的一批主候选。"""
    _generator.load_state_dict(job["state"])
    name = job["task"]["name"]
    if name not in _evaluators:
        panels = load_selection_task(job["task"], _config)
        _evaluators[name] = SelectionEvaluator(_generator.grammar, panels, _config)
    evaluator = _evaluators[name]
    return [collect_proposal(_generator, evaluator, job["observation"], _config["lower"], seed, proposal_id)
            for seed, proposal_id in job["proposals"]]


class ProposalCollector:
    """统一串行和多进程的提案身份、随机种子及冻结模型采集协议。"""

    def __init__(self, config):
        """按配置建立 CPU 工作进程，GPU 网络更新仍由主进程负责。"""
        self.config = config
        self.executor = None
        if config["rollout_workers"] > 1:
            self.executor = ProcessPoolExecutor(max_workers=config["rollout_workers"],
                                               initializer=initialize_worker, initargs=(config,))

    def collect(self, generator, evaluator, task, observation, seeds, identities, record=True):
        """为所有主提案使用同一批前观测与模型版本，按输入顺序返回结果。"""
        if self.executor is None or not record or not getattr(generator, "supports_parallel", True):
            return [collect_proposal(generator, evaluator, observation, self.config["lower"], seed, identity, record)
                    for seed, identity in zip(seeds, identities)]
        state = {name: value.detach().cpu().clone() for name, value in generator.state_dict().items()}
        jobs = [{"state": state, "task": task, "observation": observation,
                 "proposals": [(seed, identity)]} for seed, identity in zip(seeds, identities)]
        device = next(generator.parameters()).device
        result = []
        for batch in self.executor.map(collect_worker, jobs):
            result.extend(batch)
        # 采集器行中的旧观测、动作均为普通 Python/CPU 数据，可在主设备重放。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result

    def close(self):
        """等待同步采样结束并释放工作进程。"""
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


def build_candidate_pool(generator, evaluator, bank, config, rng, reference_seconds, task, count, tag, output,
                         *, collector=None, record=False, background=(), wall_seconds=None, generator_version=0):
    """为主算法及对比生成器统一计算预算、真实成本、准入、去重与批末提交。"""
    if not np.isfinite(reference_seconds) or reference_seconds <= 0:
        raise ValueError("a positive frozen reference_seconds is required")
    if hasattr(generator, "bind"):
        generator.bind(evaluator, reference_seconds)
    calls_before = evaluator.calls
    pool = CandidatePool(config["search"]["candidate_capacity"], background)
    started = perf_counter()
    deadline = started + wall_seconds if wall_seconds is not None else float("inf")
    proposals, attempts, late = [], 0, 0
    while attempts < count and perf_counter() < deadline:
        check_budget()
        size = min(config["lower"]["batch_formulas"], count - attempts)
        preparation = perf_counter()
        if getattr(generator, "uses_start_bank", True):
            observation = bank.observe(evaluator, config["lower"]["shortlist_size"], (count - attempts) / count)
        else:
            observation = Observation((), (), (), (), (), 0.0, (count - attempts) / count)
        preparation = (perf_counter() - preparation) / size
        if perf_counter() >= deadline:
            break
        seeds = [rng.randrange(2**31) for _ in range(size)]
        identities = [f"{tag}_{attempts + i}" for i in range(size)]
        if collector is None:
            batch = [collect_proposal(generator, evaluator, observation, config["lower"], seed, identity, record)
                     for seed, identity in zip(seeds, identities)]
        else:
            batch = collector.collect(generator, evaluator, task, observation, seeds, identities, record)
        credited = []
        for proposal in batch:
            proposal["seconds"] += preparation
            candidate = proposal["candidate"]
            quality = candidate.quality if candidate is not None else -1.0
            proposal["cost"] = proposal["seconds"] / reference_seconds
            proposal["base_reward"] = quality - config["reward"]["kappa"] * proposal["cost"]
            proposal["credited"] = proposal["finished_at"] <= deadline
            attempts += 1
            if proposal["credited"]:
                pool.add(proposal)
                credited.append(proposal)
                proposals.append(proposal)
            else:
                late += 1
            if output is not None:
                rows = proposal["rows"]
                error = candidate.search_metrics.get("error") if candidate else (
                    rows[0].get("error", "no generated formula") if rows else "no generated formula")
                append_json(output / "generation_attempts.jsonl", {"tag": tag, "task": task["name"],
                    "proposal_id": proposal["proposal_id"], "source_id": proposal["source_id"],
                    "expression": candidate.tree.text() if candidate else None,
                    "direction": candidate.direction if candidate else 0,
                    "metrics": candidate.search_metrics if candidate else None, "error": error,
                    "quality": quality, "cost": proposal["cost"], "seconds": proposal["seconds"],
                    "base_reward": proposal["base_reward"], "eligible": proposal["eligible"],
                    "duplicate": proposal.get("duplicate", False), "credited": proposal["credited"],
                    "generator_version": generator_version})
        bank.commit(credited)
        if hasattr(generator, "observe_batch"):
            generator.observe_batch(credited)
        if not record and tag.startswith("evaluation_") and len(pool.discoveries) >= config["search"].get("target_count", float("inf")):
            break
    elapsed = perf_counter() - started
    report = {"attempts": attempts, "completed_candidates": len(proposals), "late_results_excluded": late,
        "new_factors": len(pool.discoveries) - len(background), "wall_seconds": elapsed,
        "duplicate_count": sum(p.get("duplicate", False) for p in proposals),
        "candidate_count": len(pool.candidates), "generation_version": generator_version,
        "evaluator_calls": evaluator.calls - calls_before, "combination_seconds": 0.0,
        "budget_mode": "attempts" if wall_seconds is None else "wall_seconds", "budget_seconds": wall_seconds}
    return pool, proposals, report


def forward_exchange(evaluator, candidates, maximum):
    """只使用 A 段执行前向加入及严格改善的一换一搜索，确定性打破同分。"""
    selected = []
    value = 0.0
    while len(selected) < min(maximum, len(candidates)):
        options = [(evaluator.utility(candidates, [*selected, i], "validation")["utility"], -i, i)
                   for i in range(len(candidates)) if i not in selected]
        best, _, index = max(options)
        if best <= value + 1e-12:
            break
        selected.append(index)
        value = best
    while selected:
        options = [(evaluator.utility(candidates, [j if j != old else new for j in selected], "validation")["utility"],
                    -old, -new, old, new)
                   for old in selected for new in range(len(candidates)) if new not in selected]
        if not options:
            break
        best, _, _, old, new = max(options)
        if best <= value + 1e-12:
            break
        selected[selected.index(old)] = new
        value = best
    return tuple(selected)


def select_members(selector, observation, evaluator, candidates, *, segment="feedback", method="upper_ppo",
                   rng=None, deterministic=False, trace_lambda=1.0):
    """从空集合开始选择成员/STOP，完整终点得分反馈整条上层轨迹。"""
    rng = rng or random.Random(0)
    rows, selected = [], []
    planned = forward_exchange(evaluator, candidates, observation.maximum) if method == "forward_exchange" else None
    while len(selected) < observation.maximum:
        check_budget()
        state = replace(observation, selected=tuple(selected))
        if method in {"upper_ppo", "linucb", "double_dqn"}:
            row = selector.sample(state, deterministic)
            index = row["index"]
        else:
            available = [i for i in range(len(candidates)) if i not in selected]
            if method == "random":
                index = rng.choice([*available, len(candidates)])
            elif method == "quality":
                index = max(available, key=lambda i: state.features[i][0]) if available else len(candidates)
            elif method == "forward_selection":
                # 基线只使用 A 段，R 段完全保留给最终评价。
                scores = [(evaluator.utility(candidates, [*selected, i], "validation")["utility"], i)
                          for i in available]
                current = evaluator.utility(candidates, selected, "validation")["utility"]
                best = max(scores, default=(current, len(candidates)))
                index = best[1] if best[0] > current + 1e-12 else len(candidates)
            elif method == "forward_exchange":
                index = planned[len(selected)] if len(selected) < len(planned) else len(candidates)
            else:
                raise ValueError(f"unknown selector method {method}")
            row = {"observation": state, "index": index}
        row["reward"] = 0.0
        rows.append(row)
        if index == len(candidates):
            break
        if index in selected:
            raise AssertionError("selector repeated a member")
        selected.append(index)
    score, contributions = evaluator.contributions(candidates, selected, segment)
    rows[-1]["reward"] = score["utility"]
    complete_returns(rows)
    if method == "upper_ppo":
        gae_targets(rows, trace_lambda)
    return rows, tuple(selected), score, contributions
