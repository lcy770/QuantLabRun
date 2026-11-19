"""冻结起点批采样、候选准入以及上层成员选择的真实执行路径。"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from math import fsum
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
        self.signal_hashes = set()

    def observe(self, evaluator, limit, remaining=1.0):
        """从质量与低访问起点轮换构造固定短名单，末动作由网络表示 NEW。"""
        entries = list(self.entries.values())
        assessed = {entry["tree"].id: evaluator.assess(entry["tree"], "start")[0] for entry in entries}
        qualities = {identity: candidate.quality for identity, candidate in assessed.items()}
        # 用本任务G段信号去重，同时为旧检查点中的起点库补齐信号身份。
        self.signal_hashes = {candidate.signal_hash for candidate in assessed.values()}
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
            if proposal["eligible"] and not proposal.get("duplicate", False):
                candidate = proposal["candidate"]
                hashes = getattr(self, "signal_hashes", set())
                if candidate.tree.id not in self.entries and candidate.signal_hash not in hashes:
                    self.entries[candidate.tree.id] = {"tree": candidate.tree, "attempts": 0,
                        "reward_sum": 0.0, "qualified": 0, "duplicates": 0, "known": False}
                    hashes.add(candidate.signal_hash)
                self.signal_hashes = hashes


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


@contextmanager
def proposal_rng(device, seed):
    """只重置本次采样使用的 CPU 和目标 GPU 随机流，退出后恢复。"""
    devices = []
    if device.type == "cuda":
        # 配置中的默认cuda没有编号，fork_rng需要解析后的实际GPU编号。
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        devices = [index]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def collect_proposal(generator, evaluator, observation, settings, seed, proposal_id, record=True):
    """用独立随机流生成一个主候选，并测量选起点、扩散与真实评价的完整耗时。"""
    device = getattr(generator, "device", None)
    if device is None:
        device = next(generator.parameters()).device
    with proposal_rng(device, seed):
        started = perf_counter()
        final, rows, source = generator.generate(observation, random.Random(seed), settings["diffusion_steps"],
                                                settings["perturbation_steps"], record)
        candidate, eligible = evaluator.assess(final, proposal_id) if final is not None else (None, False)
        finished = perf_counter()
        elapsed = finished - started
    for row in rows:
        if "edit_completed_at" in row:
            row["prefix_seconds"] = row.pop("edit_completed_at") - started
    source_id = "NEW"
    if rows and rows[0].get("index", len(observation.ids)) < len(observation.ids):
        source_id = observation.ids[rows[0]["index"]]
    elif not record and source is not None and source.id in observation.ids:
        source_id = source.id
    source_id = getattr(generator, "last_source_id", None) or source_id
    return {"candidate": candidate, "eligible": eligible, "seconds": elapsed, "rows": rows,
            "source_id": source_id, "seed": seed, "proposal_id": proposal_id, "finished_at": finished}


def expected_edit_value(editor, measure_quality, cost_rate, counters):
    """枚举下一步全部合法动作，用冻结的 Critic 估计继续编辑后的远期回报。"""
    @lru_cache(maxsize=128)
    def task_query(task):
        """保持主生成器使用的条件和单状态数值布局。"""
        return editor.task_embedding(task, editor.actor_encoder, editor.actor_task) if editor.use_task else None

    @lru_cache(maxsize=1024)
    @torch.no_grad()
    def expected(tree, step, task):
        """在当前公式精确加权下一步，STOP 使用真实终点质量。"""
        check_budget()
        started = perf_counter()
        actions = editor.edit_kernel.edits(tree)
        paths, positions, replacements_at, _ = editor._sampling_distributions(
            tree, step, task_query(task), actions)
        position_probs = positions.logits.double().softmax(dim=0).tolist()
        decision_seconds = perf_counter() - started
        if len(position_probs) != len(paths) + 1:
            raise ValueError("one-step continuation requires a lower STOP action")
        counters["policy_states"] += 1
        counters["legal_transitions"] += len(actions) + 1
        terminal_quality, terminal_seconds = measure_quality(editor.as_node(tree))
        counters["terminal_evaluations"] += 1
        successor_probs = defaultdict(float)
        cost_terms = [decision_seconds, position_probs[-1] * terminal_seconds]
        for index, position_prob in enumerate(position_probs[:-1]):
            started = perf_counter()
            choices, edits = replacements_at(index)
            edit_probs = edits.logits.double().softmax(dim=0).tolist()
            cost_terms.append(position_prob * (perf_counter() - started))
            for action, edit_prob in zip(choices, edit_probs, strict=True):
                started = perf_counter()
                after = editor.edit_kernel.apply(tree, action)
                edit_seconds = perf_counter() - started
                probability = position_prob * edit_prob
                successor_probs[after] += probability
                cost_terms.append(probability * edit_seconds)
        states = [(after, step, task) for after in successor_probs]
        # 按块编码大量后继。Critic 的算分是训练开销，不向生成动作收费。
        values = []
        for offset in range(0, len(states), 256):
            check_budget()
            values.extend(editor.state_values(states[offset:offset + 256]).double().tolist())
        return (position_probs[-1] * terminal_quality
                + fsum(prob * value for prob, value in zip(successor_probs.values(), values, strict=True))
                - cost_rate * fsum(cost_terms))

    return expected


def quality_change_rewards(proposal, editor, measure_quality, cost_rate):
    """起点记初始质量，编辑只记质量差，STOP一次扣除整条轨迹成本。"""
    trajectory = proposal["rows"]
    candidate = proposal["candidate"]
    final_quality = candidate.quality if candidate is not None else -1.0
    total_cost = cost_rate * proposal["seconds"]
    if len(trajectory) == 1:
        # 新建失败时没有可编辑的公式，也没有 STOP 动作。
        trajectory[0]["reward"] = final_quality - total_cost
        return [0.0]

    def quality(program):
        """复用真实终点评分，其他已访问公式按同一 F/G 任务评价。"""
        tree = editor.as_node(program)
        if candidate is not None and tree.id == candidate.tree.id:
            return final_quality
        return measure_quality(tree)[0]

    current = quality(trajectory[1]["tree"])
    trajectory[0].update(reward=current, initial_quality=current)
    offsets = [0.0]
    preparation = proposal["preparation_seconds"]
    previous = trajectory[0]["prefix_seconds"]
    for row in trajectory[1:]:
        # 网络仍预测终点质量减剩余成本，减去当前质量和已耗成本可换算为新回报。
        offsets.append(current + cost_rate * (preparation + previous))
        following = current if row["action"] is None else quality(row["after"])
        change = following - current
        row.update(quality_before=current, quality_after=following, quality_change=change,
                   reward=-total_cost if row["action"] is None else change)
        current, previous = following, row["prefix_seconds"]
    return offsets


def lower_targets(policy, proposals, config, contributions, evaluator, reference_seconds):
    """为两种集合筛选训练入口计算同一套下层 PPO 目标。"""
    eta = config["reward"]["eta"] if contributions is not None else 0.0
    contributions = contributions or {}
    settings = config["lower"]
    learned_stop = settings.get("learned_stop", False)
    quality_delta = settings.get("quality_delta_reward", False)
    rows, feedback = [], []
    choices = [(i, j, row) for i, proposal in enumerate(proposals) for j, row in enumerate(proposal["rows"])
               if row["kind"] == "edit" and row["action"] is not None] if learned_stop else []
    if learned_stop and (reference_seconds is None or not np.isfinite(reference_seconds) or reference_seconds <= 0):
        raise ValueError("continuation requires a positive frozen reference_seconds")
    cost_rate = config["reward"]["kappa"] / reference_seconds if learned_stop else 0.0
    summary = {"states": len(choices), "policy_states": 0, "legal_transitions": 0,
               "terminal_evaluations": 0, "wall_seconds": 0.0, "evaluator_calls": 0}
    expected, measure_quality = None, None
    if choices:
        if evaluator is None:
            raise ValueError("continuation requires a task evaluator")
        started = perf_counter()
        panels = {name: evaluator.panels[name] for name in ("fit", "search")}
        branch_evaluator = SelectionEvaluator(evaluator.grammar, panels, evaluator.config)

        @lru_cache(maxsize=1024)
        def measure_quality(tree):
            """实际编辑和假设停止共用同一份质量和首次评价用时。"""
            check_budget()
            begin = perf_counter()
            quality = branch_evaluator.assess(tree, "continuation")[0].quality
            return quality, perf_counter() - begin

        expected = expected_edit_value(policy.editor, measure_quality, cost_rate, summary)
    offsets = []
    for proposal in proposals:
        d = contributions.get(proposal["proposal_id"], {}).get("value", 0.0)
        reward = proposal["base_reward"] + eta * d
        trajectory = proposal["rows"]
        if not trajectory:
            raise ValueError("financial diffusion PPO requires recorded trajectories")
        value_offsets = None
        if quality_delta:
            value_offsets = quality_change_rewards(proposal, policy.editor, measure_quality, cost_rate)
        elif learned_stop:
            previous = 0.0
            for row in trajectory:
                elapsed = row.get("prefix_seconds", previous)
                row["reward"] = -cost_rate * (elapsed - previous)
                previous = elapsed
            # 旧版协议逐步计成本，STOP 后结算质量，Critic 只拟合基础回报。
            preparation = proposal["preparation_seconds"]
            quality = proposal["candidate"].quality if proposal["candidate"] is not None else -1.0
            trajectory[0]["reward"] -= cost_rate * preparation
            trajectory[-1]["reward"] += quality - cost_rate * (proposal["seconds"] - preparation - previous)
        else:
            for row in trajectory:
                row["reward"] = 0.0
            trajectory[-1]["reward"] = reward
        complete_returns(trajectory)
        gae_targets(trajectory, settings["gae_lambda"], value_offsets=value_offsets)
        offsets.append(value_offsets)
        if learned_stop:
            for row in trajectory:
                row["advantage"] += eta * d
        rows.extend(trajectory)
        feedback.append({"proposal_id": proposal["proposal_id"], "base_reward": proposal["base_reward"],
                         "contribution": d, "reward": reward})

    if expected is not None:
        for i, j, row in choices:
            value = expected(row["after"], row["step"], row["task"])
            if not np.isfinite(value):
                raise ValueError("continuation target must be finite")
            baseline = row["value"]
            if quality_delta:
                # 下一状态的质量与已耗成本是各合法后继的共同偏移，无需逐个评价假设编辑。
                value -= offsets[i][j + 1]
                baseline -= offsets[i][j]
            row["expected_future_value"] = value
            row["advantage"] = row["reward"] + value - baseline + eta * feedback[i]["contribution"]
        summary.update(wall_seconds=perf_counter() - started, evaluator_calls=branch_evaluator.calls)
    return rows, feedback, eta, summary


def initialize_worker(config):
    """在 CPU 采集进程初始化网络和有界的任务评价器缓存。"""
    global _config, _generator, _evaluators
    from .ast import Grammar
    _config = config
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    grammar = Grammar(**config["grammar"])
    _generator = GeneratorPolicy(grammar, config["width"], config["corpus"]["noise_steps"],
                                 learned_stop=config["lower"].get("learned_stop", False))
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
            proposal["preparation_seconds"] = preparation
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
                   rng=None, deterministic=False, trace_lambda=1.0, contribution_ids=None):
    """从空集合开始选择成员/STOP，完整终点得分反馈整条上层轨迹。"""
    rng = rng or random.Random(0)
    rows, selected = [], []
    # 排名对照仍取前十；选尽整池时排序次序不会改变等权组合。
    maximum = min(observation.maximum, 10) if method == "quality" and observation.independent_stop else observation.maximum
    planned = forward_exchange(evaluator, candidates, observation.maximum) if method == "forward_exchange" else None
    while len(selected) < maximum:
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
    score, contributions = evaluator.contributions(candidates, selected, segment, proposal_ids=contribution_ids)
    rows[-1]["reward"] = score["utility"]
    complete_returns(rows)
    if method == "upper_ppo":
        gae_targets(rows, trace_lambda)
    return rows, tuple(selected), score, contributions
