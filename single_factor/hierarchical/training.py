"""组织结构预训练、下层 PPO、上层 PPO 和交替更新的完整训练流程。"""
# 训练状态机与奖励分支。原方案和后续共同奖励实验按配置分别执行。
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
import torch

from .ast import Edit, Grammar, Node, replace
from .evaluation import Evaluator, load_task, normalize
from .io import append_json, digest, event, save_json, save_torch, check_budget
from .models import LowerPolicy, UpperPolicy
from .rpn import RPNKernel
from .ppo import complete_returns, continuation_returns, update, gae_targets, immediate_targets
from .search import Archive, episode, family


def evaluator_for(grammar, panels, config):
    # 信号缓存身份包含数据、实现、文法和时间区间，隔离不同实验的计算结果。
    """根据任务面板与实验参数创建 Evaluator，按数据、实现、文法和区间隔离磁盘缓存。"""
    store = None
    if config.get("signal_cache_dir"):
        key = digest([config["data_fingerprints"], config["implementation_hash"], grammar.config(),
                      str(panels["fit"].dates[0]), str(panels["fit"].dates[-1]),
                      str(panels["search"].dates[0]), str(panels["search"].dates[-1]), config["lower"]["score_weights"]])
        store = Path(config["signal_cache_dir"]) / key
    return Evaluator(grammar, panels["fit"], panels["search"],
                     min_coverage=config["search"]["min_coverage"],
                     ridge=config["search"]["ridge"], cache_size=config["signal_cache_size"],
                     score_weights=config["lower"]["score_weights"], signal_store=store)


def create_corpus(grammar, settings, output, edit_backend="ast"):
    # 为结构预训练生成合法公式、可逆扰动及逆编辑标签，复用语料前逐项检查兼容性。
    """生成或复用结构预训练语料，返回包含公式、集合划分和逆编辑标签的记录列表。

    新语料按结构家族划分训练、开发和留出集，并写入语料及清单文件。
    复用前检查配置与当前编辑内核的逆动作兼容性。
    """
    kernel = RPNKernel(grammar) if edit_backend == "rpn" else grammar
    corpus_identity = [grammar.config(), settings] + ([edit_backend] if edit_backend != "ast" else [])
    path = output / "corpus.jsonl"
    manifest = output / "corpus_manifest.json"
    if path.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        compatible_hashes = {digest(corpus_identity)}
        if edit_backend == "rpn":
            # 旧监督语料须通过下方原生合法动作检查，检查通过后才可复用。
            # 语料来源清单保存其原始身份。
            compatible_hashes.add(digest([grammar.config(), settings]))
        if metadata["configuration_hash"] not in compatible_hashes:
            raise ValueError("corpus configuration changed; use a new output directory")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            try:
                grammar.validate(Node.from_dict(record["ast"]))
                for pair in record["pairs"]:
                    tree, _, action = decode_pair(pair)
                    grammar.validate(tree)
                    program = kernel.from_node(tree)
                    native_action = kernel.native_edit(program, action)
                    changed = (kernel.apply(program, native_action) if edit_backend == "rpn"
                               else replace(tree, action.path, action.replacement))
                    grammar.validate(kernel.to_node(changed))
                    if native_action not in kernel.edits_at(program, action.path):
                        raise ValueError("inverse label is outside the current edit kernel")
            except (ValueError, KeyError, IndexError) as error:
                raise ValueError(f"stored corpus is incompatible with the current grammar at {record['id']}; rebuild in a new directory") from error
        event("corpus_ready", formulas=len(records), reused=True)
        return records
    rng = random.Random(settings["seed"])
    unique = {}
    event("corpus_started", target_formulas=settings["count"])
    while len(unique) < settings["count"]:
        tree = grammar.sample(rng, unique)
        unique[tree.id] = tree
        if len(unique) % 16 == 0 or len(unique) == settings["count"]:
            event("corpus_sampling", formulas=len(unique), target_formulas=settings["count"])
    # 按结构家族划分训练、开发和留出集，减少近似模板跨集合复用。
    families = sorted({repr(family(t)) for t in unique.values()})
    rng.shuffle(families)
    if len(families) < 5:
        raise ValueError("corpus needs at least five distinct AST structural families")
    count = max(1, len(families) // 5)
    development, heldout = set(families[:count]), set(families[count:2 * count])
    records = []
    for tree in unique.values():
        group = repr(family(tree))
        split = "check" if group in development else "holdout" if group in heldout else "train"
        _, trace = kernel.corrupt(kernel.from_node(tree), settings["noise_steps"], rng)
        trace = [(kernel.to_node(noisy), step, kernel.legacy_edit(action)) for noisy, step, action in trace]
        records.append({"ast": tree.to_dict(), "id": tree.id, "family": group, "split": split,
                        "pairs": [{"tree": noisy.to_dict(), "step": step,
                                   "path": action.path, "replacement": action.replacement.to_dict()}
                                  for noisy, step, action in trace]})
        if len(records) % 16 == 0 or len(records) == len(unique):
            event("corpus_inverse_pairs", formulas=len(records), target_formulas=len(unique))
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    temporary.replace(path)
    save_json(manifest, {"configuration_hash": digest(corpus_identity), "grammar": grammar.config(), "edit_backend": edit_backend,
                         "settings": settings, "families": len(families),
                         "counts": {s: sum(r["split"] == s for r in records) for s in ("train", "check", "holdout")}})
    return records


def decode_pair(pair):
    """将序列化监督样本还原为公式 Node、剩余步数和逆编辑 Edit。"""
    return Node.from_dict(pair["tree"]), pair["step"], Edit(tuple(pair["path"]), Node.from_dict(pair["replacement"]))


@torch.no_grad()
def syntax_loss(model, records):
    # 监督目标为真实逆编辑标签的负对数概率，衡量结构恢复能力。
    """计算监督逆编辑标签的平均负对数概率，返回结构验证损失，空样本集报错。"""
    losses = [-float(model.evaluate_action(*decode_pair(p))[0]) for r in records for p in r["pairs"]]
    if not losses:
        raise ValueError("empty structural validation split")
    return float(np.mean(losses))


def model_bundle(lower, upper, config, **extra):
    # 检查点同时记录编码表示、编辑后端、任务条件和位置条件，供按版本恢复。
    """打包两层参数、文法、网络结构和配置身份，返回可保存的检查点字典。

    按编码器、编辑后端及条件分支确定格式版本，并合并调用方附加的训练状态。
    """
    if upper.representation == "rpn" or upper.cost_context:
        version = 9
    elif lower.edit_backend == "rpn":
        version = 8
    elif lower.use_position:
        version = 7
    elif lower.representation == "rpn":
        version = 6
    elif lower.use_task:
        version = 5
    elif upper.use_archive:
        version = 4
    else:
        version = 3
    return {"format_version": version,
            "archive_context": upper.use_archive, "lower_task_context": lower.use_task,
            "upper_representation": upper.representation, "upper_cost_context": upper.cost_context,
            "lower_representation": lower.representation,
            "lower_edit_backend": lower.edit_backend,
            "lower_position_context": lower.use_position,
            "grammar": lower.grammar.config(), "width": lower.width,
            "noise_steps": lower.noise_steps, "lower": lower.state_dict(), "upper": upper.state_dict(),
            "configuration_hash": config["fingerprint"], **extra}


def restore_models(bundle, device="cpu"):
    # 按检查点自己的元数据构建模型，历史版本缺少的新特征按兼容规则恢复。
    """按检查点元数据在指定设备重建两层模型并载入参数，返回下层与上层实例。

    检查新格式必需信息，兼容旧版本新增输入列和条件参数的迁移规则。
    """
    if bundle.get("architecture") == "candidate_selection":
        from .selection_training import restore_selection_models
        return restore_selection_models(bundle, device)
    if bundle.get("format_version", 1) >= 9 and not {"upper_representation", "upper_cost_context"} <= bundle.keys():
        raise ValueError("checkpoint is missing upper representation/context metadata")
    if bundle.get("format_version", 1) >= 8 and "lower_edit_backend" not in bundle:
        raise ValueError("checkpoint is missing edit backend metadata")
    grammar = Grammar(**bundle["grammar"])
    lower = LowerPolicy(grammar, bundle["width"], bundle["noise_steps"], use_task=bundle.get("lower_task_context", False),
                        representation=bundle.get("lower_representation", "ast"),
                        use_position=bundle.get("lower_position_context", False),
                        edit_backend=bundle.get("lower_edit_backend", "ast")).to(device)
    upper = UpperPolicy(grammar, bundle["width"], use_archive=bundle.get("archive_context", False),
                        representation=bundle.get("upper_representation", "ast"),
                        cost_context=bundle.get("upper_cost_context", False)).to(device)
    if lower.use_task:
        required = {key for key in lower.state_dict() if key.startswith(("actor_task.", "critic_task."))}
        if not required.issubset(bundle["lower"]):
            raise ValueError("task-conditioned checkpoint is missing context parameters")
    if lower.use_position and "actor_position.weight" not in bundle["lower"]:
        raise ValueError("position-conditioned checkpoint is missing position parameters")
    lower.load_state_dict(bundle["lower"])
    upper_state = dict(bundle["upper"])
    if bundle.get("format_version", 1) == 1:
        weight = upper_state["value_head.0.weight"]
        if weight.shape[1] == 2 * bundle["width"] + 1:
            # 通过新增零权重列，兼容旧模型的策略输出与价值预测。
            upper_state["value_head.0.weight"] = torch.cat((weight, torch.zeros_like(weight[:, :1])), dim=1)
    weight = upper_state["score.0.weight"]
    if weight.shape[1] == 3 * bundle["width"] + 1:
        upper_state["score.0.weight"] = torch.cat((weight, torch.zeros_like(weight[:, :1])), dim=1)
    for tower in ("actor", "critic"):
        upper_state.setdefault(f"{tower}.new_summary.weight", torch.zeros_like(upper.state_dict()[f"{tower}.new_summary.weight"]))
    upper.load_state_dict(upper_state)
    return lower, upper


def validate_forward_training(tasks, panels):
    """历史训练反馈使用的标签须在任何模型选择区间开始前成熟。"""
    # 用实际标签结束日校验，历史后续训练标签须早于开发或测试的搜索起点成熟。
    training = [task for task in tasks if task["role"] == "train"]
    evaluation = [task for task in tasks if task["role"] in {"development", "test"}]
    if not evaluation:
        raise ValueError("forward training requires a later development or test task")
    boundary = min(panels[task["name"]]["search"].dates.min() for task in evaluation)
    for task in training:
        final = panels[task["name"]].get("future")
        if final is None or final.ends.max() >= boundary:
            raise ValueError("training forward labels reach a development/test search period")


def forward_training_rewards(rows, report, evaluator, final, settings):
    """对采集完成的轨迹重算训练目标，行为阶段的可见搜索历史按原记录保留。"""
    # 历史后续上层反馈实验在采集结束后重算目标，行为阶段使用原可见状态。
    from .experiments import future_check

    if not rows:
        raise ValueError("cannot relabel an empty rollout")
    # 搜索奖励由一次合格新发现和非负的工作组合改善组成。
    # 重评分沿用当时的接纳映射，逐步回放原搜索决策。
    accepted = [i for i, row in enumerate(rows) if row["reward"] >= 1.0]
    if len(accepted) != len(report["discoveries"]):
        raise ValueError("accepted actions do not match chronological discoveries")
    trees = {tree.id: tree for row in rows for tree in row["observation"].pool_trees}
    trees.update((tree.id, tree) for tree in map(Node.from_dict, report["pool_asts"]))
    for tree in trees.values():
        evaluator.get(tree)  # 因子方向由当时可见的拟合期信息确定并锁定。
    result = future_check(report, evaluator, final)
    qualified = [int(f["reliable"] and f["coverage"] >= settings["min_coverage"] and
                     f["ic"] >= settings["quality_threshold"] and
                     f["rank_ic"] >= settings.get("min_search_rank_ic", .01))
                 for f in result["single_factors"]]
    pool_scores = {}

    def locked_pool_ic(pool_trees, weights):
        """在历史后续区间用给定成员和已锁定权重计算组合 IC，并缓存相同组合评分。"""
        key = (tuple(tree.id for tree in pool_trees), tuple(weights))
        if key not in pool_scores:
            locked = {"discoveries": [], "pool_asts": [tree.to_dict() for tree in pool_trees],
                      "pool_weights": weights}
            pool_scores[key] = future_check(locked, evaluator, final)["pool_ic"]
        return pool_scores[key]

    observation = rows[0]["observation"]
    initial_ic = before = locked_pool_ic(observation.pool_trees, observation.pool_weights)
    discoveries = dict(zip(accepted, qualified))
    for i, row in enumerate(rows):
        if i + 1 < len(rows):
            following = rows[i + 1]["observation"]
            after = locked_pool_ic(following.pool_trees, following.pool_weights)
        else:
            after = result["pool_ic"]
        row.update(search_reward=row["reward"], forward_valid=discoveries.get(i, 0),
                   forward_pool_before=before, forward_pool_after=after,
                   reward=discoveries.get(i, 0) + settings["combination_weight"] * (after - before))
        before = after
    total = complete_returns(rows)
    expected = sum(qualified) + settings["combination_weight"] * (result["pool_ic"] - initial_ic)
    if abs(total - expected) > 1e-8:
        raise AssertionError("forward training reward telescoping invariant failed")
    report.update(training_reward="forward", training_return=total, training_forward=result,
                  training_forward_valid=sum(qualified), training_initial_forward_ic=initial_ic)


def completion_reward(eligible, qualified, delta, settings, objective):
    # 这是后续共同完成反馈实验的终点评分，原方案下层评分在 Evaluator.lower_score。
    # eligible 表示搜索期合格且非重复，qualified 表示成熟历史后续期单因子达标。
    # delta 是相对动作前组合的锁定后续 IC 增量。
    """计算共同完成反馈实验的候选终点评分，返回加权目标或严格成功目标的奖励。

    eligible 为搜索合格且非重复，qualified 为历史后续单因子达标，delta 为有向组合 IC 增量。
    未通过前置检查得零，加权目标叠加达标分和增量，严格目标奖励联合成功并对组合受损扣分。
    """
    if objective not in {"forward_weighted", "forward_joint"}:
        raise ValueError("unknown completion objective")
    if not eligible:
        # 前置筛选未通过时直接置零，当前共同奖励变体的稀疏性主要来自此门槛。
        return 0.0
    if objective == "forward_weighted":
        # A 组采用后续单因子达标分与组合增量的加权形式。
        return float(qualified) + settings["combination_weight"] * delta
    # B/C 组严格成功得 1，组合受损连续扣分并截断到 1，二者奖励公式相同。
    return float(qualified and delta > 1e-12) - min(max(-delta / settings["quality_threshold"], 0.0), 1.0)


def completion_training_rewards(rows, trajectories, report, evaluator, initial, final, settings, objective):
    """采集完成后，相对每个动作发生前的实际组合重算两层目标。

    回放档案按可见搜索决策提交。历史后续标签用于计算训练目标，
    观测、接纳和主轨迹采样沿用行为阶段的搜索信息。
    """
    # 共同奖励实验显式调用此函数重标上下层目标，评价阶段也用它统计严格成功。
    archive = Archive(evaluator, initial, settings)
    signals, pool_scores = {}, {}

    def signal(tree):
        """计算并缓存公式在历史后续区间的标准化信号和单因子指标。"""
        if tree.id not in signals:
            signals[tree.id] = evaluator.segment(tree, final)
        return signals[tree.id]

    def pool_ic(pool):
        # 组合方向与权重沿用拟合结果，在历史后续区间评分时保持锁定。
        """沿用组合成员的拟合期方向和权重，计算并缓存历史后续区间的有向组合 IC。"""
        key = (tuple(tree.id for tree in pool.stats.trees), tuple(pool.score.weights))
        if key not in pool_scores:
            prediction = np.zeros_like(final.y)
            for tree, weight in zip(pool.stats.trees, pool.score.weights):
                z, _ = signal(tree)
                prediction += weight * evaluator.metrics[tree.id]["direction"] * z
            z, _, _ = normalize(prediction, final.mask)
            daily = (z * final.y).sum(1) / final.count
            pool_scores[key] = float(daily[final.active].mean())
        return pool_scores[key]

    rewards = [None] * len(trajectories)
    strict = qualified_count = 0
    for upper_row in rows:
        upper_row["search_reward"] = upper_row["reward"]
        if not upper_row.get("credited", False):
            upper_row["reward"] = 0.0
            continue
        observed = upper_row["observation"]
        if observed.pool_trees != archive.pool.stats.trees:
            raise AssertionError("completion replay changed the visible pre-action pool")
        np.testing.assert_allclose(observed.pool_weights, archive.pool.score.weights, atol=1e-12, rtol=0)
        before = pool_ic(archive.pool)
        scores = {}

        def score(tree):
            """相对当前主动作前档案试算终点，返回奖励、后续达标标记和组合 IC 增量。

            先检查搜索合格与非重复，再计算后续质量和组合影响，缓存仅用于本次动作的背景。
            """
            if tree.id not in scores:
                proposal = archive.assess(tree)
                qualified, delta = False, 0.0
                if proposal["new"]:
                    # 候选先过可见搜索期筛选，之后才计算后续单因子质量与组合影响。
                    _, metrics = signal(tree)
                    direction = proposal["metrics"]["direction"]
                    qualified = bool(metrics["reliable"] and metrics["coverage"] >= settings["min_coverage"]
                                     and direction * metrics["ic"] >= settings["quality_threshold"]
                                     and direction * metrics["rank_ic"] >= settings.get("min_search_rank_ic", .01))
                    delta = pool_ic(proposal["pool"]) - before
                scores[tree.id] = (completion_reward(proposal["new"], qualified, delta, settings, objective),
                                   qualified, delta)
            return scores[tree.id]

        primary = Node.from_dict(upper_row["candidate"])
        reward, qualified, delta = score(primary)
        upper_row.update(reward=reward, forward_valid=int(qualified), forward_pool_before=before,
                         forward_pool_after=before + delta, forward_joint_success=int(qualified and delta > 1e-12))
        qualified_count += int(qualified)
        strict += upper_row["forward_joint_success"]
        index = upper_row.get("lower_trajectory_index")
        if index is not None:
            if rewards[index] is not None:
                raise AssertionError("lower trajectory was credited twice")
            rewards[index] = reward
            for row in trajectories[index]:
                row["search_completion_scores"] = list(row["continuation_scores"])
                # 主路径与各续采样终点按同一动作前背景重评分，再重新取 Q 均值。
                outcomes = [score(Node.from_dict(endpoint))[0] for endpoint in row["continuation_endpoints"]]
                row.update(continuation_scores=outcomes, **{"return": float(np.mean(outcomes))})
        proposal = archive.assess(primary)
        if abs(proposal["reward"] - upper_row["search_reward"]) > 1e-10:
            raise AssertionError("completion replay changed search admission/reward")
        source = observed.ids[upper_row["index"]] if upper_row["index"] < len(observed.ids) else "NEW"
        # 回放只提交原搜索决策，保证历史后续标签没有改写行为阶段的档案。
        archive.commit(proposal, source, upper_row.get("expansion_seconds", 0.0))
    if any(reward is None for reward in rewards):
        raise AssertionError("unmatched lower trajectory in completion replay")
    report.update(training_reward=objective, training_return=complete_returns(rows),
                  training_forward_valid=qualified_count, training_joint_success=strict)
    return rewards


class Campaign:
    # 统一管理模型、优化器、随机状态、阶段计数及可恢复检查点。
    def __init__(self, config, output, resume=False, progress_callback=None):
        """初始化或恢复一次训练实验，建立模型、优化器、任务数据、语料和公共起点。

        恢复时核验配置指纹并还原随机流，历史后续奖励和起点筛选均检查标签时间边界。
        保存解析配置，并按配置建立并行采集器或载入共享下层初始化。
        """
        self.config, self.output = config, Path(output)
        self.progress_callback = progress_callback
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = perf_counter()
        self.rng = random.Random(config["seed"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.set_num_threads(config["torch_threads"])
        torch.use_deterministic_algorithms(True)
        self.grammar = Grammar(**config["grammar"])
        self.lower = LowerPolicy(self.grammar, config["width"], config["corpus"]["noise_steps"],
                                 use_task=config["lower"].get("task_context", False),
                                 representation=config["lower"].get("representation", "ast"),
                                 use_position=config["lower"].get("position_context", False),
                                 edit_backend=config["lower"].get("edit_backend", "ast")).to(config["device"])
        self.upper = UpperPolicy(self.grammar, config["width"], use_archive=config["upper"].get("archive_context", False),
                                 representation=config["upper"].get("representation", "ast"),
                                 cost_context=config["upper"].get("cost_context", False)).to(config["device"])
        self.lower_optimizer = torch.optim.Adam(self.lower.parameters(), lr=config["lower"]["learning_rate"])
        self.upper_optimizer = torch.optim.Adam(self.upper.parameters(), lr=config["upper"]["learning_rate"])
        self.state = {"phase": "corpus", "pretrain_epoch": 0, "lower_iteration": 0,
                      "upper_iteration": 0, "joint_iteration": 0, "best_nll": math.inf,
                      "stale": 0, "wall_seconds": 0.0, "completed_evaluations": []}
        checkpoint = self.output / "training_state.pt"
        if checkpoint.exists():
            # 恢复训练要求数据、配置和源码指纹一致，并恢复各随机数生成器状态。
            if not resume:
                raise FileExistsError("existing training state; pass --resume or use a new output")
            saved = torch.load(checkpoint, map_location=config["device"], weights_only=False)
            if saved["configuration_hash"] != config["fingerprint"]:
                raise ValueError("resume configuration/data/implementation fingerprint differs")
            self.lower.load_state_dict(saved["lower"])
            self.upper.load_state_dict(saved["upper"])
            self.lower_optimizer.load_state_dict(saved["lower_optimizer"])
            self.upper_optimizer.load_state_dict(saved["upper_optimizer"])
            self.state = saved["campaign_state"]
            self.rng.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if saved.get("cuda_rng") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            np.random.set_state(saved["numpy_rng"])
        elif resume:
            raise FileNotFoundError(checkpoint)
        self.elapsed_before = self.state["wall_seconds"]
        self.panels = {}
        for task in config["tasks"]:
            forward = task["role"] == "development" and config.get("upper_validation", {}).get("forward", False)
            forward = forward or (task["role"] == "train" and config["upper"].get("training_reward", "search").startswith("forward"))
            self.panels[task["name"]] = load_task(task, config["target_horizon"], config["lookback"], config["future_days"],
                                                   with_future=forward)
        if config["upper"].get("training_reward", "search").startswith("forward"):
            validate_forward_training(config["tasks"], self.panels)
        if config.get("corpus_source") and not (self.output / "corpus.jsonl").exists():
            for name in ("corpus.jsonl", "corpus_manifest.json"):
                shutil.copy2(Path(config["corpus_source"]) / name, self.output / name)
        self.records = create_corpus(self.grammar, config["corpus"], self.output, self.lower.edit_backend)
        train = [Node.from_dict(r["ast"]) for r in self.records if r["split"] == "train"]
        if config.get("initial_library"):
            bank = [json.loads(line) for line in Path(config["initial_library"]).read_text(encoding="utf-8").splitlines()]
            if not bank or any(not record.get("validated") for record in bank):
                raise ValueError("initial library contains unvalidated formulas")
            self.initial = [self.grammar.validate(Node.from_dict(record["ast"])) for record in bank]
            if len({tree.id for tree in self.initial}) != len(self.initial):
                raise ValueError("initial library contains duplicate structures")
            library_manifest = json.loads((Path(config["initial_library"]).parent / "library_manifest.json").read_text(encoding="utf-8"))
            cutoff = np.datetime64(library_manifest["latest_screening_label"])
            if any(cutoff >= self.panels[t["name"]]["search"].dates[0] for t in config["tasks"] if t["role"] == "train"):
                raise ValueError("library screening labels reach an upper training feedback period")
        else:
            self.initial = train[:config["search"]["initial_seeds"]]
        if len(self.initial) < config["search"]["pool_capacity"]:
            raise ValueError("insufficient common initial training formulas")
        save_json(self.output / "resolved_config.json", config)
        if config.get("initializer_checkpoint") and not checkpoint.exists():
            # 从共享结构初始化开始金融训练时，阶段直接进入 lower。
            initializer = torch.load(config["initializer_checkpoint"], map_location=config["device"], weights_only=False)
            if initializer["grammar"] != self.grammar.config():
                raise ValueError("pretrained initializer grammar differs")
            if initializer.get("lower_representation", "ast") != self.lower.representation:
                raise ValueError("pretrained initializer representation differs; use a matching initializer")
            self.lower.load_state_dict(initializer["lower"])
            self.state["phase"] = "lower"
            self.state["pretrained_initializer"] = config["initializer_checkpoint"]
        self.collector = None
        if config.get("rollout_workers", 1) > 1:
            from .parallel import Collector
            self.collector = Collector(config)

    def elapsed(self):
        """返回历史已记录耗时与本次进程运行耗时之和，单位为秒。"""
        return self.elapsed_before + perf_counter() - self.started

    def progress(self, **extra):
        """将当前训练状态和额外指标写入进度文件与事件日志，并调用可选进度回调。"""
        progress = {**{k: v for k, v in self.state.items() if k != "pretrain_optimizer"}, **extra, "wall_seconds": self.elapsed()}
        save_json(self.output / "progress.json", progress)
        event("progress", **progress)
        if self.progress_callback:
            self.progress_callback({"stage": progress["phase"], **progress})

    def commit(self):
        # 原子保存训练状态和随机流，用提交成功的批次作为恢复边界。
        """原子保存两层模型、优化器、阶段状态和随机流，随后更新进度。

        递增提交编号并记录累计耗时，可按配置另存带阶段编号的检查点副本。
        """
        self.state["commit_number"] = self.state.get("commit_number", 0) + 1
        self.state["wall_seconds"] = self.elapsed()
        save_torch(self.output / "training_state.pt", model_bundle(self.lower, self.upper, self.config,
                   lower_optimizer=self.lower_optimizer.state_dict(), upper_optimizer=self.upper_optimizer.state_dict(),
                   campaign_state=self.state.copy(), python_rng=self.rng.getstate(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], numpy_rng=np.random.get_state()))
        if self.config.get("archive_checkpoints"):
            destination = self.output / "checkpoints" / f"{self.state['commit_number']:05d}_{self.state['phase']}.pt"
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(self.output / "training_state.pt", destination)
        self.progress()

    def start_stage(self, name):
        """登记阶段首次开始时的累计耗时，恢复同一未结束阶段时保留原起点。"""
        self.state.setdefault("stage_started", {}).setdefault(name, self.elapsed())

    def finish_stage(self, name):
        """按阶段开始记录结算耗时并移除开始标记，将秒数写入阶段统计。"""
        self.state.setdefault("stage_wall_seconds", {})[name] = self.elapsed() - self.state["stage_started"].pop(name)

    def syntax_pretrain(self):
        # 先学习扰动到原公式的逆编辑，开发结构损失用于选点，金融效果另行评价。
        """用正向扰动的逆动作标签预训练下层，按开发结构损失早停并选择检查点。

        随后记录留出结构恢复诊断，保存曲线与训练状态，并将阶段推进到下层金融训练。
        """
        self.start_stage("pretrain")
        settings = self.config["pretrain"]
        train = [p for r in self.records if r["split"] == "train" for p in r["pairs"]]
        check = [r for r in self.records if r["split"] == "check"]
        if not self.state["pretrain_epoch"]:
            self.state["best_nll"] = syntax_loss(self.lower, check)
            self.state["initial_nll"] = self.state["best_nll"]
            save_torch(self.output / "syntax_best.pt", model_bundle(self.lower, self.upper, self.config))
        optimizer = torch.optim.Adam(self.lower.parameters(), lr=settings["learning_rate"])
        if "pretrain_optimizer" in self.state:
            optimizer.load_state_dict(self.state["pretrain_optimizer"])
        while self.state["pretrain_epoch"] < settings["epochs"] and self.state["stale"] < settings["patience"]:
            started = perf_counter()
            ordered = self.rng.sample(train, len(train))
            losses = []
            for cursor in range(0, len(train), settings["batch_size"]):
                loss = torch.stack([-self.lower.evaluate_action(*decode_pair(p))[0]
                                    for p in ordered[cursor:cursor + settings["batch_size"]]]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite supervised reverse-edit loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lower.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if cursor % max(settings["batch_size"] * 10, 1) == 0:
                    self.progress(training_pair_cursor=cursor, training_pairs=len(train))
            nll = syntax_loss(self.lower, check)
            self.state["pretrain_epoch"] += 1
            if nll < self.state["best_nll"] - settings["min_delta"]:
                self.state["best_nll"], self.state["stale"] = nll, 0
                save_torch(self.output / "syntax_best.pt", model_bundle(self.lower, self.upper, self.config))
            else:
                self.state["stale"] += 1
            row = {"epoch": self.state["pretrain_epoch"], "train_nll": np.mean(losses), "check_nll": nll,
                   "wall_seconds": self.elapsed(), "epoch_seconds": perf_counter() - started}
            append_json(self.output / "pretrain_curve.jsonl", row)
            self.state["pretrain_optimizer"] = optimizer.state_dict()
            self.commit()
        best = torch.load(self.output / "syntax_best.pt", map_location=self.config["device"], weights_only=False)
        self.lower.load_state_dict(best["lower"])
        heldout = [r for r in self.records if r["split"] == "holdout"]
        recovery, finals = [], []
        for record in heldout[:self.config["pretrain"]["holdout_samples"]]:
            noisy = Node.from_dict(record["pairs"][-1]["tree"])
            final, _ = self.lower.sample(noisy)
            self.grammar.validate(final)
            recovery.append(final.id == record["id"])
            finals.append(final.id)
        save_json(self.output / "structure_validation.json", {
            "initial_check_nll": self.state["initial_nll"], "best_check_nll": self.state["best_nll"],
            "heldout_nll": syntax_loss(self.lower, heldout), "multistep_exact_recovery": np.mean(recovery),
            "unique_fraction": len(set(finals)) / max(len(finals), 1), "sample_count": len(finals),
            "legal_fraction": 1.0 if finals else None, "financial_effectiveness_test": False,
            "holdout_used_for_checkpoint_selection": False})
        self.state.pop("pretrain_optimizer", None)
        self.finish_stage("pretrain")
        self.state["phase"] = "lower"
        self.commit()

    def lower_batch(self, trajectories=None, rewards=None, evaluator=None, task_name=None):
        # 独立训练时自行采集，联合训练时接收已经采集并评分的下层轨迹。
        """采集或接收一批下层轨迹，计算动作条件回报并执行一次 PPO 批次更新。

        默认使用拟合期单因子奖励，搜索训练和共同完成反馈实验按显式配置进入对应评分分支。
        已有续采样结果直接复用，按配置记录轨迹，返回训练与采集耗时等诊断。
        """
        settings = self.config["lower"]
        collection_start = perf_counter()
        if trajectories is None and settings.get("search_training"):
            # 搜索任务训练入口固定随机上层，以相同档案规则收集下层主候选。
            task = self.rng.choice([t for t in self.config["tasks"] if t["role"] == "train"])
            task_name = task["name"]
            evaluator = evaluator_for(self.grammar, self.panels[task_name], self.config).for_episode(self.initial)
            search = {**self.config["search"], "max_attempts": settings["batch_formulas"], "budget_mode": "attempts"}
            upper_rows, trajectories, rewards, report = episode(self.lower, self.upper, evaluator, self.initial,
                search, self.rng, scheduler="random", lower_training=settings,
                output=self.output / "training_attempts.jsonl", tag=f"lower_{self.state['lower_iteration']}")
            objective = self.config["upper"].get("training_reward", "search")
            if objective in {"forward_weighted", "forward_joint"}:
                # 该配置是共同奖励实验开关，会同时替换主终点和辅助分支的目标。
                rewards = completion_training_rewards(upper_rows, trajectories, report, evaluator, self.initial,
                    self.panels[task_name]["future"], search, objective)
            append_json(self.output / "lower_training_episodes.jsonl", {**report, "task": task_name})
            if not trajectories:
                raise RuntimeError("no complete lower trajectories in search training batch")
        if trajectories is None and self.collector:
            task = self.rng.choice([t for t in self.config["tasks"] if t["role"] == "train"])
            starts = [self.rng.choice(self.initial) for _ in range(settings["batch_formulas"])]
            collected = self.collector.lower(self.lower, starts, task, settings, self.rng)
            trajectories = [trajectory for result in collected for trajectory in result["trajectories"]]
            rewards = [reward for result in collected for reward in result["rewards"]]
            branches = sum(result["branches"] for result in collected)
            calls = sum(result["calls"] for result in collected)
            requests = sum(result["requests"] for result in collected)
            task_name = task["name"]
        else:
            branches = None
        if evaluator is None and branches is None:
            task = self.rng.choice([t for t in self.config["tasks"] if t["role"] == "train"])
            evaluator = evaluator_for(self.grammar, self.panels[task["name"]], self.config)
            task_name = task["name"]
        trajectories, rewards = ([], []) if trajectories is None else (trajectories, rewards)
        if not trajectories:
            # 原方案的独立下层训练从起点扰动，终点使用拟合期单因子奖励。
            for _ in range(settings["batch_formulas"]):
                start = self.rng.choice(self.initial)
                horizon = settings.get("diffusion_steps", self.lower.noise_steps)
                noisy, _ = self.lower.corrupt(start, settings.get("perturbation_steps", horizon), self.rng)
                final, rows = self.lower.sample(noisy, remaining=horizon)
                trajectories.append(rows)
                rewards.append(evaluator.lower(final))
        if branches is None:
            # 搜索采集可能已保存续采样结果，直接复用以避免重复生成和评价。
            if all("continuation_scores" in row for rows in trajectories for row in rows):
                branches = sum(len(row["continuation_scores"]) - 1 for rows in trajectories for row in rows)
            else:
                branches = continuation_returns(self.lower, trajectories, rewards, evaluator.lower,
                                                settings["continuation_samples"], settings["continuation_states"], self.rng)
            calls, requests = evaluator.calls, evaluator.requests
            task_name = task_name or "serial_or_joint"
        collection_seconds = perf_counter() - collection_start
        if self.config.get("record_lower_trajectories"):
            for trajectory, reward in zip(trajectories, rewards):
                append_json(self.output / "lower_trajectories.jsonl", {"iteration": self.state["lower_iteration"] + 1,
                    "joint_iteration": self.state["joint_iteration"], "task": task_name, "terminal_reward": reward,
                    "terminal_ast": self.lower.as_node(trajectory[-1]["after"]).to_dict(), "actions": [{
                        "tree": self.lower.as_node(r["tree"]).to_dict(), "remaining": r["step"], "path": r["action"].path,
                        "replacement": self.lower.as_node(r["action"].replacement).to_dict(), "old_logp": r["logp"], "value": r["value"],
                        "return": r["return"], "continuation_scores": r["continuation_scores"],
                        "continuation_endpoints": r["continuation_endpoints"]} for r in trajectory]})
        rows = [row for trajectory in trajectories for row in trajectory]
        if self.config.get("record_rollouts"):
            replay = [{k: v for k, v in row.items() if k != "legal_actions"} for row in rows]
            save_torch(self.output / "rollouts" / f"lower_{self.state['lower_iteration']}_{self.state['joint_iteration']}.pt",
                {"models": model_bundle(self.lower, self.upper, self.config), "rows": replay,
                 "optimizer": self.lower_optimizer.state_dict(), "shuffle_rng": self.rng.getstate(), "task": task_name})
        update_start = perf_counter()
        metrics = update(self.lower, self.lower_optimizer, rows, rng=self.rng, **settings["ppo"])
        return {**metrics, "mean_terminal_reward": np.mean(rewards), "continuation_branches": branches,
                "evaluation_calls": calls, "evaluation_requests": requests,
                "collection_seconds": collection_seconds, "update_seconds": perf_counter() - update_start}

    def train_lower(self):
        # 完成独立下层更新后导出 lower_base，供冻结下层的调度对照使用。
        """执行剩余独立下层训练批次，逐批保存曲线和状态，导出 lower_base 后进入上层阶段。"""
        self.start_stage("lower")
        while self.state["lower_iteration"] < self.config["lower"]["iterations"]:
            check_budget()
            limit = self.config["lower"].get("time_budget_seconds")
            if limit is not None and self.elapsed() - self.state["stage_started"]["lower"] >= limit:
                break
            metrics = self.lower_batch()
            self.state["lower_iteration"] += 1
            append_json(self.output / "lower_curve.jsonl", {"iteration": self.state["lower_iteration"], **metrics, "wall_seconds": self.elapsed()})
            self.commit()
        self.finish_stage("lower")
        save_torch(self.output / "lower_base.pt", model_bundle(self.lower, self.upper, self.config))
        self.state["phase"] = "upper"
        self.commit()

    def train_upper(self, joint=False):
        # joint=False 时固定下层只更新上层，joint=True 时按两层顺序交替更新。
        """训练上层调度，joint 决定固定下层还是交替更新两层。

        联合批次先采集并更新下层，再用更新后的下层重新采集上层轨迹并更新上层。
        按配置重评分、计算优势、验证选点或回退，保存工作状态和阶段最终模型。
        """
        stage = "joint" if joint else "upper"
        self.start_stage(stage)
        key = "joint_iteration" if joint else "upper_iteration"
        target = self.config["upper"]["joint_iterations" if joint else "iterations"]
        tasks = [t for t in self.config["tasks"] if t["role"] == "train"]
        if self.config.get("upper_validation") and f"best_{stage}_score" not in self.state:
            self.validate_upper(stage, 0)
        while self.state[key] < target:
            check_budget()
            limit = self.config["upper"].get("joint_time_budget_seconds" if joint else "time_budget_seconds")
            if limit is not None and self.elapsed() - self.state["stage_started"][stage] >= limit:
                break
            guarded = joint and self.config["upper"].get("joint_guard", False)
            reference_score = self.state.get("best_joint_score")
            if joint:
                # 先更新下层，再为上层采集新回合，使轨迹对应实际使用的下层版本。
                task = self.rng.choice(tasks)
                evaluator = evaluator_for(self.grammar, self.panels[task["name"]], self.config).for_episode(self.initial)
                objective = self.config["upper"].get("training_reward", "search")
                search = self.config["search"]
                if objective in {"forward_weighted", "forward_joint"}:
                    search = {**search, "max_attempts": self.config["lower"]["batch_formulas"], "budget_mode": "attempts"}
                lower_rows, trajectories, rewards, report = episode(self.lower, self.upper, evaluator, self.initial,
                    search, self.rng, output=self.output / "training_attempts.jsonl", tag=f"joint_lower_{self.state[key]}",
                    lower_training=self.config["lower"] if self.config["lower"].get("search_training") else None)
                if objective in {"forward_weighted", "forward_joint"}:
                    rewards = completion_training_rewards(lower_rows, trajectories, report, evaluator, self.initial,
                        self.panels[task["name"]]["future"], search, objective)
                append_json(self.output / "joint_lower_episodes.jsonl", report)
                for group in self.lower_optimizer.param_groups:
                    group["lr"] = self.config["lower"]["learning_rate"] * (.25 if guarded else 1.)
                lower_metrics = self.lower_batch(trajectories, rewards, evaluator, task_name=task["name"])
                for group in self.lower_optimizer.param_groups:
                    group["lr"] = self.config["lower"]["learning_rate"]
                append_json(self.output / "lower_curve.jsonl", {"joint_iteration": self.state[key], **lower_metrics})
            rollouts, reports = [], []
            # 此处采集发生在本批下层更新之后，随后才计算上层优势并执行 PPO。
            jobs = []
            for e in range(self.config["upper"]["episodes_per_batch"]):
                task = self.rng.choice(tasks)
                tag = f'{key}_{self.state[key]}_episode_{e}'
                if self.collector:
                    from .parallel import cpu_state
                    jobs.append({"lower": cpu_state(self.lower), "upper": cpu_state(self.upper), "task": task,
                        "seed": self.rng.randrange(2**31), "initial": self.initial, "search": self.config["search"],
                        "trace": self.output / "training_traces" / f"{tag}.jsonl", "tag": tag})
                    continue
                evaluator = evaluator_for(self.grammar, self.panels[task["name"]], self.config).for_episode(self.initial)
                new_rows, _, _, report = episode(self.lower, self.upper, evaluator, self.initial, self.config["search"],
                                                self.rng, output=self.output / "training_attempts.jsonl", tag=tag, record_lower=False)
                report["task"] = task["name"]
                rollouts.append(new_rows)
                reports.append(report)
                self.progress(current_episode=e + 1)
            if jobs:
                for new_rows, _, _, report in self.collector.episodes(jobs):
                    rollouts.append(new_rows)
                    reports.append(report)
            for rollout, report in zip(rollouts, reports):
                if self.config["upper"].get("training_reward") == "forward":
                    panels = self.panels[report["task"]]
                    evaluator = evaluator_for(self.grammar, panels, self.config).for_episode(self.initial)
                    forward_training_rewards(rollout, report, evaluator, panels["future"], self.config["search"])
                elif self.config["upper"].get("training_reward") in {"forward_weighted", "forward_joint"}:
                    panels = self.panels[report["task"]]
                    evaluator = evaluator_for(self.grammar, panels, self.config).for_episode(self.initial)
                    completion_training_rewards(rollout, [], report, evaluator, self.initial, panels["future"],
                                                self.config["search"], self.config["upper"]["training_reward"])
                append_json(self.output / "training_episodes.jsonl", report)
            if self.config["upper"].get("credit_assignment", "episode") == "immediate":
                # 即时信用分配属于独立实验设置，完整回报和 GAE 分支另行保留。
                for rollout in rollouts:
                    immediate_targets(rollout)
            elif self.config["upper"].get("gae_lambda") is not None:
                for rollout in rollouts:
                    gae_targets(rollout, self.config["upper"]["gae_lambda"])
            for number, rollout in enumerate(rollouts):
                append_json(self.output / "upper_trajectories.jsonl", {"phase": stage,
                    "iteration": self.state[key] + 1, "episode": number,
                    "actions": [{"source": r["observation"].ids[r["index"]]
                                 if r["index"] < len(r["observation"].ids) else "NEW",
                                 "remaining_fraction": r["observation"].remaining_fraction,
                                 "old_logp": r["logp"], "old_value": r["value"],
                                 "reward": r["reward"], "return": r["return"],
                                 **{k: r[k] for k in ("search_reward", "forward_valid", "forward_pool_before", "forward_pool_after") if k in r},
                                 "advantage": r.get("advantage", r["return"] - r["value"]),
                                 "value_target": r.get("value_target", r["return"])} for r in rollout]})
            rows = [row for rollout in rollouts for row in rollout]
            if not rows:
                raise RuntimeError("no upper actions fit the time budget; increase search.seconds")
            if self.config.get("record_rollouts"):
                save_torch(self.output / "rollouts" / f"{stage}_{self.state[key]}.pt",
                    {"models": model_bundle(self.lower, self.upper, self.config), "rows": rows,
                     "optimizer": self.upper_optimizer.state_dict(), "shuffle_rng": self.rng.getstate()})
            metrics = update(self.upper, self.upper_optimizer, rows, rng=self.rng, **self.config["upper"]["ppo"])
            self.state[key] += 1
            append_json(self.output / "upper_curve.jsonl", {"phase": key, "iteration": self.state[key],
                        **metrics, "mean_episode_return": np.mean([r["return"] for r in reports]),
                        "mean_training_return": np.mean([r.get("training_return", r["return"]) for r in reports]),
                        "new_factors": sum(r["new_factors"] for r in reports), "wall_seconds": self.elapsed()})
            rollback_each_update = guarded and self.config["upper"].get("joint_rollback", False)
            if self.config.get("upper_validation") and (rollback_each_update or self.state[key] % self.config["upper_validation"]["every"] == 0 or self.state[key] == target):
                candidate_score = self.validate_upper(stage, self.state[key])
                if guarded:
                    accepted = candidate_score > reference_score
                    rollback = self.config["upper"].get("joint_rollback", False) and not accepted
                    save_torch(self.output / "joint_candidates" / f"iteration_{self.state[key]}.pt",
                        model_bundle(self.lower, self.upper, self.config, validation_score=candidate_score,
                                     accepted=accepted, lower_optimizer=self.lower_optimizer.state_dict(),
                                     upper_optimizer=self.upper_optimizer.state_dict()))
                    append_json(self.output / "joint_decisions.jsonl", {"iteration": self.state[key],
                        "reference_score": reference_score, "candidate_score": candidate_score, "accepted": accepted,
                        "lower_reward_unchanged": True, "working_update_retained": not rollback,
                        "best_checkpoint_improved": accepted})
                    if rollback:
                        selected = torch.load(self.output / "best_joint.pt", map_location=self.config["device"], weights_only=False)
                        self.lower.load_state_dict(selected["lower"])
                        self.upper.load_state_dict(selected["upper"])
                        self.lower_optimizer.load_state_dict(selected["lower_optimizer"])
                        self.upper_optimizer.load_state_dict(selected["upper_optimizer"])
            self.commit()
        if self.config.get("upper_validation"):
            # 正式导出模型与持续训练模型分别保存。
            # 载入开发评分最优模型前，保存工作模型的优化器及随机数状态。
            self.commit()
            shutil.copy2(self.output / "training_state.pt", self.output / f"{stage}_working_final.pt")
            selected = torch.load(self.output / f"best_{stage}.pt", map_location=self.config["device"], weights_only=False)
            self.lower.load_state_dict(selected["lower"])
            self.upper.load_state_dict(selected["upper"])
            self.lower_optimizer.load_state_dict(selected["lower_optimizer"])
            self.upper_optimizer.load_state_dict(selected["upper_optimizer"])
            self.state[f"selected_{stage}_iteration"] = selected["selected_iteration"]
        name = "joint_final.pt" if joint else "upper_frozen_final.pt"
        save_torch(self.output / name, model_bundle(self.lower, self.upper, self.config))
        self.finish_stage(stage)
        self.state["phase"] = "evaluation" if joint else "joint"
        self.commit()

    def validate_upper(self, stage, iteration):
        # 开发任务评分决定选中模型，训练中的工作参数与最终导出参数分别记录。
        """在开发任务上执行固定种子的搜索验证，返回平均选点分数。

        按配置使用搜索回报或历史后续达标数与组合增量选点，记录结果并保存最佳检查点。
        串行验证保存并恢复 PyTorch 随机状态，以便继续训练沿用原有随机流。
        """
        settings = self.config["upper_validation"]
        names = settings.get("tasks") or [settings["task"]]
        tasks = [next(t for t in self.config["tasks"] if t["name"] == name) for name in names]
        if any(task["role"] != "development" for task in tasks):
            raise ValueError("checkpoint selection requires a development task")
        search = {**self.config["search"], "seconds": settings["seconds"]}
        if settings.get("max_attempts"):
            search["max_attempts"] = settings["max_attempts"]
            search["budget_mode"] = "attempts"
        reports = []
        if self.collector:
            from .parallel import cpu_state
            jobs = [{"lower": cpu_state(self.lower), "upper": cpu_state(self.upper), "task": task,
                     "seed": 74191 + i, "initial": self.initial, "search": search,
                     "trace": self.output / "validation_traces" / f"{stage}_{iteration}_{task['name']}_{i}.jsonl",
                     "tag": f"{stage}_validation_{iteration}_{task['name']}_{i}"}
                    for task in tasks for i in range(settings["episodes"])]
            reports = [r[3] for r in self.collector.episodes(jobs)]
        else:
            torch_state = torch.get_rng_state()
            try:
                for task in tasks:
                    for i in range(settings["episodes"]):
                        torch.manual_seed(74191 + i)
                        e = evaluator_for(self.grammar, self.panels[task["name"]], self.config).for_episode(self.initial)
                        report = episode(self.lower, self.upper, e, self.initial, search, random.Random(74191 + i), record_lower=False)[3]
                        report["task"] = task["name"]
                        reports.append(report)
            finally:
                torch.set_rng_state(torch_state)
        if settings.get("max_attempts") and any(r["attempts"] != settings["max_attempts"] or r["late_results_excluded"] for r in reports):
            raise RuntimeError("validation did not complete its declared fixed attempt count")
        scores = [r["return"] for r in reports]
        if settings.get("forward", False):
            from .experiments import future_check
            from .search import Archive
            scores = []
            for report in reports:
                panels = self.panels[report["task"]]
                evaluator = evaluator_for(self.grammar, panels, self.config).for_episode(self.initial)
                archive = Archive(evaluator, self.initial, self.config["search"])
                initial = {"discoveries": [], "pool_asts": [t.to_dict() for t in archive.pool.stats.trees],
                           "pool_weights": archive.pool.score.weights}
                initial_ic = future_check(initial, evaluator, panels["future"])["pool_ic"]
                for tree in report["pool_asts"]:
                    evaluator.get(Node.from_dict(tree))
                result = future_check(report, evaluator, panels["future"])
                valid = sum(f["reliable"] and f["coverage"] >= self.config["search"]["min_coverage"] and
                            f["ic"] >= self.config["search"]["quality_threshold"] and
                            f["rank_ic"] >= self.config["search"].get("min_search_rank_ic", .01)
                            for f in result["single_factors"])
                score = valid + self.config["search"]["combination_weight"] * (result["pool_ic"] - initial_ic)
                report.update(forward=result, forward_valid=valid, initial_forward_ic=initial_ic, selection_score=score)
                scores.append(score)
        value = float(np.mean(scores))
        append_json(self.output / "upper_validation.jsonl", {"stage": stage, "iteration": iteration,
                    "mean_return": value, "task": ",".join(names), "tasks": names, "episodes": reports,
                    "mean_search_return": float(np.mean([r["return"] for r in reports])),
                    "selection_metric": "forward_count_and_pool_gain" if settings.get("forward") else "search_return",
                    "budget_type": "fixed_attempts" if settings.get("max_attempts") else "wall_seconds"})
        if value > self.state.get(f"best_{stage}_score", -math.inf):
            self.state[f"best_{stage}_score"] = value
            save_torch(self.output / f"best_{stage}.pt", model_bundle(self.lower, self.upper, self.config,
                       selected_iteration=iteration, validation_score=value,
                       lower_optimizer=self.lower_optimizer.state_dict(), upper_optimizer=self.upper_optimizer.state_dict()))
        return value

    def run(self):
        """推进训练阶段并返回实验实例，无论正常结束或抛出异常均关闭并行采集器。"""
        try:
            return self._run()
        finally:
            if self.collector:
                self.collector.close()
                self.collector = None

    def _run(self):
        # 根据已保存阶段推进，父模型初始化或恢复运行可以从中间阶段开始。
        """按已保存阶段依次推进结构预训练、下层、冻结下层的上层及联合训练，返回自身。

        并行采集启用信号缓存时先准备公共起点信号，允许初始化或恢复后从中间阶段继续。
        """
        self.state.setdefault("setup_wall_seconds", self.elapsed())
        if self.collector and self.config.get("signal_cache_dir") and not self.state.get("signals_prepared"):
            started = perf_counter()
            prepared = self.collector.prepare([t for t in self.config["tasks"] if t["role"] != "test"], self.initial)
            self.state["setup_wall_seconds"] += perf_counter() - started
            self.state["signals_prepared"] = True
            save_json(self.output / "initial_signal_preparation.json", prepared)
            self.commit()
        if self.state["phase"] == "corpus":
            self.state["phase"] = "pretrain"
            self.commit()
        if self.state["phase"] == "pretrain":
            self.syntax_pretrain()
        if self.state["phase"] == "lower":
            self.train_lower()
        if self.state["phase"] == "upper":
            self.train_upper()
        if self.state["phase"] == "joint":
            self.train_upper(joint=True)
        return self
