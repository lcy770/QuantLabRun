"""用配对短实验筛查原生 RPN 双层模型的不同完成目标。"""
# 后续共同奖励实验驱动。这里的 A/B/C 方案与原 Word 分别定义两层奖励的方案区分记录。
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter

import torch

from .hierarchical.ast import Node
from .hierarchical.evaluation import load_task
from .hierarchical.experiments import future_check
from .hierarchical.io import append_json, digest, run_lock, save_json, save_torch
from .hierarchical.models import UpperPolicy
from .hierarchical.search import Archive, episode
from .hierarchical.training import Campaign, completion_training_rewards, evaluator_for, model_bundle, restore_models


PROJECT = Path(os.environ.get("QUANTLAB_PROJECT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT / "var/experiments/rpn_goal_alignment_20261110"
PARENTS = PROJECT / "var/experiments/native_rpn_speed_20261110/runs/rpn_native"
ARMS = ("weighted", "joint", "joint_cost")
# A/B/C 需要训练，C 下层配随机上层及父下层配随机上层为额外评价对照。
METHODS = (*ARMS, "joint_cost_random", "parent_random")


def protocol():
    """读取本轮实验目录中的解析协议，返回训练与评价共用的配置字典。"""
    return json.loads((ROOT / "resolved_protocol.json").read_text(encoding="utf-8"))


def train_job(job):
    """为指定实验组和训练种子执行追加训练，返回耗时及两层更新数。

    各组从同种子的父下层和配对上层初始化开始，按组切换共同奖励及成本观测。
    支持恢复已有训练状态，并在各组独立目录保存初始模型、最终状态和成本记录。
    """
    arm, seed = job
    torch.set_num_threads(1)
    config = protocol()
    config["seed"] = seed
    # upper.training_reward 在本实验中还控制采集后上下层共同重评分。
    # A 为加权终点，B/C 为严格成功终点，C 额外启用成本与贡献率观测。
    config["upper"].update(training_reward="forward_weighted" if arm == "weighted" else "forward_joint",
                           cost_context=arm == "joint_cost")
    config["fingerprint"] = digest([config, arm, "paired_rpn_completion_v1"])
    output = ROOT / "runs" / arm / f"training_seed_{seed}"
    with run_lock(output):
        campaign = Campaign(config, output, resume=(output / "training_state.pt").exists())
        if "paired_parent" not in campaign.state:
            # 各组从同一种子的已训练父下层开始，实际追加训练从 lower 阶段执行。
            parent = PARENTS / f"training_seed_{seed}/joint_final.pt"
            saved = torch.load(parent, map_location="cpu", weights_only=False)
            campaign.lower.load_state_dict(saved["lower"])
            # 各组上层配对初始化，新增成本列初值为零，编码器复用父下层的参数。
            torch.manual_seed(84291 + seed)
            baseline = UpperPolicy(campaign.grammar, config["width"], representation="rpn")
            baseline.actor.encoder.load_state_dict(campaign.lower.actor_encoder.state_dict())
            baseline.critic.encoder.load_state_dict(campaign.lower.critic_encoder.state_dict())
            campaign.upper.load_state_dict(baseline.state_dict())
            campaign.rng = random.Random(49171 + seed)
            torch.manual_seed(49171 + seed)
            campaign.state.update(phase="lower", paired_parent=str(parent))
            save_torch(output / "initial.pt", model_bundle(campaign.lower, campaign.upper, config))
            campaign.commit()
        campaign.run()
        campaign.state["phase"] = "complete"
        campaign.commit()
        result = {"arm": arm, "seed": seed, "wall_seconds": campaign.elapsed(),
                  "lower_updates": campaign.state["lower_iteration"] + campaign.state["joint_iteration"],
                  "upper_updates": campaign.state["upper_iteration"] + campaign.state["joint_iteration"]}
        save_json(output / "cost.json", result)
        return result


def evaluate_job(job):
    # 固定尝试数检验候选质量，固定 15 秒检验实际产出，两种预算分别汇总。
    """按种子、任务、方法及预算方式评价模型，返回并保存单次搜索报告。

    固定三十二次尝试比较候选质量，固定十五秒比较吞吐，父下层和随机调度用于配对对照。
    搜索后统计后续达标、严格成功和锁定组合 IC 增量，已有报告直接复用。
    """
    seed, task_name, method, timed = job
    output = ROOT / "trials" / ("timed" if timed else "fixed") / f"{method}_{seed}_{task_name}.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = protocol()
    task = next(task for task in config["tasks"] if task["name"] == task_name)
    if method == "parent_random":
        # E 组使用修改前父下层，D 组使用 C 训练后的下层，二者均随机调度。
        checkpoint = PARENTS / f"training_seed_{seed}/joint_final.pt"
    else:
        arm = "joint_cost" if method == "joint_cost_random" else method
        checkpoint = ROOT / "runs" / arm / f"training_seed_{seed}/joint_final.pt"
    lower, upper = restore_models(torch.load(checkpoint, map_location="cpu", weights_only=False))
    panels = load_task(task, config["target_horizon"], config["lookback"], config["future_days"], with_future=True)
    initial = [Node.from_dict(json.loads(line)["ast"]) for line in Path(config["initial_library"]).read_text().splitlines()]
    evaluator = evaluator_for(lower.grammar, panels, config).for_episode(initial)
    for tree in initial:
        evaluator.get(tree)
    archive = Archive(evaluator, initial, config["search"])
    initial_forward = future_check({"discoveries": [], "pool_asts": [tree.to_dict() for tree in archive.pool.stats.trees],
                                   "pool_weights": archive.pool.score.weights}, evaluator, panels["future"])["pool_ic"]
    settings = {**config["search"], "seconds": 15. if timed else 300., "max_attempts": 100000 if timed else 32,
                "budget_mode": "wall_seconds" if timed else "attempts"}
    # 各次搜索载入独立模型缓存，并固定采样随机种子。
    torch.manual_seed(101)
    rows, _, _, report = episode(lower, upper, evaluator, initial, settings, random.Random(101),
        scheduler="random" if method.endswith("random") else "ppo", record_lower=False,
        output=output.with_suffix(".jsonl"), tag=output.stem)
    # 搜索结束后才用后续区间验收，评价过程只统计指标。
    future = future_check(report, evaluator, panels["future"])
    valid = sum(factor["reliable"] and factor["coverage"] >= settings["min_coverage"]
                and factor["ic"] >= settings["quality_threshold"]
                and factor["rank_ic"] >= settings["min_search_rank_ic"] for factor in future["single_factors"])
    # 此处传入空下层轨迹，只重算严格成功计数，后续没有调用优化器更新模型。
    completion_training_rewards(rows, [], report, evaluator, initial, panels["future"], settings, "forward_joint")
    if valid != report["training_forward_valid"]:
        raise AssertionError("independent forward single-factor counts disagree")
    report.update(method=method, model_seed=seed, task=task_name, evaluation_mode="timed" if timed else "fixed",
        forward_valid=valid, forward_joint_count=report["training_joint_success"], future=future,
        forward_pool_gain=future["pool_ic"] - initial_forward, initial_forward_ic=initial_forward,
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        interpretation="reused development, three paired seeds, short screen, no checkpoint selection")
    save_json(output, report)
    return report


def summarize():
    """收集已完成搜索报告，输出逐次明细及按预算方式、方法汇总的 CSV 和 JSON。"""
    keys = ("evaluation_mode", "method", "model_seed", "task", "new_factors", "forward_valid", "forward_joint_count",
            "forward_pool_gain", "attempts", "completed_candidates", "wall_seconds", "duplicate_rate")
    rows = [{key: report[key] for key in keys}
            for path in sorted((ROOT / "trials").glob("*/*.json"))
            for report in [json.loads(path.read_text(encoding="utf-8"))]]
    if not rows:
        return
    with (ROOT / "all_trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    comparison = []
    for mode in ("fixed", "timed"):
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method and row["evaluation_mode"] == mode]
            if selected:
                comparison.append({"mode": mode, "method": method, "trials": len(selected),
                    **{key: statistics.mean(row[key] for row in selected) for key in keys[4:]}})
    save_json(ROOT / "comparison.json", comparison)
    with (ROOT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


def prepare():
    # 登记这次短实验的实际更新数、后端、奖励和预算，并保留独立输出目录。
    """从历史协议构建本轮原生 RPN 配置，登记实际训练规模、奖励、配对与评价方案。

    将源码身份、解析协议和实验预登记写入本轮目录。
    """
    config = json.loads((PROJECT / "var/experiments/coordination_study_20261109/resolved_protocol.json").read_text())
    for key in ("initializer_checkpoint", "upper_validation"):
        config.pop(key, None)
    config.update(rollout_workers=1, torch_threads=1, device="cpu", record_rollouts=True,
                  record_lower_trajectories=True,
                  corpus_source=str(PROJECT / "var/experiments/architecture_study_20261109/compatible_corpus"),
                  signal_cache_dir=str(ROOT / "signal_cache"))
    config["corpus"]["count"] = 144
    config["lower"].update(representation="rpn", edit_backend="rpn", task_context=True, position_context=True,
        iterations=4, batch_formulas=32, learning_rate=.0003, continuation_states=8, continuation_samples=3,
        completion_reward=True, search_training=True, diffusion_steps=2, perturbation_steps=2)
    config["upper"].update(representation="rpn", iterations=4, joint_iterations=2, episodes_per_batch=2,
        learning_rate=.0003, joint_guard=False, joint_rollback=False, archive_context=False,
        training_reward="forward_weighted", credit_assignment="episode", gae_lambda=.95)
    for layer in ("lower", "upper"):
        config[layer]["ppo"].update(epochs=4, batch_size=32)
    config["search"].update(seconds=8., max_attempts=100000, budget_mode="wall_seconds", diffusion_steps=2, perturbation_steps=2)
    config["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["implementation_hash"] = digest({str(path.relative_to(Path(__file__).parent)):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("hierarchical/*.py"))})
    save_json(ROOT / "resolved_protocol.json", config)
    save_json(ROOT / "preregistration.json", {"at": datetime.now().astimezone().isoformat(), "arms": ARMS,
        "seeds": [0, 1, 2], "parent": str(PARENTS), "lower_updates": "4 independent + 2 joint, 32 candidates each",
        "upper_updates": "4 frozen-lower + 2 fresh-after-lower batches, 2 x 8 seconds each",
        "reward": {"weighted": "eligible * (future_qualified + 50 * locked_future_pool_delta)",
                   "joint": "eligible * (int(future_qualified and delta > 1e-12) - clip(-delta/.02,0,1))",
                   "joint_cost": "same joint reward plus visible relative expansion cost and search contribution rate"},
        "pairing": "same trained RPN parent per seed, same initial function, zero extended columns, same task RNG",
        "observation_boundary": "mature historical future labels used only after completed collection",
        "selection": "fixed final updates, no development selection",
        "evaluation": "5 methods x 3 paired seeds x 2 development tasks, fixed32 and serial15s, sampling seed101",
        "execution": "at most six single-thread learners, no nested collector, timing serialized after fixed evaluation",
        "limits": "short screen, reused development, count joint success against pre-action locked pool, final pool gain separately"})


def main():
    """按命令行阶段分派协议准备、九个模型训练、两类预算评价和结果汇总。

    训练及固定尝试评价使用进程池，计时评价串行执行，持续写入进度和完成记录。
    """
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prepare":
        prepare()
        return
    completed = []
    if stage in {"train", "all"}:
        save_json(ROOT / "progress.json", {"stage": "training", "completed_models": 0, "total_models": 9})
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(train_job, (arm, seed)) for seed in range(3) for arm in ARMS]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                append_json(ROOT / "training_completion.jsonl", result)
                save_json(ROOT / "progress.json", {"stage": "training", "completed_models": len(completed), "total_models": 9})
    for timed, label in ((False, "fixed"), (True, "timed")):
        if stage not in {label, "all"}:
            continue
        jobs = [(seed, task, method, timed) for seed in range(3) for task in ("dev_0", "dev_1") for method in METHODS]
        random.Random(71923).shuffle(jobs)
        save_json(ROOT / "progress.json", {"stage": label, "completed_trials": 0, "total_trials": len(jobs)})
        if timed:
            for index, job in enumerate(jobs):
                result = evaluate_job(job)
                print(json.dumps({key: result[key] for key in ("method", "model_seed", "task", "forward_joint_count", "forward_pool_gain")}), flush=True)
                save_json(ROOT / "progress.json", {"stage": label, "completed_trials": index + 1, "total_trials": len(jobs)})
        else:
            with ProcessPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(evaluate_job, job) for job in jobs]
                for index, future in enumerate(as_completed(futures)):
                    future.result()
                    save_json(ROOT / "progress.json", {"stage": label, "completed_trials": index + 1, "total_trials": len(jobs)})
        summarize()
    if stage in {"all", "summarize"}:
        summarize()
        save_json(ROOT / "progress.json", {"stage": "complete", "completed_models": 9, "completed_trials": 60})


if __name__ == "__main__":
    main()
