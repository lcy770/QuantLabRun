"""Matched search/forward-reward PPO experiment on historical development data."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

import numpy as np
import torch

from .hierarchical.__main__ import load_config
from .hierarchical.ast import Node
from .hierarchical.evaluation import load_task
from .hierarchical.experiments import future_check, original_ppo
from .hierarchical.io import append_json, digest, run_lock, save_json
from .hierarchical.search import Archive, episode
from .hierarchical.training import Campaign, evaluator_for, restore_models, validate_forward_training


PROJECT = Path(__file__).resolve().parents[1]
PREVIOUS = PROJECT / "var/experiments/mechanism_study_20261109"
PARENT = PROJECT / "var/experiments/budget_study_20261108"
ROOT = PROJECT / "var/experiments/forward_reward_study_20261109"
ARMS = ("search_reward", "forward_reward")
METHODS = (*ARMS, "parent", "random", "original_rpn")
MODEL_CHECKPOINT = "upper_frozen_final.pt"


def protocol():
    return json.loads((ROOT / "resolved_protocol.json").read_text(encoding="utf-8"))


def prepare():
    if (ROOT / "resolved_protocol.json").exists():
        return protocol()
    config = json.loads((PREVIOUS / "protocol_config.json").read_text(encoding="utf-8"))
    cache = PREVIOUS / "data_missingness_reference"
    dates = np.load(cache / "dates.npy").astype("datetime64[D]")
    for task in config["tasks"]:
        for role in ("fit", "search", "future"):
            if role in task:
                task[role]["cache"] = str(cache)
        if task["role"] == "train":
            selected = dates[(dates >= np.datetime64(task["search"]["start"])) &
                             (dates <= np.datetime64(task["search"]["end"]))]
            if len(selected) < 60:
                raise ValueError("historical reward split needs sixty trading days")
            task["search"]["end"] = str(selected[19])
            task["future"] = {"cache": str(cache), "start": str(selected[40]), "end": str(selected[59])}
    config["signal_cache_dir"] = str(ROOT / "signal_cache")
    config["upper"].update(iterations=16, joint_iterations=0, joint_guard=False, joint_rollback=False,
                           learning_rate=.001, archive_context=False, training_reward="search")
    config["upper_validation"].update(every=4, episodes=2, max_attempts=32, forward=True)
    config["search"].update(max_attempts=48, budget_mode="attempts")
    config["rollout_workers"] = 3
    save_json(ROOT / "protocol_config.json", config)
    config = load_config(ROOT / "protocol_config.json")
    panels = {task["name"]: load_task(task, config["target_horizon"], config["lookback"],
              config["future_days"], with_future=True) for task in config["tasks"]}
    validate_forward_training(config["tasks"], panels)
    save_json(ROOT / "temporal_audit.json", {name: {
        role: {"days": len(panel.dates), "first": str(panel.dates.min()), "last": str(panel.dates.max()),
               "latest_label": str(panel.ends.max())} for role, panel in parts.items()}
        for name, parts in panels.items()})
    save_json(ROOT / "resolved_protocol.json", config)
    save_json(ROOT / "preregistration.json", {
        "created_at": datetime.now().astimezone().isoformat(), "training_seeds": [0, 1, 2],
        "arms": list(ARMS), "parent": "same best_upper checkpoint per seed from budget_study_20261108",
        "optimizer": "fresh upper Adam in both arms, lr=.001, GAE=.95, lower frozen",
        "training": "16 batches x 6 episodes x exactly 48 choices per model",
        "train_split": "first 20 search days, 20-day label gap, last 20 forward days of original sixty-day search",
        "teacher": "after rollout only, forward-qualified count + 50 * locked forward pool IC change",
        "observation": "unchanged fit/search-only archive history, teacher labels never reach collectors",
        "selection": "same purged historical forward-development score at batches 0/4/8/12/16",
        "data": "same repaired missingness reference for both arms and every baseline",
        "fixed_evaluation": "4 AST methods x 3 model seeds x 2 tasks x 2 sampling seeds, exactly 32 choices",
        "timed_evaluation": "5 methods x 3 model seeds x 2 tasks x sampling seed 101, serial 45 seconds",
        "evaluation_order_seed": 71923, "interpretation": "reused development data, retrospective diagnosis",
        "cloud_spent_cny": 0, "cloud_budget_cny": 20})
    snapshot = ROOT / "source_snapshot"
    for source in PROJECT.joinpath("single_factor").rglob("*.py"):
        destination = snapshot / source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    save_json(ROOT / "source_manifest.json", {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob("*.py")})
    return config


def train_job(job):
    seed, arm = job
    config = copy.deepcopy(protocol())
    config["seed"] = seed
    config["upper"]["training_reward"] = "forward" if arm == "forward_reward" else "search"
    parent = PARENT / "runs" / f"training_seed_{seed}" / "best_upper.pt"
    config["warm_start"] = {"path": str(parent), "sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}
    config["fingerprint"] = digest(config)
    output = ROOT / "runs" / arm / f"training_seed_{seed}"
    with run_lock(output):
        campaign = Campaign(config, output, resume=(output / "training_state.pt").exists())
        try:
            if "warm_start" not in campaign.state:
                bundle = torch.load(parent, map_location="cpu", weights_only=False)
                campaign.lower.load_state_dict(bundle["lower"])
                campaign.upper.load_state_dict(bundle["upper"])
                campaign.rng = random.Random(39171 + seed)
                torch.manual_seed(39171 + seed)
                campaign.state.update(phase="upper", lower_iteration=128, warm_start=config["warm_start"])
                campaign.commit()
            campaign.run()
            save_json(output / "cost.json", {"additional_wall_seconds": campaign.elapsed(), "seed": seed, "arm": arm})
            return {"seed": seed, "arm": arm, "phase": campaign.state["phase"]}
        finally:
            if campaign.collector:
                campaign.collector.close()


def train():
    # Keep arms sequential. Three learners each own three single-thread workers.
    for arm in ARMS:
        with ProcessPoolExecutor(max_workers=3) as executor:
            for result in executor.map(train_job, [(seed, arm) for seed in range(3)]):
                append_json(ROOT / "training_completion.jsonl", result)
                print(json.dumps(result), flush=True)


def evaluate_job(job):
    seed, task_name, method, sampling_seed, timed = job
    output = ROOT / "trials" / ("timed" if timed else "fixed") / f"{method}_{seed}_{task_name}_{sampling_seed}.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = protocol()
    task = next(t for t in config["tasks"] if t["name"] == task_name)
    checkpoint = (ROOT / "runs" / method / f"training_seed_{seed}" / MODEL_CHECKPOINT if method in ARMS
                  else PARENT / "runs" / f"training_seed_{seed}" / "best_upper.pt")
    lower, upper = restore_models(torch.load(checkpoint, weights_only=False, map_location="cpu"))
    panels = load_task(task, config["target_horizon"], config["lookback"], config["future_days"], with_future=True)
    initial = [Node.from_dict(json.loads(line)["ast"]) for line in Path(config["initial_library"]).read_text().splitlines()]
    evaluator = evaluator_for(lower.grammar, panels, config).for_episode(initial)
    for tree in initial:
        evaluator.get(tree)
    archive = Archive(evaluator, initial, config["search"])
    initial_report = {"discoveries": [], "pool_asts": [t.to_dict() for t in archive.pool.stats.trees],
                      "pool_weights": archive.pool.score.weights}
    initial_forward = future_check(initial_report, evaluator, panels["future"])["pool_ic"]
    settings = {**config["search"], "seconds": config["evaluation"]["seconds"] if timed else 300., "max_attempts": 100000 if timed else 32,
                "budget_mode": "wall_seconds" if timed else "attempts"}
    torch.manual_seed(sampling_seed)
    if method == "original_rpn":
        sys.path.insert(0, str(PROJECT / "backend"))
        sys.path.insert(0, str(PROJECT / "quantops"))
        report = original_ppo(evaluator, initial, settings, seed * 100000 + sampling_seed, output.with_suffix(".jsonl"))
    else:
        report = episode(lower, upper, evaluator, initial, settings, random.Random(sampling_seed),
                         scheduler="random" if method == "random" else "ppo", output=output.with_suffix(".jsonl"),
                         tag=output.stem, record_lower=False)[3]
    result = future_check(report, evaluator, panels["future"])
    valid = sum(f["reliable"] and f["coverage"] >= settings["min_coverage"] and
                f["ic"] >= settings["quality_threshold"] and f["rank_ic"] >= settings["min_search_rank_ic"]
                for f in result["single_factors"])
    report.update(model_seed=seed, method=method, task=task_name, sampling_seed=sampling_seed,
                  evaluation_mode="timed" if timed else "fixed", future=result, forward_valid=valid,
                  initial_forward_ic=initial_forward, forward_pool_gain=result["pool_ic"] - initial_forward,
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  full_trial_wall_seconds=time.perf_counter() - started,
                  interpretation="reused development slices, retrospective diagnosis")
    save_json(output, report)
    return report


def evaluate(timed):
    methods = METHODS if timed else METHODS[:-1]
    jobs = [(seed, task, method, sampling_seed, timed) for seed in range(3) for task in ("dev_0", "dev_1")
            for sampling_seed in ([101] if timed else [101, 202]) for method in methods]
    random.Random(71923).shuffle(jobs)
    if timed:
        for result in map(evaluate_job, jobs):
            print(json.dumps({k: result[k] for k in ("method", "model_seed", "task", "new_factors", "forward_valid", "forward_pool_gain")}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=6) as executor:
            for result in executor.map(evaluate_job, jobs):
                print(json.dumps({k: result[k] for k in ("method", "model_seed", "task", "new_factors", "forward_valid", "forward_pool_gain")}), flush=True)


def summarize():
    import csv
    rows = []
    for path in sorted((ROOT / "trials").glob("*/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({k: report[k] for k in ("evaluation_mode", "method", "model_seed", "task", "sampling_seed",
                     "new_factors", "forward_valid", "forward_pool_gain", "attempts", "wall_seconds")})
        rows[-1]["duplicate_rate"] = report.get("duplicate_rate")
        rows[-1]["forward_pool_ic"] = report["future"]["pool_ic"]
    with (ROOT / "all_trials.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for mode in ("fixed", "timed"):
        for method in METHODS:
            selected = [r for r in rows if r["evaluation_mode"] == mode and r["method"] == method]
            if selected:
                groups.append({"mode": mode, "method": method, "trials": len(selected),
                    **{key: statistics.mean(r[key] for r in selected) for key in
                       ("new_factors", "forward_valid", "forward_pool_ic", "forward_pool_gain", "attempts", "wall_seconds")}})
    save_json(ROOT / "comparison.json", groups)
    print(json.dumps(groups, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "fixed", "timed", "summarize", "all"))
    stage = parser.parse_args().stage
    if stage in {"prepare", "all"}:
        prepare()
    if stage in {"train", "all"}:
        train()
    if stage in {"fixed", "all"}:
        evaluate(False)
    if stage in {"timed", "all"}:
        evaluate(True)
    if stage in {"summarize", "all"}:
        summarize()


if __name__ == "__main__":
    main()
