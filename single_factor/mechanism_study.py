"""Controlled diagnostics for the hierarchical search, using train/development only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
import urllib.request

import numpy as np
import torch

from single_factor.hierarchical.ast import Grammar, Node, walk
from single_factor.hierarchical.io import append_json, digest, save_json, run_lock
from single_factor.hierarchical.training import Campaign, evaluator_for, restore_models
from single_factor.hierarchical.evaluation import load_task, load_slice
from single_factor.hierarchical.search import Archive
from single_factor.hierarchical.search import episode
from single_factor.hierarchical.ppo import update


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "var/experiments/budget_study_20261108"
ROOT = PROJECT / "var/experiments/mechanism_study_20261109"


def records(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def average(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return statistics.mean(values) if values else None


def historical_audit():
    """Read saved trajectories without evaluating or choosing on old test labels."""
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {"created_at": datetime.now().astimezone().isoformat(), "seeds": []}
    for seed in (0, 1, 2):
        parent = PARENT / "runs" / f"training_seed_{seed}"
        lower = [row for row in records(parent / "lower_curve.jsonl") if "iteration" in row]
        upper = [row for row in records(parent / "upper_curve.jsonl") if row["phase"] == "upper_iteration"]
        decisions = list(records(parent / "joint_decisions.jsonl"))
        actions, repeated, rewards = 0, 0, []
        for episode in records(parent / "upper_trajectories.jsonl"):
            if episode["phase"] != "upper":
                continue
            visited = set()
            for action in episode["actions"]:
                actions += 1
                repeated += action["source"] in visited
                visited.add(action["source"])
                rewards.append(action["reward"])
        changes = Counter()
        final_rewards = []
        for row in records(parent / "lower_trajectories.jsonl"):
            if row["joint_iteration"]:
                continue
            final = Node.from_dict(row["terminal_ast"])
            start = Node.from_dict(row["actions"][0]["tree"])
            changes["trajectories"] += 1
            changes["final_equals_noisy_input"] += final.id == start.id
            for action in row["actions"]:
                changes["actions"] += 1
                changes["root_edits"] += not action["path"]
                changes["identity_edits"] += Node.from_dict(action["tree"]).id == final.id
            final_rewards.append(row["terminal_reward"])
        item = {"seed": seed, "lower_first16_reward": average(r["mean_terminal_reward"] for r in lower[:16]),
                "lower_last16_reward": average(r["mean_terminal_reward"] for r in lower[-16:]),
                "upper_last16_entropy": average(r["entropy"] for r in upper[-16:]),
                "upper_last16_kl": average(r["approx_kl"] for r in upper[-16:]),
                "upper_last16_value_explained": average(r["value_explained_variance"] for r in upper[-16:]),
                "upper_total_actions": actions, "repeated_start_fraction": repeated / actions,
                "nonzero_reward_fraction": sum(r != 0 for r in rewards) / actions,
                "mean_reward": average(rewards), "joint_accepted": sum(r["accepted"] for r in decisions),
                "joint_equal": sum(r["candidate_score"] == r["reference_score"] for r in decisions),
                "lower_edits": dict(changes)}
        report["seeds"].append(item)
        print(json.dumps(item), flush=True)
    save_json(ROOT / "historical_audit.json", report)
    return report


def literature():
    """Save primary-source abstracts, including retrieval failures."""
    target = ROOT / "literature"
    target.mkdir(parents=True, exist_ok=True)
    sources = {"ppo": "https://arxiv.org/abs/1707.06347",
               "hiro": "https://arxiv.org/abs/1805.08296",
               "alphagen": "https://arxiv.org/abs/2306.12964"}
    results = []
    for name, url in sources.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            (target / f"{name}.html").write_bytes(content)
            results.append({"name": name, "url": url, "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()})
        except Exception as error:
            results.append({"name": name, "url": url, "error": str(error)})
    save_json(target / "sources.json", results)
    print(json.dumps(results), flush=True)


def prepare():
    from single_factor.hierarchical.__main__ import load_config
    path = ROOT / "protocol_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((PARENT / "protocol_config.json").read_text())
    config["tasks"] = [task for task in config["tasks"] if task["role"] != "test"]
    dates = np.load(Path(config["tasks"][0]["fit"]["cache"]) / "dates.npy").astype("datetime64[D]")
    latest_label = int(np.flatnonzero(dates <= np.datetime64("2024-12-31"))[-1])
    for task in config["tasks"]:
        if task["role"] != "development":
            continue
        start = int(np.flatnonzero(dates >= np.datetime64(task["search"]["start"]))[0])
        length = 30 if task["name"] == "dev_0" else 20
        task["search"]["end"] = str(dates[start + length - 1])
        forward_start = start + length + config["target_horizon"]
        forward_end = min(forward_start + length - 1, latest_label - config["target_horizon"])
        if forward_end - forward_start + 1 < 15:
            raise ValueError("insufficient mature forward-development days")
        task["future"] = {"cache": task["search"]["cache"], "start": str(dates[forward_start]),
                          "end": str(dates[forward_end])}
    config["signal_cache_dir"] = str(ROOT / "signal_cache")
    config["upper_validation"].update(every=4, episodes=2, max_attempts=32, seconds=300., forward=True)
    config["search"].update(seconds=300., max_attempts=48, budget_mode="attempts")
    config["upper"].update(iterations=16, joint_iterations=16, episodes_per_batch=6)
    config["evaluation"].update(methods=["random", "quality", "ucb", "upper_ppo"], seconds=45.)
    config["rollout_workers"] = 3
    save_json(path, config)
    config = load_config(path)
    save_json(ROOT / "resolved_protocol.json", config)
    plan = {"created_at": datetime.now().astimezone().isoformat(), "seeds": [0, 1, 2],
            "arms": {"upper_control": "frozen lower, upper lr=0.0001",
                     "upper_fast": "frozen lower, upper lr=0.001",
                     "joint_rollback": "upper lr=0.001, legacy per-update rollback",
                     "joint_accumulate": "upper lr=0.001, continuous working learner and separate best checkpoint"},
            "parent": "same saved best_upper per seed for every arm", "batches": 16,
            "episodes_per_batch": 6, "attempts_per_episode": 48,
            "development": "two shortened search slices and purged forward slices with labels mature by 2024-12-31",
            "selection": "forward valid count + 50 * locked pool forward gain against initial pool",
            "interpretation": "retrospective development diagnosis, reused development data, no new blind test",
            "probe": "fixed initial archive, 32 shortlisted seeds x 4 draws x 4 tasks x 3 sampling seeds x 2 lower models x 2 perturbation settings",
            "timed_evaluation": "serial 45s, sampling seeds 101/202, same public library, generated caches isolated",
            "cloud_budget_cny": 20, "config_fingerprint": config["fingerprint"]}
    save_json(ROOT / "preregistration.json", plan)
    snapshot = ROOT / "source_snapshot"
    for source in [*PROJECT.joinpath("single_factor").rglob("*.py"), PROJECT / "single_factor/requirements.txt"]:
        if "__pycache__" in source.parts:
            continue
        destination = snapshot / source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    save_json(ROOT / "source_manifest.json", {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in snapshot.rglob("*") if p.is_file()})
    return config


def protocol():
    return json.loads((ROOT / "resolved_protocol.json").read_text(encoding="utf-8"))


def benchmark():
    from single_factor.hierarchical.parallel import Collector
    config = protocol()
    bundle = torch.load(PARENT / "runs/training_seed_0/lower_base.pt", weights_only=False, map_location="cpu")
    lower, _ = restore_models(bundle)
    initial = [Node.from_dict(row["ast"]) for row in records(config["initial_library"])]
    results = []
    for workers in (4, 8, 12):
        candidate = {**config, "rollout_workers": workers}
        collector = Collector(candidate)
        try:
            for repeat in range(3):
                started = time.perf_counter()
                collected = collector.lower(lower, initial[:64], config["tasks"][0], config["lower"], random.Random(1903))
                result = {"workers": workers, "repeat": repeat, "seconds": time.perf_counter() - started,
                          "formulas": sum(len(r["rewards"]) for r in collected)}
                results.append(result)
                append_json(ROOT / "resource_benchmark.jsonl", result)
                print(json.dumps(result), flush=True)
        finally:
            collector.close()
    save_json(ROOT / "resource_benchmark.json", results)


def probe_job(job):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = protocol()
    seed, task_name, kind, perturbation = job
    path = ROOT / "probe" / f"{task_name}_{kind}_{seed}_noise{perturbation}.jsonl"
    if path.exists():
        return str(path)
    parent = PARENT / "runs" / f"training_seed_{seed}" / "lower_base.pt"
    bundle = torch.load(parent, weights_only=False, map_location="cpu")
    lower, _ = restore_models(bundle)
    if kind == "syntax":
        lower.load_state_dict(torch.load(config["initializer_checkpoint"], weights_only=False, map_location="cpu")["lower"])
    task = next(t for t in config["tasks"] if t["name"] == task_name)
    panels = load_task(task, config["target_horizon"], config["lookback"], config["future_days"], with_future="future" in task)
    initial = [Node.from_dict(row["ast"]) for row in records(config["initial_library"])]
    evaluator = evaluator_for(lower.grammar, panels, config).for_episode(initial)
    archive = Archive(evaluator, initial, config["search"])
    starts = archive.observe().trees
    temporary = path.with_suffix(".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    for number, start in enumerate(starts):
        for repeat in range(4):
            draw_seed = 90001 + seed * 10000 + number * 97 + repeat
            torch.manual_seed(draw_seed)
            noisy, trace = lower.grammar.corrupt(start, perturbation, random.Random(draw_seed))
            started = time.perf_counter()
            final, rows = lower.sample(noisy, remaining=2)
            proposal = archive.assess(final)
            metrics = proposal["metrics"]
            result = {"seed": seed, "task": task_name, "kind": kind, "perturbation_steps": perturbation,
                      "source": start.id, "source_expression": start.text(), "source_quality": evaluator.get(start).metrics["quality"],
                      "repeat": repeat, "expression": final.text(), "ast": final.to_dict(), "noise_ast": noisy.to_dict(),
                      "qualified_novel": proposal["new"], "duplicate": proposal["duplicate"],
                      "reward": proposal["reward"], "metrics": metrics,
                      "corruption_root_edits": sum(not inverse.path for _, _, inverse in trace),
                      "reverse_root_edits": sum(not r["action"].path for r in rows),
                      "same_as_source": final.id == start.id, "generation_and_scoring_seconds": time.perf_counter() - started}
            if "future" in panels:
                _, forward = evaluator.segment(final, panels["future"])
                direction = metrics.get("direction", 1)
                result["forward_ic"] = direction * forward["ic"]
                result["forward_valid"] = (proposal["new"] and forward["reliable"] and
                                           direction * forward["ic"] >= .02 and direction * forward["rank_ic"] >= .01)
            append_json(temporary, result)
    os.replace(temporary, path)
    return str(path)


def probe():
    jobs = [(seed, task, kind, perturbation) for seed in (0, 1, 2)
            for task in ("rolling_0", "rolling_7", "dev_0", "dev_1")
            for kind in ("syntax", "trained") for perturbation in (0, 2)]
    with ProcessPoolExecutor(max_workers=8) as executor:
        for path in executor.map(probe_job, jobs):
            print(json.dumps({"event": "probe_completed", "path": path}), flush=True)


def train_job(job):
    seed, arm = job
    config = copy.deepcopy(protocol())
    config["seed"] = seed
    config["upper"]["learning_rate"] = .0001 if arm == "upper_control" else .001
    config["upper"]["joint_rollback"] = arm == "joint_rollback"
    config["upper"]["joint_iterations"] = 16 if arm.startswith("joint_") else 0
    archive_arm = arm in {"upper_archive_control", "upper_archive"}
    if archive_arm:
        config["upper"].update(iterations=8, archive_context=arm == "upper_archive")
        config["upper_optimizer_reset"] = True
        config["runtime_implementation_hash"] = digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                       for path in (PROJECT / "single_factor/hierarchical").glob("*.py")})
    parent_path = PARENT / "runs" / f"training_seed_{seed}" / "best_upper.pt"
    config["warm_start"] = {"path": str(parent_path), "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest()}
    config["fingerprint"] = digest(config)
    output = ROOT / "runs" / arm / f"training_seed_{seed}"
    with run_lock(output):
        campaign = Campaign(config, output, resume=(output / "training_state.pt").exists())
        try:
            if "warm_start" not in campaign.state:
                bundle = torch.load(parent_path, map_location="cpu", weights_only=False)
                campaign.lower.load_state_dict(bundle["lower"])
                campaign.upper.load_state_dict(bundle["upper"])
                campaign.lower_optimizer.load_state_dict(bundle["lower_optimizer"])
                if not archive_arm:
                    campaign.upper_optimizer.load_state_dict(bundle["upper_optimizer"])
                for group in campaign.upper_optimizer.param_groups:
                    group["lr"] = config["upper"]["learning_rate"]
                campaign.rng = random.Random(39171 + seed)
                torch.manual_seed(39171 + seed)
                campaign.state.update(phase="joint" if arm.startswith("joint_") else "upper", lower_iteration=128,
                                      warm_start=config["warm_start"])
                campaign.commit()
            campaign.run()
            recovery = json.loads((output / "recovery_manifest.json").read_text()) if (output / "recovery_manifest.json").exists() else {}
            save_json(output / "cost.json", {"additional_wall_seconds": campaign.elapsed(), "seed": seed, "arm": arm,
                      "discarded_elapsed_seconds_estimate": recovery.get("discarded_elapsed_seconds_estimate", 0.)})
            return {"seed": seed, "arm": arm, "phase": campaign.state["phase"]}
        finally:
            if campaign.collector:
                campaign.collector.close()


def train(arms, seeds):
    # Three independent learners, each with three single-thread collectors.
    for arm in arms:
        with ProcessPoolExecutor(max_workers=3) as executor:
            for result in executor.map(train_job, [(seed, arm) for seed in seeds]):
                append_json(ROOT / "training_completion.jsonl", result)
                print(json.dumps(result), flush=True)


def bandit_sanity():
    """Known, dense reward on real observations isolates optimizer/encoder capacity."""
    torch.set_num_threads(1)
    results = []
    for seed in (0, 1, 2):
        saved = torch.load(PARENT / "runs" / f"training_seed_{seed}" / "rollouts/upper_0.pt",
                           map_location="cpu", weights_only=False)
        observations = [row["observation"] for row in saved["rows"][:64]]
        for rate in (.0001, .001):
            _, upper = restore_models(saved["models"])
            optimizer = torch.optim.Adam(upper.parameters(), lr=rate)
            rng = random.Random(2819 + seed)
            torch.manual_seed(2819 + seed)

            def rewards(observation):
                values = np.array([s[0] for s in observation.statistics] + [-1.])
                # Ties receive identical reward, independent of shortlist order.
                return torch.tensor([np.mean(values < value) for value in values], dtype=torch.float32)

            def measure():
                with torch.no_grad():
                    return average(float((upper.distribution(o)[0].probs * rewards(o)).sum()) for o in observations)

            initial = measure()
            started = time.perf_counter()
            for iteration in range(64):
                rows = []
                with torch.no_grad():
                    for _ in range(64):
                        observation = rng.choice(observations)
                        row = upper.sample(observation)
                        distribution, _ = upper.distribution(observation)
                        outcomes = rewards(observation)
                        expected = float((distribution.probs * outcomes).sum())
                        reward = float(outcomes[row["index"]])
                        row.update(reward=reward, return_=reward, advantage=reward - expected,
                                   value_target=expected)
                        row["return"] = reward
                        rows.append(row)
                metrics = update(upper, optimizer, rows, rng=rng, epochs=2, batch_size=64, clip=.1,
                                 entropy_weight=.005, target_kl=.01)
                if (iteration + 1) % 16 == 0:
                    append_json(ROOT / "bandit_curve.jsonl", {"seed": seed, "learning_rate": rate,
                                "iteration": iteration + 1, "expected_known_reward": measure(), **metrics})
            result = {"seed": seed, "learning_rate": rate, "initial_known_reward": initial,
                      "final_known_reward": measure(), "seconds": time.perf_counter() - started,
                      "interpretation": "capacity sanity with dense known reward, not financial performance"}
            results.append(result)
            save_json(ROOT / "bandit_sanity.json", results)
            print(json.dumps(result), flush=True)


def evaluate_job(job):
    from single_factor.hierarchical.experiments import future_check, original_ppo
    job_started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed, task_name, method, sampling_seed, timed = job
    mode = "timed" if timed else "fixed"
    output = ROOT / "trials" / mode / f"{method}_{seed}_{task_name}_{sampling_seed}.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    config = protocol()
    task = next(t for t in config["tasks"] if t["name"] == task_name)
    if method in {"upper_control", "upper_fast", "joint_rollback", "joint_accumulate", "upper_archive_control", "upper_archive"}:
        path = ROOT / "runs" / method / f"training_seed_{seed}" / ("joint_final.pt" if method.startswith("joint") else "upper_frozen_final.pt")
    else:
        path = PARENT / "runs" / f"training_seed_{seed}" / "best_upper.pt"
    lower, upper = restore_models(torch.load(path, weights_only=False, map_location="cpu"))
    if method == "syntax_random":
        lower.load_state_dict(torch.load(config["initializer_checkpoint"], weights_only=False, map_location="cpu")["lower"])
    panels = load_task(task, config["target_horizon"], config["lookback"], config["future_days"], with_future=True)
    initial = [Node.from_dict(row["ast"]) for row in records(config["initial_library"])]
    evaluator = evaluator_for(lower.grammar, panels, config).for_episode(initial)
    # Identical public cache preparation is outside the online search budget.
    # Archive construction, generated candidates and all decisions stay inside.
    for tree in initial:
        evaluator.get(tree)
    archive = Archive(evaluator, initial, config["search"])
    initial_report = {"discoveries": [], "pool_asts": [t.to_dict() for t in archive.pool.stats.trees],
                      "pool_weights": archive.pool.score.weights}
    initial_forward = future_check(initial_report, evaluator, panels["future"])["pool_ic"]
    search = {**config["search"], "seconds": 45. if timed else 300.,
              "max_attempts": 100000 if timed else 32, "budget_mode": "wall_seconds" if timed else "attempts"}
    torch.manual_seed(sampling_seed)
    trace = output.with_suffix(".jsonl")
    if method == "original_rpn":
        sys.path.insert(0, str(PROJECT / "backend"))
        sys.path.insert(0, str(PROJECT / "quantops"))
        report = original_ppo(evaluator, initial, search, seed * 100000 + sampling_seed, trace)
        report["rpn_training_seed"] = seed * 100000 + sampling_seed
    else:
        scheduler = method if method in {"random", "quality", "ucb"} else "random" if method == "syntax_random" else "ppo"
        report = episode(lower, upper, evaluator, initial, search, random.Random(sampling_seed), scheduler=scheduler,
                         output=trace, tag=output.stem, record_lower=False)[3]
    final = future_check(report, evaluator, panels["future"])
    valid = sum(f["reliable"] and f["coverage"] >= .8 and f["ic"] >= .02 and f["rank_ic"] >= .01 for f in final["single_factors"])
    report.update(model_seed=seed, task=task_name, method=method, sampling_seed=sampling_seed,
                  budget_mode=mode, future=final, forward_valid=valid, initial_forward_ic=initial_forward,
                  forward_pool_gain=final["pool_ic"] - initial_forward, checkpoint=str(path),
                  checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  full_trial_wall_seconds=time.perf_counter() - job_started,
                  interpretation="reused development slices, not a blind test")
    save_json(output, report)
    return report


def evaluate(timed=False):
    methods = (["random", "original_rpn", "upper_fast", "joint_accumulate", "upper_archive_control", "upper_archive"] if timed else
               ["syntax_random", "random", "quality", "ucb", "parent", "upper_control", "upper_fast", "joint_rollback", "joint_accumulate", "upper_archive_control", "upper_archive"])
    jobs = [(seed, task, method, sampling_seed, timed) for seed in (0, 1, 2) for task in ("dev_0", "dev_1")
            for sampling_seed in ([101] if timed else [101, 202]) for method in methods
            if method != "syntax_random" or seed == 0]
    random.Random(71923).shuffle(jobs)
    if timed:
        iterator = map(evaluate_job, jobs)
        for report in iterator:
            print(json.dumps({k: report[k] for k in ("method", "model_seed", "task", "new_factors", "forward_valid", "forward_pool_gain")}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=6) as executor:
            for report in executor.map(evaluate_job, jobs):
                print(json.dumps({k: report[k] for k in ("method", "model_seed", "task", "new_factors", "forward_valid", "forward_pool_gain")}), flush=True)


def data_audit():
    import csv
    config = protocol()
    cache = Path(config["tasks"][0]["fit"]["cache"])
    returns = np.load(cache / "reference_daily_return.npy")
    labels = np.load(cache / "label_h20.npy")
    rng = random.Random(1297)
    errors = []
    for _ in range(1000):
        day, stock = rng.randrange(len(labels) - 20), rng.randrange(labels.shape[1])
        expected = np.prod(1 + returns[day + 1:day + 21, stock]) - 1
        if np.isfinite(expected) and np.isfinite(labels[day, stock]):
            errors.append(abs(expected - labels[day, stock]))
    result = {"label_samples": len(errors), "label_chain_max_error": max(errors),
              "label_finite_fraction": float(np.isfinite(labels[:-20]).mean()),
              "label_quantiles": np.nanquantile(labels[:-20], [.001, .01, .5, .99, .999]).tolist(),
              "stock_count": labels.shape[1], "days": labels.shape[0],
              "point_in_time_universe_certified": False, "dividend_reinvestment_certified": False,
              "features": {}}
    for field in config["grammar"]["fields"]:
        values = np.load(cache / f"{field}.npy", mmap_mode="r")
        result["features"][field] = {"finite_fraction": float(np.isfinite(values).mean()),
                                     "zero_fraction": float((values == 0).mean())}
    dates = np.load(cache / "dates.npy").astype("datetime64[D]")
    turnover = np.load(cache / "TurnoverRate.npy")
    volume = np.load(cache / "volume.npy")
    from single_factor.prepare_budget_study import missing_turnover
    _, missing = missing_turnover(turnover, volume)
    indices = np.flatnonzero(missing)
    starts = indices[np.r_[True, np.diff(indices) > 1]]
    ends = indices[np.r_[np.diff(indices) > 1, True]]
    result["turnover_unavailable_periods"] = [{"start": str(dates[first]), "end": str(dates[last]), "days": int(last - first + 1)}
                                              for first, last in zip(starts, ends)]
    stratified = defaultdict(list)
    for row in csv.DictReader((PARENT / "all_discovered_factors.csv").open(encoding="utf-8-sig")):
        if row["role"] == "test":
            stratified["uses_turnover" if "TurnoverRate" in row["expression"] else "no_turnover"].append(row)
    result["historical_test_records_by_field"] = {
        group: {"records": len(rows), "forward_ic": average(float(r["future_ic"]) for r in rows if r["future_ic"]),
                "forward_valid": sum(r["future_qualified"].lower() in {"true", "1", "1.0"} for r in rows)}
        for group, rows in stratified.items()}
    save_json(ROOT / "data_audit.json", result)
    print(json.dumps(result), flush=True)


def repair_data():
    from single_factor.prepare_budget_study import missing_turnover
    config = protocol()
    source = Path(config["tasks"][0]["fit"]["cache"])
    target = ROOT / "data_missingness_reference"
    if (target / "manifest.json").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "manifest.json":
            shutil.copy2(path, target / path.name)
    values, missing = missing_turnover(np.load(source / "TurnoverRate.npy"), np.load(source / "volume.npy"))
    np.save(target / "TurnoverRate.npy", values.astype(np.float32))
    dates = np.load(source / "dates.npy").astype("datetime64[D]")
    metadata = json.loads((source / "manifest.json").read_text())
    metadata.update(missingness_base=str(source), unavailable_turnover_dates=[str(day) for day in dates[missing]],
                    missingness_policy="whole-market zero turnover with positive volume marked NaN, no imputation")
    save_json(target / "manifest.json", metadata)
    save_json(target / "repair_manifest.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in target.iterdir() if path.is_file()})
    return target


def locked_missingness_job(path):
    from single_factor.hierarchical.experiments import future_check
    torch.set_num_threads(1)
    path = Path(path)
    output = ROOT / "locked_missingness" / (path.parents[1].name + "_" + path.name)
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    config = json.loads((path.parents[1] / "resolved_config.json").read_text())
    report = json.loads(path.read_text())
    task = next(t for t in config["tasks"] if t["name"] == report["task"])
    grammar = Grammar(**config["grammar"])
    panels = load_task(task, config["target_horizon"], config["lookback"], config["future_days"], with_future=True)
    evaluator = evaluator_for(grammar, panels, config).for_episode([])
    for tree in report["pool_asts"]:
        evaluator.get(Node.from_dict(tree))
    original = future_check(report, evaluator, panels["future"])
    error = abs(original["pool_ic"] - report["future"]["pool_ic"])
    if error > 1e-12:
        raise AssertionError(f"locked historical future replay mismatch {path}: {error}")
    cleaned = load_slice({**task["future"], "cache": str(ROOT / "data_missingness_reference")},
                         config["target_horizon"], config["lookback"], config["future_days"])
    corrected = future_check(report, evaluator, cleaned)

    def valid(result):
        return sum(f["reliable"] and f["coverage"] >= .8 and f["ic"] >= .02 and f["rank_ic"] >= .01
                   for f in result["single_factors"])

    result = {"source": str(path), "seed": config["seed"], "task": task["name"], "method": report["method"],
              "replay_error": error, "old_pool_ic": original["pool_ic"], "clean_pool_ic": corrected["pool_ic"],
              "old_pool_reliable": original["pool_reliable"], "clean_pool_reliable": corrected["pool_reliable"],
              "old_valid": valid(original), "clean_valid": valid(corrected),
              "turnover_pool_members": sum("TurnoverRate" in str(tree) for tree in report["pool_asts"]),
              "formulas_weights_and_directions_locked": True}
    save_json(output, result)
    return result


def missingness_effect():
    repair_data()
    paths = sorted((PARENT / "runs").glob("training_seed_*/trials/test_*.json"))
    results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(locked_missingness_job, paths):
            results.append(result)
    save_json(ROOT / "locked_missingness_summary.json", {
        "trials": len(results), "maximum_historical_replay_error": max(r["replay_error"] for r in results),
        "mean_old_ic": average(r["old_pool_ic"] for r in results), "mean_clean_ic": average(r["clean_pool_ic"] for r in results),
        "old_reliable_pools": sum(r["old_pool_reliable"] for r in results),
        "clean_reliable_pools": sum(r["clean_pool_reliable"] for r in results),
        "old_valid_records": sum(r["old_valid"] for r in results), "clean_valid_records": sum(r["clean_valid"] for r in results),
        "interpretation": "fixed historical outputs under corrected missingness, unavailable predictions require reliability checks"})


def summarize():
    import csv
    groups = defaultdict(list)
    for path in (ROOT / "probe").glob("*.jsonl"):
        for row in records(path):
            groups[(row["task"], row["kind"], row["perturbation_steps"])].append(row)
    probes = []
    for key, rows in sorted(groups.items()):
        source = defaultdict(list)
        for row in rows:
            source[row["source"]].append(float(row["qualified_novel"]))
        means = [average(values) for values in source.values()]
        probes.append({"task": key[0], "kind": key[1], "perturbation_steps": key[2], "draws": len(rows),
                       "qualified_novel_fraction": average(r["qualified_novel"] for r in rows),
                       "duplicate_fraction": average(r["duplicate"] for r in rows),
                       "lower_reward": average(r["metrics"]["lower_reward"] for r in rows),
                       "forward_valid_fraction": average(r.get("forward_valid") for r in rows),
                       "mean_forward_ic": average(r.get("forward_ic") for r in rows),
                       "root_corruption_fraction": average(r["corruption_root_edits"] > 0 for r in rows),
                       "root_reverse_fraction": average(r["reverse_root_edits"] > 0 for r in rows),
                       "same_as_source_fraction": average(r["same_as_source"] for r in rows),
                       "source_success_rate_std": float(np.std(means)),
                       "mean_seconds": average(r["generation_and_scoring_seconds"] for r in rows)})
    save_json(ROOT / "probe_summary.json", probes)
    trials = []
    for path in (ROOT / "trials").glob("*/*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        trials.append({"budget_mode": report["budget_mode"], "method": report["method"], "seed": report["model_seed"],
                       "task": report["task"], "sampling_seed": report["sampling_seed"], "new_factors": report["new_factors"],
                       "forward_valid": report["forward_valid"], "forward_pool_ic": report["future"]["pool_ic"],
                       "forward_pool_gain": report["forward_pool_gain"], "attempts": report["attempts"],
                       "wall_seconds": report["wall_seconds"]})
    if trials:
        with (ROOT / "all_trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in trials:
        grouped[(row["budget_mode"], row["method"])][row["seed"]].append(row)
    comparison = []
    for (mode, method), seeds in sorted(grouped.items()):
        comparison.append({"budget_mode": mode, "method": method, "training_seeds": len(seeds),
                           "independent_models": 1 if method == "syntax_random" else len(seeds),
                           "trials": sum(len(rows) for rows in seeds.values()),
                           **{metric: average(average(r[metric] for r in rows) for rows in seeds.values())
                              for metric in ("new_factors", "forward_valid", "forward_pool_ic", "forward_pool_gain", "attempts", "wall_seconds")}})
    save_json(ROOT / "comparison.json", comparison)
    print(json.dumps({"probe_groups": probes, "comparison": comparison}), flush=True)


def plot_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    rows = list(csv.DictReader((ROOT / "all_trials.csv").open(encoding="utf-8-sig")))
    methods = ["random", "original_rpn", "upper_fast", "joint_accumulate", "upper_archive_control", "upper_archive"]
    labels = ["Random + lower", "Original RPN PPO", "Upper PPO, lr 0.001", "Joint PPO, accumulate", "Archive control", "Archive context"]
    metrics = [("new_factors", "Search-qualified discoveries"), ("forward_valid", "Forward-qualified discoveries"),
               ("forward_pool_gain", "Forward pool IC gain vs initial")]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for axis, (metric, label) in zip(axes, metrics):
        means = []
        for index, method in enumerate(methods):
            selected = [r for r in rows if r["budget_mode"] == "timed" and r["method"] == method]
            seeds = sorted({int(r["seed"]) for r in selected})
            values = [average(float(r[metric]) for r in selected if int(r["seed"]) == seed) for seed in seeds]
            means.append(average(values))
            axis.scatter([index] * len(values), values, color="black", s=22, zorder=3)
        axis.bar(range(len(methods)), means, color=["#99b8d1", "#c3ced8", "#6b9c88", "#bf9d6a"], alpha=.85)
        axis.set_xticks(range(len(methods)), labels, rotation=22, ha="right", fontsize=8)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
        if metric == "forward_pool_gain":
            axis.axhline(0, color="#555", linewidth=.8)
    figure.suptitle("45-second serial search on reused development periods")
    figure.text(.5, .01, "Dots are paired seed-group means. Forward periods were used for checkpoint selection.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, .95))
    figure.savefig(ROOT / "mechanism_results.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "literature", "prepare", "benchmark", "probe", "train", "bandit", "evaluate", "timed", "data", "repair", "missingness", "summarize", "plot"))
    parser.add_argument("--arms", nargs="+", choices=("upper_control", "upper_fast", "joint_rollback", "joint_accumulate", "upper_archive_control", "upper_archive"),
                        default=["upper_control", "upper_fast", "joint_rollback", "joint_accumulate"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    {"audit": historical_audit, "literature": literature, "prepare": prepare,
     "benchmark": benchmark, "probe": probe, "train": lambda: train(args.arms, args.seeds), "bandit": bandit_sanity,
     "evaluate": evaluate, "timed": lambda: evaluate(timed=True), "data": data_audit, "repair": repair_data,
     "missingness": missingness_effect, "summarize": summarize, "plot": plot_results}[args.command]()


if __name__ == "__main__":
    main()
