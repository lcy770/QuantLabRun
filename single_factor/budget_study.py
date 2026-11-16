"""Preregister, run and analyze a bounded three-seed hierarchical-PPO study."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from .hierarchical.io import save_json, event, append_json


def configure(root):
    root = Path(root).resolve()
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "single_factor/hierarchical_config.gae_20261107.json").read_text())
    cache = root / "data_calendar_reference"
    dates = np.load(cache / "dates.npy").astype("datetime64[D]")
    def task(name, role, start, future=False):
        first = int(np.searchsorted(dates, np.datetime64(start)))
        def segment(begin, end):
            if begin < 100 or end + 20 >= len(dates):
                raise ValueError(f"insufficient causal context/labels for {name}")
            return {"cache": str(cache), "start": str(dates[begin]), "end": str(dates[end])}
        result = {"name": name, "role": role, "fit": segment(first - 140, first - 21),
                  "search": segment(first, first + 59)}
        if future:
            result["future"] = segment(first + 80, first + 139)
        return result
    config["tasks"] = [task(f"rolling_{i}", "train", start) for i, start in enumerate(
        ["2023-01-03", "2023-03-01", "2023-05-05", "2023-07-03", "2023-09-01", "2023-11-01", "2024-01-02", "2024-03-01"])]
    config["tasks"] += [task("dev_0", "development", "2024-07-01"), task("dev_1", "development", "2024-09-02"),
                        task("test_0", "test", "2025-01-06", True), task("test_1", "test", "2025-06-02", True)]
    config["lower"].update(iterations=128, batch_formulas=64, continuation_states=16)
    config["lower"]["ppo"].update(epochs=2, batch_size=64)
    config["upper"].update(iterations=48, episodes_per_batch=24, joint_iterations=6,
                           joint_guard=True, learning_rate=.0001, gae_lambda=.95)
    config["upper"]["ppo"].update(epochs=2, batch_size=64, clip=.1, target_kl=.01, entropy_weight=.005)
    config["search"].update(seconds=15., min_search_rank_ic=.01)
    config["upper_validation"] = {"tasks": ["dev_0", "dev_1"], "every": 8, "seconds": 180.,
                                  "max_attempts": 32, "episodes": 2}
    config["evaluation"] = {"seconds": 45., "seeds": [101, 202], "shuffle_methods": True, "methods": [
        "original_rpn", "rpn_pretrained", "random", "quality", "ucb", "upper_mc", "upper_ppo", "alternating_ppo"]}
    config["initial_library"] = str(root / "library/library.jsonl")
    config["signal_cache_dir"] = str(root / "signal_cache")
    config["torch_threads"] = 1
    config["rollout_workers"] = 12
    config["archive_checkpoints"] = True
    config["record_rollouts"] = True
    path = root / "protocol_config.json"
    save_json(path, config)
    save_json(root / "preregistration.json", {"created_at": datetime.now().astimezone().isoformat(),
        "config": config, "training_seeds": [0, 1, 2], "primary_methods": ["upper_ppo", "alternating_ppo", "ucb", "random", "rpn_pretrained"],
        "primary_outcomes": ["search-qualified nonduplicate count /45s", "future-qualified count /45s", "locked future pool IC"],
        "future_gate": {"directed_ic": .02, "directed_rank_ic": .01, "coverage": .8},
        "independence_unit": "independently trained seed; tasks and search repeats are nested, not extra trained models",
        "evaluation_order": "seeded shuffled methods within task/search-seed blocks",
        "retrospective": True, "prior_history_observed": True,
        "market_rules_not_certified": ["delisted history completeness", "cash/reinvestment interpretation of ycp"],
        "no_post_test_hyperparameter_changes": True})
    print(path)
    return path


def analyze(root):
    """Read every completed result, including failures/zeros, without reselection."""
    root = Path(root).resolve()
    runs = root / "runs"
    trials, stages = [], []
    for run in sorted(runs.glob("training_seed_*")):
        seed = int(run.name.rsplit("_", 1)[1])
        progress = run / "progress.json"
        if progress.exists():
            state = json.loads(progress.read_text())
            stages.append({"training_seed": seed, **state})
        for path in sorted((run / "trials").glob("*.json")):
            report = json.loads(path.read_text())
            future = report.get("future", {})
            singles = future.get("single_factors", [])
            good = [f for f in singles if f["reliable"] and f["coverage"] >= .8 and f["ic"] >= .02 and f["rank_ic"] >= .01]
            trials.append({"training_seed": seed, "sampling_seed": report["seed"], "task": report["task"],
                "role": report["role"], "method": report["method"], "search_count": report["new_factors"],
                "future_count": len(good) if "future" in report else None,
                "future_mean_ic": statistics.mean(f["ic"] for f in singles) if singles else None,
                "future_pool_ic": future.get("pool_ic"), "search_pool_ic": report["pool_ic"],
                "wall_seconds": report["wall_seconds"], "end_to_end_seconds": report["end_to_end_seconds"],
                "evaluation_calls": report["evaluator_calls"], "attempts": report["attempts"],
                "duplicate_rate": report.get("duplicate_rate"), "source": str(path)})
    test = [r for r in trials if r["role"] == "test"]
    by_method = []
    for method in sorted({r["method"] for r in test}):
        selected = [r for r in test if r["method"] == method]
        seed_rows = []
        for seed in sorted({r["training_seed"] for r in selected}):
            group = [r for r in selected if r["training_seed"] == seed]
            seed_rows.append({"seed": seed, **{k: statistics.mean(r[k] for r in group if r[k] is not None)
                for k in ("search_count", "future_count", "future_pool_ic", "wall_seconds", "end_to_end_seconds")}})
        by_method.append({"method": method, "trained_seeds": len(seed_rows), "by_seed": seed_rows,
                          **{k: statistics.mean(r[k] for r in seed_rows) for k in
                             ("search_count", "future_count", "future_pool_ic", "wall_seconds", "end_to_end_seconds")}})
    paired = []
    for candidate in ("upper_ppo", "alternating_ppo"):
        for baseline in ("random", "ucb", "rpn_pretrained", "upper_mc"):
            a = next((x for x in by_method if x["method"] == candidate), None)
            b = next((x for x in by_method if x["method"] == baseline), None)
            if not a or not b:
                continue
            first = {(r["training_seed"], r["task"], r["sampling_seed"]): r for r in test if r["method"] == candidate}
            second = {(r["training_seed"], r["task"], r["sampling_seed"]): r for r in test if r["method"] == baseline}
            matched = sorted(set(first) & set(second))
            ids = sorted({k[0] for k in matched})
            if not ids:
                continue
            for metric in ("search_count", "future_count", "future_pool_ic"):
                diffs = [statistics.mean(first[k][metric] - second[k][metric] for k in matched if k[0] == s) for s in ids]
                # Exact sign-flip randomization over independent trained seeds.
                observed = abs(statistics.mean(diffs))
                means = [abs(np.mean(np.array(diffs) * np.array([1 if mask & (1 << j) else -1 for j in range(len(diffs))])))
                         for mask in range(2 ** len(diffs))]
                paired.append({"candidate": candidate, "baseline": baseline, "metric": metric,
                    "per_seed_differences": diffs, "mean_difference": statistics.mean(diffs),
                    "matched_trials": len(matched),
                    "two_sided_seed_signflip_p": sum(x >= observed - 1e-12 for x in means) / len(means),
                    "small_seed_count_limits_inference": True})
    status_path = runs / "suite_progress.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    result = {"generated_at": datetime.now().astimezone().isoformat(), "suite": status,
              "completed_trial_count": len(trials), "stages": stages, "test_aggregate": by_method,
              "paired": paired, "trials": trials, "future_was_not_blind_to_prior_research": True,
              "universe_history_not_certified": True}
    save_json(root / "analysis.json", result)
    import csv
    with (root / "all_trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        if trials:
            writer = csv.DictWriter(handle, fieldnames=list(trials[0]));writer.writeheader();writer.writerows(trials)
    lines = ["# 12小时预算研究：真实结果", "", f"更新时间：{result['generated_at']}", "",
             f"状态：{status.get('phase', 'not_started')}；已完成试验：{len(trials)}。", "",
             "所有方法使用相同500只股票、因果复权参考数据、质量门槛和45秒搜索预算。采样重复先在每个训练种子内部平均。", "",
             "| 方法 | 独立训练种子 | 搜索期合格数 | 后续期达标数 | 后续组合IC |", "|---|---:|---:|---:|---:|"]
    for r in by_method:
        lines.append(f"| {r['method']} | {r['trained_seeds']} | {r['search_count']:.3f} | {r['future_count']:.3f} | {r['future_pool_ic']:.5f} |")
    if by_method:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        metrics = ("search_count", "future_count", "future_pool_ic")
        names = [r["method"].replace("_ppo", "").replace("rpn_pretrained", "RPN trained") for r in by_method]
        for ax, metric in zip(axes, metrics):
            ax.bar(range(len(by_method)), [r[metric] for r in by_method], color="#387aab", alpha=.7)
            for i, method in enumerate(by_method):
                values = [r[metric] for r in method["by_seed"]]
                ax.scatter([i] * len(values), values, color="#222222", s=16, zorder=3)
            ax.axhline(0, color="grey", linewidth=.7)
            ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
            ax.set_title(metric.replace("_", " "))
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Bars: mean; dots: independent training seeds (tasks/repeats averaged within seed)", fontsize=10)
        fig.savefig(root / "results.png", dpi=180)
        fig.savefig(root / "results.svg")
        plt.close(fig)
        lines += ["", "![各方法真实结果](results.png)", ""]
        lines += ["## 配对比较", "", "| 候选 | 对照 | 指标 | 各训练种子的差值 | 均差 | 精确符号翻转p值 |",
                  "|---|---|---|---|---:|---:|"]
        for r in paired:
            values = ", ".join(f"{v:.4g}" for v in r["per_seed_differences"])
            lines.append(f"| {r['candidate']} | {r['baseline']} | {r['metric']} | {values} | {r['mean_difference']:.5g} | {r['two_sided_seed_signflip_p']:.3f} |")
    lines += ["", "## 解释边界", "", "后续区间在此前研究中已被查看，本次为冻结协议的回顾性复验，不是新的封闭盲测。退市历史完整性和参考昨收的现金收益口径仍未认证，不能据此宣称可交易收益。", "",
              "完整每次试验、零发现、拒绝的交替更新、耗时和独立种子差异保留在analysis.json、all_trials.csv和runs目录。三个独立种子的显著性分辨率很低，完整报告符号随机化p值，不将采样重复充当独立训练。", ""]
    (root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _rpn_job(job):
    import torch
    from .hierarchical.training import Campaign
    from .hierarchical.experiments import prepare_rpn
    from .hierarchical.io import run_lock
    config, child = job
    with run_lock(child):
        campaign = Campaign(config, child, resume=True)
        try:
            torch.set_num_threads(1)
            prepare_rpn(campaign)
        finally:
            if campaign.collector:
                campaign.collector.close()
    return config["seed"]


def phased_suite(config, output, seeds):
    """Train/freeze policies, parallelize independent controls, then time serially."""
    from concurrent.futures import ProcessPoolExecutor
    import traceback
    from .hierarchical.__main__ import finalize_suite
    from .hierarchical.training import Campaign
    from .hierarchical.experiments import prepare_ablations, evaluate_campaign
    from .hierarchical.io import run_lock, digest, BudgetExpired, check_budget
    output = Path(output)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("training seeds must be distinct nonnegative integers")
    signature = digest([config["fingerprint"], seeds])
    with run_lock(output):
        manifest = output / "suite_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["configuration_hash"] != signature:
                raise ValueError("existing study code/data/config/seed signature differs")
        else:
            save_json(manifest, {"configuration_hash": signature, "training_seeds": seeds, "config": config})
        jobs, completed = [], []
        try:
            for seed in seeds:
                check_budget()
                child_config = copy.deepcopy(config)
                child_config["seed"] = seed
                child_config.pop("fingerprint", None)
                child_config["fingerprint"] = digest(child_config)
                child = output / f"training_seed_{seed}"
                save_json(output / "suite_progress.json", {"phase": "training", "current_training_seed": seed,
                          "completed_training_seeds": completed, "child_output": str(child.resolve())})
                with run_lock(child):
                    campaign = Campaign(child_config, child, resume=(child / "training_state.pt").exists())
                    campaign.run()
                    campaign.state.setdefault("training_wall_seconds", campaign.elapsed())
                    campaign.commit()
                    prepare_ablations(campaign, include_rpn=False)
                    campaign.commit()
                jobs.append((child_config, child))
                completed.append(seed)
            if "rpn_pretrained" in config["evaluation"]["methods"]:
                save_json(output / "suite_progress.json", {"phase": "rpn_controls", "completed_training_seeds": completed})
                with ProcessPoolExecutor(max_workers=min(3, len(jobs))) as pool:
                    for seed in pool.map(_rpn_job, jobs):
                        event("rpn_control_completed", training_seed=seed)
            rows = []
            for child_config, child in jobs:
                check_budget()
                seed = child_config["seed"]
                save_json(output / "suite_progress.json", {"phase": "evaluation", "current_training_seed": seed,
                          "completed_training_seeds": completed, "child_output": str(child.resolve())})
                with run_lock(child):
                    campaign = Campaign(child_config, child, resume=True)
                    try:
                        summary = evaluate_campaign(campaign)
                    finally:
                        if campaign.collector:
                            campaign.collector.close()
                for item in summary["aggregate"]:
                    rows.append({"task": item["task"], "method": item["method"], "training_seed": seed,
                                 **{name: value["mean"] for name, value in item["metrics"].items()}})
            return finalize_suite(config, output, seeds, rows)
        except BudgetExpired:
            result = {"status": "budget_stopped", "completed_training_seeds": completed, "resumable": True}
            save_json(output / "suite_progress.json", {"phase": "budget_stopped", **result})
            return result
        except BaseException as exc:
            save_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            save_json(output / "suite_progress.json", {"phase": "failed", "completed_training_seeds": completed})
            raise


def run(root, config_path, seeds, deadline):
    from .hierarchical.__main__ import load_config
    root = Path(root).resolve()
    deadline_time = datetime.fromisoformat(deadline)
    if deadline_time.tzinfo is None:
        raise ValueError("deadline must include its timezone")
    deadline_epoch = deadline_time.timestamp()
    os.environ["QUANTLAB_DEADLINE_EPOCH"] = str(deadline_epoch)
    config = load_config(config_path)
    started = time.time()
    try:
        result = phased_suite(config, root / "runs", seeds)
        event("bounded_study_finished", **result)
    finally:
        analyze(root)
        account = {"started_epoch": started, "finished_epoch": time.time(),
                   "wall_seconds": time.time() - started, "deadline_epoch": deadline_epoch}
        append_json(root / "wall_history.jsonl", account)
        history = [json.loads(s) for s in (root / "wall_history.jsonl").read_text().splitlines()]
        save_json(root / "wall_account.json", {**account, "total_session_wall_seconds": sum(x["wall_seconds"] for x in history)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--deadline", default="2026-11-08T13:45:00+08:00")
    args = parser.parse_args()
    if args.command == "configure":
        configure(args.output)
    elif args.command == "analyze":
        summary = analyze(args.output)
        print(json.dumps({"completed_trials": summary["completed_trial_count"], "suite": summary["suite"]}))
    else:
        if args.config is None:
            parser.error("run requires --config")
        run(args.output, args.config, args.seeds, args.deadline)
