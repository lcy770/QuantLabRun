"""复用原起点调度双 PPO 的真实训练和搜索，按公共协议评价其锁定组合。"""
from __future__ import annotations

from dataclasses import replace
import json
import random
from time import perf_counter

import torch

from .ast import Node

from .comparison_metrics import prediction_for, prediction_rank_ic, top50_metrics
from .io import digest, run_lock, save_json, save_torch
from .selection_search import CandidatePool


def run_legacy(study, name, seed):
    """共享结构初始化后，运行原起点调度、原拟合期下层奖励与原 Ridge 工作池。"""
    from .__main__ import load_config
    from .models import LowerPolicy, UpperPolicy
    from .training import Campaign, model_bundle
    from .comparison_study import REPOSITORY
    from .comparison_training import seed_all
    root = study.node_path(name, seed)
    source = study.saved_node("syntax", seed)
    identity = digest([study.protocol["fingerprint"], name, seed, source["identity"]])
    with run_lock(root):
        if (root / "checkpoint.pt").exists():
            saved = torch.load(root / "checkpoint.pt", weights_only=False)
            if saved["identity"] != identity:
                raise ValueError("legacy experiment identity changed")
            if saved["state"]["phase"] == "complete":
                return
        config = json.loads((REPOSITORY / "single_factor/hierarchical_config.fullcheck.json").read_text(encoding="utf-8"))
        for key in ("seed", "device", "target_horizon", "lookback", "future_days", "width", "torch_threads",
                    "signal_cache_size", "data_mode", "grammar", "initial_library", "corpus"):
            config[key] = seed if key == "seed" else study.config[key]
        config["tasks"] = [{"name": task["name"], "role": task["role"], "fit": task["fit"], "search": task["search"]}
                           for task in study.config["tasks"]]
        config["lower"].update(iterations=study.profile["lower_iterations"], batch_formulas=study.profile["batch_formulas"],
            time_budget_seconds=study.profile["lower_seconds"], representation="rpn", edit_backend="rpn",
            diffusion_steps=study.profile["noise_steps"], perturbation_steps=study.profile["noise_steps"],
            continuation_samples=1, continuation_states=0)
        config["upper"].update(iterations=study.profile["upper_iterations"], joint_iterations=0,
            episodes_per_batch=study.profile["episodes_per_batch"], representation="rpn",
            time_budget_seconds=study.profile["upper_seconds"])
        for layer in ("lower", "upper"):
            config[layer]["ppo"]["epochs"] = study.profile["ppo_epochs"]
        config["search"].update(pool_capacity=study.profile["max_members"], shortlist_size=study.config["lower"]["shortlist_size"],
            max_attempts=study.profile["batch_formulas"], budget_mode="attempts", seconds=study.profile["time_budget"],
            quality_threshold=study.config["reward"]["thresholds"][0], min_coverage=study.config["reward"]["thresholds"][3],
            min_search_rank_ic=study.config["reward"]["thresholds"][1])
        config["evaluation"].update(methods=["upper_ppo"], seconds=study.profile["time_budget"])
        seed_all(seed)
        lower = LowerPolicy(study.grammar, config["width"], config["corpus"]["noise_steps"], representation="rpn", edit_backend="rpn")
        upper = UpperPolicy(study.grammar, config["width"], representation="rpn")
        editor = {key[len("editor."):]: value for key, value in source["generator"]["policy"].items() if key.startswith("editor.")}
        lower.load_state_dict({key: editor[key] for key in lower.state_dict()})
        initializer = root / "syntax_initializer.pt"
        if not initializer.exists():
            save_torch(initializer, model_bundle(lower, upper, {"fingerprint": source["identity"]}))
        config["initializer_checkpoint"] = str(initializer.resolve())
        config["corpus_source"] = str(study.node_path("syntax", seed).resolve())
        path = root / "legacy_config.json"
        save_json(path, config)
        resolved = load_config(path)
        output = root / ("legacy_training" if study.comparison_split == "training" else "training")
        started = perf_counter()
        campaign = Campaign(resolved, output, resume=(output / "training_state.pt").exists())
        campaign.run()
        stage_times = campaign.state.get("stage_wall_seconds", {})
        bundle = torch.load(output / "joint_final.pt", map_location="cpu", weights_only=False)
        state = {"phase": "complete", "costs": {"generator_training": stage_times.get("lower", 0.0),
            "selector_training": stage_times.get("upper", 0.0), "total_training": perf_counter() - started,
            "shared_syntax": source["state"]["costs"]["syntax"]}, "generator_version": campaign.state["lower_iteration"],
            "selector_version": campaign.state["upper_iteration"]}
        save_torch(root / "checkpoint.pt", {"identity": identity, "state": state, "bundle": bundle, "config": resolved})
        save_json(root / "progress.json", {**state, "identity": identity, "architecture": "legacy_start_scheduler"})


def evaluate_legacy(study, name, seed, split, selection_identity=None):
    """保留旧策略决策和 F 段 Ridge 权重，在公共 R/future 上计算同一套评价指标。"""
    from .training import restore_models, evaluator_for
    from .evaluation import load_task
    from .search import episode
    from .comparison_training import seed_all, seed_number
    saved = study.saved_node(name, seed)
    tasks = study.evaluation_tasks(split)
    for task in tasks:
        for sampling_seed in study.profile["evaluation_seeds"]:
            for budget in ("attempts", "wall_seconds"):
                key = f"{task['name']}_{budget}_{sampling_seed}"
                root = study.node_path(name, seed) / split
                path = root / f"{key}.json"
                if path.exists():
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    if previous["model_identity"] != saved["identity"] or previous.get("selection_identity") != selection_identity:
                        raise ValueError("legacy evaluation identity differs")
                    continue
                root.mkdir(parents=True, exist_ok=True)
                lower, upper = restore_models(saved["bundle"], study.config["device"])
                old_config = saved["config"]
                panels = load_task({"fit": task["fit"], "search": task["search"]}, old_config["target_horizon"],
                                   old_config["lookback"], old_config["future_days"])
                evaluator = evaluator_for(study.grammar, panels, old_config).for_episode(study.initial)
                search = {**old_config["search"], "seconds": study.profile["time_budget"],
                          "budget_mode": budget, "max_attempts": study.profile["attempt_budget"] if budget == "attempts" else 10000000}
                trace = root / f"{key}.jsonl"
                trace.unlink(missing_ok=True)
                trial_seed = seed_number(seed, task["name"], budget, sampling_seed)
                seed_all(trial_seed)
                _, _, _, legacy = episode(lower, upper, evaluator, study.initial, search, random.Random(trial_seed),
                                          scheduler="ppo", output=trace, record_lower=False)
                scoring = study.evaluator(task, future=split == "test", scope=f"legacy_{seed}_{budget}_{sampling_seed}")
                records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
                credited = {row["attempt"] for row in records if row.get("event") == "credit" and row["credited"]}
                common_pool = CandidatePool(study.config["search"]["candidate_capacity"])
                for row in records:
                    if "ast" in row and row["attempt"] in credited:
                        candidate, eligible = scoring.assess(Node.from_dict(row["ast"]), f"legacy_{row['attempt']}")
                        common_pool.add({"candidate": candidate, "eligible": eligible})
                members = []
                for index, record in enumerate(legacy["pool_asts"]):
                    tree = Node.from_dict(record)
                    candidate = scoring.assess(tree, f"legacy_selected_{index}")[0]
                    members.append(replace(candidate, direction=evaluator.get(tree).metrics["direction"]))
                indices = tuple(range(len(members)))
                weights = list(legacy["pool_weights"])
                segment = "future" if split == "test" else "feedback"
                score = scoring.utility(members, indices, segment, weights)
                signal = prediction_for(scoring, members, indices, segment, weights)
                portfolio = top50_metrics(scoring, members, indices, segment, study.config["portfolio"], weights)
                initial = {tree.id for tree in study.initial}
                report = {"node": name, "group": "pairing", "training_seed": seed, "sampling_seed": sampling_seed,
                    "task": task["name"], "budget": budget, "model_identity": saved["identity"],
                    "evaluation_scope": split, "evaluation_uses_training_data": split == "training",
                    "selection_identity": selection_identity,
                    "architecture": "legacy_start_scheduler", "utility": score["utility"], "ic": score["ic"],
                    "rank_ic": prediction_rank_ic(signal, scoring.panels[segment].target) if members else 0.0,
                    "new_factors": len(common_pool.discoveries),
                    "new_to_initial_library": sum(c.tree.id not in initial for c in common_pool.discoveries),
                    "member_count": len(members), "selected_factors": [{"expression": c.tree.text(), "direction": c.direction,
                        "ast": c.tree.to_dict()} for c in members], "locked_weights": weights,
                    "search_seconds": legacy["wall_seconds"], "net_return": portfolio["net_return"],
                    "net_excess_return": portfolio["net_excess_return"], "maximum_drawdown": portfolio["maximum_drawdown"],
                    "score": score, "portfolio": portfolio, "generation": legacy, "costs": saved["state"]["costs"],
                    "legacy_uses_validation_features": False, "weights_refitted_on_evaluation": False}
                save_json(path, report)
