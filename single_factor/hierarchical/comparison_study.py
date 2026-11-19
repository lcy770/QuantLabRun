"""五组实验的依赖图、统一预算、配对汇总及跨机器配置准备。"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from .comparison_data import design_tasks, validate_cache
from .io import digest, save_json
from .selection_config import load_selection_config


REPOSITORY = Path(__file__).resolve().parents[2]
TEMPLATES = REPOSITORY / "experiments" / "dual_ppo"


def experiment_nodes():
    """按用户五组实验声明真实方法和可复用依赖，数量扣分消融已按确认删除。"""
    nodes = {
        "syntax": {"group": "shared", "generator": "diffusion_ppo", "syntax": True},
        "g_rpn_original": {"group": "generation", "generator": "rpn_original", "lower": True},
        "g_rpn_unified": {"group": "generation", "generator": "rpn_unified", "lower": True},
        "g_random": {"group": "generation", "generator": "random_edit"},
        "g_pretrained": {"group": "generation", "generator": "pretrained_diffusion", "generator_source": "syntax"},
        "g_diffusion": {"group": "generation", "generator": "diffusion_ppo", "generator_source": "syntax", "lower": True},
        "g_gp": {"group": "generation", "generator": "gp", "lower": True},
    }
    for method in ("random", "quality", "forward_exchange", "linucb", "double_dqn", "upper_ppo"):
        nodes[f"s_{method}"] = {"group": "selection", "generator": "diffusion_ppo", "generator_source": "g_diffusion",
                               "selector": method, "upper": method in {"linucb", "double_dqn", "upper_ppo"},
                               "shared_pools": True}
    for name, generator in (("rpn", "rpn_original"), ("random", "random_edit"), ("gp", "gp")):
        source = {"rpn": "g_rpn_original", "random": "g_random", "gp": "g_gp"}[name]
        nodes[f"p_{name}"] = {"group": "pairing", "generator": generator, "generator_source": source,
                              "selector": "upper_ppo", "upper": True, "shared_pools": True}
    nodes.update({
        "p_old_dual": {"group": "pairing", "legacy": True, "generator_source": "syntax"},
        "p_new_classical": {"group": "pairing", "alias": "g_diffusion"},
        "p_new_dual": {"group": "pairing", "alias": "s_upper_ppo"},
        "a_full": {"group": "reward", "alias": "s_upper_ppo"},
        "a_no_stability": {"group": "reward", "generator": "diffusion_ppo", "generator_source": "g_diffusion",
                           "selector": "upper_ppo", "upper": True, "shared_pools": True, "beta": 0.0},
        "a_no_quality": {"group": "reward", "generator": "diffusion_ppo", "generator_source": "g_diffusion",
                         "selector": "upper_ppo", "upper": True, "shared_pools": True, "gamma": 0.0},
        "c_lower": {"group": "continuation", "generator": "diffusion_ppo", "generator_source": "s_upper_ppo",
                    "selector_source": "s_upper_ppo", "selector": "upper_ppo", "continuation": "lower", "shared_pools": True},
        "c_upper": {"group": "continuation", "generator": "diffusion_ppo", "generator_source": "s_upper_ppo",
                    "selector_source": "s_upper_ppo", "selector": "upper_ppo", "continuation": "upper", "shared_pools": True},
        "c_alternating": {"group": "continuation", "generator": "diffusion_ppo", "generator_source": "s_upper_ppo",
                          "selector_source": "s_upper_ppo", "selector": "upper_ppo", "continuation": "alternating", "shared_pools": True},
        "w_equal_ridge": {"group": "weighting", "alias": "s_upper_ppo"},
    })
    return nodes


def resolve_nodes(names=None, groups=None):
    """按依赖拓扑排序，选择部分组时自动补齐必要训练来源。"""
    definitions = experiment_nodes()
    requested = names or [name for name, node in definitions.items() if node["group"] in (groups or
                        ["generation", "selection", "pairing", "reward", "continuation", "weighting"])]
    if not set(requested) <= definitions.keys():
        raise ValueError(f"unknown nodes {sorted(set(requested) - definitions.keys())}")
    ordered, visited = [], set()

    def visit(name):
        """先提交来源节点，再提交当前节点。"""
        if name in visited:
            return
        visited.add(name)
        node = definitions[name]
        for dependency in (node.get("alias"), node.get("generator_source"), node.get("selector_source")):
            if dependency:
                visit(dependency)
        ordered.append(name)

    for name in requested:
        visit(name)
    return ordered, requested


def prepare_protocol(cache, output, profile="short", device="auto", tasks=None, *, training_start=None, cutoff=None,
                     width=None, candidate_capacity=None, pool_attempts=None):
    """生成冻结协议，记录模型宽度、候选池容量和生成预算。"""
    output = Path(output).resolve()
    if (output / "protocol.json").exists():
        raise FileExistsError("protocol already exists; use it unchanged or choose a new output")
    profiles = json.loads((TEMPLATES / "profiles.json").read_text(encoding="utf-8"))
    settings = profiles[profile]
    for name, value in (("width", width), ("candidate_capacity", candidate_capacity), ("pool_attempts", pool_attempts)):
        if value is not None:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            settings[name] = value
    config = json.loads((TEMPLATES / "base.json").read_text(encoding="utf-8"))
    cache = Path(cache).resolve()
    if tasks is None:
        tasks, dates = design_tasks(cache, start=training_start, test_cutoff=cutoff)
        coverage = validate_cache(cache, tasks, config["lookback"], config["target_horizon"], config["grammar"]["fields"])
    else:
        if training_start or cutoff:
            raise ValueError("explicit task dates cannot be combined with rolling date overrides")
        tasks = copy.deepcopy(tasks)
        dates = {"mode": "explicit_historical_or_test_protocol", "training_tasks": sum(t["role"] == "train" for t in tasks),
                 "comparison_uses_training_data": not any(t["role"] == "development" for t in tasks),
                 "independent_test_available": any(t["role"] == "test" for t in tasks)}
        coverage = {"cache": str(cache), "mode": "explicit_tasks"}
    config.update(tasks=tasks, device=device, width=settings["width"], rollout_workers=1)
    config["initial_library"] = str((TEMPLATES / config["initial_library"]).resolve())
    config["corpus"].update(count=settings["corpus_count"], noise_steps=settings["noise_steps"])
    config["pretrain"]["epochs"] = settings["pretrain_epochs"]
    config["lower"].update(iterations=settings["lower_iterations"], batch_formulas=settings["batch_formulas"],
                           diffusion_steps=settings["noise_steps"], perturbation_steps=settings["noise_steps"])
    config["upper"].update(iterations=settings["upper_iterations"], episodes_per_batch=settings["episodes_per_batch"],
                           max_members=settings["max_members"], pool_attempts=settings["pool_attempts"], joint_iterations=0)
    config["search"].update(candidate_capacity=settings["candidate_capacity"], max_attempts=settings["attempt_budget"])
    for layer in ("lower", "upper"):
        config[layer]["ppo"]["epochs"] = settings["ppo_epochs"]
    config["autoregressive"] = {"n_steps": settings["rpn_steps"], "batch_size": min(64, settings["rpn_steps"]),
                                "epochs": settings["ppo_epochs"], "width": settings["rpn_width"]}
    config["upper"]["double_dqn"] = {"gradient_steps": settings["ppo_epochs"], "batch_size": 16,
        "target_update_interval": 10, "replay_capacity": 10000, "epsilon": 0.1, "discount": 1.0,
        "learning_rate": config["upper"]["learning_rate"]}
    config["portfolio"] = {"top_k": 50, "rebalance_days": 20, "initial_cash": 100000000.0,
                            "fee_rate": 0.0015, "slippage": 0.0005, "minimum_fee": 5.0}
    config["study_profile"] = settings
    config["evaluation"]["seeds"] = settings["evaluation_seeds"]
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    resolved = load_selection_config(output / "config.json")
    from .selection_evaluation import load_selection_task
    checked = {task["name"]: load_selection_task(task, resolved, with_feedback=task["role"] != "test") for task in tasks}
    evaluation_starts = [checked[task["name"]]["search"].dates.min() for task in tasks if task["role"] != "train"]
    if evaluation_starts and any(checked[task["name"]]["feedback"].ends.max() >= min(evaluation_starts)
                                 for task in tasks if task["role"] == "train"):
        raise ValueError("training feedback labels overlap development/test generation dates")
    if dates.get("comparison_uses_training_data"):
        dates.setdefault("comparison_information_cutoff", str(max(checked[task["name"]]["feedback"].ends.max()
                             for task in tasks if task["role"] == "train"))[:10])
    protocol = {"schema_version": 1, "profile": profile, "settings": settings, "dates": dates,
                "coverage": coverage, "config": resolved, "nodes": experiment_nodes(),
                "training_seeds": [0, 1, 2], "quantity_penalty": False}
    protocol["fingerprint"] = digest(protocol)
    save_json(output / "protocol.json", protocol)
    return protocol


def summarize_study(output, definitions=None, split=None):
    """先合并同一训练种子的搜索重复，再计算三训练种子的配对差和区间。"""
    output = Path(output)
    if split is None:
        manifest = output / "manifest.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        split = metadata.get("comparison_split", "development")
    if split not in {"training", "development", "test"}:
        raise ValueError("summary split must be training, development or test")
    prefix = "test_" if split == "test" else ""
    definitions = definitions or experiment_nodes()
    records = []
    for path in sorted(output.glob(f"seed_*/nodes/*/{split}/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        records.append(row)
    means = {}
    metric_names = ("utility", "ic", "rank_ic", "new_factors", "new_to_initial_library", "search_seconds",
                    "net_return", "net_excess_return", "maximum_drawdown", "member_count")
    for row in records:
        key = (row["node"], row["task"], row["budget"], row["training_seed"])
        means.setdefault(key, []).append(row)
    seed_rows, weight_rows = [], []
    for key, rows in means.items():
        seed_rows.append({"node": key[0], "task": key[1], "budget": key[2], "training_seed": key[3],
                          "evaluation_scope": split,
                          **{metric: float(np.mean([row[metric] for row in rows])) for metric in metric_names}})
        weighted = [row for row in rows if "weight_comparison" in row]
        if weighted:
            paired = {}
            for metric in ("utility", "ic", "net_return", "net_excess_return"):
                equal = float(np.mean([row[metric] for row in weighted]))
                ridge = float(np.mean([row["weight_comparison"]["ridge_score" if metric in {"utility", "ic"}
                    else "ridge_portfolio"][metric] for row in weighted]))
                paired.update({f"equal_{metric}": equal, f"ridge_{metric}": ridge,
                               f"ridge_minus_equal_{metric}": ridge - equal})
            weight_rows.append({"node": key[0], "task": key[1], "budget": key[2], "training_seed": key[3], **paired})
    comparisons = []
    rng = np.random.default_rng(20261116)
    baselines = {"generation": "g_diffusion", "selection": "s_upper_ppo", "pairing": "p_new_dual",
                 "reward": "a_full", "continuation": "c_alternating"}
    indexed = {(row["node"], row["task"], row["budget"], row["training_seed"]): row for row in seed_rows}
    # 别名共享真实模型，汇总时保留其组名，同时保持训练种子为统计单位。
    for name, node in definitions.items():
        if node.get("alias"):
            for row in list(seed_rows):
                if row["node"] == node["alias"]:
                    clone = {**row, "node": name}
                    seed_rows.append(clone)
                    indexed[(name, row["task"], row["budget"], row["training_seed"])] = clone
    for name, node in definitions.items():
        baseline = baselines.get(node["group"])
        if not baseline or name == baseline:
            continue
        contexts = {(r["task"], r["budget"]) for r in seed_rows if r["node"] == name}
        for task, budget in contexts:
            pairs = [(row, indexed[(baseline, task, budget, row["training_seed"])]) for row in seed_rows
                     if row["node"] == name and row["task"] == task and row["budget"] == budget
                     and (baseline, task, budget, row["training_seed"]) in indexed]
            if not pairs:
                continue
            for metric in metric_names:
                differences = np.array([a[metric] - b[metric] for a, b in pairs])
                interval = None
                if len(differences) > 1:
                    boot = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(1)
                    interval = np.quantile(boot, [.025, .975]).tolist()
                comparisons.append({"node": name, "baseline": baseline, "task": task, "budget": budget,
                    "metric": metric, "training_seed_differences": differences.tolist(),
                    "n_training_seeds": len(differences), "mean_difference": float(differences.mean()), "bootstrap_ci95": interval})
    save_json(output / f"{prefix}paired_differences.json", comparisons)
    save_json(output / f"{prefix}seed_means.json", seed_rows)
    for filename, rows, columns in (("comparison.csv", seed_rows, ["node", "task", "budget", "training_seed", "evaluation_scope", *metric_names]),
                                    ("weight_comparison.csv", weight_rows, list(weight_rows[0]) if weight_rows else
                                     ["node", "task", "budget", "training_seed"])):
        with (output / f"{prefix}{filename}").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    result = {f"{split}_reports": len(records), "trained_seed_means": len(seed_rows),
              "evaluation_scope": split, "evaluation_uses_training_data": split == "training",
              "statistical_unit": "independent_training_seed", "paired_comparisons": len(comparisons)}
    save_json(output / f"{prefix}summary.json", result)
    return result
