"""候选生成与集合选择架构的配置契约和数据/源码身份。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .ast import Grammar
from .io import digest


ARCHITECTURE = "candidate_selection"


def file_hash(path):
    """流式计算真实文件内容身份，避免仅信任缓存清单。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_selection_config(path):
    """读取并验证新架构配置，解析相对路径，固定全部奖励与任务身份。"""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError("expected candidate_selection architecture")
    defaults = {"seed": 0, "device": "cpu", "data_mode": "exploratory", "target_horizon": 20,
                "lookback": 100, "future_days": 20, "width": 32, "torch_threads": 2,
                "signal_cache_size": 64, "rollout_workers": 1, "record_rollouts": True}
    for name, value in defaults.items():
        config.setdefault(name, value)
    if config["device"] == "auto":
        import torch
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    for name in ("target_horizon", "lookback", "future_days", "width", "torch_threads", "signal_cache_size", "rollout_workers"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if config["data_mode"] not in {"exploratory", "formal"}:
        raise ValueError("data_mode must be exploratory or formal")
    grammar = Grammar(**config["grammar"])
    if grammar.max_lookback > config["lookback"] or config["future_days"] < config["target_horizon"]:
        raise ValueError("panel buffers do not support grammar/target horizon")
    reward = config.setdefault("reward", {})
    for name, value in {"thresholds": [0.02, 0.01, 0.10, 0.80], "kappa": 0.05,
                        "alpha": 50.0, "beta": 25.0, "gamma": 0.5, "eta": 0.0,
                        "member_segment": "feedback", "block_days": 10,
                        "reference_seconds": None, "calibration_samples": 8}.items():
        reward.setdefault(name, value)
    if len(reward["thresholds"]) != 4 or any(not math.isfinite(x) or x <= 0 for x in reward["thresholds"]):
        raise ValueError("four finite positive quality thresholds are required")
    if reward["thresholds"][3] > 1:
        raise ValueError("coverage threshold exceeds one")
    for name in ("kappa", "alpha", "beta", "gamma", "eta"):
        if not math.isfinite(reward[name]) or reward[name] < 0:
            raise ValueError(f"reward {name} must be finite and nonnegative")
    if reward["alpha"] <= 0 or reward["member_segment"] not in {"search", "feedback"}:
        raise ValueError("positive alpha and an explicit search/feedback H segment are required")
    if reward["reference_seconds"] is not None and (not math.isfinite(reward["reference_seconds"]) or reward["reference_seconds"] <= 0):
        raise ValueError("reference_seconds must be finite and positive")
    config.setdefault("initial_seeds", 16)
    config.setdefault("corpus", {"count": 144, "noise_steps": 4, "seed": 0})
    pretrain = config.setdefault("pretrain", {})
    for name, value in {"epochs": 0, "learning_rate": 0.001, "batch_size": 8}.items():
        pretrain.setdefault(name, value)
    if not math.isfinite(pretrain["learning_rate"]) or pretrain["learning_rate"] <= 0:
        raise ValueError("pretrain learning_rate must be finite and positive")
    for layer in ("lower", "upper"):
        settings = config.setdefault(layer, {})
        for name, value in {"iterations": 128 if layer == "lower" else 16, "learning_rate": 0.0003,
                            "gae_lambda": 1.0}.items():
            settings.setdefault(name, value)
        if type(settings["iterations"]) is not int or settings["iterations"] < 0:
            raise ValueError(f"invalid {layer} iterations")
        if not 0 <= settings["gae_lambda"] <= 1 or not math.isfinite(settings["gae_lambda"]):
            raise ValueError("GAE lambda must be finite and between zero and one")
        ppo = settings.setdefault("ppo", {})
        for name, value in {"epochs": 4, "batch_size": 16, "clip": 0.2, "entropy_weight": 0.01,
                            "value_weight": 0.5, "max_grad_norm": 1.0}.items():
            ppo.setdefault(name, value)
        if (type(ppo["epochs"]) is not int or ppo["epochs"] < 1 or type(ppo["batch_size"]) is not int
                or ppo["batch_size"] < 1 or not 0 < ppo["clip"] < 1):
            raise ValueError("invalid PPO epochs/batch_size/clip")
        for name in ("learning_rate",):
            if not math.isfinite(settings[name]) or settings[name] <= 0:
                raise ValueError(f"invalid {layer} {name}")
        for name in ("entropy_weight", "value_weight", "max_grad_norm"):
            if not math.isfinite(ppo[name]) or ppo[name] < 0:
                raise ValueError(f"invalid PPO {name}")
        if ppo.get("target_kl") is not None and (not math.isfinite(ppo["target_kl"]) or ppo["target_kl"] <= 0):
            raise ValueError("invalid PPO target_kl")
    lower, upper = config["lower"], config["upper"]
    lower.setdefault("batch_formulas", 16)
    lower.setdefault("shortlist_size", 32)
    lower.setdefault("diffusion_steps", config["corpus"]["noise_steps"])
    lower.setdefault("perturbation_steps", lower["diffusion_steps"])
    upper.setdefault("episodes_per_batch", 4)
    upper.setdefault("joint_iterations", 4)
    upper.setdefault("max_members", 10)
    upper.setdefault("pool_attempts", 128)
    search = config.setdefault("search", {})
    for name, value in {"candidate_capacity": 128, "max_attempts": 10000,
                        "seconds": 1800.0, "budget_mode": "wall_seconds"}.items():
        search.setdefault(name, value)
    evaluation = config.setdefault("evaluation", {})
    evaluation.setdefault("methods", ["upper_ppo", "random", "quality", "forward_selection"])
    evaluation.setdefault("seeds", [0])
    evaluation.setdefault("deterministic", False)
    if not evaluation["methods"] or not set(evaluation["methods"]) <= {"upper_ppo", "random", "quality", "forward_selection"}:
        raise ValueError("unknown/empty candidate selection evaluation methods")
    if not evaluation["seeds"] or any(type(seed) is not int or seed < 0 for seed in evaluation["seeds"]):
        raise ValueError("evaluation seeds must be nonnegative integers")
    if len(set(evaluation["seeds"])) != len(evaluation["seeds"]):
        raise ValueError("evaluation seeds must be distinct")
    if search["budget_mode"] not in {"wall_seconds", "attempts"} or not math.isfinite(search["seconds"]) or search["seconds"] <= 0:
        raise ValueError("invalid generation budget")
    if "target_count" in search and (type(search["target_count"]) is not int or search["target_count"] < 1):
        raise ValueError("target_count must be a positive integer")
    positive = [(config, "initial_seeds"), (reward, "block_days"), (reward, "calibration_samples"),
                (config["corpus"], "noise_steps"), (config["corpus"], "count"), (pretrain, "batch_size"),
                (lower, "batch_formulas"), (lower, "shortlist_size"), (lower, "diffusion_steps"),
                (upper, "episodes_per_batch"), (upper, "max_members"), (upper, "pool_attempts"),
                (search, "candidate_capacity"), (search, "max_attempts")]
    if any(type(section[name]) is not int or section[name] < 1 for section, name in positive):
        raise ValueError("positive integer size/horizon settings are required")
    for section, name in ((pretrain, "epochs"), (upper, "joint_iterations"), (lower, "perturbation_steps")):
        if type(section[name]) is not int or section[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if (lower["diffusion_steps"] > config["corpus"]["noise_steps"]
            or lower["perturbation_steps"] > config["corpus"]["noise_steps"]):
        raise ValueError("diffusion horizon exceeds the time embedding vocabulary")
    tasks = config["tasks"]
    names = [task["name"] for task in tasks]
    if len(set(names)) != len(names) or any(not name.replace("_", "").isalnum() for name in names):
        raise ValueError("task names must be unique path-safe identifiers")
    if not any(task["role"] == "train" for task in tasks):
        raise ValueError("at least one training task is required")
    identities = {}
    config["data_fingerprints"] = {}
    for task in tasks:
        if task["role"] not in {"train", "development", "test"}:
            raise ValueError("unknown task role")
        required = {"fit", "search", "validation"}
        if task["role"] in {"train", "development"}:
            required.add("feedback")
        if not required <= task.keys():
            raise ValueError(f"{task['name']}: F/G/A/R task sections are missing")
        for name in ("fit", "search", "validation", "feedback", "future"):
            if name not in task:
                continue
            spec = task[name]
            cache = Path(spec["cache"]).expanduser()
            cache = (path.parent / cache).resolve() if not cache.is_absolute() else cache.resolve()
            spec["cache"] = str(cache)
            if str(cache) not in identities:
                manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
                if not set(grammar.fields) <= set(manifest["fields"]):
                    raise ValueError("grammar fields missing from cache")
                if config["data_mode"] == "formal" and not manifest.get("data_conventions_verified", False):
                    raise ValueError("formal data conventions have not been verified")
                files = {"manifest.json", "dates.npy", "codes.json"}
                files.update(f"{field}.npy" for field in manifest["fields"])
                if manifest.get("label_file"):
                    files.add(manifest["label_file"])
                else:
                    files.add("close.npy")
                identities[str(cache)] = digest([(filename, file_hash(cache / filename)) for filename in sorted(files)])
            config["data_fingerprints"][f"{task['name']}.{name}"] = identities[str(cache)]
    for name in ("initial_library", "initializer_checkpoint"):
        if config.get(name):
            artifact = Path(config[name]).expanduser()
            artifact = (path.parent / artifact).resolve() if not artifact.is_absolute() else artifact.resolve()
            config[name] = str(artifact)
            config.setdefault("artifact_hashes", {})[name] = file_hash(artifact)
            if name == "initial_library":
                config["artifact_hashes"]["library_manifest"] = file_hash(artifact.parent / "library_manifest.json")
    root = Path(__file__).parent
    sources = {p.name: file_hash(p) for p in root.glob("*.py")}
    for name in ("data.py", "expression.py", "environment.py", "evaluator.py", "config.py",
                 "hierarchical_config.fullcheck.json", "_vendor/alphagen/config.py",
                 "_vendor/alphagen/data/expression.py", "_vendor/alphagen/data/tree.py",
                 "_vendor/alphagen/data/tokens.py", "_vendor/alphagen/rl/policy.py",
                 "_vendor/alphagen/utils/correlation.py"):
        sources[f"dependency:{name}"] = file_hash(root.parent / name)
    repository = root.parents[1]
    for directory in (repository / "quantops/quantops", repository / "backend/adapters"):
        for source in directory.glob("*.py"):
            sources[str(source.relative_to(repository))] = file_hash(source)
    for name in ("canonical_expression.py", "expression_parser.py"):
        source = repository / "backend/custom_bt" / name
        sources[str(source.relative_to(repository))] = file_hash(source)
    config["implementation_hash"] = digest(sources)
    config.pop("fingerprint", None)
    config["fingerprint"] = digest(config)
    return config
