# 命令行入口，负责读取实验配置、核验数据和源码身份，再启动训练及评价。
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from .io import event, run_lock, save_json, digest, BudgetExpired, check_budget


def load_config(path):
    # 配置决定编码器、编辑后端、奖励分支、训练规模和时间任务，需与结果一起保存。
    """读取实验配置，校验参数、任务划分和数据约定，返回解析后的配置字典。

    相对路径按配置文件所在目录展开，并对实际数据、源码和初始化文件计算指纹。
    后续恢复训练使用这些指纹检查实验身份是否一致。
    """
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("architecture") == "candidate_selection":
        from .selection_config import load_selection_config
        return load_selection_config(path)
    if config.get("architecture") not in {None, "start_scheduler"}:
        raise ValueError("unknown hierarchical architecture")
    config["device"] = config.get("device", "cpu")
    config["data_mode"] = config.get("data_mode", "exploratory")
    if config["data_mode"] not in {"exploratory", "formal"}:
        raise ValueError("data_mode must be exploratory or formal")
    if config["device"] == "auto":
        import torch
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    for key in ("target_horizon", "lookback", "future_days", "width", "torch_threads", "signal_cache_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["future_days"] < config["target_horizon"]:
        raise ValueError("future_days is shorter than target horizon")
    if config["grammar"]["max_lookback"] > config["lookback"]:
        raise ValueError("grammar lookback exceeds panel history")
    if not config.get("initial_library") and config["corpus"]["count"] < config["search"]["initial_seeds"]:
        raise ValueError("initial library exceeds corpus")
    if config["search"]["seconds"] <= 0 or config["evaluation"]["seconds"] <= 0:
        raise ValueError("search time budgets must be positive")
    if config["search"]["combination_weight"] < 0 or config["search"]["pair_gain_threshold"] <= 0:
        raise ValueError("combination weight must be nonnegative; novelty gain must be positive")
    if not any(t["role"] == "train" for t in config["tasks"]):
        raise ValueError("at least one historical training task is required")
    allowed_methods = {"original_rpn", "rpn_pretrained", "ast_random_init", "random", "quality", "productivity", "ucb", "upper_ppo", "alternating_ppo", "lambda_zero", "upper_mc"}
    if not set(config["evaluation"]["methods"]) <= allowed_methods:
        raise ValueError("unknown evaluation method")
    for section, names in {"corpus": ("count", "noise_steps"), "pretrain": ("epochs", "batch_size", "patience", "holdout_samples"),
                           "lower": ("iterations", "batch_formulas", "continuation_samples"),
                           "upper": ("iterations", "episodes_per_batch"),
                           "search": ("initial_seeds", "shortlist_size", "pool_capacity", "max_attempts")}.items():
        if any(type(config[section][name]) is not int or config[section][name] < 1 for name in names):
            raise ValueError(f"invalid positive integer setting in {section}")
    if config["upper"]["joint_iterations"] < 0 or config["lower"]["continuation_states"] < 0:
        raise ValueError("negative update/continuation counts")
    if config.get("rollout_workers", 1) < 1:
        raise ValueError("rollout_workers must be positive")
    if not 0 <= config["upper"].get("gae_lambda", 1.0) <= 1:
        raise ValueError("upper GAE lambda must be between zero and one")
    if config["upper"].get("joint_guard") and not config.get("upper_validation"):
        raise ValueError("guarded joint training requires development validation")
    if type(config["upper"].get("joint_rollback", False)) is not bool:
        raise ValueError("joint_rollback must be a boolean")
    if type(config["upper"].get("archive_context", False)) is not bool:
        raise ValueError("archive_context must be a boolean")
    for name in ("task_context", "search_training", "completion_reward", "position_context"):
        if type(config["lower"].get(name, False)) is not bool:
            raise ValueError(f"lower {name} must be a boolean")
    if config["lower"].get("representation", "ast") not in {"ast", "rpn"}:
        raise ValueError("lower representation must be ast or rpn")
    backend = config["lower"].get("edit_backend", "ast")
    if backend not in {"ast", "rpn"} or backend == "rpn" and config["lower"].get("representation", "ast") != "rpn":
        raise ValueError("native RPN edit backend requires the RPN representation")
    if config["lower"].get("completion_reward") and not config["lower"].get("search_training"):
        raise ValueError("completion reward requires search-task lower training")
    if config["upper"].get("representation", "ast") not in {"ast", "rpn"}:
        raise ValueError("upper representation must be ast or rpn")
    if type(config["upper"].get("cost_context", False)) is not bool:
        raise ValueError("upper cost_context must be a boolean")
    objective = config["upper"].get("training_reward", "search")
    if objective not in {"search", "forward", "forward_weighted", "forward_joint"}:
        raise ValueError("unknown upper training_reward")
    if objective in {"forward_weighted", "forward_joint"} and not config["lower"].get("completion_reward"):
        raise ValueError("aligned forward objectives require lower completion reward")
    if objective == "forward_joint" and config["search"]["quality_threshold"] <= 0:
        raise ValueError("joint completion requires a positive quality threshold")
    if config["upper"].get("credit_assignment", "episode") not in {"episode", "immediate"}:
        raise ValueError("upper credit_assignment must be episode or immediate")
    if objective.startswith("forward"):
        if any("future" not in task for task in config["tasks"] if task["role"] == "train"):
            raise ValueError("forward training requires a future slice for every training task")
    if config["upper"].get("joint_rollback") and not config["upper"].get("joint_guard"):
        raise ValueError("joint rollback requires guarded joint training")
    if type(config.get("upper_validation", {}).get("forward", False)) is not bool:
        raise ValueError("forward validation must be a boolean")
    if config.get("upper_validation", {}).get("forward"):
        names = config["upper_validation"].get("tasks") or [config["upper_validation"].get("task")]
        selected = [task for task in config["tasks"] if task["name"] in names]
        if len(selected) != len(names) or any(task["role"] != "development" or "future" not in task for task in selected):
            raise ValueError("forward validation requires development tasks with a future slice")
    for section in ("lower", "search"):
        if not 1 <= config[section].get("diffusion_steps", config["corpus"]["noise_steps"]) <= config["corpus"]["noise_steps"]:
            raise ValueError("effective diffusion horizon exceeds the pretrained time vocabulary")
        perturbation = config[section].get("perturbation_steps", config[section].get("diffusion_steps", config["corpus"]["noise_steps"]))
        if type(perturbation) is not int or not 0 <= perturbation <= config["corpus"]["noise_steps"]:
            raise ValueError("perturbation steps must be an integer within the noise vocabulary")
    names = [t["name"] for t in config["tasks"]]
    if len(names) != len(set(names)) or any(not name.replace("_", "").isalnum() for name in names):
        raise ValueError("task names must be unique path-safe identifiers")
    config["data_fingerprints"] = {}
    cache_fingerprints = {}
    for task in config["tasks"]:
        for role in ("fit", "search", "future"):
            if role not in task:
                continue
            spec = task[role]
            cache = Path(spec["cache"]).expanduser()
            if not cache.is_absolute():
                cache = path.parent / cache
            spec["cache"] = str(cache.resolve())
            metadata = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
            if not set(config["grammar"]["fields"]) <= set(metadata["fields"]):
                raise ValueError("configured grammar fields missing from panel")
            if config["data_mode"] == "formal" and not metadata.get("data_conventions_verified", False):
                raise ValueError("formal experiments require verified point-in-time data conventions")
    # 对实际特征与标识文件的字节计算哈希，防止清单中的旧指纹与真实数据不一致。
            files = ["manifest.json", "dates.npy", "codes.json", *(f"{field}.npy" for field in config["grammar"]["fields"]), "close.npy"]
            if metadata.get("label_file"):
                files.append(metadata["label_file"])
            if cache in cache_fingerprints:
                config["data_fingerprints"][f"{task['name']}.{role}"] = cache_fingerprints[cache]
                continue
            fingerprint = hashlib.sha256()
            for name in sorted(set(files)):
                fingerprint.update(name.encode())
                with (cache / name).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        fingerprint.update(chunk)
            config["data_fingerprints"][f"{task['name']}.{role}"] = fingerprint.hexdigest()
            cache_fingerprints[cache] = fingerprint.hexdigest()
    # 源码按原始字节计算指纹，增加注释也会改变新运行的实现身份。
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    factor_root = Path(__file__).parents[1]
    for name in ("data.py", "expression.py", "_vendor/alphagen/data/expression.py",
                 "_vendor/alphagen/utils/correlation.py"):
        sources[f"dependency:{name}"] = hashlib.sha256((factor_root / name).read_bytes()).hexdigest()
    config["implementation_hash"] = digest(sources)
    for name in ("initial_library", "initializer_checkpoint", "corpus_source", "signal_cache_dir"):
        if config.get(name):
            resolved = Path(config[name]).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            config[name] = str(resolved.resolve())
            if name in {"initial_library", "initializer_checkpoint"}:
                config.setdefault("artifact_hashes", {})[name] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    config["fingerprint"] = digest(config)
    return config


def run(config, output, resume=False, progress_callback=None):
    # 一个输出目录对应一个训练模型，运行锁防止多个进程写入相同检查点。
    """在指定输出目录执行单模型训练和评价，返回评价汇总。

    通过目录锁保护检查点，首次运行保存源码快照，resume 控制是否恢复已有状态。
    进度可交给回调处理，预算中止和异常会写入对应记录后继续抛出。
    """
    from .training import Campaign
    from .experiments import evaluate_campaign
    output = Path(output)
    with run_lock(output):
        try:
            from shutil import copy2
            snapshot = output / "source_snapshot"
            if not snapshot.exists():
                # 首次运行冻结源码，历史实验的重放依据是该目录自己的快照。
                snapshot.mkdir()
                for source in Path(__file__).parent.glob("*.py"):
                    copy2(source, snapshot / source.name)
            if not (output / "progress.json").exists():
                save_json(output / "progress.json", {"phase": "initialization", "configuration_hash": config["fingerprint"]})
            if config.get("architecture") == "candidate_selection":
                from .selection_training import SelectionCampaign
                return SelectionCampaign(config, output, resume, progress_callback).run()
            campaign = Campaign(config, output, resume, progress_callback)
            campaign.run()
            if campaign.state["phase"] != "complete":
                campaign.state.setdefault("training_wall_seconds", campaign.elapsed())
                campaign.commit()
            return evaluate_campaign(campaign)
        except BudgetExpired as exc:
            save_json(output / "budget_stop.json", {"message": str(exc), "resumable": True})
            raise
        except BaseException as exc:
            save_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            raise


def run_suite(config, output, seeds, resume=False):
    """按不同种子独立训练网络，每个模型的重复搜索采样在该模型内部统计。"""
    # suite 的种子控制独立模型训练，evaluation.seeds 只控制同一模型的搜索采样。
    from copy import deepcopy
    from .experiments import aggregate
    output = Path(output)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("training seeds must be distinct nonnegative integers")
    signature = digest([config["fingerprint"], seeds])
    with run_lock(output):
        manifest_path = output / "suite_manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not resume or saved["configuration_hash"] != signature:
                raise ValueError("existing suite requires --resume and identical code/data/config/seeds")
        else:
            save_json(manifest_path, {"configuration_hash": signature, "training_seeds": seeds,
                                      "config": config})
        rows, completed = [], []
        for seed in seeds:
            child_config = deepcopy(config)
            child_config["seed"] = seed
            child_config.pop("fingerprint", None)
            child_config["fingerprint"] = digest(child_config)
            child = output / f"training_seed_{seed}"
            save_json(output / "suite_progress.json", {"phase": "training_and_evaluation",
                      "current_training_seed": seed, "completed_training_seeds": completed,
                      "child_output": str(child.resolve())})
            try:
                check_budget()
                summary = run(child_config, child, resume=(child / "training_state.pt").exists())
            except BudgetExpired:
                result = {"status": "budget_stopped", "completed_training_seeds": completed,
                          "current_training_seed": seed, "resumable": True}
                save_json(output / "suite_progress.json", {"phase": "budget_stopped", **result})
                return result
    # 先在同一个训练种子内部汇总重复搜索，再根据独立训练模型计算区间。
    # 模型独立性按训练种子统计，同一模型的多次搜索属于重复采样。
            for item in summary["aggregate"]:
                rows.append({"task": item["task"], "method": item["method"], "training_seed": seed,
                             **{name: value["mean"] for name, value in item["metrics"].items()}})
            completed.append(seed)
        return finalize_suite(config, output, seeds, rows)


def finalize_suite(config, output, seeds, rows):
    """汇总各训练种子的模型内搜索均值，返回多模型实验结果。

    rows 以独立训练模型为统计单位，结果及完成进度分别写入输出目录。
    """
    from .experiments import aggregate
    output = Path(output)
    result = {"status": "complete", "training_seeds": seeds,
              "independent_training_runs": len(seeds), "interval_unit": "training_seed",
              "sampling_repeats_per_training_seed": len(config["evaluation"]["seeds"]),
              "aggregate": aggregate(rows), "configuration_hash": digest([config["fingerprint"], seeds]),
              "data_mode": config["data_mode"]}
    save_json(output / "suite_summary.json", result)
    save_json(output / "suite_progress.json", {"phase": "complete", "completed_training_seeds": seeds})
    return result


def main():
    # run 训练单模型，suite 训练多个独立种子，status 读取已保存进度。
    """解析命令行并分派单模型训练、多种子实验或进度查询。

    训练命令先加载并校验配置，再记录实验开始和结束事件。
    """
    parser = argparse.ArgumentParser(description="Two-level PPO native RPN generation and equal-weight selection")
    parser.add_argument("command", choices=("run", "suite", "status"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="independent training seeds for suite")
    args = parser.parse_args()
    if args.command == "status":
        suite_status = args.output / "suite_progress.json"
        if suite_status.exists():
            state = json.loads(suite_status.read_text(encoding="utf-8"))
            child = Path(state.get("child_output", args.output)) / "progress.json"
            if child.exists():
                state["current_run"] = json.loads(child.read_text(encoding="utf-8"))
            print(json.dumps(state, ensure_ascii=False, indent=2))
        else:
            print((args.output / "progress.json").read_text(encoding="utf-8"))
        return
    if args.config is None:
        parser.error("run requires --config")
    config = load_config(args.config)
    event("campaign_started", output=str(args.output.resolve()), fingerprint=config["fingerprint"])
    result = (run_suite(config, args.output, args.seeds, args.resume) if args.command == "suite" else
              run(config, args.output, args.resume))
    event("campaign_completed", **result)


if __name__ == "__main__":
    main()
