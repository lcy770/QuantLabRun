"""可迁移实验入口，覆盖准备、五组对比、冻结验收和配对汇总。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import stat
from tempfile import TemporaryDirectory
from time import perf_counter
import zipfile

import torch

from .comparison_data import TRAINING_START, design_tasks, prepare_database_panel, prepare_panel
from .comparison_study import TEMPLATES, experiment_nodes, prepare_protocol, resolve_nodes, summarize_study
from .comparison_training import Study
from .io import append_json, digest, run_lock, save_json
from .selection_config import file_hash


def environment_report():
    """记录实际运行依赖、CUDA 与硬件，便于另一台机器核验。"""
    import platform
    packages = {}
    for name in ("torch", "numpy", "pandas", "pyarrow", "gymnasium", "stable-baselines3", "sb3-contrib", "exchange-calendars"):
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def copy_source_snapshot(output):
    """冻结实验实际调用的本地源码，包含未提交到 Git 的算法文件。"""
    from .comparison_study import REPOSITORY
    target = output / "source_snapshot"
    if target.exists():
        return
    for directory in (REPOSITORY / "single_factor", REPOSITORY / "quantops/quantops", REPOSITORY / "backend/adapters"):
        for source in directory.rglob("*.py"):
            destination = target / source.relative_to(REPOSITORY)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for filename in ("canonical_expression.py", "expression_parser.py"):
        source = REPOSITORY / "backend/custom_bt" / filename
        destination = target / source.relative_to(REPOSITORY)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_study(protocol, output, seeds, nodes=None, groups=None):
    """按依赖顺序执行完整真实实验，恢复时跳过已提交模型与评价。"""
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("distinct nonnegative paired seeds are required")
    study = Study(protocol, output)
    ordered, requested = resolve_nodes(nodes, groups)
    with run_lock(study.output):
        manifest_path = study.output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["protocol_fingerprint"] != study.protocol["fingerprint"]:
                raise ValueError("study protocol differs from existing output")
        else:
            manifest = {"protocol_fingerprint": study.protocol["fingerprint"], "protocol": str(Path(protocol).resolve()),
                        "seeds_seen": [], "requested_nodes": [], "environment": environment_report()}
        manifest["seeds_seen"] = sorted(set(manifest["seeds_seen"]) | set(seeds))
        manifest["requested_nodes"] = sorted(set(manifest["requested_nodes"]) | set(requested))
        manifest["comparison_split"] = study.comparison_split
        manifest["width"] = study.config["width"]
        manifest["rpn_width"] = study.config.get("autoregressive", {}).get("width", 128)
        save_json(manifest_path, manifest)
        copy_source_snapshot(study.output)
        for seed in seeds:
            for name in ordered:
                started = perf_counter()
                save_json(study.output / "progress.json", {"phase": "running", "training_seed": seed, "node": name})
                try:
                    study.run_node(name, seed)
                    if not study.definitions[name].get("alias"):
                        study.evaluate_node(name, seed)
                except BaseException as error:
                    save_json(study.output / "progress.json", {"phase": "failed", "training_seed": seed,
                        "node": name, "error_type": type(error).__name__, "error": str(error)})
                    raise
                append_json(study.output / "execution.jsonl", {"training_seed": seed, "node": name,
                            "wall_seconds": perf_counter() - started, "status": "complete"})
        summary = summarize_study(study.output)
        save_json(study.output / "progress.json", {"phase": "complete", "completed_nodes": ordered,
                  "training_seeds": seeds, **summary})
        return summary


def freeze_selection(protocol, output, nodes, seeds, destination):
    """根据开发结果显式冻结待验收方法和检查点内容身份。"""
    study = Study(protocol, output)
    items = []
    for seed in seeds:
        for name in nodes:
            canonical = study.canonical(name)
            study.saved_node(canonical, seed)
            path = study.node_path(canonical, seed) / "checkpoint.pt"
            if not list((path.parent / study.comparison_split).glob("*.json")):
                raise ValueError(f"{name}/{seed} has no completed {study.comparison_split} evaluation")
            items.append({"name": name, "canonical": canonical, "training_seed": seed, "checkpoint_sha256": file_hash(path)})
    selection = {"protocol_fingerprint": study.protocol["fingerprint"], "items": items,
                 "information_cutoff": study.protocol["dates"].get("comparison_information_cutoff",
                     study.protocol["dates"].get("development_information_cutoff")),
                 "selection_basis": study.comparison_split}
    selection["identity"] = digest(selection)
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != selection:
            raise FileExistsError("selection is immutable; use a new filename for a different selection")
    else:
        save_json(destination, selection)
    return selection


def evaluate_frozen(protocol, output, selection_path):
    """验证冻结检查点后运行独立验收，记录每次测试数据访问。"""
    study = Study(protocol, output)
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    identity_payload = {key: value for key, value in selection.items() if key != "identity"}
    if selection["identity"] != digest(identity_payload) or selection["protocol_fingerprint"] != study.protocol["fingerprint"]:
        raise ValueError("selection/protocol identity mismatch")
    study.evaluation_tasks("test")
    with run_lock(study.output):
        for item in selection["items"]:
            checkpoint = study.node_path(item["canonical"], item["training_seed"]) / "checkpoint.pt"
            if file_hash(checkpoint) != item["checkpoint_sha256"]:
                raise ValueError("selected model changed after freezing")
        append_json(study.output / "test_access.jsonl", {"selection_identity": selection["identity"],
                    "items": selection["items"], "stage": "before_locked_test"})
        for item in selection["items"]:
            study.evaluate_node(item["canonical"], item["training_seed"], "test", selection["identity"])
    return {"status": "complete", "selection_identity": selection["identity"], "model_evaluations": len(selection["items"])}


def transfer_data(cache, archive, unpack=False):
    """打包或校验解包数据，路径检查阻止越界写入及符号链接。"""
    cache, archive = Path(cache).resolve(), Path(archive).resolve()
    if unpack:
        if cache.exists() and any(cache.iterdir()):
            raise FileExistsError("unpack into an empty cache directory")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (cache / member.filename).resolve()
                if not destination.is_relative_to(cache) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("unsafe archive entry")
            with TemporaryDirectory(prefix=f".{cache.name}_", dir=cache.parent) as temporary:
                staging = Path(temporary).resolve()
                if not staging.is_relative_to(cache.parent):
                    raise ValueError("temporary transfer directory escaped its parent")
                bundle.extractall(staging)
                hashes = json.loads((staging / "transfer_hashes.json").read_text(encoding="utf-8"))
                for name, expected in hashes.items():
                    if not (staging / name).resolve().is_relative_to(staging):
                        raise ValueError("unsafe transfer hash path")
                    if file_hash(staging / name) != expected:
                        raise ValueError(f"transferred data hash mismatch: {name}")
                if cache.exists():
                    cache.rmdir()
                staging.replace(cache)
        return {"cache": str(cache), "verified_files": len(hashes)}
    if archive.exists():
        raise FileExistsError(archive)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    names = ["manifest.json", "codes.json", "dates.npy", *(f"{field}.npy" for field in manifest["fields"])]
    if manifest.get("label_file"):
        names.append(manifest["label_file"])
    if (cache / "source_hashes.json").exists():
        names.append("source_hashes.json")
    if any(not (cache / name).resolve().is_relative_to(cache) for name in names):
        raise ValueError("manifest references data outside the requested cache")
    hashes = {name: file_hash(cache / name) for name in names}
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for name in names:
            bundle.write(cache / name, name)
        bundle.writestr("transfer_hashes.json", json.dumps(hashes))
    temporary.replace(archive)
    return {"archive": str(archive), "files": len(names), "sha256": file_hash(archive)}


def main():
    """解析跨 Windows/Linux 一致的实验命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    plan = commands.add_parser("plan")
    plan.add_argument("--nodes", nargs="+")
    plan.add_argument("--groups", nargs="+")
    plan.add_argument("--cache", help="read the actual information cutoff from a prepared panel")
    plan.add_argument("--training-start")
    plan.add_argument("--end", help="information cutoff including matured labels")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cache", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--profile", choices=("smoke", "short", "long"), default="short")
    prepare.add_argument("--device", default="auto")
    prepare.add_argument("--width", type=int, help="覆盖上下层网络隐藏宽度；原生RPN PPO仍使用独立的rpn_width")
    prepare.add_argument("--tasks-config", help="explicit historical/test tasks for local smoke validation")
    prepare.add_argument("--training-start")
    prepare.add_argument("--end", help="information cutoff, defaults to the prepared panel end")
    panel = commands.add_parser("prepare-panel")
    panel.add_argument("--source", choices=("clickhouse", "master"), help="defaults to ClickHouse; --master preserves historical imports")
    panel.add_argument("--master")
    panel.add_argument("--codes", default=str(TEMPLATES / "assets/codes.json"))
    panel.add_argument("--output", required=True)
    panel.add_argument("--training-start", default=TRAINING_START)
    panel.add_argument("--start", help="raw data start, defaults to 100 sessions before training")
    panel.add_argument("--end", help="defaults to the latest available completed trading day")
    panel.add_argument("--snapshot-id", help="reuse an existing database snapshot offline")
    panel.add_argument("--snapshot-root", help="directory holding immutable database snapshots")
    panel.add_argument("--refresh", action="store_true", help="capture revised database data in a new snapshot")
    for command in ("run", "freeze", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--protocol", required=True)
        sub.add_argument("--output", required=True)
        if command in {"run", "freeze"}:
            sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
            sub.add_argument("--nodes", nargs="+", required=command == "freeze")
        if command == "run":
            sub.add_argument("--groups", nargs="+")
        else:
            sub.add_argument("--selection", required=True)
    for command in ("status", "summarize"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", required=True)
        if command == "summarize":
            sub.add_argument("--split", choices=("training", "development", "test"))
    for command in ("pack-data", "unpack-data"):
        sub = commands.add_parser(command)
        sub.add_argument("--cache", required=True)
        sub.add_argument("--archive", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        result = environment_report()
    elif args.command == "plan":
        ordered, requested = resolve_nodes(args.nodes, args.groups)
        tasks, dates = design_tasks(args.cache or "<DATA_CACHE>", start=args.training_start, test_cutoff=args.end)
        result = {"ordered_nodes": ordered, "requested_nodes": requested, "definitions": experiment_nodes(), "dates": dates,
                  "first_training_task": tasks[0], "latest_training_task": tasks[-1],
                  "comparison_task_names": [task["name"] for task in tasks], "locked_test": None}
    elif args.command == "prepare-panel":
        source = args.source or ("master" if args.master else "clickhouse")
        if source == "master":
            if not args.master or args.snapshot_id or args.snapshot_root or args.refresh:
                parser.error("master source requires --master and accepts no snapshot options")
            result = prepare_panel(args.master, args.codes, args.output,
                                   start=args.start or "2023-07-03", end=args.end or "2026-06-16")
        else:
            if args.master:
                parser.error("ClickHouse source cannot be combined with --master")
            result = prepare_database_panel(args.codes, args.output, training_start=args.training_start,
                start=args.start, end=args.end, snapshot_id=args.snapshot_id,
                snapshot_root=args.snapshot_root, refresh=args.refresh)
    elif args.command == "prepare":
        tasks = None
        if args.tasks_config:
            task_config = Path(args.tasks_config).resolve()
            tasks = json.loads(task_config.read_text(encoding="utf-8-sig"))["tasks"]
            for task in tasks:
                for segment in ("fit", "search", "validation", "feedback", "future"):
                    if segment in task and "cache" in task[segment]:
                        cache = Path(task[segment]["cache"])
                        if not cache.is_absolute():
                            task[segment]["cache"] = str((task_config.parent / cache).resolve())
        overrides = {"training_start": args.training_start, "cutoff": args.end} if args.training_start or args.end else {}
        if args.width is not None:
            overrides["width"] = args.width
        result = prepare_protocol(args.cache, args.output, args.profile, args.device, tasks, **overrides)
        result = {"protocol": str(Path(args.output).resolve() / "protocol.json"), "width": result["config"]["width"],
                  "dates": result["dates"]}
    elif args.command == "run":
        result = run_study(args.protocol, args.output, args.seeds, args.nodes, args.groups)
    elif args.command == "freeze":
        result = freeze_selection(args.protocol, args.output, args.nodes, args.seeds, args.selection)
    elif args.command == "evaluate":
        result = evaluate_frozen(args.protocol, args.output, args.selection)
    elif args.command == "summarize":
        result = summarize_study(args.output, split=args.split)
    elif args.command == "status":
        result = json.loads((Path(args.output) / "progress.json").read_text(encoding="utf-8"))
    else:
        result = transfer_data(args.cache, args.archive, unpack=args.command == "unpack-data")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
