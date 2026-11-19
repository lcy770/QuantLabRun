"""从完整训练状态追加时间，保留父轮模型、协议和评价结果。"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch

from .comparison_study import experiment_nodes, resolve_nodes
from .io import digest, run_lock, save_json, save_torch
from .selection_config import file_hash, load_selection_config


def _prepare_extension(protocol, source, output, request):
    """校验父轮数据与配置，复制学习状态并原子发布独立续训轮次。"""
    parent = json.loads(protocol.read_text(encoding="utf-8"))
    if parent["fingerprint"] != digest({key: value for key, value in parent.items() if key != "fingerprint"}):
        raise ValueError("source protocol fingerprint differs")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_fingerprint"] != parent["fingerprint"]:
        raise ValueError("source results belong to a different protocol")
    config = load_selection_config(protocol.with_name("config.json"))
    ignored = {"implementation_hash", "fingerprint"}
    if ({key: value for key, value in config.items() if key not in ignored}
            != {key: value for key, value in parent["config"].items() if key not in ignored}):
        raise ValueError("source data or training configuration changed")

    definitions = experiment_nodes()
    node, seed = request["node"], request["seed"]
    ordered, _ = resolve_nodes([node])
    saved, checksums = {}, {}
    for name in ordered:
        if parent["nodes"].get(name) != definitions[name]:
            raise ValueError(f"source algorithm definition changed: {name}")
        path = source / f"seed_{seed}" / "nodes" / name / "checkpoint.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        dependencies = {key: saved[definitions[name][key]]["identity"]
                        for key in ("generator_source", "selector_source") if key in definitions[name]}
        if state["identity"] != digest([parent["fingerprint"], name, seed, dependencies]):
            raise ValueError(f"source checkpoint identity differs: {name}")
        if name != node and state["state"]["phase"] != "complete":
            raise ValueError(f"source dependency has not completed: {name}")
        saved[name], checksums[name] = state, file_hash(path)

    spec = definitions[node]
    phase = "continuation" if spec.get("continuation") else "upper" if spec.get("upper") else "lower"
    cost_key = {"lower": "generator_training", "upper": "selector_training", "continuation": "continuation"}[phase]
    state = saved[node]["state"]
    spent = state["costs"][cost_key]
    target = spent + request["additional_hours"] * 3600
    if not math.isfinite(target) or target <= spent or spent < 0:
        raise ValueError("additional training time is invalid or too small")
    settings = copy.deepcopy(parent["settings"])
    settings[f"{phase}_seconds"] = target
    # 追加时间不能再次被旧轮数上限截断，原有轮数和采样种子序列继续累计。
    settings[f"{phase}_iterations"] = max(settings[f"{phase}_iterations"], state[f"{phase}_iteration"] + 1_000_000_000)
    config["study_profile"] = settings
    if phase in {"lower", "upper"}:
        config[phase]["iterations"] = settings[f"{phase}_iterations"]
    lineage = {"request": request, "phase": phase, "previous_seconds": spent, "target_seconds": target,
               "source_protocol_fingerprint": parent["fingerprint"], "source_checkpoints": checksums,
               "source_implementation_hash": parent["config"]["implementation_hash"]}

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        ready = Path(temporary) / "ready"
        config_path = ready / "protocol" / "config.json"
        save_json(config_path, config)
        resolved = load_selection_config(config_path)
        child = {**parent, "config": resolved, "settings": settings, "nodes": definitions, "continuation": lineage}
        child.pop("fingerprint")
        child["fingerprint"] = digest(child)
        save_json(config_path.with_name("protocol.json"), child)
        identities = {}
        for name in ordered:
            item = saved[name]
            dependencies = {key: identities[definitions[name][key]]
                            for key in ("generator_source", "selector_source") if key in definitions[name]}
            item["identity"] = identities[name] = digest([child["fingerprint"], name, seed, dependencies])
            item["state"]["continued_from"] = {"identity": parent["fingerprint"], "checkpoint_sha256": checksums[name]}
            stages = ("syntax", "lower", "freeze", "upper", "continuation", "complete")
            if name == node and stages.index(item["state"]["phase"]) > stages.index(phase):
                item["state"]["phase"] = phase
            save_torch(ready / "results" / f"seed_{seed}" / "nodes" / name / "checkpoint.pt", item)
        calibration = source / f"seed_{seed}" / "calibration.json"
        shutil.copy2(calibration, ready / "results" / f"seed_{seed}" / "calibration.json")
        save_json(ready / "source_protocol.json", parent)
        ready.replace(output)


def extend_study(protocol, source, output, node, seed, hours, *, train_only=False):
    """追加指定小时；重复同一命令恢复同一轮，下一轮需指定新的输出目录。"""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0:
        raise ValueError("hours must be finite and positive")
    definitions = experiment_nodes()
    if node not in definitions:
        raise ValueError(f"unknown node: {node}")
    while "alias" in definitions[node]:
        node = definitions[node]["alias"]
    spec = definitions[node]
    if spec.get("legacy") or not any(spec.get(key) for key in ("lower", "upper", "continuation")):
        raise ValueError("extension requires a trainable generator, selector or continuation node")
    protocol, source, output = Path(protocol).resolve(), Path(source).resolve(), Path(output).resolve()
    request = {"protocol": str(protocol), "source": str(source), "node": node,
               "seed": seed, "additional_hours": float(hours)}
    child_protocol = output / "protocol" / "protocol.json"
    if output.exists():
        if not child_protocol.is_file():
            raise FileExistsError("output already exists and is not an extension round")
        recorded = json.loads(child_protocol.read_text(encoding="utf-8"))
        if recorded.get("continuation", {}).get("request") != request:
            raise FileExistsError("extension request differs; choose a new output directory")
    else:
        if not source.is_dir() or not protocol.is_file():
            raise FileNotFoundError("source protocol and result directory must exist")
        with run_lock(source):
            _prepare_extension(protocol, source, output, request)
    from .comparison_cli import run_study
    result = run_study(child_protocol, output / "results", [seed], nodes=[node], train_only=train_only)
    return {"protocol": str(child_protocol), "output": str(output / "results"), **result}
