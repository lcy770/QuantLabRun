"""将平台 train/valid/test 数据契约映射到因果的 F/G/A/R 训练与冻结验收。"""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from ..data import PanelData
from ..storage import FactorStorage
from .io import save_json


def chronological_sections(cache, config, names, before=None):
    """在真实交易日及标签成熟约束下均分多个评分段，段间清除未成熟标签。"""
    panel = PanelData.from_memmap(cache, config["lookback"], config["future_days"])
    back, horizon = config["lookback"], config["target_horizon"]
    dates = panel.dates[back:back + panel.n_days].values.astype("datetime64[D]")
    ends = panel.dates[back + horizon:back + horizon + panel.n_days].values.astype("datetime64[D]")
    limit = len(dates) if before is None else int(np.count_nonzero(ends < np.datetime64(before)))
    size = (limit - (len(names) - 1) * horizon) // len(names)
    if size < 3:
        raise ValueError(f"cache is too short for {len(names)} mature F/G/A/R sections")
    result, first = {}, 0
    for index, name in enumerate(names):
        last = limit - 1 if index == len(names) - 1 else first + size - 1
        if last < first or last >= limit:
            raise ValueError("insufficient dates after label-maturity purging")
        result[name] = {"cache": str(cache), "start": str(dates[first]), "end": str(dates[last])}
        first = int(np.searchsorted(dates, ends[last], side="right"))
    return result


def run_selection_from_runtime(runtime, output, settings, template, seed, device, target_count,
                               max_attempts, operators=None, progress_callback=None):
    """通过正常平台入口执行新双层训练，test 仅用于全部选择冻结后的验收。"""
    from .__main__ import load_config, run
    from .bridge import GenerationResult
    config = copy.deepcopy(template)
    config.update(seed=seed, device=device, target_horizon=runtime["target_horizon"],
                  lookback=runtime["max_backtrack_days"], future_days=runtime["max_future_days"])
    config["grammar"]["fields"] = runtime["fields"]
    config["grammar"]["max_lookback"] = config["lookback"]
    if operators:
        config["grammar"]["operators"] = list(operators)
    # 研究示例库的筛选日期与平台任务无关，平台仅使用显式指定的可追溯库。
    config.pop("initial_library", None)
    config.pop("initializer_checkpoint", None)
    for name in ("initial_library", "initializer_checkpoint", "rollout_workers", "initial_seeds"):
        if name in settings:
            config[name] = settings[name]
    for name in ("corpus", "pretrain", "reward", "lower", "upper", "search", "evaluation"):
        config.setdefault(name, {}).update(settings.get(name, {}))
    quality = settings.get("quality", {})
    thresholds = list(config["reward"].get("thresholds", [0.02, 0.01, 0.10, 0.80]))
    for index, name in enumerate(("min_valid_ic", "min_rank_ic", "min_icir", "min_coverage")):
        if quality.get(name) is not None:
            thresholds[index] = quality[name]
    config["reward"]["thresholds"] = thresholds
    splits = runtime["splits"]
    if not {"train", "valid", "test"} <= splits.keys():
        raise ValueError("hierarchical generation requires train/valid/sealed test caches")
    train, valid, test = [splits[name]["cache_dir"] for name in ("train", "valid", "test")]
    valid_panel = PanelData.from_memmap(valid, config["lookback"], config["future_days"])
    test_panel = PanelData.from_memmap(test, config["lookback"], config["future_days"])
    train_sections = chronological_sections(train, config, ("fit", "search", "validation", "feedback"),
                                           before=valid_panel.dates[config["lookback"]])
    search_sections = chronological_sections(valid, config, ("search", "validation"),
                                            before=test_panel.dates[config["lookback"]])
    config["tasks"] = [
        {"name": "platform_train", "role": "train", **train_sections},
        {"name": "platform_search", "role": "test", "fit": train_sections["fit"],
         **search_sections, "future": {"cache": str(test)}}]
    config["search"]["max_attempts"] = max_attempts
    config["search"]["target_count"] = target_count
    config["search"]["candidate_capacity"] = min(config["search"].get("candidate_capacity", 128), target_count)
    config["evaluation"].update(methods=["upper_ppo"], seeds=[seed])
    config["storage_metadata"] = {name: runtime[name] for name in
                                  ("pool_id", "runtime_id", "source_signature", "fields") if name in runtime}
    output = Path(output)
    path = output / "hierarchical_runtime_config.json"
    save_json(path, config)
    started = datetime.now().astimezone().isoformat()
    summary = run(load_config(path), output, resume=(output / "training_state.pt").exists(),
                  progress_callback=progress_callback)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (output / "trials").glob("*_upper_ppo.json")]
    timesteps = 0
    for name in ("lower_curve.jsonl", "upper_curve.jsonl"):
        timesteps += sum(json.loads(line).get("samples", 0) for line in
                         (output / name).read_text(encoding="utf-8").splitlines())
    return GenerationResult(FactorStorage(output).accepted_count, sum(report["attempts"] for report in reports),
        target_count, max_attempts, device, timesteps=timesteps, training_started_at=started,
        training_finished_at=datetime.now().astimezone().isoformat(), training_seconds=summary["training_wall_seconds"])
