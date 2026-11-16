"""Integrate the full hierarchy with the existing platform generation contract."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .__main__ import load_config, run
from .io import save_json


@dataclass
class GenerationResult:
    accepted_count: int
    attempts: int
    target_count: int
    max_attempts: int
    device: str
    timesteps: int = 0
    training_started_at: str | None = None
    training_finished_at: str | None = None
    training_seconds: float | None = None

    @property
    def target_reached(self):
        return self.accepted_count >= self.target_count

    @property
    def attempts_exhausted(self):
        return self.attempts >= self.max_attempts


def run_from_runtime(runtime, output, settings, seed, device, target_count, max_attempts,
                     operators=None, progress_callback=None):
    """Historical training inside train; valid is search reward; test stays sealed."""
    from datetime import datetime
    from ..data import PanelData
    template = settings.get("config")
    template = Path(template) if template else Path(__file__).parents[1] / "hierarchical_config.example.json"
    config = json.loads(template.read_text(encoding="utf-8-sig"))
    if config.get("architecture") == "candidate_selection":
        from .selection_bridge import run_selection_from_runtime
        return run_selection_from_runtime(runtime, output, settings, config, seed, device,
                                          target_count, max_attempts, operators, progress_callback)
    config.update(seed=seed, device=device, target_horizon=runtime["target_horizon"],
                  lookback=runtime["max_backtrack_days"], future_days=runtime["max_future_days"])
    config["grammar"]["fields"] = runtime["fields"]
    config["grammar"]["max_lookback"] = config["lookback"]
    quality = settings.get("quality", {})
    config["search"]["min_coverage"] = quality.get("min_coverage", config["search"]["min_coverage"])
    if quality.get("min_valid_ic") is not None:
        config["search"]["quality_threshold"] = quality["min_valid_ic"]
    for name in ("min_train_ic", "min_icir"):
        if quality.get(name) is not None:
            config["search"][name] = quality[name]
    config["lower"]["score_weights"] = [quality.get(name, default) for name, default in
        zip(("score_ic_weight", "score_rank_ic_weight", "score_icir_weight"), config["lower"]["score_weights"])]
    if operators:
        config["grammar"]["operators"] = list(operators)
    splits = runtime["splits"]
    if not all(name in splits for name in ("train", "valid", "test")):
        raise ValueError("hierarchical generation requires train, valid and sealed test caches")
    panel = PanelData.from_memmap(splits["train"]["cache_dir"], config["lookback"], config["future_days"])
    back, horizon = config["lookback"], config["target_horizon"]
    dates = panel.dates[back:back + panel.n_days]
    if len(dates) < 2 * horizon + 12:
        raise ValueError("train split too short for causal upper training tasks")
    boundary = len(dates) // 2
    cache = str(splits["train"]["cache_dir"])
    train_task = {"name": "historical_train", "role": "train",
                  "fit": {"cache": cache, "end": str(dates[boundary - horizon - 1].date())},
                  "search": {"cache": cache, "start": str(dates[boundary].date())}}
    # Explicitly purge the fit scoring tail against valid scoring start.
    valid = PanelData.from_memmap(splits["valid"]["cache_dir"], config["lookback"], config["future_days"])
    valid_start = valid.dates[back]
    ends = panel.dates[back + horizon:back + horizon + panel.n_days]
    allowed = dates[ends < valid_start]
    if len(allowed) < 3:
        raise ValueError("no label-mature fit rows before valid period")
    if allowed[-1] <= dates[boundary]:
        raise ValueError("no historical upper-training feedback period before valid")
    train_task["search"]["end"] = str(allowed[-1].date())
    test_panel = PanelData.from_memmap(splits["test"]["cache_dir"], config["lookback"], config["future_days"])
    valid_dates = valid.dates[back:back + valid.n_days]
    valid_ends = valid.dates[back + horizon:back + horizon + valid.n_days]
    valid_allowed = valid_dates[valid_ends < test_panel.dates[back]]
    if len(valid_allowed) < 3:
        raise ValueError("no mature search labels before sealed test period")
    eval_task = {"name": "platform_search", "role": "test",
                 "fit": {"cache": cache, "end": str(allowed[-1].date())},
                 "search": {"cache": str(splits["valid"]["cache_dir"]), "end": str(valid_allowed[-1].date())},
                 "future": {"cache": str(splits["test"]["cache_dir"])}}
    config["tasks"] = [train_task, eval_task]
    config["evaluation"]["methods"] = ["upper_ppo"]
    config["evaluation"]["seeds"] = [seed]
    config["search"]["max_attempts"] = max_attempts
    config["storage_metadata"] = {name: runtime[name] for name in ("pool_id", "runtime_id", "source_signature", "fields")
                                  if name in runtime}
    # Overrides are explicit research settings; no model mechanism is substituted.
    for section in ("corpus", "pretrain", "lower", "upper", "search", "evaluation"):
        config[section].update(settings.get(section, {}))
    output = Path(output)
    save_json(output / "hierarchical_runtime_config.json", config)
    started = datetime.now().astimezone().isoformat()
    summary = run(load_config(output / "hierarchical_runtime_config.json"), output,
                  resume=(output / "training_state.pt").exists(), progress_callback=progress_callback)
    finished = datetime.now().astimezone().isoformat()
    from ..storage import FactorStorage
    storage = FactorStorage(output)
    for name in ("accepted_factors.jsonl", "rejected_candidates.jsonl"):
        (output / name).touch(exist_ok=True)
    trials = list((output / "trials").glob("*_upper_ppo.json"))
    attempts = sum(json.loads(p.read_text(encoding="utf-8"))["attempts"] for p in trials)
    timesteps = 0
    for name in ("lower_curve.jsonl", "upper_curve.jsonl"):
        path = output / name
        if path.exists():
            timesteps += sum(json.loads(line).get("samples", 0) for line in path.read_text(encoding="utf-8").splitlines())
    return GenerationResult(storage.accepted_count, attempts, target_count, max_attempts, device,
                            timesteps=timesteps,
                            training_started_at=started, training_finished_at=finished,
                            training_seconds=summary["end_to_end_wall_seconds"])
