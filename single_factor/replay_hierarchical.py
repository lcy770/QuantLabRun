"""Replay a saved PPO update without re-querying prices or wall-clock sampling."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import torch

from .hierarchical.io import save_json
from .hierarchical.training import restore_models
from .hierarchical.ppo import update


def replay(path, config_path, output, threads=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if config["fingerprint"] != saved["models"]["configuration_hash"]:
        raise ValueError("rollout/configuration fingerprint differs")
    torch.set_num_threads(threads or config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    layer = "lower" if "task" in saved else "upper"
    reconstruction_started = perf_counter()
    if layer == "lower" and saved["models"].get("architecture") != "candidate_selection":
        builder, _ = restore_models(saved["models"])
        for row in saved["rows"]:
            row["legal_actions"] = builder.edit_kernel.edits(builder.edit_kernel.from_node(row["tree"]))
    reconstruction_seconds = perf_counter() - reconstruction_started
    results, metrics = [], []
    for _ in range(2):
        lower, upper = restore_models(saved["models"])
        model = lower if layer == "lower" else upper
        optimizer = torch.optim.Adam(model.parameters(), lr=config[layer]["learning_rate"])
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        rng = random.Random()
        rng.setstate(saved["shuffle_rng"])
        started = perf_counter()
        metrics.append(update(model, optimizer, saved["rows"], rng=rng, **config[layer]["ppo"]))
        metrics[-1]["update_seconds"] = perf_counter() - started
        results.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    difference = max(float((results[0][k] - results[1][k]).abs().max()) for k in results[0])
    if difference > 2e-6:
        raise AssertionError(f"replayed PPO updates differ: {difference}")
    result = {"rollout": str(Path(path).resolve()), "rollout_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "configuration_hash": config["fingerprint"], "layer": layer, "rows": len(saved["rows"]),
              "execution_threads": torch.get_num_threads(), "recording_threads": config["torch_threads"],
              "legal_action_reconstruction_seconds": reconstruction_seconds,
              "repeat_max_parameter_difference": difference, "ppo_metrics": metrics,
              "market_data_requeried": False, "observed_actions_replayed": True}
    save_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    print(json.dumps(replay(args.rollout, args.config, args.output, args.threads)))
