"""Cross-domain hierarchy transfer: task completion feedback and persistent tasks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import torch

from . import forward_reward_study as evaluation
from .hierarchical.__main__ import load_config
from .hierarchical.io import append_json, digest, run_lock, save_json
from .hierarchical.training import Campaign


PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir() and (p / "pytest.ini").exists())
ROOT = PROJECT / "var/experiments/coordination_study_20261109"
PARENT = PROJECT / "var/experiments/budget_study_20261108"
ARMS = ("interface_control", "completion", "completion_task")

# Reuse the same complete evaluation protocol, with joint checkpoints.
evaluation.PROJECT = PROJECT
evaluation.ROOT = ROOT
evaluation.PARENT = PARENT
evaluation.ARMS = ARMS
evaluation.METHODS = (*ARMS, "random", "original_rpn")
evaluation.MODEL_CHECKPOINT = "joint_final.pt"


def prepare():
    if (ROOT / "resolved_protocol.json").exists():
        return evaluation.protocol()
    config = json.loads((PROJECT / "var/experiments/forward_reward_study_20261109/protocol_config.json").read_text(encoding="utf-8"))
    config["signal_cache_dir"] = str(ROOT / "signal_cache")
    config["upper"].update(iterations=1, joint_iterations=8, joint_guard=True, joint_rollback=False,
                           learning_rate=.001, archive_context=False, training_reward="search")
    # The existing guarded schedule uses one quarter of this rate for the lower.
    # Every arm therefore has the same actual lower learning rate of .0003.
    config["lower"].update(learning_rate=.0012, search_training=True, task_context=False, completion_reward=False)
    config["upper_validation"].update(every=4, episodes=2, max_attempts=32, forward=True)
    config["rollout_workers"] = 3
    save_json(ROOT / "protocol_config.json", config)
    config = load_config(ROOT / "protocol_config.json")
    save_json(ROOT / "resolved_protocol.json", config)
    save_json(ROOT / "preregistration.json", {
        "created_at": datetime.now().astimezone().isoformat(), "arms": list(ARMS), "training_seeds": [0, 1, 2],
        "purpose": "transfer cross-domain resource allocation, persistent subtask and coordinated completion mechanisms into current AST dual PPO",
        "initialization": "same legacy best_upper per seed; both optimizers freshly initialized in every arm",
        "task_projection": "source AST, weighted current pool AST summary and pool IC, persistent over every edit; zero output at migration",
        "control": "search-task collection, original fit single-factor reward, no persistent task",
        "completion": "same input; terminal reward for discovery and pool gain against pre-action archive",
        "completion_task": "same completion reward plus persistent task input",
        "lower_updates": "8 batches of 48 main candidates; 16 non-terminal states x3 extra continuations, fixed old lower",
        "upper_updates": "after every lower update, recollect 6 episodes x48 upper choices, then update upper",
        "actual_learning_rates": {"lower": .0003, "upper": .001},
        "continuation_invariants": "same pre-commit archive and task; no branch discoveries; primary Torch RNG restored after auxiliary continuations",
        "selection": "same forward-development criterion at updates 0/4/8, continuous learner separate from selected model",
        "fixed_trials": 48, "serial_45_second_trials": 30,
        "sampling": "fixed uses101/202, timed uses101, order seed71923",
        "limitations": ["historical development reused", "eight updates screen mechanisms, convergence not established",
                        "completion vs control changes both lower objective and its historical feedback period",
                        "completion_task vs completion isolates persistent task architecture under matched feedback"],
        "source_root": str(ROOT / "source_snapshot"), "cloud_spent_cny": 0})
    snapshot = ROOT / "source_snapshot"
    save_json(ROOT / "source_manifest.json", {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob("*.py")})
    return config


def train_job(job):
    seed, arm = job
    config = copy.deepcopy(evaluation.protocol())
    config["seed"] = seed
    config["lower"].update(completion_reward=arm != "interface_control", task_context=arm == "completion_task")
    parent = PARENT / "runs" / f"training_seed_{seed}" / "best_upper.pt"
    config["warm_start"] = {"path": str(parent), "sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}
    config["fingerprint"] = digest(config)
    output = ROOT / "runs" / arm / f"training_seed_{seed}"
    with run_lock(output):
        campaign = Campaign(config, output, resume=(output / "training_state.pt").exists())
        try:
            # Backend adapters can prepend the repository root while resolving
            # operators. Nested Windows collectors must import this source tree.
            source_root = str(Path(__file__).resolve().parents[1])
            if sys.path[0] != source_root:
                sys.path.insert(0, source_root)
            if "warm_start" not in campaign.state:
                bundle = torch.load(parent, map_location="cpu", weights_only=False)
                campaign.lower.load_state_dict(bundle["lower"])
                campaign.upper.load_state_dict(bundle["upper"])
                campaign.rng = random.Random(39171 + seed)
                torch.manual_seed(39171 + seed)
                campaign.state.update(phase="joint", lower_iteration=128, warm_start=config["warm_start"])
                campaign.commit()
            campaign.run()
            save_json(output / "cost.json", {"additional_wall_seconds": campaign.elapsed(), "seed": seed, "arm": arm})
            return {"seed": seed, "arm": arm, "phase": campaign.state["phase"]}
        finally:
            if campaign.collector:
                campaign.collector.close()


def train():
    for arm in ARMS:
        with ProcessPoolExecutor(max_workers=3) as executor:
            for result in executor.map(train_job, [(seed, arm) for seed in range(3)]):
                append_json(ROOT / "training_completion.jsonl", result)
                print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "fixed", "timed", "summarize", "all"))
    stage = parser.parse_args().stage
    if stage in {"prepare", "all"}:
        prepare()
    if stage in {"train", "all"}:
        train()
    if stage in {"fixed", "all"}:
        evaluation.evaluate(False)
    if stage in {"timed", "all"}:
        evaluation.evaluate(True)
    if stage in {"summarize", "all"}:
        evaluation.summarize()


if __name__ == "__main__":
    main()
