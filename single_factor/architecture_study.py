"""Isolated lower-policy contributions and upper scheduling diagnostics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil

import torch

from . import forward_reward_study as evaluation
from .hierarchical.io import append_json, digest, run_lock, save_json, save_torch
from .hierarchical.models import LowerPolicy
from .hierarchical.training import Campaign, model_bundle, restore_models
from .hierarchical.ast import Grammar, Node


PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir() and (p / "pytest.ini").exists())
ROOT = PROJECT / "var/experiments/architecture_study_20261109"
PARENT = PROJECT / "var/experiments/budget_study_20261108"
COORDINATION = PROJECT / "var/experiments/coordination_study_20261109"
LOWER_METHODS = ("seed_hold", "noise_only", "random_edits", "syntax", "parent_lower",
                 "interface_control", "completion", "completion_task")


def protocol():
    return json.loads((COORDINATION / "resolved_protocol.json").read_text(encoding="utf-8"))


def prepare_lower():
    output = ROOT / "lower_only"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in range(3):
        parent = PARENT / "runs" / f"training_seed_{seed}" / "best_upper.pt"
        for method in LOWER_METHODS:
            source = (COORDINATION / "runs" / method / f"training_seed_{seed}" / "joint_final.pt"
                      if method in {"interface_control", "completion", "completion_task"} else
                      PARENT / "artifacts/syntax_initializer.pt" if method == "syntax" else parent)
            target = output / "runs" / method / f"training_seed_{seed}" / "joint_final.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            if method in {"syntax", "random_edits"}:
                base = torch.load(parent, weights_only=False, map_location="cpu")
                lower, upper = restore_models(base)
                if method == "syntax":
                    saved = torch.load(source, weights_only=False, map_location="cpu")
                    lower.load_state_dict(saved["lower"])
                else:
                    with torch.no_grad():
                        for head in (lower.node_head, lower.edit_head):
                            for parameter in head.parameters():
                                parameter.zero_()
                if not target.exists():
                    save_torch(target, model_bundle(lower, upper, {"fingerprint": base["configuration_hash"]}))
            elif not target.exists():
                shutil.copy2(source, target)
            manifest.append({"method": method, "seed": seed, "source": str(source), "checkpoint": str(target),
                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    save_json(output / "preregistration.json", {"created_at": datetime.now().astimezone().isoformat(),
        "methods": LOWER_METHODS, "training_seeds": [0, 1, 2], "tasks": ["dev_0", "dev_1"],
        "sampling_seeds": [101, 202], "attempts": 32, "total_fixed_trials": 96, "new_training": False,
        "replication": "Sampling seed 202 was added after inspecting the first 48 trials. This is a development replication, with initial results preserved.",
        "upper": "Uniform random origin selection for every method; identical initial library and common scoring protocol.",
        "seed_hold": "No perturbation and no reverse edits.", "noise_only": "Two forward perturbations, no reverse edits.",
        "random_edits": "Uniform legal position followed by uniform legal replacement, two reverse edits.",
        "syntax": "One shared structural initializer. Seed labels identify paired search randomness, not independent syntax training.",
        "financial_methods": "Selected models, their historical selection and training costs reported separately.",
        "time_interpretation": "Fixed-attempt jobs may run concurrently. Their wall times do not establish runtime speed.",
        "checkpoints": manifest})


def lower_job(job):
    seed, task, method, sampling_seed = job
    config = copy.deepcopy(protocol())
    if method in {"seed_hold", "noise_only"}:
        config["search"]["diffusion_steps"] = 0
        config["search"]["perturbation_steps"] = 0 if method == "seed_hold" else 2
    evaluation.ROOT = ROOT / "lower_only"
    evaluation.PROJECT, evaluation.PARENT = PROJECT, PARENT
    evaluation.ARMS = LOWER_METHODS
    evaluation.MODEL_CHECKPOINT = "joint_final.pt"
    evaluation.protocol = lambda: config
    original_episode = evaluation.episode

    def random_upper_episode(*args, **kwargs):
        kwargs["scheduler"] = "random"
        return original_episode(*args, **kwargs)

    evaluation.episode = random_upper_episode
    try:
        report = evaluation.evaluate_job((seed, task, method, sampling_seed, False))
        return {key: report[key] for key in ("model_seed", "task", "method", "sampling_seed", "new_factors", "forward_valid", "forward_pool_gain")}
    finally:
        evaluation.episode = original_episode


def run_lower():
    prepare_lower()
    jobs = [(seed, task, method, sampling_seed) for seed in range(3) for task in ("dev_0", "dev_1")
            for method in LOWER_METHODS for sampling_seed in (101, 202)]
    random.Random(71923).shuffle(jobs)
    with ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(lower_job, jobs):
            append_json(ROOT / "lower_only/completion.jsonl", result)
            print(json.dumps(result), flush=True)
    evaluation.ROOT, evaluation.METHODS = ROOT / "lower_only", LOWER_METHODS
    evaluation.summarize()


def prepare_training(study, arms, purpose):
    output = ROOT / study
    output.mkdir(parents=True, exist_ok=True)
    if (output / "preregistration.json").exists():
        return
    save_json(output / "preregistration.json", {"created_at": datetime.now().astimezone().isoformat(),
        "arms": arms, "seeds": [0, 1, 2], "purpose": purpose,
        "models": 6, "lower_training_batches": 8 if study == "representation" else 0,
        "upper_training_batches": 8 if study == "upper_credit" else 0,
        "sampling_seeds": [101, 202], "evaluation_attempts": 32,
        "selection": "Representation uses the final eighth lower update. Upper credit uses the same reused development criterion at 0/4/8.",
        "representation_scope": "RPN postfix GRU versus bottom-up AST encoding. Both keep the same typed reversible edit actions, corruption, PPO estimator and financial evaluator.",
        "pretraining": "Same first 96 currently valid train, 24 check and 24 holdout formulas from the existing family split. All inverse pairs regenerated with the current kernel, two epochs.",
        "shared_initialization": "Same token embeddings and shape-compatible heads, fresh optimizers. Recurrent and tree composition weights are architecture-specific.",
        "upper_credit": "Frozen parent lower. Full episode/GAE control versus immediate-reward actor advantage and critic target. Original episode returns remain logged.",
        "timing": "Fixed-attempt comparisons can run concurrently. Any timed search is serial after training and tests.",
        "limits": ["retrospective reused development", "short mechanism screen, convergence unproven", "three paired financial seeds", "historical training costs separate"]})


def prepare_compatible_corpus():
    output = ROOT / "compatible_corpus"
    if (output / "corpus_manifest.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    config = protocol()
    grammar = Grammar(**config["grammar"])
    old_path = PARENT / "artifacts/corpus/corpus.jsonl"
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    valid = []
    for record in old:
        try:
            grammar.validate(Node.from_dict(record["ast"]))
            valid.append(record)
        except ValueError:
            continue
    selected = []
    rng = random.Random(0)
    for split, count in (("train", 96), ("check", 24), ("holdout", 24)):
        candidates = [row for row in valid if row["split"] == split][:count]
        if len(candidates) != count:
            raise ValueError("insufficient valid formulas in the existing family split")
        for record in candidates:
            tree = Node.from_dict(record["ast"])
            _, trace = grammar.corrupt(tree, config["corpus"]["noise_steps"], rng)
            pairs = [{"tree": noisy.to_dict(), "step": step, "path": action.path,
                      "replacement": action.replacement.to_dict()} for noisy, step, action in trace]
            selected.append({**record, "pairs": pairs})
    settings = {**config["corpus"], "count": len(selected)}
    (output / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    save_json(output / "corpus_manifest.json", {"configuration_hash": digest([grammar.config(), settings]),
        "grammar": grammar.config(), "settings": settings,
        "families": len({row["family"] for row in selected}),
        "counts": {split: sum(row["split"] == split for row in selected) for split in ("train", "check", "holdout")},
        "source_corpus": str(old_path), "source_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "old_clean_formulas": len(old), "currently_valid_clean_formulas": len(valid),
        "repair": "Retain valid formulas and original family split. Regenerate every corruption and inverse label using the current legal kernel."})


def training_job(job):
    study, arm, seed = job
    torch.set_num_threads(1)
    config = copy.deepcopy(protocol())
    config["seed"] = seed
    config["implementation_hash"] = digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in (Path(__file__).parent / "hierarchical").glob("*.py")})
    config["upper"].update(iterations=8, joint_iterations=0, joint_guard=False, joint_rollback=False,
                           credit_assignment="immediate" if arm == "immediate" else "episode")
    config["lower"].update(task_context=False, completion_reward=False, search_training=False,
                           representation="rpn" if arm == "rpn" else "ast", iterations=8, batch_formulas=32,
                           continuation_states=8, continuation_samples=4, learning_rate=.0003)
    config["search"].update(max_attempts=32, budget_mode="attempts")
    config["rollout_workers"] = 3
    config["pretrain"].update(epochs=2, patience=3, holdout_samples=12)
    config["corpus_source"] = str(ROOT / "compatible_corpus")
    config["corpus"]["count"] = 144
    if study == "representation":
        config.pop("initializer_checkpoint", None)
    config["fingerprint"] = digest([config, study, arm, "architecture_screen_v1"])
    output = ROOT / study / "runs" / arm / f"training_seed_{seed}"
    with run_lock(output):
        campaign = Campaign(config, output, resume=(output / "training_state.pt").exists())
        try:
            source_root = str(Path(__file__).resolve().parents[1])
            import sys
            sys.path.insert(0, source_root)
            if study == "representation":
                used = []
                for split, count in (("train", 96), ("check", 24), ("holdout", 24)):
                    used.extend([row for row in campaign.records if row["split"] == split][:count])
                campaign.records = used
                save_json(output / "pretrain_subset.json", [{"id": r["id"], "split": r["split"]} for r in used])
                if campaign.state["phase"] == "corpus":
                    torch.manual_seed(seed)
                    reference = LowerPolicy(campaign.grammar, config["width"], config["corpus"]["noise_steps"])
                    shared = campaign.lower.state_dict()
                    shared.update({key: value for key, value in reference.state_dict().items()
                                   if key in shared and shared[key].shape == value.shape})
                    campaign.lower.load_state_dict(shared)
                    torch.manual_seed(48171 + seed)
                    campaign.state["phase"] = "pretrain"
                    campaign.commit()
                if campaign.state["phase"] == "pretrain":
                    campaign.syntax_pretrain()
                    save_torch(output / "syntax_after_pretrain.pt", model_bundle(campaign.lower, campaign.upper, config))
                if campaign.state["phase"] == "lower":
                    campaign.train_lower()
                save_torch(output / "joint_final.pt", model_bundle(campaign.lower, campaign.upper, config))
                syntax_target = ROOT / study / "runs" / f"{arm}_syntax" / f"training_seed_{seed}" / "joint_final.pt"
                syntax_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output / "syntax_after_pretrain.pt", syntax_target)
            else:
                if "warm_start" not in campaign.state:
                    parent = PARENT / "runs" / f"training_seed_{seed}" / "best_upper.pt"
                    saved = torch.load(parent, weights_only=False, map_location="cpu")
                    campaign.lower.load_state_dict(saved["lower"])
                    campaign.upper.load_state_dict(saved["upper"])
                    campaign.rng = random.Random(39171 + seed)
                    torch.manual_seed(39171 + seed)
                    campaign.state.update(phase="upper", lower_iteration=128, warm_start=str(parent))
                    campaign.commit()
                if campaign.state["phase"] == "upper":
                    campaign.train_upper()
                shutil.copy2(output / "upper_frozen_final.pt", output / "joint_final.pt")
            save_json(output / "cost.json", {"wall_seconds": campaign.elapsed(), "study": study, "arm": arm, "seed": seed})
            return {"study": study, "arm": arm, "seed": seed, "phase": campaign.state["phase"]}
        finally:
            if campaign.collector:
                campaign.collector.close()


def train_study(study):
    prepare_compatible_corpus()
    arms = ("ast", "rpn") if study == "representation" else ("episode", "immediate")
    prepare_training(study, arms, "Isolate lower representation" if study == "representation" else "Isolate upper credit horizon")
    for arm in arms:
        with ProcessPoolExecutor(max_workers=3) as executor:
            for result in executor.map(training_job, [(study, arm, seed) for seed in range(3)]):
                append_json(ROOT / study / "training_completion.jsonl", result)
                print(json.dumps(result), flush=True)


def study_evaluation_job(job):
    study, seed, task, arm, sampling_seed = job
    evaluation.PROJECT, evaluation.PARENT = PROJECT, PARENT
    evaluation.ROOT = ROOT / study
    arms = ("ast", "rpn", "ast_syntax", "rpn_syntax") if study == "representation" else ("episode", "immediate")
    evaluation.ARMS = arms
    evaluation.MODEL_CHECKPOINT = "joint_final.pt"
    evaluation.protocol = protocol
    original_episode = evaluation.episode

    def fixed_upper_episode(*args, **kwargs):
        if study == "representation":
            kwargs["scheduler"] = "random"
        return original_episode(*args, **kwargs)

    evaluation.episode = fixed_upper_episode
    try:
        return evaluation.evaluate_job((seed, task, arm, sampling_seed, False))
    finally:
        evaluation.episode = original_episode


def evaluate_study(study):
    arms = ("ast", "rpn", "ast_syntax", "rpn_syntax") if study == "representation" else ("episode", "immediate", "random", "parent")
    jobs = [(study, seed, task, arm, sampling_seed) for seed in range(3) for task in ("dev_0", "dev_1")
            for arm in arms for sampling_seed in (101, 202)]
    with ProcessPoolExecutor(max_workers=6) as executor:
        for report in executor.map(study_evaluation_job, jobs):
            print(json.dumps({key: report[key] for key in ("method", "model_seed", "task", "sampling_seed", "new_factors", "forward_valid", "forward_pool_gain")}), flush=True)
    evaluation.ROOT, evaluation.METHODS = ROOT / study, arms
    evaluation.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["lower", "train-representation", "train-upper", "evaluate-representation", "evaluate-upper"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "lower":
        run_lower()
    elif args.stage.startswith("train-"):
        train_study("representation" if args.stage == "train-representation" else "upper_credit")
    else:
        evaluate_study("representation" if args.stage == "evaluate-representation" else "upper_credit")
