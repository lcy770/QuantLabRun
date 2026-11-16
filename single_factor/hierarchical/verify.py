"""Reproducible real-panel combination identity and conditional-sampling checks."""
from __future__ import annotations

import argparse
import random
from time import perf_counter

import numpy as np
import torch

from .__main__ import load_config
from .ast import Grammar, replace
from .evaluation import PoolStats, load_task, normalize
from .io import event, save_json
from .models import LowerPolicy
from .training import evaluator_for


def full_panel_score(stats):
    e = stats.evaluator
    groups = sorted(set(stats.groups))
    transform = np.zeros((len(stats.trees), len(groups)))
    for j, group in enumerate(groups):
        members = [i for i, g in enumerate(stats.groups) if g == group]
        for i in members:
            transform[i, j] = stats.signs[i] / len(members)
    x = np.stack([s.fit for s in stats.signals], -1) @ transform
    search = np.stack([s.search for s in stats.signals], -1) @ transform
    g = np.einsum("dsi,dsj,ds->ij", x, x, e.fit.fit_weights)
    b = np.einsum("dsi,ds,ds->i", x, e.fit.y, e.fit.fit_weights)
    beta = np.linalg.solve(g + e.ridge * np.eye(len(groups)), b)
    z, _, _ = normalize(search @ beta, e.search.mask)
    daily = (z * e.search.y).sum(1) / e.search.count
    return daily[e.search.active].mean()


def check(config):
    if config.get("architecture") == "candidate_selection":
        return check_selection(config)
    torch.set_num_threads(config["torch_threads"])
    rng = random.Random(9)
    grammar = Grammar(**config["grammar"])
    task = next(t for t in config["tasks"] if t["role"] == "development")
    panels = load_task(task, config["target_horizon"], config["lookback"], config["future_days"])
    evaluator = evaluator_for(grammar, panels, config)
    trees = []
    for _ in range(256):
        tree = grammar.sample(rng, [t.id for t in trees])
        if evaluator.get(tree).metrics["reliable"]:
            trees.append(tree)
        if len(trees) == 6:
            break
    if len(trees) < 6:
        raise RuntimeError("insufficient reliable actual-panel factors for timing")
    base = PoolStats(evaluator, trees[:5])
    start = perf_counter()
    extended = base.extend(trees[5])
    options = [extended.subset([j for j in range(6) if j != i]) for i in range(5)]
    cached = [option.score().ic for option in options]
    cached_seconds = perf_counter() - start
    start = perf_counter()
    reference = [full_panel_score(option) for option in options]
    reference_seconds = perf_counter() - start
    np.testing.assert_allclose(cached, reference, atol=1e-10, rtol=1e-10)

    torch.manual_seed(1)
    model = LowerPolicy(grammar, config["width"], config["corpus"]["noise_steps"],
                        representation=config["lower"].get("representation", "ast"),
                        edit_backend=config["lower"].get("edit_backend", "ast"),
                        use_task=config["lower"].get("task_context", False),
                        use_position=config["lower"].get("position_context", False))
    errors, slow, fast = [], [], []
    # Warm masks are common to both timing paths; all probabilities use actual NN.
    for tree in trees[:3]:
        with torch.no_grad():
            torch.manual_seed(41)
            current, expected_rows = model.edit_kernel.from_node(tree), []
            started = perf_counter()
            for step in range(model.noise_steps, 0, -1):
                paths, grouped, positions, edits, _ = model.distributions(current, step)
                i = int(positions.sample())
                j = int(edits[i].sample())
                action = grouped[paths[i]][j]
                expected_rows.append((action, float(positions.logits[i] + edits[i].logits[j])))
                current = (model.edit_kernel.apply(current, action) if model.edit_backend == "rpn"
                           else replace(current, action.path, action.replacement))
            slow.append(perf_counter() - started)
            torch.manual_seed(41)
            started = perf_counter()
            actual, rows = model.sample(tree)
            fast.append(perf_counter() - started)
            assert model.as_node(current) == actual
            for expected, row in zip(expected_rows, rows):
                assert expected[0] == row["action"]
                errors.append(abs(expected[1] - row["logp"]))
    if max(errors) > 2e-5:
        raise AssertionError("conditional sampler probability mismatch")
    return {"task": task["name"], "edit_backend": model.edit_backend, "lower_representation": model.representation,
            "stocks": panels["fit"].panel.n_stocks,
            "fit_days": len(panels["fit"].dates), "search_days": len(panels["search"].dates),
            "combination_ic_max_error": float(np.max(abs(np.array(cached) - reference))),
            "five_combinations_cached_seconds": cached_seconds,
            "five_combinations_full_panel_seconds": reference_seconds,
            "combination_values_precomputed_for_both_methods": True,
            "sampler_logp_max_error": max(errors), "same_sampled_actions_and_endpoints": True,
            "sampling_reference_median_seconds": float(np.median(slow)),
            "sampling_conditional_median_seconds": float(np.median(fast)),
            "sampling_trials": len(fast), "configuration_hash": config["fingerprint"]}


def check_selection(config):
    """用实际任务面板独立复算等权 IC，并核验原生生成和集合策略的概率重放。"""
    import json
    from pathlib import Path
    from .ast import Node
    from .selection_evaluation import SelectionEvaluator, load_selection_task
    from .selection_models import GeneratorPolicy, SelectorPolicy
    from .selection_search import StartBank

    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(41)
    grammar = Grammar(**config["grammar"])
    training = next(task for task in config["tasks"] if task["role"] == "train")
    task = next((task for task in config["tasks"] if task["role"] == "development"), training)
    panels = load_selection_task(task, config, with_feedback=True)
    evaluator = SelectionEvaluator(grammar, panels, config)
    rng = random.Random(41)
    if config.get("initial_library"):
        trees = [Node.from_dict(json.loads(line)["ast"]) for line in
                 Path(config["initial_library"]).read_text(encoding="utf-8").splitlines()[:16]]
    else:
        trees = [grammar.sample(rng) for _ in range(16)]
    candidates, seen = [], set()
    for tree in trees:
        candidate, _ = evaluator.assess(tree, tree.id)
        metrics = candidate.search_metrics
        if candidate.direction and np.isfinite([metrics[n] for n in ("ic", "rank_ic", "icir")]).all() and candidate.signal_hash not in seen:
            candidates.append(candidate)
            seen.add(candidate.signal_hash)
        if len(candidates) == 3:
            break
    if not candidates:
        raise RuntimeError("no computable actual-panel candidates for verification")
    indices = tuple(range(min(2, len(candidates))))
    score, contributions = evaluator.contributions(candidates, indices)
    prediction = sum(evaluator.segment(candidates[i].tree, "feedback")[1] * candidates[i].direction for i in indices) / len(indices)
    daily = []
    for signal, target in zip(prediction, panels["feedback"].target):
        valid = np.isfinite(signal) & np.isfinite(target)
        value = 0.0
        if valid.sum() >= 3 and np.std(signal[valid]) > 1e-12 and np.std(target[valid]) > 1e-12:
            value = float(np.corrcoef(signal[valid], target[valid])[0, 1])
        daily.append(value)
    ic_error = abs(score["ic"] - float(np.mean(daily)))
    if ic_error > 1e-9:
        raise AssertionError(f"equal-weight IC differs from numpy: {ic_error}")
    generator = GeneratorPolicy(grammar, config["width"], config["corpus"]["noise_steps"])
    source = StartBank([candidate.tree for candidate in candidates]).observe(evaluator, config["lower"]["shortlist_size"])
    torch.manual_seed(41)
    final, rows, _ = generator.generate(source, random.Random(41), config["lower"]["diffusion_steps"],
                                       config["lower"]["perturbation_steps"], True)
    torch.manual_seed(41)
    fast, _, _ = generator.generate(source, random.Random(41), config["lower"]["diffusion_steps"],
                                    config["lower"]["perturbation_steps"], False)
    if final != fast:
        raise AssertionError("recorded/lazy generator endpoints differ")
    selector = SelectorPolicy(grammar, config["width"])
    observation = evaluator.observation(candidates, config["upper"]["max_members"])
    selection = selector.sample(observation)
    with torch.no_grad():
        errors = [abs(float(actual[0]) - row["logp"]) for actual, row in zip(generator.rows(rows), rows)]
        errors.append(abs(float(selector.rows([selection])[0][0]) - selection["logp"]))
    if max(errors) > 2e-5:
        raise AssertionError("candidate selection old-policy probability mismatch")
    return {"architecture": "candidate_selection", "task": task["name"], "stocks": panels["fit"].panel.n_stocks,
            "combination_ic_max_error": ic_error, "sampler_logp_max_error": max(errors),
            "same_sampled_actions_and_endpoints": True, "contributions": contributions,
            "member_segment": config["reward"]["member_segment"], "future_loaded": False,
            "configuration_hash": config["fingerprint"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check(load_config(args.config))
    save_json(args.output, result)
    event("numeric_verification", **result)
