"""Data, finite-horizon policy, trust checks and baseline isolation."""
import copy
import json
import random

import numpy as np
import pytest


def increment_under_cache_lock(job):
    import json
    import time
    from pathlib import Path
    from single_factor.hierarchical.io import cache_write_lock
    directory, iterations = job
    directory = Path(directory)
    for _ in range(iterations):
        with cache_write_lock(directory / "empty.lock"):
            path = directory / "counter.json"
            count = json.loads(path.read_text())
            time.sleep(.002)
            path.write_text(json.dumps(count + 1))


def test_spawned_processes_can_lock_empty_file_without_initialization_race(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    (tmp_path / "counter.json").write_text("0")
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as executor:
        list(executor.map(increment_under_cache_lock, [(str(tmp_path), 8)] * 4))
    assert (tmp_path / "counter.json").read_text() == "32"


def test_missing_turnover_masks_marketwide_placeholders_and_preserves_stock_zeros():
    from single_factor.prepare_budget_study import missing_turnover
    values = np.array([[0., 0., 0.], [0., .5, 0.], [0., 0., 0.]])
    volume = np.array([[10., 20., 0.], [10., 20., 0.], [0., 0., 0.]])
    original = values.copy()
    cleaned, missing = missing_turnover(values, volume)
    np.testing.assert_array_equal(missing, [True, False, False])
    assert np.isnan(cleaned[0]).all()
    np.testing.assert_array_equal(cleaned[1:], values[1:])
    np.testing.assert_array_equal(values, original)
import torch

from single_factor.prepare_budget_study import reference_panel, forward_labels
from single_factor.hierarchical.ast import Node
from single_factor.hierarchical.evaluation import load_slice, PoolStats
from single_factor.hierarchical.models import UpperPolicy
from single_factor.hierarchical.ppo import update
from single_factor.hierarchical.search import Archive
from single_factor.tests.test_hierarchical_search import evaluator, grammar, search_settings, write_panel_cache


def test_cached_tree_size_counts_repeated_subtrees_and_survives_pickle():
    import pickle
    from single_factor.hierarchical.ast import walk
    branch = Node("add", (Node("close"), Node("open")))
    tree = Node("add", (branch, branch))
    assert tree.node_count == 7
    assert tree.depth == 3
    restored = pickle.loads(pickle.dumps(tree))
    assert restored.node_count == len(list(walk(restored))) == 7
    assert restored.depth == max(len(path) + 1 for path, _ in walk(restored)) == 3


@pytest.mark.parametrize("text", [
    "close", "add(close,open)", "ts_mean(close,5)",
    "sub(ts_mean(close,5),open)", "add(add(close,open),add(close,open))",
])
def test_fast_edit_pruning_preserves_full_validated_action_set(monkeypatch, text):
    from single_factor.hierarchical.ast import Grammar
    settings = dict(fields=["open", "close", "volume"],
                    operators=["add", "sub", "div", "ts_mean", "ts_std"],
                    windows=[1, 5], constants=[0., 1.], max_nodes=7,
                    max_depth=4, max_lookback=10, strict_types=True)
    fast, reference = Grammar(**settings), Grammar(**settings)
    tree = fast.parse(text)
    actual = fast.edits(tree)
    # Disable only the new cheap pruning. The original complete validator
    # remains the oracle, including types, size, depth and rolling support.
    with monkeypatch.context() as patch:
        patch.setattr(Node, "node_count", property(lambda _: 0))
        patch.setattr(Node, "depth", property(lambda _: 0))
        expected = reference.edits(tree)
    assert actual == expected


def test_reference_split_adjustment_and_causal_future_labels():
    close = np.array([[10.], [5.], [6.], [6.], [6.]])
    raw = {k: close.copy() for k in ("open", "close", "high", "low")}
    raw.update(volume=np.full_like(close, 100), amount=close * 100, TurnoverRate=np.ones_like(close))
    reference = np.array([[10.], [5.], [5.], [6.], [6.]])
    adjusted, returns, factors = reference_panel(raw, reference)
    np.testing.assert_allclose(adjusted["close"].ravel(), [10, 10, 12, 12, 12])
    np.testing.assert_allclose(adjusted["volume"].ravel(), [100, 50, 50, 50, 50])
    assert forward_labels(returns, 2)[0, 0] == pytest.approx(.2)
    future_changed = {k: v.copy() for k, v in raw.items()}
    future_changed["close"][-1] *= 10
    alternative, _, _ = reference_panel(future_changed, reference)
    np.testing.assert_array_equal(alternative["close"][:-1], adjusted["close"][:-1])
    returns[2] = np.nan
    assert np.isnan(forward_labels(returns, 2)[:2]).all()


def test_target_sidecar_is_loaded_and_horizon_checked(tmp_path):
    e = evaluator()
    cache = write_panel_cache(e.fit, tmp_path / "panel")
    path = tmp_path / "panel"
    labels = np.tile(np.arange(24) / 100, (36, 1))
    np.save(path / "label_h2.npy", labels)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(label_file="label_h2.npy", label_horizon=2)
    (path / "manifest.json").write_text(json.dumps(manifest))
    loaded = load_slice({"cache": cache}, 2, 5, 2)
    np.testing.assert_array_equal(loaded.target, labels[5:-2])
    with pytest.raises(ValueError, match="horizon"):
        load_slice({"cache": cache}, 1, 5, 2)


def test_new_arm_outcomes_and_recent_discoveries_reach_observation():
    e = evaluator()
    a = Archive(e, [Node("close")], search_settings())
    proposal = a.assess(Node("volume"))
    assert proposal["new"]
    a.commit(proposal, "NEW")
    observation = a.observe(.3)
    assert observation.ids[0] == Node("volume").id
    assert observation.new_statistics[1] == pytest.approx(np.log(2))
    assert observation.new_statistics[2] == proposal["reward"]
    assert observation.new_statistics[3] == 1
    model = UpperPolicy(e.grammar, 8)
    distribution, value = model.distribution(observation)
    (distribution.logits[-1] + value).backward()
    assert model.actor.new_summary.weight.grad.abs().sum() > 0
    assert model.score[0].weight.grad[:, -1].abs().sum() > 0


def test_ppo_kl_limit_stops_additional_updates():
    torch.manual_seed(71)
    e = evaluator()
    model = UpperPolicy(e.grammar, 8)
    observation = Archive(e, [Node("close")], search_settings()).observe()
    rows = [model.sample(observation) for _ in range(16)]
    for i, row in enumerate(rows):
        row["return"] = float(row["index"] == 0) + i / 100
    metrics = update(model, torch.optim.Adam(model.parameters(), lr=.02), rows,
                     epochs=8, batch_size=16, target_kl=1e-6, rng=random.Random(0))
    assert metrics["kl_early_stop"] and metrics["optimizer_steps"] < 8


def test_wall_deadline_is_explicit_and_resumable(monkeypatch):
    from single_factor.hierarchical.io import BudgetExpired, check_budget
    monkeypatch.setenv("QUANTLAB_DEADLINE_EPOCH", "0")
    with pytest.raises(BudgetExpired):
        check_budget()


def test_rpn_training_reward_never_accesses_search_panel(tmp_path):
    from single_factor.hierarchical.experiments import original_ppo
    e = evaluator()
    e.search.panel._arrays.clear()
    settings = {**search_settings(), "seconds": 5.0, "max_attempts": 20}
    report = original_ppo(e, [], settings, 0, tmp_path / "rpn.jsonl", fit_only=True)
    assert not report["search_labels_used_for_policy_reward"]
    assert report["attempts"] > 0
    assert (tmp_path / "rpn.zip").exists()
    resumed = original_ppo(e, [], {**settings, "seconds": 1.0}, 17, tmp_path / "rpn_loaded.jsonl",
                           checkpoint=tmp_path / "rpn.zip", fit_only=True)
    assert torch.initial_seed() == 17 and resumed["sampling_seed"] == 17


def test_novelty_checks_all_highly_correlated_archive_members():
    # Build an actual numerical counterexample: the closest neighbor benefits,
    # but a second near neighbor already contains the candidate's information.
    found = False
    for seed in range(60):
        e = evaluator(seed)
        previous = [Node("close"), e.grammar.parse("delay(close,1)")]
        candidate = e.grammar.parse("ts_mean(close,2)")
        x = e.get(candidate).fit
        rho, gains = [], []
        for tree in previous:
            y, w = e.get(tree).fit, e.fit.weights
            norm = np.sqrt((w*x*x).sum(1) * (w*y*y).sum(1))
            daily = np.divide((w*x*y).sum(1), norm, out=np.zeros(len(x)), where=norm > 1e-24)
            rho.append(float(np.mean(abs(daily[e.fit.active]))))
            pair = PoolStats(e, [candidate, tree])
            gains.append(pair.score().ic - max(pair.subset([0]).score().ic, pair.subset([1]).score().ic))
        nearest = int(np.argmax(rho));other = 1 - nearest
        if min(rho) > .9 and gains[nearest] > gains[other] + 1e-5:
            settings = {**search_settings(), "correlation_threshold": .9,
                        "pair_gain_threshold": (gains[nearest] + gains[other]) / 2}
            accepted, details = Archive(e, previous, settings).novel(candidate)
            assert not accepted and details["redundant_to"] == previous[other].id
            assert details["neighbors_checked"] == 2
            found = True
            break
    assert found, "fixture must exercise a second-neighbor redundancy decision"
