"""Numerical, causal and end-to-end checks with actual policy/evaluator code."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from single_factor.hierarchical.ast import Edit, Grammar, Node, replace, walk
from single_factor.hierarchical.evaluation import Evaluator, PoolStats, Slice, WorkingPool, load_task, normalize
from single_factor.hierarchical.models import LowerPolicy, Observation, UpperPolicy
from single_factor.hierarchical.ppo import complete_returns, continuation_returns, gae_targets, update
from single_factor.hierarchical.search import Archive, episode
from single_factor.data import PanelData


torch.set_num_threads(1)


def grammar():
    return Grammar(["close", "volume", "amount"], operators=["add", "sub", "mul", "div", "ts_mean", "delay"],
                   windows=[1, 2], constants=[-1.0, 0.0, 1.0, 2.0], max_nodes=7, max_depth=5, max_lookback=5)


def slices(seed=0):
    rng = np.random.default_rng(seed)
    def panel(start):
        close = np.exp(rng.normal(0, .03, (36, 24)).cumsum(0))
        arrays = {"close": close, "volume": rng.lognormal(size=close.shape), "amount": 2 * close}
        dates = np.datetime64(start) + np.arange(len(close)).astype("timedelta64[D]")
        return Slice(PanelData(arrays, dates, list(map(str, range(24))), 5, 2), 2)
    return panel("2020-01-01"), panel("2020-05-01")


def evaluator(seed=0):
    fit, search = slices(seed)
    return Evaluator(grammar(), fit, search)


def search_settings():
    return {"pool_capacity": 2, "shortlist_size": 4, "quality_threshold": -1.0,
            "correlation_threshold": .98, "pair_gain_threshold": .001,
            "combination_weight": 50.0, "seconds": 1.0, "max_attempts": 100}


def test_frozen_lower_recording_can_be_skipped_with_identical_actions_and_rng():
    g = grammar()
    lower, upper = LowerPolicy(g, 8, 2), UpperPolicy(g, 8)
    config = {**search_settings(), "seconds": 30., "max_attempts": 8, "budget_mode": "attempts"}
    initial = [Node("close"), Node("volume")]
    outcomes = []
    for recording in (True, False):
        torch.manual_seed(541)
        evaluator = Evaluator(g, *slices())
        rows, trajectories, scores, report = episode(lower, upper, evaluator, initial, config, random.Random(719),
                                                    record_lower=recording)
        outcomes.append((rows, report, torch.get_rng_state()))
        assert bool(trajectories) == recording
        assert bool(scores) == recording
    first, second = outcomes
    for a, b in zip(first[0], second[0]):
        for key in ("index", "logp", "value", "reward", "return", "observation"):
            assert a[key] == b[key]
    for key in ("discoveries", "pool_asts", "pool_weights", "return", "new_factors", "attempts", "duplicate_rate"):
        assert first[1][key] == second[1][key]
    assert torch.equal(first[2], second[2])


def test_fixed_attempt_budget_is_independent_of_wall_reference():
    g = grammar()
    lower, upper = LowerPolicy(g, 8, 2), UpperPolicy(g, 8)
    settings = {**search_settings(), "seconds": 1e-12, "max_attempts": 3, "budget_mode": "attempts"}
    rows, _, _, report = episode(lower, upper, Evaluator(g, *slices()), [Node("close"), Node("volume")],
                                 settings, random.Random(41), record_lower=False)
    assert len(rows) == report["attempts"] == 3
    assert report["late_results_excluded"] == 0
    assert report["budget_mode"] == "attempts"
    assert report["wall_seconds"] > settings["seconds"]


def test_forward_training_relabels_targets_with_locked_pools_and_unchanged_observations(monkeypatch):
    from single_factor.hierarchical.experiments import future_check
    from single_factor.hierarchical.training import forward_training_rewards

    e = evaluator()
    first, second = Node("close"), Node("volume")
    discovery = e.get(second).metrics
    observation = Observation((first.id,), (first,), ((0.,) * 7,), (first,), (1.,), 0., 1.)
    following = Observation((second.id,), (second,), ((0.,) * 7,), (first, second), (.25, -.75), 0., .5)
    original = [{"observation": observation, "reward": 1.2, "value": .1},
                {"observation": following, "reward": 0., "value": .2}]
    report = {"discoveries": [{"ast": second.to_dict(), **discovery}], "return": 1.2,
              "pool_asts": [first.to_dict(), second.to_dict()], "pool_weights": [.25, -.75]}
    final = slices(71)[1]
    initial = {"discoveries": [], "pool_asts": [first.to_dict()], "pool_weights": [1.]}
    e.get(first)
    expected_initial = future_check(initial, e, final)["pool_ic"]
    expected_final = future_check(report, e, final)["pool_ic"]
    # A refit would silently change the teacher objective.
    monkeypatch.setattr(PoolStats, "score", lambda *a, **k: pytest.fail("teacher refitted weights"))
    settings = {**search_settings(), "min_coverage": .8, "quality_threshold": .02}
    outcomes = []
    for sign in (1, -1):
        target = copy.deepcopy(final)
        target.y = sign * final.y
        rows, result = copy.deepcopy(original), copy.deepcopy(report)
        forward_training_rewards(rows, result, e, target, settings)
        assert [r["observation"] for r in rows] == [observation, following]
        assert [r["search_reward"] for r in rows] == [1.2, 0.]
        assert result["return"] == report["return"]
        assert result["pool_weights"] == report["pool_weights"]
        assert result["discoveries"][0]["direction"] == discovery["direction"]
        assert rows[1]["reward"] == pytest.approx(0., abs=1e-12)
        assert result["training_initial_forward_ic"] == pytest.approx(sign * expected_initial)
        assert result["training_forward"]["pool_ic"] == pytest.approx(sign * expected_final)
        assert not result["training_forward"]["weights_refitted"]
        expected = result["training_forward_valid"] + 50 * sign * (expected_final - expected_initial)
        assert rows[0]["return"] == pytest.approx(expected)
        assert sum(r["reward"] for r in rows) == pytest.approx(expected)
        outcomes.append(rows[0]["return"])
    assert outcomes[0] != pytest.approx(outcomes[1])


def test_forward_training_requires_all_historical_labels_to_mature_before_development():
    from types import SimpleNamespace
    from single_factor.hierarchical.training import validate_forward_training
    tasks = [{"name": "train", "role": "train"}, {"name": "dev", "role": "development"}]
    panels = {"train": {"future": SimpleNamespace(ends=np.array([np.datetime64("2024-06-28")]))},
              "dev": {"search": SimpleNamespace(dates=np.array([np.datetime64("2024-07-01")]))}}
    validate_forward_training(tasks, panels)
    panels["train"]["future"].ends[0] = np.datetime64("2024-07-01")
    with pytest.raises(ValueError, match="reach"):
        validate_forward_training(tasks, panels)
    panels["train"].clear()
    with pytest.raises(ValueError, match="reach"):
        validate_forward_training(tasks, panels)


def test_archive_background_includes_seen_formulas_outside_shortlist():
    e = evaluator()
    config = {**search_settings(), "shortlist_size": 1}
    archive = Archive(e, [Node("close"), Node("volume")], config)
    candidate = e.grammar.parse("delay(close,1)")
    archive.commit(archive.assess(candidate), Node("close").id)
    observation = archive.observe(include_archive=True)
    assert len(observation.trees) == 1
    assert {tree.id for tree in observation.archive_trees} == archive.seen
    assert candidate.id in {tree.id for tree in observation.archive_trees}


def test_archive_policy_migrates_legacy_weights_and_learns_background():
    from dataclasses import replace as replace_observation
    from single_factor.hierarchical.training import model_bundle, restore_models
    g, width = grammar(), 8
    legacy = UpperPolicy(g, width)
    current = UpperPolicy(g, width, use_archive=True)
    current.load_state_dict(legacy.state_dict())
    observation = Observation(("c", "v"), (Node("close"), Node("volume")), ((0.,) * 7, (0.,) * 7),
                              (Node("close"),), (1.,), .03, archive_trees=(Node("close"), Node("volume"), Node("amount")))
    old_distribution, old_value = legacy.distribution(observation)
    distribution, value = current.distribution(observation)
    assert torch.allclose(distribution.probs, old_distribution.probs, atol=2e-6)
    assert torch.allclose(value, old_value, atol=2e-6)
    (-distribution.logits[0] + value).backward()
    assert current.value_head[0].weight.grad[:, 2 * width + 2:].abs().sum() > 0
    with torch.no_grad():
        current.value_head[0].weight[:, 2 * width + 2:].fill_(.2)
    shuffled = replace_observation(observation, archive_trees=tuple(reversed(observation.archive_trees)))
    assert torch.allclose(current.distribution(observation)[1], current.distribution(shuffled)[1], atol=2e-6)
    empty = replace_observation(observation, archive_trees=())
    assert not torch.allclose(current.distribution(observation)[1], current.distribution(empty)[1], atol=1e-7)
    bundle = model_bundle(LowerPolicy(g, width, 2), current, {"fingerprint": "test"})
    _, restored = restore_models(bundle)
    assert restored.use_archive and bundle["format_version"] == 4
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in current.state_dict().items())


def test_forward_inverse_support_and_topology_changes():
    g, rng = grammar(), random.Random(71)
    changed_topology = False
    for _ in range(20):
        tree = g.sample(rng)
        noisy, trace = g.corrupt(tree, 3, rng)
        for state, _, inverse in reversed(trace):
            assert inverse in g.edits(state)
            previous = replace(state, inverse.path, inverse.replacement)
            changed_topology |= len(list(walk(previous))) != len(list(walk(state)))
            noisy = previous
        assert noisy == tree
    assert changed_topology


def test_typed_masks_reject_future_and_bad_units_and_canonical_duplicates():
    g = grammar()
    with pytest.raises(ValueError):
        g.parse("add(close,volume)")
    with pytest.raises(ValueError):
        g.parse("delay(close,-1)")
    relaxed = Grammar(["close", "volume"], operators=["add"], windows=[1], constants=[1.0], strict_types=False)
    assert relaxed.parse("add(close,volume)").id == relaxed.parse("add(volume,close)").id
    # Actual signal execution uses the existing past-only rolling implementation.
    e = evaluator()
    tree = e.grammar.parse("delay(close,2)")
    values = e.grammar.compile(tree).evaluate(e.fit.panel).numpy()
    np.testing.assert_allclose(values, e.fit.panel._arrays["close"][3:3 + e.fit.panel.n_days], rtol=1e-6)


def test_lower_probabilities_normalize_and_replay_without_missing_factors():
    g = grammar()
    model = LowerPolicy(g, 8, 2)
    tree = g.parse("div(close,amount)")
    paths, groups, positions, edits, _ = model.distributions(tree, 2)
    probability = sum(float((positions.probs[i] * distribution.probs).sum().detach()) for i, distribution in enumerate(edits))
    assert probability == pytest.approx(1, abs=1e-6)
    final, rows = model.sample(tree)
    for row in rows:
        assert float(model.row(row)[0].detach()) == pytest.approx(row["logp"], abs=1e-6)
    g.validate(final)


def test_continuation_fixes_action_and_decrements_remaining_horizon():
    torch.manual_seed(4)
    g = Grammar(["close", "volume"], operators=["add"], windows=[1], constants=[1.0],
                strict_types=False, max_nodes=3, max_depth=2, max_lookback=2)
    model = LowerPolicy(g, 8, 2)
    _, rows = model.sample(Node("close"))
    def terminal(tree):
        return float(tree.symbol == "volume") + .25 * float(bool(tree.children))
    paths, grouped, position, edits, _ = model.distributions(rows[0]["after"], 1)
    expected = sum(float((position.probs[i] * edits[i].probs[j]).detach()) * terminal(replace(rows[0]["after"], path, a.replacement))
                   for i, path in enumerate(paths) for j, a in enumerate(grouped[path]))
    reward = terminal(rows[-1]["after"])
    branches = continuation_returns(model, [rows], [reward], terminal, samples=1024, states=1, rng=random.Random(0))
    assert branches == 1023
    assert rows[0]["return"] == pytest.approx(expected, abs=.04)
    assert len(rows[0]["continuation_scores"]) == 1024
    assert rows[1]["return"] == reward


def brute_score(stats):
    e = stats.evaluator
    ids = list(dict.fromkeys(stats.groups))
    transform = np.zeros((len(stats.trees), len(ids)))
    for j, group in enumerate(ids):
        members = [i for i, g in enumerate(stats.groups) if g == group]
        for i in members:
            transform[i, j] = stats.signs[i] / len(members)
    x = np.stack([s.fit for s in stats.signals], -1) @ transform
    z = np.stack([s.search for s in stats.signals], -1) @ transform
    gram = np.einsum("dsi,dsj,ds->ij", x, x, e.fit.fit_weights)
    rhs = np.einsum("dsi,ds,ds->i", x, e.fit.y, e.fit.fit_weights)
    beta = np.linalg.solve(gram + e.ridge * np.eye(len(ids)), rhs)
    prediction, _, _ = normalize(z @ beta, e.search.mask)
    ic = ((prediction * e.search.y).sum(1) / e.search.count)[e.search.active].mean()
    return transform @ beta, ic


def test_cached_ridge_ic_matches_full_panel_with_duplicates_missing_and_replacement():
    e = evaluator()
    # Finite/missing values and signed aliases exercise the actual normalizer.
    e.fit.panel._arrays["volume"][::10, ::5] = np.nan
    e.search.panel._arrays["volume"][::10, ::5] = np.nan
    trees = [Node("close"), Node("amount"), Node("volume"), e.grammar.parse("div(volume,close)")]
    stats = PoolStats(e)
    for tree in trees:
        stats = stats.extend(tree)
    for deleted in range(4):
        option = stats.subset([i for i in range(4) if i != deleted])
        actual = option.score()
        weights, ic = brute_score(option)
        np.testing.assert_allclose(actual.weights, weights, atol=1e-10)
        assert actual.ic == pytest.approx(ic, abs=1e-10)


def test_search_labels_cannot_change_fitted_weights_or_fit_direction():
    normal, reversed_labels = evaluator(), evaluator()
    reversed_labels.search.target *= -1
    reversed_labels.search.y *= -1
    trees = [Node("close"), Node("volume")]
    first, second = PoolStats(normal, trees).score(), PoolStats(reversed_labels, trees).score()
    np.testing.assert_allclose(first.weights, second.weights, atol=1e-12)
    assert first.ic == pytest.approx(-second.ic, abs=1e-12)
    for tree in trees:
        assert normal.metrics[tree.id]["direction"] == reversed_labels.metrics[tree.id]["direction"]


def test_working_pool_never_decreases_and_exact_alias_cannot_relax_ridge():
    e = evaluator()
    original = PoolStats(e, [Node("close")]).score()
    duplicate = PoolStats(e, [Node("close"), Node("amount")]).score()
    assert duplicate.ic == pytest.approx(original.ic, abs=1e-12)
    pool = WorkingPool(e, 2, [Node("close")])
    old = pool.score.ic
    gain = pool.consider(Node("volume"))
    assert gain >= 0 and pool.score.ic >= old
    with pytest.raises(ValueError, match="unreliable"):
        PoolStats(e, [e.grammar.parse("div(close,amount)")])


def test_discovery_count_reward_is_independent_of_combination_improvement():
    e = evaluator()
    archive = Archive(e, [Node("close")], search_settings())
    proposal = archive.assess(Node("volume"))
    assert proposal["new"]
    assert proposal["reward"] == pytest.approx(1 + 50 * proposal["gain"])
    archive.commit(proposal, Node("close").id)
    assert not archive.assess(Node("volume"))["new"]
    assert archive.assess(Node("amount"))["duplicate"]
    assert not archive.assess(Node("close"))["new"]
    assert archive.entries[Node("close").id].attempts == 1


def test_upper_set_invariance_gradient_and_layer_freezing():
    g = grammar()
    lower, upper = LowerPolicy(g, 8, 2), UpperPolicy(g, 8)
    stats = ((.02, 0, 0, 0, 0, 1, 1), (.01, 1, .2, .1, .3, 0, 0))
    observation = Observation(("a", "b"), (Node("close"), Node("volume")), stats,
                              (Node("close"), Node("volume")), (.2, -.1), .03)
    swapped = Observation(("b", "a"), tuple(reversed(observation.trees)), tuple(reversed(stats)),
                          tuple(reversed(observation.pool_trees)), tuple(reversed(observation.pool_weights)), .03)
    p, value = upper.distribution(observation)
    q, qvalue = upper.distribution(swapped)
    np.testing.assert_allclose(p.probs.detach(), q.probs.detach()[[1, 0, 2]], atol=1e-7)
    assert float(value.detach()) == pytest.approx(float(qvalue.detach()), abs=1e-7)
    rewards = torch.tensor([0.1, .3, .8])
    expectation = (p.probs * rewards).sum()
    expectation.backward()
    parameter = upper.actor.new
    analytic = float(parameter.grad[0])
    with torch.no_grad():
        original = float(parameter[0])
        parameter[0] = original + .001
        plus = float((upper.distribution(observation)[0].probs * rewards).sum())
        parameter[0] = original - .001
        minus = float((upper.distribution(observation)[0].probs * rewards).sum())
        parameter[0] = original
    assert analytic == pytest.approx((plus - minus) / .002, abs=1e-4)
    rows = [upper.sample(observation) for _ in range(4)]
    for i, row in enumerate(rows):
        row["return"] = i / 10
    before = {k: v.clone() for k, v in lower.state_dict().items()}
    metrics = update(upper, torch.optim.Adam(upper.parameters(), lr=.001), rows, epochs=1, batch_size=4)
    assert metrics["old_logp_max_error"] < 1e-6
    assert all(torch.equal(v, lower.state_dict()[k]) for k, v in before.items())


def test_complete_episode_returns_cross_formula_boundaries():
    rows = [{"reward": v} for v in (0.0, 1.1, 0.0, 1.2)]
    assert complete_returns(rows) == pytest.approx(2.3)
    assert [r["return"] for r in rows] == pytest.approx([2.3, 2.3, 1.2, 1.2])


def test_gae_matches_hand_computation_and_monte_carlo_limit():
    rows = [{"reward": r, "value": v} for r, v in zip((0., 1., 2.), (.5, .7, .2))]
    complete_returns(rows)
    gae_targets(rows, .5)
    # TD residuals: [.2, .5, 1.8]; zero bootstrap after the true episode end.
    assert [r["advantage"] for r in rows] == pytest.approx([.9, 1.4, 1.8])
    assert [r["value_target"] for r in rows] == pytest.approx([1.4, 2.1, 2.])
    assert [r["return"] for r in rows] == [3., 3., 2.]
    gae_targets(rows, 1.)
    assert [r["advantage"] for r in rows] == pytest.approx([r["return"] - r["value"] for r in rows])
    gae_targets(rows, 0.)
    assert [r["advantage"] for r in rows] == pytest.approx([.2, .5, 1.8])
    with pytest.raises(ValueError):
        gae_targets(rows, 1.1)


def test_budget_context_and_legacy_checkpoint_are_compatible():
    from dataclasses import replace as changed
    from single_factor.hierarchical.training import model_bundle, restore_models
    g = grammar()
    lower, upper = LowerPolicy(g, 8, 2), UpperPolicy(g, 8)
    # A legacy actor has no budget column. Zero extension preserves it exactly.
    with torch.no_grad():
        upper.score[0].weight[:, -1] = 0
    o = Observation(("a",), (Node("close"),), ((.1, 0, 0, 0, 0, 1, 1),), (), (), 0., .9)
    later = changed(o, remaining_fraction=.1)
    p, _ = upper.distribution(o)
    q, value = upper.distribution(later)
    torch.testing.assert_close(p.probs, q.probs)
    value.backward()
    assert upper.value_head[0].weight.grad[:, -1].abs().sum() > 0
    bundle = model_bundle(lower, upper, {"fingerprint": "test"})
    bundle["format_version"] = 1
    bundle["upper"] = dict(bundle["upper"])
    bundle["upper"]["value_head.0.weight"] = bundle["upper"]["value_head.0.weight"][:, :-1]
    bundle["upper"]["score.0.weight"] = bundle["upper"]["score.0.weight"][:, :-1]
    for tower in ("actor", "critic"):
        bundle["upper"].pop(f"{tower}.new_summary.weight")
    _, migrated = restore_models(bundle)
    _, c = upper.critic(o)
    old_input = torch.nn.functional.linear(c, bundle["upper"]["value_head.0.weight"], upper.value_head[0].bias)
    expected_value = upper.value_head[2](torch.tanh(old_input)).squeeze()
    for obs in (o, later):
        distribution, actual = migrated.distribution(obs)
        torch.testing.assert_close(actual, expected_value)
        torch.testing.assert_close(distribution.probs, p.probs)


def test_vectorized_novelty_prefix_matches_full_daily_correlations():
    e = evaluator()
    e.fit.panel._arrays["volume"][::9, ::4] = np.nan
    initial = [Node("close"), Node("volume")]
    archive = Archive(e, initial, search_settings())
    candidate = e.grammar.parse("div(volume,close)")
    signal = e.get(candidate)
    rho = archive.prefix_correlations(signal)
    indices = archive.novelty_indices[:8]
    for number, tree in enumerate(initial):
        x, y, w = signal.fit[indices], e.get(tree).fit[indices], e.fit.weights[indices]
        denominator = np.sqrt((w * x * x).sum(1) * (w * y * y).sum(1))
        expected = np.divide((w*x*y).sum(1), denominator, out=np.zeros(len(indices)), where=denominator > 1e-24)
        np.testing.assert_allclose(rho[number], expected, atol=1e-12)
    # Low threshold forces full evaluation; check the exact closest correlation.
    archive.config["correlation_threshold"] = .01
    _, detail = archive.novel(candidate)
    exact = []
    for tree in initial:
        x, y, w = signal.fit, e.get(tree).fit, e.fit.weights
        denominator = np.sqrt((w * x*x).sum(1) * (w * y*y).sum(1))
        daily = np.divide((w*x*y).sum(1), denominator, out=np.zeros(len(x)), where=denominator > 1e-24)
        exact.append(np.clip(abs(daily[e.fit.active]), 0, 1).mean())
    assert detail["rho"] == pytest.approx(max(exact), abs=1e-12)


def test_initialization_and_late_work_cannot_escape_time_budget():
    e = evaluator()
    g = e.grammar
    config = {**search_settings(), "seconds": 1e-9}
    rows, _, _, report = episode(LowerPolicy(g, 8, 2), UpperPolicy(g, 8), e,
                                 [Node("close")], config, random.Random(0))
    assert not rows and report["new_factors"] == 0
    assert report["wall_seconds"] > config["seconds"]


def test_scoring_slice_rejects_label_overlap(tmp_path):
    fit, search = slices()
    def write(scoring, name):
        root = tmp_path / name
        root.mkdir()
        for field, values in scoring.panel._arrays.items():
            np.save(root / f"{field}.npy", values)
        np.save(root / "dates.npy", scoring.panel.dates.values)
        (root / "codes.json").write_text(json.dumps(scoring.panel.stock_ids))
        (root / "manifest.json").write_text(json.dumps({"fields": list(scoring.panel._arrays)}))
        return {"cache": str(root)}
    spec = write(fit, "fit")
    task = {"fit": spec, "search": spec}
    with pytest.raises(ValueError, match="overlap"):
        load_task(task, 2, 5, 2)


def test_lower_reward_never_reads_search_labels_or_search_features():
    baseline, poisoned = evaluator(), evaluator()
    poisoned.search.target[:] = np.nan
    poisoned.search.y[:] = np.nan
    # A malformed search panel would fail formula execution if it were accessed.
    poisoned.search.panel._arrays.clear()
    tree = Node("close")
    assert poisoned.lower(tree) == pytest.approx(baseline.lower(tree), abs=1e-12)
    assert poisoned.calls == 1
    assert poisoned.lower(tree) == baseline.lower(tree)
    assert poisoned.calls == 1 and poisoned.hits == 1


def test_lower_expected_reward_gradient_matches_finite_difference():
    torch.manual_seed(12)
    g = Grammar(["close", "volume"], operators=["add"], windows=[1], constants=[1.0],
                strict_types=False, max_nodes=3, max_depth=2, max_lookback=2)
    model, tree = LowerPolicy(g, 8, 1), Node("close")
    def expectation():
        paths, grouped, positions, edits, _ = model.distributions(tree, 1)
        return sum(positions.probs[i] * edits[i].probs[j] *
                   (float(a.replacement.symbol == "volume") + .3 * len(a.replacement.children))
                   for i, path in enumerate(paths) for j, a in enumerate(grouped[path]))
    expectation().backward()
    parameter = model.edit_head[2].weight
    analytical = float(parameter.grad[0, 0])
    with torch.no_grad():
        original = float(parameter[0, 0])
        parameter[0, 0] = original + .001
        plus = float(expectation())
        parameter[0, 0] = original - .001
        minus = float(expectation())
        parameter[0, 0] = original
    assert analytical == pytest.approx((plus - minus) / .002, abs=5e-5)


def test_batched_lower_has_same_probabilities_entropy_values_and_gradients():
    torch.manual_seed(71)
    g = grammar()
    reference = LowerPolicy(g, 8, 2)
    batched = copy.deepcopy(reference)
    rows = []
    for tree in (Node("close"), Node("volume"), g.parse("div(close,amount)")):
        rows.extend(reference.sample(tree)[1])
    expected = [reference.row(row) for row in rows]
    actual = batched.rows(rows)
    for first, second in zip(expected, actual):
        torch.testing.assert_close(torch.stack(first), torch.stack(second), atol=2e-6, rtol=2e-6)
    sum(v[0] + .1 * v[1] + .01 * v[2] for v in expected).backward()
    sum(v[0] + .1 * v[1] + .01 * v[2] for v in actual).backward()
    for (_, first), (_, second) in zip(reference.named_parameters(), batched.named_parameters()):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, atol=1e-5, rtol=2e-5)


def test_persistent_signal_cache_matches_original_and_is_memory_mapped(tmp_path):
    first = evaluator()
    first.signal_store = tmp_path
    tree = Node("close")
    original = first.get(tree)
    second = evaluator()
    second.signal_store = tmp_path
    restored = second.get(tree)
    assert second.disk_hits == 1 and second.calls == 0
    assert isinstance(restored.fit, np.memmap)
    np.testing.assert_array_equal(original.fit, restored.fit)
    assert original.metrics == restored.metrics


def test_generated_signal_cache_is_isolated_between_episodes(tmp_path):
    shared = evaluator()
    shared.signal_store = tmp_path
    initial, generated = Node("close"), Node("volume")
    shared.get(initial)
    shared.get(generated)  # Even a pre-existing generated cache must be ignored.
    first, second = shared.for_episode([initial]), shared.for_episode([initial])
    for trial in (first, second):
        trial.get(initial)
        assert trial.disk_hits == 1 and trial.calls == 0
        trial.get(generated)
        assert trial.disk_hits == 1 and trial.calls == 1
        trial.get(generated)
        assert trial.calls == 1  # Within-episode memoization remains available.
    assert first.signals is not second.signals
    assert first.metrics is not shared.metrics


def test_public_signals_stay_pinned_but_generated_cache_remains_bounded():
    e = evaluator()
    initial = [Node("close"), Node("volume"), Node("amount")]
    trial = e.for_episode(initial)
    trial.cache_size = 2
    for tree in initial:
        trial.get(tree)
    calls = trial.calls
    for tree in initial:
        trial.get(tree)
    assert trial.calls == calls and len(trial.public_signals) == 3
    for expression in ("div(volume,close)", "delay(close,1)", "delay(close,2)"):
        trial.get(e.grammar.parse(expression))
    assert len(trial.signals) == 2 and len(trial.public_signals) == 3


def test_rolling_rank_and_sign_preserve_undefined_values():
    from single_factor._vendor.alphagen.data.expression import Rank, Sign
    values = torch.tensor([[[1., 3., 2.], [2., 2., 2.], [1., float("nan"), 2.],
                            [1., float("inf"), 2.], [1., 2., float("nan")]]])
    actual = Rank(1., 3)._apply(values)
    # Preserve the original AlphaGen rank convention on fully finite windows.
    torch.testing.assert_close(actual[0, :2], torch.tensor([2 / 3, 2 / 3]))
    assert torch.isnan(actual[0, 2:]).all()
    signed = Sign(1.)._apply(torch.tensor([-2., 0., 2., float("nan"), float("inf")]))
    torch.testing.assert_close(signed[:3], torch.tensor([-1., 0., 1.]))
    assert torch.isnan(signed[3:]).all()


def test_correlation_is_scale_invariant_and_zero_variance_is_undefined():
    from single_factor._vendor.alphagen.data.expression import Corr
    lhs = torch.tensor([[[1., 4., 2., 8.], [1., 1., 1., 1.],
                          [1., 2., float("nan"), 4.]]])
    rhs = torch.tensor([[[4., 1., 3., 2.], [1., 2., 3., 4.], [1., 2., 3., 4.]]])
    corr = Corr(1., 1., 4)
    expected = np.corrcoef(lhs[0, 0].numpy(), rhs[0, 0].numpy())[0, 1]
    for scale in (1., 1e-18, 1e18):
        actual = corr._apply(lhs * scale, rhs * scale)
        assert actual[0, 0].item() == pytest.approx(expected, abs=2e-6)
        assert torch.isnan(actual[0, 1:]).all()
    assert torch.isnan(Corr(1., 1., 1)._apply(lhs[..., :1], rhs[..., :1])).all()


def test_binary_operations_cannot_sanitize_nonfinite_operands():
    e = evaluator()
    back = e.fit.panel.max_backtrack_days
    e.fit.panel._arrays["amount"][back, 0] = np.inf
    e.fit.panel._arrays["amount"][back, 1] = np.nan
    values = e.grammar.compile(e.grammar.parse("div(close,amount)")).evaluate(e.fit.panel)
    assert torch.isnan(values[0, :2]).all()  # Finite/Inf must not become a signal of zero.
    assert torch.isfinite(values[0, 2:]).all()


def test_nested_undefined_formulas_cannot_receive_financial_reward():
    rng = np.random.default_rng(42)
    close = np.exp(rng.normal(0, .03, (160, 24)).cumsum(0))
    dates = np.datetime64("2020-01-01") + np.arange(len(close)).astype("timedelta64[D]")
    scoring = Slice(PanelData({"close": close, "high": close * 1.05, "open": close * .99},
                              dates, list(map(str, range(24))), 100, 2), 2)
    g = Grammar(["close", "high", "open"],
                operators=["mul", "div", "delta", "ts_corr", "ts_rank", "ts_std"],
                windows=[1, 5, 20, 40], constants=[0.0], max_nodes=31, max_depth=12, max_lookback=100)
    e = Evaluator(g, scoring, scoring)
    invalid = Node("ts_rank", (Node("ts_corr", (Node("high"), Node("close"), Node("window", value=1))), Node("window", value=40)))
    with pytest.raises(ValueError, match="two observations"):
        g.parse("ts_rank(ts_corr(high,close,1),40)")
    with pytest.raises(ValueError, match="zero"):
        g.parse("mul(open,ts_rank(div(delta(close,5),ts_std(delta(0.0,5),20)),40))")
    for tree in (invalid,):
        assert e.lower(tree) == 0


def write_panel_cache(scoring, root):
    root.mkdir(parents=True)
    for field, values in scoring.panel._arrays.items():
        np.save(root / f"{field}.npy", values)
    np.save(root / "dates.npy", scoring.panel.dates.values)
    (root / "codes.json").write_text(json.dumps(scoring.panel.stock_ids))
    (root / "manifest.json").write_text(json.dumps({"fields": list(scoring.panel._arrays)}))
    return str(root)


def tiny_training_config(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    fit, search = slices()
    config = json.loads((Path(__file__).parents[1] / "hierarchical_config.fullcheck.json").read_text())
    config.update(target_horizon=2, lookback=5, future_days=2, width=8, torch_threads=1)
    config["grammar"] = grammar().config()
    config["corpus"].update(count=24, noise_steps=2)
    config["pretrain"].update(epochs=1, holdout_samples=2)
    config["lower"].update(iterations=1, batch_formulas=2, continuation_states=1)
    config["upper"].update(iterations=1, joint_iterations=1, episodes_per_batch=1)
    for layer in ("lower", "upper"):
        config[layer]["ppo"]["epochs"] = 1
    config["search"].update(seconds=.5, pool_capacity=2, initial_seeds=4, shortlist_size=4)
    config["evaluation"].update(seconds=.5, methods=["upper_ppo"])
    config["tasks"] = [{"name": "training", "role": "train",
                         "fit": {"cache": write_panel_cache(fit, tmp_path / "fit")},
                         "search": {"cache": write_panel_cache(search, tmp_path / "search")}}]
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(config))
    return load_config(path)


@pytest.mark.parametrize("workers", [1, 2])
def test_forward_teacher_trains_through_campaign_with_serial_and_parallel_collection(tmp_path, workers):
    from single_factor.hierarchical.training import Campaign
    config = tiny_training_config(tmp_path)
    final = copy.deepcopy(slices(41)[1])
    final.panel.dates = final.panel.dates + np.timedelta64(120, "D")
    config["tasks"][0]["future"] = {"cache": write_panel_cache(final, tmp_path / "teacher")}
    later = copy.deepcopy(slices(51)[1])
    later.panel.dates = later.panel.dates + np.timedelta64(365, "D")
    config["tasks"].append({"name": "development", "role": "development",
                            "fit": config["tasks"][0]["fit"],
                            "search": {"cache": write_panel_cache(later, tmp_path / "dev")}})
    config["upper"].update(training_reward="forward", joint_iterations=0, gae_lambda=.95)
    config["search"].update(max_attempts=3, budget_mode="attempts")
    config["rollout_workers"] = workers
    campaign = Campaign(config, tmp_path / "teacher_run")
    try:
        campaign.train_upper()
    finally:
        if campaign.collector:
            campaign.collector.close()
    episodes = [json.loads(line) for line in (campaign.output / "training_episodes.jsonl").read_text().splitlines()]
    trajectories = [json.loads(line) for line in (campaign.output / "upper_trajectories.jsonl").read_text().splitlines()]
    assert episodes[0]["training_reward"] == "forward"
    assert episodes[0]["attempts"] == 3
    assert len(trajectories[0]["actions"]) == 3
    actions = trajectories[0]["actions"]
    assert sum(a["search_reward"] for a in actions) == pytest.approx(episodes[0]["return"])
    assert sum(a["reward"] for a in actions) == pytest.approx(episodes[0]["training_return"])
    assert all("value_target" in action for action in actions)


def test_checkpoint_restores_optimizer_models_and_all_rng_and_rejects_mismatch(tmp_path):
    from single_factor.hierarchical.training import Campaign
    config = tiny_training_config(tmp_path)
    campaign = Campaign(config, tmp_path / "run")
    _, rows = campaign.lower.sample(Node("close"))
    for row in rows:
        row["return"] = .2
    update(campaign.lower, campaign.lower_optimizer, rows, epochs=1)
    campaign.state["phase"] = "lower"
    campaign.commit()
    expected = (campaign.rng.random(), np.random.rand(), torch.rand(4))
    restored = Campaign(config, tmp_path / "run", resume=True)
    assert restored.rng.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])
    for name, value in campaign.lower.state_dict().items():
        assert torch.equal(restored.lower.state_dict()[name], value)
    original_state = campaign.lower_optimizer.state_dict()
    restored_state = restored.lower_optimizer.state_dict()
    assert original_state["param_groups"] == restored_state["param_groups"]
    for key, state in original_state["state"].items():
        for name, value in state.items():
            assert torch.equal(restored_state["state"][key][name], value)
    with pytest.raises(ValueError, match="fingerprint"):
        Campaign({**config, "fingerprint": "different"}, tmp_path / "run", resume=True)


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("every", [1, 2])
def test_joint_tied_validation_preserves_working_updates_and_separate_best(tmp_path, monkeypatch, rollback, every):
    from single_factor.hierarchical.io import save_torch
    from single_factor.hierarchical.training import Campaign, model_bundle
    config = tiny_training_config(tmp_path)
    config["upper"].update(joint_iterations=2, joint_guard=True, joint_rollback=rollback)
    config["search"].update(seconds=30, max_attempts=2, budget_mode="attempts")
    config["upper_validation"] = {"every": every}
    campaign = Campaign(config, tmp_path / "joint")
    initial = {name: value.clone() for name, value in campaign.upper.state_dict().items()}

    def tied_validation(stage, iteration):
        if iteration == 0:
            campaign.state["best_joint_score"] = 0.
            save_torch(campaign.output / "best_joint.pt", model_bundle(campaign.lower, campaign.upper, config,
                       selected_iteration=0, validation_score=0., lower_optimizer=campaign.lower_optimizer.state_dict(),
                       upper_optimizer=campaign.upper_optimizer.state_dict()))
        return 0.

    monkeypatch.setattr(campaign, "validate_upper", tied_validation)
    campaign.train_upper(joint=True)
    working = torch.load(campaign.output / "joint_working_final.pt", weights_only=False)
    steps = [int(value["step"]) for value in working["upper_optimizer"]["state"].values()]
    if rollback:
        assert not steps
    else:
        assert max(steps) >= 2
        assert any(not torch.equal(value, working["upper"][name]) for name, value in initial.items())
        assert "python_rng" in working and "torch_rng" in working and "numpy_rng" in working
    assert all(torch.equal(value, campaign.upper.state_dict()[name]) for name, value in initial.items())
    decisions = [json.loads(line) for line in (campaign.output / "joint_decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == (2 if rollback else 2 // every)
    assert all(row["working_update_retained"] == (not rollback) for row in decisions)


def test_forward_checkpoint_selection_uses_locked_development_results(tmp_path, monkeypatch):
    from single_factor.hierarchical.training import Campaign
    from single_factor.hierarchical import experiments
    from single_factor.hierarchical.search import Archive
    config = tiny_training_config(tmp_path)
    config["upper_validation"] = {"task": "development", "every": 1, "episodes": 1,
                                  "seconds": 30, "max_attempts": 1, "forward": False}
    task = {**copy.deepcopy(config["tasks"][0]), "name": "development", "role": "development"}
    config["tasks"].append(task)
    campaign = Campaign(config, tmp_path / "forward")
    campaign.panels["development"]["future"] = campaign.panels["development"]["search"]
    config["upper_validation"]["forward"] = True
    calls = []

    def locked_result(report, evaluator, final):
        calls.append(report)
        assert all(tree["symbol"] for tree in report["pool_asts"])
        return {"pool_ic": .01 if len(calls) == 1 else .04,
                "single_factors": [{"reliable": True, "coverage": 1., "ic": .03, "rank_ic": .02}]}

    monkeypatch.setattr(experiments, "future_check", locked_result)
    score = campaign.validate_upper("upper", 0)
    assert score == pytest.approx(1 + config["search"]["combination_weight"] * .03)
    saved = json.loads((campaign.output / "upper_validation.jsonl").read_text())
    assert saved["selection_metric"] == "forward_count_and_pool_gain"
    assert saved["episodes"][0]["forward_valid"] == 1


def test_runtime_bridge_executes_all_training_layers_exports_and_resumes(tmp_path):
    from single_factor.hierarchical.bridge import run_from_runtime
    config = tiny_training_config(tmp_path)
    fit, _ = slices()
    # The API fixture uses real files and actual neural training, never mocks.
    final = copy.deepcopy(fit)
    final.panel.dates = final.panel.dates + np.timedelta64(250, "D")
    test_cache = write_panel_cache(final, tmp_path / "future")
    template = tmp_path / "adapter_template.json"
    template.write_text(json.dumps(config))
    runtime = {"target_horizon": 2, "max_backtrack_days": 5, "max_future_days": 2,
               "fields": list(fit.panel._arrays),
               "splits": {"train": {"cache_dir": str(tmp_path / "fit")},
                          "valid": {"cache_dir": str(tmp_path / "search")},
                          "test": {"cache_dir": test_cache}}}
    output = tmp_path / "adapter"
    result = run_from_runtime(runtime, output, {"config": str(template)}, 0, "cpu", 10, 1000)
    assert result.timesteps > 0
    for filename in ("lower_base.pt", "upper_frozen_final.pt", "joint_final.pt", "experiment_summary.json",
                     "accepted_factors.jsonl", "rejected_candidates.jsonl"):
        assert (output / filename).exists()
    progress = json.loads((output / "progress.json").read_text())
    assert progress["phase"] == "complete"
    report = json.loads(next((output / "trials").glob("*.json")).read_text())
    assert report["future"]["weights_refitted"] is False
    assert report["future"]["directions_refitted"] is False
    resumed = run_from_runtime(runtime, output, {"config": str(template)}, 0, "cpu", 10, 1000)
    assert resumed.accepted_count == result.accepted_count


def test_suite_really_trains_independent_networks_and_resumes_without_duplicate_trials(tmp_path):
    from single_factor.hierarchical.__main__ import run_suite
    config = tiny_training_config(tmp_path)
    result = run_suite(config, tmp_path / "suite", [1, 2])
    assert result["independent_training_runs"] == 2
    assert result["interval_unit"] == "training_seed"
    assert all(item["metrics"]["new_factors"]["n"] == 2 for item in result["aggregate"])
    first, second = [torch.load(tmp_path / "suite" / f"training_seed_{seed}" / "lower_base.pt", weights_only=False)
                     for seed in (1, 2)]
    assert any(not torch.equal(value, second["lower"][name]) for name, value in first["lower"].items())
    repeated = run_suite(config, tmp_path / "suite", [1, 2], resume=True)
    assert repeated["aggregate"] == result["aggregate"]


def test_parallel_training_cache_validation_and_detailed_lower_logs(tmp_path):
    from single_factor.hierarchical.__main__ import run
    config = tiny_training_config(tmp_path)
    config["rollout_workers"] = 2
    config["record_lower_trajectories"] = True
    config["upper"]["gae_lambda"] = .95
    config["upper"]["joint_guard"] = True
    config["evaluation"]["methods"].append("upper_mc")
    config["signal_cache_dir"] = str(tmp_path / "signals")
    config["lower"]["diffusion_steps"] = 1
    config["search"]["diffusion_steps"] = 1
    config["search"].update(budget_mode="attempts", max_attempts=3, seconds=10.)
    config["tasks"].append({**copy.deepcopy(config["tasks"][0]), "name": "validation", "role": "development"})
    config["upper_validation"] = {"task": "validation", "every": 1, "seconds": 10,
                                  "episodes": 1, "max_attempts": 3}
    output = tmp_path / "parallel"
    summary = run(config, output)
    assert summary["status"] == "complete"
    assert (output / "best_upper.pt").exists() and (output / "best_joint.pt").exists()
    logs = [json.loads(s) for s in (output / "lower_trajectories.jsonl").read_text().splitlines()]
    assert logs and all(len(r["actions"]) == 1 for r in logs)
    assert all("old_logp" in r["actions"][0] and "continuation_endpoints" in r["actions"][0] for r in logs)
    assert list((tmp_path / "signals").glob("*/*.fit.npy"))
    validations = [json.loads(s) for s in (output / "upper_validation.jsonl").read_text().splitlines()]
    assert all(v["budget_type"] == "fixed_attempts" for v in validations)
    assert all(r["attempts"] == 3 and r["late_results_excluded"] == 0
               for v in validations for r in v["episodes"])
    state = json.loads((output / "progress.json").read_text())
    selected = next(v for v in validations if v["stage"] == "upper" and v["iteration"] == state["selected_upper_iteration"])
    joint_baseline = next(v for v in validations if v["stage"] == "joint" and v["iteration"] == 0)
    assert selected["mean_return"] == joint_baseline["mean_return"]
    assert selected["episodes"][0]["discoveries"] == joint_baseline["episodes"][0]["discoveries"]
    recorded = [json.loads(s) for s in (output / "upper_trajectories.jsonl").read_text().splitlines()]
    assert all(0 <= a["remaining_fraction"] <= 1 for r in recorded for a in r["actions"])
    mc = json.loads((output / "upper_mc" / "resolved_config.json").read_text())
    assert "gae_lambda" not in mc["upper"]
    assert (output / "upper_mc" / "upper_frozen_final.pt").exists()
    decisions = [json.loads(s) for s in (output / "joint_decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == 1
    assert state["best_joint_score"] >= joint_baseline["mean_return"]
    assert (output / "joint_candidates" / "iteration_1.pt").exists()
    trial = json.loads(next((output / "trials").glob("*_upper_ppo.json")).read_text())
    assert trial["budget_mode"] == "wall_seconds"


def test_batched_upper_is_exact_and_correlation_pruning_keeps_near_duplicates():
    g = grammar()
    model = UpperPolicy(g, 8)
    o = Observation(("c", "v"), (Node("close"), Node("volume")),
                    ((.1, 0, 0, 0, 0, 1, 1), (.2, 1, .1, .1, .1, 0, 0)), (Node("close"),), (.1,), .02)
    rows = [model.sample(o) for _ in range(3)]
    for a, b in zip([model.row(r) for r in rows], model.rows(rows)):
        torch.testing.assert_close(torch.stack(a), torch.stack(b), atol=1e-6, rtol=1e-6)
    e = evaluator()
    settings = search_settings()
    settings["correlation_threshold"] = .5
    a = Archive(e, [Node("close")], settings)
    candidate = e.grammar.parse("mul(close,2.0)")
    # Exact/near signal handling still evaluates the full mean when its bound permits it.
    result = a.assess(candidate)
    assert not result["new"] or result.get("novelty", {}).get("pair_gain", 0) >= settings["pair_gain_threshold"]
