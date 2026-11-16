import copy
import json
import random

import pytest
import torch

from single_factor.hierarchical.ast import Node
from single_factor.hierarchical.evaluation import Evaluator
from single_factor.hierarchical.models import LowerPolicy, LowerTask, UpperPolicy
from single_factor.hierarchical.ppo import continuation_returns
from single_factor.hierarchical.search import episode
from single_factor.hierarchical.training import Campaign, model_bundle, restore_models
from single_factor.tests.test_hierarchical_search import grammar, slices, search_settings, tiny_training_config


def task(source="close"):
    return LowerTask(Node(source), (Node("close"), Node("volume")), (.2, -.4), .03)


def test_task_migration_preserves_legacy_sampling_and_checkpoint_roundtrip():
    g = grammar()
    legacy, current = LowerPolicy(g, 8, 2), LowerPolicy(g, 8, 2, use_task=True)
    current.load_state_dict(legacy.state_dict())
    outcomes = []
    for model, options in ((legacy, {}), (current, {"task": task()})):
        torch.manual_seed(741)
        final, rows = model.sample(Node("volume"), **options)
        outcomes.append((final, rows, torch.get_rng_state()))
    assert outcomes[0][0] == outcomes[1][0]
    assert torch.equal(outcomes[0][2], outcomes[1][2])
    for first, second in zip(outcomes[0][1], outcomes[1][1]):
        assert first["action"] == second["action"]
        assert first["logp"] == pytest.approx(second["logp"], abs=2e-6)
        assert first["value"] == pytest.approx(second["value"], abs=2e-6)
        assert second["task"] == task()
    bundle = model_bundle(current, UpperPolicy(g, 8), {"fingerprint": "test"})
    restored, _ = restore_models(bundle)
    assert bundle["format_version"] == 5 and restored.use_task
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in current.state_dict().items())


def test_conditioned_batch_matches_individual_probabilities_values_entropy_and_gradients():
    g = grammar()
    model = LowerPolicy(g, 8, 2, use_task=True)
    with torch.no_grad():
        model.actor_task[-1].weight.normal_(std=.1)
        model.critic_task[-1].weight.normal_(std=.1)
    rows = []
    for source in ("close", "volume", "amount"):
        rows.extend(model.sample(Node("volume"), task=task(source))[1])
    batched = copy.deepcopy(model)
    first, second = [model.row(row) for row in rows], batched.rows(rows)
    for a, b, row in zip(first, second, rows):
        torch.testing.assert_close(torch.stack(a), torch.stack(b), atol=2e-6, rtol=2e-6)
        assert float(a[0].detach()) == pytest.approx(row["logp"], abs=2e-6)
    sum(v[0] + .1 * v[1] + .01 * v[2] for v in first).backward()
    sum(v[0] + .1 * v[1] + .01 * v[2] for v in second).backward()
    for (_, a), (_, b) in zip(model.named_parameters(), batched.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=2e-5)
    assert model.actor_task[-1].weight.grad.abs().sum() > 0
    assert model.critic_task[-1].weight.grad.abs().sum() > 0
    first_value = model.distributions(Node("volume"), 2, task("close"))[-1]
    second_value = model.distributions(Node("volume"), 2, task("amount"))[-1]
    assert not torch.allclose(first_value, second_value, atol=1e-7)
    first_edits = model.distributions(Node("volume"), 2, task("close"))[3][0].probs
    second_edits = model.distributions(Node("volume"), 2, task("amount"))[3][0].probs
    assert not torch.allclose(first_edits, second_edits, atol=1e-7)


@pytest.mark.parametrize("missing", ["one", "all"])
def test_task_checkpoint_rejects_missing_context_parameters(missing):
    g = grammar()
    lower = LowerPolicy(g, 8, 2, use_task=True)
    bundle = model_bundle(lower, UpperPolicy(g, 8), {"fingerprint": "test"})
    keys = [key for key in bundle["lower"] if key.startswith(("actor_task.", "critic_task."))]
    for key in keys[:1] if missing == "one" else keys:
        del bundle["lower"][key]
    with pytest.raises(ValueError, match="context parameters"):
        restore_models(bundle)
    if missing == "one":
        with pytest.raises(RuntimeError, match="Missing key"):
            lower.load_state_dict(bundle["lower"])


def test_continuations_preserve_the_persistent_task(monkeypatch):
    model = LowerPolicy(grammar(), 8, 2, use_task=True)
    _, rows = model.sample(Node("volume"), task=task())
    sample = model.sample
    received = []
    def tracked(tree, **kwargs):
        received.append(kwargs["task"])
        return sample(tree, **kwargs)
    monkeypatch.setattr(model, "sample", tracked)
    branches = continuation_returns(model, [rows], [.2], lambda tree: .3, samples=4, selected={(0, 0)})
    assert branches == 3 and received == [task()] * 3
    assert rows[0]["return"] == pytest.approx(.275)


def test_worker_observes_new_discoveries_outside_the_current_pool():
    from dataclasses import replace
    model = LowerPolicy(grammar(), 8, 2, use_task=True)
    with torch.no_grad():
        model.critic_task[-1].weight.normal_(std=.1)
    base = task()
    extended = replace(base, discoveries=(Node("amount"), Node("volume")))
    reversed_task = replace(extended, discoveries=tuple(reversed(extended.discoveries)))
    first = model.distributions(Node("volume"), 2, base)[-1]
    second = model.distributions(Node("volume"), 2, extended)[-1]
    third = model.distributions(Node("volume"), 2, reversed_task)[-1]
    assert not torch.allclose(first, second, atol=1e-7)
    torch.testing.assert_close(second, third, atol=2e-6, rtol=2e-6)


def test_completion_branches_preserve_primary_rollout_archive_and_rng():
    g = grammar()
    lower, upper = LowerPolicy(g, 8, 2), UpperPolicy(g, 8)
    config = {**search_settings(), "seconds": 30., "max_attempts": 8, "budget_mode": "attempts"}
    initial = [Node("close"), Node("volume")]
    results = []
    for training in (None, {"completion_reward": True, "continuation_states": 3, "continuation_samples": 4}):
        torch.manual_seed(712)
        result = episode(lower, upper, Evaluator(g, *slices()), initial, config, random.Random(41), lower_training=training)
        results.append((*result, torch.get_rng_state()))
    a, b = results
    assert torch.equal(a[-1], b[-1])
    for x, y in zip(a[0], b[0]):
        assert x == y
    assert [[r["after"] for r in t] for t in a[1]] == [[r["after"] for r in t] for t in b[1]]
    for key in ("discoveries", "pool_asts", "pool_weights", "return", "new_factors", "duplicate_count"):
        assert a[3][key] == b[3][key]
    assert b[3]["lower_continuation_branches"] == 9
    assert b[2] == [row["reward"] for row in b[0]]
    assert sum(len(row["continuation_scores"]) - 1 for t in b[1] for row in t) == 9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_auxiliary_completions_restore_cuda_random_state(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    g = grammar()
    lower, upper = LowerPolicy(g, 8, 2).cuda(), UpperPolicy(g, 8).cuda()
    settings = {**search_settings(), "seconds": 30., "max_attempts": 3, "budget_mode": "attempts"}
    outcomes = []
    for training in (None, {"completion_reward": True, "continuation_states": 1, "continuation_samples": 4}):
        torch.manual_seed(718)
        rows, _, _, report = episode(lower, upper, Evaluator(g, *slices()), [Node("close"), Node("volume")],
                                    settings, random.Random(918), lower_training=training)
        outcomes.append((rows, report["discoveries"], torch.cuda.get_rng_state()))
    assert outcomes[0][:2] == outcomes[1][:2]
    assert torch.equal(outcomes[0][2], outcomes[1][2])


def test_episode_task_locks_source_directed_weights_and_pre_action_discoveries():
    g = grammar()
    lower, upper = LowerPolicy(g, 8, 2, use_task=True), UpperPolicy(g, 8)
    e = Evaluator(g, *slices())
    settings = {**search_settings(), "seconds": 30., "max_attempts": 4, "budget_mode": "attempts"}
    upper_rows, trajectories, _, report = episode(lower, upper, e, [Node("close"), Node("volume")],
                                                  settings, random.Random(79))
    discoveries = tuple(Node.from_dict(d["ast"]) for d in report["discoveries"])
    count = 0
    assert len(upper_rows) == len(trajectories) == 4
    for row, trajectory in zip(upper_rows, trajectories):
        condition, observation = trajectory[0]["task"], row["observation"]
        if row["index"] < len(observation.trees):
            assert condition.source == observation.trees[row["index"]]
        assert all(edit["task"] == condition for edit in trajectory)
        assert condition.pool_weights == tuple(w * e.metrics[t.id]["direction"]
                                               for t, w in zip(observation.pool_trees, observation.pool_weights))
        assert condition.pool_ic == observation.pool_ic
        assert condition.discoveries == discoveries[:count]
        count += int(row["reward"] >= 1.)


def test_joint_campaign_updates_task_worker_then_collects_new_upper_rollouts(tmp_path):
    config = tiny_training_config(tmp_path)
    config["lower"].update(task_context=True, search_training=True, completion_reward=True)
    config["upper"].update(joint_guard=True, joint_iterations=1)
    config["search"].update(max_attempts=4, budget_mode="attempts", seconds=30.)
    config["record_rollouts"] = True
    campaign = Campaign(config, tmp_path / "joint_task")
    campaign.train_upper(joint=True)
    lower = torch.load(campaign.output / "rollouts/lower_0_0.pt", weights_only=False)
    upper = torch.load(campaign.output / "rollouts/joint_0.pt", weights_only=False)
    assert all("task" in row for row in lower["rows"])
    assert lower["task"] == "training"
    assert any(not torch.equal(v, upper["models"]["lower"][k]) for k, v in lower["models"]["lower"].items())
    reports = [json.loads(line) for line in (campaign.output / "joint_lower_episodes.jsonl").read_text().splitlines()]
    assert reports[0]["attempts"] == 4 and reports[0]["lower_continuation_branches"] == 3
