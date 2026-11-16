"""Regression checks for common two-layer completion targets and RPN scheduling."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from single_factor.hierarchical.models import LowerPolicy, UpperPolicy, RPNEncoder
from single_factor.hierarchical.ast import Node
from single_factor.hierarchical.search import Archive, episode
from single_factor.hierarchical.training import (
    Campaign, completion_reward, completion_training_rewards, model_bundle, restore_models,
)
from single_factor.tests.test_hierarchical_search import (
    evaluator, grammar, search_settings, slices, tiny_training_config, write_panel_cache,
)


def test_joint_objective_rewards_only_intersection_and_bounds_damage():
    settings = {"quality_threshold": .02, "combination_weight": 50.}
    assert completion_reward(True, True, .001, settings, "forward_joint") == 1
    assert completion_reward(True, False, .001, settings, "forward_joint") == 0
    assert completion_reward(True, True, 0., settings, "forward_joint") == 0
    assert completion_reward(True, True, -.01, settings, "forward_joint") == -.5
    assert completion_reward(True, False, -.2, settings, "forward_joint") == -1
    assert completion_reward(False, True, .2, settings, "forward_joint") == 0


def test_rpn_upper_cost_migration_preserves_initial_distribution_and_roundtrip():
    g, e = grammar(), evaluator()
    settings = search_settings()
    archive = Archive(e, [Node("close"), Node("volume")], settings)
    old = UpperPolicy(g, 8, representation="rpn")
    current = UpperPolicy(g, 8, representation="rpn", cost_context=True)
    current.load_state_dict(old.state_dict())
    first = old.distribution(archive.observe())[0].probs
    second = current.distribution(archive.observe(cost_context=True))[0].probs
    torch.testing.assert_close(first, second)
    lower = LowerPolicy(g, 8, 2, representation="rpn", edit_backend="rpn")
    bundle = model_bundle(lower, current, {"fingerprint": "aligned"})
    _, upper = restore_models(bundle)
    assert bundle["format_version"] == 9 and isinstance(upper.actor.encoder, RPNEncoder)
    torch.testing.assert_close(upper.distribution(archive.observe(cost_context=True))[0].probs, second)
    bundle.pop("upper_representation")
    with pytest.raises(ValueError, match="metadata"):
        restore_models(bundle)


def test_postcollection_targets_preserve_visible_state_rng_and_match_layers():
    g, e = grammar(), evaluator()
    settings = {**search_settings(), "quality_threshold": .000001, "min_coverage": .8,
                "max_attempts": 8, "budget_mode": "attempts"}
    lower = LowerPolicy(g, 8, 2, representation="rpn", edit_backend="rpn")
    upper = UpperPolicy(g, 8, representation="rpn", cost_context=True)
    initial, rng = [Node("close"), Node("volume")], random.Random(41)
    rows, trajectories, _, report = episode(lower, upper, e, initial, settings, rng,
        lower_training={"completion_reward": True, "continuation_states": 3, "continuation_samples": 3})
    observations = [row["observation"] for row in rows]
    before, torch_rng, python_rng = copy.deepcopy(report), torch.get_rng_state(), rng.getstate()
    # An uncredited NEW failure has no corresponding lower trajectory.
    rows.insert(1, {"observation": observations[0], "index": 0, "reward": 0., "credited": False})
    rewards = completion_training_rewards(rows, trajectories, report, e, initial, slices(41)[1], settings, "forward_joint")
    assert torch.equal(torch_rng, torch.get_rng_state()) and rng.getstate() == python_rng
    assert [row["observation"] for row in rows if row.get("credited")] == observations
    for key in ("pool_asts", "pool_weights", "return", "discoveries"):
        assert before[key] == report[key]
    assert sum(row["reward"] for row in rows) == report["training_return"]
    for row in rows:
        if "lower_trajectory_index" in row:
            index = row["lower_trajectory_index"]
            assert rewards[index] == row["reward"]
            assert trajectories[index][-1]["continuation_scores"] == [row["reward"]]
            for action in trajectories[index]:
                assert action["return"] == np.mean(action["continuation_scores"])


@pytest.mark.parametrize("workers", [1, 2])
def test_aligned_campaign_search_lower_then_fresh_upper_and_checkpoint(tmp_path, workers):
    config = tiny_training_config(tmp_path)
    final = copy.deepcopy(slices(41)[1])
    final.panel.dates += np.timedelta64(120, "D")
    config["tasks"][0]["future"] = {"cache": write_panel_cache(final, tmp_path / "teacher")}
    later = copy.deepcopy(slices(51)[1])
    later.panel.dates += np.timedelta64(365, "D")
    config["tasks"].append({"name": "dev", "role": "development", "fit": config["tasks"][0]["fit"],
                            "search": {"cache": write_panel_cache(later, tmp_path / "dev")}})
    config["lower"].update(representation="rpn", edit_backend="rpn", search_training=True, completion_reward=True)
    config["upper"].update(representation="rpn", cost_context=True, training_reward="forward_joint", joint_guard=False)
    config["search"].update(max_attempts=3, budget_mode="attempts", quality_threshold=.000001)
    config.update(rollout_workers=workers, record_rollouts=True)
    campaign = Campaign(config, tmp_path / "run")
    try:
        campaign.train_lower()
        reports = [json.loads(line) for line in (campaign.output / "lower_training_episodes.jsonl").read_text().splitlines()]
        assert reports[0]["attempts"] == 2 and reports[0]["training_reward"] == "forward_joint"
        campaign.train_upper(joint=True)
        joint_lower = torch.load(campaign.output / "rollouts/lower_1_0.pt", weights_only=False)
        joint_upper = torch.load(campaign.output / "rollouts/joint_0.pt", weights_only=False)
        assert any(not torch.equal(v, joint_upper["models"]["lower"][k])
                   for k, v in joint_lower["models"]["lower"].items())
        lower, upper = restore_models(torch.load(campaign.output / "joint_final.pt", weights_only=False))
        assert lower.edit_backend == upper.representation == "rpn" and upper.cost_context
    finally:
        if campaign.collector:
            campaign.collector.close()
