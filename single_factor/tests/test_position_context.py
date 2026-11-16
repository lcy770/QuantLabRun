import random

import pytest
import torch

from single_factor.hierarchical.ast import Grammar, Node
from single_factor.hierarchical.models import LowerPolicy, LowerTask, UpperPolicy
from single_factor.hierarchical.training import Campaign, model_bundle, restore_models
from single_factor.tests.test_hierarchical_search import tiny_training_config


def grammar():
    return Grammar(["close", "open"], operators=["sub", "div", "delay"], windows=[1, 5], constants=[1., 2.],
                   max_nodes=9, max_depth=4)


@pytest.mark.parametrize("representation", ["ast", "rpn"])
def test_position_context_breaks_the_unlearnable_tie_between_identical_subtrees(representation):
    torch.manual_seed(917)
    g = grammar()
    tree = g.parse("sub(close,delay(close,1))")
    legacy = LowerPolicy(g, 8, 2, representation=representation)
    paths, _, positions, _, _ = legacy.distributions(tree, 2)
    left, right = paths.index((0,)), paths.index((1, 0))
    assert positions.probs[left].item() == positions.probs[right].item()
    model = LowerPolicy(g, 8, 2, representation=representation, use_position=True)
    model.load_state_dict(legacy.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.actor_position.weight.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor_position.parameters(), lr=.1)
    for _ in range(60):
        _, _, distribution, _, _ = model.distributions(tree, 2)
        logits = distribution.logits[[left, right]]
        loss = -torch.log_softmax(logits, dim=0)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, _, learned, _, _ = model.distributions(tree, 2)
    conditional = learned.probs[left] / (learned.probs[left] + learned.probs[right])
    assert float(conditional) > .65


@pytest.mark.parametrize("representation", ["ast", "rpn"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_position_migration_sampling_replay_and_checkpoint_metadata(representation, conditioned):
    torch.manual_seed(918)
    g = grammar()
    tree = g.parse("sub(close,delay(close,1))")
    legacy = LowerPolicy(g, 8, 2, representation=representation, use_task=conditioned)
    model = LowerPolicy(g, 8, 2, representation=representation, use_task=conditioned, use_position=True)
    model.load_state_dict(legacy.state_dict())
    task = LowerTask(tree, (Node("open"),), (.2,), .03) if conditioned else None
    first, second = legacy.distributions(tree, 2, task), model.distributions(tree, 2, task)
    assert torch.equal(first[2].probs, second[2].probs)
    assert all(torch.equal(a.probs, b.probs) for a, b in zip(first[3], second[3]))
    with torch.no_grad():
        model.actor_position.weight.normal_(std=.2)
    outcomes = []
    for record in (True, False):
        torch.manual_seed(719)
        outcomes.append(model.sample(tree, remaining=2, record=record, task=task))
    assert outcomes[0][0] == outcomes[1][0]
    replay = model.rows(outcomes[0][1])
    assert max(abs(float(row[0]) - actual["logp"]) for row, actual in zip(replay, outcomes[0][1])) < 2e-5
    bundle = model_bundle(model, UpperPolicy(g, 8), {"fingerprint": "positions"})
    restored, _ = restore_models(bundle)
    assert restored.use_position and bundle["format_version"] == 7
    assert torch.allclose(restored.rows(outcomes[0][1])[0][0], replay[0][0], atol=2e-6)
    del bundle["lower"]["actor_position.weight"]
    with pytest.raises(ValueError, match="position parameters"):
        restore_models(bundle)


def test_campaign_rejects_an_initializer_from_another_representation(tmp_path):
    config = tiny_training_config(tmp_path)
    g = Grammar(**config["grammar"])
    initializer = tmp_path / "ast.pt"
    torch.save(model_bundle(LowerPolicy(g, config["width"], config["corpus"]["noise_steps"]),
                           UpperPolicy(g, config["width"]), config), initializer)
    config["initializer_checkpoint"] = str(initializer)
    config["lower"]["representation"] = "rpn"
    with pytest.raises(ValueError, match="initializer representation differs"):
        Campaign(config, tmp_path / "incompatible")


@pytest.mark.parametrize("workers", [1, 2])
def test_position_policy_trains_through_normal_campaign_and_collectors(tmp_path, workers):
    config = tiny_training_config(tmp_path)
    config["lower"]["position_context"] = True
    config["rollout_workers"] = workers
    campaign = Campaign(config, tmp_path / "positions")
    try:
        campaign.train_lower()
        saved = torch.load(campaign.output / "lower_base.pt", weights_only=False)
        restored, _ = restore_models(saved)
        assert restored.use_position
    finally:
        if campaign.collector:
            campaign.collector.close()
