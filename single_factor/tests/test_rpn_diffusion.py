import random
import json

import pytest
import torch

from single_factor.hierarchical.ast import Grammar, Node, walk
from single_factor.hierarchical.models import LowerPolicy, LowerTask, RPNEncoder, UpperPolicy
from single_factor.hierarchical.ppo import continuation_returns, update
from single_factor.hierarchical.training import Campaign, create_corpus, model_bundle, restore_models
from single_factor.hierarchical.io import digest
from single_factor.tests.test_hierarchical_search import tiny_training_config


def grammar():
    return Grammar(["close", "volume"], operators=["sub", "div", "ts_mean", "delay"],
                   windows=[1, 5], constants=[1., 2.], max_nodes=9)


def test_rpn_encoder_preserves_postfix_operand_order_and_all_subtree_keys():
    g = grammar()
    first = g.parse("sub(close,delay(close,1))")
    second = g.parse("sub(delay(close,1),close)")
    encoder = RPNEncoder(g, 8)
    actual = encoder.encode([first, second])
    assert set(actual) == {node for tree in (first, second) for _, node in walk(tree)}
    assert not torch.allclose(actual[first], actual[second])
    assert encoder.encode([]) == {}
    assert torch.allclose(encoder.encode([first])[first], actual[first], atol=2e-6)


@pytest.mark.parametrize("conditioned", [False, True])
def test_rpn_diffusion_probabilities_checkpoint_and_ppo_replay(conditioned):
    torch.manual_seed(713)
    g = grammar()
    lower = LowerPolicy(g, 8, 2, use_task=conditioned, representation="rpn")
    source = g.parse("div(close,delay(close,1))")
    task = LowerTask(source, (Node("volume"),), (.2,), .03) if conditioned else None
    trajectories, rewards = [], []
    for _ in range(3):
        noisy, _ = g.corrupt(source, 2, random.Random(719))
        final, rows = lower.sample(noisy, task=task)
        g.validate(final)
        trajectories.append(rows)
        rewards.append(final.node_count / 10)
    continuation_returns(lower, trajectories, rewards, lambda tree: tree.node_count / 10,
                         samples=2, states=2, rng=random.Random(719))
    rows = [row for trajectory in trajectories for row in trajectory]
    replayed = lower.rows(rows)
    assert max(abs(float(actual[0]) - row["logp"]) for actual, row in zip(replayed, rows)) < 2e-5
    bundle = model_bundle(lower, UpperPolicy(g, 8), {"fingerprint": "rpn-test"})
    restored, _ = restore_models(bundle)
    assert restored.representation == "rpn" and bundle["format_version"] == 6
    for expected, actual in zip(replayed, restored.rows(rows)):
        assert torch.allclose(expected[0], actual[0], atol=2e-6)
    before = {key: value.clone() for key, value in lower.state_dict().items() if "recurrent" in key}
    result = update(lower, torch.optim.Adam(lower.parameters(), lr=.001), rows, epochs=1, batch_size=3)
    assert result["old_logp_max_error"] < 2e-5
    assert any(not torch.equal(value, lower.state_dict()[key]) for key, value in before.items())


def test_legacy_checkpoint_uses_ast_representation():
    g = grammar()
    bundle = model_bundle(LowerPolicy(g, 8, 2), UpperPolicy(g, 8), {"fingerprint": "legacy"})
    bundle.pop("lower_representation")
    lower, _ = restore_models(bundle)
    assert lower.representation == "ast"


def test_unknown_lower_representation_is_rejected():
    with pytest.raises(ValueError, match="representation"):
        LowerPolicy(grammar(), representation="unknown")


@pytest.mark.parametrize("workers", [1, 2])
def test_rpn_policy_trains_through_campaign_and_parallel_collectors(tmp_path, workers):
    config = tiny_training_config(tmp_path)
    config["lower"]["representation"] = "rpn"
    config["rollout_workers"] = workers
    campaign = Campaign(config, tmp_path / "rpn_campaign")
    try:
        campaign.train_lower()
        saved = torch.load(campaign.output / "lower_base.pt", weights_only=False)
        lower, _ = restore_models(saved)
        assert lower.representation == "rpn"
        curves = [json.loads(line) for line in (campaign.output / "lower_curve.jsonl").read_text().splitlines()]
        assert curves and all(row["old_logp_max_error"] < 2e-5 for row in curves)
    finally:
        if campaign.collector:
            campaign.collector.close()


@pytest.mark.parametrize("invalid_part", ["source", "inverse_pair"])
def test_reused_corpus_rejects_labels_invalid_under_current_grammar(tmp_path, invalid_part):
    g = grammar()
    invalid = Node("div", (Node("close"), Node("sub", (Node("close"), Node("close")))))
    settings = {"count": 1, "noise_steps": 2, "seed": 0}
    record = {"id": "old-formula", "ast": (invalid if invalid_part == "source" else Node("close")).to_dict(),
              "pairs": [{"tree": invalid.to_dict(), "step": 1, "path": [], "replacement": Node("close").to_dict()}]}
    (tmp_path / "corpus.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "corpus_manifest.json").write_text(json.dumps({"configuration_hash": digest([g.config(), settings])}))
    with pytest.raises(ValueError, match="stored corpus is incompatible"):
        create_corpus(g, settings, tmp_path)
