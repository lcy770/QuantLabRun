"""Native token edits, full PPO replay, and normal collection entry points."""
import json
import random
from copy import deepcopy

import pytest
import torch

import single_factor.hierarchical.ast as ast_module
import single_factor.hierarchical.models as models_module
from single_factor.hierarchical.ast import Grammar, Node
from single_factor.hierarchical.models import LowerPolicy, LowerTask, UpperPolicy, RPNEncoder
from single_factor.hierarchical.ppo import continuation_returns, update
from single_factor.hierarchical.rpn import RPNKernel, RPNProgram
from single_factor.hierarchical.training import Campaign, model_bundle, restore_models
from single_factor.tests.test_hierarchical_search import tiny_training_config


def grammar(strict=True):
    return Grammar(["close", "volume", "amount"],
                   operators=["add", "sub", "mul", "div", "ts_mean", "ts_std", "ts_corr", "delay"],
                   windows=[1, 2, 5], constants=[-1., 0., 1., 2.], strict_types=strict,
                   max_nodes=15, max_depth=8, max_lookback=10)


@pytest.mark.parametrize("strict", [False, True])
def test_native_kernel_exactly_matches_tree_actions_and_corruption(strict):
    g = grammar(strict)
    kernel = RPNKernel(g)
    rng = random.Random(1713)
    formulas = [g.parse(text) for text in ["close", "sub(close,delay(close,1))", "ts_corr(close,volume,2)"]]
    formulas.extend(g.sample(rng) for _ in range(24))
    for tree in formulas:
        program = kernel.from_node(tree)
        assert kernel.to_node(kernel.validate(program)) == tree
        actual = kernel.edits(program)
        assert tuple(kernel.legacy_edit(action) for action in actual) == g.edits(tree)
        assert kernel.edit_paths(program) == g.edit_paths(tree)
        for path in kernel.edit_paths(program):
            assert tuple(kernel.legacy_edit(a) for a in kernel.edits_at(program, path)) == g.edits_at(tree, path)
        for seed in [101, 202]:
            noisy, trace = kernel.corrupt(program, 2, random.Random(seed))
            expected, old_trace = g.corrupt(tree, 2, random.Random(seed))
            assert kernel.to_node(noisy) == expected
            assert [(kernel.to_node(p), step, kernel.legacy_edit(a)) for p, step, a in trace] == old_trace
    # Exercise the uncached position mask and selected-position action path.
    for tree in formulas[:6]:
        program = kernel.from_node(tree)
        kernel._cache.clear()
        assert kernel.edit_paths(program) == g.edit_paths(tree)
        for path in kernel.edit_paths(program):
            assert tuple(kernel.legacy_edit(a) for a in kernel.edits_at(program, path)) == g.edits_at(tree, path)


def test_native_diffusion_works_with_all_ast_editing_entry_points_disabled(monkeypatch):
    g = grammar()
    source = g.parse("sub(close,delay(close,1))")
    model = LowerPolicy(g, 8, 2, representation="rpn", edit_backend="rpn")
    def forbidden(*args, **kwargs):
        raise AssertionError("AST editing was called by native RPN diffusion")
    for name in ["edits", "edits_at", "edit_paths", "corrupt", "_iter_edits", "_edit_candidates", "validate"]:
        monkeypatch.setattr(Grammar, name, forbidden)
    monkeypatch.setattr(ast_module, "replace", forbidden)
    monkeypatch.setattr(models_module, "replace", forbidden)
    kernel = model.edit_kernel
    program = kernel.from_node(source)
    monkeypatch.setattr(kernel, "legacy_edit", forbidden)
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "to_node", forbidden)
        noisy, _ = kernel.corrupt(program, 2, random.Random(101))
        assert kernel.validate(noisy)
    for recording in [False, True]:
        torch.manual_seed(101)
        final, rows = model.sample(noisy, record=recording)
        assert isinstance(final, Node)
        assert len(rows) == (2 if recording else 0)
        if recording:
            assert all("token_edit" in row for row in rows)
            assert max(abs(float(actual[0]) - row["logp"]) for actual, row in zip(model.rows(rows), rows)) < 2e-5


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("position", [False, True])
def test_native_sampling_replay_checkpoint_and_update_match_rpn_tree_backend(conditioned, position):
    torch.manual_seed(1717)
    g = grammar()
    reference = LowerPolicy(g, 8, 2, representation="rpn", use_task=conditioned, use_position=position)
    native = LowerPolicy(g, 8, 2, representation="rpn", use_task=conditioned,
                         use_position=position, edit_backend="rpn")
    if conditioned:
        with torch.no_grad():
            reference.actor_task[-1].weight.normal_(std=.1)
            reference.critic_task[-1].weight.normal_(std=.1)
    if position:
        with torch.no_grad():
            reference.actor_position.weight.normal_(std=.1)
    native.load_state_dict(reference.state_dict())
    source = g.parse("sub(close,delay(close,1))")
    task = LowerTask(source, (Node("volume"),), (.2,), .03) if conditioned else None
    trajectories, rewards = [], []
    for seed in [101, 202, 303]:
        outcomes = []
        for model in [reference, native]:
            torch.manual_seed(seed)
            noisy, _ = model.corrupt(source, 2, random.Random(seed))
            final, rows = model.sample(noisy, task=task)
            outcomes.append((final, rows, torch.get_rng_state()))
        assert outcomes[0][0] == outcomes[1][0]
        assert torch.equal(outcomes[0][2], outcomes[1][2])
        for left, right in zip(outcomes[0][1], outcomes[1][1]):
            assert left["action"] == native.edit_kernel.legacy_edit(right["action"])
            assert isinstance(right["tree"], RPNProgram) and isinstance(right["after"], RPNProgram)
            assert left["logp"] == pytest.approx(right["logp"], abs=2e-5)
            assert left["value"] == pytest.approx(right["value"], abs=2e-5)
        trajectories.append(outcomes[1][1])
        rewards.append(outcomes[1][0].node_count / 10)
    continuation_returns(native, trajectories, rewards, lambda tree: tree.node_count / 10,
                         samples=2, states=2, rng=random.Random(1717))
    rows = [row for trajectory in trajectories for row in trajectory]
    replay = native.rows(rows)
    assert max(abs(float(value[0]) - row["logp"]) for value, row in zip(replay, rows)) < 2e-5
    bundle = model_bundle(native, UpperPolicy(g, 8), {"fingerprint": "native-rpn"})
    restored, _ = restore_models(bundle)
    assert bundle["format_version"] == 8 and restored.edit_backend == "rpn"
    for left, right in zip(replay, restored.rows(rows)):
        for a, b in zip(left, right):
            assert torch.allclose(a, b, atol=2e-6)
    before = native.actor_encoder.embedding.weight.detach().clone()
    result = update(native, torch.optim.Adam(native.parameters(), lr=.001), rows, epochs=1, batch_size=3)
    assert result["old_logp_max_error"] < 2e-5
    assert not torch.equal(before, native.actor_encoder.embedding.weight)


@pytest.mark.parametrize("workers", [1, 2])
def test_native_backend_trains_through_campaign_and_collectors(tmp_path, workers):
    config = tiny_training_config(tmp_path)
    config["lower"].update(representation="rpn", edit_backend="rpn", position_context=True, task_context=True)
    config["rollout_workers"] = workers
    config["record_rollouts"] = True
    campaign = Campaign(config, tmp_path / "native_campaign")
    try:
        campaign.train_lower()
        saved = torch.load(campaign.output / "lower_base.pt", weights_only=False)
        lower, _ = restore_models(saved)
        assert lower.edit_backend == "rpn"
        curves = [json.loads(line) for line in (campaign.output / "lower_curve.jsonl").read_text().splitlines()]
        assert curves and all(row["old_logp_max_error"] < 2e-5 for row in curves)
        from single_factor.replay_hierarchical import replay
        rollout = next((campaign.output / "rollouts").glob("lower_*.pt"))
        result = replay(rollout, campaign.output / "resolved_config.json", tmp_path / "replay.json", threads=1)
        assert result["repeat_max_parameter_difference"] < 2e-6
    finally:
        if campaign.collector:
            campaign.collector.close()


def test_native_program_validation_and_backend_compatibility():
    g = grammar()
    kernel = RPNKernel(g)
    for tokens in [(), (kernel.vocabulary["sub"],), (kernel.vocabulary["close"],) * 2, (-1,), (True,)]:
        with pytest.raises(ValueError):
            kernel.validate(kernel.program(tokens))
    with pytest.raises(ValueError, match="vocabulary"):
        kernel.from_node(RPNProgram((0,), "other"))
    with pytest.raises(ValueError, match="backend"):
        LowerPolicy(g, edit_backend="rpn")


def test_padded_gru_encoding_matches_packed_outputs_and_gradients():
    torch.manual_seed(1719)
    g = grammar()
    encoder = RPNEncoder(g, 8)
    reference = deepcopy(encoder)
    forest = [g.parse(text) for text in ["close", "sub(close,delay(close,1))", "ts_corr(close,volume,2)"]]
    encoded = encoder.encode(forest)
    keys = list(encoded)
    sequences = [reference.kernel.from_node(node).tokens for node in keys]
    lengths = torch.tensor(list(map(len, sequences)))
    tokens = torch.tensor([sequence + (0,) * (int(lengths.max()) - len(sequence)) for sequence in sequences])
    packed = torch.nn.utils.rnn.pack_padded_sequence(reference.embedding(tokens), lengths, batch_first=True, enforce_sorted=False)
    _, hidden = reference.recurrent(packed)
    expected = torch.tanh(reference.norm(hidden.squeeze(0)))
    actual = torch.stack([encoded[node] for node in keys])
    assert torch.allclose(actual, expected, atol=2e-6)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for left, right in zip(encoder.parameters(), reference.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=4e-6)


def test_native_full_operator_registry_and_duplicate_window_vocabulary():
    g = Grammar(["close", "volume", "amount"], windows=[1, 2, 2, 5], max_nodes=12, max_lookback=10)
    kernel = RPNKernel(g)
    rng = random.Random(1723)
    for _ in range(16):
        tree = g.sample(rng)
        program = kernel.from_node(tree)
        assert kernel.to_node(kernel.validate(program)) == tree
        assert tuple(kernel.legacy_edit(a) for a in kernel.edits(program)) == g.edits(tree)
    model = LowerPolicy(g, 8, 2, representation="rpn", edit_backend="rpn")
    final, rows = model.sample(Node("close"))
    assert g.validate(final)
    assert len(model.rows(rows)) == 2
