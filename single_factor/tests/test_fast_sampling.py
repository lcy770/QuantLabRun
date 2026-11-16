import random

import pytest
import torch

from single_factor.hierarchical.ast import Grammar, Node, replace
from single_factor.hierarchical.models import LowerPolicy, LowerTask
import single_factor.hierarchical.models as models


def kernel(strict=True):
    return Grammar(["close", "volume", "amount"], operators=["add", "sub", "mul", "div", "ts_mean", "ts_std", "ts_corr", "delay"],
                   windows=[1, 2, 5], constants=[-1., 0., 1., 2.], strict_types=strict,
                   max_nodes=15, max_depth=8, max_lookback=10)


@pytest.mark.parametrize("strict", [False, True])
def test_lazy_positions_and_replacements_equal_the_full_legal_kernel(strict):
    grammar = kernel(strict)
    rng = random.Random(713)
    for _ in range(32):
        tree = grammar.sample(rng)
        expected = grammar.edits(tree)
        grammar._cache.clear()
        paths = grammar.edit_paths(tree)
        assert paths == tuple(sorted({action.path for action in expected}))
        actual = []
        for path in paths:
            grammar._cache.clear()
            choices = grammar.edits_at(tree, path)
            assert choices == tuple(action for action in expected if action.path == path)
            actual.extend(choices)
        assert tuple(actual) == expected


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("remaining", [0, 1, 2, 4])
def test_recorded_and_lazy_sampling_have_identical_actions_and_random_states(monkeypatch, conditioned, remaining):
    grammar = kernel()
    lower = LowerPolicy(grammar, 8, 4, use_task=conditioned)
    source = grammar.parse("div(ts_mean(close,2),delay(close,1))")
    options = {}
    if conditioned:
        with torch.no_grad():
            lower.actor_task[-1].weight.normal_(std=.1)
            lower.critic_task[-1].weight.normal_(std=.1)
        options["task"] = LowerTask(source, (Node("close"), Node("volume")), (.3, -.2), .03,
                                    (grammar.parse("ts_mean(volume,2)"),))
    chosen = []
    def track(tree, path, replacement):
        chosen.append((path, replacement))
        return replace(tree, path, replacement)
    monkeypatch.setattr(models, "replace", track)
    outcomes = []
    for recording in (True, False):
        grammar._cache.clear()
        torch.manual_seed(717)
        rng = random.Random(717)
        noisy, trace = grammar.corrupt(source, 2, rng)
        chosen.clear()
        final, rows = lower.sample(noisy, remaining=remaining, record=recording, **options)
        outcomes.append((final, trace, tuple(chosen), torch.get_rng_state(), rng.getstate()))
        assert len(rows) == (remaining if recording else 0)
    a, b = outcomes
    assert a[:3] == b[:3]
    assert torch.equal(a[3], b[3])
    assert a[4] == b[4]


def test_position_mask_excludes_nodes_with_no_legal_alternative():
    grammar = Grammar(["close"], operators=["delay"], windows=[1], constants=[1.],
                      max_nodes=3, max_depth=2, max_lookback=1)
    tree = grammar.parse("delay(close,1)")
    assert grammar.edit_paths(tree) == ((),)
    assert grammar.edits_at(tree, (0,)) == ()
    assert grammar.edits_at(tree, (1,)) == ()
    assert grammar.edits_at(tree, ()) == grammar.edits(tree)
