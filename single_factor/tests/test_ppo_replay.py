"""概率重放的批次数值漂移、真实错配拒绝及宽网络GPU回归。"""
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from single_factor.hierarchical.ast import Grammar
from single_factor.hierarchical.models import LowerPolicy, LowerTask, Observation
from single_factor.hierarchical.ppo import complete_returns, update
from single_factor.hierarchical.selection_models import GeneratorPolicy, SelectionObservation, SelectorPolicy


class LayoutPolicy(nn.Module):
    """构造可控批次漂移，单次采样分布与真实参数始终一致。"""

    def __init__(self, drift=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.3))
        self.drift = drift

    def row(self, row):
        distribution = Categorical(logits=torch.stack((self.bias, -self.bias)))
        return distribution.logits[row["index"]], self.bias, distribution.entropy()

    def rows(self, rows):
        return [(value + self.drift, critic, entropy) for value, critic, entropy in map(self.row, rows)]

    def replay_logp(self, row):
        return self.row(row)[0]


def rollout(model):
    """保留来自采样分布的旧概率，构造非零优势。"""
    rows = [{"index": index, "return": float(index), "value": 0.3} for index in (0, 1)]
    for row in rows:
        row["logp"] = float(model.row(row)[0].detach())
    return rows


@pytest.mark.parametrize("drift", [0.0, 4.9591064453125e-05])
def test_small_batch_drift_requires_exact_replay_and_keeps_old_probabilities(drift):
    """复现用户的误差量级，单条复核成功后允许更新但不重写行为策略概率。"""
    model = LayoutPolicy(drift)
    rows = rollout(model)
    old = [row["logp"] for row in rows]
    before = model.bias.detach().clone()
    result = update(model, torch.optim.Adam(model.parameters(), lr=.01), rows, epochs=1, batch_size=2)
    assert result["old_logp_max_error"] < 2e-5
    assert result["old_logp_batch_max_error"] == pytest.approx(drift, abs=1e-7)
    assert result["old_logp_scalar_checks"] == (len(rows) if drift else 0)
    assert old == [row["logp"] for row in rows]
    assert not torch.equal(model.bias.detach(), before)


@pytest.mark.parametrize("failure", ["stored", "batch", "nan"])
def test_real_or_large_replay_mismatches_fail_before_optimization(failure):
    """旧概率损坏、批量实现大偏差及非有限数值都必须继续拦截。"""
    model = LayoutPolicy(.01 if failure == "batch" else float("nan") if failure == "nan" else 0.0)
    rows = rollout(model)
    if failure == "stored":
        rows[0]["logp"] += .001
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = model.bias.detach().clone()
    with pytest.raises(AssertionError, match="old-policy"):
        update(model, optimizer, rows, epochs=1, batch_size=2)
    assert torch.equal(model.bias.detach(), before)
    assert not optimizer.state


def test_sampling_layout_fallback_checks_every_old_probability_before_optimization(monkeypatch):
    model = LayoutPolicy()
    rows = rollout(model)
    monkeypatch.setattr(model, "replay_row", model.row, raising=False)
    drift = (0.0002454519271850586, 3e-5)
    batched = model.rows

    def drifted(batch):
        return [(logp + offset, value, entropy)
                for (logp, value, entropy), offset in zip(batched(batch), drift, strict=True)]

    monkeypatch.setattr(model, "rows", drifted)
    # The second row agrees with the batched evaluator but disagrees with sampling.
    rows[1]["logp"] += drift[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    with pytest.raises(AssertionError, match=r"old-policy action probability.*row=1"):
        update(model, optimizer, rows, epochs=1, batch_size=2)
    assert not optimizer.state


@pytest.mark.parametrize("corrupt_tail", [False, True])
def test_large_rollout_replays_in_minibatches_before_any_update(monkeypatch, corrupt_tail):
    """长轨迹的重放显存受小批限制，末批损坏仍在任何参数更新之前拦截。"""
    model = LayoutPolicy()
    rows = [dict(row) for _ in range(20) for row in rollout(model)]
    if corrupt_tail:
        rows[-1]["logp"] += .01
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    replay = model.rows
    check_sizes = []

    def bounded(batch):
        assert len(batch) <= 16
        if not torch.is_grad_enabled():
            check_sizes.append(len(batch))
            assert not optimizer.state
        return replay(batch)

    monkeypatch.setattr(model, "rows", bounded)
    if corrupt_tail:
        with pytest.raises(AssertionError, match="old-policy.*row=39"):
            update(model, optimizer, rows, epochs=1, batch_size=16)
        assert not optimizer.state
    else:
        assert update(model, optimizer, rows, epochs=1, batch_size=16)["optimizer_steps"] == 3
    assert check_sizes == [16, 16, 8]


def tiny_grammar():
    return Grammar(["close", "volume"], operators=["sub", "div", "delay"], windows=[1, 5],
                   constants=[1., 2.], max_nodes=9)


@pytest.mark.parametrize("kind", ["editor", "generator", "selector"])
@pytest.mark.parametrize("drift", [5e-5, 0.0002454519271850586])
def test_scalar_replay_uses_sampling_path_without_consuming_rng(kind, drift, monkeypatch):
    """用真实三种策略核验共享采样路径，复核不消耗额外随机数。"""
    torch.manual_seed(19)
    grammar = tiny_grammar()
    source = grammar.parse("div(close,delay(close,1))")
    if kind == "editor":
        model = LowerPolicy(grammar, 32, 2, use_task=True, representation="rpn", edit_backend="rpn")
        _, rows = model.sample(source, task=LowerTask(source, (), (), 0.0))
    elif kind == "generator":
        model = GeneratorPolicy(grammar, 32, 2)
        observation = Observation(("source",), (source,), ((0.,) * 7,), (), (), 0.0)
        _, rows, _ = model.generate(observation, random.Random(19), 2, 1)
    else:
        model = SelectorPolicy(grammar, 32)
        observation = SelectionObservation((source,), ((0.,) * 6,), ((1.,),), (), 2, "test")
        rows = [model.sample(observation)]
    state = torch.get_rng_state().clone()
    with torch.no_grad():
        assert max(abs(float(model.replay_logp(row)) - row["logp"]) for row in rows) < 2e-5
        assert max(abs(float(model.replay_row(row)[1]) - row["value"]) for row in rows) < 2e-5
    assert torch.equal(state, torch.get_rng_state())
    for index, row in enumerate(rows):
        row["return"] = float(index)
    batched = model.rows

    def drifted(batch):
        """模拟 GPU 合并批次的两种漂移量级。"""
        if drift > 2e-4:
            assert not torch.is_grad_enabled()
        return [(probability + drift, value, entropy) for probability, value, entropy in batched(batch)]

    monkeypatch.setattr(model, "rows", drifted)
    old = [row["logp"] for row in rows]
    metrics = update(model, torch.optim.Adam(model.parameters(), lr=.0003), rows, epochs=1, batch_size=16)
    assert metrics["old_logp_scalar_checks"] == len(rows)
    assert metrics["old_logp_max_error"] < 2e-5
    assert metrics["old_logp_sampling_layout_update"] is (drift > 2e-4)
    assert metrics["optimizer_steps"] > 0
    assert [row["logp"] for row in rows] == old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA设备")
def test_cuda_width256_replay_after_repeated_updates(monkeypatch):
    """在实际GPU上用完整文法、256宽度和多轮更新验证采样及重放。"""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(19)
    path = Path(__file__).parents[2] / "experiments/dual_ppo/base.json"
    grammar = Grammar(**json.loads(path.read_text(encoding="utf-8"))["grammar"])
    model = GeneratorPolicy(grammar, 256, 4, learned_stop=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    source = grammar.parse("div(ts_mean(close,5),ts_std(close,20))")
    observation = Observation(("source",), (source,), ((0.,) * 7,), (), (), 0.0)
    for iteration in range(3):
        rows = []
        for index in range(4):
            _, trajectory, _ = model.generate(observation, random.Random(iteration * 4 + index), 4, 4)
            for row in trajectory:
                row["reward"] = 0.0
            trajectory[-1]["reward"] = (index - 1.5) / 10
            complete_returns(trajectory)
            rows.extend(trajectory)
        metrics = update(model, optimizer, rows, epochs=4, batch_size=16)
        assert metrics["old_logp_max_error"] <= 2e-5
        assert np.isfinite(metrics["actor_loss"])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要CUDA设备"))])
def test_width256_upper_ppo_replay_with_512_candidates(monkeypatch, device):
    """在主实验的候选规模、网络宽度和小批设置下执行上层真实梯度更新。"""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(19)
    path = Path(__file__).parents[2] / "experiments/dual_ppo/base.json"
    grammar = Grammar(**json.loads(path.read_text(encoding="utf-8"))["grammar"])
    rng = random.Random(33)
    trees, seen = [], set()
    for _ in range(512):
        tree = grammar.sample(rng, exclude=seen)
        trees.append(tree)
        seen.add(tree.id)
    selector = SelectorPolicy(grammar, 256).to(device)
    observation = SelectionObservation(tuple(trees), ((0.,) * 6,) * len(trees),
        tuple(map(tuple, np.eye(len(trees)))), (), len(trees), "width256", True)
    rows = []
    for i in range(17):
        selected = tuple(range(i))
        row = selector.sample(replace(observation, selected=selected))
        assert row["index"] not in selected
        row["return"] = i / 17
        rows.append(row)
    metrics = update(selector, torch.optim.Adam(selector.parameters(), lr=.0003), rows, epochs=4, batch_size=16)
    assert metrics["optimizer_steps"] == 8
    assert metrics["old_logp_max_error"] <= 2e-5
    assert np.isfinite(metrics["actor_loss"])
