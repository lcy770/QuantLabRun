"""新分工的公式、因果边界、实际梯度与正常入口端到端回归。"""
import copy
from contextlib import nullcontext
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from single_factor.data import PanelData
from single_factor.hierarchical.ast import Grammar
from single_factor.hierarchical.evaluation import Slice, normalize
from single_factor.hierarchical.selection_evaluation import (
    SelectionEvaluator, block_deviation, daily_correlation, load_selection_task,
    member_deficiency, quality_score,
)
from single_factor.hierarchical.selection_models import GeneratorPolicy, SelectorPolicy
from single_factor.hierarchical.selection_search import (
    CandidatePool, ProposalCollector, StartBank, collect_proposal, select_members,
)
from single_factor.hierarchical.selection_training import SelectionCampaign, restore_selection_models


torch.set_num_threads(1)


def grammar():
    return Grammar(["close", "volume", "amount"], operators=["abs", "add", "sub"],
                   windows=[1], constants=[-1.0, 1.0], max_nodes=5, max_depth=3,
                   max_lookback=2, strict_types=False)


def panel(seed, start, count=28):
    rng = np.random.default_rng(seed)
    shape = (count + 4, 12)
    first = rng.normal(size=shape)
    second = rng.normal(size=shape)
    arrays = {"close": 20 + rng.normal(size=shape), "volume": first, "amount": second}
    data = PanelData(arrays, pd.date_range(start, periods=shape[0], freq="B"), list(map(str, range(12))), 2, 2)
    target = 0.03 * first[2:2 + count] + rng.normal(0, 0.01, (count, 12))
    return Slice(data, 2, target)


def write_cache(scoring, path):
    path.mkdir(parents=True)
    for field, values in scoring.panel._arrays.items():
        np.save(path / f"{field}.npy", values)
    np.save(path / "dates.npy", scoring.panel.dates.values)
    np.save(path / "labels.npy", np.pad(scoring.target, ((2, 2), (0, 0)), constant_values=np.nan))
    (path / "codes.json").write_text(json.dumps(scoring.panel.stock_ids))
    (path / "manifest.json").write_text(json.dumps({"fields": list(scoring.panel._arrays),
        "label_file": "labels.npy", "label_horizon": 2}))
    return str(path)


def config_file(tmp_path, workers=1, eta=0.0, pretrain=0):
    task = {"name": "training", "role": "train"}
    for index, name in enumerate(("fit", "search", "validation", "feedback")):
        task[name] = {"cache": write_cache(panel(index, f"2020-{1 + 3 * index:02d}-01"), tmp_path / name)}
    task["future"] = {"cache": write_cache(panel(8, "2021-04-01"), tmp_path / "future")}
    library = tmp_path / "library.jsonl"
    library.write_text(json.dumps({"validated": True, "ast": grammar().parse("volume").to_dict()}) + "\n")
    (tmp_path / "library_manifest.json").write_text(json.dumps({"latest_screening_label": "2019-12-01"}))
    config = {"architecture": "candidate_selection", "seed": 7, "width": 8, "target_horizon": 2,
        "lookback": 2, "future_days": 2, "torch_threads": 1, "signal_cache_size": 16,
        "rollout_workers": workers, "grammar": grammar().config(), "initial_library": str(library),
        "initial_seeds": 1, "corpus": {"count": 24, "noise_steps": 2, "seed": 0},
        "pretrain": {"epochs": pretrain, "batch_size": 8},
        "reward": {"reference_seconds": 0.02, "eta": eta},
        "lower": {"iterations": 1, "batch_formulas": 6, "shortlist_size": 4,
                  "diffusion_steps": 2, "perturbation_steps": 0, "ppo": {"epochs": 1}},
        "upper": {"iterations": 1, "joint_iterations": 1, "episodes_per_batch": 2,
                  "max_members": 2, "pool_attempts": 8, "ppo": {"epochs": 1}},
        "search": {"max_attempts": 8, "budget_mode": "attempts", "candidate_capacity": 8},
        "evaluation": {"methods": ["upper_ppo", "random", "quality", "forward_selection"], "seeds": [3]},
        "tasks": [task]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("device_name, expected_devices", [
    ("cpu", []), ("cuda", [2]), ("cuda:0", [0]), ("cuda:1", [1]),
])
def test_collect_proposal_rng_resolves_cuda_device(monkeypatch, device_name, expected_devices):
    """默认CUDA解析当前编号，显式编号保留，并恢复提案前的随机状态。"""
    current_device = Mock(return_value=2)
    get_state = Mock(side_effect=lambda index: torch.tensor([index], dtype=torch.uint8))
    seed_all_cuda = Mock()
    seed_cuda = Mock()
    cuda_context = Mock(side_effect=lambda index: nullcontext())
    restored_devices = []

    def restore_state(state, device):
        """记录实际恢复的设备，兼容Torch的位置参数和关键字参数调用。"""
        restored_devices.append(device)

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "get_rng_state", get_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", restore_state)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_all_cuda)
    monkeypatch.setattr(torch.cuda, "manual_seed", seed_cuda)
    monkeypatch.setattr(torch.cuda, "device", cuda_context)

    def generate(observation, rng, steps, perturbation_steps, record):
        """通过真实CPU随机流记录采样值，以核验上下文退出后的恢复。"""
        return None, [{"index": 0, "draw": float(torch.rand(()))}], None

    generator = SimpleNamespace(device=torch.device(device_name), generate=generate)
    observation = SimpleNamespace(ids=())
    settings = {"diffusion_steps": 1, "perturbation_steps": 0}
    before = torch.get_rng_state().clone()
    first = collect_proposal(generator, None, observation, settings, 42, "first")
    assert torch.equal(torch.get_rng_state(), before)
    second = collect_proposal(generator, None, observation, settings, 42, "second")
    assert torch.equal(torch.get_rng_state(), before)
    assert first["rows"] == second["rows"]
    assert [call.args[0] for call in get_state.call_args_list] == expected_devices * 2
    assert restored_devices == expected_devices * 2
    assert current_device.call_count == (2 if device_name == "cuda" else 0)
    assert [call.args[0] for call in cuda_context.call_args_list] == expected_devices * 2
    assert seed_cuda.call_count == (2 if expected_devices else 0)
    seed_all_cuda.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA设备")
@pytest.mark.parametrize("device_name", ["cuda", "cuda:0"])
def test_collect_proposal_cuda_backend_restores_rng(tmp_path, monkeypatch, device_name):
    """在GPU上通过实际扩散后端验证默认设备、重复采样和随机状态恢复。"""
    from single_factor.hierarchical.__main__ import load_config
    from single_factor.hierarchical.comparison_generators import DiffusionBackend

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = load_config(config_file(tmp_path))
    config["device"] = device_name
    generator = DiffusionBackend(grammar(), config)
    evaluator = SelectionEvaluator(grammar(), load_selection_task(config["tasks"][0], config), config)
    observation = StartBank([grammar().parse("volume")]).observe(evaluator, 4)
    device = next(generator.policy.parameters()).device
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()
    first = collect_proposal(generator, evaluator, observation, config["lower"], 42, "first")
    second = collect_proposal(generator, evaluator, observation, config["lower"], 42, "second")
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)
    assert first["source_id"] == second["source_id"]
    assert [row["logp"] for row in first["rows"]] == pytest.approx([row["logp"] for row in second["rows"]])


def test_quality_deficiency_and_blocks_match_word_arithmetic():
    thresholds = [0.02, 0.01, 0.10, 0.8]
    metrics = {"ic": 0.03, "rank_ic": 0.015, "icir": 0.15, "coverage": 0.6}
    assert quality_score(metrics, thresholds) == pytest.approx(0.25)
    assert member_deficiency(metrics, thresholds) == pytest.approx(0.0625)
    metrics.update(ic=0.01, rank_ic=0.007, icir=0.2, coverage=0.9)
    assert member_deficiency(metrics, thresholds) == pytest.approx(0.20)
    metrics["ic"] = float("nan")
    assert quality_score(metrics, thresholds) == -1.0
    assert member_deficiency(metrics, thresholds) == pytest.approx((1 + .3) / 4)
    metrics.update(ic=-1, rank_ic=-1, icir=-1, coverage=0.1)
    assert quality_score(metrics, thresholds) == pytest.approx(-1.875)
    assert block_deviation(np.array([.025] * 10 + [.035] * 10)) == pytest.approx(.005)
    assert block_deviation(np.array([np.nan] * 10 + [.02] * 3)) == pytest.approx(.01)


def test_utility_equal_weights_missing_and_leave_one_out_match_direct_calculation(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    config = load_config(config_file(tmp_path))
    panels = load_selection_task(config["tasks"][0], config, with_feedback=True)
    evaluator = SelectionEvaluator(grammar(), panels, config)
    candidates = [evaluator.assess(grammar().parse(name), name)[0] for name in ("volume", "amount")]
    indices = [0, 1]
    score, contributions = evaluator.contributions(candidates, indices)
    prediction = sum(evaluator.segment(c.tree, "feedback")[1] * c.direction for c in candidates) / 2
    daily = daily_correlation(prediction, panels["feedback"].target)
    mean = float(np.nan_to_num(daily).mean())
    deviation = block_deviation(daily)
    h = np.mean([member_deficiency(evaluator.directed_metrics(c.tree, "feedback"), evaluator.thresholds)
                 for c in candidates])
    assert score["utility"] == pytest.approx(50 * mean - 25 * deviation - .5 * h)
    for index, candidate in enumerate(candidates):
        reduced = evaluator.utility(candidates, [1 - index])
        assert contributions[candidate.proposal_id]["value"] == pytest.approx(score["utility"] - reduced["utility"])
    assert evaluator.utility(candidates, [])["utility"] == 0.0
    only = evaluator.contributions(candidates, [0])[1]["volume"]
    assert only["value"] == pytest.approx(only["whole_utility"])
    values = evaluator.segment(candidates[0].tree, "feedback")[0].copy()
    values[:, 0] = np.nan
    z, observed, _ = normalize(values, np.isfinite(values))
    assert not observed[:, 0].any()
    assert not z[:, 0].any()
    with pytest.raises(ValueError, match="distinct"):
        evaluator.utility(candidates, [0, 0])


def test_upper_inputs_never_depend_on_feedback_labels_and_direction_is_fit_only(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    config = load_config(config_file(tmp_path))
    panels = load_selection_task(config["tasks"][0], config, with_feedback=True)
    first = SelectionEvaluator(grammar(), panels, config)
    changed = dict(panels)
    changed["feedback"] = Slice(panels["feedback"].panel, 2, -panels["feedback"].target)
    second = SelectionEvaluator(grammar(), changed, config)
    trees = [grammar().parse(name) for name in ("volume", "amount")]
    left = [first.assess(tree, str(i))[0] for i, tree in enumerate(trees)]
    right = [second.assess(tree, str(i))[0] for i, tree in enumerate(trees)]
    assert first.observation(left, 2) == second.observation(right, 2)
    assert [c.direction for c in left] == [c.direction for c in right]
    assert [c.quality for c in left] == [c.quality for c in right]
    assert first.utility(left, [0])["utility"] != second.utility(right, [0])["utility"]
    bank = StartBank(trees)
    observation = bank.observe(first, 2)
    proposal = {"source_id": trees[0].id, "base_reward": .2, "eligible": False, "terminal_reward": 900}
    bank.commit([proposal])
    assert bank.entries[trees[0].id]["reward_sum"] == .2
    assert all(len(row) == 7 for row in observation.statistics)


def test_selector_mask_permutation_empty_stop_and_nonzero_gradients(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    config = load_config(config_file(tmp_path))
    evaluator = SelectionEvaluator(grammar(), load_selection_task(config["tasks"][0], config, with_feedback=True), config)
    candidates = [evaluator.assess(grammar().parse(name), name)[0] for name in ("volume", "amount", "close")]
    observation = evaluator.observation(candidates, 2)
    selector = SelectorPolicy(grammar(), 8)
    state = replace(observation, selected=(1,))
    distribution, value = selector.distribution(state)
    assert distribution.probs[1] == 0
    permutation = [2, 0, 1]
    shuffled = replace(observation, trees=tuple(observation.trees[i] for i in permutation),
        features=tuple(observation.features[i] for i in permutation),
        correlations=tuple(tuple(observation.correlations[i][j] for j in permutation) for i in permutation), selected=(2,))
    actual, actual_value = selector.distribution(shuffled)
    assert torch.allclose(actual.probs, distribution.probs[[2, 0, 1, 3]], atol=1e-7)
    assert float(actual_value.detach()) == pytest.approx(float(value.detach()), abs=1e-7)
    (distribution.logits[0] + value).backward()
    assert selector.member_head[0].weight.grad.abs().sum() > 0
    assert selector.value_head[0].weight.grad.abs().sum() > 0
    empty = evaluator.observation([], 2)
    assert selector.sample(empty)["index"] == 0
    trajectory, indices, score, _ = select_members(selector, empty, evaluator, [])
    assert len(trajectory) == 1 and indices == () and score["utility"] == 0
    full, _ = selector.distribution(replace(observation, selected=(0, 1)))
    assert full.probs[-1] == 1


@pytest.mark.parametrize("workers", [1, 2])
def test_real_campaign_updates_both_layers_feedback_checkpoint_and_resume(tmp_path, workers):
    from single_factor.hierarchical.__main__ import load_config, run
    config = load_config(config_file(tmp_path, workers=workers, eta=0.7, pretrain=1))
    output = tmp_path / "run"
    summary = run(config, output)
    assert summary["status"] == "complete" and summary["trial_count"] == 4
    state = torch.load(output / "training_state.pt", weights_only=False)
    assert state["campaign_state"]["lower_updates"] == 2
    assert state["campaign_state"]["upper_updates"] == 2
    initial_lower = torch.load(output / "rollouts" / "lower_0.pt", weights_only=False)
    final_lower = torch.load(output / "lower_base.pt", weights_only=False)
    for prefix in ("starter.", "editor."):
        assert any(not torch.equal(v, final_lower["lower"][k]) for k, v in initial_lower["models"]["lower"].items() if k.startswith(prefix))
    initial_upper = torch.load(output / "rollouts" / "upper_0.pt", weights_only=False)
    final_upper = torch.load(output / "upper_frozen_final.pt", weights_only=False)
    assert any(not torch.equal(v, final_upper["upper"][k]) for k, v in initial_upper["models"]["upper"].items())
    for path in (output / "rollouts").glob("*.pt"):
        rollout = torch.load(path, weights_only=False)
        lower, upper = restore_selection_models(rollout["models"])
        policy = lower if "feedback" in rollout else upper
        with torch.no_grad():
            error = max(abs(float(actual[0]) - row["logp"]) for actual, row in zip(policy.rows(rollout["rows"]), rollout["rows"]))
        assert error < 2e-5
    joint = torch.load(output / "rollouts" / "joint_0_lower.pt", weights_only=False)
    for feedback in joint["feedback"]:
        assert feedback["terminal_reward"] == pytest.approx(feedback["base_reward"] + .7 * feedback["contribution"])
    for line in (output / "generation_attempts.jsonl").read_text().splitlines():
        proposal = json.loads(line)
        assert proposal["base_reward"] == pytest.approx(proposal["quality"] - .05 * proposal["cost"])
        assert proposal["cost"] == pytest.approx(proposal["seconds"] / .02)
    fresh = torch.load(output / "candidate_pools" / "joint_0_fresh_upper_training.pt", weights_only=False)
    assert fresh["generator_version"] == 2
    report = json.loads((output / "trials" / "training_3_upper_ppo.json").read_text())
    assert report["future"]["directions_refitted"] is False
    if report["member_count"]:
        assert report["pool_weights"] == [1 / report["member_count"]] * report["member_count"]
    portfolio = json.loads((output / "portfolios" / "training_3.json").read_text())
    assert portfolio["information_cutoff"] < "2021-04-01"
    from single_factor.replay_hierarchical import replay
    for layer in ("lower", "upper"):
        proof = replay(output / "rollouts" / f"{layer}_0.pt", output / "resolved_config.json",
                       output / f"{layer}_replay.json")
        assert proof["repeat_max_parameter_difference"] == 0
    before = (output / "lower_curve.jsonl").read_bytes()
    assert run(config, output, resume=True) == summary
    assert (output / "lower_curve.jsonl").read_bytes() == before
    mismatched = copy.deepcopy(config)
    mismatched["fingerprint"] = "different"
    with pytest.raises(ValueError, match="identity"):
        run(mismatched, output, resume=True)


def test_parallel_actions_and_old_probabilities_equal_serial_collection(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    config = load_config(config_file(tmp_path))
    generator = GeneratorPolicy(grammar(), 8, 2)
    task = config["tasks"][0]
    evaluator = SelectionEvaluator(grammar(), load_selection_task(task, config), config)
    bank = StartBank([grammar().parse("volume")])
    observation = bank.observe(evaluator, 4)
    serial = ProposalCollector(config)
    parallel = ProposalCollector({**config, "rollout_workers": 2})
    try:
        left = serial.collect(generator, evaluator, task, observation, [12, 13], ["a", "b"])
        right = parallel.collect(generator, evaluator, task, observation, [12, 13], ["a", "b"])
        assert [p["candidate"].tree for p in left] == [p["candidate"].tree for p in right]
        for first, second in zip(left, right):
            assert [r["logp"] for r in first["rows"]] == pytest.approx([r["logp"] for r in second["rows"]], abs=1e-6)
    finally:
        serial.close()
        parallel.close()


def test_positive_eta_reaches_start_and_edit_returns(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    config = load_config(config_file(tmp_path, eta=0.7))
    campaign = SelectionCampaign(config, tmp_path / "credit")
    try:
        task = config["tasks"][0]
        _, proposals, _ = campaign.generate_pool(task, 2, "explicit_credit", record=True)
        contribution = {proposals[0]["proposal_id"]: {"value": .25}}
        campaign._lower_update(proposals, task, "explicit_credit", contribution)
        expected = proposals[0]["base_reward"] + .7 * .25
        assert proposals[0]["rows"][0]["return"] == pytest.approx(expected)
        assert proposals[0]["rows"][-1]["return"] == pytest.approx(expected)
        assert proposals[1]["rows"][0]["return"] == pytest.approx(proposals[1]["base_reward"])
    finally:
        campaign.collector.close()


def test_config_rejects_bad_weights_and_label_overlap(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    path = config_file(tmp_path)
    raw = json.loads(path.read_text())
    raw["reward"]["alpha"] = float("nan")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="alpha"):
        load_config(path)
    raw["reward"]["alpha"] = 50
    raw["tasks"][0]["validation"] = raw["tasks"][0]["search"]
    path.write_text(json.dumps(raw))
    config = load_config(path)
    with pytest.raises(ValueError, match="immature"):
        SelectionCampaign(config, tmp_path / "invalid")


def test_platform_bridge_uses_four_training_sections_and_sealed_test(tmp_path):
    from single_factor.hierarchical.bridge import run_from_runtime
    path = config_file(tmp_path)
    template = json.loads(path.read_text())
    template["upper"]["joint_iterations"] = 0
    path.write_text(json.dumps(template))
    train = write_cache(panel(40, "2017-01-01", 160), tmp_path / "platform_train")
    valid = write_cache(panel(41, "2021-01-01", 64), tmp_path / "platform_valid")
    test = write_cache(panel(42, "2022-01-01", 32), tmp_path / "platform_test")
    runtime = {"target_horizon": 2, "max_backtrack_days": 2, "max_future_days": 2,
               "fields": ["close", "volume", "amount"],
               "splits": {name: {"cache_dir": cache} for name, cache in zip(("train", "valid", "test"), (train, valid, test))}}
    settings = {"config": str(path)}
    output = tmp_path / "platform_run"
    result = run_from_runtime(runtime, output, settings, 9, "cpu", 8, 8)
    assert result.timesteps > 0 and result.attempts == 8
    resolved = json.loads((output / "resolved_config.json").read_text())
    assert {"fit", "search", "validation", "feedback"} <= resolved["tasks"][0].keys()
    assert resolved["reward"]["member_segment"] == "feedback"
    assert "initial_library" not in resolved
    report = json.loads((output / "trials" / "platform_search_9_upper_ppo.json").read_text())
    assert report["future"]["weights_refitted"] is False
    resumed = run_from_runtime(runtime, output, settings, 9, "cpu", 8, 8)
    assert resumed.accepted_count == result.accepted_count


def test_cost_calibration_and_interrupted_stage_resume(tmp_path):
    from single_factor.hierarchical.__main__ import load_config, run
    path = config_file(tmp_path)
    raw = json.loads(path.read_text())
    raw["reward"].update(reference_seconds=None, calibration_samples=2)
    raw["lower"]["iterations"] = 0
    raw["upper"].update(iterations=0, joint_iterations=0)
    path.write_text(json.dumps(raw))
    config = load_config(path)
    output = tmp_path / "calibration"
    campaign = SelectionCampaign(config, output)
    try:
        campaign._pretrain()
        campaign._calibrate()
        reference = campaign.state["reference_seconds"]
    finally:
        campaign.collector.close()
    calibration = json.loads((output / "cost_calibration.json").read_text())
    assert reference == pytest.approx(float(np.median(calibration["samples_seconds"])))
    assert reference > 0
    summary = run(config, output, resume=True)
    assert summary["reference_seconds"] == reference
    assert summary["status"] == "complete"


def test_candidate_pool_deduplication_capacity_and_late_results(tmp_path, monkeypatch):
    from single_factor.hierarchical.__main__ import load_config
    config = load_config(config_file(tmp_path))
    evaluator = SelectionEvaluator(grammar(), load_selection_task(config["tasks"][0], config), config)
    first = evaluator.assess(grammar().parse("volume"), "first")[0]
    second = evaluator.assess(grammar().parse("amount"), "second")[0]
    pool = CandidatePool(1)
    original = {"candidate": first, "eligible": True}
    pool.add(original)
    duplicate = {"candidate": replace(first, proposal_id="copy"), "eligible": True}
    pool.add(duplicate)
    assert duplicate["duplicate"] and not duplicate["accepted"]
    same_signal = {"candidate": replace(second, signal_hash=first.signal_hash), "eligible": True}
    pool.add(same_signal)
    assert same_signal["duplicate"]
    distinct = {"candidate": replace(second, quality=first.quality + .1), "eligible": True}
    pool.add(distinct)
    assert pool.candidates[0].proposal_id == "second"
    assert len(pool.discoveries) == 2
    campaign = SelectionCampaign(config, tmp_path / "late")
    try:
        collect = campaign.collector.collect
        def late(*args, **kwargs):
            result = collect(*args, **kwargs)
            for proposal in result:
                proposal["finished_at"] = float("inf")
            return result
        monkeypatch.setattr(campaign.collector, "collect", late)
        pool, proposals, report = campaign.generate_pool(config["tasks"][0], 2, "late", wall_seconds=2)
        assert report["attempts"] == report["late_results_excluded"] == 2
        assert not pool.candidates and not proposals
        assert campaign.banks["training"].new["attempts"] == 0
    finally:
        campaign.collector.close()


def test_atomic_platform_exports_promote_accepted_and_retry_without_duplicates(tmp_path):
    campaign = SelectionCampaign.__new__(SelectionCampaign)
    campaign.output = tmp_path
    rejected = {"expression": "close", "accepted": False, "score": -.1}
    accepted = {"expression": "close", "accepted": True, "score": .2}
    campaign._platform_exports([rejected], "a")
    campaign._platform_exports([accepted], "b")
    campaign._platform_exports([accepted], "b")
    records = (tmp_path / "accepted_factors.jsonl").read_text().splitlines()
    assert len(records) == 1 and json.loads(records[0])["score"] == .2
    assert not (tmp_path / "rejected_candidates.jsonl").read_text()
    assert json.loads((tmp_path / "expression_index.json").read_text()) == {"close": 1}


def test_new_architecture_suite_trains_independent_seeds(tmp_path):
    from single_factor.hierarchical.__main__ import load_config, run_suite
    path = config_file(tmp_path)
    raw = json.loads(path.read_text())
    raw["upper"]["joint_iterations"] = 0
    raw["evaluation"]["methods"] = ["upper_ppo"]
    path.write_text(json.dumps(raw))
    config = load_config(path)
    output = tmp_path / "suite"
    summary = run_suite(config, output, [1, 2])
    assert summary["independent_training_runs"] == 2
    states = [torch.load(output / f"training_seed_{seed}" / "lower_base.pt", weights_only=False) for seed in (1, 2)]
    assert any(not torch.equal(value, states[1]["lower"][name]) for name, value in states[0]["lower"].items())
    assert run_suite(config, output, [1, 2], resume=True)["aggregate"] == summary["aggregate"]


def test_daily_pearson_matches_numpy_on_common_finite_samples():
    x = np.array([[1, 2, np.nan, 4], [5, 5, 5, 5], [9, 1, 6, 3]], dtype=float)
    y = np.array([[4, 3, 2, 0], [1, 2, 3, 4], [2, np.nan, 7, 8]], dtype=float)
    actual = daily_correlation(x, y)
    assert actual[0] == pytest.approx(np.corrcoef([1, 2, 4], [4, 3, 0])[0, 1])
    assert np.isnan(actual[1])
    assert actual[2] == pytest.approx(np.corrcoef([9, 6, 3], [2, 7, 8])[0, 1])


def test_numeric_verification_entry_point_supports_new_architecture(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    from single_factor.hierarchical.verify import check
    result = check(load_config(config_file(tmp_path)))
    assert result["combination_ic_max_error"] < 1e-9
    assert result["sampler_logp_max_error"] < 2e-5
    assert result["future_loaded"] is False


def test_immediate_stop_exports_zero_member_portfolio_through_complete_campaign(tmp_path):
    from single_factor.hierarchical.__main__ import load_config
    path = config_file(tmp_path)
    raw = json.loads(path.read_text())
    raw["lower"]["iterations"] = 0
    raw["upper"].update(iterations=0, joint_iterations=0)
    raw["evaluation"]["methods"] = ["upper_ppo"]
    path.write_text(json.dumps(raw))
    output = tmp_path / "empty_portfolio"
    campaign = SelectionCampaign(load_config(path), output)
    with torch.no_grad():
        campaign.upper.stop_head[-1].weight.zero_()
        campaign.upper.stop_head[-1].bias.fill_(40)
        campaign.upper.member_head[-1].weight.zero_()
        campaign.upper.member_head[-1].bias.fill_(-40)
    summary = campaign.run()
    assert summary["status"] == "complete"
    report = json.loads((output / "trials" / "training_3_upper_ppo.json").read_text())
    assert report["pool_weights"] == [] and report["member_count"] == 0
    assert report["return"] == 0 and report["future"]["utility"] == 0
    assert report["future"]["contributions"] == {}
    portfolio = json.loads((output / "portfolios" / "training_3.json").read_text())
    assert portfolio["members"] == []
    with np.load(output / "portfolios" / "training_3_signals.npz") as signals:
        assert not signals["scores"].any()
