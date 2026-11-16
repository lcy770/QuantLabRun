"""实验正常入口、真实学习器、交易记账及跨机数据传输的工程验收。"""
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from single_factor.hierarchical.comparison_cli import (
    evaluate_frozen, freeze_selection, run_study, transfer_data,
)
from single_factor.hierarchical.comparison_data import design_tasks, validate_cache
from single_factor.hierarchical.comparison_metrics import portfolio_curve
from single_factor.hierarchical.comparison_study import experiment_nodes, summarize_study, TEMPLATES
from single_factor.hierarchical.comparison_training import Study
from single_factor.hierarchical.io import digest, save_json
from single_factor.hierarchical.selection_config import load_selection_config
from single_factor.tests.test_candidate_selection import config_file, grammar, panel, write_cache


@pytest.fixture
def protocol(tmp_path):
    """三个时间隔离任务，明确标记为合成数据工程验收。"""
    path = config_file(tmp_path, pretrain=1)
    config = json.loads(path.read_text())
    Path(config["initial_library"]).write_text("\n".join(json.dumps({"validated": True,
        "ast": grammar().parse(expression).to_dict()}) for expression in ("volume", "close")) + "\n")
    tasks = []
    for year, role in ((2020, "train"), (2022, "development"), (2024, "test")):
        task = {"name": role, "role": role}
        segments = ("fit", "search", "validation", "future") if role == "test" else ("fit", "search", "validation", "feedback")
        for index, segment in enumerate(segments):
            scoring = panel(index, f"{year}-{1 + index * 3:02d}-01")
            arrays = scoring.panel._arrays
            arrays["open"] = arrays["close"].copy()
            arrays["volume"] = np.abs(arrays["volume"]) + 1
            task[segment] = {"cache": write_cache(scoring, tmp_path / f"{role}_{segment}")}
        tasks.append(task)
    settings = json.loads((TEMPLATES / "profiles.json").read_text())["smoke"]
    settings.update(pool_attempts=4, attempt_budget=4, time_budget=1e-9)
    config.update(tasks=tasks, device="cpu", record_rollouts=True, data_mode="exploratory")
    config["autoregressive"] = {"n_steps": 32, "batch_size": 16, "epochs": 1, "width": 16}
    config["portfolio"] = {"top_k": 2, "initial_cash": 100000, "rebalance_days": 5,
                           "fee_rate": .0015, "minimum_fee": 5, "slippage": .0005}
    config["upper"]["joint_iterations"] = 0
    config["upper"]["double_dqn"] = {"gradient_steps": 1, "target_update_interval": 1}
    save_json(path, config)
    resolved = load_selection_config(path)
    result = {"config": resolved, "settings": settings, "nodes": experiment_nodes(),
              "dates": {"mode": "synthetic_engineering_validation"}, "quantity_penalty": False}
    result["fingerprint"] = digest(result)
    protocol_path = tmp_path / "protocol.json"
    save_json(protocol_path, result)
    return protocol_path


@pytest.mark.parametrize("comparison_scope", ["development", "training"])
def test_all_experiment_nodes_train_restore_freeze_and_evaluate(protocol, tmp_path, comparison_scope):
    if comparison_scope == "training":
        definition = json.loads(protocol.read_text())
        config = definition["config"]
        config["tasks"] = [task for task in config["tasks"] if task["role"] == "train"]
        config_path = protocol.with_name("config.json")
        save_json(config_path, config)
        definition["config"] = load_selection_config(config_path)
        definition["dates"] = {"mode": "shared_training_comparison", "comparison_uses_training_data": True}
        definition["fingerprint"] = digest({key: value for key, value in definition.items() if key != "fingerprint"})
        save_json(protocol, definition)
    output = tmp_path / "results"
    result = run_study(protocol, output, [0])
    assert result[f"{comparison_scope}_reports"] > 0
    assert result["evaluation_scope"] == comparison_scope
    assert result["evaluation_uses_training_data"] == (comparison_scope == "training")
    study = Study(protocol, output)
    if comparison_scope == "training":
        assert study.evaluation_tasks("training") == study.training_tasks
        reports = list(output.glob("seed_*/nodes/*/training/*.json"))
        assert reports
        assert all(json.loads(path.read_text())["evaluation_uses_training_data"] for path in reports)
        assert not list(output.glob("seed_*/nodes/*/development/*.json"))
        assert summarize_study(output)["training_reports"] == len(reports)
    assert torch.get_num_threads() == study.config["torch_threads"]
    for name, definition in experiment_nodes().items():
        saved = study.saved_node(name, 0)
        assert saved["state"]["phase"] == "complete"
        if definition.get("lower"):
            assert saved["state"]["generator_version"] > 0
        if definition.get("upper"):
            assert saved["state"]["selector_version"] > 0
    dqn = study.saved_node("s_double_dqn", 0)["selector"]
    assert dqn["updates"] > 0 and dqn["replay"]
    for name in dqn["online"]:
        assert torch.equal(dqn["online"][name], dqn["target"][name])
    assert np.trace(study.saved_node("s_linucb", 0)["selector"]["a"]) > 24
    pools = [study.saved_node(name, 0)["pools"]["train"]["observation"].pool_id
             for name in ("s_random", "s_quality", "s_forward_exchange", "s_linucb", "s_double_dqn", "s_upper_ppo", "a_no_quality", "a_no_stability")]
    assert len(set(pools)) == 1
    baseline = study.saved_node("s_upper_ppo", 0)
    upper_only = study.saved_node("c_upper", 0)
    lower_only = study.saved_node("c_lower", 0)
    assert upper_only["pools"]["train"]["observation"].pool_id == baseline["pools"]["train"]["observation"].pool_id
    for name, value in baseline["generator"]["policy"].items():
        assert torch.equal(value, upper_only["generator"]["policy"][name])
    for name, value in baseline["selector"]["policy"].items():
        assert torch.equal(value, lower_only["selector"]["policy"][name])
    assert not (output / "test_access.jsonl").exists()
    checkpoint = study.node_path("g_diffusion", 0) / "checkpoint.pt"
    before = checkpoint.read_bytes()
    run_study(protocol, output, [0])
    assert checkpoint.read_bytes() == before
    selection = tmp_path / "selection.json"
    freeze_selection(protocol, output, ["p_new_dual", "p_old_dual"], [0], selection)
    if comparison_scope == "training":
        assert json.loads(selection.read_text())["selection_basis"] == "training"
        with pytest.raises(ValueError, match="no test tasks"):
            evaluate_frozen(protocol, output, selection)
        assert not (output / "test_access.jsonl").exists()
        return
    assert evaluate_frozen(protocol, output, selection)["status"] == "complete"
    reports = list(output.glob("seed_*/nodes/*/test/*.json"))
    assert len(reports) == 4
    assert all(json.loads(p.read_text())["selection_identity"] for p in reports)
    assert summarize_study(output, split="test")["test_reports"] == 4
    assert (output / "test_weight_comparison.csv").exists()
    assert (output / "weight_comparison.csv").exists()


def test_unified_rpn_failure_reward_includes_measured_cost(protocol, tmp_path, monkeypatch):
    from single_factor.hierarchical import comparison_generators as generators
    from single_factor.environment import SingleFactorEnvWrapper
    study = Study(protocol, tmp_path / "reward_check")
    generator = generators.create_generator("rpn_unified", study.grammar, study.config)
    generator.bind(study.evaluator(study.training_tasks[0]), 2.0)
    generator.started = 10.0
    monkeypatch.setattr(generators, "perf_counter", lambda: 11.0)
    monkeypatch.setattr("custom_bt.canonical_expression.serialize_alphagen_expression", lambda expression: "unknown_feature")
    assert generator.evaluate(None).score == pytest.approx(-1.025)
    monkeypatch.setattr(SingleFactorEnvWrapper, "step", lambda self, action: ([], -1.0, True, False, {"reason": "too long"}))
    assert generator.environment.step(0)[1] == pytest.approx(-1.025)


def test_ridge_coefficients_use_only_fit_segment(protocol, tmp_path):
    from single_factor.hierarchical.comparison_metrics import locked_ridge_weights
    study = Study(protocol, tmp_path / "ridge_check")
    evaluator = study.evaluator(study.training_tasks[0])
    candidate = evaluator.assess(study.grammar.parse("volume"), "ridge")[0]
    before = locked_ridge_weights(evaluator, [candidate], [0])
    evaluator.panels["feedback"].target[:] *= -100
    assert locked_ridge_weights(evaluator, [candidate], [0]) == before
    signal = evaluator.segment(candidate.tree, "fit")[1] * candidate.direction
    fitting = evaluator.panels["fit"]
    expected = np.sum(fitting.fit_weights * signal * fitting.y) / (np.sum(fitting.fit_weights * signal**2) + .001)
    assert before == pytest.approx([expected])


def test_next_open_accounting_fees_missing_prices_and_cash():
    scoring = panel(1, "2024-01-01", count=4)
    arrays = scoring.panel._arrays
    arrays["open"] = np.full_like(arrays["close"], 10.0)
    arrays["close"][:] = 10
    arrays["volume"][:] = 100
    arrays["close"][3:, 0] = 20
    signal = np.zeros_like(scoring.target)
    signal[:, 0] = 1
    settings = {"initial_cash": 1000, "top_k": 1, "rebalance_days": 20,
                "fee_rate": 0, "minimum_fee": 0, "slippage": 0}
    result = portfolio_curve(scoring, signal, settings)
    assert result["daily"][0]["nav"] == 1
    assert result["daily"][1]["nav"] == 2
    fees = portfolio_curve(scoring, signal, {**settings, "minimum_fee": 5})
    assert fees["daily"][1]["nav"] == pytest.approx(1.99)
    assert fees["daily"][1]["commission"] == .005
    assert fees["daily"][1]["cash_weight"] >= 0
    arrays["open"][3, 0] = np.nan
    blocked = portfolio_curve(scoring, signal, settings)
    assert blocked["net_return"] == 0
    assert portfolio_curve(scoring, signal * 0, settings)["net_return"] == 0


def test_calendar_split_and_insufficient_cache(protocol, tmp_path):
    tasks, summary = design_tasks(tmp_path)
    assert summary["mode"] == "shared_training_comparison"
    assert summary["training_tasks"] == summary["comparison_tasks"] == len(tasks)
    assert len(tasks) > 27
    assert summary["training_information_start"] == "2024-01-02"
    assert summary["training_label_cutoff"] == "2026-06-16"
    assert summary["comparison_information_cutoff"] == "2026-06-16"
    assert summary["latest_signal_date"] == "2026-05-19"
    assert summary["comparison_uses_training_data"] is True
    assert summary["independent_test_available"] is False
    assert all(task["role"] == "train" for task in tasks)
    assert tasks[-1]["feedback"]["end"] == "2026-05-19"
    heldout, heldout_summary = design_tasks(tmp_path, development_cutoff="2026-03-31")
    assert heldout_summary["training_tasks"] == 27
    assert heldout[-2]["search"]["start"] == "2025-09-25"
    assert heldout[-1]["validation"]["end"] == "2026-03-03"
    _, previous = design_tasks(tmp_path, start="2025-01-02", development_cutoff="2026-08-31",
                               test_cutoff="2026-11-11")
    assert previous["training_tasks"] == 13
    assert previous["test_signal_start"] == "2026-09-01"
    cache = Path(json.loads(protocol.read_text())["config"]["tasks"][0]["fit"]["cache"])
    with pytest.raises(ValueError, match="coverage insufficient"):
        validate_cache(cache, tasks, 2, 20, ["close"])


def test_prepare_cli_resolves_relative_task_caches(tmp_path, monkeypatch):
    """迁移机器后，显式任务缓存仍相对于源配置解析。"""
    import sys
    from single_factor.hierarchical import comparison_cli

    config = tmp_path / "configs" / "tasks.json"
    save_json(config, {"tasks": [{"name": "train", "role": "train", "fit": {"cache": "../data"}}]})
    captured = {}

    def prepare(cache, output, profile, device, tasks):
        """捕获命令行传入准备器的实际任务路径。"""
        captured["tasks"] = tasks
        return {"dates": {"comparison_uses_training_data": True}}

    monkeypatch.setattr(comparison_cli, "prepare_protocol", prepare)
    monkeypatch.setattr(sys, "argv", ["comparison_cli", "prepare", "--cache", str(tmp_path / "data"),
                        "--output", str(tmp_path / "protocol"), "--tasks-config", str(config)])
    comparison_cli.main()
    assert captured["tasks"][0]["fit"]["cache"] == str((tmp_path / "data").resolve())


def test_transfer_round_trip_and_unsafe_archive(protocol, tmp_path):
    cache = Path(json.loads(protocol.read_text())["config"]["tasks"][0]["fit"]["cache"])
    archive = tmp_path / "market.zip"
    transfer_data(cache, archive)
    destination = tmp_path / "received"
    result = transfer_data(destination, archive, unpack=True)
    assert result["verified_files"] > 0
    assert np.array_equal(np.load(cache / "close.npy"), np.load(destination / "close.npy"))
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        transfer_data(tmp_path / "unsafe_destination", unsafe, unpack=True)
    corrupted = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(corrupted, "w") as changed:
        for entry in original.namelist():
            changed.writestr(entry, b"corrupted" if entry == "close.npy" else original.read(entry))
    with pytest.raises(ValueError, match="hash mismatch"):
        transfer_data(tmp_path / "corrupted_destination", corrupted, unpack=True)
    assert not (tmp_path / "corrupted_destination").exists()
