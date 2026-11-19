"""实验正常入口、真实学习器、交易记账及跨机数据传输的工程验收。"""
import json
from pathlib import Path
import random
import zipfile
from unittest.mock import Mock

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


@pytest.mark.parametrize("entry_point", ["comparison", "selection", "legacy"])
def test_resume_normalizes_cuda_rng_states_to_cpu(protocol, tmp_path, monkeypatch, entry_point):
    """模拟检查点映射到GPU后，三个恢复入口均先还原CPU随机状态张量。"""
    from single_factor.hierarchical.comparison_training import NodeRunner
    from single_factor.hierarchical.selection_training import SelectionCampaign
    from single_factor.hierarchical.training import Campaign
    from single_factor.tests.test_hierarchical_search import tiny_training_config

    output = tmp_path / "restore_rng"
    if entry_point == "comparison":
        study = Study(protocol, output)
        create = lambda: NodeRunner(study, "syntax", 0)
        original = create()
        checkpoint = original.root / "checkpoint.pt"
    else:
        if entry_point == "selection":
            config = load_selection_config(protocol.with_name("config.json"))
            campaign_type = SelectionCampaign
        else:
            config_root = tmp_path / "legacy_config"
            config_root.mkdir()
            config = tiny_training_config(config_root)
            campaign_type = Campaign
        original = campaign_type(config, output)
        create = lambda: campaign_type(config, output, resume=True)
        checkpoint = output / "training_state.pt"
    original.commit()
    metadata = torch.load(checkpoint, weights_only=False)["model_metadata"]
    assert metadata["width"] == original.config["width"]
    assert metadata["total_parameters"] > 0
    if getattr(original, "collector", None):
        original.collector.close()
    original_load = torch.load
    cpu_state = torch.zeros(8, dtype=torch.uint8)
    gpu_mapped_state = Mock()
    gpu_mapped_state.cpu.return_value = cpu_state
    restore_cuda = Mock()

    def load_mapped_rng(path, *args, **kwargs):
        """保持真实检查点加载，仅模拟CUDA随机状态被map_location搬到GPU。"""
        saved = original_load(path, *args, **kwargs)
        if Path(path) == checkpoint:
            saved["cuda_rng"] = [gpu_mapped_state]
        return saved

    monkeypatch.setattr(torch, "load", load_mapped_rng)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", Mock())
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore_cuda)
    restored = create()
    try:
        gpu_mapped_state.cpu.assert_called_once_with()
        restore_cuda.assert_called_once()
        assert restore_cuda.call_args.args[0][0] is cpu_state
    finally:
        if getattr(restored, "collector", None):
            restored.collector.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA设备")
def test_cuda_checkpoint_resume_preserves_rng_and_optimizer(protocol, tmp_path, monkeypatch):
    """在真实GPU上完成预训练、保存与恢复，并核验后续CPU/CUDA随机流。"""
    from single_factor.hierarchical.comparison_training import NodeRunner
    from single_factor.hierarchical.training import create_corpus

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_path = protocol.with_name("config.json")
    config = json.loads(config_path.read_text())
    config["device"] = "cuda"
    save_json(config_path, config)
    definition = json.loads(protocol.read_text())
    definition["config"] = load_selection_config(config_path)
    definition["fingerprint"] = digest({key: value for key, value in definition.items() if key != "fingerprint"})
    save_json(protocol, definition)
    study = Study(protocol, tmp_path / "gpu_resume")
    original = NodeRunner(study, "syntax", 0)
    original.run()
    expected_cpu = torch.rand(4)
    expected_cuda = torch.rand(4, device="cuda")
    restored = NodeRunner(study, "syntax", 0)
    assert torch.equal(torch.rand(4), expected_cpu)
    assert torch.equal(torch.rand(4, device="cuda"), expected_cuda)
    old_optimizer = original.generator.syntax_optimizer.state_dict()
    new_optimizer = restored.generator.syntax_optimizer.state_dict()
    assert old_optimizer["state"] and old_optimizer["param_groups"] == new_optimizer["param_groups"]
    for parameter, state in old_optimizer["state"].items():
        for name, value in state.items():
            assert torch.equal(new_optimizer["state"][parameter][name].cpu(), value.cpu())
    records = create_corpus(study.grammar, study.config["corpus"], restored.root, "rpn")
    metrics = restored.generator.pretrain(records, random.Random(42))
    assert np.isfinite(metrics["train_nll"])


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
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["width"] == study.config["width"]
    assert manifest["rpn_width"] == study.config["autoregressive"]["width"]
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
        metadata = saved["model_metadata"]
        assert metadata["width"] == study.config["width"]
        actual_name = study.canonical(name)
        metadata_path = study.node_path(actual_name, 0) / "model_metadata.json"
        assert json.loads(metadata_path.read_text()) == metadata
        if actual_name in {"g_rpn_original", "g_rpn_unified"}:
            assert metadata["lower"]["width"] == study.config["autoregressive"]["width"]
            assert metadata["lower"]["parameter_count"] > 0
        if actual_name in {"g_random", "g_gp"}:
            assert metadata["lower"]["parameter_count"] == 0
            assert metadata["lower"]["width"] is None
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
    tasks, summary = design_tasks(tmp_path, start="2024-01-02", test_cutoff="2026-06-16")
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
    heldout, heldout_summary = design_tasks(tmp_path, start="2024-01-02", development_cutoff="2026-03-31", test_cutoff="2026-06-16")
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


@pytest.mark.parametrize("width", [None, 128, 256])
def test_prepare_cli_resolves_relative_task_caches(tmp_path, monkeypatch, width):
    """显式任务缓存相对于源配置解析，网络宽度只在用户指定时覆盖。"""
    import sys
    from single_factor.hierarchical import comparison_cli

    config = tmp_path / "configs" / "tasks.json"
    save_json(config, {"tasks": [{"name": "train", "role": "train", "fit": {"cache": "../data"}}]})
    captured = {}

    def prepare(cache, output, profile, device, tasks, **overrides):
        """捕获命令行传入准备器的实际任务路径和宽度覆盖。"""
        captured["tasks"] = tasks
        captured["overrides"] = overrides
        return {"dates": {"comparison_uses_training_data": True}, "config": {"width": overrides.get("width", 32)}}

    monkeypatch.setattr(comparison_cli, "prepare_protocol", prepare)
    argv = ["comparison_cli", "prepare", "--cache", str(tmp_path / "data"),
            "--output", str(tmp_path / "protocol"), "--tasks-config", str(config)]
    if width is not None:
        argv += ["--width", str(width)]
    monkeypatch.setattr(sys, "argv", argv)
    comparison_cli.main()
    assert captured["tasks"][0]["fit"]["cache"] == str((tmp_path / "data").resolve())
    assert captured["overrides"] == ({"width": width} if width is not None else {})


@pytest.mark.parametrize("width", [None, 128, 256])
@pytest.mark.parametrize("profile", ["smoke", "primary"])
def test_prepare_protocol_freezes_model_width(protocol, tmp_path, monkeypatch, width, profile):
    """真实准备并加载协议，确认上下层使用覆盖宽度而自回归对照保持原规模。"""
    from single_factor.hierarchical import comparison_study
    from single_factor.hierarchical.comparison_generators import create_generator
    from single_factor.hierarchical.comparison_selectors import create_selector

    config = json.loads(protocol.read_text())["config"]
    templates = tmp_path / "templates"
    save_json(templates / "base.json", config)
    profiles = json.loads((TEMPLATES / "profiles.json").read_text())
    save_json(templates / "profiles.json", profiles)
    monkeypatch.setattr(comparison_study, "TEMPLATES", templates)
    root = tmp_path / "prepared"
    result = comparison_study.prepare_protocol(tmp_path, root, profile=profile, device="cpu",
                                               tasks=config["tasks"], width=width)
    expected = profiles[profile]["width"] if width is None else width
    assert result["settings"]["width"] == result["config"]["width"] == expected
    assert result["config"]["study_profile"]["width"] == expected
    assert result["config"]["autoregressive"]["width"] == profiles[profile]["rpn_width"]
    if profile == "primary":
        assert result["settings"]["lower_seconds"] == 48 * 3600
        assert result["settings"]["upper_seconds"] == 24 * 3600
        assert result["settings"]["continuation_seconds"] == 48 * 3600
    study = Study(root / "protocol.json", tmp_path / "prepared_results")
    generator = create_generator("diffusion_ppo", study.grammar, study.config)
    selector = create_selector("upper_ppo", study.grammar, study.config)
    assert generator.policy.starter.width == generator.policy.editor.width == expected
    assert selector.policy.actor.encoder.width == selector.policy.critic.encoder.width == expected
    assert json.loads((templates / "profiles.json").read_text()) == profiles
    if width is not None and profile == "smoke":
        run_study(root / "protocol.json", study.output, [0], nodes=["p_new_dual"])
        assert study.saved_node("g_diffusion", 0)["state"]["lower_iteration"] == 1
        saved = study.saved_node("s_upper_ppo", 0)
        assert saved["state"]["upper_iteration"] == 1
        metadata = saved["model_metadata"]
        assert metadata["width"] == metadata["lower"]["width"] == metadata["upper"]["width"] == width
        assert metadata["lower"]["parameter_count"] == sum(parameter.numel() for parameter in generator.policy.parameters())
        assert metadata["upper"]["parameter_count"] == sum(parameter.numel() for parameter in selector.policy.parameters())
        assert metadata["total_parameters"] == metadata["lower"]["parameter_count"] + metadata["upper"]["parameter_count"]
        assert metadata["total_weight_bytes"] == 4 * metadata["total_parameters"]
        assert metadata["lower"]["dtypes"] == metadata["upper"]["dtypes"] == ["float32"]
        assert metadata["node"] == "s_upper_ppo" and metadata["training_seed"] == 0
        node_root = study.node_path("s_upper_ppo", 0)
        assert json.loads((node_root / "model_metadata.json").read_text()) == metadata
        assert json.loads((node_root / "progress.json").read_text())["model_metadata"] == metadata
        # 重复运行同时核验大模型检查点及其优化器状态可被完整加载。
        run_study(root / "protocol.json", study.output, [0], nodes=["p_new_dual"])


def test_train_only_then_evaluate_reuses_models(protocol, tmp_path, monkeypatch):
    """主算法先完成各阶段训练，之后同协议评价且不重复优化或改写检查点。"""
    import sys
    from single_factor.hierarchical import comparison_cli
    from single_factor.hierarchical.selection_config import file_hash

    output = tmp_path / "primary_results"
    with monkeypatch.context() as guard:
        evaluate = Mock(side_effect=AssertionError("训练阶段不应执行完整评价"))
        guard.setattr(Study, "evaluate_node", evaluate)
        guard.setattr(sys, "argv", ["comparison_cli", "run", "--protocol", str(protocol),
            "--output", str(output), "--seeds", "0", "--nodes", "c_alternating", "--train-only"])
        comparison_cli.main()
        evaluate.assert_not_called()
    study = Study(protocol, output)
    assert not list(output.glob("seed_*/nodes/*/development/*.json"))
    assert not (output / "comparison.csv").exists()
    assert json.loads((output / "manifest.json").read_text())["evaluation_deferred"] is True
    progress = json.loads((output / "progress.json").read_text())
    assert progress["phase"] == "complete" and progress["evaluation_deferred"] is True
    assert progress["completed_nodes"] == ["syntax", "g_diffusion", "s_upper_ppo", "c_alternating"]
    assert study.saved_node("g_diffusion", 0)["state"]["lower_iteration"] == 1
    assert study.saved_node("s_upper_ppo", 0)["state"]["upper_iteration"] == 1
    assert study.saved_node("c_alternating", 0)["state"]["continuation_iteration"] == 1
    hashes = {path: file_hash(path) for path in output.glob("seed_*/nodes/*/checkpoint.pt")}
    result = run_study(protocol, output, [0], nodes=["c_alternating"])
    assert result["development_reports"] > 0
    assert json.loads((output / "manifest.json").read_text())["evaluation_deferred"] is False
    assert {path: file_hash(path) for path in hashes} == hashes


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_prepare_protocol_rejects_invalid_width(tmp_path, width):
    """非法宽度在生成配置和协议前失败，避免发布不可训练的实验目录。"""
    from single_factor.hierarchical.comparison_study import prepare_protocol

    output = tmp_path / "invalid_width"
    with pytest.raises(ValueError, match="width must be a positive integer"):
        prepare_protocol(tmp_path, output, width=width)
    assert not output.exists()


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
