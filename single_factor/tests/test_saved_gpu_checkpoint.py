"""仅在显式指定的公司训练检查点副本上复核下一批，原文件不覆盖。"""
import json
import os
from pathlib import Path

import pytest
import torch

from single_factor.hierarchical.comparison_resume import _prepare_extension
from single_factor.hierarchical.comparison_training import NodeRunner, Study
from single_factor.hierarchical.io import run_lock
from single_factor.hierarchical.selection_config import file_hash


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA设备")
def test_saved_generator_next_batch_on_actual_gpu(tmp_path, monkeypatch):
    """恢复实际失败模型，按原批量与PPO配置验证下一批更新，保留父轮权重。"""
    protocol_value = os.environ.get("QUANTLAB_AUDIT_PROTOCOL")
    results_value = os.environ.get("QUANTLAB_AUDIT_RESULTS")
    if not protocol_value or not results_value:
        pytest.skip("需要显式设置审计协议及结果路径")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    protocol, results = Path(protocol_value).resolve(), Path(results_value).resolve()
    checkpoint = results / "seed_0/nodes/g_diffusion/checkpoint.pt"
    request = {"protocol": str(protocol), "source": str(results), "node": "g_diffusion",
               "seed": 0, "additional_hours": 1e-9}
    original_hash = file_hash(checkpoint)
    output = tmp_path / "actual_checkpoint_audit"
    with run_lock(results):
        _prepare_extension(protocol, results, output, request)
    child_protocol = output / "protocol/protocol.json"
    study = Study(child_protocol, output / "results")
    assert study.config["device"].startswith("cuda")
    assert study.config["width"] == 256
    assert study.config["lower"]["batch_formulas"] == 16
    runner = NodeRunner(study, "g_diffusion", 0)
    assert next(runner.generator.policy.parameters()).is_cuda
    iteration = runner.state["lower_iteration"]
    runner.lower_step(runner.task_for("lower", iteration), f"lower_{iteration}")
    metrics = runner.state["history"][-1]
    assert metrics["optimizer_steps"] > 0
    assert metrics["old_logp_max_error"] <= 2e-5
    assert file_hash(checkpoint) == original_hash
    # 明确输出实际恢复轮数、批量数和两种误差，便于对照公司机器的失败现场。
    print(json.dumps({"source_lower_iteration": iteration, "width": study.config["width"],
        "batch_formulas": study.config["lower"]["batch_formulas"],
        "ppo_epochs": study.config["lower"]["ppo"]["epochs"],
        "old_logp_max_error": metrics["old_logp_max_error"],
        "old_logp_batch_max_error": metrics["old_logp_batch_max_error"],
        "old_logp_scalar_checks": metrics["old_logp_scalar_checks"],
        "optimizer_steps": metrics["optimizer_steps"], "source_checkpoint_unchanged": True}, indent=2))
