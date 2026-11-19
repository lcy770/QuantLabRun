"""候选生成、冻结集合训练与交替反馈的可恢复完整训练流程。"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from .ast import Grammar, Node
from .io import append_json, check_budget, clean, event, save_json, save_torch
from .models import model_metadata
from .ppo import complete_returns, gae_targets, update
from .selection_config import ARCHITECTURE
from .selection_evaluation import SelectionEvaluator, load_selection_task
from .selection_models import GeneratorPolicy, SelectorPolicy
from .selection_search import CandidatePool, ProposalCollector, StartBank, select_members, build_candidate_pool


def selection_bundle(generator, selector, config, **extra):
    """保存架构、完整两层参数、文法和奖励身份，禁止混用旧调度检查点。"""
    return {"format_version": 10, "architecture": ARCHITECTURE, "grammar": generator.grammar.config(),
            "width": generator.width, "noise_steps": generator.noise_steps,
            "lower": generator.state_dict(), "upper": selector.state_dict(),
            "model_metadata": model_metadata(generator, selector, generator.width),
            "configuration_hash": config["fingerprint"], "reward": config["reward"], **extra}


def restore_selection_models(bundle, device="cpu"):
    """按新架构检查点重建起点编辑生成器与集合选择器。"""
    if bundle.get("architecture") != ARCHITECTURE or bundle.get("format_version") != 10:
        raise ValueError("not a candidate_selection checkpoint")
    grammar = Grammar(**bundle["grammar"])
    generator = GeneratorPolicy(grammar, bundle["width"], bundle["noise_steps"]).to(device)
    selector = SelectorPolicy(grammar, bundle["width"]).to(device)
    generator.load_state_dict(bundle["lower"])
    selector.load_state_dict(bundle["upper"])
    return generator, selector


class SelectionCampaign:
    """两层各自奖励、批前冻结支持及显式阶段提交的训练所有者。"""

    def __init__(self, config, output, resume=False, progress_callback=None):
        """初始化模型、任务与来源档案，恢复时核对配置并最后恢复所有随机流。"""
        self.config, self.output = config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.callback, self.started = progress_callback, perf_counter()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config["torch_threads"])
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        self.rng = random.Random(config["seed"])
        self.grammar = Grammar(**config["grammar"])
        self.lower = GeneratorPolicy(self.grammar, config["width"], config["corpus"]["noise_steps"]).to(config["device"])
        self.upper = SelectorPolicy(self.grammar, config["width"]).to(config["device"])
        self.lower_optimizer = torch.optim.Adam(self.lower.parameters(), lr=config["lower"]["learning_rate"])
        self.upper_optimizer = torch.optim.Adam(self.upper.parameters(), lr=config["upper"]["learning_rate"])
        self.pretrain_optimizer = torch.optim.Adam(self.lower.editor.parameters(), lr=config["pretrain"]["learning_rate"])
        self.tasks = {task["name"]: task for task in config["tasks"]}
        self.training_tasks = [task for task in config["tasks"] if task["role"] == "train"]
        self.panels, self.evaluators = {}, {}
        for task in config["tasks"]:
            # 测试段与独立验收数据在决策完成后按需加载。
            feedback = task["role"] in {"train", "development"}
            panels = load_selection_task(task, config, with_feedback=feedback)
            self.panels[task["name"]] = panels
            self.evaluators[task["name"]] = SelectionEvaluator(self.grammar, panels, config)
        later = [self.panels[t["name"]]["search"].dates.min() for t in config["tasks"] if t["role"] != "train"]
        if later and any(self.panels[t["name"]]["feedback"].ends.max() >= min(later) for t in self.training_tasks):
            raise ValueError("training feedback labels reach development/test generation dates")
        self.initial = self._initial_formulas()
        self.banks = {name: StartBank(self.initial) for name in self.tasks}
        self.pools = {}
        self.state = {"phase": "pretrain", "pretrain_epoch": 0, "lower_iteration": 0,
                      "upper_iteration": 0, "joint_iteration": 0, "lower_updates": 0, "upper_updates": 0,
                      "reference_seconds": config["reward"]["reference_seconds"], "wall_seconds": 0.0,
                      "history": {"lower": [], "upper": [], "pretrain": []}, "evaluations": []}
        checkpoint = self.output / "training_state.pt"
        saved = None
        if checkpoint.exists():
            if not resume:
                raise FileExistsError("existing training state requires --resume or a new output")
            saved = torch.load(checkpoint, map_location=config["device"], weights_only=False)
            if saved.get("architecture") != ARCHITECTURE or saved["configuration_hash"] != config["fingerprint"]:
                raise ValueError("resume architecture/configuration/data/implementation identity differs")
            self.lower.load_state_dict(saved["lower"])
            self.upper.load_state_dict(saved["upper"])
            self.lower_optimizer.load_state_dict(saved["lower_optimizer"])
            self.upper_optimizer.load_state_dict(saved["upper_optimizer"])
            self.pretrain_optimizer.load_state_dict(saved["pretrain_optimizer"])
            self.state, self.banks, self.pools = saved["campaign_state"], saved["banks"], saved["pools"]
        elif resume:
            raise FileNotFoundError(checkpoint)
        elif config.get("initializer_checkpoint"):
            initializer = torch.load(config["initializer_checkpoint"], map_location=config["device"], weights_only=False)
            if initializer["grammar"] != self.grammar.config() or initializer["noise_steps"] != self.lower.noise_steps:
                raise ValueError("initializer grammar/diffusion horizon differs")
            if initializer.get("architecture") == ARCHITECTURE:
                self.lower.load_state_dict(initializer["lower"])
                self.upper.load_state_dict(initializer["upper"])
                if self.state["reference_seconds"] is None:
                    self.state["reference_seconds"] = initializer.get("reference_seconds")
            else:
                if initializer.get("lower_representation") != "rpn":
                    raise ValueError("initializer must use the RPN representation")
                self.lower.editor.load_state_dict(initializer["lower"])
            self.state["initializer_checkpoint"] = config["initializer_checkpoint"]
        self.elapsed_before = self.state["wall_seconds"]
        self.collector = ProposalCollector(config)
        save_json(self.output / "resolved_config.json", config)
        if saved:
            self.rng.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if saved["cuda_rng"] and torch.cuda.is_available():
                # map_location可能搬动随机状态，CUDA恢复接口要求CPU字节张量。
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])

    def _initial_formulas(self):
        """读取可追溯初始库或从固定文法采样合法唯一公式。"""
        if self.config.get("initial_library"):
            path = Path(self.config["initial_library"])
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not records or any(not record.get("validated") for record in records):
                raise ValueError("initial library contains unvalidated records")
            manifest = json.loads((path.parent / "library_manifest.json").read_text(encoding="utf-8"))
            boundary = min(self.panels[t["name"]]["search"].dates.min() for t in self.training_tasks)
            if np.datetime64(manifest["latest_screening_label"]) >= boundary:
                raise ValueError("initial library screening labels reach training generation dates")
            result = [self.grammar.validate(Node.from_dict(record["ast"])) for record in records]
            if len({tree.id for tree in result}) != len(result):
                raise ValueError("initial library contains canonical duplicates")
            return result
        rng = random.Random(self.config["seed"])
        result, identities = [], set()
        while len(result) < self.config["initial_seeds"]:
            tree = self.grammar.sample(rng, identities)
            result.append(tree)
            identities.add(tree.id)
        return result

    def elapsed(self):
        """返回已提交历史耗时加本次进程实际墙钟耗时。"""
        return self.elapsed_before + perf_counter() - self.started

    def commit(self):
        """原子提交模型、优化器、池与随机流，并由已提交历史重写无重复曲线。"""
        self.state["wall_seconds"] = self.elapsed()
        save_torch(self.output / "training_state.pt", selection_bundle(self.lower, self.upper, self.config,
            campaign_state=copy.deepcopy(self.state), banks=self.banks, pools=self.pools,
            lower_optimizer=self.lower_optimizer.state_dict(), upper_optimizer=self.upper_optimizer.state_dict(),
            pretrain_optimizer=self.pretrain_optimizer.state_dict(), python_rng=self.rng.getstate(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        for name, history in self.state["history"].items():
            path = self.output / f"{name}_curve.jsonl"
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(clean(row), ensure_ascii=False) + "\n" for row in history), encoding="utf-8")
            temporary.replace(path)
        progress = {key: value for key, value in self.state.items() if key not in {"history", "evaluations"}}
        progress.update(architecture=ARCHITECTURE, completed_trials=len(self.state["evaluations"]))
        save_json(self.output / "progress.json", progress)
        if self.callback:
            self.callback({"stage": self.state["phase"], **progress})
        event("selection_progress", **progress)

    def _pretrain(self):
        """使用合法原生逆编辑预训练编辑器，按 epoch 提交并恢复优化器。"""
        settings = self.config["pretrain"]
        if settings["epochs"]:
            from .training import create_corpus, decode_pair, syntax_loss
            records = create_corpus(self.grammar, self.config["corpus"], self.output, "rpn")
            train = [pair for record in records if record["split"] == "train" for pair in record["pairs"]]
            check = [record for record in records if record["split"] == "check"]
            while self.state["pretrain_epoch"] < settings["epochs"]:
                check_budget()
                losses = []
                ordered = self.rng.sample(train, len(train))
                for start in range(0, len(ordered), settings["batch_size"]):
                    loss = torch.stack([-self.lower.editor.evaluate_action(*decode_pair(pair))[0]
                                        for pair in ordered[start:start + settings["batch_size"]]]).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite inverse editing loss")
                    self.pretrain_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.lower.editor.parameters(), 1.0, error_if_nonfinite=True)
                    self.pretrain_optimizer.step()
                    losses.append(float(loss.detach()))
                validation = syntax_loss(self.lower.editor, check)
                self.state["pretrain_epoch"] += 1
                self.state["history"]["pretrain"].append({"epoch": self.state["pretrain_epoch"],
                    "train_nll": float(np.mean(losses)), "check_nll": validation})
                if validation < self.state.get("best_nll", float("inf")):
                    self.state["best_nll"] = validation
                    save_torch(self.output / "syntax_best.pt", selection_bundle(self.lower, self.upper, self.config))
                self.commit()
            best = torch.load(self.output / "syntax_best.pt", map_location=self.config["device"], weights_only=False)
            self.lower.load_state_dict(best["lower"])
        self.state["phase"] = "calibration"
        self.commit()

    def _calibrate(self):
        """在训练任务上测量基准主候选中位耗时，固定后用于全部后续质量成本奖励。"""
        if self.state["reference_seconds"] is None:
            task = self.training_tasks[0]
            evaluator = self.evaluators[task["name"]]
            started = perf_counter()
            observation = self.banks[task["name"]].observe(evaluator, self.config["lower"]["shortlist_size"])
            count = self.config["reward"]["calibration_samples"]
            preparation = (perf_counter() - started) / count
            seeds = [310000 + self.config["seed"] * count + i for i in range(count)]
            proposals = self.collector.collect(self.lower, evaluator, task, observation, seeds,
                                               [f"calibration_{i}" for i in range(count)])
            samples = [proposal["seconds"] + preparation for proposal in proposals]
            self.state["reference_seconds"] = float(np.median(samples))
            save_json(self.output / "cost_calibration.json", {"samples_seconds": samples,
                "reference_seconds": self.state["reference_seconds"], "task": task["name"],
                "statistic": "median", "parameters_updated": False, "pool_updated": False})
        self.state["phase"] = "lower"
        self.commit()

    def generate_pool(self, task, count, tag, *, record=False, background=(), wall_seconds=None):
        """通过共享采集协议逐批生成、评价和提交候选。"""
        return build_candidate_pool(self.lower, self.evaluators[task["name"]], self.banks[task["name"]],
            self.config, self.rng, self.state["reference_seconds"], task, count, tag, self.output,
            collector=self.collector, record=record, background=background, wall_seconds=wall_seconds,
            generator_version=self.state["lower_updates"])

    def _lower_update(self, proposals, task, tag, contributions=None):
        """同一终点奖励训练起点及所有编辑，D 只归因到本批真实生成来源。"""
        contributions = contributions or {}
        trajectories = []
        rewards = []
        feedback = []
        for proposal in proposals:
            contribution = contributions.get(proposal["proposal_id"], {}).get("value", 0.0)
            reward = proposal["base_reward"] + self.config["reward"]["eta"] * contribution
            rows = proposal["rows"]
            if not rows:
                raise AssertionError("lower PPO requires a recorded start decision")
            for row in rows:
                row["reward"] = 0.0
            rows[-1]["reward"] = reward
            complete_returns(rows)
            gae_targets(rows, self.config["lower"]["gae_lambda"])
            trajectories.extend(rows)
            rewards.append(reward)
            feedback.append({"proposal_id": proposal["proposal_id"], "base_reward": proposal["base_reward"],
                "contribution": contribution, "eta": self.config["reward"]["eta"], "terminal_reward": reward})
        if self.config["record_rollouts"]:
            save_torch(self.output / "rollouts" / f"{tag}.pt", {"models": selection_bundle(self.lower, self.upper, self.config),
                "rows": trajectories, "feedback": feedback, "task": task["name"],
                "optimizer": self.lower_optimizer.state_dict(), "shuffle_rng": self.rng.getstate(),
                "generator_version": self.state["lower_updates"], "selector_version": self.state["upper_updates"]})
        metrics = update(self.lower, self.lower_optimizer, trajectories, rng=self.rng, **self.config["lower"]["ppo"])
        self.state["lower_updates"] += 1
        self.state["history"]["lower"].append({"tag": tag, "task": task["name"], **metrics,
            "mean_terminal_reward": float(np.mean(rewards)), "contribution_feedback": feedback,
            "generator_version": self.state["lower_updates"], "wall_seconds": self.elapsed()})

    def _upper_update(self, task=None, tag="upper"):
        """在冻结候选池上采集当前上层轨迹后更新，只将 A 段输入网络。"""
        rows, reports = [], []
        for episode in range(self.config["upper"]["episodes_per_batch"]):
            current = task or self.rng.choice(self.training_tasks)
            candidates = self.pools[current["name"]]["candidates"]
            evaluator = self.evaluators[current["name"]]
            observation = self.pools[current["name"]]["observation"]
            trajectory, indices, score, contributions = select_members(self.upper, observation, evaluator, candidates,
                rng=self.rng, trace_lambda=self.config["upper"]["gae_lambda"])
            rows.extend(trajectory)
            reports.append({"task": current["name"], "pool_id": observation.pool_id,
                "generator_version": self.pools[current["name"]]["generator_version"],
                "selected": indices, "score": score, "contributions": contributions})
        if self.config["record_rollouts"]:
            save_torch(self.output / "rollouts" / f"{tag}.pt", {"models": selection_bundle(self.lower, self.upper, self.config),
                "rows": rows, "reports": reports, "selector_version": self.state["upper_updates"],
                "optimizer": self.upper_optimizer.state_dict(), "shuffle_rng": self.rng.getstate()})
        metrics = update(self.upper, self.upper_optimizer, rows, rng=self.rng, **self.config["upper"]["ppo"])
        self.state["upper_updates"] += 1
        self.state["history"]["upper"].append({"tag": tag, **metrics, "episodes": reports,
            "mean_episode_return": float(np.mean([report["score"]["utility"] for report in reports])),
            "selector_version": self.state["upper_updates"], "wall_seconds": self.elapsed()})

    def _freeze_pool(self, task, tag):
        """用当前下层重新生成训练候选并冻结来源版本，随后上层可重复采样组合。"""
        pool, _, report = self.generate_pool(task, self.config["upper"]["pool_attempts"], tag)
        self.pools[task["name"]] = {"candidates": pool.candidates,
            "generator_version": self.state["lower_updates"], "generation_report": report,
            "observation": self.evaluators[task["name"]].observation(pool.candidates, self.config["upper"]["max_members"])}
        save_torch(self.output / "candidate_pools" / f"{tag}.pt", self.pools[task["name"]])

    def _snapshot(self, filename):
        """保存可部署的两层快照及实际固定成本尺度。"""
        save_torch(self.output / filename, selection_bundle(self.lower, self.upper, self.config,
            reference_seconds=self.state["reference_seconds"], generator_version=self.state["lower_updates"],
            selector_version=self.state["upper_updates"]))

    def run(self):
        """执行全部已授权阶段，任何退出均释放采集进程。"""
        try:
            return self._run()
        finally:
            self.collector.close()

    def _run(self):
        """按已提交阶段推进预训练、下层、冻结上层、交替训练和最终评价。"""
        if self.state["phase"] == "complete":
            return json.loads((self.output / "experiment_summary.json").read_text(encoding="utf-8"))
        if self.state["phase"] == "pretrain":
            self._pretrain()
        if self.state["phase"] == "calibration":
            self._calibrate()
        if self.state["phase"] == "lower":
            while self.state["lower_iteration"] < self.config["lower"]["iterations"]:
                check_budget()
                task = self.rng.choice(self.training_tasks)
                tag = f"lower_{self.state['lower_iteration']}"
                _, proposals, _ = self.generate_pool(task, self.config["lower"]["batch_formulas"], tag, record=True)
                self._lower_update(proposals, task, tag)
                self.state["lower_iteration"] += 1
                self.commit()
            self._snapshot("lower_base.pt")
            self.state["phase"] = "freeze_pools"
            self.commit()
        if self.state["phase"] == "freeze_pools":
            for task in self.training_tasks:
                if task["name"] not in self.pools:
                    self._freeze_pool(task, f"frozen_{task['name']}")
                    self.commit()
            self.state["phase"] = "upper"
            self.commit()
        if self.state["phase"] == "upper":
            while self.state["upper_iteration"] < self.config["upper"]["iterations"]:
                check_budget()
                self._upper_update(tag=f"upper_{self.state['upper_iteration']}")
                self.state["upper_iteration"] += 1
                self.commit()
            self._snapshot("upper_frozen_final.pt")
            self.state["phase"] = "joint"
            self.commit()
        if self.state["phase"] == "joint":
            while self.state["joint_iteration"] < self.config["upper"]["joint_iterations"]:
                check_budget()
                task = self.rng.choice(self.training_tasks)
                tag = f"joint_{self.state['joint_iteration']}"
                background = self.pools[task["name"]]["candidates"]
                pool, proposals, _ = self.generate_pool(task, self.config["lower"]["batch_formulas"],
                                                        f"{tag}_lower", record=True, background=background)
                evaluator = self.evaluators[task["name"]]
                observation = evaluator.observation(pool.candidates, self.config["upper"]["max_members"])
                _, selected, score, contributions = select_members(self.upper, observation, evaluator, pool.candidates, rng=self.rng)
                save_json(self.output / "contributions" / f"{tag}.json", {"pool_id": observation.pool_id,
                    "selected": selected, "score": score, "contributions": contributions,
                    "generator_version": self.state["lower_updates"], "selector_version": self.state["upper_updates"]})
                self._lower_update(proposals, task, f"{tag}_lower", contributions)
                # 下层参数更新后重新生成候选、重新采集上层轨迹，概率均对应当前策略。
                self._freeze_pool(task, f"{tag}_fresh_upper_{task['name']}")
                self._upper_update(task, f"{tag}_upper")
                self.state["joint_iteration"] += 1
                self.commit()
            self._snapshot("joint_final.pt")
            self.state["training_wall_seconds"] = self.elapsed()
            self.state["phase"] = "evaluation"
            self.commit()
        return self.evaluate()

    def evaluate(self):
        """同一冻结候选池比较选择器，封存选择后才载入独立验收数据。"""
        from .experiments import aggregate
        tasks = [task for task in self.tasks.values() if task["role"] != "train"] or self.training_tasks
        evaluation = self.config["evaluation"]
        for task in tasks:
            for seed in evaluation["seeds"]:
                key = f"{task['name']}_{seed}"
                if key in self.state["evaluations"]:
                    continue
                # 初始化和训练统计不能跨评价种子污染起点策略。
                self.banks[task["name"]] = StartBank(self.initial)
                self.rng = random.Random(seed)
                seconds = self.config["search"]["seconds"] if self.config["search"]["budget_mode"] == "wall_seconds" else None
                pool, proposals, generation = self.generate_pool(task, self.config["search"]["max_attempts"],
                                                         f"evaluation_{key}", wall_seconds=seconds)
                candidates = pool.candidates
                evaluator = self.evaluators[task["name"]]
                preparation_started = perf_counter()
                observation = evaluator.observation(candidates, self.config["upper"]["max_members"])
                preparation_seconds = perf_counter() - preparation_started
                # 所有方法首先完成选择，之后统一加载封存的 future，避免影响后续方法。
                choices = []
                for method in evaluation["methods"]:
                    selection_started = perf_counter()
                    combination_before = evaluator.combination_seconds
                    device = next(self.upper.parameters()).device
                    devices = [device.index] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        torch.random.default_generator.manual_seed(seed)
                        if devices:
                            with torch.cuda.device(devices[0]):
                                torch.cuda.manual_seed(seed)
                        segment = "feedback" if "feedback" in evaluator.panels else "validation"
                        rows, indices, score, contributions = select_members(self.upper, observation, evaluator, candidates,
                            method=method, rng=random.Random(seed), deterministic=evaluation["deterministic"], segment=segment)
                    choices.append((method, rows, indices, score, contributions,
                                    perf_counter() - selection_started, evaluator.combination_seconds - combination_before))
                if "future" in task:
                    panels = load_selection_task(task, self.config, with_feedback="feedback" in evaluator.panels, with_future=True)
                    evaluator.panels["future"] = panels["future"]
                for method, rows, indices, score, contributions, selection_seconds, combination_seconds in choices:
                    report = {"task": task["name"], "role": task["role"], "seed": seed, "method": method,
                        **generation, "pool_id": observation.pool_id, "return": score["utility"],
                        "pool_ic": score["ic"], "initial_pool_ic": 0.0, "score": score,
                        "generation_seconds": generation["wall_seconds"], "validation_preparation_seconds": preparation_seconds,
                        "wall_seconds": generation["wall_seconds"] + preparation_seconds + selection_seconds,
                        "selection_seconds": selection_seconds, "combination_seconds": combination_seconds,
                        "scoring_segment": "feedback" if "feedback" in evaluator.panels else "validation",
                        "selected_indices": indices, "member_count": len(indices),
                        "pool_ids": [candidates[i].tree.id for i in indices],
                        "pool_weights": [1.0 / len(indices) for _ in indices],
                        "pool_directions": [candidates[i].direction for i in indices],
                        "pool_asts": [candidates[i].tree.to_dict() for i in indices],
                        "contributions": contributions,
                        "actions": [{"index": row["index"], "selected_before": row["observation"].selected,
                                     "old_logp": row.get("logp"), "reward": row["reward"]} for row in rows],
                        "discoveries": [{"expression": c.tree.text(), "direction": c.direction,
                            "ast": c.tree.to_dict(), "search": c.search_metrics, "quality": c.quality,
                            "proposal_id": c.proposal_id} for c in pool.discoveries],
                        "new_per_budget_second": generation["new_factors"] / max(seconds or generation["wall_seconds"], 1e-12),
                        "new_per_actual_generation_second": generation["new_factors"] / max(generation["wall_seconds"], 1e-12)}
                    if "future" in task:
                        future_score, future_contributions = evaluator.contributions(candidates, indices, "future")
                        report["future"] = {**future_score, "contributions": future_contributions,
                            "weights_refitted": False, "directions_refitted": False}
                    save_json(self.output / "trials" / f"{key}_{method}.json", report)
                    if method == "upper_ppo":
                        self._export(task, evaluator, candidates, indices, report, key, proposals)
                self.state["evaluations"].append(key)
                self.commit()
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((self.output / "trials").glob("*.json"))]
        summary = {"status": "complete", "architecture": ARCHITECTURE, "configuration_hash": self.config["fingerprint"],
            "training_seed": self.config["seed"], "aggregate": aggregate(reports), "trial_count": len(reports),
            "training_wall_seconds": self.state["training_wall_seconds"], "end_to_end_wall_seconds": self.elapsed(),
            "reference_seconds": self.state["reference_seconds"], "data_mode": self.config["data_mode"],
            "financial_performance_proven": False}
        save_json(self.output / "experiment_summary.json", summary)
        self.state["phase"] = "complete"
        self.commit()
        return summary

    def _export(self, task, evaluator, candidates, indices, report, key, proposals):
        """保存可部署等权成员、每日信号及平台单因子成果，重复运行按公式去重。"""
        from ..evaluator import SegmentMetrics, SingleFactorMetrics
        records = []
        for proposal in proposals:
            candidate = proposal["candidate"]
            if candidate is None:
                continue
            tree = candidate.tree
            fm = evaluator.directed_metrics(tree, "fit")
            sm = evaluator.directed_metrics(tree, "search")
            segments = {name: SegmentMetrics(m["ic"], m["rank_ic"], m["icir"], m["coverage"], m["days"])
                        for name, m in (("train", fm), ("valid", sm))}
            expression = tree.text() if candidate.direction >= 0 else f"mul(-1.0,{tree.text()})"
            accepted = proposal.get("accepted", False)
            reason = None if accepted else ("duplicate" if proposal.get("duplicate") else "search_quality")
            record = asdict(SingleFactorMetrics(expression, fm["ic"], fm["rank_ic"], fm["icir"], fm["coverage"],
                                                candidate.quality, accepted, reason, segments))
            record.update(self.config.get("storage_metadata", {}))
            record.update(source_expression=tree.text(), direction=candidate.direction, task=task["name"])
            records.append(record)
        self._platform_exports(records, key)
        members = [candidates[i] for i in indices]
        deployment = {"architecture": ARCHITECTURE, "task": task["name"], "pool_id": report["pool_id"],
            "members": [{"expression": c.tree.text(), "direction": c.direction, "weight": 1 / len(members),
                         "ast": c.tree.to_dict()} for c in members],
            "information_cutoff": str(evaluator.panels["validation"].ends.max()),
            "normalization": "daily_cross_section_population_std_missing_zero", "score": report["score"]}
        save_json(self.output / "portfolios" / f"{key}.json", deployment)
        panel = evaluator.panels["validation"]
        signal = sum((evaluator.segment(c.tree, "validation")[1] * c.direction for c in members),
                     np.zeros_like(panel.target)) / max(len(members), 1)
        path = self.output / "portfolios" / f"{key}_signals.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, scores=signal, dates=panel.dates,
                                stock_ids=np.asarray(panel.panel.stock_ids))
        temporary.replace(path)

    def _platform_exports(self, records, key):
        """由原子提交的逐次成果重建平台文件，失败重试不会重复追加或丢失已接纳公式。"""
        root = self.output / "platform_exports"
        save_json(root / f"{key}.json", records)
        merged = {}
        for path in sorted(root.glob("*.json")):
            for record in json.loads(path.read_text(encoding="utf-8")):
                previous = merged.get(record["expression"])
                if previous is None or record["accepted"]:
                    merged[record["expression"]] = record
        for filename, accepted in (("accepted_factors.jsonl", True), ("rejected_candidates.jsonl", False)):
            destination = self.output / filename
            temporary = destination.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged.values()
                                         if record["accepted"] == accepted), encoding="utf-8")
            temporary.replace(destination)
        save_json(self.output / "expression_index.json", {name: int(record["accepted"]) for name, record in merged.items()})
