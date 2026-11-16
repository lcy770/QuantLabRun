"""实验节点训练、同池评价、旧双 PPO 对照与可恢复执行。"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from .ast import Grammar, Node
from .comparison_generators import AutoregressivePPO, DiffusionBackend, RandomEditor, create_generator
from .comparison_metrics import locked_ridge_weights, prediction_for, top50_metrics, prediction_rank_ic
from .comparison_selectors import create_selector
from .io import append_json, check_budget, digest, run_lock, save_json, save_torch
from .selection_evaluation import SelectionEvaluator, load_selection_task
from .selection_search import CandidatePool, StartBank, build_candidate_pool, select_members


def seed_number(*parts):
    """为阶段、任务和重复采样生成跨进程稳定种子，排除实验臂名称。"""
    return int(digest(parts)[:8], 16) % (2**31)


def seed_all(seed):
    """统一初始化实际 CPU/CUDA 随机流。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def policy_hash(state):
    """对实际张量内容求身份，支持核对配对模型的初始参数。"""
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            result.update(name.encode())
            result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


class Study:
    """管理协议、共享预训练、固定候选池、节点来源与封存验收。"""

    def __init__(self, protocol_path, output):
        """载入已核验协议并检查当前代码、数据与协议身份一致。"""
        from .selection_config import load_selection_config
        self.protocol_path = Path(protocol_path).resolve()
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        if digest({key: value for key, value in self.protocol.items() if key != "fingerprint"}) != self.protocol["fingerprint"]:
            raise ValueError("protocol contents differ from its frozen fingerprint")
        current = load_selection_config(self.protocol_path.with_name("config.json"))
        if current["fingerprint"] != self.protocol["config"]["fingerprint"]:
            raise ValueError("protocol code/data/config identity changed; prepare a new protocol")
        self.config, self.profile = current, self.protocol["settings"]
        torch.set_num_threads(current["torch_threads"])
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.definitions = self.protocol["nodes"]
        self.grammar = Grammar(**current["grammar"])
        self.initial = [self.grammar.validate(Node.from_dict(json.loads(line)["ast"])) for line in
                        Path(current["initial_library"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.training_tasks = [task for task in current["tasks"] if task["role"] == "train"]
        self.comparison_split = ("training" if self.protocol["dates"].get("comparison_uses_training_data")
                                 else "development")
        self.evaluators = {}

    def evaluation_tasks(self, split):
        """按显式报告用途选择同一训练任务或历史开发、测试任务。"""
        roles = {"training": "train", "development": "development", "test": "test"}
        if split not in roles:
            raise ValueError("unknown evaluation split")
        tasks = [task for task in self.config["tasks"] if task["role"] == roles[split]]
        if not tasks:
            raise ValueError(f"protocol has no {split} tasks")
        return tasks

    def evaluator(self, task, reward=None, future=False, scope="evaluation"):
        """按数据角色延迟加载，独立验收数据仅由显式验收调用打开。"""
        reward = reward or self.config["reward"]
        key = (task["name"], digest(reward), future, scope)
        if key not in self.evaluators:
            panels = load_selection_task(task, self.config, with_feedback=task["role"] != "test", with_future=future)
            config = {**self.config, "reward": reward}
            self.evaluators[key] = SelectionEvaluator(self.grammar, panels, config)
        return self.evaluators[key]

    def node_path(self, name, seed):
        """返回固定的配对种子与实验节点路径。"""
        return self.output / f"seed_{seed}" / "nodes" / name

    def canonical(self, name):
        """解析共享模型别名，避免重复训练同一实验。"""
        while "alias" in self.definitions[name]:
            name = self.definitions[name]["alias"]
        return name

    def saved_node(self, name, seed):
        """只接受已经提交完成的来源节点。"""
        path = self.node_path(self.canonical(name), seed) / "checkpoint.pt"
        saved = torch.load(path, map_location=self.config["device"], weights_only=False)
        if saved["state"]["phase"] != "complete":
            raise ValueError(f"source node {name} has not completed")
        return saved

    def reference_seconds(self, seed):
        """同一配对种子共用一次训练前基准中位耗时。"""
        path = self.output / f"seed_{seed}" / "calibration.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["reference_seconds"]
        started = perf_counter()
        evaluator = self.evaluator(self.training_tasks[0], scope=f"calibration_{seed}")
        generator = RandomEditor(self.grammar, self.config)
        bank = StartBank(self.initial)
        observation = bank.observe(evaluator, self.config["lower"]["shortlist_size"])
        from .selection_search import collect_proposal
        count = self.config["reward"]["calibration_samples"]
        preparation = (perf_counter() - started) / count
        samples = [collect_proposal(generator, evaluator, observation, self.config["lower"],
                   seed_number(seed, "calibration", i), f"calibration_{i}", False)["seconds"] + preparation
                   for i in range(count)]
        value = float(np.median(samples))
        save_json(path, {"reference_seconds": value, "samples": samples, "preparation_seconds_per_candidate": preparation,
                         "wall_seconds": perf_counter() - started, "method": "common_random_legal_rpn_editor"})
        return value

    def shared_pool(self, source, seed, task, budget, sampling_seed):
        """同一生成器检查点、数据、种子和预算只生成一次候选池，供所有上层复用。"""
        source = self.canonical(source)
        root = self.output / f"seed_{seed}" / "shared_pools" / source / task["name"] / f"{budget}_{sampling_seed}"
        path = root / "pool.pt"
        saved = self.saved_node(source, seed)
        identity = digest([saved["identity"], task, budget, sampling_seed])
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if cached["identity"] != identity:
                raise ValueError("shared candidate-pool identity changed")
            return cached
        root.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        config["seed"] = seed
        seed_all(seed_number(seed, "pool_model"))
        generator = create_generator(self.definitions[source]["generator"], self.grammar, config)
        generator.restore(saved["generator"])
        evaluator = self.evaluator(task, scope=f"pool_{source}_{seed}_{budget}_{sampling_seed}")
        bank = StartBank(self.initial)
        attempts = self.profile["attempt_budget"] if budget == "attempts" else 10000000
        seconds = None if budget == "attempts" else self.profile["time_budget"]
        pool, _, report = build_candidate_pool(generator, evaluator, bank, config,
            random.Random(seed_number(seed, task["name"], budget, sampling_seed)), self.reference_seconds(seed),
            task, attempts, f"evaluation_{task['name']}_{budget}_{sampling_seed}", root, wall_seconds=seconds)
        preparation = perf_counter()
        observation = evaluator.observation(pool.candidates, config["upper"]["max_members"])
        preparation = perf_counter() - preparation
        cached = {"identity": identity, "candidates": pool.candidates, "discoveries": pool.discoveries,
                  "report": report, "generator_node": source, "generator_identity": saved["identity"],
                  "observation": observation, "validation_preparation_seconds": preparation}
        save_torch(path, cached)
        return cached

    def run_node(self, name, seed):
        """在目录锁保护下执行单个节点，失败保留完整错误并可从批末恢复。"""
        definition = self.definitions[name]
        path = self.node_path(name, seed)
        if "alias" in definition:
            save_json(path / "alias.json", {"source": self.canonical(name), "shared_model": True})
            return
        if definition.get("legacy"):
            from .comparison_legacy import run_legacy
            return run_legacy(self, name, seed)
        with run_lock(path):
            try:
                runner = NodeRunner(self, name, seed)
                runner.run()
            except BaseException as error:
                import traceback
                save_json(path / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                   "traceback": traceback.format_exc()})
                raise

    def evaluate_node(self, name, seed, split=None, selection_identity=None):
        """按协议在训练R段或历史保留段比较，保存选股及配权结果。"""
        split = split or self.comparison_split
        name = self.canonical(name)
        definition = self.definitions[name]
        if definition.get("legacy"):
            from .comparison_legacy import evaluate_legacy
            return evaluate_legacy(self, name, seed, split, selection_identity)
        if definition.get("syntax"):
            return
        saved = self.saved_node(name, seed)
        config = copy.deepcopy(self.config)
        config["seed"] = seed
        selector_method = definition.get("selector", "forward_exchange")
        seed_all(seed_number(seed, "selector_initial"))
        selector = create_selector(selector_method, self.grammar, config)
        if selector is not None:
            selector.restore(saved["selector"])
        source = name if definition.get("lower") or definition.get("continuation") or not definition.get("generator_source") else definition["generator_source"]
        if definition.get("generator_source") == "syntax":
            source = name
        tasks = self.evaluation_tasks(split)
        for task in tasks:
            for sampling_seed in self.profile["evaluation_seeds"]:
                for budget in ("attempts", "wall_seconds"):
                    key = f"{task['name']}_{budget}_{sampling_seed}"
                    output = self.node_path(name, seed) / split / f"{key}.json"
                    if output.exists():
                        prior = json.loads(output.read_text(encoding="utf-8"))
                        if prior["model_identity"] != saved["identity"] or prior.get("selection_identity") != selection_identity:
                            raise ValueError("existing evaluation identity differs")
                        continue
                    cached = self.shared_pool(source, seed, task, budget, sampling_seed)
                    candidates = cached["candidates"]
                    evaluator = self.evaluator(task, scope=f"selection_{name}_{seed}_{budget}_{sampling_seed}")
                    observation = cached["observation"]
                    preparation = cached["validation_preparation_seconds"]
                    seed_all(seed_number(seed, task["name"], budget, sampling_seed, "selection"))
                    started = perf_counter()
                    _, selected, _, _ = select_members(selector, observation, evaluator, candidates,
                        segment="validation", method=selector_method, rng=random.Random(sampling_seed), deterministic=True)
                    selection_seconds = perf_counter() - started
                    # 此处开始使用独立验收标签，候选和成员已全部锁定。
                    final = self.evaluator(task, future=True, scope=f"final_{name}_{seed}_{budget}_{sampling_seed}") if split == "test" else evaluator
                    segment = "future" if split == "test" else "feedback"
                    score, contributions = final.contributions(candidates, selected, segment)
                    prediction = prediction_for(final, candidates, selected, segment)
                    rank_ic = prediction_rank_ic(prediction, final.panels[segment].target) if selected else 0.0
                    portfolio = top50_metrics(final, candidates, selected, segment, config["portfolio"])
                    ridge = locked_ridge_weights(final, candidates, selected)
                    ridge_score = final.utility(candidates, selected, segment, ridge)
                    ridge_portfolio = top50_metrics(final, candidates, selected, segment, config["portfolio"], ridge)
                    initial_ids = {tree.id for tree in self.initial}
                    generated = cached["discoveries"]
                    report = {"node": name, "group": definition["group"], "training_seed": seed,
                        "sampling_seed": sampling_seed, "task": task["name"], "budget": budget,
                        "evaluation_scope": split, "evaluation_uses_training_data": split == "training",
                        "model_identity": saved["identity"], "selection_identity": selection_identity,
                        "candidate_pool_identity": cached["identity"], "pool_id": observation.pool_id,
                        "generator_node": source, "member_count": len(selected), "selected_indices": selected,
                        "selected_factors": [{"expression": candidates[i].tree.text(), "ast": candidates[i].tree.to_dict(),
                                              "direction": candidates[i].direction} for i in selected],
                        "utility": score["utility"], "ic": score["ic"], "rank_ic": rank_ic,
                        "new_factors": len(generated), "new_to_initial_library": sum(c.tree.id not in initial_ids for c in generated),
                        "search_seconds": cached["report"]["wall_seconds"] + preparation + selection_seconds,
                        "generation": cached["report"], "validation_preparation_seconds": preparation,
                        "selection_seconds": selection_seconds, "score": score, "contributions": contributions,
                        "net_return": portfolio["net_return"], "net_excess_return": portfolio["net_excess_return"],
                        "maximum_drawdown": portfolio["maximum_drawdown"], "portfolio": portfolio,
                        "weight_comparison": {"same_members": True, "equal_weights": [1 / len(selected) for _ in selected],
                            "ridge_weights": ridge, "fit_segment": "fit", "refitted_on_evaluation": False,
                            "ridge_score": ridge_score, "ridge_portfolio": ridge_portfolio},
                        "costs": saved["state"]["costs"], "costs_scope": "incremental_node",
                        "cost_dependencies": sorted({self.canonical(definition[key]) for key in
                            ("generator_source", "selector_source") if key in definition}),
                        "common_evaluation_reward": self.config["reward"]}
                    save_json(output, report)


class NodeRunner:
    """在共同协议下执行一个生成器/选择器组合，实际学习器各自保存完整状态。"""

    def __init__(self, study, name, seed):
        """恢复批末检查点，或从指定配对来源创建模型。"""
        self.study, self.name, self.seed = study, name, seed
        self.spec = study.definitions[name]
        self.config = copy.deepcopy(study.config)
        self.config["seed"] = seed
        for weight in ("beta", "gamma"):
            if weight in self.spec:
                self.config["reward"][weight] = self.spec[weight]
        self.root = study.node_path(name, seed)
        self.root.mkdir(parents=True, exist_ok=True)
        seed_all(seed_number(seed, "generator_initial"))
        self.generator = create_generator(self.spec["generator"], study.grammar, self.config)
        seed_all(seed_number(seed, "selector_initial"))
        self.method = self.spec.get("selector", "forward_exchange")
        self.selector = create_selector(self.method, study.grammar, self.config)
        self.banks, self.pools = {}, {}
        dependencies = {}
        for key in ("generator_source", "selector_source"):
            if key in self.spec:
                dependencies[key] = study.saved_node(self.spec[key], seed)
        self.identity = digest([study.protocol["fingerprint"], name, seed,
                                {key: value["identity"] for key, value in dependencies.items()}])
        self.state = {"phase": "syntax", "syntax_epoch": 0, "lower_iteration": 0, "upper_iteration": 0,
            "continuation_iteration": 0, "generator_version": 0, "selector_version": 0,
            "costs": {"syntax": 0.0, "generator_training": 0.0, "pool_preparation": 0.0,
                      "selector_training": 0.0, "continuation": 0.0}, "history": []}
        checkpoint = self.root / "checkpoint.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=self.config["device"], weights_only=False)
            if saved["identity"] != self.identity:
                raise ValueError("node identity changed; use a new output")
            self.generator.restore(saved["generator"])
            if self.selector is not None:
                self.selector.restore(saved["selector"])
            self.state, self.banks, self.pools = saved["state"], saved["banks"], saved["pools"]
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if saved["cuda_rng"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        else:
            if "generator_source" in dependencies:
                source = dependencies["generator_source"]
                self.generator.restore(source["generator"])
                self.state["generator_version"] = source["state"]["generator_version"]
                if self.spec.get("shared_pools"):
                    self.pools = copy.deepcopy(source["pools"])
                    self.banks = copy.deepcopy(source["banks"])
            if "selector_source" in dependencies:
                self.selector.restore(dependencies["selector_source"]["selector"])
                self.state["selector_version"] = dependencies["selector_source"]["state"]["selector_version"]
            initial = self.generator.checkpoint()
            save_json(self.root / "initialization.json", {"generator_source": self.spec.get("generator_source"),
                "selector_source": self.spec.get("selector_source"), "paired_training_seed": seed,
                "generator_parameter_hash": policy_hash(initial.get("policy", {})),
                "selector_parameter_hash": policy_hash(self.selector.checkpoint().get("policy", {})) if self.selector else None})
        self.reference = study.reference_seconds(seed)

    def commit(self):
        """原子保存所有模型、优化器、种群、经验池、候选池和随机流。"""
        save_torch(self.root / "checkpoint.pt", {"identity": self.identity, "state": self.state,
            "generator": self.generator.checkpoint(), "selector": self.selector.checkpoint() if self.selector else None,
            "banks": self.banks, "pools": self.pools, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
        save_json(self.root / "progress.json", {**self.state, "history": None, "identity": self.identity})
        save_json(self.root / "training_history.json", self.state["history"])

    def evaluator(self, task):
        """训练采用本节点消融权重，F/G 准入门槛始终相同。"""
        return self.study.evaluator(task, self.config["reward"], scope=f"train_{self.name}_{self.seed}")

    def generate(self, task, count, tag, record=False, background=()):
        """调用与主算法完全相同的成本、预算和去重实现。"""
        self.banks.setdefault(task["name"], StartBank(self.study.initial))
        return build_candidate_pool(self.generator, self.evaluator(task), self.banks[task["name"]],
            self.config, random.Random(seed_number(self.seed, task["name"], tag)), self.reference,
            task, count, tag, self.root, record=record, background=background,
            generator_version=self.state["generator_version"])

    def lower_step(self, task, tag, feedback=False):
        """执行真实扩散 PPO、自回归 PPO 或一代 GP，分别登记实际更新类型。"""
        if isinstance(self.generator, AutoregressivePPO):
            metrics = self.generator.learn_task(self.evaluator(task), self.reference)
        else:
            background = self.pools.get(task["name"], {}).get("candidates", ()) if feedback else ()
            before = copy.deepcopy(self.generator.checkpoint()) if self.config["record_rollouts"] else None
            pool, proposals, report = self.generate(task, self.config["lower"]["batch_formulas"], tag,
                                                    record=isinstance(self.generator, DiffusionBackend), background=background)
            contributions = {}
            if feedback:
                observation = self.evaluator(task).observation(pool.candidates, self.config["upper"]["max_members"])
                _, _, _, contributions = select_members(self.selector, observation, self.evaluator(task), pool.candidates,
                                                         method=self.method, trace_lambda=self.config["upper"]["gae_lambda"])
            if isinstance(self.generator, DiffusionBackend):
                metrics = self.generator.learn(proposals, random.Random(seed_number(self.seed, tag, "update")), contributions)
            else:
                metrics = {"samples": len(proposals), "optimizer_steps": 0, "update_type": "genetic_generation"}
            metrics["generation"] = report
            if self.config["record_rollouts"] and self.state["generator_version"] % self.study.profile.get("rollout_every", 10) == 0:
                save_torch(self.root / "rollouts" / f"{tag}.pt", {"generator_before": before,
                    "proposals": proposals, "contributions": contributions, "task": task["name"]})
        self.state["generator_version"] += 1
        self.state["history"].append({"stage": "lower", "tag": tag, "task": task["name"], **metrics})

    def freeze_pool(self, task, tag):
        """生成器当前版本生成候选，缓存 A 段特征以复用同池学习与消融。"""
        started = perf_counter()
        pool, _, report = self.generate(task, self.config["upper"]["pool_attempts"], tag)
        self.pools[task["name"]] = {"candidates": pool.candidates, "report": report,
            "generator_version": self.state["generator_version"],
            "observation": self.evaluator(task).observation(pool.candidates, self.config["upper"]["max_members"])}
        self.state["costs"]["pool_preparation"] += perf_counter() - started

    def upper_step(self, task, tag):
        """同一冻结池采集完整组合回报，然后更新实际 PPO、LinUCB 或 Double DQN。"""
        frozen = self.pools[task["name"]]
        rollouts = []
        for episode in range(self.config["upper"]["episodes_per_batch"]):
            seed_all(seed_number(self.seed, tag, episode))
            rows, selected, score, contributions = select_members(self.selector, frozen["observation"], self.evaluator(task),
                frozen["candidates"], method=self.method, trace_lambda=self.config["upper"]["gae_lambda"])
            rollouts.append(rows)
        metrics = self.selector.learn(rollouts, random.Random(seed_number(self.seed, tag, "update")))
        self.state["selector_version"] += 1
        self.state["history"].append({"stage": "upper", "tag": tag, "task": task["name"],
            "pool_id": frozen["observation"].pool_id, "generator_version": frozen["generator_version"],
            "mean_return": float(np.mean([rows[0]["return"] for rows in rollouts])), **metrics})
        if self.config["record_rollouts"] and self.state["selector_version"] % self.study.profile.get("rollout_every", 10) == 0:
            save_torch(self.root / "rollouts" / f"{tag}.pt", {"rows": rollouts, "task": task["name"], "metrics": metrics})

    def task_for(self, phase, iteration):
        """以独立阶段种子固定任务顺序，防止动作长度改变配对任务抽样。"""
        return self.study.training_tasks[seed_number(self.seed, phase, iteration) % len(self.study.training_tasks)]

    def run(self):
        """推进所有未提交阶段，时间预算只在完整更新边界结束。"""
        if self.state["phase"] == "complete":
            return
        profile = self.study.profile
        if self.state["phase"] == "syntax":
            if self.spec.get("syntax"):
                from .training import create_corpus
                records = create_corpus(self.study.grammar, self.config["corpus"], self.root, "rpn")
                while self.state["syntax_epoch"] < profile["pretrain_epochs"]:
                    check_budget()
                    started = perf_counter()
                    metrics = self.generator.pretrain(records, random.Random(seed_number(self.seed, "syntax", self.state["syntax_epoch"])))
                    self.state["syntax_epoch"] += 1
                    self.state["costs"]["syntax"] += perf_counter() - started
                    self.state["history"].append({"stage": "syntax", **metrics})
                    self.commit()
            self.state["phase"] = "lower"
            self.commit()
        if self.state["phase"] == "lower":
            if self.spec.get("lower"):
                while self.state["lower_iteration"] < profile["lower_iterations"] and self.state["costs"]["generator_training"] < profile["lower_seconds"]:
                    check_budget()
                    started = perf_counter()
                    iteration = self.state["lower_iteration"]
                    self.lower_step(self.task_for("lower", iteration), f"lower_{iteration}")
                    self.state["lower_iteration"] += 1
                    self.state["costs"]["generator_training"] += perf_counter() - started
                    self.commit()
            self.state["phase"] = "freeze"
            self.commit()
        if self.state["phase"] == "freeze":
            if not self.spec.get("syntax"):
                for task in self.study.training_tasks:
                    if task["name"] not in self.pools:
                        self.freeze_pool(task, f"frozen_{task['name']}")
                        self.commit()
            self.state["phase"] = "upper"
            self.commit()
        if self.state["phase"] == "upper":
            if self.spec.get("upper"):
                while self.state["upper_iteration"] < profile["upper_iterations"] and self.state["costs"]["selector_training"] < profile["upper_seconds"]:
                    check_budget()
                    started = perf_counter()
                    iteration = self.state["upper_iteration"]
                    self.upper_step(self.task_for("upper", iteration), f"upper_{iteration}")
                    self.state["upper_iteration"] += 1
                    self.state["costs"]["selector_training"] += perf_counter() - started
                    self.commit()
            self.state["phase"] = "continuation"
            self.commit()
        if self.state["phase"] == "continuation":
            mode = self.spec.get("continuation")
            while mode and self.state["continuation_iteration"] < profile["continuation_iterations"] and self.state["costs"]["continuation"] < profile["continuation_seconds"]:
                check_budget()
                started = perf_counter()
                iteration = self.state["continuation_iteration"]
                task = self.task_for("continuation", iteration)
                if mode in {"lower", "alternating"}:
                    self.lower_step(task, f"continuation_lower_{iteration}", feedback=mode == "alternating")
                    self.freeze_pool(task, f"continuation_fresh_{iteration}")
                if mode in {"upper", "alternating"}:
                    self.upper_step(task, f"continuation_upper_{iteration}")
                self.state["continuation_iteration"] += 1
                self.state["costs"]["continuation"] += perf_counter() - started
                self.commit()
            self.state["phase"] = "complete"
            self.commit()
