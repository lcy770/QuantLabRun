"""扩散、自回归 PPO、随机合法编辑和 GP 的真实训练及生成后端。"""
from __future__ import annotations

import copy
import io
import math
from time import perf_counter

import numpy as np
import torch

from .ast import Node
from .models import LowerTask
from .ppo import update
from .rpn import RPNKernel, TokenEdit
from .selection_models import GeneratorPolicy
from .selection_search import lower_targets


class DiffusionBackend:
    """复用原生 RPN 扩散及同一下层起点策略，支持仅结构初始化对照。"""
    uses_start_bank = True
    supports_parallel = False

    def __init__(self, grammar, config, pretrained_only=False):
        """创建实际生成策略和金融、结构两个优化器。"""
        self.grammar, self.config = grammar, config
        self.device = torch.device(config["device"])
        self.policy = GeneratorPolicy(grammar, config["width"], config["corpus"]["noise_steps"],
                                      learned_stop=config["lower"].get("learned_stop", False)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["lower"]["learning_rate"])
        self.syntax_optimizer = torch.optim.Adam(self.policy.editor.parameters(), lr=config["pretrain"]["learning_rate"])
        self.pretrained_only = pretrained_only

    def generate(self, observation, rng, steps, perturbation_steps, record=True):
        """PPO 使用学习起点，仅预训练对照使用均匀起点及已预训练编辑器。"""
        if not self.pretrained_only:
            return self.policy.generate(observation, rng, steps, perturbation_steps, record)
        index = rng.randrange(len(observation.trees) + 1)
        source = observation.trees[index] if index < len(observation.trees) else self.grammar.sample(rng)
        self.last_source_id = source.id if index < len(observation.trees) else "NEW"
        noisy, _ = self.policy.editor.corrupt(source, perturbation_steps, rng)
        final, rows = self.policy.editor.sample(noisy, remaining=steps, record=record,
                                               task=LowerTask(source, (), (), 0.0), allow_stop=False)
        return final, rows, source

    def pretrain(self, records, rng):
        """训练一轮真实逆编辑标签，返回训练损失和独立结构检查损失。"""
        from .training import decode_pair, syntax_loss
        pairs = [pair for record in records if record["split"] == "train" for pair in record["pairs"]]
        ordered = rng.sample(pairs, len(pairs))
        losses = []
        size = self.config["pretrain"]["batch_size"]
        for start in range(0, len(ordered), size):
            loss = torch.stack([-self.policy.editor.evaluate_action(*decode_pair(pair))[0]
                                for pair in ordered[start:start + size]]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite structural loss")
            self.syntax_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.editor.parameters(), 1.0, error_if_nonfinite=True)
            self.syntax_optimizer.step()
            losses.append(float(loss.detach()))
        return {"train_nll": float(np.mean(losses)),
                "check_nll": syntax_loss(self.policy.editor, [r for r in records if r["split"] == "check"])}

    def learn(self, proposals, rng, contributions=None, *, evaluator=None, reference_seconds=None):
        """冻结策略完成主轨迹与续采样的信用分配，再执行下层 PPO 更新。"""
        rows, feedback, eta, continuation = lower_targets(
            self.policy, proposals, self.config, contributions, evaluator, reference_seconds)
        result = update(self.policy, self.optimizer, rows, rng=rng, **self.config["lower"]["ppo"])
        return {**result, "feedback": feedback, "update_type": "ppo", "eta_effective": eta,
                "edit_continuations": continuation}

    def checkpoint(self):
        """保存权重及两类优化器状态。"""
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                "syntax_optimizer": self.syntax_optimizer.state_dict()}

    def restore(self, state):
        """完整恢复训练状态。"""
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.syntax_optimizer.load_state_dict(state["syntax_optimizer"])


class RandomEditor:
    """在相同原生 RPN 合法动作核上均匀选择起点、位置和条件替换。"""
    uses_start_bank = True
    supports_parallel = False
    device = torch.device("cpu")

    def __init__(self, grammar, config):
        """绑定共享文法与编辑内核。"""
        self.grammar, self.config = grammar, config
        self.kernel = RPNKernel(grammar)

    def generate(self, observation, rng, steps, perturbation_steps, record=True):
        """执行固定正向扰动及随机反向编辑，NOOP 使用相同合法支持。"""
        index = rng.randrange(len(observation.trees) + 1)
        source = observation.trees[index] if index < len(observation.trees) else self.grammar.sample(rng)
        self.last_source_id = source.id if index < len(observation.trees) else "NEW"
        program, _ = self.kernel.corrupt(self.kernel.from_node(source), perturbation_steps, rng)
        for _ in range(steps):
            paths = self.kernel.edit_paths(program)
            path = rng.choice(paths)
            action = rng.choice(self.kernel.edits_at(program, path))
            program = self.kernel.apply(program, action)
        return self.kernel.to_node(program), [], source

    def checkpoint(self):
        """随机策略不含拟合参数。"""
        return {}

    def restore(self, state):
        """核验随机策略的空状态。"""
        if state:
            raise ValueError("random editor has no learned state")


class GeneticProgramming(RandomEditor):
    """按 G 段质量成本适应度进行锦标赛、合法子式交叉及变异的 GP。"""

    def __init__(self, grammar, config):
        """每个历史任务保存独立种群。"""
        super().__init__(grammar, config)
        self.populations = {}
        self.current = None

    def bind(self, evaluator, reference_seconds):
        """以 G 段身份隔离种群，冻结本轮成本尺度。"""
        self.evaluator = evaluator
        self.current = (str(evaluator.panels["search"].dates[0]), str(evaluator.panels["search"].dates[-1]))
        self.populations.setdefault(self.current, {})

    def generate(self, observation, rng, steps, perturbation_steps, record=True):
        """用适应度锦标赛选父本，原生词元交叉后执行合法变异。"""
        population = self.populations[self.current]
        for tree in observation.trees:
            if tree.id not in population:
                candidate, _ = self.evaluator.assess(tree, "gp_initial")
                population[tree.id] = (tree, candidate.quality)
        if not population:
            tree = self.grammar.sample(rng)
            population[tree.id] = (tree, self.evaluator.assess(tree, "gp_initial")[0].quality)
        values = list(population.values())
        settings = self.config.get("gp", {})

        def parent():
            """从有限个体中进行无放回锦标赛。"""
            choices = rng.sample(values, min(settings.get("tournament_size", 3), len(values)))
            return max(choices, key=lambda entry: (entry[1], entry[0].id))[0]

        first, second = parent(), parent()
        program, donor = self.kernel.from_node(first), self.kernel.from_node(second)
        if rng.random() < settings.get("crossover_probability", 0.8):
            positions, sources = self.kernel.positions(program), self.kernel.positions(donor)
            for _ in range(32):
                path, replacement = rng.choice(list(positions)), rng.choice(list(sources.values()))
                try:
                    legal = self.kernel._legal_replacement(program, path, replacement)
                except (ValueError, KeyError):
                    legal = False
                if legal:
                    spans, indices = self.kernel.analyze(program)
                    span = spans[indices[path]]
                    program = self.kernel.apply(program, TokenEdit(path, span.start, span.stop, replacement))
                    break
        if rng.random() < settings.get("mutation_probability", 0.3):
            path = rng.choice(self.kernel.edit_paths(program))
            program = self.kernel.apply(program, rng.choice(self.kernel.edits_at(program, path)))
        self.kernel.validate(program)
        self.last_source_id = first.id if first.id in observation.ids else "NEW"
        return self.kernel.to_node(program), [], first

    def observe_batch(self, proposals):
        """批末按实测 Q−κC 适应度执行精英保留，失败候选仍有连续适应度。"""
        population = self.populations[self.current]
        for proposal in proposals:
            candidate = proposal["candidate"]
            if candidate is not None:
                population[candidate.tree.id] = (candidate.tree, proposal["base_reward"])
        limit = self.config.get("gp", {}).get("population_size", 64)
        ranked = sorted(population.values(), key=lambda entry: (-entry[1], entry[0].id))[:limit]
        self.populations[self.current] = {tree.id: (tree, score) for tree, score in ranked}

    def checkpoint(self):
        """保存全部任务的种群与适应度。"""
        return {"populations": copy.deepcopy(self.populations)}

    def restore(self, state):
        """恢复任务种群。"""
        self.populations = copy.deepcopy(state["populations"])


class AutoregressivePPO:
    """使用仓库原 RPN 环境和 LSTM 的真实 MaskablePPO。"""
    uses_start_bank = False
    supports_parallel = False
    last_source_id = "NEW"

    def __init__(self, grammar, config, original_reward=False):
        """构建原词元生成模型，限制常量和窗口到公共文法。"""
        from sb3_contrib import MaskablePPO
        from ..environment import SingleFactorEnvCore, SingleFactorEnvWrapper
        from .._vendor.alphagen.config import CONSTANTS, DELTA_TIMES
        from .._vendor.alphagen.rl.policy import LSTMSharedNet
        self.grammar, self.config, self.original_reward = grammar, config, original_reward
        self.device = torch.device(config["device"])
        self.evaluator, self.reference_seconds = None, None
        self.last_tree, self.last_reward, self.last_error = None, 0.0, None
        owner = self

        class VocabularyEnvironment(SingleFactorEnvWrapper):
            def reset(self, **kwargs):
                """从完整公式起点开始计量本次自回归生成成本。"""
                owner.started = perf_counter()
                owner.last_tree, owner.last_error = None, None
                return super().reset(**kwargs)

            def step(self, action):
                """未形成完整公式的终点同样承担统一质量成本奖励。"""
                state, reward, done, truncated, info = super().step(action)
                if (done or truncated) and "metrics" not in info:
                    owner.last_error = info.get("reason", "incomplete expression")
                    if not owner.original_reward:
                        reward = -1.0 - config["reward"]["kappa"] * (perf_counter() - owner.started) / owner.reference_seconds
                return state, reward, done, truncated, info

            def action_masks(self):
                """沿用原 RPN 栈掩码，并使用所有生成器相同的常量和窗口。"""
                mask = super().action_masks()
                offset = self.size_op + self.size_feature
                mask[offset:offset + self.size_constant] &= np.array([x in grammar.constants for x in CONSTANTS])
                offset += self.size_constant
                mask[offset:offset + self.size_delta_time] &= np.array([x in grammar.windows for x in DELTA_TIMES])
                return mask

        self.environment = VocabularyEnvironment(SingleFactorEnvCore(grammar.fields, self),
                                                [grammar.classes[name] for name in grammar.operators])
        settings = config.get("autoregressive", {})
        self.model = MaskablePPO("MlpPolicy", self.environment,
            policy_kwargs={"features_extractor_class": LSTMSharedNet,
                "features_extractor_kwargs": {"n_layers": 2, "d_model": settings.get("width", 128),
                                               "dropout": 0.0, "device": config["device"]}},
            n_steps=settings.get("n_steps", 128), batch_size=settings.get("batch_size", 64),
            n_epochs=settings.get("epochs", 4), gamma=1.0, ent_coef=0.01,
            learning_rate=config["lower"]["learning_rate"], seed=config["seed"], device=config["device"], verbose=0)

    def bind(self, evaluator, reference_seconds):
        """训练及推理都绑定同一任务 F/G，保持 A/R 不进入生成奖励。"""
        self.evaluator, self.reference_seconds = evaluator, reference_seconds

    def evaluate(self, expression):
        """环境终点奖励使用原加权绝对指标，或统一的有向 Q−κC。"""
        from ..evaluator import SingleFactorMetrics
        from custom_bt.canonical_expression import serialize_alphagen_expression
        text = str(expression)
        try:
            text = serialize_alphagen_expression(expression)
            self.last_tree = self.grammar.parse(text)
            candidate, eligible = self.evaluator.assess(self.last_tree, "autoregressive_training")
            metrics = candidate.search_metrics
            if self.original_reward:
                valid = np.isfinite([metrics[name] for name in ("ic", "rank_ic", "icir")]).all()
                reward = (.5 * abs(metrics["ic"]) + .3 * abs(metrics["rank_ic"]) + .2 * np.tanh(abs(metrics["icir"]))) if valid and metrics["coverage"] >= self.evaluator.thresholds[3] else 0.0
            else:
                reward = candidate.quality - self.config["reward"]["kappa"] * (perf_counter() - self.started) / self.reference_seconds
            result = SingleFactorMetrics(text, metrics["ic"], metrics["rank_ic"], metrics["icir"],
                                         metrics["coverage"], float(reward), eligible, None, {})
        except (ValueError, KeyError, RuntimeError) as error:
            self.last_tree, self.last_error = None, str(error)
            reward = -1.0
            if not self.original_reward:
                reward -= self.config["reward"]["kappa"] * (perf_counter() - self.started) / self.reference_seconds
            result = SingleFactorMetrics(text, float("nan"), float("nan"), float("nan"), 0.0,
                                         reward, False, str(error), {})
        self.last_reward = result.score
        return result

    def generate(self, observation, rng, steps, perturbation_steps, record=True):
        """冻结自回归网络逐词元生成完整公式，金融评价通过同一入口完成。"""
        state, _ = self.environment.reset(seed=rng.randrange(2**31))
        for _ in range(32):
            masks = self.environment.action_masks()
            if not masks.any():
                self.last_tree, self.last_error = None, "no legal RPN token"
                break
            action, _ = self.model.predict(state, action_masks=masks, deterministic=False)
            state, _, done, truncated, _ = self.environment.step(int(action))
            if done or truncated:
                break
        self.environment.env.drain_completed_metrics()
        return self.last_tree, [], None

    def learn_task(self, evaluator, reference_seconds):
        """收集完整在策略 SB3 rollout 后执行真实 PPO 更新。"""
        from stable_baselines3.common.callbacks import BaseCallback
        self.bind(evaluator, reference_seconds)
        owner = self
        completed = []

        class Capture(BaseCallback):
            def _on_step(self):
                """记录合法与失败的每个完整公式，持续完成当前 PPO rollout。"""
                metrics = owner.environment.env.drain_completed_metrics()
                if bool(self.locals["dones"][0]):
                    completed.append({"expression": metrics[-1].expression if metrics else None,
                        "reward": float(self.locals["rewards"][0]),
                        "error": metrics[-1].reason if metrics else self.locals["infos"][0].get("reason")})
                return True

        before = self.model.num_timesteps
        self.model.set_env(self.environment, force_reset=True)
        self.model.learn(total_timesteps=self.config.get("autoregressive", {}).get("n_steps", 128),
                         reset_num_timesteps=False, callback=Capture())
        return {"samples": self.model.num_timesteps - before, "completed_formulas": len(completed),
                "terminals": completed, "update_type": "maskable_ppo", "optimizer_steps": self.model.n_epochs}

    def checkpoint(self):
        """用 SB3 官方序列化同时保存策略及其真实优化器。"""
        stream = io.BytesIO()
        self.model.save(stream)
        return {"sb3_zip": stream.getvalue()}

    def restore(self, state):
        """恢复 SB3 模型并保持当前任务的评价环境。"""
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(io.BytesIO(state["sb3_zip"]), env=self.environment, device=self.config["device"])


def create_generator(method, grammar, config):
    """创建实验列出的实际生成算法。"""
    if method in {"diffusion_ppo", "pretrained_diffusion"}:
        return DiffusionBackend(grammar, config, pretrained_only=method == "pretrained_diffusion")
    if method == "random_edit":
        return RandomEditor(grammar, config)
    if method == "gp":
        return GeneticProgramming(grammar, config)
    if method in {"rpn_original", "rpn_unified"}:
        return AutoregressivePPO(grammar, config, original_reward=method == "rpn_original")
    raise ValueError(f"unknown comparison generator {method}")
