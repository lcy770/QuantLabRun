"""同一下层 PPO 训练起点与扩散编辑，上层 PPO 构造等权成员集合。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .models import LowerPolicy, LowerTask, RPNEncoder, UpperPolicy, mlp


class GeneratorPolicy(nn.Module):
    """起点选择头与原生 RPN 去噪策略共用一个下层优化器和终点奖励。"""

    def __init__(self, grammar, width, noise_steps):
        """复用现有起点网络及合法编辑策略，保留源公式的持续条件。"""
        super().__init__()
        self.grammar, self.width, self.noise_steps = grammar, width, noise_steps
        self.starter = UpperPolicy(grammar, width, representation="rpn")
        self.editor = LowerPolicy(grammar, width, noise_steps, use_task=True,
                                  representation="rpn", edit_backend="rpn")

    def rows(self, rows):
        """按动作种类重放同一下层轨迹，返回各决策自己的概率、价值和熵。"""
        result = [None] * len(rows)
        for kind, policy in (("start", self.starter), ("edit", self.editor)):
            positions = [i for i, row in enumerate(rows) if row["kind"] == kind]
            if positions:
                evaluated = policy.rows([rows[i] for i in positions])
                for index, value in zip(positions, evaluated):
                    result[index] = value
        if any(value is None for value in result):
            raise ValueError("unknown lower action kind")
        return result

    @torch.no_grad()
    def generate(self, observation, rng, steps, perturbation_steps, record=True):
        """采样起点或 NEW，经固定正向扰动与原生 RPN 反向编辑生成终点。"""
        start_row = self.starter.sample(observation)
        index = start_row["index"]
        try:
            source = observation.trees[index] if index < len(observation.trees) else self.grammar.sample(rng)
        except RuntimeError as error:
            if "grammar prior exhausted" not in str(error):
                raise
            start_row["kind"] = "start"
            start_row["error"] = str(error)
            return None, [start_row], None
        noisy, _ = self.editor.corrupt(source, perturbation_steps, rng)
        task = LowerTask(source, (), (), 0.0)
        final, edits = self.editor.sample(noisy, remaining=steps, record=record, task=task)
        start_row["kind"] = "start"
        for row in edits:
            row["kind"] = "edit"
        return final, [start_row, *edits], source


@dataclass(frozen=True)
class SelectionObservation:
    """冻结的 A 段候选特征、相关矩阵与当前成员索引，完全不含 R 数据。"""
    trees: tuple
    features: tuple
    correlations: tuple
    selected: tuple
    maximum: int
    pool_id: str

    def __post_init__(self):
        """验证候选、特征、相关矩阵和已选成员的一致性。"""
        count = len(self.trees)
        if (len(self.features) != count or len(self.correlations) != count or self.maximum < 1
                or any(len(row) != count for row in self.correlations)
                or any(len(row) != 6 for row in self.features)
                or len(set(self.selected)) != len(self.selected)
                or any(index < 0 or index >= count for index in self.selected)
                or len(self.selected) > self.maximum):
            raise ValueError("inconsistent selection observation")


class SelectionTower(nn.Module):
    """对候选置换等变、对已选集合置换不变的成员条件编码。"""

    def __init__(self, grammar, width):
        """分别创建公式编码器与 A 段指标投影。"""
        super().__init__()
        self.encoder = RPNEncoder(grammar, width)
        self.candidate = mlp(width + 6, width, width)
        self.width = width

    def forward(self, observation, encoded=None):
        """汇总候选及已选集合，并为每个候选计算对已选集合的有向相关特征。"""
        reference = self.encoder.embedding.weight
        zero = reference.new_zeros(self.width)
        if observation.trees:
            encoded = self.encoder.encode([], roots=observation.trees) if encoded is None else encoded
            features = reference.new_tensor(observation.features)
            values = self.candidate(torch.cat((torch.stack([encoded[t] for t in observation.trees]), features), dim=1))
            whole = values.mean(0)
        else:
            values, whole = reference.new_empty((0, self.width)), zero
        fraction = reference.new_tensor([len(observation.selected) / observation.maximum])
        selected = values[list(observation.selected)].mean(0) if observation.selected else zero
        context = torch.cat((whole, selected, fraction))
        relations = reference.new_zeros((len(observation.trees), 3))
        if observation.selected:
            rho = reference.new_tensor(observation.correlations)[:, list(observation.selected)]
            relations = torch.stack((rho.min(1).values, rho.max(1).values, rho.mean(1)), dim=1)
        return values, context, relations


class SelectorPolicy(nn.Module):
    """上层仅选择未选成员或 STOP，状态价值与策略使用独立参数。"""

    def __init__(self, grammar, width):
        """创建成员、停止与价值输出头。"""
        super().__init__()
        self.actor = SelectionTower(grammar, width)
        self.critic = SelectionTower(grammar, width)
        self.member_head = mlp(3 * width + 4, width, 1)
        self.stop_head = mlp(2 * width + 1, width, 1)
        self.value_head = mlp(2 * width + 1, width, 1)

    def distribution(self, observation, actor_encoding=None, critic_encoding=None):
        """在完整候选索引上屏蔽已选成员，末索引始终表示 STOP。"""
        values, context, relations = self.actor(observation, actor_encoding)
        features = torch.cat((values, context.expand(len(values), -1), relations), dim=1)
        members = self.member_head(features).flatten()
        blocked = list(observation.selected)
        if len(observation.selected) >= observation.maximum:
            blocked = list(range(len(observation.trees)))
        if blocked:
            mask = torch.zeros(len(members), dtype=torch.bool, device=members.device)
            mask[blocked] = True
            members = members.masked_fill(mask, -torch.inf)
        logits = torch.cat((members, self.stop_head(context).reshape(1)))
        _, critic_context, _ = self.critic(observation, critic_encoding)
        return Categorical(logits=logits), self.value_head(critic_context).squeeze()

    @torch.no_grad()
    def sample(self, observation, deterministic=False):
        """采样或确定性选择一个成员/STOP，并保存 PPO 旧策略证据。"""
        distribution, value = self.distribution(observation)
        index = int(distribution.logits.argmax() if deterministic else distribution.sample())
        return {"observation": observation, "index": index,
                "logp": float(distribution.logits[index]), "value": float(value)}

    def rows(self, rows):
        """共享批内公式编码，准确重放每行的冻结池、已选掩码与 STOP 概率。"""
        trees = [tree for row in rows for tree in row["observation"].trees]
        actor = self.actor.encoder.encode([], roots=trees)
        critic = self.critic.encoder.encode([], roots=trees)
        result = []
        for row in rows:
            distribution, value = self.distribution(row["observation"], actor, critic)
            result.append((distribution.logits[row["index"]], value, distribution.entropy()))
        return result
