"""同池集合选择对照，包含 LinUCB、Double DQN 和现有 PPO。"""
from __future__ import annotations

import copy
from dataclasses import replace
import random

import numpy as np
import torch
from torch import nn

from .models import mlp
from .selection_models import SelectionTower, SelectorPolicy
from .ppo import update


def available_actions(observation):
    """返回未选成员及 STOP，达到容量时只保留 STOP。"""
    count = len(observation.trees)
    if len(observation.selected) >= observation.maximum:
        return [count]
    return [i for i in range(count) if i not in observation.selected] + [count]


def bandit_features(observation):
    """用 A 段指标、候选与已选集合相关性及 STOP 标记构造共享线性上下文。"""
    count = len(observation.trees)
    values = np.asarray(observation.features, dtype=float).reshape(count, 6)
    whole = values.mean(0) if count else np.zeros(6)
    selected = values[list(observation.selected)].mean(0) if observation.selected else np.zeros(6)
    relation = np.zeros((count, 3))
    if observation.selected:
        rho = np.asarray(observation.correlations)[:, list(observation.selected)]
        relation = np.stack((rho.min(1), rho.max(1), rho.mean(1)), axis=1)
    rows = []
    for index in range(count + 1):
        stop = index == count
        rows.append(np.concatenate(([1.0], np.zeros(6) if stop else values[index], whole, selected,
            np.zeros(3) if stop else relation[index], [len(observation.selected) / observation.maximum, float(stop)])))
    return np.asarray(rows, dtype=np.float64)


class LinUCBSelector:
    """共享线性参数的上下文 UCB，用实际完整组合回报训练各步上下文。"""

    def __init__(self, alpha=1.0, regularization=1.0):
        """初始化正定岭统计量，维度由固定特征定义给出。"""
        if alpha < 0 or regularization <= 0:
            raise ValueError("invalid LinUCB configuration")
        self.alpha = alpha
        self.a = np.eye(24) * regularization
        self.b = np.zeros(24)

    def sample(self, observation, deterministic=False):
        """在相同成员/STOP 支持上取 UCB 最大值，验收确定性模式使用线性均值。"""
        features = bandit_features(observation)
        theta = np.linalg.solve(self.a, self.b)
        inverse_features = np.linalg.solve(self.a, features.T).T
        bonus = np.sqrt(np.maximum(np.sum(features * inverse_features, axis=1), 0))
        scores = features @ theta + (0.0 if deterministic else self.alpha) * bonus
        allowed = available_actions(observation)
        index = max(allowed, key=lambda i: (scores[i], -i))
        return {"observation": observation, "index": index, "value": float(features[index] @ theta)}

    def learn(self, rollouts, rng=None):
        """批末用每个已选动作的完整回报更新 A、b，动作采集期间统计量冻结。"""
        count = 0
        for rows in rollouts:
            for row in rows:
                feature = bandit_features(row["observation"])[row["index"]]
                target = row["return"]
                if not np.isfinite(target):
                    raise ValueError("non-finite bandit target")
                self.a += np.outer(feature, feature)
                self.b += feature * target
                count += 1
        return {"samples": count, "update_type": "ridge_statistics", "optimizer_steps": 0}

    def checkpoint(self):
        """保存充分统计量。"""
        return {"a": self.a, "b": self.b}

    def restore(self, state):
        """恢复统计量并核验正定性及有限值。"""
        self.a, self.b = state["a"].copy(), state["b"].copy()
        if self.a.shape != (24, 24) or self.b.shape != (24,) or not np.isfinite(self.a).all() or not np.isfinite(self.b).all():
            raise ValueError("invalid LinUCB checkpoint")
        np.linalg.cholesky(self.a)


class SetQNetwork(nn.Module):
    """复用同一集合编码表示，以 Q 值输出成员和 STOP。"""

    def __init__(self, grammar, width):
        """创建候选、已选集合及相关性条件的动作价值网络。"""
        super().__init__()
        self.tower = SelectionTower(grammar, width)
        self.member = mlp(3 * width + 4, width, 1)
        self.stop = mlp(2 * width + 1, width, 1)

    def forward(self, observation):
        """输出完整候选索引与 STOP 的动作值，并屏蔽不可行动作。"""
        values, context, relations = self.tower(observation)
        features = torch.cat((values, context.expand(len(values), -1), relations), dim=1)
        result = torch.cat((self.member(features).flatten(), self.stop(context).reshape(1)))
        mask = torch.ones(len(result), dtype=torch.bool, device=result.device)
        mask[available_actions(observation)] = False
        return result.masked_fill(mask, -torch.inf)


class DoubleDQNSelector:
    """在线网络选择下一动作，目标网络评价下一动作的 Double DQN。"""

    def __init__(self, grammar, width, settings, device="cpu"):
        """初始化双网络、优化器及有限经验池。"""
        self.settings = settings
        self.online = SetQNetwork(grammar, width).to(device)
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=settings.get("learning_rate", 0.0003))
        self.replay = []
        self.updates = 0

    @torch.no_grad()
    def sample(self, observation, deterministic=False):
        """按 epsilon-greedy 选择合法成员/STOP，冻结验收时关闭探索。"""
        q = self.online(observation)
        epsilon = 0.0 if deterministic else self.settings.get("epsilon", 0.1)
        allowed = available_actions(observation)
        if float(torch.rand(())) < epsilon:
            index = allowed[int(torch.randint(len(allowed), (1,)))]
        else:
            index = int(q.argmax())
        return {"observation": observation, "index": index, "value": float(q[index])}

    def learn(self, rollouts, rng=None):
        """存储真实逐步奖励，使用在线 argmax 与目标网络估值形成 TD 目标。"""
        rng = rng or random.Random(0)
        for rows in rollouts:
            for index, row in enumerate(rows):
                following = rows[index + 1]["observation"] if index + 1 < len(rows) else None
                self.replay.append((row["observation"], row["index"], row["reward"], following))
        capacity = self.settings.get("replay_capacity", 10000)
        del self.replay[:-capacity]
        losses = []
        for _ in range(self.settings.get("gradient_steps", 4)):
            batch = rng.sample(self.replay, min(self.settings.get("batch_size", 16), len(self.replay)))
            predictions, targets = [], []
            for observation, action, reward, following in batch:
                predictions.append(self.online(observation)[action])
                with torch.no_grad():
                    target = float(reward)
                    if following is not None:
                        action_next = int(self.online(following).argmax())
                        target += self.settings.get("discount", 1.0) * float(self.target(following)[action_next])
                targets.append(target)
            prediction = torch.stack(predictions)
            target = prediction.new_tensor(targets)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Double DQN loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.updates += 1
            if self.updates % self.settings.get("target_update_interval", 10) == 0:
                self.target.load_state_dict(self.online.state_dict())
            losses.append(float(loss.detach()))
        return {"samples": sum(map(len, rollouts)), "optimizer_steps": len(losses),
                "td_loss": float(np.mean(losses)), "update_type": "double_dqn"}

    def checkpoint(self):
        """保存网络、优化器、经验池和目标更新计数。"""
        return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "replay": self.replay, "updates": self.updates}

    def restore(self, state):
        """完整恢复学习状态。"""
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay, self.updates = state["replay"], state["updates"]


class PPOSelector:
    """为实验驱动包装现有集合 PPO，保持相同概率重放核验。"""

    def __init__(self, grammar, width, settings, device="cpu"):
        """创建实际策略与优化器。"""
        self.policy = SelectorPolicy(grammar, width).to(device)
        self.settings = settings
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=settings["learning_rate"])

    def sample(self, observation, deterministic=False):
        """使用现有集合策略采样。"""
        return self.policy.sample(observation, deterministic)

    def learn(self, rollouts, rng=None):
        """按已保存旧概率执行 PPO。"""
        return update(self.policy, self.optimizer, [row for rows in rollouts for row in rows],
                      rng=rng, **self.settings["ppo"])

    def checkpoint(self):
        """保存权重和优化器。"""
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}

    def restore(self, state):
        """恢复权重和优化器。"""
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])


def create_selector(method, grammar, config):
    """按实验名称创建实际学习器，常规筛选由公共选择函数执行。"""
    settings = config["upper"]
    if method == "upper_ppo":
        return PPOSelector(grammar, config["width"], settings, config["device"])
    if method == "linucb":
        return LinUCBSelector(**settings.get("linucb", {}))
    if method == "double_dqn":
        return DoubleDQNSelector(grammar, config["width"], settings.get("double_dqn", {}), config["device"])
    if method not in {"random", "quality", "forward_exchange"}:
        raise ValueError(f"unknown comparison selector {method}")
    return None
