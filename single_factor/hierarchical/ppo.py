"""使用固定旧策略概率和动作条件回报目标进行 PPO 裁剪更新。"""
# 上下层共用的 PPO 更新实现，同时提供下层续采样和上层回报计算。
from __future__ import annotations

import random
import numpy as np
import torch


def sample_edit_tail(model, row):
    """固定已执行动作及任务条件，只采样剩余编辑。"""
    condition = {"task": row["task"]} if "task" in row else {}
    final, _ = model.sample(row["after"], remaining=row["step"] - 1, record=False, **condition)
    return final


@torch.no_grad()
def continuation_returns(model, trajectories, rewards, evaluate, samples=4, states=8, rng=None, selected=None):
    """共使用 K 个终点样本，包含原路径和额外 K−1 条独立续采样路径。

    固定已经选择的当前动作，再采样剩余编辑。
    分支的真实终点评分用于动作价值估计，上层发现数量只统计主候选。
    主路径和续采样收集完成后，再进入策略更新阶段。
    """
    # samples 是终点样本总数。例如取 3 时，包含原路径和额外的 2 条续采样路径。
    if samples < 1 or states < 0 or len(trajectories) != len(rewards):
        raise ValueError("invalid continuation settings")
    rng = rng or random.Random(0)
    # 只在还有后续编辑的状态中抽样。最后一步的真实终点奖励已经确定。
    choices = [(i, j) for i, rows in enumerate(trajectories) for j, row in enumerate(rows) if row["step"] > 1]
    selected = set(rng.sample(choices, min(states, len(choices)))) if selected is None else set(selected)
    if not selected <= set(choices):
        raise ValueError("continuation selection outside collected non-terminal states")
    branches = 0
    for i, (rows, reward) in enumerate(zip(trajectories, rewards)):
        if not np.isfinite(reward):
            raise ValueError("terminal reward must be finite")
        for j, row in enumerate(rows):
            # 原轨迹是旧策略生成的一个终点样本。未选中的状态直接使用这个样本。
            outcomes = [float(reward)]
            terminal = rows[-1]["after"]
            if hasattr(model, "as_node"):
                terminal = model.as_node(terminal)
            endpoints = [terminal.to_dict()]
            if (i, j) in selected:
                for _ in range(samples - 1):
                    final = sample_edit_tail(model, row)
                    outcomes.append(float(evaluate(final)))
                    endpoints.append(final.to_dict())
                    branches += 1
            if not np.isfinite(outcomes).all():
                raise ValueError("continuation score must be finite")
            # 用终点奖励均值估计 Q(s,a)，这里只估计下层本次编辑的条件期望。
            row["return"] = float(np.mean(outcomes))
            row["continuation_scores"] = outcomes
            row["continuation_endpoints"] = endpoints
    return branches


def complete_returns(rows):
    # 上层完整搜索回合的累计回报，折扣因子为 1。
    """按回合末端反向累计奖励，原地写入每条轨迹行的 return，并返回总回报。

    使用不折扣的有限回合回报，折扣因子为一。
    """
    total = 0.0
    for row in reversed(rows):
        total += row["reward"]
        row["return"] = total
    return total


def immediate_targets(rows):
    """使用单步信用分配计算训练目标，原回合累计回报另行保存。"""
    # 这是即时信用分配实验分支，策略和价值网络都使用当前这次扩展的奖励。
    for row in rows:
        row["value_target"] = row["reward"]
        row["advantage"] = row["reward"] - row["value"]


def gae_targets(rows, trace_lambda, *, value_offsets=None):
    """在奖励的尺度上估计优势，再把价值目标换回网络输出的尺度。"""
    # 从回合结束处反推 GAE，结束后的状态价值取 0。
    if not 0 <= trace_lambda <= 1:
        raise ValueError("GAE lambda must be between zero and one")
    advantage, next_value = 0.0, 0.0
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        value = row["value"] - value_offsets[index] if value_offsets is not None else row["value"]
        delta = row["reward"] + next_value - value
        advantage = delta + trace_lambda * advantage
        row["advantage"] = advantage
        row["value_target"] = row["value"] + advantage
        next_value = value


def update(model, optimizer, rows, *, epochs=4, batch_size=16, clip=0.2,
           entropy_weight=0.01, value_weight=0.5, max_grad_norm=1.0, target_kl=None, rng=None):
    """根据旧策略轨迹执行 PPO 更新，返回损失、熵、KL、概率重放误差等诊断。

    先核验旧动作概率可重放，再归一化优势，优化裁剪策略目标、价值误差和熵项。
    支持预设优势及价值目标、梯度裁剪和 KL 提前停止，会修改模型与优化器状态。
    """
    if not rows or epochs < 1 or batch_size < 1 or not 0 < clip < 1:
        raise ValueError("invalid PPO batch/settings")
    if target_kl is not None and target_kl <= 0:
        raise ValueError("target KL must be positive")
    rng = rng or random.Random(0)
    device = next(model.parameters()).device
    # 下层默认使用续采样 Q 减去旧价值估计，上层可事先写入 GAE 等优势目标。
    advantages = np.array([row.get("advantage", row["return"] - row["value"]) for row in rows], dtype=np.float64)
    raw_std = float(advantages.std())
    targets = np.array([row.get("value_target", row["return"]) for row in rows])
    predicted = np.array([row["value"] for row in rows])
    explained = float(1 - np.var(targets - predicted) / np.var(targets)) if np.var(targets) > 1e-12 else None
    if advantages.std() > 1e-8 and len(rows) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std()
    evaluate_rows = model.rows if hasattr(model, "rows") else lambda batch: [model.row(row) for row in batch]
    history = []
    stopped = False
    # 更新前重算旧动作的对数概率，核验合法动作集合、任务条件和概率重放一致。
    with torch.no_grad():
        # 轨迹长度可变，重放也使用优化小批的大小，避免整批所有动作同时进入显存。
        observed = np.empty(len(rows), dtype=np.float64)
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            observed[start:start + len(batch)] = [float(value[0]) for value in evaluate_rows(batch)]
        stored = np.array([row["logp"] for row in rows])
        if not np.isfinite(observed).all() or not np.isfinite(stored).all():
            raise AssertionError("old-policy action probability replay contains non-finite values")
        differences = np.abs(observed - stored)
        batch_error = float(differences.max())
        batch_mismatch = batch_error > min(2e-3, clip * 1e-2)
        replay_row = getattr(model, "replay_row", None)
        exact_update = batch_mismatch and replay_row is not None
        fallback = range(len(rows)) if exact_update else np.flatnonzero(differences > 2e-5)
        # CUDA批次形状可改变数值归约；超差时按实际采样路径复核，原阈值保持不变。
        for index in fallback:
            row = rows[index]
            if exact_update:
                actual = replay_row(row)[0]
            elif hasattr(model, "replay_logp"):
                actual = model.replay_logp(row)
            elif hasattr(model, "row"):
                actual = model.row(row)[0]
            else:
                actual = model.rows([row])[0][0]
            difference = abs(float(actual) - stored[index])
            if not np.isfinite(difference) or difference > 2e-5:
                raise AssertionError(f"old-policy action probability replay mismatch: {difference} "
                                     f"(row={index}, batch_error={batch_error})")
            differences[index] = difference
        if batch_mismatch and not exact_update:
            raise AssertionError(f"old-policy batch probability replay mismatch: {batch_error}")
        error = float(differences.max())
    if exact_update:
        # 批量结果超差时不用它做梯度更新，整个 PPO 批次沿用已核验的原采样布局。
        evaluate_rows = lambda batch: [replay_row(row) for row in batch]
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for cursor in range(0, len(order), batch_size):
            indices = order[cursor:cursor + batch_size]
            evaluated = evaluate_rows([rows[i] for i in indices])
            logp = torch.stack([x[0] for x in evaluated])
            values = torch.stack([x[1] for x in evaluated])
            entropy = torch.stack([x[2] for x in evaluated]).mean()
            old = torch.tensor([rows[i]["logp"] for i in indices], device=device)
            returns = torch.tensor(targets[indices], dtype=torch.float32, device=device)
            advantage = torch.tensor(advantages[indices], dtype=torch.float32, device=device)
            # 新旧策略概率比用于 PPO 裁剪，下层传入的是位置与替换的联合概率。
            ratio = (logp - old).exp()
            measured_kl = ((ratio - 1) - (logp - old)).mean()
            if target_kl is not None and float(measured_kl.detach()) > 1.5 * target_kl:
                stopped = True
                break
            # 策略网络提高高优势动作的概率，价值网络拟合回报，熵项维持探索。
            actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
            critic_loss = (values - returns).square().mean()
            loss = actor_loss + value_weight * critic_loss - entropy_weight * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            history.append({"actor_loss": float(actor_loss.detach()), "value_loss": float(critic_loss.detach()),
                            "entropy": float(entropy.detach()), "approx_kl": float(((ratio - 1) - (logp - old)).mean().detach())})
        if stopped:
            break
    losses = {key: float(np.mean([h[key] for h in history])) if history else 0.0
              for key in ("actor_loss", "value_loss", "entropy", "approx_kl")}
    return losses | {
        "old_logp_max_error": error, "samples": len(rows),
        "old_logp_batch_max_error": batch_error, "old_logp_scalar_checks": len(fallback),
        "old_logp_sampling_layout_update": exact_update,
        "advantage_std_before_normalization": raw_std, "value_explained_variance": explained,
        "kl_early_stop": stopped, "optimizer_steps": len(history)}
