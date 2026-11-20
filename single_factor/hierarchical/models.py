# 上下层策略网络、价值网络与公式编码器，奖励定义和训练阶段由 training.py 负责。
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat
import math
from time import perf_counter

import torch
from torch import nn
from torch.distributions import Categorical

from .ast import Node, replace, walk
from .io import check_budget
from .rpn import RPNKernel, RPNProgram, token_vocabulary


def model_metadata(lower, upper, width, *, lower_width=None):
    """统计实际策略网络规模；非神经方法记零，优化器和DQN目标副本不计入。"""
    result = {"width": width, "parameter_scope": "neural_policy_and_value_networks"}
    for name, model, hidden_width in (("lower", lower, width if lower_width is None else lower_width),
                                      ("upper", upper, width)):
        parameters = list(model.parameters()) if model is not None else []
        result[name] = {"network": type(model).__name__ if model is not None else None,
                        "width": hidden_width if model is not None else None,
                        "parameter_count": sum(parameter.numel() for parameter in parameters),
                        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters
                                                         if parameter.requires_grad),
                        "dtypes": sorted({str(parameter.dtype).removeprefix("torch.") for parameter in parameters}),
                        "weight_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters)}
    result["total_parameters"] = result["lower"]["parameter_count"] + result["upper"]["parameter_count"]
    result["total_weight_bytes"] = result["lower"]["weight_bytes"] + result["upper"]["weight_bytes"]
    return result


def token_name(node):
    """将公式节点转换为词表键，常量和窗口键包含规范化后的数值。"""
    if node.symbol == "constant":
        return f"constant:{float(node.value)}"
    if node.symbol == "window":
        return f"window:{int(node.value)}"
    return node.symbol


class TreeEncoder(nn.Module):
    """按深度批量编码不重复子树，从叶节点到根节点保留子节点的顺序信息。"""
    def __init__(self, grammar, width):
        """按文法词表和表示宽度创建树编码器的嵌入、子节点角色投影及归一化层。"""
        super().__init__()
        self.vocabulary = token_vocabulary(grammar)
        self.embedding = nn.Embedding(len(self.vocabulary), width)
        self.roles = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(3))
        self.norm = nn.LayerNorm(width)
        self.width = width

    def encode(self, trees, roots=()):
        """编码输入公式与额外根节点，返回节点到表示张量的映射。

        先按子树高度分组，再自底向上批量计算，相同子树共用本次编码结果。
        """
        levels, heights = defaultdict(list), {}
        def visit(node):
            """递归计算节点高度并登记各层唯一子树，返回当前节点的高度。"""
            if node not in heights:
                depth = 1 + max((visit(c) for c in node.children), default=0)
                heights[node] = depth
                levels[depth].append(node)
            return heights[node]
        for tree in (*trees, *roots):
            visit(tree)
        device = self.embedding.weight.device
        encoded = {}
        for depth in sorted(levels):
            nodes = levels[depth]
            tokens = [token_name(node) for node in nodes]
            values = self.embedding(torch.tensor([self.vocabulary[t] for t in tokens], device=device))
            zero = values.new_zeros(self.width)
            for position, role in enumerate(self.roles):
                children = torch.stack([encoded[n.children[position]] if position < len(n.children) else zero for n in nodes])
                values = values + role(children)
            values = torch.tanh(self.norm(values))
            encoded.update(zip(nodes, values.unbind()))
        return encoded


class RPNEncoder(nn.Module):
    # 词元嵌入经GRU得到后缀表达式表示，同一批相同序列共用编码结果。
    def __init__(self, grammar, width):
        """按文法创建词元嵌入、GRU 编码层和原生 RPN 内核，width 指定表示宽度。"""
        super().__init__()
        self.vocabulary = token_vocabulary(grammar)
        self.embedding = nn.Embedding(len(self.vocabulary), width)
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.width = width
        self.kernel = RPNKernel(grammar)

    def encode(self, trees, roots=()):
        """将公式转换为后缀序列，返回公式或子表达式到 GRU 表示的映射。

        trees 提供可编辑的内部位置，roots 只编码整条表达式，相同词元序列合并计算。
        """
        sequences = {}

        def postfix(node):
            """取得并缓存节点的后缀序列，同时登记树节点或原生 RPN 的各个子表达式。"""
            if node not in sequences:
                if isinstance(node, RPNProgram):
                    sequences.update((part, part.tokens) for part in self.kernel.positions(node).values())
                else:
                    sequences[node] = tuple(index for child in node.children for index in postfix(child)) + (self.vocabulary[token_name(node)],)
            return sequences[node]

        for tree in trees:
            postfix(tree)
        # 替换候选和任务成员只需完整序列的表示。
        # 单独编码其内部所有区间会重复执行 GRU。
        for tree in roots:
            sequences[tree] = self.kernel.from_node(tree).tokens
        if not sequences:
            return {}
        values = list(dict.fromkeys(sequences.values()))
        lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
        device = self.embedding.weight.device
        maximum = int(lengths.max())
        tokens = torch.tensor([sequence + (0,) * (maximum - len(sequence)) for sequence in values],
                              dtype=torch.long, device=device)
        embedded = self.embedding(tokens)
        outputs, _ = self.recurrent(embedded)
        final = outputs[torch.arange(len(values), device=device), lengths.to(device) - 1]
        encoded = torch.tanh(self.norm(final))
        by_sequence = dict(zip(values, encoded.unbind()))
        return {node: by_sequence[sequence] for node, sequence in sequences.items()}


def mlp(input_size, width, output_size):
    """按输入、隐藏和输出维度构造两层全连接网络，中间使用 Tanh 激活。"""
    return nn.Sequential(nn.Linear(input_size, width), nn.Tanh(), nn.Linear(width, output_size))


@dataclass(frozen=True)
class LowerTask:
    """可见的任务状态，其中组合权重已合并拟合期锁定的因子方向。"""
    # 后续任务条件实验使用的可见背景，包含起点、动作前组合和本回合新发现。
    source: Node
    pool_trees: tuple[Node, ...]
    pool_weights: tuple[float, ...]
    pool_ic: float
    discoveries: tuple[Node, ...] = ()

    def __post_init__(self):
        """校验组合成员与权重一一对应，且所有权重及组合 IC 均为有限值。"""
        if len(self.pool_trees) != len(self.pool_weights) or not all(math.isfinite(value) for value in (*self.pool_weights, self.pool_ic)):
            raise ValueError("invalid lower task pool")

    @property
    def trees(self):
        """返回任务中需编码的起点、组合成员和本回合发现，保留各组原有顺序。"""
        return (self.source, *self.pool_trees, *self.discoveries)


class LowerPolicy(nn.Module):
    # 下层将每次编辑分为选择位置、选择该位置的合法替换两个条件动作。
    def __init__(self, grammar, width=32, noise_steps=4, use_task=False, representation="ast", use_position=False,
                 edit_backend="ast", learned_stop=False):
        """创建下层编辑策略及独立的价值编码器，校验编码表示与编辑后端的兼容性。

        noise_steps 决定去噪时间嵌入范围，可选任务条件和位置条件均以零贡献初始化。
        位置头选择编辑位置，替换头在该位置的合法动作中选择修改内容。
        """
        super().__init__()
        if representation not in {"ast", "rpn"}:
            raise ValueError("lower representation must be ast or rpn")
        if edit_backend not in {"ast", "rpn"} or edit_backend == "rpn" and representation != "rpn":
            raise ValueError("native RPN edit backend requires the RPN representation")
        self.grammar, self.width, self.noise_steps = grammar, width, noise_steps
        self.representation = representation
        self.edit_backend = edit_backend
        self.edit_kernel = RPNKernel(grammar) if edit_backend == "rpn" else grammar
        # representation 决定编码器，edit_backend 决定实际执行编辑的内核。
        encoder = RPNEncoder if representation == "rpn" else TreeEncoder
        self.actor_encoder = encoder(grammar, width)
        self.critic_encoder = encoder(grammar, width)
        if edit_backend == "rpn":
            self.actor_encoder.kernel = self.critic_encoder.kernel = self.edit_kernel
        self.actor_time = nn.Embedding(noise_steps + 1, width)
        self.critic_time = nn.Embedding(noise_steps + 1, width)
        self.node_head = mlp(3 * width, width, 1)
        self.edit_head = mlp(4 * width, width, 1)
        self.value_head = mlp(2 * width, width, 1)
        self.stop_head = nn.Linear(2 * width, 1) if learned_stop else None
        self.use_task = use_task
        if use_task:
            # 新条件投影末层零初始化，旧检查点迁移时新增分支的初始贡献为 0。
            self.actor_task = mlp(3 * width + 2, width, width)
            self.critic_task = mlp(3 * width + 2, width, width)
            for projection in (self.actor_task, self.critic_task):
                nn.init.zeros_(projection[-1].weight)
                nn.init.zeros_(projection[-1].bias)
        self.use_position = use_position
        if use_position:
            self.position_roles = max(definition.arity for definition in grammar.definitions.values())
            self.actor_position = nn.Embedding(grammar.max_depth * self.position_roles + 1, width, padding_idx=0)
            nn.init.zeros_(self.actor_position.weight)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """载入下层参数，兼容旧检查点缺少的任务条件分支和位置嵌入。

        缺少的新增参数沿用当前模型初值，其余校验和返回值交给 PyTorch。
        """
        if self.use_position and "actor_position.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["actor_position.weight"] = self.state_dict()["actor_position.weight"]
        if self.use_task and not any(key.startswith(("actor_task.", "critic_task.")) for key in state_dict):
            state_dict = dict(state_dict)
            for key, value in self.state_dict().items():
                if key.startswith(("actor_task.", "critic_task.")):
                    state_dict.setdefault(key, value)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def position_bias(self, paths):
        # 用深度和子节点角色区分相同子表达式在不同位置的出现。
        """将每条出现路径的深度和子节点角色编码并求和，返回各位置的偏置表示。"""
        maximum = max(1, max(map(len, paths)))
        indices = [[1 + depth * self.position_roles + role for depth, role in enumerate(path)]
                   + [0] * (maximum - len(path)) for path in paths]
        values = torch.tensor(indices, dtype=torch.long, device=self.actor_position.weight.device)
        features = self.actor_position(values) * values.ne(0).unsqueeze(-1)
        return features.sum(1)

    def task_embedding(self, task, encoder, projection, encoded=None):
        # 起点、带方向的组合权重及已发现背景共同构成本次扩展的任务表示。
        """编码起点、带方向权重的组合与已发现背景，返回当前任务的条件向量。

        可复用传入的编码映射，任务为空时返回零向量。
        """
        if task is None:
            return self.actor_time.weight.new_zeros(self.width)
        encoded = encoder.encode([], roots=task.trees) if encoded is None else encoded
        pool = sum((weight * encoded[tree] for tree, weight in zip(task.pool_trees, task.pool_weights)),
                   encoded[task.source].new_zeros(self.width))
        background = torch.stack([encoded[tree] for tree in task.discoveries]).mean(0) if task.discoveries else pool.new_zeros(self.width)
        return projection(torch.cat((encoded[task.source], pool, background,
                                     pool.new_tensor([task.pool_ic, math.log1p(len(task.discoveries))]))))

    def distributions(self, tree, step, task=None):
        # 对全部合法动作构造精确联合分布，训练重放需要相同的动作支持集合。
        """为当前公式、剩余编辑步数和任务构造完整合法动作分布。

        返回位置列表、分组动作、位置分布、各位置的条件替换分布和价值估计。
        策略与价值使用各自的编码器，step 须位于训练的去噪时间范围内。
        """
        if not 1 <= step <= self.noise_steps:
            raise ValueError("remaining-step index outside diffusion horizon")
        tree = self.edit_kernel.from_node(tree)
        grouped = defaultdict(list)
        for action in self.edit_kernel.edits(tree):
            grouped[action.path].append(action)
        paths = sorted(grouped)
        nodes = self.edit_kernel.positions(tree)
        all_actions = [a for p in paths for a in grouped[p]]
        task_trees = task.trees if self.use_task and task is not None else ()
        enc = self.actor_encoder.encode([tree], roots=[*(a.replacement for a in all_actions), *task_trees])
        time = self.actor_time.weight[step]
        node_features = torch.stack([enc[nodes[path]] for path in paths])
        if self.use_position:
            node_features = node_features + self.position_bias(paths)
        by_path = dict(zip(paths, node_features.unbind()))
        contexts = torch.stack([torch.cat((enc[tree], by_path[p], time)) for p in paths])
        position_logits = self.node_head(contexts).flatten()
        edit_inputs = torch.stack([torch.cat((enc[tree], by_path[a.path], time, enc[a.replacement])) for a in all_actions])
        logits = self.edit_head(edit_inputs).flatten()
        if self.use_task:
            query = self.task_embedding(task, self.actor_encoder, self.actor_task, enc)
            position_logits = position_logits + node_features @ query / math.sqrt(self.width)
            logits = logits + torch.stack([enc[a.replacement] for a in all_actions]) @ query / math.sqrt(self.width)
        position = Categorical(logits=position_logits)
        edits, cursor = [], 0
        for p in paths:
            size = len(grouped[p])
            edits.append(Categorical(logits=logits[cursor:cursor + size]))
            cursor += size
        critic = self.critic_encoder.encode([], roots=[tree, *task_trees])
        critic_time = self.critic_time.weight[step]
        if self.use_task:
            critic_time = critic_time + self.task_embedding(task, self.critic_encoder, self.critic_task, critic)
        value = self.value_head(torch.cat((critic[tree], critic_time))).squeeze()
        return paths, grouped, position, edits, value

    def evaluate_action(self, tree, step, action, task=None):
        """重算指定编辑的联合对数概率、状态价值和精确联合熵。

        动作概率由位置概率与该位置的条件替换概率相乘得到，任务条件参与原样重放。
        """
        tree = self.edit_kernel.from_node(tree)
        action = self.edit_kernel.native_edit(tree, action)
        paths, grouped, positions, edits, value = self.distributions(tree, step, task)
        i = paths.index(action.path)
        j = grouped[action.path].index(action)
        # log P(位置, 替换) = log P(位置) + log P(替换 | 位置)。
        logp = positions.logits[i] + edits[i].logits[j]
        # 联合熵包含位置熵，以及按位置概率加权的条件替换熵。
        entropy = positions.entropy() + sum(positions.probs[k] * d.entropy() for k, d in enumerate(edits))
        return logp, value, entropy

    def evaluate_batch(self, specifications):
        """批量共享公式编码，计算与逐条求值一致的联合概率分布及精确熵。"""
        prepared, forest, roots = [], [], []
        for specification in specifications:
            tree, step, action = specification[:3]
            tree = self.edit_kernel.from_node(tree)
            if action is not None:
                action = self.edit_kernel.native_edit(tree, action)
            elif self.stop_head is None:
                raise ValueError("STOP requires a learned lower stop action")
            actions = (tuple(self.edit_kernel.native_edit(tree, edit) for edit in specification[3])
                       if len(specification) > 3 and specification[3] else self.edit_kernel.edits(tree))
            task = specification[4] if len(specification) > 4 else None
            grouped = defaultdict(list)
            for edit in actions:
                grouped[edit.path].append(edit)
            paths, nodes = sorted(grouped), self.edit_kernel.positions(tree)
            ordered = [edit for path in paths for edit in grouped[path]]
            prepared.append((tree, step, action, paths, nodes, grouped, ordered, task))
            forest.append(tree)
            roots.extend(edit.replacement for edit in ordered)
            if self.use_task and task is not None:
                roots.extend(task.trees)
        encoded = self.actor_encoder.encode(forest, roots=roots)
        indices = {node: i for i, node in enumerate(encoded)}
        matrix = torch.stack(list(encoded.values()))
        position_roots, position_nodes, position_times = [], [], []
        edit_roots, edit_nodes, edit_times, edit_replacements = [], [], [], []
        position_tasks, edit_tasks = [], []
        position_paths, edit_paths = [], []
        stop_contexts = []
        for tree, step, action, paths, nodes, grouped, ordered, task in prepared:
            position_roots.extend([indices[tree]] * len(paths))
            position_nodes.extend(indices[nodes[path]] for path in paths)
            position_times.extend([step] * len(paths))
            edit_roots.extend([indices[tree]] * len(ordered))
            edit_nodes.extend(indices[nodes[a.path]] for a in ordered)
            edit_times.extend([step] * len(ordered))
            edit_replacements.extend(indices[a.replacement] for a in ordered)
            if self.use_position:
                position_paths.extend(paths)
                edit_paths.extend(action.path for action in ordered)
            if self.use_task:
                bias = self.task_embedding(task, self.actor_encoder, self.actor_task, encoded)
                position_tasks.extend([bias] * len(paths))
                edit_tasks.extend([bias] * len(ordered))
            if self.stop_head is not None:
                time = self.actor_time.weight[step]
                if self.use_task:
                    time = time + bias
                stop_contexts.append(torch.cat((encoded[tree], time)))
        def select(values):
            """按节点索引从共享编码矩阵取行，返回当前设备上的批量表示。"""
            return matrix[torch.tensor(values, device=matrix.device)]
        position_time = self.actor_time(torch.tensor(position_times, device=matrix.device))
        edit_time = self.actor_time(torch.tensor(edit_times, device=matrix.device))
        position_features, edit_features = select(position_nodes), select(edit_nodes)
        if self.use_position:
            position_features = position_features + self.position_bias(position_paths)
            edit_features = edit_features + self.position_bias(edit_paths)
        contexts = torch.cat((select(position_roots), position_features, position_time), dim=1)
        edit_inputs = torch.cat((select(edit_roots), edit_features,
                                edit_time, select(edit_replacements)), dim=1)
        position_logits = self.node_head(contexts).flatten()
        edit_logits = self.edit_head(edit_inputs).flatten()
        if self.use_task:
            position_logits = position_logits + (position_features * torch.stack(position_tasks)).sum(1) / math.sqrt(self.width)
            edit_logits = edit_logits + (select(edit_replacements) * torch.stack(edit_tasks)).sum(1) / math.sqrt(self.width)
        values = self.state_values([(tree, step, task) for tree, step, *_, task in prepared])
        stop_logits = self.stop_head(torch.stack(stop_contexts)).flatten() if stop_contexts else None
        result, cursor, edit_cursor = [], 0, 0
        for index, (_, _, action, paths, _, grouped, ordered, _) in enumerate(prepared):
            logits = position_logits[cursor:cursor + len(paths)]
            if stop_logits is not None:
                logits = torch.cat((logits, stop_logits[index:index+1]))
            positions = Categorical(logits=logits)
            cursor += len(paths)
            distributions = []
            for path in paths:
                size = len(grouped[path])
                distributions.append(Categorical(logits=edit_logits[edit_cursor:edit_cursor + size]))
                edit_cursor += size
            entropy = positions.entropy() + sum(positions.probs[k] * d.entropy() for k, d in enumerate(distributions))
            if action is None:
                logp = positions.logits[-1]
            else:
                i = paths.index(action.path)
                j = grouped[action.path].index(action)
                logp = positions.logits[i] + distributions[i].logits[j]
            result.append((logp, values[index], entropy))
        return result

    def state_values(self, states):
        """共享编码，批量估计给定公式状态的后续回报。"""
        roots = [tree for tree, _, _ in states]
        if self.use_task:
            roots.extend(tree for _, _, task in states if task is not None for tree in task.trees)
        critic = self.critic_encoder.encode([], roots=roots)
        contexts = []
        for tree, step, task in states:
            time = self.critic_time.weight[step]
            if self.use_task:
                time = time + self.task_embedding(task, self.critic_encoder, self.critic_task, critic)
            contexts.append(torch.cat((critic[tree], time)))
        return self.value_head(torch.stack(contexts)).flatten()

    def rows(self, rows):
        """从轨迹行提取状态、动作、合法动作集合和任务，批量返回 PPO 所需的概率、价值及熵。"""
        return self.evaluate_batch([(r["tree"], r["step"], r["action"], r.get("legal_actions"), r.get("task")) for r in rows])

    def _sampling_distributions(self, tree, step, query, legal_actions=None, *, allow_stop=True):
        """保持采样时的编码批次，构造位置分布及指定位置的条件替换分布。"""
        grouped = defaultdict(list)
        if legal_actions is not None:
            for action in legal_actions:
                grouped[action.path].append(action)
            paths = sorted(grouped)
        else:
            paths = self.edit_kernel.edit_paths(tree)
        nodes = self.edit_kernel.positions(tree)
        enc = self.actor_encoder.encode([tree])
        time = self.actor_time.weight[step]
        node_features = torch.stack([enc[nodes[path]] for path in paths])
        if self.use_position:
            node_features = node_features + self.position_bias(paths)
        contexts = torch.stack([torch.cat((enc[tree], feature, time)) for feature in node_features])
        position_logits = self.node_head(contexts).flatten()
        if self.use_task:
            position_logits = position_logits + node_features @ query / math.sqrt(self.width)
        if self.stop_head is not None and allow_stop:
            stop_time = time + query if self.use_task else time
            position_logits = torch.cat((position_logits, self.stop_head(torch.cat((enc[tree], stop_time)))))

        def replacements_at(index):
            """仅编码当前位置的替换，与实际采样使用相同矩阵形状及计算顺序。"""
            choices = grouped[paths[index]] if legal_actions is not None else self.edit_kernel.edits_at(tree, paths[index])
            replacements = self.actor_encoder.encode([], roots=[action.replacement for action in choices])
            inputs = torch.stack([torch.cat((enc[tree], node_features[index], time, replacements[action.replacement]))
                                  for action in choices])
            logits = self.edit_head(inputs).flatten()
            if self.use_task:
                logits = logits + torch.stack([replacements[action.replacement] for action in choices]) @ query / math.sqrt(self.width)
            return choices, Categorical(logits=logits)

        ordered = tuple(action for path in paths for action in grouped[path]) if legal_actions is not None else None
        return paths, Categorical(logits=position_logits), replacements_at, ordered

    def _replay_action(self, row):
        """按录制时的批次布局重建位置及条件替换分布。"""
        tree = self.edit_kernel.from_node(row["tree"])
        actions = tuple(self.edit_kernel.native_edit(tree, edit)
                        for edit in (row.get("legal_actions") or self.edit_kernel.edits(tree)))
        query = self.task_embedding(row.get("task"), self.actor_encoder, self.actor_task) if self.use_task else None
        paths, positions, replacements_at, _ = self._sampling_distributions(tree, row["step"], query, actions)
        if row["action"] is None:
            if self.stop_head is None:
                raise ValueError("STOP requires a learned lower stop action")
            logp = positions.logits[-1]
        else:
            action = self.edit_kernel.native_edit(tree, row["action"])
            index = paths.index(action.path)
            choices, edits = replacements_at(index)
            logp = positions.logits[index] + edits.logits[choices.index(action)]
        return tree, logp, positions, replacements_at, len(paths)

    def replay_logp(self, row):
        """不消耗随机数，只复核所选动作的联合概率。"""
        return self._replay_action(row)[1]

    def replay_row(self, row):
        """用原采样布局求可微的概率、价值和联合熵。"""
        tree, logp, positions, replacements_at, count = self._replay_action(row)
        entropy = positions.entropy()
        for index in range(count):
            _, edits = replacements_at(index)
            entropy = entropy + positions.probs[index] * edits.entropy()
        bias = self.task_embedding(row.get("task"), self.critic_encoder, self.critic_task) if self.use_task else None
        return logp, self._sampling_value(tree, row["step"], bias), entropy

    def _sampling_value(self, tree, step, bias):
        """按采样布局独立编码当前公式和任务条件。"""
        critic = self.critic_encoder.encode([], roots=[tree])[tree]
        time = self.critic_time.weight[step]
        if bias is not None:
            time = time + bias
        return self.value_head(torch.cat((critic, time))).squeeze()

    @torch.no_grad()
    def sample(self, tree, remaining=None, record=True, task=None, *, time_edits=False, allow_stop=True):
        # 从扰动后的表达式开始，按剩余步数逐步生成反向编辑轨迹。
        """执行反向编辑，返回终点和可选轨迹。学习 STOP 时一直编辑到模型选择停止。

        每步先采样位置，再采样该处替换，整个过程沿用同一任务条件。
        record 启用时保存旧概率、价值与合法动作，关闭时按选中位置枚举替换以节省计算。
        """
        if self.stop_head is not None and allow_stop:
            if remaining is not None:
                raise ValueError("learned STOP does not use a fixed editor horizon")
            # 学习何时停止的策略只看当前公式和任务，时间嵌入保持固定。
            steps = repeat(1)
        else:
            remaining = self.noise_steps if remaining is None else remaining
            if not 0 <= remaining <= self.noise_steps:
                raise ValueError("remaining diffusion steps invalid")
            steps = range(remaining, 0, -1)
        tree = self.edit_kernel.from_node(tree)
        if self.edit_backend == "rpn":
            self.edit_kernel.validate(tree)
        rows = []
        query = self.task_embedding(task, self.actor_encoder, self.actor_task) if self.use_task else None
        critic_bias = self.task_embedding(task, self.critic_encoder, self.critic_task) if self.use_task and record else None
        for step in steps:
            check_budget()
            actions = self.edit_kernel.edits(tree) if record else None
            paths, positions, replacements_at, ordered = self._sampling_distributions(
                tree, step, query, actions, allow_stop=allow_stop)
            if self.stop_head is not None and allow_stop and not bool(positions.probs[-1] > 0):
                raise FloatingPointError("learned STOP probability underflowed to zero")
            i = int(positions.sample())
            action, logp = None, positions.logits[i]
            if i < len(paths):
                choices, edits = replacements_at(i)
                j = int(edits.sample())
                action, logp = choices[j], logp + edits.logits[j]
            if record:
                # 保存旧策略概率、价值和动作前状态，PPO 更新时按原条件重放。
                value = self._sampling_value(tree, step, critic_bias)
                row = {"tree": tree, "step": step, "action": action,
                       "logp": float(logp), "value": float(value),
                       "legal_actions": ordered}
                if self.edit_backend == "rpn" and action is not None:
                    row["token_edit"] = {"start": action.start, "stop": action.stop,
                                         "replacement_tokens": action.replacement.tokens}
                if task is not None:
                    row["task"] = task
            # 原生 RPN 在内核中直接替换词元区间，轨迹保存其起止位置。
            if action is not None:
                tree = (self.edit_kernel.apply(tree, action) if self.edit_backend == "rpn"
                        else replace(tree, action.path, action.replacement))
            if record:
                row["after"] = tree
                if time_edits:
                    row["edit_completed_at"] = perf_counter()
                rows.append(row)
            if action is None:
                break
        # 在金融评价和存储边界转换为项目通用 Node，内部编辑状态由后端管理。
        return self.edit_kernel.to_node(tree), rows

    def as_node(self, program):
        """在金融评价或日志记录的边界，将轨迹状态转换为通用公式节点。"""
        return self.edit_kernel.to_node(program)

    def corrupt(self, tree, steps, rng):
        """用指定随机数生成器执行正向扰动，返回扰动公式和各步逆编辑标签。

        实际扰动由配置的编辑内核完成，接口输出转换为通用 Node 和 Edit。
        """
        noisy, trace = self.edit_kernel.corrupt(self.edit_kernel.from_node(tree), steps, rng)
        return self.edit_kernel.to_node(noisy), [(self.edit_kernel.to_node(program), step,
                                                self.edit_kernel.legacy_edit(action)) for program, step, action in trace]

    def row(self, row):
        """按单条轨迹记录重算下层动作的联合对数概率、价值和熵。"""
        return self.evaluate_action(row["tree"], row["step"], row["action"], row.get("task"))


@dataclass(frozen=True)
class Observation:
    # 上层一次决策可见的公共短名单、候选统计、当前组合与剩余预算。
    ids: tuple[str, ...]
    trees: tuple[Node, ...]
    statistics: tuple[tuple[float, ...], ...]
    pool_trees: tuple[Node, ...]
    pool_weights: tuple[float, ...]
    pool_ic: float
    remaining_fraction: float = 1.0
    new_statistics: tuple[float, ...] = (0., 0., 0., 0., 0., 0., 1.)
    archive_trees: tuple[Node, ...] = ()


class UpperTower(nn.Module):
    # 分别编码候选和组合，再用集合汇总表示构建调度上下文。
    def __init__(self, grammar, width, use_archive=False, representation="ast", cost_context=False):
        """创建上层单个网络分支的公式编码器及候选、组合和 NEW 特征层。

        成本条件决定统计特征维数，档案条件决定是否编码已见公式背景。
        """
        super().__init__()
        self.encoder = (RPNEncoder if representation == "rpn" else TreeEncoder)(grammar, width)
        self.statistics_width = 9 if cost_context else 7
        self.candidate = mlp(width + self.statistics_width, width, width)
        self.member = mlp(width + 1, width, width)
        self.new = nn.Parameter(torch.zeros(width))
        self.new_summary = nn.Linear(self.statistics_width, width, bias=False)
        nn.init.zeros_(self.new_summary.weight)
        self.width = width
        self.use_archive = use_archive

    def forward(self, observation, enc=None):
        """将观测编码为候选表示矩阵和全局上下文向量。

        矩阵末行对应 NEW，上下文汇总候选、当前组合和可选档案，可复用传入的公式编码。
        """
        archive = observation.archive_trees if self.use_archive else ()
        enc = self.encoder.encode([], roots=[*observation.trees, *observation.pool_trees, *archive]) if enc is None else enc
        device = self.new.device
        new = self.new + self.new_summary(self.new.new_tensor(observation.new_statistics))
        if observation.trees:
            features = torch.cat((torch.stack([enc[tree] for tree in observation.trees]),
                                  torch.tensor(observation.statistics, dtype=torch.float32, device=device)), dim=1)
            candidates = torch.cat((self.candidate(features), new[None, :]), dim=0)
        else:
            candidates = new[None, :]
        if observation.pool_trees:
            members = torch.cat((torch.stack([enc[tree] for tree in observation.pool_trees]),
                                 self.new.new_tensor(observation.pool_weights)[:, None]), dim=1)
            pool = self.member(members).mean(0)
        else:
            pool = self.new.new_zeros(self.width)
        context = torch.cat((candidates.mean(0), pool, self.new.new_tensor([observation.pool_ic])))
        if self.use_archive:
            background = torch.stack([enc[tree] for tree in archive]).mean(0) if archive else self.new.new_zeros(self.width)
            context = torch.cat((context, background, self.new.new_tensor([math.log1p(len(archive))])))
        return candidates, context


class UpperPolicy(nn.Module):
    # 上层输出候选起点和 NEW 的概率，策略与价值分别使用独立网络。
    def __init__(self, grammar, width=32, use_archive=False, representation="ast", cost_context=False):
        """创建上层起点调度策略，分别构建策略分支、价值分支及输出头。

        representation 选择公式编码方式，档案和成本条件控制额外观测特征。
        """
        super().__init__()
        if representation not in {"ast", "rpn"}:
            raise ValueError("upper representation must be ast or rpn")
        self.width, self.use_archive = width, use_archive
        self.representation, self.cost_context = representation, cost_context
        self.actor = UpperTower(grammar, width, use_archive, representation, cost_context)
        self.critic = UpperTower(grammar, width, use_archive, representation, cost_context)
        archive_width = width + 1 if use_archive else 0
        self.score = mlp(3 * width + 2 + archive_width, width, 1)
        self.value_head = mlp(2 * width + 2 + archive_width, width, 1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """载入上层参数，旧检查点新增的成本或档案输入列补零后交给 PyTorch 校验。"""
        state_dict = dict(state_dict)
        if self.cost_context:
            for tower in ("actor", "critic"):
                for suffix in ("candidate.0.weight", "new_summary.weight"):
                    key = f"{tower}.{suffix}"
                    weight, target = state_dict[key], self.state_dict()[key]
                    if weight.shape[1] + 2 == target.shape[1]:
                        state_dict[key] = torch.cat((weight, weight.new_zeros((weight.shape[0], 2))), dim=1)
        if self.use_archive:
            for key, columns in (("score.0.weight", self.score[0].in_features),
                                 ("value_head.0.weight", self.value_head[0].in_features)):
                weight = state_dict[key]
                if weight.shape[1] + self.width + 1 == columns:
                    state_dict[key] = torch.cat((weight, weight.new_zeros((weight.shape[0], self.width + 1))), dim=1)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def budget_context(self, observation, context):
        # 剩余预算比例同时进入策略打分和价值估计。
        """在上下文约定位置插入剩余预算比例，返回供策略头和价值头使用的向量。"""
        boundary = 2 * self.width + 1
        return torch.cat((context[:boundary], context.new_tensor([observation.remaining_fraction]), context[boundary:]))

    def value(self, observation, context):
        """将价值分支的上下文与剩余预算合并，返回当前调度状态的标量价值估计。"""
        return self.value_head(self.budget_context(observation, context)).squeeze()

    def logits(self, observation, candidates, context):
        """拼接各候选表示与共享预算上下文，返回短名单及 NEW 的未归一化分数。"""
        context = self.budget_context(observation, context)
        return self.score(torch.cat((candidates, context.expand(len(candidates), -1)), dim=1)).flatten()

    def distribution(self, observation):
        """分别调用策略和价值分支，返回起点选择的分类分布及状态价值。"""
        candidates, context = self.actor(observation)
        logits = self.logits(observation, candidates, context)
        _, critic_context = self.critic(observation)
        return Categorical(logits=logits), self.value(observation, critic_context)

    @torch.no_grad()
    def sample(self, observation):
        # 采样一次起点选择，并保存 PPO 所需的旧对数概率和旧价值。
        """采样一次起点调度，返回观测、动作索引、旧对数概率和旧价值。

        短名单末尾的额外索引代表 NEW，用于从文法先验另取起点。
        """
        distribution, value = self.distribution(observation)
        index = int(distribution.sample())
        return {"observation": observation, "index": index, "logp": float(distribution.logits[index]), "value": float(value)}

    def row(self, row):
        """按保存的观测和起点索引重算单条上层动作的对数概率、价值及熵。"""
        distribution, value = self.distribution(row["observation"])
        return distribution.logits[row["index"]], value, distribution.entropy()

    replay_row = row

    def rows(self, rows):
        """共享一批观测中重复公式的编码，逐行返回上层动作对数概率、价值及熵。

        策略与价值各自编码，保留两套独立参数的计算路径。
        """
        forest = [tree for row in rows for tree in (*row["observation"].trees, *row["observation"].pool_trees,
                  *(row["observation"].archive_trees if self.use_archive else ()))]
        actor = self.actor.encoder.encode([], roots=forest)
        critic = self.critic.encoder.encode([], roots=forest)
        result = []
        for row in rows:
            candidates, context = self.actor(row["observation"], actor)
            logits = self.logits(row["observation"], candidates, context)
            distribution = Categorical(logits=logits)
            _, critic_context = self.critic(row["observation"], critic)
            value = self.value(row["observation"], critic_context)
            result.append((distribution.logits[row["index"]], value, distribution.entropy()))
        return result
