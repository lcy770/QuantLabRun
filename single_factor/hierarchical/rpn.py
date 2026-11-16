"""带类型约束的可逆后缀表达式编辑，采用原生词元掩码与区间替换。"""
# 原生 RPN 编辑内核，使用栈扫描定位完整子表达式并直接替换词元区间。
from collections import OrderedDict
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache, cached_property
from itertools import accumulate
from typing import NamedTuple

from .ast import Edit, Node, fold_constant
from .io import digest


def token_vocabulary(grammar):
    """按文法中的字段、算子、窗口和常量构建去重词表，返回名称到词元编号的映射。"""
    tokens = [*grammar.fields, *grammar.operators,
              *(f"window:{w}" for w in grammar.windows),
              *(f"constant:{float(c)}" for c in grammar.constants)]
    return {token: i for i, token in enumerate(dict.fromkeys(tokens))}


@dataclass(frozen=True)
class RPNProgram:
    # 不可变词元序列携带词表身份，便于校验、去重和缓存。
    tokens: tuple[int, ...]
    vocabulary_id: str

    @cached_property
    def _structural_hash(self):
        """计算并缓存词元序列与词表身份共同确定的进程内哈希值。"""
        return hash((self.tokens, self.vocabulary_id))

    def __hash__(self):
        """返回缓存的结构哈希，供公式去重、字典索引和编辑缓存使用。"""
        return self._structural_hash

    def __reduce__(self):
        # 新启动的采集进程会重新计算本进程使用的字符串哈希。
        """返回序列化重建所需的类型和参数，使接收进程按自身哈希种子重新计算哈希。"""
        return RPNProgram, (self.tokens, self.vocabulary_id)


@dataclass(frozen=True)
class TokenEdit:
    # path 表示出现位置，start/stop 是左闭右开的词元区间。
    path: tuple[int, ...]
    start: int
    stop: int
    replacement: RPNProgram


class Summary(NamedTuple):
    # 局部语义摘要用于检查量纲、深度、回看长度、字段数量和常量退化。
    dimension: tuple[int, int] | None
    depth: int
    lookback: int
    fields: int
    constant: float | None


class Span(NamedTuple):
    start: int
    stop: int
    children: tuple[int, ...]
    summary: Summary


class RPNKernel:
    """与 Grammar 共用算子语义的有限后缀表达式编辑内核。

    路径标识子表达式的出现位置，供策略的位置嵌入使用。
    编辑通过栈扫描得到的词元区间、语义信息和元组拼接完成。
    在公共评价及存储边界构造 Node 对象。
    """
    def __init__(self, grammar):
        """由文法预建词表、算子元数、原子表达式和编辑缓存，供原生词元操作复用。"""
        self.grammar = grammar
        self.vocabulary = token_vocabulary(grammar)
        self.vocabulary_id = digest(self.vocabulary)
        self.symbols, self.values, self.arities = [], [], []
        for token in self.vocabulary:
            if token.startswith("window:"):
                symbol, value = "window", int(token.split(":", 1)[1])
            elif token.startswith("constant:"):
                symbol, value = "constant", float(token.split(":", 1)[1])
            else:
                symbol, value = token, None
            self.symbols.append(symbol)
            self.values.append(value)
            self.arities.append(grammar.definitions[symbol].arity if symbol in grammar.definitions else 0)
        self.atoms = tuple(self.program((self.vocabulary[token],)) for token in
                           (*grammar.fields, *(f"constant:{float(c)}" for c in grammar.constants)))
        self.atom_tokens = {program.tokens for program in self.atoms}
        self._cache = OrderedDict()

    def program(self, tokens):
        """将词元序列封装为携带当前词表身份的不可变 RPNProgram，语义校验由后续接口负责。"""
        return RPNProgram(tuple(tokens), self.vocabulary_id)

    @lru_cache(maxsize=8192)
    def from_node(self, tree):
        # 将公共 Node 转为后缀词元，已属于本词表的 RPNProgram 可直接复用。
        """将 Node 递归转换为后缀词元序列，并缓存转换结果。

        输入已是 RPNProgram 时校验词表身份后直接返回。
        """
        if isinstance(tree, RPNProgram):
            if tree.vocabulary_id != self.vocabulary_id:
                raise ValueError("RPN program uses a different token vocabulary")
            return tree
        token = (f"constant:{float(tree.value)}" if tree.symbol == "constant" else
                 f"window:{int(tree.value)}" if tree.symbol == "window" else tree.symbol)
        return self.program(tuple(index for child in tree.children for index in self.from_node(child).tokens)
                            + (self.vocabulary[token],))

    @lru_cache(maxsize=8192)
    def to_node(self, program):
        # 在求值、日志和旧接口边界恢复 Node，编辑过程中使用原生词元状态。
        """校验后缀程序并用操作数栈还原通用 Node，供评价、存储和兼容接口使用。"""
        self.analyze(program)
        stack = []
        for token in program.tokens:
            arity = self.arities[token]
            children = tuple(stack[-arity:]) if arity else ()
            if arity:
                del stack[-arity:]
            stack.append(Node(self.symbols[token], children, self.values[token]))
        return stack[0]

    def legacy_edit(self, action):
        """把原生区间编辑转换为包含相同出现路径和 Node 替换项的通用 Edit。"""
        return Edit(action.path, self.to_node(action.replacement))

    def native_edit(self, program, action):
        """把通用编辑的路径映射为源程序中的词元区间，返回 TokenEdit。

        输入已经是 TokenEdit 时直接复用，区间与词表的匹配在执行编辑时检查。
        """
        if isinstance(action, TokenEdit):
            return action
        spans, positions = self.analyze(program)
        span = spans[positions[action.path]]
        return TokenEdit(action.path, span.start, span.stop, self.from_node(action.replacement))

    def _summary(self, token, summaries, child_tokens):
        """从子表达式摘要和词元序列提取根信息，组合当前节点语义并识别相同操作数的减法。"""
        return self.compose(token, summaries, tuple(tokens[-1] for tokens in child_tokens),
                            self.symbols[token] == "sub" and child_tokens[0] == child_tokens[1])

    @lru_cache(maxsize=16384)
    def compose(self, token, summaries, child_roots, equal_operands=False):
        """等价局部状态共用类型与数值合法性检查结果。"""
        try:
            return self._compose(token, summaries, child_roots, equal_operands)
        except (ValueError, KeyError):
            return None

    def _compose(self, token, summaries, child_roots, equal_operands):
        """调用文法检查量纲与操作数约束，计算深度、回看长度、字段数和常量摘要。

        参数不满足算子语义时抛出异常，合法时返回 Summary。
        """
        symbol, value = self.symbols[token], self.values[token]
        child_symbols = tuple(self.symbols[child] for child in child_roots)
        child_values = tuple(self.values[child] for child in child_roots)
        dimension = self.grammar.infer_dimension(symbol, value, tuple(s.dimension for s in summaries),
                                                  child_symbols, child_values)
        constants = tuple(s.constant for s in summaries)
        window = child_values[-1] if self.grammar.windowed.get(symbol) else None
        self.grammar.validate_operands(symbol, constants, window)
        back = max((s.lookback for s in summaries), default=0)
        if window is not None:
            back += window if symbol in {"delay", "delta"} else window - 1
        constant = fold_constant(symbol, value, constants[0] if constants else None,
                                 equal_operands)
        return Summary(dimension, 1 + max((s.depth for s in summaries), default=0), back,
                       int(symbol in self.grammar.fields) + sum(s.fields for s in summaries), constant)

    @lru_cache(maxsize=16384)
    def summary(self, tokens):
        """缓存局部后缀表达式的语义摘要，同时缓存被拒绝的子表达式结果。"""
        try:
            arity = self.arities[tokens[-1]]
            children, cursor = [], len(tokens) - 1
            for _ in range(arity):
                stop, needed = cursor, 1
                while needed:
                    cursor -= 1
                    if cursor < 0:
                        return None
                    needed += self.arities[tokens[cursor]] - 1
                children.append(tokens[cursor:stop])
            if cursor != 0:
                return None
            children.reverse()
            summaries = tuple(self.summary(child) for child in children)
            if any(item is None for item in summaries):
                return None
            return self._summary(tokens[-1], summaries, tuple(children))
        except (ValueError, KeyError, IndexError):
            return None

    @lru_cache(maxsize=8192)
    def analyze(self, program):
        # 从左到右用操作数栈检查表达式，记录每个子表达式的完整区间与出现路径。
        """扫描后缀程序并检查词表、操作数栈及局部语义，返回区间表与出现路径索引。

        每个区间记录完整子表达式边界、子节点索引和摘要，最终栈须归约为一个表达式。
        """
        if program.vocabulary_id != self.vocabulary_id or not program.tokens:
            raise ValueError("invalid or incompatible RPN program")
        spans, stack = [], []
        for index, token in enumerate(program.tokens):
            if type(token) is not int or not 0 <= token < len(self.arities):
                raise ValueError("invalid RPN token index")
            arity = self.arities[token]
            if len(stack) < arity:
                raise ValueError("RPN operand stack underflow")
            children = tuple(stack[-arity:]) if arity else ()
            if arity:
                del stack[-arity:]
            start = spans[children[0]].start if children else index
            summary = self.summary(program.tokens[start:index + 1])
            if summary is None:
                raise ValueError("invalid RPN operator semantics")
            spans.append(Span(start, index + 1, children, summary))
            stack.append(index)
        if len(stack) != 1:
            raise ValueError("RPN program must reduce to one expression")
        positions, pending = {}, [((), len(spans) - 1)]
        while pending:
            path, index = pending.pop()
            positions[path] = index
            pending.extend((path + (role,), child) for role, child in reversed(tuple(enumerate(spans[index].children))))
        return tuple(spans), positions

    def validate(self, program):
        # 限制长度、深度、回看范围，并要求最终表达式包含行情字段且量纲合法。
        """检查完整因子的词元数、深度、回看预算、字段存在性与量纲，合法时返回原程序。"""
        spans, _ = self.analyze(program)
        root = spans[-1].summary
        if len(program.tokens) > self.grammar.max_nodes or root.depth > self.grammar.max_depth:
            raise ValueError("RPN size/depth budget exceeded")
        if root.lookback > self.grammar.max_lookback:
            raise ValueError("RPN lookback budget exceeded")
        if not root.fields or root.dimension is None:
            raise ValueError("RPN factor must contain a field")
        return program

    @lru_cache(maxsize=4096)
    def positions(self, program):
        """根据区间分析结果，返回每个出现路径对应的原生后缀子表达式。"""
        spans, positions = self.analyze(program)
        return {path: self.program(program.tokens[spans[index].start:spans[index].stop])
                for path, index in positions.items()}

    @lru_cache(maxsize=8192)
    def text(self, program):
        """将合法后缀程序格式化为函数式文本，供稳定排序与展示使用。"""
        stack = []
        for token in program.tokens:
            symbol, value, arity = self.symbols[token], self.values[token], self.arities[token]
            children = stack[-arity:] if arity else []
            if arity:
                del stack[-arity:]
            stack.append(repr(float(value)) if symbol == "constant" else str(int(value)) if symbol == "window"
                         else f"{symbol}({','.join(children)})" if arity else symbol)
        return stack[0]

    def _candidates(self, program):
        # 有限动作空间包括字段/常量/窗口替换、同类算子替换、展开和简化。
        """为局部后缀程序生成有限替换候选，涵盖原子替换、同类算子替换、展开和简化。

        这里只生成局部候选，放回完整公式后的合法性由后续步骤检查。
        """
        tokens, grammar = program.tokens, self.grammar
        symbol = self.symbols[tokens[-1]]
        if symbol in grammar.fields:
            yield from (self.program((self.vocabulary[field],)) for field in grammar.fields)
        elif symbol == "window":
            yield from (self.program((self.vocabulary[f"window:{w}"],)) for w in grammar.windows)
        elif symbol == "constant":
            yield from (self.program((self.vocabulary[f"constant:{float(c)}"],)) for c in grammar.constants)
        else:
            old = grammar.definitions[symbol]
            yield from (self.program(tokens[:-1] + (self.vocabulary[name],)) for name, definition in grammar.definitions.items()
                        if definition.arity == old.arity and definition.category == old.category)
            spans, _ = self.analyze(program)
            children = [tokens[spans[c].start:spans[c].stop] for c in spans[-1].children
                        if self.symbols[tokens[spans[c].stop - 1]] != "window"]
            for index, child in enumerate(children):
                if len(children) <= 2 and all(other in self.atom_tokens for j, other in enumerate(children) if index != j):
                    yield self.program(child)
        if symbol == "window":
            return
        for name, definition in grammar.definitions.items():
            windowed = grammar.windowed[name]
            operands = definition.arity - int(windowed)
            for window in grammar.windows if windowed else (None,):
                tail = ((self.vocabulary[f"window:{window}"],) if windowed else ()) + (self.vocabulary[name],)
                if operands == 1:
                    yield self.program(tokens + tail)
                elif operands == 2:
                    for atom in self.atoms:
                        yield self.program(tokens + atom.tokens + tail)
                        yield self.program(atom.tokens + tokens + tail)

    def _legal_replacement(self, program, path, replacement):
        """复用未改动区间，重新检查替换表达式及其祖先的合法性。"""
        # 沿替换位置向根检查语义，复用未改动区间的摘要，降低合法性检查成本。
        spans, positions = self.analyze(program)
        target = spans[positions[path]]
        if len(program.tokens) - (target.stop - target.start) + len(replacement.tokens) > self.grammar.max_nodes:
            return False
        summary = self.summary(replacement.tokens)
        if summary is None:
            return False
        if len(path) + summary.depth > self.grammar.max_depth:
            return False
        current = replacement.tokens
        for depth in range(len(path) - 1, -1, -1):
            parent = spans[positions[path[:depth]]]
            role = path[depth]
            children = tuple(current if i == role else program.tokens[spans[c].start:spans[c].stop]
                             for i, c in enumerate(parent.children))
            token = program.tokens[parent.stop - 1]
            current = tuple(t for child in children for t in child) + (token,)
            summary = self.summary(current)
            if summary is None:
                return False
        return bool(summary.fields and summary.dimension is not None and summary.depth <= self.grammar.max_depth
                    and summary.lookback <= self.grammar.max_lookback)

    def _iter_edits(self, program, path, node):
        """生成指定出现位置的合法区间编辑，根位置额外提供保持原公式的动作。

        逐项检查替换后的完整公式约束，产出 TokenEdit。
        """
        spans, positions = self.analyze(program)
        span = spans[positions[path]]
        if not path:
            yield TokenEdit((), 0, len(program.tokens), program)
        for candidate in self._candidates(node):
            if candidate == node:
                continue
            try:
                legal = self._legal_replacement(program, path, candidate)
            except (ValueError, KeyError):
                continue
            if legal:
                yield TokenEdit(path, span.start, span.stop, candidate)

    @lru_cache(maxsize=512)
    def raw_candidates(self, node):
        """缓存局部替换候选，去掉原表达式、去重并按文本排序，返回候选元组。"""
        return tuple(sorted(set(candidate for candidate in self._candidates(node) if candidate != node), key=self.text))

    def edit_paths(self, program):
        """返回至少存在一个合法动作的出现路径，优先复用完整动作缓存。"""
        if program in self._cache:
            self._cache.move_to_end(program)
            return tuple(sorted({action.path for action in self._cache[program]}))
        return tuple(sorted(path for path, node in self.positions(program).items()
                            if next(self._iter_edits(program, path, node), None) is not None))

    def edits_at(self, program, path):
        """返回指定出现路径的全部合法编辑，去重排序并优先复用动作缓存。"""
        if program in self._cache:
            self._cache.move_to_end(program)
            return tuple(action for action in self._cache[program] if action.path == path)
        return tuple(sorted(set(self._iter_edits(program, path, self.positions(program)[path])),
                            key=lambda action: self.text(action.replacement)))

    def edits(self, program):
        # 合法动作按位置和替换内容确定性排序，缓存同时受公式数与动作总数限制。
        """枚举完整程序所有合法编辑，按位置及替换文本排序并缓存结果。

        缓存按最近使用顺序淘汰，容量同时受公式数量和动作总量约束。
        """
        if program in self._cache:
            self._cache.move_to_end(program)
            return self._cache[program]
        actions = {action for path, node in self.positions(program).items() for action in self._iter_edits(program, path, node)}
        result = tuple(sorted(actions, key=lambda action: (action.path, self.text(action.replacement))))
        self._cache[program] = result
        while len(self._cache) > 1 and (len(self._cache) > 128 or sum(map(len, self._cache.values())) > 80000):
            self._cache.popitem(last=False)
        return result

    def apply(self, program, action):
        # 核验动作区间和词表后，直接拼接词元元组执行替换。
        """检查编辑区间及替换词表与源程序匹配，直接拼接词元返回新程序。

        动作的语义合法性由合法动作枚举保证，此处执行已经选定的区间替换。
        """
        spans, positions = self.analyze(program)
        span = spans[positions[action.path]]
        if (action.start, action.stop) != (span.start, span.stop) or action.replacement.vocabulary_id != self.vocabulary_id:
            raise ValueError("RPN edit span or vocabulary does not match source program")
        if action.start == 0 and action.stop == len(program.tokens):
            return action.replacement
        return self.program(program.tokens[:action.start] + action.replacement.tokens + program.tokens[action.stop:])

    def corrupt(self, program, steps, rng):
        # 正向扰动用于构造去噪任务，同时记录可还原前一状态的逆编辑标签。
        """执行指定步数的原生 RPN 正向扰动，返回终态和各步逆动作标签。

        优先抽样局部候选，失败后从合法编辑中选择，并核验每个逆动作受支持且能精确还原。
        """
        self.validate(program)
        trace = []
        for step in range(1, steps + 1):
            spans, positions = self.analyze(program)
            nodes = self.positions(program)
            paths = sorted(nodes)
            candidates = [self.raw_candidates(nodes[path]) for path in paths]
            cumulative = tuple(accumulate(map(len, candidates)))
            total = cumulative[-1] if cumulative else 0
            action = None
            for _ in range(64 if total else 0):
                rank = rng.randrange(total)
                index = bisect_right(cumulative, rank)
                offset = cumulative[index - 1] if index else 0
                path = paths[index]
                span = spans[positions[path]]
                proposed = TokenEdit(path, span.start, span.stop, candidates[index][rank - offset])
                if self._legal_replacement(program, proposed.path, proposed.replacement):
                    action = proposed
                    break
            if action is None:
                choices = [edit for edit in self.edits(program) if edit.replacement != nodes[edit.path]]
                action = rng.choice(choices) if choices else TokenEdit((), 0, len(program.tokens), program)
            original = self.positions(program)[action.path]
            changed = self.apply(program, action)
            changed_spans, changed_positions = self.analyze(changed)
            target = changed_spans[changed_positions[action.path]]
            inverse = TokenEdit(action.path, target.start, target.stop, original)
            changed_node = self.positions(changed)[action.path]
            inverse_supported = (not action.path and changed == program or
                                 any(candidate == original for candidate in self._candidates(changed_node)))
            # 监督标签必须属于当前合法编辑核，且执行逆动作后精确回到原表达式。
            if not inverse_supported or self.apply(changed, inverse) != program:
                raise AssertionError("native RPN forward kernel lacks inverse support")
            trace.append((changed, step, inverse))
            program = changed
        return program, trace
