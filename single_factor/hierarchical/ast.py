"""Finite typed grammar, ordered ASTs, and an explicitly reversible edit kernel."""
from __future__ import annotations

import ast as python_ast
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, cached_property
from itertools import accumulate
import math
import random

from adapters.operator_registry import get_operator, generation_operator_names, load_alphagen_operator_map
from ..expression import NamedFeature
from .._vendor.alphagen.data.expression import Constant
from .._vendor.alphagen.data.expression import RollingOperator, PairRollingOperator
from .io import digest


@dataclass(frozen=True)
class Node:
    symbol: str
    children: tuple["Node", ...] = ()
    value: float | None = None

    def to_dict(self):
        return {"symbol": self.symbol, "value": self.value,
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, data):
        return cls(str(data["symbol"]), tuple(cls.from_dict(c) for c in data.get("children", [])), data.get("value"))

    @lru_cache(maxsize=32768)
    def text(self):
        if self.symbol == "constant":
            return repr(float(self.value))
        if self.symbol == "window":
            return str(int(self.value))
        return self.symbol if not self.children else f"{self.symbol}({','.join(c.text() for c in self.children)})"

    def canonical(self):
        children = [c.canonical() for c in self.children]
        if self.symbol in {"add", "mul", "max", "min"}:
            children.sort(key=repr)
        return [self.symbol, self.value, children]

    @cached_property
    def node_count(self):
        return 1 + sum(child.node_count for child in self.children)

    @cached_property
    def depth(self):
        return 1 + max((child.depth for child in self.children), default=0)

    @cached_property
    def id(self):
        return digest(self.canonical())

    @cached_property
    def _structural_hash(self):
        return hash((self.symbol, self.children, self.value))

    def __hash__(self):
        return self._structural_hash

    def __reduce__(self):
        # Python string hashes differ between spawned processes. Never pickle
        # a cached process-local hash alongside an immutable tree.
        return Node, (self.symbol, self.children, self.value)


def walk(tree, path=()):
    yield path, tree
    for i, child in enumerate(tree.children):
        yield from walk(child, path + (i,))


def replace(tree, path, replacement):
    if not path:
        return replacement
    children = list(tree.children)
    children[path[0]] = replace(children[path[0]], path[1:], replacement)
    return Node(tree.symbol, tuple(children), tree.value)


@lru_cache(maxsize=8192)
def constant_value(node):
    """Only identities that are constant on every finite input are folded."""
    return fold_constant(node.symbol, node.value,
                         constant_value(node.children[0]) if node.children else None,
                         node.symbol == "sub" and node.children[0] == node.children[1])


def fold_constant(symbol, value, first_constant, equal_operands=False):
    """Local semantic rule shared by tree and postfix implementations."""
    if symbol == "constant":
        return float(value)
    if symbol == "sub" and equal_operands:
        return 0.0
    if first_constant is not None and symbol in {"ts_std", "ts_var", "delta"}:
        return 0.0
    if first_constant is not None and symbol in {"ts_mean", "delay"}:
        return first_constant
    if first_constant is not None and symbol == "abs":
        return abs(first_constant)
    return None


@dataclass(frozen=True)
class Edit:
    path: tuple[int, ...]
    replacement: Node


class Grammar:
    def __init__(self, fields, operators=None, windows=(1, 5, 10, 20, 40),
                 constants=(-1.0, -0.5, 0.0, 0.5, 1.0, 2.0), max_nodes=15,
                 max_depth=8, max_lookback=100, strict_types=True):
        self.fields = tuple(fields)
        self.operators = tuple(get_operator(x).canonical_name for x in
                               (generation_operator_names() if operators is None else operators))
        self.windows, self.constants = tuple(windows), tuple(map(float, constants))
        self.max_nodes, self.max_depth, self.max_lookback = max_nodes, max_depth, max_lookback
        self.strict_types = strict_types
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must be nonempty and unique")
        if not self.operators or len(set(self.operators)) != len(self.operators):
            raise ValueError("operators must be nonempty and unique")
        if not self.windows or any(type(w) is not int or w < 1 for w in self.windows):
            raise ValueError("windows must be positive integers")
        if not self.constants or not all(math.isfinite(c) for c in self.constants):
            raise ValueError("constants must be finite")
        if max_nodes < 3 or max_depth < 2 or max_lookback < 1:
            raise ValueError("insufficient AST budgets")
        self.definitions = {x: get_operator(x) for x in self.operators}
        self.classes = load_alphagen_operator_map()
        self.windowed = {name: issubclass(self.classes[name], (RollingOperator, PairRollingOperator)) for name in self.operators}
        self.atoms = tuple(Node(f) for f in self.fields) + tuple(Node("constant", value=c) for c in self.constants)
        self._cache = OrderedDict()

    def config(self):
        return {k: list(getattr(self, k)) if isinstance(getattr(self, k), tuple) else getattr(self, k)
                for k in ("fields", "operators", "windows", "constants", "max_nodes", "max_depth", "max_lookback", "strict_types")}

    def from_node(self, tree):
        return tree

    def to_node(self, tree):
        return tree

    def positions(self, tree):
        return dict(walk(tree))

    def native_edit(self, tree, action):
        return action

    def legacy_edit(self, action):
        return action

    @lru_cache(maxsize=8192)
    def dimension(self, node):
        return self.infer_dimension(node.symbol, node.value, tuple(self.dimension(c) for c in node.children),
                                    tuple(c.symbol for c in node.children), tuple(c.value for c in node.children))

    @lru_cache(maxsize=8192)
    def infer_dimension(self, symbol, value, dims, child_symbols, child_values):
        """Infer one operator's units without choosing an editing representation."""
        if symbol in self.fields:
            if symbol in {"open", "close", "high", "low", "ycp", "vwap"} or symbol.startswith(("ma", "ema")):
                return (1, 0)
            return (0, 1) if symbol == "volume" else (1, 1) if symbol == "amount" else (0, 0)
        if symbol == "constant":
            if value not in self.constants or dims:
                raise ValueError("constant outside finite grammar")
            return (0, 0)
        if symbol == "window":
            if value not in self.windows or dims:
                raise ValueError("window outside finite grammar")
            return None
        definition = self.definitions.get(symbol)
        if definition is None or len(dims) != definition.arity:
            raise ValueError("unknown operator or wrong arity")
        if self.windowed[symbol]:
            if child_symbols[-1] != "window" or any(d is None for d in dims[:-1]):
                raise ValueError("time-series windows must be final arguments")
            if symbol in {"ts_corr", "ts_rank", "ts_arg_max", "ts_arg_min"}:
                return (0, 0)
            if symbol == "ts_covariance":
                return tuple(a + b for a, b in zip(dims[0], dims[1]))
            if symbol == "ts_var":
                return tuple(2 * a for a in dims[0])
            return dims[0]
        if any(d is None for d in dims):
            raise ValueError("window cannot be an arithmetic operand")
        if len(dims) == 1:
            if symbol == "inverse":
                return tuple(-d for d in dims[0])
            return (0, 0) if symbol in {"log", "sign", "rank", "zscore", "scale", "quantile", "is_nan", "not_"} else dims[0]
        if symbol == "if_else":
            if self.strict_types and dims[1] != dims[2]:
                raise ValueError("conditional branches have incompatible units")
            return dims[1]
        if symbol in {"equal", "not_equal", "greater", "greater_equal", "less", "less_equal", "and_", "or_"}:
            return (0, 0)
        if symbol in {"add", "sub", "max", "min"}:
            if self.strict_types and dims[0] != dims[1]:
                raise ValueError("incompatible conceptual units")
            return dims[0]
        if symbol == "div" and child_symbols[1] == "constant" and child_values[1] == 0.0:
            raise ValueError("known zero denominator")
        return tuple(a + (-b if symbol == "div" else b) for a, b in zip(dims[0], dims[1]))

    def validate_operands(self, symbol, constants, window=None):
        if symbol in {"ts_std", "ts_var", "ts_corr", "ts_covariance"} and window < 2:
            raise ValueError("sample statistic requires at least two observations")
        if symbol == "div" and constants[1] == 0:
            raise ValueError("denominator is identically zero")
        if symbol == "ts_corr" and any(value is not None for value in constants[:2]):
            raise ValueError("correlation with a constant operand is undefined")

    @lru_cache(maxsize=8192)
    def lookback(self, node):
        back = max((self.lookback(c) for c in node.children), default=0)
        if node.symbol in self.definitions and self.windowed[node.symbol]:
            window = int(node.children[-1].value)
            back += window if node.symbol in {"delay", "delta"} else window - 1
        return back

    def validate(self, tree):
        self.dimension(tree)
        nodes = list(walk(tree))
        if len(nodes) > self.max_nodes or max(len(p) for p, _ in nodes) + 1 > self.max_depth:
            raise ValueError("AST size/depth budget exceeded")
        if self.lookback(tree) > self.max_lookback:
            raise ValueError("AST lookback budget exceeded")
        if not any(n.symbol in self.fields for _, n in nodes) or tree.symbol == "window":
            raise ValueError("factor must contain a field")
        for _, node in nodes:
            if node.symbol in {"ts_std", "ts_var", "ts_corr", "ts_covariance", "div"}:
                self.validate_operands(node.symbol, tuple(constant_value(c) for c in node.children),
                                       node.children[-1].value if self.windowed.get(node.symbol) else None)
        return tree

    def parse(self, text):
        def read(item, is_window=False):
            if isinstance(item, python_ast.Name) and item.id in self.fields:
                return Node(item.id)
            if isinstance(item, (python_ast.Constant, python_ast.UnaryOp)):
                value = python_ast.literal_eval(item)
                return Node("window" if is_window else "constant", value=value)
            if isinstance(item, python_ast.Call) and isinstance(item.func, python_ast.Name) and not item.keywords:
                definition = get_operator(item.func.id)
                return Node(definition.canonical_name, tuple(read(c, self.windowed.get(definition.canonical_name, False) and j == len(item.args) - 1)
                            for j, c in enumerate(item.args)))
            raise ValueError("unsupported formula syntax")
        return self.validate(read(python_ast.parse(text.replace("$", ""), mode="eval").body))

    def compile(self, tree):
        self.validate(tree)
        def build(node):
            if node.symbol in self.fields:
                return NamedFeature(node.symbol)
            if node.symbol == "constant":
                return Constant(node.value)
            if node.symbol == "window":
                return int(node.value)
            return self.classes[node.symbol](*(build(c) for c in node.children))
        return build(tree)

    def _edit_candidates(self, node):
        if node.symbol in self.fields:
            yield from (Node(field) for field in self.fields)
        elif node.symbol == "window":
            yield from (Node("window", value=window) for window in self.windows)
        elif node.symbol == "constant":
            yield from (Node("constant", value=value) for value in self.constants)
        elif node.symbol in self.definitions:
            old = self.definitions[node.symbol]
            yield from (Node(name, node.children) for name, definition in self.definitions.items()
                        if definition.arity == old.arity and definition.category == old.category)
            non_windows = [child for child in node.children if child.symbol != "window"]
            for i, child in enumerate(non_windows):
                if len(non_windows) <= 2 and all(other in self.atoms for j, other in enumerate(non_windows) if i != j):
                    yield child
        if node.symbol != "window":
            for name, definition in self.definitions.items():
                windowed = self.windowed[name]
                operands = definition.arity - int(windowed)
                for window in self.windows if windowed else (None,):
                    tail = (Node("window", value=window),) if windowed else ()
                    if operands == 1:
                        yield Node(name, (node,) + tail)
                    elif operands == 2:
                        for atom in self.atoms:
                            yield Node(name, (node, atom) + tail)
                            yield Node(name, (atom, node) + tail)

    def _iter_edits(self, tree, path, node):
        if not path:
            yield Edit((), tree)  # The root hold action retains its probability.
        for candidate in self._edit_candidates(node):
            if candidate == node:
                continue
            if tree.node_count - node.node_count + candidate.node_count > self.max_nodes:
                continue
            if len(path) + candidate.depth > self.max_depth:
                continue
            try:
                self.validate(replace(tree, path, candidate))
            except (ValueError, KeyError):
                continue
            yield Edit(path, candidate)

    @lru_cache(maxsize=512)
    def raw_candidates(self, node):
        return tuple(sorted(set(candidate for candidate in self._edit_candidates(node) if candidate != node),
                            key=lambda candidate: candidate.text()))

    def edit_paths(self, tree):
        """Exact position mask, stopping after the first legal edit per node."""
        if tree in self._cache:
            self._cache.move_to_end(tree)
            return tuple(sorted({action.path for action in self._cache[tree]}))
        return tuple(sorted(path for path, node in walk(tree)
                            if next(self._iter_edits(tree, path, node), None) is not None))

    def edits_at(self, tree, path):
        """Materialize the replacement choices for one already selected node."""
        if tree in self._cache:
            self._cache.move_to_end(tree)
            return tuple(action for action in self._cache[tree] if action.path == path)
        node = dict(walk(tree))[path]
        return tuple(sorted(set(self._iter_edits(tree, path, node)), key=lambda action: action.replacement.text()))

    def edits(self, tree):
        if tree in self._cache:
            self._cache.move_to_end(tree)
            return self._cache[tree]
        result = {action for path, node in walk(tree) for action in self._iter_edits(tree, path, node)}
        ordered = tuple(sorted(result, key=lambda a: (a.path, a.replacement.text())))
        self._cache[tree] = ordered
        while len(self._cache) > 1 and (len(self._cache) > 128 or sum(map(len, self._cache.values())) > 40000):
            self._cache.popitem(last=False)
        return ordered

    def corrupt(self, tree, steps, rng):
        trace = []
        for step in range(1, steps + 1):
            nodes = dict(walk(tree))
            # Uniform rejection over a deduplicated superset gives exactly a
            # uniform draw over legal non-hold edits. The fallback is uniform
            # over that same set. Cache warmth never changes RNG consumption.
            paths = sorted(nodes)
            candidates = [self.raw_candidates(nodes[path]) for path in paths]
            cumulative = tuple(accumulate(map(len, candidates)))
            total = cumulative[-1] if cumulative else 0
            edit = None
            for _ in range(64 if total else 0):
                rank = rng.randrange(total)
                index = bisect_right(cumulative, rank)
                offset = cumulative[index - 1] if index else 0
                proposed = Edit(paths[index], candidates[index][rank - offset])
                try:
                    self.validate(replace(tree, proposed.path, proposed.replacement))
                except (ValueError, KeyError):
                    continue
                edit = proposed
                break
            if edit is None:
                choices = [action for action in self.edits(tree) if action.replacement != nodes[action.path]]
                edit = rng.choice(choices) if choices else Edit((), tree)
            original = nodes[edit.path]
            changed = replace(tree, edit.path, edit.replacement)
            inverse = Edit(edit.path, original)
            changed_node = dict(walk(changed))[edit.path]
            inverse_supported = (not edit.path and changed == tree or
                                 any(candidate == original for candidate in self._edit_candidates(changed_node)))
            if not inverse_supported:
                raise AssertionError("forward kernel lacks inverse support")
            if replace(changed, inverse.path, inverse.replacement) != tree:
                raise AssertionError("inverse label does not restore previous AST")
            trace.append((changed, step, inverse))
            tree = changed
        return tree, trace

    def sample(self, rng, exclude=(), max_attempts=256):
        """Recursive grammar prior, not selection from a fixed formula template list."""
        excluded = set(exclude)
        self.last_prior_attempts = 0
        for _ in range(max_attempts):
            self.last_prior_attempts += 1
            def generate(budget, depth):
                if budget < 3 or depth >= self.max_depth - 1 or rng.random() < 0.3:
                    return rng.choice(self.atoms)
                name = rng.choice(self.operators)
                definition = self.definitions[name]
                n = definition.arity - int(self.windowed[name])
                available = budget - 1 - int(self.windowed[name])
                if available < n:
                    return rng.choice(self.atoms)
                if n == 1:
                    children = (generate(available, depth + 1),)
                elif n == 2:
                    split = rng.randint(1, available - 1)
                    children = (generate(split, depth + 1), generate(available - split, depth + 1))
                else:
                    cuts = sorted(rng.sample(range(1, available), n - 1))
                    lengths = [b - a for a, b in zip([0] + cuts, cuts + [available])]
                    children = tuple(generate(size, depth + 1) for size in lengths)
                if self.windowed[name]:
                    children += (Node("window", value=rng.choice(self.windows)),)
                return Node(name, children)
            node = generate(rng.randint(1, self.max_nodes), 0)
            try:
                self.validate(node)
            except ValueError:
                continue
            if node.id not in excluded:
                return node
        raise RuntimeError("grammar prior exhausted its bounded unique-start attempts")
