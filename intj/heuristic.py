"""Reading and lowering of `@triton.heuristics` functions.

Finds which names each heuristic reads, and lowers the ones keyed at run time
to the steps entry.c.jinja writes as C.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import inspect
import linecache
import types
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from . import _imports
from .render.key import Operand, Step

# Python 3.8 wraps a subscript's index in ast.Index; later versions do not.
_INDEX: Any = getattr(ast, "Index", ())


class HeuristicError(ValueError):
    """A heuristic function is outside the subset intj reads."""


@dataclasses.dataclass(frozen=True)
class Heuristic:
    """One parsed `@triton.heuristics` entry."""

    name: str  # the value it assigns
    inputs: tuple[str, ...]  # names it reads, first-use order
    arg: str  # its one parameter
    expr: ast.expr = dataclasses.field(compare=False, hash=False, repr=False)
    fn: Any = dataclasses.field(compare=False, hash=False, repr=False)
    where: str = ""  # file:line, for messages


def parse_heuristic(name: str, fn: object) -> Heuristic:
    """Locates `fn`'s source without calling it, and lists the names it reads.

    Raises:
        HeuristicError: `fn` is not a one-argument lambda or def with a single
            return that reads its argument only as `arg["name"]`.
    """
    if type(fn) is not types.FunctionType:
        raise HeuristicError(f"heuristic {name!r} must be a lambda or def")
    code = fn.__code__
    if code.co_freevars:
        raise HeuristicError(
            f"heuristic {name!r} cannot reference free variables {code.co_freevars}"
        )
    varargs = code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
    if (
        code.co_argcount != 1
        or code.co_posonlyargcount
        or code.co_kwonlyargcount
        or varargs
        or fn.__defaults__
    ):
        raise HeuristicError(f"heuristic {name!r} must take exactly one argument")
    node = _locate(fn)
    where = f"{code.co_filename}:{node.lineno}"
    if isinstance(node, ast.Lambda):
        expr = node.body
        arg = node.args.args[0].arg
    else:
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and type(body[0].value.value) is str
        ):
            body = body[1:]
        if (
            len(body) != 1
            or not isinstance(body[0], ast.Return)
            or body[0].value is None
        ):
            raise HeuristicError(f"{where}: heuristic {name!r} must be a single return")
        expr = body[0].value
        arg = (node.args.posonlyargs + node.args.args)[0].arg
    inputs: dict[str, None] = {}
    subscripts = 0
    for sub in ast.walk(expr):
        key = _subscript_key(sub, arg, where)
        if key is not None:
            inputs[key] = None
            subscripts += 1
    uses = sum(isinstance(sub, ast.Name) and sub.id == arg for sub in ast.walk(expr))
    if uses != subscripts:
        raise HeuristicError(
            f'{where}: heuristic {name!r} may use {arg!r} only as {arg}["name"] with a string literal'
        )
    return Heuristic(name, tuple(inputs), arg, expr, fn, where)


def _subscript_key(node: ast.AST, arg: str, where: str) -> str | None:
    """Returns `"name"` if `node` is `arg["name"]`, or None if it does not index `arg`.

    Raises:
        HeuristicError: `node` indexes `arg` with anything but a string literal.
    """
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == arg
    ):
        return None
    index: Any = node.slice
    if isinstance(index, _INDEX):
        index = index.value
    if isinstance(index, ast.Constant) and type(index.value) is str:
        return index.value
    raise HeuristicError(f"{where}: index {arg!r} with a string literal")


def _locate(fn: types.FunctionType) -> ast.Lambda | ast.FunctionDef:
    """Returns the one lambda or def in `fn`'s file that compiles to `fn`'s code.

    Several lambdas can share a line, and inspect.getsource returns lines, not
    nodes; compiling each candidate and comparing bytecode picks the right one.
    """
    code = fn.__code__
    lines = linecache.getlines(code.co_filename, fn.__globals__)
    if not lines:
        raise HeuristicError(f"heuristic {fn.__name__!r} needs available Python source")
    try:
        tree = ast.parse("".join(lines))
    except SyntaxError as error:
        raise HeuristicError(f"cannot parse {code.co_filename}") from error
    matches: list[ast.Lambda | ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            first = node.lineno
        elif isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        else:
            continue
        if first == code.co_firstlineno and _same_code(node, code, strict=False):
            matches.append(node)
    if len(matches) > 1:
        matches = [node for node in matches if _same_code(node, code, strict=True)]
    if len(matches) != 1:
        raise HeuristicError(
            f"cannot locate the source of heuristic {fn.__name__!r} at "
            f"{code.co_filename}:{code.co_firstlineno}"
        )
    return matches[0]


def _same_code(
    node: ast.Lambda | ast.FunctionDef, code: types.CodeType, strict: bool
) -> bool:
    """Returns whether `node` compiles to `code`.

    The bytecode is only compared when `strict`: a file that imports `triton`
    compiles `triton.cdiv(...)` in another form than the node does alone, so it
    breaks ties only.
    """
    if isinstance(node, ast.Lambda):
        compiled = compile(ast.Expression(body=node), code.co_filename, "eval")
    else:
        compiled = compile(
            ast.Module(body=[node], type_ignores=[]), code.co_filename, "exec"
        )
    return any(
        isinstance(const, types.CodeType)
        and (not strict or const.co_code == code.co_code)
        and const.co_names == code.co_names
        and const.co_varnames == code.co_varnames
        and const.co_consts == code.co_consts
        for const in compiled.co_consts
    )


_BUILTIN = {"min": builtins.min, "max": builtins.max}
_TRITON = ("cdiv", "next_power_of_2")
#: tensor method -> takes one literal index
_TENSOR_METHODS = {
    "numel": False,
    "size": True,
    "stride": True,
    "dim": False,
    "element_size": False,
    "is_contiguous": False,
    "data_ptr": False,
}
_ARITH: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.FloorDiv: "floor",
    ast.Mod: "mod",
}
_COMPARE: dict[type[ast.cmpop], str] = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}


@dataclasses.dataclass(frozen=True)
class Source:
    """Where a heuristic input comes from in C."""

    kind: Literal["arg", "fixed", "dep", "comp"]
    index: int = 0  # args index, dep slot or comp slot
    value: object = None  # fixed: the baked value


def lower(
    computed: Sequence[tuple[Heuristic, int]],
    sources: Mapping[str, Source],
    levels: int,
) -> tuple[tuple[Step, ...], ...]:
    """Lowers `computed` to the steps of each level's `intj_tuned_level_<n>`.

    Each level writes (value, is_bool) pairs to `comp`.

    `computed` is level-major; its position is the comp slot.  Every read of a
    tensor attribute sits behind the short-circuit it has in Python.
    """
    out: list[tuple[Step, ...]] = []
    for level in range(levels):
        steps: list[Step] = []
        serial = [0]  # temp numbers: deterministic, since steps are in the digest
        for slot, (h, at) in enumerate(computed):
            if at != level:
                continue
            value, is_bool = _Emitter(h, sources, serial).emit(h.expr, steps)
            steps.append(
                Step(
                    "assign",
                    Operand("comp", 2 * slot),
                    Operand("comp", 2 * slot + 1),
                    value,
                    is_bool,
                )
            )
        out.append(tuple(steps))
    return tuple(out)


_ZERO = Operand("int", 0)
_ONE = Operand("int", 1)


class _Emitter:
    """Lowers one heuristic's expression to steps appended to a caller's list."""

    def __init__(
        self,
        h: Heuristic,
        sources: Mapping[str, Source],
        serial: list[int],
    ):
        self.h, self.sources, self.serial = h, sources, serial
        namespace = h.fn.__globals__
        self.builtins_ok = {
            name: namespace.get(name, getattr(builtins, name)) is fn
            for name, fn in _BUILTIN.items()
        }
        _imports.ensure_triton_imported()
        self.triton_ok = namespace.get("triton") is _imports.triton

    def fail(self, node: ast.AST, message: str) -> HeuristicError:
        return HeuristicError(f"{self.h.where}: heuristic {self.h.name!r}: {message}")

    def temp(self) -> Operand:
        self.serial[0] += 1
        return Operand("temp", self.serial[0])

    def pair(self, steps: list[Step]) -> tuple[Operand, Operand]:
        v, b = self.temp(), self.temp()
        steps.append(Step("declare", v, b, _ZERO, _ZERO))
        return v, b

    def fixed_kind(self, steps: list[Step], is_bool: bool) -> tuple[Operand, Operand]:
        """Returns a temp for a result whose kind the syntax decides."""
        v = self.temp()
        steps.append(Step("declare", v, a=_ZERO))
        return v, _ONE if is_bool else _ZERO

    def input(self, node: ast.expr) -> tuple[str, Source] | None:
        key = _subscript_key(node, self.h.arg, self.h.where)
        if key is not None and key not in self.sources:
            # a dynamic value: C reads the heuristic's inputs from the kernel arguments
            raise self.fail(node, f"cannot read dynamic option {key!r}")
        return None if key is None else (key, self.sources[key])

    def tensor_arg(self, node: ast.expr, what: str) -> tuple[str, int]:
        found = self.input(node)
        if found is None or found[1].kind != "arg":
            raise self.fail(node, f"{what} needs a caller tensor argument")
        return found[0], found[1].index

    def emit(self, node: ast.expr, steps: list[Step]) -> tuple[Operand, Operand]:
        """Appends the steps for `node` and returns its (value, is_bool) operands.

        Raises:
            HeuristicError: `node` is outside the subset intj lowers.
        """
        found = self.input(node)
        if found is not None:
            name, source = found
            if source.kind == "arg":
                v, b = self.pair(steps)
                steps.append(
                    Step("heur_arg", v, b, Operand("arg", source.index), name=name)
                )
                return v, b
            if source.kind == "fixed":
                if type(source.value) not in (int, bool):
                    raise self.fail(node, f"baked {name!r} must be an int or bool")
                is_bool = type(source.value) is bool
                return Operand("int", int(cast(int, source.value))), (
                    _ONE if is_bool else _ZERO
                )
            return (
                Operand(source.kind, 2 * source.index),
                Operand(source.kind, 2 * source.index + 1),
            )
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            value = int(cast(int, node.value))
            if not -(1 << 63) < value < (1 << 63):
                raise self.fail(node, "integer literal is outside int64")
            return Operand("int", value), _ONE if type(node.value) is bool else _ZERO
        if isinstance(node, ast.BinOp):
            op = _ARITH.get(type(node.op))
            if op is None:
                raise self.fail(node, "unsupported operator")
            left, _ = self.emit(node.left, steps)
            right, _ = self.emit(node.right, steps)
            v, b = self.fixed_kind(steps, False)
            steps.append(Step("checked", v, a=left, b=right, fn=op))
            return v, b
        if isinstance(node, ast.UnaryOp):
            operand, _ = self.emit(node.operand, steps)
            v, b = self.fixed_kind(steps, isinstance(node.op, ast.Not))
            if isinstance(node.op, ast.Not):
                steps.append(Step("not", v, a=operand))
            elif isinstance(node.op, ast.USub):
                steps.append(Step("checked", v, a=_ZERO, b=operand, fn="sub"))
            elif isinstance(node.op, ast.UAdd):
                steps.append(Step("assign", v, a=operand))
            else:
                raise self.fail(node, "unsupported operator")
            return v, b
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise self.fail(node, "chained comparisons are not supported")
            op, right_node = node.ops[0], node.comparators[0]
            v, b = self.fixed_kind(steps, True)
            if isinstance(op, (ast.Is, ast.IsNot)):
                found = self.input(node.left)
                if found is None or not (
                    isinstance(right_node, ast.Constant) and right_node.value is None
                ):
                    raise self.fail(node, "`is` only compares an input with None")
                source = found[1]
                tested = (
                    Operand("arg", source.index)
                    if source.kind == "arg"
                    else _ONE
                    if source.kind == "fixed" and source.value is None
                    else _ZERO
                )
                steps.append(
                    Step("is_none", v, a=tested, negate=isinstance(op, ast.IsNot))
                )
                return v, b
            compare = _COMPARE.get(type(op))
            if compare is None:
                raise self.fail(node, "unsupported comparison")
            if _is_dtype(node.left) or _is_dtype(right_node):
                if compare not in ("eq", "ne") or not (
                    _is_dtype(node.left) and _is_dtype(right_node)
                ):
                    raise self.fail(
                        node, ".dtype only compares with another .dtype by == or !="
                    )
                left, right = (
                    self.dtype(node.left, steps),
                    self.dtype(right_node, steps),
                )
            else:
                left, _ = self.emit(node.left, steps)
                right, _ = self.emit(right_node, steps)
            steps.append(Step("compare", v, a=left, b=right, fn=compare))
            return v, b
        if isinstance(node, ast.BoolOp):
            v, b = self.pair(steps)
            first, first_bool = self.emit(node.values[0], steps)
            steps.append(Step("assign", v, b, first, first_bool))
            # `a and b and c`: each later operand, nested in its predecessor's test
            blocks: list[list[Step]] = []
            for value in node.values[1:]:
                then: list[Step] = []
                nv, nb = self.emit(value, then)
                then.append(Step("assign", v, b, nv, nb))
                blocks.append(then)
            nested: tuple[Step, ...] = ()
            for then in reversed(blocks):
                negate = isinstance(node.op, ast.Or)
                nested = (Step("if", a=v, negate=negate, then=(*then, *nested)),)
            steps += nested
            return v, b
        if isinstance(node, ast.IfExp):
            v, b = self.pair(steps)
            test, _ = self.emit(node.test, steps)
            yes: list[Step] = []
            yes_value, yes_bool = self.emit(node.body, yes)
            yes.append(Step("assign", v, b, yes_value, yes_bool))
            no: list[Step] = []
            no_value, no_bool = self.emit(node.orelse, no)
            no.append(Step("assign", v, b, no_value, no_bool))
            steps.append(Step("if", a=test, then=tuple(yes), orelse=tuple(no)))
            return v, b
        if isinstance(node, ast.Subscript) and _is_attr(node.value, "shape"):
            index = node.slice
            if isinstance(index, _INDEX):
                index = index.value  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 ast.Index
            return self.tensor_call(
                cast(ast.Attribute, node.value).value, "size", [index], node, steps
            )
        if isinstance(node, ast.Call) and not node.keywords:
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id in _BUILTIN
                and len(node.args) == 2
            ):
                if not self.builtins_ok[func.id]:
                    raise self.fail(node, f"{func.id} must resolve to the builtin")
                a, a_bool = self.emit(node.args[0], steps)
                c, c_bool = self.emit(node.args[1], steps)
                v, b = self.pair(steps)
                # Python returns the first argument unless the second beats it.
                fn = "lt" if func.id == "min" else "gt"
                steps.append(Step("pick", v, b, a, a_bool, c, c_bool, fn=fn))
                return v, b
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "triton"
                and func.attr in _TRITON
            ):
                if not self.triton_ok:
                    raise self.fail(node, "triton must resolve to the triton module")
                arity = 2 if func.attr == "cdiv" else 1
                if len(node.args) != arity:
                    raise self.fail(
                        node, f"triton.{func.attr} takes {arity} argument(s)"
                    )
                operands = [self.emit(arg, steps)[0] for arg in node.args]
                v, b = self.fixed_kind(steps, False)
                fn = "cdiv" if func.attr == "cdiv" else "next_pow2"
                second = operands[1] if arity == 2 else None
                steps.append(Step("checked", v, a=operands[0], b=second, fn=fn))
                return v, b
            if isinstance(func, ast.Attribute) and func.attr in _TENSOR_METHODS:
                return self.tensor_call(func.value, func.attr, node.args, node, steps)
        raise self.fail(
            node,
            f"unsupported {type(node).__name__} call"
            if isinstance(node, ast.Call)
            else f"unsupported {type(node).__name__}",
        )

    def tensor_call(
        self,
        target: ast.expr,
        method: str,
        args: Sequence[ast.expr],
        node: ast.AST,
        steps: list[Step],
    ) -> tuple[Operand, Operand]:
        """Appends a call of tensor method `method` on the caller tensor `target`."""
        name, index = self.tensor_arg(target, f".{method}")
        takes_index = _TENSOR_METHODS[method]
        if len(args) != int(takes_index):
            raise self.fail(node, f".{method} takes {int(takes_index)} argument(s)")
        literal = None
        if takes_index:
            arg = args[0]
            if not (isinstance(arg, ast.Constant) and type(arg.value) is int):
                raise self.fail(node, f".{method} needs a literal int index")
            literal = Operand("int", int(arg.value))
        v, b = self.fixed_kind(steps, method == "is_contiguous")
        steps.append(
            Step(
                "heur_tensor",
                v,
                a=Operand("arg", index),
                b=literal,
                fn=method,
                name=name,
            )
        )
        return v, b

    def dtype(self, node: ast.expr, steps: list[Step]) -> Operand:
        name, index = self.tensor_arg(cast(ast.Attribute, node).value, ".dtype")
        v = self.temp()
        steps.append(Step("declare", v))
        steps.append(Step("heur_dtype", v, a=Operand("arg", index), name=name))
        return v


def _is_attr(node: ast.expr, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == attr


def _is_dtype(node: ast.expr) -> bool:
    return _is_attr(node, "dtype")
