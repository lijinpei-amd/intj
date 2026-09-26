"""Read `@triton.heuristics` functions: which names each one reads, and lower
the ones keyed at run time to C.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import inspect
import json
import linecache
import types
from collections.abc import Mapping, Sequence
from typing import Any, cast

import triton

# Python 3.8 wraps a subscript's index in ast.Index; later versions do not.
_INDEX: Any = getattr(ast, "Index", ())


class HeuristicError(ValueError):
    """A heuristic function is outside the subset intj reads."""


@dataclasses.dataclass(frozen=True)
class Heuristic:
    name: str  # the value it assigns
    inputs: tuple[str, ...]  # names it reads, first-use order
    arg: str  # its one parameter
    expr: ast.expr = dataclasses.field(compare=False, hash=False, repr=False)
    fn: Any = dataclasses.field(compare=False, hash=False, repr=False)
    where: str = ""  # file:line, for messages


def parse_heuristic(name: str, fn: object) -> Heuristic:
    """Locate `fn`'s source without calling it, and list the names it reads."""
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
    """The one lambda or def in `fn`'s file that compiles to `fn`'s code.

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
    """Whether `node` compiles to `code`.  The bytecode is only compared when
    `strict`: a file that imports `triton` compiles `triton.cdiv(...)` in
    another form than the node does alone, so it breaks ties only."""
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
_TRITON = {"cdiv": triton.cdiv, "next_power_of_2": triton.next_power_of_2}
#: tensor method -> takes one literal index
_TENSOR_METHODS = {
    "numel": False,
    "size": True,
    "stride": True,
    "dim": False,
    "element_size": False,
    "is_contiguous": False,
}
_ARITH: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.FloorDiv: "floor",
    ast.Mod: "mod",
}
_COMPARE: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


@dataclasses.dataclass(frozen=True)
class Source:
    """Where a heuristic input comes from in C."""

    kind: str  # "arg" | "fixed" | "dep" | "comp"
    index: int = 0  # args index, dep slot or comp slot
    value: object = None  # fixed: the baked value


def lower(
    computed: Sequence[tuple[Heuristic, int]],
    sources: Mapping[str, Source],
    levels: int,
) -> str:
    """One `intj_tuned_level_<n>` per level, writing (value, is_bool) pairs to `comp`.

    `computed` is level-major; its position is the comp slot.  Every read of a
    tensor attribute sits behind the short-circuit it has in Python.
    """
    out: list[str] = []
    for level in range(levels):
        lines = [
            f"static int intj_tuned_level_{level}(intj_state *st, PyObject *const *args, "
            "const int64_t *dep, int64_t *comp) {",
            "  (void)st; (void)args; (void)dep; (void)comp;",
        ]
        serial = [
            0
        ]  # temp names: deterministic, since the source is in the module digest
        for slot, (h, at) in enumerate(computed):
            if at != level:
                continue
            emitter = _Emitter(h, sources, lines, serial)
            value, is_bool = emitter.emit(h.expr, "  ")
            lines.append(
                f"  comp[{2 * slot}] = {value}; comp[{2 * slot + 1}] = {is_bool};"
            )
        lines += ["  return 0;", "}"]
        out.append("\n".join(lines))
    return "\n\n".join(out)


class _Emitter:
    def __init__(
        self,
        h: Heuristic,
        sources: Mapping[str, Source],
        lines: list[str],
        serial: list[int],
    ):
        self.h, self.sources, self.lines, self.serial = h, sources, lines, serial
        namespace = h.fn.__globals__
        self.builtins_ok = {
            name: namespace.get(name, getattr(builtins, name)) is fn
            for name, fn in _BUILTIN.items()
        }
        self.triton_ok = namespace.get("triton") is triton

    def fail(self, node: ast.AST, message: str) -> HeuristicError:
        return HeuristicError(f"{self.h.where}: heuristic {self.h.name!r}: {message}")

    def temp(self) -> str:
        self.serial[0] += 1
        return f"hv_{self.serial[0]}"

    def pair(self, indent: str) -> tuple[str, str]:
        v, b = self.temp(), self.temp()
        self.lines.append(f"{indent}int64_t {v} = 0, {b} = 0;")
        return v, b

    def fixed_kind(self, indent: str, is_bool: bool) -> tuple[str, str]:
        """A temp for a result whose kind the syntax decides."""
        v = self.temp()
        self.lines.append(f"{indent}int64_t {v} = 0;")
        return v, "1" if is_bool else "0"

    def input(self, node: ast.expr) -> tuple[str, Source] | None:
        key = _subscript_key(node, self.h.arg, self.h.where)
        return None if key is None else (key, self.sources[key])

    def tensor_arg(self, node: ast.expr, what: str) -> tuple[str, int]:
        found = self.input(node)
        if found is None or found[1].kind != "arg":
            raise self.fail(node, f"{what} needs a caller tensor argument")
        return found[0], found[1].index

    def emit(self, node: ast.expr, indent: str) -> tuple[str, str]:
        found = self.input(node)
        if found is not None:
            name, source = found
            if source.kind == "arg":
                v, b = self.pair(indent)
                self.lines.append(
                    f"{indent}if (intj_heur_arg(args[{source.index}], {json.dumps(name)}, &{v}, &{b}) != 0) return -1;"
                )
                return v, b
            if source.kind == "fixed":
                if type(source.value) not in (int, bool):
                    raise self.fail(node, f"baked {name!r} must be an int or bool")
                return str(int(cast(int, source.value))), "1" if type(
                    source.value
                ) is bool else "0"
            array = "dep" if source.kind == "dep" else "comp"
            return f"{array}[{2 * source.index}]", f"{array}[{2 * source.index + 1}]"
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            value = int(cast(int, node.value))
            if not -(1 << 63) < value < (1 << 63):
                raise self.fail(node, "integer literal is outside int64")
            return str(value), "1" if type(node.value) is bool else "0"
        if isinstance(node, ast.BinOp):
            op = _ARITH.get(type(node.op))
            if op is None:
                raise self.fail(node, "unsupported operator")
            left, _ = self.emit(node.left, indent)
            right, _ = self.emit(node.right, indent)
            v, b = self.fixed_kind(indent, False)
            self.lines.append(
                f"{indent}if (intj_grid_{op}({left}, {right}, &{v}) != 0) return -1;"
            )
            return v, b
        if isinstance(node, ast.UnaryOp):
            operand, operand_bool = self.emit(node.operand, indent)
            v, b = self.fixed_kind(indent, isinstance(node.op, ast.Not))
            if isinstance(node.op, ast.Not):
                self.lines.append(f"{indent}{v} = !{operand};")
            elif isinstance(node.op, ast.USub):
                self.lines.append(
                    f"{indent}if (intj_grid_sub(0, {operand}, &{v}) != 0) return -1;"
                )
            elif isinstance(node.op, ast.UAdd):
                self.lines.append(f"{indent}{v} = {operand};")
            else:
                raise self.fail(node, "unsupported operator")
            del operand_bool
            return v, b
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise self.fail(node, "chained comparisons are not supported")
            op, right_node = node.ops[0], node.comparators[0]
            v, b = self.fixed_kind(indent, True)
            if isinstance(op, (ast.Is, ast.IsNot)):
                found = self.input(node.left)
                if found is None or not (
                    isinstance(right_node, ast.Constant) and right_node.value is None
                ):
                    raise self.fail(node, "`is` only compares an input with None")
                name, source = found
                negate = "!" if isinstance(op, ast.IsNot) else ""
                test = (
                    f"args[{source.index}] == Py_None"
                    if source.kind == "arg"
                    else (
                        "1" if source.kind == "fixed" and source.value is None else "0"
                    )
                )
                del name
                self.lines.append(f"{indent}{v} = {negate}({test});")
                return v, b
            symbol = _COMPARE.get(type(op))
            if symbol is None:
                raise self.fail(node, "unsupported comparison")
            if _is_dtype(node.left) or _is_dtype(right_node):
                if symbol not in ("==", "!=") or not (
                    _is_dtype(node.left) and _is_dtype(right_node)
                ):
                    raise self.fail(
                        node, ".dtype only compares with another .dtype by == or !="
                    )
                left, right = (
                    self.dtype(node.left, indent),
                    self.dtype(right_node, indent),
                )
            else:
                left, _ = self.emit(node.left, indent)
                right, _ = self.emit(right_node, indent)
            self.lines.append(f"{indent}{v} = {left} {symbol} {right};")
            return v, b
        if isinstance(node, ast.BoolOp):
            v, b = self.pair(indent)
            first, first_bool = self.emit(node.values[0], indent)
            self.lines.append(f"{indent}{v} = {first}; {b} = {first_bool};")
            test = f"{v}" if isinstance(node.op, ast.And) else f"!{v}"
            depth = indent
            for value in node.values[1:]:
                self.lines.append(f"{depth}if ({test}) {{")
                inner = depth + "  "
                nv, nb = self.emit(value, inner)
                self.lines.append(f"{inner}{v} = {nv}; {b} = {nb};")
                depth = inner
            for _ in node.values[1:]:
                depth = depth[:-2]
                self.lines.append(f"{depth}}}")
            return v, b
        if isinstance(node, ast.IfExp):
            v, b = self.pair(indent)
            test, _ = self.emit(node.test, indent)
            self.lines.append(f"{indent}if ({test}) {{")
            yes, yes_bool = self.emit(node.body, indent + "  ")
            self.lines.append(f"{indent}  {v} = {yes}; {b} = {yes_bool};")
            self.lines.append(f"{indent}}} else {{")
            no, no_bool = self.emit(node.orelse, indent + "  ")
            self.lines.append(f"{indent}  {v} = {no}; {b} = {no_bool};")
            self.lines.append(f"{indent}}}")
            return v, b
        if isinstance(node, ast.Subscript) and _is_attr(node.value, "shape"):
            index = node.slice
            if isinstance(index, _INDEX):
                index = index.value  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 ast.Index
            return self.tensor_call(
                cast(ast.Attribute, node.value).value, "size", [index], node, indent
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
                a, a_bool = self.emit(node.args[0], indent)
                c, c_bool = self.emit(node.args[1], indent)
                v, b = self.pair(indent)
                op = "<" if func.id == "min" else ">"
                # Python returns the first argument unless the second beats it.
                self.lines.append(
                    f"{indent}if ({c} {op} {a}) {{ {v} = {c}; {b} = {c_bool}; }} else {{ {v} = {a}; {b} = {a_bool}; }}"
                )
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
                operands = [self.emit(arg, indent)[0] for arg in node.args]
                v, b = self.fixed_kind(indent, False)
                helper = (
                    "intj_grid_cdiv" if func.attr == "cdiv" else "intj_grid_next_pow2"
                )
                self.lines.append(
                    f"{indent}if ({helper}({', '.join(operands)}, &{v}) != 0) return -1;"
                )
                return v, b
            if isinstance(func, ast.Attribute) and func.attr in _TENSOR_METHODS:
                return self.tensor_call(func.value, func.attr, node.args, node, indent)
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
        indent: str,
    ) -> tuple[str, str]:
        name, index = self.tensor_arg(target, f".{method}")
        takes_index = _TENSOR_METHODS[method]
        if len(args) != int(takes_index):
            raise self.fail(node, f".{method} takes {int(takes_index)} argument(s)")
        literal = 0
        if takes_index:
            arg = args[0]
            if not (isinstance(arg, ast.Constant) and type(arg.value) is int):
                raise self.fail(node, f".{method} needs a literal int index")
            literal = int(arg.value)
        v, b = self.fixed_kind(indent, method == "is_contiguous")
        self.lines.append(
            f"{indent}if (intj_heur_tensor(args[{index}], st->tensor_type, st->param_type, "
            f"{json.dumps(name)}, {json.dumps(method)}, {int(takes_index)}, {literal}L, &{v}) != 0) return -1;"
        )
        return v, b

    def dtype(self, node: ast.expr, indent: str) -> str:
        name, index = self.tensor_arg(cast(ast.Attribute, node).value, ".dtype")
        v = self.temp()
        self.lines.append(f"{indent}int64_t {v};")
        self.lines.append(
            f"{indent}if (intj_heur_dtype(args[{index}], st->tensor_type, st->param_type, {json.dumps(name)}, &{v}) != 0) return -1;"
        )
        return v


def _is_attr(node: ast.expr, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == attr


def _is_dtype(node: ast.expr) -> bool:
    return _is_attr(node, "dtype")
