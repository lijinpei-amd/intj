"""Lower the supported annotated grid function subset to C."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import inspect
import json
import textwrap
import types
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import triton

if TYPE_CHECKING:
    from .launcher import Param

_BUILTIN_MIN = builtins.min
_TRITON_CDIV = triton.cdiv


class GridError(ValueError):
    """An annotated grid is outside the native subset."""


@dataclasses.dataclass(frozen=True)
class GridCode:
    extras: tuple[str, ...]
    source: str


def compile_grid(
    fn: object, params: Sequence[Param], baked: Mapping[int, object]
) -> GridCode:
    """Validate `fn` without executing it and emit one checked native evaluator."""
    if type(fn) is not types.FunctionType:
        raise GridError("grid_cpp requires a Python def function")
    if fn.__code__.co_freevars:
        raise GridError(
            f"grid_cpp cannot reference free variables {fn.__code__.co_freevars}"
        )
    try:
        raw, start_line = inspect.getsourcelines(fn)
        tree = ast.parse(textwrap.dedent("".join(raw)))
    except (OSError, IOError, SyntaxError) as error:
        raise GridError("grid_cpp requires available Python source") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise GridError("grid_cpp requires one Python def function")
    body = tree.body[0]

    def fail(node: ast.AST, message: str) -> GridError:
        line = start_line + getattr(node, "lineno", 1) - 1
        return GridError(f"{fn.__code__.co_filename}:{line}: {message}")

    if body.decorator_list or body.name != fn.__name__:
        raise fail(body, "grid_cpp source must be an undecorated def")
    signature = body.args
    ordered_args = (*signature.posonlyargs, *signature.args, *signature.kwonlyargs)
    if (
        signature.vararg
        or signature.kwarg
        or signature.defaults
        or any(default is not None for default in signature.kw_defaults)
        or fn.__defaults__
        or fn.__kwdefaults__
    ):
        raise fail(body, "grid_cpp parameters cannot have defaults, *args or **kwargs")
    if (
        tuple(arg.arg for arg in ordered_args)
        != fn.__code__.co_varnames[: len(ordered_args)]
    ):
        raise fail(body, "grid_cpp source does not match the function parameters")

    by_name = {param.name: param for param in params}
    kinds: dict[str, str] = {}
    extras: list[str] = []
    for arg in ordered_args:
        annotation = fn.__annotations__.get(arg.arg)
        kind = (
            "int"
            if annotation is int or type(annotation) is str and annotation == "int"
            else "bool"
            if annotation is bool or type(annotation) is str and annotation == "bool"
            else None
        )
        if arg.annotation is None or kind is None:
            raise fail(
                arg, f"grid_cpp parameter {arg.arg!r} needs an int or bool annotation"
            )
        kinds[arg.arg] = kind
        if arg in signature.kwonlyargs:
            if arg.arg in by_name:
                raise fail(
                    arg, f"grid_cpp extra {arg.arg!r} must not name a kernel parameter"
                )
            extras.append(arg.arg)
        elif arg.arg not in by_name:
            raise fail(
                arg, f"grid_cpp parameter {arg.arg!r} must name a kernel parameter"
            )
    local_names = set(kinds) | {
        node.id
        for node in ast.walk(body)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    builtin_namespace = getattr(fn, "__builtins__", None)
    if builtin_namespace is None:
        builtin_namespace = fn.__globals__.get("__builtins__", builtins)
    if isinstance(builtin_namespace, types.ModuleType):
        builtin_namespace = vars(builtin_namespace)
    builtin_min = (
        builtin_namespace.get("min") if isinstance(builtin_namespace, dict) else None
    )
    min_is_builtin = fn.__globals__.get("min", builtin_min) is _BUILTIN_MIN
    cdiv_is_triton = (
        fn.__globals__.get("triton") is triton and triton.cdiv is _TRITON_CDIV
    )

    lines = ["static int intj_eval_grid(PyObject *const *args, uint32_t dims[3]) {"]
    bound: dict[str, tuple[str, str]] = {}
    serial = 0

    def temp() -> str:
        nonlocal serial
        name = f"gv_{serial}"
        serial += 1
        return name

    for arg in ordered_args:
        name = arg.arg
        kind = kinds[name]
        var = temp()
        if name in extras:
            index = extras.index(name)
            lines.append(f"  int64_t {var};")
            lines.append(
                f"  if (intj_grid_input_{kind}(args[{index}], {json.dumps(name)}, &{var}) != 0) return -1;"
            )
        else:
            param = by_name[name]
            if param.call_index is not None:
                index = len(extras) + param.call_index
                lines.append(f"  int64_t {var};")
                lines.append(
                    f"  if (intj_grid_input_{kind}(args[{index}], {json.dumps(name)}, &{var}) != 0) return -1;"
                )
            elif param.annotation.bind_value is not None:
                raise fail(arg, f"grid_cpp cannot read bound parameter {name!r}")
            else:
                value = baked[param.index]
                if (
                    kind == "int"
                    and type(value) is not int
                    or kind == "bool"
                    and type(value) is not bool
                ):
                    raise fail(
                        arg, f"grid_cpp baked parameter {name!r} has the wrong type"
                    )
                scalar = cast(int, value)
                if kind == "int" and not -(1 << 63) <= scalar < (1 << 63):
                    raise fail(
                        arg, f"grid_cpp baked parameter {name!r} is outside int64"
                    )
                literal = "INT64_MIN" if scalar == -(1 << 63) else str(int(scalar))
                lines.append(f"  int64_t {var} = {literal};")
        bound[name] = var, kind

    def emit(node: ast.AST, indent: str = "  ") -> tuple[str, str]:
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            value = int(cast(int, node.value))
            if type(node.value) is int and not -(1 << 63) <= value < (1 << 63):
                raise fail(node, "grid_cpp integer literal is outside int64")
            var = temp()
            lines.append(
                f"{indent}int64_t {var} = {'INT64_MIN' if value == -(1 << 63) else value};"
            )
            return var, "bool" if type(node.value) is bool else "int"
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound:
                raise fail(node, f"grid_cpp unknown or global name {node.id!r}")
            return bound[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            # Python parses -2**63 as a unary operation on a positive literal.
            if (
                isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and type(node.operand.value) is int
                and node.operand.value == 1 << 63
            ):
                var = temp()
                lines.append(f"{indent}int64_t {var} = INT64_MIN;")
                return var, "int"
            operand, _ = emit(node.operand, indent)
            if isinstance(node.op, ast.UAdd):
                return operand, "int"
            var = temp()
            lines.append(f"{indent}int64_t {var};")
            lines.append(
                f"{indent}if (intj_grid_sub(0, {operand}, &{var}) != 0) return -1;"
            )
            return var, "int"
        if isinstance(node, ast.BinOp):
            operations: dict[type[ast.operator], str] = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.FloorDiv: "floor",
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise fail(node, "grid_cpp unsupported arithmetic operator")
            left, _ = emit(node.left, indent)
            right, _ = emit(node.right, indent)
            var = temp()
            lines.append(f"{indent}int64_t {var};")
            lines.append(
                f"{indent}if (intj_grid_{operation}({left}, {right}, &{var}) != 0) return -1;"
            )
            return var, "int"
        if isinstance(node, ast.Call) and len(node.args) == 2 and not node.keywords:
            if isinstance(node.func, ast.Name) and node.func.id == "min":
                if not min_is_builtin or "min" in local_names:
                    raise fail(node, "grid_cpp min must resolve to the builtin")
                operation = "min"
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "cdiv"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "triton"
            ):
                if not cdiv_is_triton or "triton" in local_names:
                    raise fail(node, "grid_cpp triton.cdiv must resolve to Triton")
                operation = "cdiv"
            else:
                raise fail(node, "grid_cpp unsupported call")
            left, left_kind = emit(node.args[0], indent)
            right, right_kind = emit(node.args[1], indent)
            if left_kind != "int" or right_kind != "int":
                raise fail(node, f"grid_cpp {operation} needs integer operands")
            var = temp()
            lines.append(f"{indent}int64_t {var};")
            if operation == "min":
                lines.append(f"{indent}{var} = {left} < {right} ? {left} : {right};")
            else:
                lines.append(
                    f"{indent}if (intj_grid_cdiv({left}, {right}, &{var}) != 0) return -1;"
                )
            return var, "int"
        if isinstance(node, ast.IfExp):
            test, _ = emit(node.test, indent)
            var = temp()
            lines.append(f"{indent}int64_t {var};")
            lines.append(f"{indent}if ({test}) {{")
            yes, yes_kind = emit(node.body, indent + "  ")
            lines.append(f"{indent}  {var} = {yes};")
            lines.append(f"{indent}}} else {{")
            no, no_kind = emit(node.orelse, indent + "  ")
            lines.append(f"{indent}  {var} = {no};")
            lines.append(f"{indent}}}")
            if yes_kind != no_kind:
                raise fail(
                    node, "grid_cpp conditional branches need the same scalar type"
                )
            return var, yes_kind
        raise fail(node, f"grid_cpp unsupported {type(node).__name__}")

    statements = body.body
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and type(statements[0].value.value) is str
    ):
        statements = statements[1:]
    if not statements or not isinstance(statements[-1], ast.Return):
        raise fail(body, "grid_cpp needs a final return")
    for statement in statements[:-1]:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise fail(statement, "grid_cpp supports only simple local assignments")
        target = statement.targets[0].id
        value = emit(statement.value)
        bound[target] = value
    result = statements[-1].value
    if not isinstance(result, (ast.Tuple, ast.List)) or not 1 <= len(result.elts) <= 3:
        raise fail(statements[-1], "grid_cpp must return a 1-3 element tuple or list")
    dimensions = [emit(element) for element in result.elts]
    for i, (var, kind) in enumerate(dimensions):
        if kind != "int":
            raise fail(result.elts[i], "grid_cpp dimensions must be integers")
        lines.append(f"  if (intj_grid_output({var}, &dims[{i}]) != 0) return -1;")
    lines.extend(("  return 0;", "}"))
    return GridCode(tuple(extras), "\n".join(lines))
