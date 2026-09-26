"""Read `@triton.heuristics` functions: which names each one reads.

Task-4 lowering to C lives here too; this part only parses.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import linecache
import types
from typing import Any

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
        if first == code.co_firstlineno and _same_code(node, code):
            matches.append(node)
    if len(matches) != 1:
        raise HeuristicError(
            f"cannot locate the source of heuristic {fn.__name__!r} at "
            f"{code.co_filename}:{code.co_firstlineno}"
        )
    return matches[0]


def _same_code(node: ast.Lambda | ast.FunctionDef, code: types.CodeType) -> bool:
    if isinstance(node, ast.Lambda):
        compiled = compile(ast.Expression(body=node), code.co_filename, "eval")
    else:
        compiled = compile(
            ast.Module(body=[node], type_ignores=[]), code.co_filename, "exec"
        )
    return any(
        isinstance(const, types.CodeType)
        and const.co_code == code.co_code
        and const.co_names == code.co_names
        and const.co_varnames == code.co_varnames
        and const.co_consts == code.co_consts
        for const in compiled.co_consts
    )
