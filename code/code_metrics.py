"""
Code complexity metrics for generated examples.

Computes per-file metrics:
  - loc: lines of code (excluding comments and blank lines)
  - num_functions: number of function definitions
  - type_imports: number of type-related imports
  - type_density: (typed_objects - untyped_objects) / loc
  - internal_calls: number of functions calling other locally-defined functions
"""

import ast
import re
from dataclasses import dataclass


TYPING_MODULES = {"typing", "typing_extensions", "collections.abc"}


@dataclass
class CodeMetrics:
    loc: int
    num_functions: int
    type_imports: int
    type_density: float
    internal_calls: int


def _count_loc(source_code: str) -> int:
    """Count lines of code excluding blank lines and comment-only lines."""
    count = 0
    for line in source_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        count += 1
    return count


def _count_functions(tree: ast.Module) -> int:
    """Count all function and async function definitions."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            count += 1
    return count


def _count_type_imports(tree: ast.Module) -> int:
    """Count imports from typing-related modules."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # Direct typing module imports
            if node.module in TYPING_MODULES or node.module.startswith("typing."):
                count += len(node.names)
    return count


def _compute_type_density(tree: ast.Module, loc: int) -> float:
    """Compute (typed_objects - untyped_objects) / loc.

    Typed objects: annotated function parameters, annotated return types,
    and annotated variable assignments.
    Untyped objects: function parameters without annotations (excluding self/cls),
    functions without return annotations, bare assignments to new names.
    """
    if loc == 0:
        return 0.0

    typed = 0
    untyped = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Return annotation
            if node.returns:
                typed += 1
            else:
                untyped += 1

            # Parameters (skip self/cls)
            all_args = (
                node.args.posonlyargs
                + node.args.args
                + node.args.kwonlyargs
            )
            if node.args.vararg:
                all_args = list(all_args) + [node.args.vararg]
            if node.args.kwarg:
                all_args = list(all_args) + [node.args.kwarg]

            for arg in all_args:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.annotation:
                    typed += 1
                else:
                    untyped += 1

        elif isinstance(node, ast.AnnAssign):
            # x: int = 5 (annotation)
            typed += 1

        elif isinstance(node, ast.Assign):
            # x = 5 (no annotation)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    untyped += 1

    return (typed - untyped) / loc


def _count_internal_calls(tree: ast.Module) -> int:
    """Count how many locally-defined functions call other locally-defined functions."""
    # Collect all function names defined at any level
    function_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)

    if not function_names:
        return 0

    # For each function check if it calls any other local function
    call_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        caller = node.name
        called_locals: set[str] = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                callee = child.func.id
                if callee in function_names and callee != caller:
                    called_locals.add(callee)

        call_count += len(called_locals)

    return call_count


def compute_metrics(source_code: str) -> CodeMetrics:
    """Compute all code metrics for a source file."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return CodeMetrics(
            loc=_count_loc(source_code),
            num_functions=0,
            type_imports=0,
            type_density=0.0,
            internal_calls=0,
        )

    loc = _count_loc(source_code)

    return CodeMetrics(
        loc=loc,
        num_functions=_count_functions(tree),
        type_imports=_count_type_imports(tree),
        type_density=round(_compute_type_density(tree, loc), 4),
        internal_calls=_count_internal_calls(tree),
    )


def metrics_to_dict(m: CodeMetrics) -> dict:
    """Convert metrics to a JSON-serializable dict."""
    return {
        "loc": m.loc,
        "num_functions": m.num_functions,
        "type_imports": m.type_imports,
        "type_density": m.type_density,
        "internal_calls": m.internal_calls,
    }

