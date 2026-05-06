"""
Phase 2: Signature-Driven Hypothesis Property Testing for Type Constraint Validation.

This module exercises actual code paths by extracting function/method definitions
from source, introspecting their signatures + type hints, building Hypothesis
strategies, and running @given tests with generated inputs.

Architecture:
1. Parse the AST to find all user-defined functions and class methods.
2. Execute source once (with __name__ != "__main__") to build a live namespace.
3. Resolve each definition's callable and introspect signature + type hints.
4. Build Hypothesis strategies from concrete parameter type hints.
5. Run @given(...) tests that call real code and catch runtime exceptions.
6. Check return types (Self substitution, typeguard enforcement).

Usage:
    from hypothesis_tier2 import run_hypothesis_tier2
    bugs = run_hypothesis_tier2(source_code, checker_outputs=outputs)
"""

import ast
import inspect
import io
import os
import re
import contextlib
import typing
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

try:
    from hypothesis import given, strategies as st, settings, HealthCheck, Verbosity
    from hypothesis.errors import Unsatisfiable
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    st = None  # type: ignore

try:
    from typeguard import check_type, TypeCheckError
    HAS_TYPEGUARD = True
except ImportError:
    HAS_TYPEGUARD = False


class Verdict(Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class TypeBug:
    line: int
    bug_type: str
    message: str
    source: str
    confidence: float
    details: dict = field(default_factory=dict)


@dataclass
class TypeAnnotation:
    line: int
    variable_name: str
    annotation: str
    value_expr: Optional[str] = None


@dataclass
class HypothesisTestResult:
    annotation: str
    verdict: Verdict
    failure_type: Optional[str]
    failure_message: Optional[str]
    test_cases_run: int
    failing_example: Optional[Any] = None


class CallKind:
    FUNCTION = "function"
    CONSTRUCTOR = "constructor"
    METHOD = "method"


@dataclass
class InvocationPlan:
    line: int
    kind: str
    call_text: str
    call_node: Optional[ast.Call] = None
    func_name: Optional[str] = None
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    receiver_class_name: Optional[str] = None
    receiver_ctor_args: Optional[list] = None
    receiver_ctor_kwargs: Optional[dict] = None
    callable_obj: Any = None
    sig: Any = None
    hints: dict = field(default_factory=dict)
    return_hint: Any = None
    skipped: Optional[str] = None
    ast_node: Optional[ast.FunctionDef] = None
    strategy_descriptions: dict = field(default_factory=dict)
    used_fallback_strategy: bool = False
    skipped_params: list = field(default_factory=list)


MAX_EXAMPLES = 30
TYPE_ERROR_EXCEPTIONS = (TypeError, KeyError, AttributeError, ValueError)

_TYPE_CORRELATED_VALUE_ERROR_PATTERNS = (
    "unpack",
    "invalid literal",
    "could not convert",
    "not enough values",
    "too many values",
    "expected at least",
    "expected at most",
    "must be a string",
    "must be an integer",
    "must be a real number",
    "cannot be interpreted as",
    "unsupported format",
    "year is out of range",
    "month must be in",
    "day is out of range",
)


def _is_deliberate_key_error(exc: KeyError) -> bool:
    """Return True if a KeyError appears to be a deliberate domain-level raise
    rather than an accidental dict-key miss indicating a type bug.
    
    Deliberate raises typically have human-readable sentences as messages,
    while accidental misses have bare key values.
    """
    msg = str(exc)
    if len(msg) > 60:
        return True
    lower = msg.lower()
    if any(word in lower for word in ("not found", "not exist", "unknown", "invalid", "missing", "no ")):
        return True
    return False


def _is_missing_arg_error(exc: TypeError) -> bool:
    """Return True if a TypeError is about missing positional arguments,
    which typically means the test harness failed to provide required args
    (e.g., due to a decorator hiding the signature) not a real type bug."""
    msg = str(exc).lower()
    return "missing" in msg and "required" in msg and "argument" in msg


def _is_type_correlated_error(exc: Exception) -> bool:
    if isinstance(exc, KeyError):
        return not _is_deliberate_key_error(exc)
    if isinstance(exc, TypeError) and _is_missing_arg_error(exc):
        return False
    if not isinstance(exc, ValueError):
        return True
    msg = str(exc).lower()
    return any(pat in msg for pat in _TYPE_CORRELATED_VALUE_ERROR_PATTERNS)


def _is_paramspec_component(hint: Any) -> bool:
    if hasattr(typing, "ParamSpec"):
        if isinstance(hint, typing.ParamSpec):
            return True
    hint_str = str(hint)
    if ".args" in hint_str or ".kwargs" in hint_str:
        return True
    origin = getattr(hint, "__origin__", None)
    if origin is not None and hasattr(typing, "ParamSpec"):
        if isinstance(origin, typing.ParamSpec):
            return True
    return False


BUILTIN_NAMES = frozenset({
    "print", "len", "range", "enumerate", "zip", "sorted", "reversed",
    "list", "dict", "set", "tuple", "int", "str", "float", "bool",
    "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "repr", "id", "hash", "reveal_type", "super", "property",
    "staticmethod", "classmethod", "object", "Exception", "BaseException",
})

SKIP_DUNDER = frozenset({
    "__init__", "__repr__", "__str__", "__hash__", "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__", "__bool__", "__len__",
    "__del__", "__new__", "__init_subclass__", "__class_getitem__",
    "__set_name__", "__slots__",
})


# DEFINITION DISCOVERY (extract functions and methods from AST)

def _extract_definitions(tree: ast.Module) -> list[InvocationPlan]:
    plans: list[InvocationPlan] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("_") or name in BUILTIN_NAMES:
                continue
            plans.append(InvocationPlan(
                line=node.lineno,
                kind=CallKind.FUNCTION,
                call_text=f"{name}(...)",
                func_name=name,
                ast_node=node,
            ))

        elif isinstance(node, ast.ClassDef):
            cls_name = node.name
            if cls_name.startswith("_"):
                continue

            has_init = any(
                isinstance(m, ast.FunctionDef) and m.name == "__init__"
                for m in node.body
            )
            if has_init:
                init_node = next(
                    m for m in node.body
                    if isinstance(m, ast.FunctionDef) and m.name == "__init__"
                )
                plans.append(InvocationPlan(
                    line=init_node.lineno,
                    kind=CallKind.CONSTRUCTOR,
                    call_text=f"{cls_name}(...)",
                    class_name=cls_name,
                    ast_node=init_node,
                ))
            else:
                plans.append(InvocationPlan(
                    line=node.lineno,
                    kind=CallKind.CONSTRUCTOR,
                    call_text=f"{cls_name}()",
                    class_name=cls_name,
                ))

            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                mname = member.name
                if mname in SKIP_DUNDER or mname.startswith("__"):
                    continue

                decorators = set()
                for dec in member.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorators.add(dec.id)
                    elif isinstance(dec, ast.Attribute):
                        decorators.add(dec.attr)

                if "staticmethod" in decorators:
                    kind = CallKind.FUNCTION
                    call_text = f"{cls_name}.{mname}(...)"
                    plans.append(InvocationPlan(
                        line=member.lineno,
                        kind=kind,
                        call_text=call_text,
                        func_name=f"{cls_name}.{mname}",
                        class_name=cls_name,
                        method_name=mname,
                        ast_node=member,
                    ))
                elif "classmethod" in decorators:
                    kind = CallKind.FUNCTION
                    call_text = f"{cls_name}.{mname}(...)"
                    plans.append(InvocationPlan(
                        line=member.lineno,
                        kind=kind,
                        call_text=call_text,
                        func_name=f"{cls_name}.{mname}",
                        class_name=cls_name,
                        method_name=mname,
                        ast_node=member,
                    ))
                else:
                    plans.append(InvocationPlan(
                        line=member.lineno,
                        kind=CallKind.METHOD,
                        call_text=f"{cls_name}.{mname}(...)",
                        method_name=mname,
                        receiver_class_name=cls_name,
                        class_name=cls_name,
                        ast_node=member,
                    ))

    return plans


# RUNTIME RESOLUTION (execute source code, inspect signatures & hints)

def _build_source_env(source_code: str) -> Optional[dict[str, Any]]:
    env: dict[str, Any] = {"__name__": "__hypothesis_tier2__", "__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(compile(source_code, "<source>", "exec"), env)
    except Exception:
        return None
    return env


def _is_protocol_class(cls: type) -> bool:
    if getattr(cls, "_is_protocol", False):
        return True
    if hasattr(typing, "Protocol") and typing.Protocol in getattr(cls, "__mro__", []):
        return True
    return False


def _has_erased_signature(sig: inspect.Signature) -> bool:
    """Detect if a signature has been erased by a decorator to (*args, **kwargs)."""
    params = list(sig.parameters.values())
    if not params:
        return False
    kinds = {p.kind for p in params}
    named_params = [p for p in params if p.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )]
    has_var_pos = inspect.Parameter.VAR_POSITIONAL in kinds
    has_var_kw = inspect.Parameter.VAR_KEYWORD in kinds
    return has_var_pos and has_var_kw and len(named_params) == 0


def _extract_hints_from_ast(
    node: ast.FunctionDef, env: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Extract parameter type hints and return hint from an AST FunctionDef node.

    Resolves annotation strings against the live namespace via eval().
    Returns (param_hints, return_hint).
    """
    hints: dict[str, Any] = {}
    for arg in node.args.args:
        if arg.arg == "self" or arg.arg == "cls":
            continue
        if arg.annotation is not None:
            ann_str = ast.unparse(arg.annotation)
            try:
                hints[arg.arg] = eval(ann_str, env)
            except Exception:
                pass

    for arg in node.args.kwonlyargs:
        if arg.annotation is not None:
            ann_str = ast.unparse(arg.annotation)
            try:
                hints[arg.arg] = eval(ann_str, env)
            except Exception:
                pass

    return_hint = None
    if node.returns is not None:
        ret_str = ast.unparse(node.returns)
        try:
            return_hint = eval(ret_str, env)
        except Exception:
            pass

    return hints, return_hint


def _build_sig_from_ast(
    node: ast.FunctionDef,
) -> Optional[inspect.Signature]:
    """Build an inspect.Signature from an AST FunctionDef node,
    bypassing the live (potentially decorated) object's signature."""
    params: list[inspect.Parameter] = []
    all_args = node.args

    num_defaults = len(all_args.defaults)
    num_pos_args = len(all_args.args)
    first_default_idx = num_pos_args - num_defaults

    for i, arg in enumerate(all_args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        default = inspect.Parameter.empty
        if i >= first_default_idx:
            default_node = all_args.defaults[i - first_default_idx]
            try:
                default = eval(ast.unparse(default_node), {})
            except Exception:
                default = None
        params.append(inspect.Parameter(
            arg.arg,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default,
        ))

    for i, arg in enumerate(all_args.kwonlyargs):
        default = inspect.Parameter.empty
        if i < len(all_args.kw_defaults) and all_args.kw_defaults[i] is not None:
            try:
                default = eval(ast.unparse(all_args.kw_defaults[i]), {})
            except Exception:
                default = None
        params.append(inspect.Parameter(
            arg.arg,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
        ))

    return inspect.Signature(params) if params else None


def _is_safe_to_hypothesis_test(fn, return_hint) -> bool:
    if inspect.iscoroutinefunction(fn):
        return False
    if inspect.isgeneratorfunction(fn):
        return False
    unwrapped = inspect.unwrap(fn) if hasattr(fn, '__wrapped__') else fn
    try:
        sig = inspect.signature(unwrapped)
    except (ValueError, TypeError):
        return False
    params = list(sig.parameters.values())
    if params and all(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    ):
        return False
    if return_hint is None:
        return False
    hint_str = str(return_hint)
    if any(x in hint_str for x in ["TypeVar", "ParamSpec", "Coroutine", "Generator",
                                     "Iterator", "AsyncIterator", "Union", "Optional",
                                     "~", "T_", "TypeVarTuple"]):
        return False
    return True


def _resolve_plan(plan: InvocationPlan, env: dict[str, Any]) -> bool:
    try:
        if plan.kind == CallKind.CONSTRUCTOR:
            cls = env.get(plan.class_name)
            if cls is None or not isinstance(cls, type):
                return False
            if _is_protocol_class(cls):
                plan.skipped = f"{plan.class_name} is a Protocol and cannot be instantiated"
                return False
            if issubclass(cls, Enum):
                plan.skipped = f"{plan.class_name} is an Enum and uses value-based construction"
                return False
            plan.callable_obj = cls
            try:
                plan.sig = inspect.signature(cls)
            except (ValueError, TypeError):
                return False

            if plan.ast_node and _has_erased_signature(plan.sig):
                ast_sig = _build_sig_from_ast(plan.ast_node)
                if ast_sig is not None:
                    plan.sig = ast_sig
                ast_hints, _ = _extract_hints_from_ast(plan.ast_node, env)
                plan.hints = ast_hints
            else:
                try:
                    unwrapped = inspect.unwrap(cls.__init__) if hasattr(cls.__init__, '__wrapped__') else cls.__init__
                    plan.hints = typing.get_type_hints(unwrapped, globalns=env, localns=env)
                except Exception:
                    plan.hints = {}
            plan.hints.pop("return", None)
            plan.return_hint = cls
            if not _is_safe_to_hypothesis_test(plan.callable_obj, plan.return_hint):
                plan.skipped = f"{plan.class_name} is not safe to hypothesis test"
                return False

        elif plan.kind == CallKind.FUNCTION:
            if plan.class_name and plan.method_name:
                cls = env.get(plan.class_name)
                if cls is None or not isinstance(cls, type):
                    return False
                fn = getattr(cls, plan.method_name, None)
            else:
                fn = env.get(plan.func_name)
            if fn is None or not callable(fn):
                return False
            plan.callable_obj = fn
            try:
                plan.sig = inspect.signature(fn)
            except (ValueError, TypeError):
                return False

            if plan.ast_node and _has_erased_signature(plan.sig):
                ast_sig = _build_sig_from_ast(plan.ast_node)
                if ast_sig is not None:
                    plan.sig = ast_sig
                ast_hints, ast_return = _extract_hints_from_ast(plan.ast_node, env)
                plan.hints = ast_hints
                plan.return_hint = ast_return
            else:
                try:
                    target = inspect.unwrap(fn) if hasattr(fn, '__wrapped__') else fn
                    plan.hints = typing.get_type_hints(target, globalns=env, localns=env)
                except Exception:
                    plan.hints = {}
                plan.return_hint = plan.hints.pop("return", None)
            if not _is_safe_to_hypothesis_test(plan.callable_obj, plan.return_hint):
                plan.skipped = f"{plan.func_name} is not safe to hypothesis test"
                return False

        elif plan.kind == CallKind.METHOD:
            cls = env.get(plan.receiver_class_name)
            if cls is None or not isinstance(cls, type):
                return False
            method = getattr(cls, plan.method_name, None)
            if method is None:
                return False
            plan.callable_obj = method
            try:
                plan.sig = inspect.signature(method)
            except (ValueError, TypeError):
                return False

            if plan.ast_node and _has_erased_signature(plan.sig):
                ast_sig = _build_sig_from_ast(plan.ast_node)
                if ast_sig is not None:
                    plan.sig = ast_sig
                ast_hints, ast_return = _extract_hints_from_ast(plan.ast_node, env)
                plan.hints = ast_hints
                plan.return_hint = ast_return
            else:
                try:
                    target = inspect.unwrap(method) if hasattr(method, '__wrapped__') else method
                    plan.hints = typing.get_type_hints(target, globalns=env, localns=env)
                except Exception:
                    plan.hints = {}
                plan.return_hint = plan.hints.pop("return", None)
            if not _is_safe_to_hypothesis_test(plan.callable_obj, plan.return_hint):
                plan.skipped = f"{plan.method_name} is not safe to hypothesis test"
                return False

        else:
            return False

        return True
    except Exception:
        return False


# STRATEGY BUILDING (from concrete parameter type hints)

def _strategy_for_hint(hint: Any, env: dict[str, Any], depth: int = 0) -> Optional[Any]:
    if depth > 4 or st is None:
        return None

    if hint is inspect.Parameter.empty:
        return st.one_of(st.integers(), st.text(max_size=20), st.booleans())

    if hint is int:
        return st.integers(min_value=-1000, max_value=1000)
    if hint is float:
        return st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000)
    if hint is str:
        return st.text(max_size=30)
    if hint is bool:
        return st.booleans()
    if hint is bytes:
        return st.binary(max_size=30)
    if hint is type(None):
        return st.none()
    if hint is Any:
        return st.one_of(st.integers(), st.text(max_size=10), st.booleans(), st.none())

    if hasattr(typing, "TypeVarTuple") and isinstance(hint, typing.TypeVarTuple):
        return None

    if _is_paramspec_component(hint):
        return None

    if isinstance(hint, typing.TypeVar):
        if hint.__bound__ is not None:
            return _strategy_for_hint(hint.__bound__, env, depth + 1)
        if hint.__constraints__:
            sub = [_strategy_for_hint(c, env, depth + 1) for c in hint.__constraints__]
            valid = [s for s in sub if s is not None]
            if valid:
                return st.one_of(*valid)
        return st.one_of(st.integers(), st.text(max_size=10), st.booleans())

    if hasattr(hint, "__supertype__"):
        base_strat = _strategy_for_hint(hint.__supertype__, env, depth + 1)
        if base_strat is not None:
            return base_strat.map(hint)
        return None

    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", None) or ()

    if origin is typing.Union:
        sub = [_strategy_for_hint(a, env, depth + 1) for a in args]
        valid = [s for s in sub if s is not None]
        if valid:
            return st.one_of(*valid)
        return None

    if origin in (list, typing.List) or hint is list:
        if args:
            inner = _strategy_for_hint(args[0], env, depth + 1)
            if inner is not None:
                return st.lists(inner, max_size=5)
        return st.lists(st.integers(), max_size=5)

    if origin in (dict, typing.Dict) or hint is dict:
        if len(args) >= 2:
            ks = _strategy_for_hint(args[0], env, depth + 1)
            vs = _strategy_for_hint(args[1], env, depth + 1)
            if ks and vs:
                return st.dictionaries(ks, vs, max_size=3)
        return st.dictionaries(st.text(max_size=10), st.integers(), max_size=3)

    if origin in (tuple, typing.Tuple) or hint is tuple:
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                inner = _strategy_for_hint(args[0], env, depth + 1)
                if inner:
                    return st.lists(inner, max_size=5).map(tuple)
            else:
                subs = [_strategy_for_hint(a, env, depth + 1) for a in args]
                if all(s is not None for s in subs):
                    return st.tuples(*subs)
        return st.just(())

    if origin in (set, typing.Set) or hint is set:
        if args:
            inner = _strategy_for_hint(args[0], env, depth + 1)
            if inner:
                return st.frozensets(inner, max_size=5).map(set)
        return st.frozensets(st.integers(), max_size=5).map(set)

    if origin is typing.Literal or (hasattr(typing, "Literal") and origin is getattr(typing, "Literal", None)):
        if args:
            return st.sampled_from(list(args))

    if origin is type:
        candidates = [
            v for v in env.values()
            if isinstance(v, type) and v is not type and not v.__name__.startswith("_")
        ]
        if args:
            base = args[0]
            if isinstance(base, type):
                candidates = [c for c in candidates if issubclass(c, base)]
        if candidates:
            return st.sampled_from(candidates)
        return None

    if hasattr(hint, "__annotations__") and hasattr(hint, "__required_keys__"):
        return _strategy_for_typeddict(hint, env, depth)

    if isinstance(hint, type) and issubclass(hint, (int, float, str, bool, bytes)):
        return _strategy_for_hint(hint.__bases__[0], env, depth + 1)

    if isinstance(hint, type) and (inspect.isabstract(hint) or _is_protocol(hint)):
        implementors = _find_protocol_implementors(hint, env)
        if implementors:
            sub = [_try_construct_instance_strategy(c, env, depth + 1) for c in implementors]
            valid = [s for s in sub if s is not None]
            if valid:
                return st.one_of(*valid)
        return None

    if isinstance(hint, type):
        return _try_construct_instance_strategy(hint, env, depth)

    try:
        return st.from_type(hint)
    except Exception:
        pass

    return None


def _strategy_for_typeddict(td_type: Any, env: dict[str, Any], depth: int) -> Optional[Any]:
    hints = getattr(td_type, "__annotations__", {})
    required = getattr(td_type, "__required_keys__", set())
    optional_keys = getattr(td_type, "__optional_keys__", set())

    fixed = {}
    for key, val_type in hints.items():
        val_strat = _strategy_for_hint(val_type, env, depth + 1)
        if val_strat is None:
            if key in required:
                return None
            continue
        fixed[key] = (key in required, val_strat)

    if not fixed:
        return None

    @st.composite
    def build_td(draw):
        result = {}
        for key, (is_req, strat) in fixed.items():
            if is_req or draw(st.booleans()):
                result[key] = draw(strat)
        return result

    return build_td()


def _is_protocol(cls: type) -> bool:
    return getattr(cls, "_is_protocol", False) or (
        hasattr(typing, "Protocol") and typing.Protocol in getattr(cls, "__mro__", [])
    )


def _find_protocol_implementors(protocol: type, env: dict[str, Any]) -> list[type]:
    is_runtime_checkable = getattr(protocol, "__protocol_attrs__", None) is not None and (
        hasattr(protocol, "_is_runtime_protocol") and protocol._is_runtime_protocol
    )

    implementors = []
    for name, obj in env.items():
        if not isinstance(obj, type) or obj is protocol:
            continue
        if inspect.isabstract(obj) or _is_protocol(obj):
            continue

        if is_runtime_checkable:
            try:
                if issubclass(obj, protocol):
                    implementors.append(obj)
                    continue
            except TypeError:
                pass

        required_members = set(getattr(protocol, "__protocol_attrs__", []))
        if not required_members:
            required_members = getattr(protocol, "__abstractmethods__", set())
        if required_members and all(hasattr(obj, m) for m in required_members):
            implementors.append(obj)

    return implementors


def _try_construct_instance_strategy(
    cls: type, env: dict[str, Any], depth: int,
) -> Optional[Any]:
    if depth > 4:
        return None
    if inspect.isabstract(cls):
        return None
    if _is_protocol_class(cls):
        return None
    if issubclass(cls, Enum):
        return None
    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return None

    try:
        hints = typing.get_type_hints(cls.__init__, globalns=env, localns=env)
    except Exception:
        hints = {}
    hints.pop("return", None)

    param_strats: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        hint = hints.get(pname, param.annotation)
        if hint is inspect.Parameter.empty:
            if param.default is not inspect.Parameter.empty:
                continue
            return None
        s = _strategy_for_hint(hint, env, depth + 1)
        if s is None:
            if param.default is not inspect.Parameter.empty:
                continue
            return None
        param_strats[pname] = s

    if not param_strats:
        return st.just(cls())

    @st.composite
    def build_instance(draw):
        kwargs = {k: draw(v) for k, v in param_strats.items()}
        return cls(**kwargs)

    return build_instance()


def _describe_strategy(strat: Any) -> str:
    try:
        r = repr(strat)
        if len(r) > 80:
            return r[:77] + "..."
        return r
    except Exception:
        return "<strategy>"


def _strategy_for_hint_no_fallback(hint: Any, env: dict[str, Any]) -> Optional[Any]:
    if hint is inspect.Parameter.empty or hint is Any:
        return _strategy_for_hint(hint, env)
    if hint in (int, float, str, bool, bytes, type(None)):
        return _strategy_for_hint(hint, env)
    origin = getattr(hint, "__origin__", None)
    if origin is not None:
        return _strategy_for_hint(hint, env)
    if hasattr(hint, "__supertype__"):
        return _strategy_for_hint(hint, env)
    if hasattr(hint, "__annotations__") and hasattr(hint, "__required_keys__"):
        return _strategy_for_hint(hint, env)
    if isinstance(hint, type):
        if issubclass(hint, (int, float, str, bool, bytes)):
            return _strategy_for_hint(hint, env)
        if inspect.isabstract(hint) or _is_protocol(hint):
            return _strategy_for_hint(hint, env)
        return _try_construct_instance_strategy(hint, env, 0)
    return None


# STRATEGY BUILDING FROM SIGNATURES

def _build_param_strats_from_sig(
    plan: InvocationPlan, env: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if plan.sig is None:
        return None

    strats: dict[str, Any] = {}
    descriptions: dict[str, str] = {}
    typevar_groups: dict[int, list[str]] = {}
    typevar_strats: dict[int, Any] = {}

    for pname, param in plan.sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        hint = plan.hints.get(pname, param.annotation)

        if _is_paramspec_component(hint):
            plan.skipped_params.append(pname)
            continue

        if isinstance(hint, typing.TypeVar):
            tv_id = id(hint)
            if tv_id not in typevar_strats:
                s = _strategy_for_hint(hint, env)
                if s is None:
                    if param.default is not inspect.Parameter.empty:
                        continue
                    plan.skipped = f"no strategy for param '{pname}' (hint={hint})"
                    return None
                typevar_strats[tv_id] = s
                typevar_groups[tv_id] = []
            typevar_groups[tv_id].append(pname)
            continue

        s = _strategy_for_hint_no_fallback(hint, env)
        if s is not None:
            strats[pname] = s
            hint_name = getattr(hint, "__name__", None) or str(hint)
            descriptions[pname] = f"{hint_name} -> {_describe_strategy(s)}"
        else:
            s_fallback = _strategy_for_hint(hint, env)
            if s_fallback is not None:
                strats[pname] = s_fallback
                plan.used_fallback_strategy = True
                hint_name = getattr(hint, "__name__", None) or str(hint)
                descriptions[pname] = f"{hint_name} -> {_describe_strategy(s_fallback)} [from_type fallback]"
            elif param.default is not inspect.Parameter.empty:
                continue
            else:
                plan.skipped = f"no strategy for param '{pname}' (hint={hint})"
                return None

    if typevar_groups:
        strats = _wrap_with_typevar_sharing(strats, typevar_groups, typevar_strats)
        for tv_id, pnames in typevar_groups.items():
            strat = typevar_strats[tv_id]
            for pname in pnames:
                descriptions[pname] = f"TypeVar (shared) -> {_describe_strategy(strat)}"

    plan.strategy_descriptions = descriptions
    return strats


def _wrap_with_typevar_sharing(
    base_strats: dict[str, Any],
    typevar_groups: dict[int, list[str]],
    typevar_strats: dict[int, Any],
) -> dict[str, Any]:
    @st.composite
    def shared_draw(draw):
        result = {}
        for pname, strat in base_strats.items():
            result[pname] = draw(strat)
        for tv_id, pnames in typevar_groups.items():
            val = draw(typevar_strats[tv_id])
            for pname in pnames:
                result[pname] = val
        return result

    return {"__shared__": shared_draw()}


def _unpack_shared_kwargs(kwargs: dict) -> dict:
    if "__shared__" in kwargs:
        return dict(kwargs["__shared__"])
    return dict(kwargs)


# TEST EXECUTION — run Hypothesis tests per invocation plan

def _run_plan_test(
    plan: InvocationPlan,
    env: dict[str, Any],
) -> list[TypeBug]:
    if plan.skipped:
        return []

    param_strats = _build_param_strats_from_sig(plan, env)
    if param_strats is None:
        return []

    if plan.kind == CallKind.CONSTRUCTOR:
        return _test_constructor(plan, param_strats, env)
    elif plan.kind == CallKind.FUNCTION:
        return _test_function(plan, param_strats, env)
    elif plan.kind == CallKind.METHOD:
        return _test_method(plan, param_strats, env)

    return []


def _test_constructor(
    plan: InvocationPlan,
    param_strats: dict[str, Any],
    env: dict[str, Any],
) -> list[TypeBug]:
    bugs: list[TypeBug] = []
    cls = plan.callable_obj
    crash_examples: list[dict] = []
    cases_run = 0

    def check_ctor(**kwargs):
        nonlocal cases_run
        kwargs = _unpack_shared_kwargs(kwargs)
        cases_run += 1
        pos_args = []
        real_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("_pos_"):
                pos_args.append((int(k.split("_")[-1]), v))
            else:
                real_kwargs[k] = v
        pos_args.sort()
        positional = [v for _, v in pos_args]

        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                instance = cls(*positional, **real_kwargs)
            if plan.return_hint and isinstance(plan.return_hint, type):
                if not isinstance(instance, plan.return_hint):
                    crash_examples.append({
                        "kwargs": _safe_repr(kwargs),
                        "error": f"Expected {plan.return_hint.__name__}, got {type(instance).__name__}",
                        "type": "ReturnTypeMismatch",
                    })
                    raise AssertionError("type mismatch")
        except TYPE_ERROR_EXCEPTIONS as e:
            if not _is_type_correlated_error(e):
                return
            crash_examples.append({
                "kwargs": _safe_repr(kwargs),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "type": type(e).__name__,
            })
            raise AssertionError("crash")

    test_fn = _build_hypothesis_test(check_ctor, param_strats)
    _run_hypothesis_fn(test_fn)

    for ex in crash_examples[:3]:
        bugs.append(TypeBug(
            line=plan.line,
            bug_type=ex["type"],
            message=f"{plan.call_text} -> {ex['error']}",
            source="hypothesis_tier2",
            confidence=0.65 if plan.used_fallback_strategy else 0.95,
            details={
                "call_text": plan.call_text,
                "kind": plan.kind,
                "failing_args": ex["kwargs"],
                "test_cases_run": cases_run,
                "used_fallback_strategy": plan.used_fallback_strategy,
            },
        ))

    return bugs


def _test_function(
    plan: InvocationPlan,
    param_strats: dict[str, Any],
    env: dict[str, Any],
) -> list[TypeBug]:
    bugs: list[TypeBug] = []
    fn = plan.callable_obj
    crash_examples: list[dict] = []
    return_mismatches: list[dict] = []
    cases_run = 0

    def check_fn(**kwargs):
        nonlocal cases_run
        kwargs = _unpack_shared_kwargs(kwargs)
        cases_run += 1
        pos_args = []
        real_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("_pos_"):
                pos_args.append((int(k.split("_")[-1]), v))
            else:
                real_kwargs[k] = v
        pos_args.sort()
        positional = [v for _, v in pos_args]

        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = fn(*positional, **real_kwargs)

            if plan.return_hint is not None and HAS_TYPEGUARD:
                try:
                    check_type(result, plan.return_hint)
                except (TypeCheckError, TypeError) as te:
                    return_mismatches.append({
                        "kwargs": _safe_repr(kwargs),
                        "error": f"Return type mismatch: {te}",
                        "type": "ReturnTypeMismatch",
                    })
        except TYPE_ERROR_EXCEPTIONS as e:
            if not _is_type_correlated_error(e):
                return
            crash_examples.append({
                "kwargs": _safe_repr(kwargs),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "type": type(e).__name__,
            })
            raise AssertionError("crash")

    test_fn = _build_hypothesis_test(check_fn, param_strats)
    _run_hypothesis_fn(test_fn)

    for ex in crash_examples[:3]:
        bugs.append(TypeBug(
            line=plan.line,
            bug_type=ex["type"],
            message=f"{plan.call_text} -> {ex['error']}",
            source="hypothesis_tier2",
            confidence=0.65 if plan.used_fallback_strategy else 0.95,
            details={
                "call_text": plan.call_text,
                "kind": plan.kind,
                "failing_args": ex["kwargs"],
                "test_cases_run": cases_run,
                "used_fallback_strategy": plan.used_fallback_strategy,
            },
        ))

    for ex in return_mismatches[:2]:
        bugs.append(TypeBug(
            line=plan.line,
            bug_type=ex["type"],
            message=f"{plan.call_text} -> {ex['error']}",
            source="hypothesis_tier2",
            confidence=0.60 if plan.used_fallback_strategy else 0.85,
            details={
                "call_text": plan.call_text,
                "kind": plan.kind,
                "failing_args": ex["kwargs"],
                "test_cases_run": cases_run,
                "used_fallback_strategy": plan.used_fallback_strategy,
            },
        ))

    return bugs


def _test_method(
    plan: InvocationPlan,
    param_strats: dict[str, Any],
    env: dict[str, Any],
) -> list[TypeBug]:
    bugs: list[TypeBug] = []
    cls = env.get(plan.receiver_class_name)
    if cls is None or not isinstance(cls, type):
        return bugs

    recv_strat = _try_construct_instance_strategy(cls, env, depth=0)
    if recv_strat is None:
        plan.skipped = f"cannot construct receiver {plan.receiver_class_name}"
        return bugs

    crash_examples: list[dict] = []
    return_mismatches: list[dict] = []
    cases_run = 0

    expected_return = plan.return_hint
    is_self_return = False
    if expected_return is not None:
        if hasattr(typing, 'Self') and expected_return is typing.Self:
            is_self_return = True
        elif hasattr(expected_return, '__name__') and expected_return.__name__ == 'Self':
            is_self_return = True

    def check_method(receiver, **kwargs):
        nonlocal cases_run
        kwargs = _unpack_shared_kwargs(kwargs)
        cases_run += 1
        method = getattr(receiver, plan.method_name, None)
        if method is None:
            return
        pos_args = []
        real_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("_pos_"):
                pos_args.append((int(k.split("_")[-1]), v))
            else:
                real_kwargs[k] = v
        pos_args.sort()
        positional = [v for _, v in pos_args]

        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = method(*positional, **real_kwargs)

            if is_self_return:
                if not isinstance(result, type(receiver)):
                    return_mismatches.append({
                        "kwargs": _safe_repr(kwargs),
                        "error": f"Self return: expected {type(receiver).__name__}, got {type(result).__name__}",
                        "type": "ReturnTypeMismatch",
                    })
            elif expected_return is not None and HAS_TYPEGUARD:
                try:
                    check_type(result, expected_return)
                except (TypeCheckError, TypeError) as te:
                    return_mismatches.append({
                        "kwargs": _safe_repr(kwargs),
                        "error": f"Return type mismatch: {te}",
                        "type": "ReturnTypeMismatch",
                    })
        except TYPE_ERROR_EXCEPTIONS as e:
            if not _is_type_correlated_error(e):
                return
            crash_examples.append({
                "kwargs": _safe_repr(kwargs),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "type": type(e).__name__,
            })
            raise AssertionError("crash")

    all_strats = {"receiver": recv_strat}
    all_strats.update(param_strats)

    test_fn = _build_hypothesis_test(check_method, all_strats)
    _run_hypothesis_fn(test_fn)

    for ex in crash_examples[:3]:
        bugs.append(TypeBug(
            line=plan.line,
            bug_type=ex["type"],
            message=f"{plan.call_text} -> {ex['error']}",
            source="hypothesis_tier2",
            confidence=0.65 if plan.used_fallback_strategy else 0.95,
            details={
                "call_text": plan.call_text,
                "kind": plan.kind,
                "receiver_class": plan.receiver_class_name,
                "method": plan.method_name,
                "failing_args": ex["kwargs"],
                "test_cases_run": cases_run,
                "used_fallback_strategy": plan.used_fallback_strategy,
            },
        ))

    for ex in return_mismatches[:2]:
        bugs.append(TypeBug(
            line=plan.line,
            bug_type=ex["type"],
            message=f"{plan.call_text} -> {ex['error']}",
            source="hypothesis_tier2",
            confidence=0.60 if plan.used_fallback_strategy else 0.90,
            details={
                "call_text": plan.call_text,
                "kind": plan.kind,
                "receiver_class": plan.receiver_class_name,
                "method": plan.method_name,
                "failing_args": ex["kwargs"],
                "test_cases_run": cases_run,
                "used_fallback_strategy": plan.used_fallback_strategy,
            },
        ))

    return bugs


def _build_hypothesis_test(fn, param_strats: dict[str, Any]):
    if not param_strats:
        def wrapped():
            fn()
        return settings(
            max_examples=1,
            suppress_health_check=list(HealthCheck),
            verbosity=Verbosity.quiet,
            deadline=None,
            database=None,
        )(wrapped)

    return settings(
        max_examples=MAX_EXAMPLES,
        suppress_health_check=list(HealthCheck),
        verbosity=Verbosity.quiet,
        deadline=None,
        database=None,
    )(given(**param_strats)(fn))


def _run_hypothesis_fn(test_fn):
    try:
        test_fn()
    except AssertionError:
        pass
    except Unsatisfiable:
        pass
    except Exception:
        pass


def _safe_repr(obj: Any) -> str:
    try:
        r = repr(obj)
        if len(r) > 200:
            return r[:200] + "..."
        return r
    except Exception:
        return "<unrepresentable>"


# ARTIFACT SAVING

def _save_artifacts(
    plans: list[InvocationPlan],
    bugs: list[TypeBug],
    source_code: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    tested_plans = [p for p in plans if not p.skipped]
    skipped_plans = [p for p in plans if p.skipped]

    summary_lines = [
        '"""',
        "Hypothesis Tier 2 — Signature-Driven Property Test Summary",
        "",
        f"Definitions found: {len(plans)}",
        f"Testable (strategies built): {len(tested_plans)}",
        f"Skipped (no strategies): {len(skipped_plans)}",
        f"Bugs found: {len(bugs)}",
        "",
    ]

    for i, plan in enumerate(plans):
        if plan.skipped:
            summary_lines.append(f"Plan {i}: [{plan.kind}] {plan.call_text} (line {plan.line}) -> SKIPPED: {plan.skipped}")
        else:
            status = "BUG FOUND" if any(b.line == plan.line for b in bugs) else "ok"
            strat_info = ""
            if plan.strategy_descriptions:
                strat_parts = [f"{k}={v}" for k, v in plan.strategy_descriptions.items()]
                strat_info = f" strategies=[{', '.join(strat_parts)}]"
            skipped_info = ""
            if plan.skipped_params:
                skipped_info = f" [skipped params: {', '.join(plan.skipped_params)}]"
            summary_lines.append(
                f"Plan {i}: [{plan.kind}] {plan.call_text} (line {plan.line}) -> {status}"
                f" (max_examples={MAX_EXAMPLES}){strat_info}{skipped_info}"
            )

    summary_lines.append("")
    if bugs:
        summary_lines.append("Bugs:")
        for i, bug in enumerate(bugs):
            details = bug.details or {}
            cases = details.get("test_cases_run", "?")
            args = details.get("failing_args", "?")
            summary_lines.append(f"  {i}: L{bug.line} [{bug.bug_type}] {bug.message}")
            summary_lines.append(f"      test_cases_run={cases}, failing_args={args}")
    else:
        summary_lines.append("No bugs found by Tier 2.")
    summary_lines.append('"""')

    summary_path = os.path.join(output_dir, "tier2_summary.py")
    try:
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
    except Exception:
        pass

    for i, plan in enumerate(plans):
        if plan.skipped:
            continue
        plan_bugs = [b for b in bugs if b.line == plan.line]
        _save_plan_test_file(plan, plan_bugs, source_code, output_dir, i)


def _save_plan_test_file(
    plan: InvocationPlan,
    bugs: list[TypeBug],
    source_code: str,
    output_dir: str,
    index: int,
) -> None:
    safe_name = re.sub(r"[^\w]", "_", plan.call_text)[:50]
    status = "FAIL" if bugs else "PASS"
    filename = f"tier2_{index}_{status}_{safe_name}.py"
    filepath = os.path.join(output_dir, filename)

    lines = [
        '"""',
        "Hypothesis Tier 2 — Generated Property Test",
        "",
        f"Target: {plan.call_text}",
        f"Kind: {plan.kind}",
        f"Line: {plan.line}",
        f"Status: {status}",
        f"Max examples: {MAX_EXAMPLES}",
    ]
    if plan.strategy_descriptions:
        lines.append("")
        lines.append("Strategies:")
        for pname, desc in plan.strategy_descriptions.items():
            lines.append(f"  {pname}: {desc}")
    if bugs:
        lines.append("")
        for b in bugs:
            lines.append(f"Bug: [{b.bug_type}] {b.message}")
            details = b.details or {}
            lines.append(f"  test_cases_run={details.get('test_cases_run', '?')}")
            lines.append(f"  failing_args={details.get('failing_args', '?')}")
    lines.append('"""')
    lines.append("")

    lines.append("# --- Original source (full context) ---")
    lines.append("")
    for src_line in source_code.splitlines():
        lines.append(src_line)
    lines.append("")
    lines.append("")

    lines.append("# --- Tier 2 property test ---")
    lines.append("")

    if plan.kind == CallKind.CONSTRUCTOR:
        lines.extend(_gen_constructor_test(plan, source_code))
    elif plan.kind == CallKind.FUNCTION:
        lines.extend(_gen_function_test(plan, source_code))
    elif plan.kind == CallKind.METHOD:
        lines.extend(_gen_method_test(plan, source_code))

    try:
        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def _gen_constructor_test(plan: InvocationPlan, source_code: str) -> list[str]:
    lines = []
    cls_name = plan.class_name or "UnknownClass"

    lines.append("from hypothesis import given, strategies as st, settings")
    lines.append("")

    if plan.strategy_descriptions:
        lines.append(f"@settings(max_examples={MAX_EXAMPLES}, deadline=None)")
        strat_args = ", ".join(
            f"{pname}=..."
            for pname in plan.strategy_descriptions
        )
        lines.append(f"@given({strat_args})")
        param_list = ", ".join(plan.strategy_descriptions.keys())
        lines.append(f"def test_{cls_name}_constructor({param_list}):")
        lines.append(f'    """Property test: {cls_name}() with generated inputs."""')
        lines.append(f"    instance = {cls_name}({param_list})")
        lines.append(f"    assert isinstance(instance, {cls_name})")
    else:
        lines.append(f"def test_{cls_name}_constructor():")
        lines.append(f'    """Test that {cls_name}() can be constructed."""')
        lines.append(f"    instance = {cls_name}()")
        lines.append(f"    assert isinstance(instance, {cls_name})")

    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append(f"    test_{cls_name}_constructor()")
    return lines


def _gen_function_test(plan: InvocationPlan, source_code: str) -> list[str]:
    lines = []
    fn_name = plan.func_name or "unknown_func"
    safe_test_name = re.sub(r"[^\w]", "_", fn_name)

    lines.append("from hypothesis import given, strategies as st, settings")
    lines.append("")

    if plan.strategy_descriptions:
        lines.append(f"@settings(max_examples={MAX_EXAMPLES}, deadline=None)")
        strat_args = ", ".join(
            f"{pname}=..."
            for pname in plan.strategy_descriptions
        )
        lines.append(f"@given({strat_args})")
        param_list = ", ".join(plan.strategy_descriptions.keys())
        lines.append(f"def test_{safe_test_name}({param_list}):")
        lines.append(f'    """Property test: {fn_name}() with generated inputs."""')
        lines.append(f"    result = {fn_name}({param_list})")
    else:
        lines.append(f"def test_{safe_test_name}():")
        lines.append(f'    """Test that {fn_name}() runs without type errors."""')
        lines.append(f"    result = {fn_name}()")

    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append(f"    test_{safe_test_name}()")
    return lines


def _gen_method_test(plan: InvocationPlan, source_code: str) -> list[str]:
    lines = []
    cls_name = plan.receiver_class_name or "UnknownClass"
    method_name = plan.method_name or "unknown_method"
    test_name = f"{cls_name}_{method_name}"

    lines.append("from hypothesis import given, strategies as st, settings")
    lines.append("")

    all_descs = {"receiver": f"{cls_name} instance"}
    all_descs.update(plan.strategy_descriptions)

    lines.append(f"@settings(max_examples={MAX_EXAMPLES}, deadline=None)")
    strat_args = ", ".join(f"{pname}=..." for pname in all_descs)
    lines.append(f"@given({strat_args})")
    param_list = ", ".join(all_descs.keys())
    lines.append(f"def test_{test_name}({param_list}):")
    lines.append(f'    """Property test: {cls_name}.{method_name}() with generated inputs."""')

    method_params = ", ".join(
        pname for pname in plan.strategy_descriptions
    )
    lines.append(f"    result = receiver.{method_name}({method_params})")

    if plan.return_hint is not None:
        hint_name = getattr(plan.return_hint, "__name__", repr(plan.return_hint))
        lines.append(f"    # Expected return type: {hint_name}")

    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append(f"    test_{test_name}()")
    return lines


# ANNOTATION EXTRACTION (kept for API compatibility)

def extract_type_annotations(source_code: str) -> list[TypeAnnotation]:
    annotations: list[TypeAnnotation] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return annotations

    class AnnotationVisitor(ast.NodeVisitor):
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name):
                ann = ast.unparse(node.annotation)
                val = ast.unparse(node.value) if node.value else None
                annotations.append(TypeAnnotation(
                    line=node.lineno,
                    variable_name=node.target.id,
                    annotation=ann,
                    value_expr=val,
                ))
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node.returns:
                annotations.append(TypeAnnotation(
                    line=node.lineno,
                    variable_name=f"{node.name}.__return__",
                    annotation=ast.unparse(node.returns),
                ))
            for arg in node.args.args:
                if arg.annotation:
                    annotations.append(TypeAnnotation(
                        line=node.lineno,
                        variable_name=f"{node.name}.{arg.arg}",
                        annotation=ast.unparse(arg.annotation),
                    ))
            self.generic_visit(node)

    visitor = AnnotationVisitor()
    visitor.visit(tree)
    return annotations


# MAIN ENTRY POINT

def run_hypothesis_tier2(
    source_code: str,
    annotations: list[TypeAnnotation] | None = None,
    checker_outputs: dict[str, str] | None = None,
    output_dir: str | None = None,
) -> list[TypeBug]:
    """
    Run signature-driven Hypothesis property testing (Phase 2).

    1. Parse AST to find all user-defined functions and class methods.
    2. Execute source once to build a live namespace.
    3. Resolve each definition's callable, introspect signature + type hints.
    4. Build Hypothesis strategies from concrete parameter type hints.
    5. Run @given(...) tests that call real code, catching runtime exceptions.
    6. Check return types (including Self substitution).
    """
    if not HAS_HYPOTHESIS:
        return []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    env = _build_source_env(source_code)
    if env is None:
        return []

    plans = _extract_definitions(tree)
    if not plans:
        return []

    resolved_plans: list[InvocationPlan] = []
    for plan in plans:
        if _resolve_plan(plan, env):
            resolved_plans.append(plan)
        else:
            plan.skipped = "could not resolve callable in live namespace"
            resolved_plans.append(plan)

    bugs: list[TypeBug] = []
    seen_bugs: set[tuple[int, str, str]] = set()
    for plan in resolved_plans:
        plan_bugs = _run_plan_test(plan, env)
        for bug in plan_bugs:
            key = (bug.line, bug.bug_type, bug.message[:100])
            if key not in seen_bugs:
                seen_bugs.add(key)
                bugs.append(bug)

    if output_dir:
        _save_artifacts(resolved_plans, bugs, source_code, output_dir)

    return bugs

