import ast
import inspect
import typing
import io
import os
import re
import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from .hypothesis_tier2 import TypeBug
except ImportError:
    from hypothesis_tier2 import TypeBug


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
    msg = str(exc)
    if len(msg) > 60:
        return True
    lower = msg.lower()
    if any(word in lower for word in ("not found", "not exist", "unknown", "invalid", "missing", "no ")):
        return True
    return False


def _is_missing_arg_error(exc: TypeError) -> bool:
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


def _cls_needs_init_args(cls: type) -> bool:
    """Return True if the class __init__ has required positional args beyond self."""
    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return True
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.default is inspect.Parameter.empty and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


def _is_protocol_class(cls: type) -> bool:
    """Return True if cls is a Protocol (cannot be instantiated)."""
    if getattr(cls, "_is_protocol", False):
        return True
    if hasattr(typing, "Protocol") and typing.Protocol in getattr(cls, "__mro__", []):
        return True
    return False


def _is_abstract_class(cls: type) -> bool:
    """Return True if cls is an abstract class (cannot be instantiated)."""
    return inspect.isabstract(cls)


def _callable_needs_args(fn) -> bool:
    """Return True if function has required positional args (excluding self/cls)."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return True
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.default is inspect.Parameter.empty and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


def _ast_callable_needs_args(func: ast.FunctionDef, in_class: bool) -> bool:
    """Check if function has required positional args based on AST (pre-decoration)."""
    pos = list(func.args.posonlyargs) + list(func.args.args)
    defaults = func.args.defaults or []
    defaults_offset = len(pos) - len(defaults)
    for i, arg in enumerate(pos):
        if in_class and i == 0 and arg.arg in ("self", "cls"):
            continue
        if i < defaults_offset:
            return True
    for arg, dflt in zip(func.args.kwonlyargs, func.args.kw_defaults or []):
        if dflt is None:
            return True
    return False


def _get_typeguard_param_annotation(tree: ast.Module, func_name: str) -> Optional[str]:
    """Get the annotation string of the first parameter of a TypeGuard function."""
    for func in _get_functions(tree):
        if func.name == func_name and func.args.args:
            first_arg = func.args.args[0]
            if first_arg.annotation:
                return _get_annotation_str(first_arg.annotation)
    return None


def _build_source_env(source_code: str) -> Optional[dict[str, Any]]:
    env: dict[str, Any] = {"__name__": "__targeted_tests__", "__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(compile(source_code, "<source>", "exec"), env)
    except Exception:
        return None
    return env


def _get_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append(node)
    return result


def _get_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def _get_annotation_str(ann) -> str:
    return ast.unparse(ann) if ann else ""


def _has_tuple_param(func: ast.FunctionDef) -> list[tuple[str, str, int]]:
    results = []
    for arg in func.args.args:
        if arg.annotation:
            ann_str = _get_annotation_str(arg.annotation)
            if "Tuple" in ann_str or "tuple" in ann_str:
                results.append((arg.arg, ann_str, func.lineno))
    return results


def _find_typeguard_functions(tree: ast.Module) -> list[tuple[str, int]]:
    results = []
    for func in _get_functions(tree):
        if func.returns:
            ret_str = _get_annotation_str(func.returns)
            if "TypeGuard" in ret_str:
                results.append((func.name, func.lineno))
    return results


def _find_newtypes(tree: ast.Module) -> list[tuple[str, str, int]]:
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.value, ast.Call):
                func_str = _get_annotation_str(node.value.func)
                if func_str == "NewType" and len(node.value.args) >= 2:
                    name = _get_annotation_str(node.targets[0])
                    base = _get_annotation_str(node.value.args[1])
                    results.append((name, base, node.lineno))
    return results


def _find_inheritance(tree: ast.Module) -> list[tuple[str, list[str], int]]:
    results = []
    for cls in _get_classes(tree):
        bases = [_get_annotation_str(b) for b in cls.bases if _get_annotation_str(b) != "object"]
        if bases:
            results.append((cls.name, bases, cls.lineno))
    return results


def _find_overridden_methods(tree: ast.Module) -> list[tuple[str, str, str, int]]:
    class_methods: dict[str, list[tuple[str, int]]] = {}
    class_bases: dict[str, list[str]] = {}
    for cls in _get_classes(tree):
        methods = []
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append((item.name, item.lineno))
        class_methods[cls.name] = methods
        class_bases[cls.name] = [_get_annotation_str(b) for b in cls.bases]

    results = []
    for cls_name, methods in class_methods.items():
        for base_name in class_bases.get(cls_name, []):
            if base_name in class_methods:
                base_method_names = {m[0] for m in class_methods[base_name]}
                for method_name, lineno in methods:
                    if method_name in base_method_names and method_name != "__init__":
                        results.append((cls_name, base_name, method_name, lineno))
    return results


def _find_decorated_functions(tree: ast.Module) -> list[tuple[str, list[str], int, Optional[str], ast.FunctionDef]]:
    results = []
    for func in _get_functions(tree):
        if func.decorator_list:
            dec_names = [_get_annotation_str(d) for d in func.decorator_list]
            class_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if item is func:
                            class_name = node.name
            results.append((func.name, dec_names, func.lineno, class_name, func))
    return results


def _find_typevar_functions(tree: ast.Module, env: dict) -> list[tuple[str, dict[str, str], int]]:
    results = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = env.get(node.name)
            if fn is None or not callable(fn):
                continue
            try:
                hints = typing.get_type_hints(fn, globalns=env, localns=env)
            except Exception:
                continue
            tv_params = {}
            for pname, hint in hints.items():
                if pname == "return":
                    continue
                if isinstance(hint, typing.TypeVar):
                    tv_params[pname] = str(hint)
            if tv_params:
                results.append((node.name, tv_params, node.lineno))
    return results


def _find_callable_params(tree: ast.Module) -> list[tuple[str, str, int]]:
    results = []
    for func in _get_functions(tree):
        for arg in func.args.args:
            if arg.annotation:
                ann_str = _get_annotation_str(arg.annotation)
                if "Callable" in ann_str:
                    results.append((func.name, arg.arg, func.lineno))
    return results


def _gen_tuple_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    for func in _get_functions(tree):
        tuple_params = _has_tuple_param(func)
        if not tuple_params:
            continue
        fname = func.name
        if fname.startswith("_") or fname not in env:
            continue
        for pname, ann_str, lineno in tuple_params:
            tests.append(f'''
def test_{fname}_empty_tuple():
    """Call {fname} with empty tuple for param '{pname}'."""
    try:
        {fname}({pname}=())
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "empty_tuple"}})
''')
            tests.append(f'''
def test_{fname}_single_element_tuple():
    """Call {fname} with single-element tuple for param '{pname}'."""
    try:
        {fname}({pname}=(1,))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "single_element_tuple"}})
''')
            tests.append(f'''
def test_{fname}_none_in_tuple():
    """Call {fname} with None element in tuple for param '{pname}'."""
    try:
        {fname}({pname}=(None,))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "none_in_tuple"}})
''')
            tests.append(f'''
def test_{fname}_wrong_types_in_tuple():
    """Call {fname} with wrong types in tuple for param '{pname}'."""
    try:
        {fname}({pname}=(123, 456, 789))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "wrong_types_in_tuple"}})
''')
            tests.append(f'''
def test_{fname}_string_instead_of_tuple():
    """Call {fname} with a string instead of tuple for param '{pname}'."""
    try:
        {fname}({pname}="not a tuple")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "string_instead_of_tuple"}})
''')
    return tests


def _gen_typeguard_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    guards = _find_typeguard_functions(tree)
    for guard_name, lineno in guards:
        if guard_name not in env:
            continue
        ann = _get_typeguard_param_annotation(tree, guard_name)
        param_expects_iterable = ann and ("List" in ann or "list" in ann
                                          or "Iterable" in ann or "Sequence" in ann)

        tests.append(f'''
def test_{guard_name}_returns_bool():
    """Verify {guard_name} returns a boolean."""
    try:
        result = {guard_name}([])
        if not isinstance(result, bool):
            BUGS.append({{"line": {lineno}, "type": "ReturnTypeMismatch", "error": f"TypeGuard {guard_name} returned {{type(result).__name__}}, expected bool", "test": "typeguard_returns_bool"}})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_empty_list"}})
''')
        if not param_expects_iterable:
            tests.append(f'''
def test_{guard_name}_with_none():
    """Call {guard_name} with None."""
    try:
        {guard_name}(None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_none"}})
''')
        tests.append(f'''
def test_{guard_name}_with_ints():
    """Call {guard_name} with list of ints."""
    try:
        result = {guard_name}([1, 2, 3])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_ints"}})
''')
        tests.append(f'''
def test_{guard_name}_with_strings():
    """Call {guard_name} with list of strings."""
    try:
        result = {guard_name}(["a", "b", "c"])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_strings"}})
''')
        tests.append(f'''
def test_{guard_name}_with_mixed():
    """Call {guard_name} with mixed type list."""
    try:
        result = {guard_name}([1, "hello", True, 3.14])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_mixed"}})
''')
        tests.append(f'''
def test_{guard_name}_with_bools():
    """Call {guard_name} with list of booleans."""
    try:
        result = {guard_name}([True, False, True])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_bools"}})
''')
    return tests


def _gen_newtype_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    newtypes = _find_newtypes(tree)
    for nt_name, base_type, lineno in newtypes:
        if nt_name not in env:
            continue
        if base_type == "str":
            tests.append(f'''
def test_{nt_name}_from_string():
    """Create {nt_name} from a plain string."""
    try:
        val = {nt_name}("test_value")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_from_base"}})
''')
            tests.append(f'''
def test_{nt_name}_from_int():
    """Create {nt_name} from an int (wrong base type)."""
    try:
        val = {nt_name}(42)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_wrong_base"}})
''')
            tests.append(f'''
def test_{nt_name}_from_none():
    """Create {nt_name} from None."""
    try:
        val = {nt_name}(None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_none"}})
''')
        elif base_type == "int":
            tests.append(f'''
def test_{nt_name}_from_int():
    """Create {nt_name} from a plain int."""
    try:
        val = {nt_name}(42)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_from_base"}})
''')
            tests.append(f'''
def test_{nt_name}_from_string():
    """Create {nt_name} from a string (wrong base type)."""
    try:
        val = {nt_name}("not_an_int")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_wrong_base"}})
''')
        elif base_type == "bytes":
            tests.append(f'''
def test_{nt_name}_from_bytes():
    """Create {nt_name} from bytes."""
    try:
        val = {nt_name}(b"test_bytes")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_from_base"}})
''')
            tests.append(f'''
def test_{nt_name}_from_string():
    """Create {nt_name} from a string (wrong base type for bytes)."""
    try:
        val = {nt_name}("not_bytes")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_wrong_base"}})
''')

    for func in _get_functions(tree):
        if func.name.startswith("_") or func.name not in env:
            continue
        for arg in func.args.args:
            if not arg.annotation:
                continue
            ann_str = _get_annotation_str(arg.annotation)
            for nt_name, base_type, nt_lineno in newtypes:
                if nt_name in ann_str and nt_name in env:
                    other_newtypes = [(n, b) for n, b, _ in newtypes if n != nt_name and b == base_type]
                    for other_name, _ in other_newtypes:
                        if other_name in env:
                            if base_type == "str":
                                wrong_val = f'{other_name}("wrong_newtype")'
                            elif base_type == "int":
                                wrong_val = f'{other_name}(99)'
                            elif base_type == "bytes":
                                wrong_val = f'{other_name}(b"wrong")'
                            else:
                                continue
                            tests.append(f'''
def test_{func.name}_wrong_newtype_{arg.arg}_{other_name}():
    """Call {func.name} passing {other_name} where {nt_name} expected for param '{arg.arg}'."""
    try:
        {func.name}({arg.arg}={wrong_val})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {func.lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "wrong_newtype"}})
''')
    return tests


def _gen_inheritance_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    overrides = _find_overridden_methods(tree)
    for cls_name, base_name, method_name, lineno in overrides:
        cls = env.get(cls_name)
        base_cls = env.get(base_name)
        if cls is None or base_cls is None:
            continue
        if not isinstance(cls, type) or not isinstance(base_cls, type):
            continue
        if _cls_needs_init_args(cls) or _is_protocol_class(cls):
            continue

        tests.append(f'''
def test_{cls_name}_{method_name}_via_base_ref():
    """Call {cls_name}.{method_name} through a {base_name} reference."""
    try:
        obj: {base_name} = {cls_name}()
        result = obj.{method_name}()
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "override_via_base"}})
''')
        tests.append(f'''
def test_{cls_name}_{method_name}_direct():
    """Call {cls_name}.{method_name} directly."""
    try:
        obj = {cls_name}()
        result = obj.{method_name}()
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "override_direct"}})
''')
        tests.append(f'''
def test_{cls_name}_isinstance_{base_name}():
    """Verify {cls_name} is an instance of {base_name}."""
    try:
        obj = {cls_name}()
        if not isinstance(obj, {base_name}):
            BUGS.append({{"line": {lineno}, "type": "InheritanceError", "error": "{cls_name} is not instance of {base_name}", "test": "isinstance_check"}})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "isinstance_check"}})
''')
        tests.append(f'''
def test_{cls_name}_super_{method_name}():
    """Verify super().{method_name}() works from {cls_name}."""
    try:
        obj = {cls_name}()
        base_method = getattr(super(type(obj), obj), "{method_name}", None)
        if base_method is not None:
            result = base_method()
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "super_call"}})
''')
    return tests


def _gen_decorator_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    decorated = _find_decorated_functions(tree)
    for func_name, decorators, lineno, class_name, func_node in decorated:
        if func_name.startswith("_"):
            continue

        if class_name:
            cls = env.get(class_name)
            if cls is None or not isinstance(cls, type):
                continue
            if _cls_needs_init_args(cls) or _is_protocol_class(cls) or _is_abstract_class(cls):
                continue
            method = getattr(cls, func_name, None)
            if method is None:
                continue
            if inspect.iscoroutinefunction(method):
                continue
            if inspect.isgeneratorfunction(method):
                continue
            if _callable_needs_args(method):
                continue

            tests.append(f'''
def test_{class_name}_{func_name}_decorated_callable():
    """Verify decorated method {class_name}.{func_name} is callable."""
    try:
        obj = {class_name}()
        method = getattr(obj, "{func_name}", None)
        if method is None:
            BUGS.append({{"line": {lineno}, "type": "AttributeError", "error": "{class_name} has no method {func_name} after decoration", "test": "decorated_callable"}})
        elif not callable(method):
            BUGS.append({{"line": {lineno}, "type": "TypeError", "error": "{class_name}.{func_name} is not callable after decoration", "test": "decorated_callable"}})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "decorated_callable"}})
''')
            if not _ast_callable_needs_args(func_node, in_class=True):
                tests.append(f'''
def test_{class_name}_{func_name}_no_args():
    """Call decorated {class_name}.{func_name} with no extra args."""
    try:
        obj = {class_name}()
        result = obj.{func_name}()
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "decorated_no_args"}})
''')
        else:
            fn = env.get(func_name)
            if fn is None or not callable(fn):
                continue
            tests.append(f'''
def test_{func_name}_decorated_callable():
    """Verify decorated function {func_name} is callable."""
    if not callable({func_name}):
        BUGS.append({{"line": {lineno}, "type": "TypeError", "error": "{func_name} is not callable after decoration", "test": "decorated_callable"}})
''')
            if not _ast_callable_needs_args(func_node, in_class=False):
                tests.append(f'''
def test_{func_name}_no_args():
    """Call decorated function {func_name} with no args."""
    try:
        result = {func_name}()
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "decorated_no_args"}})
''')
    return tests


def _gen_callable_param_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    callable_params = _find_callable_params(tree)
    for func_name, param_name, lineno in callable_params:
        if func_name.startswith("_") or func_name not in env:
            continue

        tests.append(f'''
def test_{func_name}_none_callable():
    """Call {func_name} with None for Callable param '{param_name}'."""
    try:
        {func_name}({param_name}=None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "none_callable"}})
''')
        tests.append(f'''
def test_{func_name}_string_callable():
    """Call {func_name} with a string for Callable param '{param_name}'."""
    try:
        {func_name}({param_name}="not_callable")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "string_callable"}})
''')
        tests.append(f'''
def test_{func_name}_wrong_arity_callable():
    """Call {func_name} with a zero-arg callable for param '{param_name}'."""
    try:
        {func_name}({param_name}=lambda: None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "wrong_arity_callable"}})
''')
    return tests


def _gen_protocol_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    for cls_node in _get_classes(tree):
        cls = env.get(cls_node.name)
        if cls is None or not isinstance(cls, type):
            continue
        is_protocol = _is_protocol_class(cls)
        if not is_protocol:
            continue

        protocol_attrs = getattr(cls, "__protocol_attrs__", set())
        abstract_methods = getattr(cls, "__abstractmethods__", set())
        required = protocol_attrs or abstract_methods
        if not required:
            continue

        callable_methods = set()
        for attr_name in required:
            proto_attr = getattr(cls, attr_name, None)
            if callable(proto_attr) or isinstance(proto_attr, (classmethod, staticmethod, property)):
                callable_methods.add(attr_name)
            else:
                try:
                    hints = typing.get_type_hints(cls, include_extras=True)
                    if attr_name in hints:
                        ann = hints[attr_name]
                        ann_str = str(ann)
                        if "ClassVar" in ann_str:
                            continue
                except Exception:
                    pass
                if proto_attr is not None and not callable(proto_attr):
                    continue
                callable_methods.add(attr_name)

        concrete_classes = []
        for name, obj in env.items():
            if isinstance(obj, type) and obj is not cls and not _is_protocol_class(obj):
                if all(hasattr(obj, m) for m in required):
                    if not _cls_needs_init_args(obj):
                        concrete_classes.append(name)

        for concrete_name in concrete_classes:
            for method_name in callable_methods:
                concrete_cls = env.get(concrete_name)
                concrete_attr = getattr(concrete_cls, method_name, None) if concrete_cls else None
                if concrete_attr is not None and not callable(concrete_attr):
                    continue
                tests.append(f'''
def test_{concrete_name}_has_{method_name}():
    """Verify {concrete_name} has required protocol method '{method_name}'."""
    try:
        obj = {concrete_name}()
        method = getattr(obj, "{method_name}", None)
        if method is None:
            BUGS.append({{"line": {cls_node.lineno}, "type": "AttributeError", "error": "{concrete_name} missing protocol method {method_name}", "test": "protocol_method_exists"}})
        elif not callable(method):
            BUGS.append({{"line": {cls_node.lineno}, "type": "TypeError", "error": "{concrete_name}.{method_name} is not callable", "test": "protocol_method_callable"}})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {cls_node.lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "protocol_check"}})
''')

        tests.append(f'''
def test_{cls_node.name}_non_conforming_object():
    """Pass a non-conforming object where Protocol {cls_node.name} is expected."""
    class _FakeNonConforming:
        pass
    fake = _FakeNonConforming()
    for func_name_check, func_obj in [(k, v) for k, v in globals().items() if callable(v)]:
        pass
''')

    return tests


def _gen_classmethod_super_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    for cls_node in _get_classes(tree):
        cls = env.get(cls_node.name)
        if cls is None or not isinstance(cls, type):
            continue
        bases = [_get_annotation_str(b) for b in cls_node.bases]
        if not bases:
            continue
        for item in cls_node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_classmethod = any(
                _get_annotation_str(d) == "classmethod" for d in item.decorator_list
            )
            if not is_classmethod:
                continue

            cm = getattr(cls, item.name, None)
            if cm is not None and inspect.iscoroutinefunction(cm):
                continue
            if cm is not None and inspect.isgeneratorfunction(cm):
                continue
            if cm is not None and _callable_needs_args(cm):
                continue

            tests.append(f'''
def test_{cls_node.name}_{item.name}_classmethod_call():
    """Call classmethod {cls_node.name}.{item.name}() directly."""
    try:
        result = {cls_node.name}.{item.name}()
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {item.lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "classmethod_call"}})
''')

            for base_name in bases:
                if base_name in env:
                    base_cls = env[base_name]
                    if isinstance(base_cls, type) and hasattr(base_cls, item.name):
                        tests.append(f'''
def test_{cls_node.name}_{item.name}_matches_base_{base_name}():
    """Verify {cls_node.name}.{item.name} return matches {base_name}.{item.name}."""
    try:
        derived_result = {cls_node.name}.{item.name}()
        base_result = {base_name}.{item.name}()
        if type(derived_result) != type(base_result):
            BUGS.append({{"line": {item.lineno}, "type": "ReturnTypeMismatch", "error": f"{{type(derived_result).__name__}} vs {{type(base_result).__name__}}", "test": "classmethod_return_type"}})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({{"line": {item.lineno}, "type": type(e).__name__, "error": str(e)[:200], "test": "classmethod_return_type"}})
''')
    return tests


def _gen_main_block_tests(tree: ast.Module, env: dict) -> list[str]:
    tests = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__":
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        call_str = ast.unparse(stmt.value)
                        referenced_names = [n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)]
                        if not all(name in env for name in referenced_names):
                            continue
                        safe_name = re.sub(r"[^\w]", "_", call_str)[:40]
                        tests.append(f'''
def test_main_call_{safe_name}():
    """Execute main block call: {call_str}"""
    import traceback as _tb, sys as _sys
    try:
        {call_str}
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        _fault_line = {stmt.lineno}
        _root = e
        while getattr(_root, '__cause__', None) or getattr(_root, '__context__', None):
            _root = _root.__cause__ or _root.__context__
        _frames = _tb.extract_tb(_root.__traceback__)
        if _frames:
            _fault_line = _frames[-1].lineno - _SOURCE_LINE_OFFSET
        BUGS.append({{"line": _fault_line, "type": type(e).__name__, "error": str(e)[:200], "test": "main_block_call"}})
''')
                    elif isinstance(stmt, ast.Assign):
                        assign_str = ast.unparse(stmt)
                        referenced_names = [n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)]
                        if not all(name in env for name in referenced_names):
                            continue
                        safe_name = re.sub(r"[^\w]", "_", assign_str)[:40]
                        tests.append(f'''
def test_main_assign_{safe_name}():
    """Execute main block assignment: {assign_str}"""
    import traceback as _tb, sys as _sys
    try:
        {assign_str}
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        _fault_line = {stmt.lineno}
        _root = e
        while getattr(_root, '__cause__', None) or getattr(_root, '__context__', None):
            _root = _root.__cause__ or _root.__context__
        _frames = _tb.extract_tb(_root.__traceback__)
        if _frames:
            _fault_line = _frames[-1].lineno - _SOURCE_LINE_OFFSET
        BUGS.append({{"line": _fault_line, "type": type(e).__name__, "error": str(e)[:200], "test": "main_block_assign"}})
''')
    return tests


def generate_test_file(
    source_code: str,
    filename: str,
    env: dict,
    tree: ast.Module,
) -> tuple[str, list[str]]:
    all_tests: list[str] = []
    patterns_found: list[str] = []

    generators = [
        ("tuple_length", _gen_tuple_tests),
        ("typeguard_narrowing", _gen_typeguard_tests),
        ("newtype", _gen_newtype_tests),
        ("inheritance_override", _gen_inheritance_tests),
        ("decorator_signature", _gen_decorator_tests),
        ("callable_param", _gen_callable_param_tests),
        ("protocol_conformance", _gen_protocol_tests),
        ("classmethod_super", _gen_classmethod_super_tests),
        ("main_block_replay", _gen_main_block_tests),
    ]

    for pattern_name, gen_fn in generators:
        tests = gen_fn(tree, env)
        if tests:
            patterns_found.append(f"{pattern_name} ({len(tests)} tests)")
            all_tests.extend(tests)

    header = f'''"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: {filename}
Patterns detected: {len(patterns_found)}
  {chr(10).join("  - " + p for p in patterns_found) if patterns_found else "  (none)"}
Test cases generated: {len(all_tests)}
"""

'''

    source_section = "# --- Original source ---\n\n" + source_code + "\n\n"
    source_line_offset = header.count("\n") + 2
    bugs_init = f"# --- Test infrastructure ---\nBUGS = []\n_SOURCE_LINE_OFFSET = {source_line_offset}\n\n"
    tests_section = "# --- Test cases ---\n" + "\n".join(all_tests) + "\n"

    runner = '''
# --- Runner ---
if __name__ == "__main__":
    import sys
    _test_fns = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    print(f"Running {len(_test_fns)} targeted tests...")
    _passed = 0
    _failed = 0
    for _name, _fn in _test_fns:
        try:
            _fn()
            _passed += 1
        except Exception as _e:
            _failed += 1
    print(f"Passed: {_passed}, Failed: {_failed}, Bugs found: {len(BUGS)}")
    for _bug in BUGS:
        print(f"  BUG L{_bug['line']} [{_bug['type']}] {_bug['error']}")
'''

    full_file = header + source_section + bugs_init + tests_section + runner
    return full_file, patterns_found


def run_targeted_tests(
    source_code: str,
    output_dir: str | None = None,
    filename: str = "unknown.py",
) -> list[TypeBug]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    env = _build_source_env(source_code)
    if env is None:
        return []

    test_file_content, patterns_found = generate_test_file(source_code, filename, env, tree)

    if not patterns_found:
        return []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        test_path = os.path.join(output_dir, "targeted_tests.py")
        try:
            with open(test_path, "w") as f:
                f.write(test_file_content)
        except Exception:
            pass

    test_env: dict[str, Any] = {"__name__": "__targeted_test_runner__", "__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(compile(test_file_content, "<targeted_tests>", "exec"), test_env)
    except Exception:
        return []

    test_fns = [(name, fn) for name, fn in test_env.items() if name.startswith("test_") and callable(fn)]
    for name, fn in test_fns:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                fn()
        except Exception:
            pass

    raw_bugs = test_env.get("BUGS", [])
    bugs: list[TypeBug] = []
    seen: set[tuple[int, str, str]] = set()
    _NON_TYPE_EXCEPTIONS = {"ValidationError", "PydanticValidationError"}
    for b in raw_bugs:
        if b["type"] in _NON_TYPE_EXCEPTIONS:
            continue
        error_msg = b.get("error", "")
        if b["type"] == "TypeError" and _is_missing_arg_error(TypeError(error_msg)):
            continue
        if b["type"] == "KeyError" and _is_deliberate_key_error(KeyError(error_msg)):
            continue
        if b["type"] == "ValueError" and not _is_type_correlated_error(
            ValueError(error_msg)
        ):
            continue
        key = (b["line"], b["type"], b.get("test", ""))
        if key in seen:
            continue
        seen.add(key)
        bugs.append(TypeBug(
            line=b["line"],
            bug_type=b["type"],
            message=f'{b.get("test", "targeted")}: {b["error"]}',
            source="tier2_targeted",
            confidence=0.90,
            details={
                "test_name": b.get("test", "unknown"),
                "error": b["error"],
            },
        ))

    return bugs

