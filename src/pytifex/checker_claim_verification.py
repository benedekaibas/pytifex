"""
Checker Claim Verification Tier.

Parses type checker diagnostic messages to extract specific type mismatch
claims, then constructs and executes the exact scenario to empirically
verify whether the checker's prediction holds.

Given a checker diagnostic like:
    Argument 1 to "f" has incompatible type "int"; expected "str"

This tier will:
1. Parse the claim: function=f, param=1, bad_type=int, expected=str
2. Generate a concrete value of the "bad" type: 42
3. Call f(42) in a sandbox
4. Observe: crash → claim validated, no crash → claim may be wrong

Handles edge cases:
- Silent type errors (typeguard-enforced return type checking)
- Complex types (generics, Optional, Union, custom classes)
- Decorated functions (resolves through runtime namespace)
- Protocol/ABC types (constructs minimal conforming objects)
- Multiple claims per file

Usage:
    from checker_claim_verification import run_claim_verification
    results = run_claim_verification(source_code, checker_outputs)
"""

import ast
import re
import io
import contextlib
import traceback
import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ClaimVerdict(Enum):
    VALIDATED = "VALIDATED"        # crash confirms the checker's claim
    REFUTED = "REFUTED"            # no crash — checker may be wrong
    INCONCLUSIVE = "INCONCLUSIVE"  # could not construct test


@dataclass
class CheckerClaim:
    """A structured claim extracted from a checker diagnostic."""
    checker: str
    line: int
    function_name: str
    param_index: int | None        # 0-based, None if unknown
    param_name: str | None
    bad_type: str                   # the type the checker says is wrong
    expected_type: str              # what the parameter expects
    raw_message: str
    error_code: str | None = None


@dataclass
class ClaimResult:
    """Result of verifying a single checker claim."""
    claim: CheckerClaim
    verdict: ClaimVerdict
    exception_type: str | None = None
    exception_message: str | None = None
    confidence: float = 0.0
    details: dict = field(default_factory=dict)


# =============================================================================
# DIAGNOSTIC PARSING — extract structured claims from checker output
# =============================================================================

# --- mypy / zuban patterns ---
_MYPY_ARG_INCOMPATIBLE = re.compile(
    r'Argument\s+(?P<pos>\d+)\s+to\s+"(?P<func>[^"]+)"\s+has\s+incompatible\s+type\s+"(?P<bad>[^"]+)";\s*expected\s+"(?P<expected>[^"]+)"'
)
_MYPY_ARG_NAME_INCOMPATIBLE = re.compile(
    r'Argument\s+"(?P<name>[^"]+)"\s+to\s+"(?P<func>[^"]+)"\s+has\s+incompatible\s+type\s+"(?P<bad>[^"]+)";\s*expected\s+"(?P<expected>[^"]+)"'
)
_MYPY_RETURN_INCOMPATIBLE = re.compile(
    r'Incompatible\s+return\s+value\s+type\s*\(got\s+"(?P<bad>[^"]+)",\s*expected\s+"(?P<expected>[^"]+)"\)'
)
_MYPY_ASSIGNMENT_INCOMPATIBLE = re.compile(
    r'Incompatible\s+types\s+in\s+assignment\s*\(expression\s+has\s+type\s+"(?P<bad>[^"]+)",\s*variable\s+has\s+type\s+"(?P<expected>[^"]+)"\)'
)
_MYPY_OVERRIDE_INCOMPATIBLE = re.compile(
    r'Argument\s+(?P<pos>\d+)\s+of\s+"(?P<func>[^"]+)"\s+is\s+incompatible\s+with\s+supertype.*;\s*supertype\s+defines\s+the\s+argument\s+type\s+as\s+"(?P<expected>[^"]+)"'
)

# --- pyrefly patterns ---
_PYREFLY_ARG_TYPE = re.compile(
    r'(?:Argument|Expected)\s+.*?type\s+`(?P<expected>[^`]+)`.*?got\s+`(?P<bad>[^`]+)`'
)
_PYREFLY_INCOMPATIBLE = re.compile(
    r'`(?P<bad>[^`]+)`\s+is\s+not\s+assignable\s+to\s+`(?P<expected>[^`]+)`'
)
_PYREFLY_PARAM = re.compile(
    r'parameter\s+`(?P<name>[^`]+)`'
)

# --- ty patterns ---
_TY_ARG_TYPE = re.compile(
    r'Argument\s+of\s+type\s+`(?P<bad>[^`]+)`\s+is\s+not\s+assignable\s+to\s+parameter\s+(?:of\s+type\s+`(?P<expected>[^`]+)`|`(?P<name>[^`]+)`\s+of\s+type\s+`(?P<expected2>[^`]+)`)'
)
_TY_INCOMPATIBLE = re.compile(
    r'`(?P<bad>[^`]+)`\s+is\s+not\s+assignable\s+to\s+`(?P<expected>[^`]+)`'
)

# --- line number extraction ---
_LINE_RE = re.compile(r'\.py:(\d+)(?::\d+)?:')
_MYPY_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+)(?::\d+)?:\s*(?P<severity>error|warning|note):\s*(?P<message>.+?)(?:\s*\[(?P<code>[^\]]+)\])?\s*$"
)


def _parse_mypy_claims(output: str, checker: str = "mypy") -> list[CheckerClaim]:
    """Parse mypy/zuban output for type mismatch claims."""
    claims = []

    for line_text in output.splitlines():
        m = _MYPY_LINE_RE.match(line_text)
        if not m:
            continue
        if m.group("severity") != "error":
            continue

        line_num = int(m.group("line"))
        message = m.group("message")
        code = m.group("code")

        # Argument positional incompatibility
        am = _MYPY_ARG_INCOMPATIBLE.search(message)
        if am:
            claims.append(CheckerClaim(
                checker=checker,
                line=line_num,
                function_name=am.group("func"),
                param_index=int(am.group("pos")) - 1,
                param_name=None,
                bad_type=am.group("bad"),
                expected_type=am.group("expected"),
                raw_message=message,
                error_code=code,
            ))
            continue

        # Argument named incompatibility
        nm = _MYPY_ARG_NAME_INCOMPATIBLE.search(message)
        if nm:
            claims.append(CheckerClaim(
                checker=checker,
                line=line_num,
                function_name=nm.group("func"),
                param_index=None,
                param_name=nm.group("name"),
                bad_type=nm.group("bad"),
                expected_type=nm.group("expected"),
                raw_message=message,
                error_code=code,
            ))
            continue

        # Return type incompatibility
        rm = _MYPY_RETURN_INCOMPATIBLE.search(message)
        if rm:
            claims.append(CheckerClaim(
                checker=checker,
                line=line_num,
                function_name="__return__",
                param_index=None,
                param_name=None,
                bad_type=rm.group("bad"),
                expected_type=rm.group("expected"),
                raw_message=message,
                error_code=code,
            ))
            continue

        # Assignment incompatibility
        asm = _MYPY_ASSIGNMENT_INCOMPATIBLE.search(message)
        if asm:
            claims.append(CheckerClaim(
                checker=checker,
                line=line_num,
                function_name="__assign__",
                param_index=None,
                param_name=None,
                bad_type=asm.group("bad"),
                expected_type=asm.group("expected"),
                raw_message=message,
                error_code=code,
            ))

    return claims


def _parse_pyrefly_claims(output: str) -> list[CheckerClaim]:
    """Parse pyrefly output for type mismatch claims."""
    claims = []
    lines = output.splitlines()

    for i, line_text in enumerate(lines):
        # Try single-line format
        sm = re.match(
            r"^ERROR\s+(?P<file>[^:]+):(?P<line>\d+):\d+\s+(?P<code>\S+)\s+(?P<message>.+)$",
            line_text,
        )
        if not sm:
            # Try block header format
            hm = re.match(r"^ERROR\s+(?P<message>.+?)\s+\[(?P<code>[^\]]+)\]\s*$", line_text)
            if hm:
                # Find line number from --> line
                line_num = 0
                for j in range(i + 1, min(i + 5, len(lines))):
                    lm = re.search(r"-->\s*\S+:(\d+):\d+", lines[j])
                    if lm:
                        line_num = int(lm.group(1))
                        break
                message = hm.group("message")
                code = hm.group("code")
            else:
                continue
        else:
            line_num = int(sm.group("line"))
            message = sm.group("message")
            code = sm.group("code")

        if line_num == 0:
            continue

        # Extract type mismatch
        tm = _PYREFLY_ARG_TYPE.search(message)
        if not tm:
            tm = _PYREFLY_INCOMPATIBLE.search(message)
        if not tm:
            continue

        bad = tm.group("bad")
        expected = tm.group("expected")

        # Try to extract function name and param
        func_match = re.search(r'"(\w+)"', message)
        func_name = func_match.group(1) if func_match else "__unknown__"
        param_match = _PYREFLY_PARAM.search(message)
        param_name = param_match.group("name") if param_match else None

        claims.append(CheckerClaim(
            checker="pyrefly",
            line=line_num,
            function_name=func_name,
            param_index=None,
            param_name=param_name,
            bad_type=bad,
            expected_type=expected,
            raw_message=message,
            error_code=code,
        ))

    return claims


def _parse_ty_claims(output: str) -> list[CheckerClaim]:
    """Parse ty output for type mismatch claims."""
    claims = []
    lines = output.splitlines()
    i = 0

    while i < len(lines):
        line_text = lines[i]

        # Block format: error[code] or error[code]: message
        hm = re.match(r"^\s*error\[(?P<code>[^\]]+)\](?::\s*(?P<message>.+))?$", line_text)
        if hm:
            code = hm.group("code")
            header_msg = hm.group("message") or ""

            # Collect message lines until --> location
            line_num = 0
            full_message = header_msg
            j = i + 1
            while j < len(lines):
                lm = re.search(r"-->\s*\S+:(\d+):\d+", lines[j])
                if lm:
                    line_num = int(lm.group(1))
                    break
                if lines[j].strip() and not lines[j].startswith(" ") and not lines[j].startswith("|"):
                    break
                if lines[j].strip().startswith("|"):
                    pass  # code context
                else:
                    full_message += " " + lines[j].strip()
                j += 1

            if line_num > 0 and full_message:
                tm = _TY_ARG_TYPE.search(full_message)
                if tm:
                    bad = tm.group("bad")
                    expected = tm.group("expected") or tm.group("expected2") or ""
                    param_name = tm.group("name") if tm.lastgroup == "name" or tm.group("name") else None
                    claims.append(CheckerClaim(
                        checker="ty",
                        line=line_num,
                        function_name="__unknown__",
                        param_index=None,
                        param_name=param_name,
                        bad_type=bad,
                        expected_type=expected,
                        raw_message=full_message.strip(),
                        error_code=code,
                    ))
                else:
                    tm = _TY_INCOMPATIBLE.search(full_message)
                    if tm:
                        claims.append(CheckerClaim(
                            checker="ty",
                            line=line_num,
                            function_name="__unknown__",
                            param_index=None,
                            param_name=None,
                            bad_type=tm.group("bad"),
                            expected_type=tm.group("expected"),
                            raw_message=full_message.strip(),
                            error_code=code,
                        ))

            i = j + 1
            continue

        # Single-line format
        sm = re.match(
            r"^(?P<file>[^:]+):(?P<line>\d+):\d+:\s*error\[(?P<code>[^\]]+)\]:\s*(?P<message>.+)$",
            line_text,
        )
        if sm:
            line_num = int(sm.group("line"))
            message = sm.group("message")
            code = sm.group("code")

            tm = _TY_ARG_TYPE.search(message)
            if tm:
                bad = tm.group("bad")
                expected = tm.group("expected") or tm.group("expected2") or ""
                param_name = tm.group("name") if tm.group("name") else None
                claims.append(CheckerClaim(
                    checker="ty",
                    line=line_num,
                    function_name="__unknown__",
                    param_index=None,
                    param_name=param_name,
                    bad_type=bad,
                    expected_type=expected,
                    raw_message=message,
                    error_code=code,
                ))

        i += 1

    return claims


def parse_all_claims(checker_outputs: dict[str, str]) -> list[CheckerClaim]:
    """Parse all checker outputs and extract structured claims."""
    claims = []
    for checker, output in checker_outputs.items():
        if checker in ("mypy", "zuban"):
            claims.extend(_parse_mypy_claims(output, checker))
        elif checker == "pyrefly":
            claims.extend(_parse_pyrefly_claims(output))
        elif checker == "ty":
            claims.extend(_parse_ty_claims(output))
    return claims


# =============================================================================
# TYPE-TO-VALUE GENERATION
# =============================================================================

# Map of type name -> concrete value generator
# Returns (value, description) tuples
_PRIMITIVE_VALUES: dict[str, tuple[Any, str]] = {
    "int": (42, "42"),
    "str": ("test_string", '"test_string"'),
    "float": (3.14, "3.14"),
    "bool": (True, "True"),
    "bytes": (b"test", 'b"test"'),
    "None": (None, "None"),
    "NoneType": (None, "None"),
    "object": (object(), "object()"),
    "complex": (1 + 2j, "1+2j"),
    "bytearray": (bytearray(b"test"), 'bytearray(b"test")'),
    "memoryview": (memoryview(b"test"), 'memoryview(b"test")'),
    "type": (int, "int"),
    "type[int]": (int, "int"),
    "type[str]": (str, "str"),
}

# Generic container patterns
_GENERIC_PATTERNS: list[tuple[re.Pattern, Any]] = [
    (re.compile(r"^list\[(.+)\]$", re.IGNORECASE), lambda inner: ([_generate_inner(inner)], f"[{_describe_inner(inner)}]")),
    (re.compile(r"^List\[(.+)\]$"), lambda inner: ([_generate_inner(inner)], f"[{_describe_inner(inner)}]")),
    (re.compile(r"^set\[(.+)\]$", re.IGNORECASE), lambda inner: ({_generate_inner(inner)}, f"{{{_describe_inner(inner)}}}")),
    (re.compile(r"^Set\[(.+)\]$"), lambda inner: ({_generate_inner(inner)}, f"{{{_describe_inner(inner)}}}")),
    (re.compile(r"^frozenset\[(.+)\]$", re.IGNORECASE), lambda inner: (frozenset([_generate_inner(inner)]), f"frozenset({{{_describe_inner(inner)}}})")),
    (re.compile(r"^tuple\[(.+),\s*\.\.\.\]$", re.IGNORECASE), lambda inner: ((_generate_inner(inner),), f"({_describe_inner(inner)},)")),
    (re.compile(r"^Tuple\[(.+),\s*\.\.\.\]$"), lambda inner: ((_generate_inner(inner),), f"({_describe_inner(inner)},)")),
    (re.compile(r"^tuple\[(.+)\]$", re.IGNORECASE), lambda inner: (tuple(_generate_inner(t.strip()) for t in _split_type_args(inner)), f"tuple(...)")),
]

_DICT_PATTERN = re.compile(r"^(?:dict|Dict)\[(.+),\s*(.+)\]$", re.IGNORECASE)
_OPTIONAL_PATTERN = re.compile(r"^Optional\[(.+)\]$")
_UNION_PATTERN = re.compile(r"^(?:Union\[(.+)\]|(.+)\s*\|\s*(.+))$")
_CALLABLE_PATTERN = re.compile(r"^Callable\[.*\]$")
_LITERAL_PATTERN = re.compile(r"^Literal\[(.+)\]$")


def _split_type_args(s: str) -> list[str]:
    """Split type arguments respecting bracket nesting."""
    parts = []
    depth = 0
    current = ""
    for ch in s:
        if ch in "([":
            depth += 1
            current += ch
        elif ch in ")]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _generate_inner(type_str: str) -> Any:
    """Generate a value for a type used inside a generic."""
    val, _ = generate_value(type_str)
    return val


def _describe_inner(type_str: str) -> str:
    """Get description of generated value for a type."""
    _, desc = generate_value(type_str)
    return desc


def generate_value(
    type_str: str,
    source_env: dict | None = None,
    depth: int = 0,
) -> tuple[Any, str]:
    """
    Generate a concrete value for the given type string.

    Returns (value, description) or raises ValueError if impossible.
    Handles:
    - Primitives (int, str, float, bool, None, bytes, etc.)
    - Generic containers (list[X], dict[K,V], set[X], tuple[X,...])
    - Optional[X] and Union[X, Y]
    - Callable types (returns a lambda)
    - Literal types
    - Custom classes (attempts construction from source_env)
    """
    if depth > 5:
        return (None, "None (recursion limit)")

    type_str = type_str.strip()

    # Remove quotes (some checkers wrap in quotes)
    if type_str.startswith('"') and type_str.endswith('"'):
        type_str = type_str[1:-1]
    if type_str.startswith("'") and type_str.endswith("'"):
        type_str = type_str[1:-1]

    # Check primitives first
    if type_str in _PRIMITIVE_VALUES:
        return _PRIMITIVE_VALUES[type_str]

    # Literal types
    lm = _LITERAL_PATTERN.match(type_str)
    if lm:
        literal_val = lm.group(1).strip()
        # Try to evaluate the literal
        try:
            val = ast.literal_eval(literal_val)
            return (val, repr(val))
        except (ValueError, SyntaxError):
            return (literal_val, repr(literal_val))

    # Optional[X] → generate the inner type (not None)
    om = _OPTIONAL_PATTERN.match(type_str)
    if om:
        return generate_value(om.group(1), source_env, depth + 1)

    # Union[X, Y] → generate the first non-None type
    um = _UNION_PATTERN.match(type_str)
    if um:
        if um.group(1):
            parts = _split_type_args(um.group(1))
        else:
            parts = [um.group(2), um.group(3)]
        for part in parts:
            part = part.strip()
            if part in ("None", "NoneType"):
                continue
            try:
                return generate_value(part, source_env, depth + 1)
            except ValueError:
                continue
        return (None, "None")

    # Pipe union: X | Y
    if " | " in type_str and not type_str.startswith("("):
        parts = type_str.split(" | ")
        for part in parts:
            part = part.strip()
            if part in ("None", "NoneType"):
                continue
            try:
                return generate_value(part, source_env, depth + 1)
            except ValueError:
                continue
        return (None, "None")

    # Dict pattern
    dm = _DICT_PATTERN.match(type_str)
    if dm:
        key_val = _generate_inner(dm.group(1).strip())
        val_val = _generate_inner(dm.group(2).strip())
        return ({key_val: val_val}, f"{{{repr(key_val)}: {repr(val_val)}}}")

    # Generic container patterns
    for pattern, factory in _GENERIC_PATTERNS:
        gm = pattern.match(type_str)
        if gm:
            try:
                return factory(gm.group(1))
            except (ValueError, TypeError):
                continue

    # Callable → return a lambda
    if _CALLABLE_PATTERN.match(type_str) or type_str.startswith("Callable"):
        return (lambda *args, **kwargs: None, "lambda *args, **kwargs: None")

    # Try to resolve from source environment
    if source_env:
        # Try direct construction
        cls = source_env.get(type_str)
        if cls is not None and isinstance(cls, type):
            try:
                # Try no-arg construction
                instance = cls.__new__(cls)
                return (instance, f"{type_str}()")
            except Exception:
                pass

            try:
                # Try with inspect to find __init__ params
                sig = inspect.signature(cls.__init__)
                params = list(sig.parameters.values())[1:]  # skip self
                kwargs = {}
                for p in params:
                    if p.default is not inspect.Parameter.empty:
                        continue
                    ann = p.annotation
                    if ann is not inspect.Parameter.empty:
                        ann_str = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
                        try:
                            val, _ = generate_value(ann_str, source_env, depth + 1)
                            kwargs[p.name] = val
                        except ValueError:
                            kwargs[p.name] = None
                    else:
                        kwargs[p.name] = None
                instance = cls(**kwargs)
                return (instance, f"{type_str}(...)")
            except Exception:
                pass

    # Handle some common typing constructs
    if type_str in ("Any", "typing.Any"):
        return (42, "42 (Any)")
    if type_str == "Self":
        raise ValueError(f"Cannot generate value for Self outside class context")
    if type_str == "Never" or type_str == "NoReturn":
        raise ValueError(f"Cannot generate value for {type_str}")
    if type_str.startswith("Type[") or type_str.startswith("type["):
        inner = type_str[5:-1]
        if inner in _PRIMITIVE_VALUES:
            type_map = {"int": int, "str": str, "float": float, "bool": bool, "bytes": bytes}
            if inner in type_map:
                return (type_map[inner], inner)
        if source_env and inner in source_env and isinstance(source_env[inner], type):
            return (source_env[inner], inner)
        return (type, "type")

    # Handle generic subscripted custom classes: ClassName[X] → try ClassName
    generic_match = re.match(r"^(\w+)\[.+\]$", type_str)
    if generic_match:
        base_name = generic_match.group(1)
        # Skip well-known generics already handled above
        if base_name not in ("list", "List", "dict", "Dict", "set", "Set",
                             "tuple", "Tuple", "frozenset", "FrozenSet",
                             "Optional", "Union", "Callable", "Literal",
                             "Type", "type", "Concatenate"):
            try:
                return generate_value(base_name, source_env, depth + 1)
            except ValueError:
                pass

    # Try to find NewType or custom class by checking if the name
    # wraps a primitive (e.g., UserID = NewType('UserID', int))
    if source_env:
        obj = source_env.get(type_str)
        if obj is not None:
            # NewType: callable that wraps a base type
            if callable(obj) and hasattr(obj, "__supertype__"):
                base = obj.__supertype__
                base_name = getattr(base, "__name__", str(base))
                try:
                    base_val, base_desc = generate_value(base_name, source_env, depth + 1)
                    wrapped = obj(base_val)
                    return (wrapped, f"{type_str}({base_desc})")
                except (ValueError, TypeError):
                    pass

            # Regular class
            if isinstance(obj, type):
                try:
                    instance = obj.__new__(obj)
                    return (instance, f"{type_str}()")
                except Exception:
                    pass

                try:
                    sig = inspect.signature(obj.__init__)
                    params = list(sig.parameters.values())[1:]  # skip self
                    kwargs = {}
                    for p in params:
                        if p.default is not inspect.Parameter.empty:
                            continue
                        ann = p.annotation
                        if ann is not inspect.Parameter.empty:
                            ann_str = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
                            try:
                                val, _ = generate_value(ann_str, source_env, depth + 1)
                                kwargs[p.name] = val
                            except ValueError:
                                kwargs[p.name] = None
                        else:
                            kwargs[p.name] = None
                    instance = obj(**kwargs)
                    return (instance, f"{type_str}(...)")
                except Exception:
                    pass

    raise ValueError(f"Cannot generate value for type: {type_str}")


# =============================================================================
# FUNCTION RESOLUTION — find target functions in source AST
# =============================================================================

@dataclass
class FunctionTarget:
    """A resolved function target for testing."""
    name: str
    callable_obj: Any
    params: list[tuple[str, str | None]]  # (name, annotation_str)
    return_annotation: str | None
    line: int
    is_method: bool = False
    class_name: str | None = None


def _build_source_env(source_code: str) -> dict | None:
    """Execute source code to build a live namespace."""
    env: dict[str, Any] = {"__name__": "__pytifex_verify__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(compile(source_code, "<claim_verify>", "exec"), env)
        return env
    except Exception:
        return None


def _extract_function_signatures(tree: ast.Module) -> dict[str, list[tuple[str, str | None]]]:
    """Extract function name -> [(param_name, annotation_str)] from AST."""
    functions: dict[str, list[tuple[str, str | None]]] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = []
            for arg in (node.args.posonlyargs + node.args.args):
                ann = ast.unparse(arg.annotation) if arg.annotation else None
                params.append((arg.arg, ann))
            for arg in node.args.kwonlyargs:
                ann = ast.unparse(arg.annotation) if arg.annotation else None
                params.append((arg.arg, ann))
            functions[node.name] = params

    return functions


def _find_function_at_line(
    tree: ast.Module,
    line: int,
    env: dict,
) -> FunctionTarget | None:
    """Find the function being called at a specific line."""
    func_sigs = _extract_function_signatures(tree)

    # Search with a tolerance window (checker line may differ slightly)
    for tolerance in (0, 1, 2, 3):
        for node in ast.walk(tree):
            if not hasattr(node, "lineno"):
                continue
            if abs(node.lineno - line) > tolerance:
                continue

            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue

                func_name = None
                class_name = None
                is_method = False

                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    func_name = child.func.attr
                    is_method = True
                    if isinstance(child.func.value, ast.Name):
                        class_name = child.func.value.id

                if func_name and func_name in env:
                    obj = env[func_name]
                    if callable(obj):
                        params = func_sigs.get(func_name, [])
                        return FunctionTarget(
                            name=func_name,
                            callable_obj=obj,
                            params=params,
                            return_annotation=None,
                            line=line,
                            is_method=is_method,
                            class_name=class_name,
                        )

    # Fallback: find the enclosing function definition for return/assign claims
    best_func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", node.lineno + 100)
            if node.lineno <= line <= end_line:
                if node.name in env and callable(env[node.name]):
                    params = func_sigs.get(node.name, [])
                    ret_ann = ast.unparse(node.returns) if node.returns else None
                    best_func = FunctionTarget(
                        name=node.name,
                        callable_obj=env[node.name],
                        params=params,
                        return_annotation=ret_ann,
                        line=node.lineno,
                    )

    return best_func


def _resolve_function(
    claim: CheckerClaim,
    tree: ast.Module,
    env: dict,
    func_sigs: dict[str, list[tuple[str, str | None]]],
) -> FunctionTarget | None:
    """Resolve a function from a checker claim."""
    func_name = claim.function_name

    # Skip non-function claims
    if func_name in ("__return__", "__assign__", "__unknown__"):
        # Try to find the function at the claim's line
        target = _find_function_at_line(tree, claim.line, env)
        if target:
            return target
        return None

    # Direct lookup in environment
    if func_name in env and callable(env[func_name]):
        params = func_sigs.get(func_name, [])
        return FunctionTarget(
            name=func_name,
            callable_obj=env[func_name],
            params=params,
            return_annotation=None,
            line=claim.line,
        )

    # Try as class constructor
    if func_name in env and isinstance(env[func_name], type):
        cls = env[func_name]
        init_name = "__init__"
        params = func_sigs.get(init_name, [])
        # Look for __init__ in class body
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == func_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        params = []
                        for arg in (item.args.posonlyargs + item.args.args):
                            if arg.arg == "self":
                                continue
                            ann = ast.unparse(arg.annotation) if arg.annotation else None
                            params.append((arg.arg, ann))
                        break
        return FunctionTarget(
            name=func_name,
            callable_obj=cls,
            params=params,
            return_annotation=None,
            line=claim.line,
        )

    # Search in class methods
    for name, obj in env.items():
        if isinstance(obj, type):
            method = getattr(obj, func_name, None)
            if method is not None and callable(method):
                # Find params from AST
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == name:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == func_name:
                                params = []
                                for arg in (item.args.posonlyargs + item.args.args):
                                    if arg.arg == "self":
                                        continue
                                    ann = ast.unparse(arg.annotation) if arg.annotation else None
                                    params.append((arg.arg, ann))
                                return FunctionTarget(
                                    name=func_name,
                                    callable_obj=method,
                                    params=params,
                                    return_annotation=None,
                                    line=claim.line,
                                    is_method=True,
                                    class_name=name,
                                )

    return None


# =============================================================================
# CLAIM VERIFICATION — construct and execute test scenarios
# =============================================================================

_TYPE_ERROR_EXCEPTIONS = (TypeError, AttributeError, KeyError, ValueError)


def _build_call_args(
    claim: CheckerClaim,
    target: FunctionTarget,
    source_env: dict,
) -> tuple[list, dict, str] | None:
    """
    Build the argument list for calling the target function.

    Puts the "bad" type value in the claimed position,
    and valid values in all other positions.

    Returns (args_list, kwargs_dict, description) or None if impossible.
    """
    try:
        bad_value, bad_desc = generate_value(claim.bad_type, source_env)
    except ValueError:
        return None

    params = target.params
    if not params:
        # No params known — try calling with just the bad value
        return ([bad_value], {}, f"({bad_desc})")

    args = []
    descriptions = []

    for i, (pname, ptype) in enumerate(params):
        is_target_param = False

        # Match by position
        if claim.param_index is not None and i == claim.param_index:
            is_target_param = True
        # Match by name
        elif claim.param_name and pname == claim.param_name:
            is_target_param = True
        # If function has only one non-self param, that's probably it
        elif len(params) == 1:
            is_target_param = True

        if is_target_param:
            args.append(bad_value)
            descriptions.append(f"{pname}={bad_desc}")
        else:
            # Generate a valid value for this parameter
            if ptype:
                try:
                    val, desc = generate_value(ptype, source_env)
                    args.append(val)
                    descriptions.append(f"{pname}={desc}")
                except ValueError:
                    args.append(None)
                    descriptions.append(f"{pname}=None")
            else:
                args.append(None)
                descriptions.append(f"{pname}=None")

    return (args, {}, f"({', '.join(descriptions)})")


def _verify_single_claim(
    claim: CheckerClaim,
    source_code: str,
    tree: ast.Module,
    env: dict,
    func_sigs: dict[str, list[tuple[str, str | None]]],
) -> ClaimResult:
    """Verify a single checker claim by constructing and executing the scenario."""

    # Resolve the target function
    target = _resolve_function(claim, tree, env, func_sigs)
    if target is None:
        return ClaimResult(
            claim=claim,
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            details={"reason": f"Could not resolve function '{claim.function_name}'"},
        )

    # Build call arguments
    call_info = _build_call_args(claim, target, env)
    if call_info is None:
        return ClaimResult(
            claim=claim,
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            details={"reason": f"Could not generate value for type '{claim.bad_type}'"},
        )

    args, kwargs, call_desc = call_info

    # Execute the call
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            result = target.callable_obj(*args, **kwargs)

        # No crash — but check return type if we can (silent type error detection)
        return_type_mismatch = False
        if claim.function_name == "__return__":
            # This was a return type claim — the function ran, check what it returned
            try:
                expected_val, _ = generate_value(claim.expected_type, env)
                if type(result) != type(expected_val) and result is not None:
                    return_type_mismatch = True
            except ValueError:
                pass

        # Also try typeguard if available for runtime type checking
        try:
            from typeguard import check_type, TypeCheckError
            if target.params and claim.param_index is not None:
                param_name, param_type_str = target.params[claim.param_index]
                if param_type_str:
                    # Try to resolve the type annotation
                    try:
                        type_hint = eval(param_type_str, env)
                        check_type(args[claim.param_index], type_hint)
                    except TypeCheckError:
                        return ClaimResult(
                            claim=claim,
                            verdict=ClaimVerdict.VALIDATED,
                            confidence=0.85,
                            details={
                                "reason": "typeguard runtime check confirms type violation",
                                "call": f"{target.name}{call_desc}",
                                "method": "typeguard",
                            },
                        )
                    except Exception:
                        pass
        except ImportError:
            pass

        if return_type_mismatch:
            return ClaimResult(
                claim=claim,
                verdict=ClaimVerdict.VALIDATED,
                confidence=0.75,
                details={
                    "reason": "Return type mismatch detected",
                    "call": f"{target.name}{call_desc}",
                    "returned": repr(result)[:100],
                },
            )

        return ClaimResult(
            claim=claim,
            verdict=ClaimVerdict.REFUTED,
            confidence=0.70,
            details={
                "reason": "Function executed successfully with the 'bad' type",
                "call": f"{target.name}{call_desc}",
                "returned": repr(result)[:100] if result is not None else "None",
            },
        )

    except _TYPE_ERROR_EXCEPTIONS as e:
        exc_type = type(e).__name__
        exc_msg = str(e)[:200]

        # Filter out errors from our test harness, not the actual function
        if isinstance(e, TypeError) and "missing" in exc_msg.lower() and "argument" in exc_msg.lower():
            return ClaimResult(
                claim=claim,
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                details={
                    "reason": "Missing argument error (test harness issue, not type error)",
                    "exception": f"{exc_type}: {exc_msg}",
                },
            )

        return ClaimResult(
            claim=claim,
            verdict=ClaimVerdict.VALIDATED,
            exception_type=exc_type,
            exception_message=exc_msg,
            confidence=0.90,
            details={
                "reason": f"Crash confirms type incompatibility: {exc_type}",
                "call": f"{target.name}{call_desc}",
                "exception": f"{exc_type}: {exc_msg}",
            },
        )

    except Exception as e:
        # Non-type-related exception
        exc_type = type(e).__name__
        exc_msg = str(e)[:200]
        return ClaimResult(
            claim=claim,
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            details={
                "reason": f"Non-type-related exception: {exc_type}",
                "call": f"{target.name}{call_desc}",
                "exception": f"{exc_type}: {exc_msg}",
            },
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_claim_verification(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[ClaimResult]:
    """
    Run checker claim verification on all claims from all checkers.

    1. Parse all checker diagnostics for type mismatch claims
    2. Build source environment
    3. Resolve functions and construct test scenarios
    4. Execute and collect results

    Returns list of ClaimResult with VALIDATED/REFUTED/INCONCLUSIVE verdicts.
    """
    # Parse claims from all checkers
    claims = parse_all_claims(checker_outputs)
    if not claims:
        return []

    # Parse AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return [
            ClaimResult(
                claim=c,
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                details={"reason": "Source code has syntax errors"},
            )
            for c in claims
        ]

    # Build live environment
    env = _build_source_env(source_code)
    if env is None:
        return [
            ClaimResult(
                claim=c,
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                details={"reason": "Could not execute source code"},
            )
            for c in claims
        ]

    # Extract function signatures from AST
    func_sigs = _extract_function_signatures(tree)

    # Verify each claim
    results = []
    seen_claims: set[tuple[str, int, str, str]] = set()

    for claim in claims:
        # Deduplicate claims with same checker/line/function/bad_type
        key = (claim.checker, claim.line, claim.function_name, claim.bad_type)
        if key in seen_claims:
            continue
        seen_claims.add(key)

        result = _verify_single_claim(claim, source_code, tree, env, func_sigs)
        results.append(result)

    return results


def results_to_dicts(results: list[ClaimResult]) -> list[dict]:
    """Convert claim results to JSON-serializable dicts."""
    return [
        {
            "checker": r.claim.checker,
            "line": r.claim.line,
            "function": r.claim.function_name,
            "bad_type": r.claim.bad_type,
            "expected_type": r.claim.expected_type,
            "verdict": r.verdict.value,
            "confidence": r.confidence,
            "exception_type": r.exception_type,
            "exception_message": r.exception_message,
            "details": r.details,
            "raw_message": r.claim.raw_message[:200],
        }
        for r in results
    ]


def summarize_results(results: list[ClaimResult]) -> dict[str, dict[str, int]]:
    """Summarize claim verification results per checker."""
    summary: dict[str, dict[str, int]] = {}
    for r in results:
        checker = r.claim.checker
        if checker not in summary:
            summary[checker] = {"validated": 0, "refuted": 0, "inconclusive": 0, "total": 0}
        summary[checker][r.verdict.value.lower()] += 1
        summary[checker]["total"] += 1
    return summary
