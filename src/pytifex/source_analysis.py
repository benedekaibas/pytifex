"""
Source-Aware AST Analysis for PEP Typing Rule Compliance.

Analyzes Python source code independently of any type checker output to
determine where errors MUST exist according to PEP specifications. Produces
findings like "line X violates PEP Y, an error SHOULD exist here."

Only emits high-confidence findings for definitive violations — never
flags ambiguous cases. Uses only locally-defined symbols for resolution.

Usage:
    from .source_analysis import analyze_source
    findings = analyze_source(source_code)
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SourceFinding:
    line: int
    rule_id: str
    pep: int
    message: str
    expected_behavior: str  # always "error" NOTE: we only emit when an error MUST exist
    confidence: float


# CLASS/MODULE MODEL

@dataclass
class ParamInfo:
    name: str
    annotation: str
    has_default: bool
    kind: str  # "positional", "keyword", "var_positional", "var_keyword"


@dataclass
class MethodInfo:
    name: str
    line: int
    params: list[ParamInfo]
    return_annotation: str
    decorators: list[str]
    is_abstract: bool = False
    is_final: bool = False
    is_override: bool = False
    is_overload: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False


@dataclass
class ClassInfo:
    name: str
    line: int
    base_names: list[str]
    methods: dict[str, MethodInfo]
    attributes: dict[str, str]  # name -> annotation string
    decorators: list[str]
    is_protocol: bool = False
    is_typeddict: bool = False
    is_abc: bool = False
    is_final: bool = False
    is_runtime_checkable: bool = False
    is_abstract: bool = False
    type_params: list[str] = field(default_factory=list)
    typeddict_total: bool = True  # TypedDict total= keyword (default True per PEP 589)


@dataclass
class ModuleSymbols:
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    functions: dict[str, MethodInfo] = field(default_factory=dict)
    typevars: dict[str, str] = field(default_factory=dict)  # name -> kind (TypeVar/ParamSpec/TypeVarTuple)
    typevar_string_args: dict[str, str] = field(default_factory=dict)  # name -> string literal arg
    typevar_variance: dict[str, str] = field(default_factory=dict)  # name -> "covariant"/"contravariant"/"invariant"
    typevar_lines: dict[str, int] = field(default_factory=dict)  # name -> declaration line
    newtypes: dict[str, str] = field(default_factory=dict)  # name -> string literal arg
    newtype_lines: dict[str, int] = field(default_factory=dict)
    newtype_bases: dict[str, str] = field(default_factory=dict)  # name -> base type string
    final_vars: dict[str, int] = field(default_factory=dict)  # name -> line
    typing_imports: set[str] = field(default_factory=set)
    all_imports: dict[str, str] = field(default_factory=dict)  # name -> module


TYPING_PROTOCOL_NAMES = {"Protocol", "typing.Protocol", "typing_extensions.Protocol"}
TYPING_TYPEDDICT_NAMES = {"TypedDict", "typing.TypedDict", "typing_extensions.TypedDict"}
ABC_NAMES = {"ABC", "abc.ABC"}
ABSTRACT_BASE_NAMES = {"ABCMeta", "abc.ABCMeta"}

PARAM_KIND_MAP = {
    ast.arg: "positional",
}


def _get_decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(dec.attr)
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                names.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                names.append(dec.func.attr)
    return names


def _extract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> MethodInfo:
    decorators = _get_decorator_names(node)
    params: list[ParamInfo] = []

    defaults_offset = len(node.args.args) - len(node.args.defaults)
    for i, arg in enumerate(node.args.args):
        ann = ast.unparse(arg.annotation) if arg.annotation else ""
        has_default = i >= defaults_offset
        params.append(ParamInfo(arg.arg, ann, has_default, "positional"))

    for i, arg in enumerate(node.args.kwonlyargs):
        ann = ast.unparse(arg.annotation) if arg.annotation else ""
        default = node.args.kw_defaults[i]
        params.append(ParamInfo(arg.arg, ann, default is not None, "keyword"))

    if node.args.vararg:
        ann = ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else ""
        params.append(ParamInfo(node.args.vararg.arg, ann, False, "var_positional"))

    if node.args.kwarg:
        ann = ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else ""
        params.append(ParamInfo(node.args.kwarg.arg, ann, False, "var_keyword"))

    for arg in node.args.posonlyargs:
        ann = ast.unparse(arg.annotation) if arg.annotation else ""
        params.append(ParamInfo(arg.arg, ann, False, "positional"))

    ret = ast.unparse(node.returns) if node.returns else ""

    return MethodInfo(
        name=node.name,
        line=node.lineno,
        params=params,
        return_annotation=ret,
        decorators=decorators,
        is_abstract="abstractmethod" in decorators,
        is_final="final" in decorators,
        is_override="override" in decorators,
        is_overload="overload" in decorators,
        is_classmethod="classmethod" in decorators,
        is_staticmethod="staticmethod" in decorators,
        is_property="property" in decorators,
    )


def _has_type_params(node: ast.ClassDef) -> bool:
    if hasattr(node, 'type_params') and node.type_params:
        return True
    return False


def _extract_type_param_names(node: ast.ClassDef) -> list[str]:
    names = []
    if hasattr(node, 'type_params'):
        for tp in node.type_params:
            if hasattr(tp, 'name'):
                names.append(tp.name)
    return names


def _is_generic_base(base_name: str, symbols: ModuleSymbols) -> bool:
    return base_name in ("Generic", "Protocol", "TypedDict") or base_name in symbols.typing_imports


def build_module_symbols(tree: ast.Module) -> ModuleSymbols:
    symbols = ModuleSymbols()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _process_import(node, symbols)
        elif isinstance(node, ast.ClassDef):
            _process_class(node, symbols)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.functions[node.name] = _extract_method(node)
        elif isinstance(node, ast.Assign):
            _process_assignment(node, symbols)
        elif isinstance(node, ast.AnnAssign):
            _process_ann_assign(node, symbols)

    return symbols


def _process_import(node: ast.Import | ast.ImportFrom, symbols: ModuleSymbols):
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module in ("typing", "typing_extensions"):
            for alias in node.names:
                name = alias.asname or alias.name
                symbols.typing_imports.add(name)
                symbols.all_imports[name] = module
        elif module in ("abc",):
            for alias in node.names:
                name = alias.asname or alias.name
                symbols.all_imports[name] = module
        else:
            for alias in node.names:
                name = alias.asname or alias.name
                symbols.all_imports[name] = module


def _process_class(node: ast.ClassDef, symbols: ModuleSymbols):
    decorators = _get_decorator_names(node)
    base_names = []
    is_generic = _has_type_params(node)
    type_params = _extract_type_param_names(node)

    subscript_type_params: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Subscript):
            if isinstance(base.value, ast.Name):
                base_names.append(base.value.id)
                if base.value.id in ("Generic", "Protocol"):
                    is_generic = True
                    if isinstance(base.slice, ast.Tuple):
                        for elt in base.slice.elts:
                            if isinstance(elt, ast.Name):
                                subscript_type_params.append(elt.id)
                    elif isinstance(base.slice, ast.Name):
                        subscript_type_params.append(base.slice.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(ast.unparse(base))

    is_protocol = any(
        b in TYPING_PROTOCOL_NAMES or b == "Protocol"
        or (b in symbols.classes and symbols.classes[b].is_protocol)
        for b in base_names
    )
    is_typeddict = any(
        b in TYPING_TYPEDDICT_NAMES or b == "TypedDict"
        or (b in symbols.classes and symbols.classes[b].is_typeddict)
        for b in base_names
    )
    is_abc = any(b in ABC_NAMES or b == "ABC" for b in base_names)

    methods: dict[str, MethodInfo] = {}
    attributes: dict[str, str] = {}

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mi = _extract_method(item)
            methods[item.name] = mi
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name):
                ann = ast.unparse(item.annotation) if item.annotation else ""
                attributes[item.target.id] = ann

    has_abstract = any(m.is_abstract for m in methods.values())

    # Parse total= keyword for TypedDict classes
    typeddict_total = True  # default per PEP 589
    if is_typeddict:
        for kw in node.keywords:
            if kw.arg == "total" and isinstance(kw.value, ast.Constant):
                typeddict_total = bool(kw.value.value)

    ci = ClassInfo(
        name=node.name,
        line=node.lineno,
        base_names=base_names,
        methods=methods,
        attributes=attributes,
        decorators=decorators,
        is_protocol=is_protocol,
        is_typeddict=is_typeddict,
        is_abc=is_abc or has_abstract,
        is_final="final" in decorators,
        is_runtime_checkable="runtime_checkable" in decorators,
        is_abstract=has_abstract,
        type_params=type_params,
        typeddict_total=typeddict_total,
    )

    if is_generic or type_params:
        ci.type_params = type_params or subscript_type_params or ["_T"]

    symbols.classes[node.name] = ci


def _process_assignment(node: ast.Assign, symbols: ModuleSymbols):
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return
    target_name = node.targets[0].id

    if not isinstance(node.value, ast.Call):
        return

    call = node.value
    func_name = ""
    if isinstance(call.func, ast.Name):
        func_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        func_name = call.func.attr

    if func_name in ("TypeVar", "ParamSpec", "TypeVarTuple"):
        symbols.typevars[target_name] = func_name
        symbols.typevar_lines[target_name] = node.lineno
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            symbols.typevar_string_args[target_name] = call.args[0].value
        variance = "invariant"
        for kw in call.keywords:
            if kw.arg == "covariant" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                variance = "covariant"
            elif kw.arg == "contravariant" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                variance = "contravariant"
        symbols.typevar_variance[target_name] = variance

    elif func_name == "NewType":
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            symbols.newtypes[target_name] = call.args[0].value
            symbols.newtype_lines[target_name] = node.lineno
            if len(call.args) >= 2:
                symbols.newtype_bases[target_name] = ast.unparse(call.args[1])


def _process_ann_assign(node: ast.AnnAssign, symbols: ModuleSymbols):
    if not isinstance(node.target, ast.Name):
        return
    ann_str = ast.unparse(node.annotation) if node.annotation else ""
    if "Final" in ann_str:
        symbols.final_vars[node.target.id] = node.lineno


# ANALYSES

def _analyze_lsp(symbols: ModuleSymbols) -> list[SourceFinding]:
    """LSP001: Method override signature compatibility (PEP 484)."""
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        for base_name in cls.base_names:
            base = symbols.classes.get(base_name)
            if base is None:
                continue

            for method_name, method in cls.methods.items():
                if method_name.startswith("_"):
                    continue
                base_method = base.methods.get(method_name)
                if base_method is None:
                    continue

                sub_required = [
                    p for p in method.params
                    if p.name != "self" and p.name != "cls"
                    and not p.has_default
                    and p.kind not in ("var_positional", "var_keyword")
                ]
                base_required = [
                    p for p in base_method.params
                    if p.name != "self" and p.name != "cls"
                    and not p.has_default
                    and p.kind not in ("var_positional", "var_keyword")
                ]

                if len(sub_required) > len(base_required):
                    findings.append(SourceFinding(
                        line=method.line,
                        rule_id="LSP001",
                        pep=484,
                        message=(
                            f"`{cls_name}.{method_name}` requires {len(sub_required)} "
                            f"params but base `{base_name}.{method_name}` requires "
                            f"{len(base_required)} — incompatible override (LSP)"
                        ),
                        expected_behavior="error",
                        confidence=0.90,
                    ))

                sub_non_self = [p for p in method.params if p.name not in ("self", "cls")]
                base_non_self = [p for p in base_method.params if p.name not in ("self", "cls")]

                base_has_var = any(p.kind == "var_positional" for p in base_non_self)
                sub_has_var = any(p.kind == "var_positional" for p in sub_non_self)

                if not base_has_var and not sub_has_var:
                    base_positional = [p for p in base_non_self if p.kind == "positional"]
                    sub_positional = [p for p in sub_non_self if p.kind == "positional"]
                    if len(sub_positional) != len(base_positional) and len(sub_required) != len(base_required):
                        pass

    return findings


def _analyze_self_context(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """SELF001: Self used outside class scope (PEP 673)."""
    findings: list[SourceFinding] = []

    class SelfVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_class = False
            self.in_method = False

        def visit_ClassDef(self, node):
            old = self.in_class
            self.in_class = True
            self.generic_visit(node)
            self.in_class = old

        def visit_FunctionDef(self, node):
            if not self.in_class:
                if node.returns:
                    ret = ast.unparse(node.returns)
                    if ret == "Self" and "Self" in symbols.typing_imports:
                        findings.append(SourceFinding(
                            line=node.lineno,
                            rule_id="SELF001",
                            pep=673,
                            message=f"`Self` used as return type of module-level function `{node.name}`",
                            expected_behavior="error",
                            confidence=0.95,
                        ))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    SelfVisitor().visit(tree)
    return findings


def _analyze_isinstance_protocol(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """PROTO001: isinstance/issubclass with non-@runtime_checkable Protocol (PEP 544)."""
    findings: list[SourceFinding] = []

    protocol_classes = {
        name for name, cls in symbols.classes.items()
        if cls.is_protocol and not cls.is_runtime_checkable
    }

    if not protocol_classes:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("isinstance", "issubclass"):
            continue
        if len(node.args) < 2:
            continue

        cls_arg = node.args[1]
        cls_names = _extract_type_names(cls_arg)

        for name in cls_names:
            if name in protocol_classes:
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id="PROTO001",
                    pep=544,
                    message=(
                        f"`{node.func.id}()` called with Protocol `{name}` "
                        f"which is not decorated with `@runtime_checkable`"
                    ),
                    expected_behavior="error",
                    confidence=0.95,
                ))

    return findings


def _extract_type_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple):
        names = []
        for elt in node.elts:
            names.extend(_extract_type_names(elt))
        return names
    return []


def _analyze_abstract_not_implemented(symbols: ModuleSymbols) -> list[SourceFinding]:
    """ABC001: Abstract method not implemented in concrete subclass (PEP 3119)."""
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        if cls.is_abstract or cls.is_protocol:
            continue

        all_abstract: dict[str, str] = {}
        for base_name in cls.base_names:
            base = symbols.classes.get(base_name)
            if base is None:
                continue
            _collect_abstract_methods(base, symbols, all_abstract)

        for method_name in list(all_abstract.keys()):
            if method_name in cls.methods:
                del all_abstract[method_name]

        if all_abstract:
            missing = ", ".join(f"`{m}`" for m in sorted(all_abstract.keys()))
            findings.append(SourceFinding(
                line=cls.line,
                rule_id="ABC001",
                pep=3119,
                message=(
                    f"Class `{cls_name}` doesn't implement abstract method(s) "
                    f"{missing} from {', '.join(all_abstract.values())}"
                ),
                expected_behavior="error",
                confidence=0.90,
            ))

    return findings


def _collect_abstract_methods(cls: ClassInfo, symbols: ModuleSymbols, result: dict[str, str]):
    for method_name, method in cls.methods.items():
        if method.is_abstract:
            result[method_name] = cls.name

    for base_name in cls.base_names:
        base = symbols.classes.get(base_name)
        if base is not None:
            _collect_abstract_methods(base, symbols, result)


def _analyze_override(symbols: ModuleSymbols) -> list[SourceFinding]:
    """OVERRIDE001: @override on method that doesn't exist in parent (PEP 698)."""
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        for method_name, method in cls.methods.items():
            if not method.is_override:
                continue

            found_in_parent = False
            for base_name in cls.base_names:
                if _method_in_hierarchy(base_name, method_name, symbols):
                    found_in_parent = True
                    break

            if not found_in_parent and all(
                b in symbols.classes for b in cls.base_names
            ):
                findings.append(SourceFinding(
                    line=method.line,
                    rule_id="OVERRIDE001",
                    pep=698,
                    message=(
                        f"`{cls_name}.{method_name}` decorated with @override "
                        f"but no base class defines `{method_name}`"
                    ),
                    expected_behavior="error",
                    confidence=0.90,
                ))

    return findings


def _method_in_hierarchy(cls_name: str, method_name: str, symbols: ModuleSymbols) -> bool:
    cls = symbols.classes.get(cls_name)
    if cls is None:
        return False
    if method_name in cls.methods:
        return True
    for base_name in cls.base_names:
        if _method_in_hierarchy(base_name, method_name, symbols):
            return True
    return False


def _analyze_final(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """FINAL001/002/003: Final violations (PEP 591)."""
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        for base_name in cls.base_names:
            base = symbols.classes.get(base_name)
            if base is not None and base.is_final:
                findings.append(SourceFinding(
                    line=cls.line,
                    rule_id="FINAL002",
                    pep=591,
                    message=f"Class `{cls_name}` subclasses `@final` class `{base_name}`",
                    expected_behavior="error",
                    confidence=0.95,
                ))

        for method_name, method in cls.methods.items():
            for base_name in cls.base_names:
                base = symbols.classes.get(base_name)
                if base is None:
                    continue
                base_method = base.methods.get(method_name)
                if base_method is not None and base_method.is_final:
                    findings.append(SourceFinding(
                        line=method.line,
                        rule_id="FINAL003",
                        pep=591,
                        message=(
                            f"`{cls_name}.{method_name}` overrides `@final` method "
                            f"in `{base_name}`"
                        ),
                        expected_behavior="error",
                        confidence=0.95,
                    ))

    assigned_names: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.setdefault(target.id, []).append(node.lineno)

    for name, decl_line in symbols.final_vars.items():
        assign_lines = assigned_names.get(name, [])
        reassignments = [ln for ln in assign_lines if ln != decl_line]
        for ln in reassignments:
            findings.append(SourceFinding(
                line=ln,
                rule_id="FINAL001",
                pep=591,
                message=f"Reassignment to `Final` variable `{name}` (declared at line {decl_line})",
                expected_behavior="error",
                confidence=0.95,
            ))

    return findings


def _analyze_generic_subscript(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """GENERIC001: Non-generic class subscripted (PEP 484)."""
    findings: list[SourceFinding] = []

    non_generic_classes = set()
    for name, cls in symbols.classes.items():
        if not cls.type_params and not cls.is_protocol and not cls.is_typeddict:
            has_generic_base = any(
                b in ("Generic", "Protocol", "TypedDict") for b in cls.base_names
            )
            if not has_generic_base:
                non_generic_classes.add(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in non_generic_classes:
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id="GENERIC001",
                    pep=484,
                    message=(
                        f"`{node.value.id}` is not a generic class but is subscripted "
                        f"with type arguments"
                    ),
                    expected_behavior="error",
                    confidence=0.85,
                ))

    return findings


def _analyze_typevar_names(symbols: ModuleSymbols) -> list[SourceFinding]:
    """TVAR001: TypeVar/ParamSpec/TypeVarTuple name mismatch (PEP 484/612/646)."""
    findings: list[SourceFinding] = []

    for name, kind in symbols.typevars.items():
        string_arg = symbols.typevar_string_args.get(name)
        if string_arg is not None and string_arg != name:
            findings.append(SourceFinding(
                line=0,
                rule_id="TVAR001",
                pep=484 if kind == "TypeVar" else 612 if kind == "ParamSpec" else 646,
                message=(
                    f"`{kind}` assigned to `{name}` but string argument is "
                    f'`"{string_arg}"` — names must match'
                ),
                expected_behavior="error",
                confidence=0.95,
            ))

    return findings


def _analyze_newtype_names(symbols: ModuleSymbols) -> list[SourceFinding]:
    """NEWTYPE001: NewType name mismatch (PEP 484)."""
    findings: list[SourceFinding] = []

    for name, string_arg in symbols.newtypes.items():
        if string_arg != name:
            line = symbols.newtype_lines.get(name, 0)
            findings.append(SourceFinding(
                line=line,
                rule_id="NEWTYPE001",
                pep=484,
                message=(
                    f"`NewType` assigned to `{name}` but string argument is "
                    f'`"{string_arg}"` — names must match'
                ),
                expected_behavior="error",
                confidence=0.95,
            ))

    return findings


def _analyze_typing_forms(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """FORM001: Typing special forms arity/context violations (multiple PEPs)."""
    findings: list[SourceFinding] = []

    class FormVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_class = False
            self.in_function = False
            self.in_typeddict = False
            self.context_stack: list[str] = []

        def visit_ClassDef(self, node):
            old = self.in_class
            self.in_class = True
            cls = symbols.classes.get(node.name)
            old_td = self.in_typeddict
            if cls and cls.is_typeddict:
                self.in_typeddict = True
            self.context_stack.append("class")
            self.generic_visit(node)
            self.context_stack.pop()
            self.in_class = old
            self.in_typeddict = old_td

        def visit_FunctionDef(self, node):
            self.in_function = True
            self.context_stack.append("function")

            if node.returns:
                self._check_annotation(node.returns, "return", node.lineno)

            for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                if arg.annotation:
                    self._check_annotation(arg.annotation, "param", node.lineno)

            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.arguments,)):
                    self.visit(child)

            self.context_stack.pop()
            self.in_function = False

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node):
            if node.annotation:
                ctx = "class_body" if "class" in self.context_stack and "function" not in self.context_stack else "other"
                self._check_annotation(node.annotation, ctx, node.lineno)
            self.generic_visit(node)

        def _check_annotation(self, node: ast.expr, context: str, line: int):
            if not isinstance(node, ast.Subscript):
                return

            if isinstance(node.value, ast.Name):
                name = node.value.id
            elif isinstance(node.value, ast.Attribute):
                name = node.value.attr
            else:
                return

            if name not in symbols.typing_imports and name not in (
                "ClassVar", "Final", "Required", "NotRequired",
                "TypeGuard", "TypeIs", "Literal", "Annotated",
                "Concatenate", "Unpack",
            ):
                return

            args = _get_subscript_args(node)
            n_args = len(args)

            if name == "ClassVar":
                if context not in ("class_body",):
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="FORM001",
                        pep=526,
                        message="`ClassVar` can only be used in class body annotations",
                        expected_behavior="error",
                        confidence=0.90,
                    ))
                if n_args > 1:
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="FORM001",
                        pep=526,
                        message=f"`ClassVar` takes exactly 1 type argument, got {n_args}",
                        expected_behavior="error",
                        confidence=0.90,
                    ))

            elif name == "Final":
                if context == "param":
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="FORM001",
                        pep=591,
                        message="`Final` cannot be used as a function parameter annotation",
                        expected_behavior="error",
                        confidence=0.90,
                    ))

            elif name in ("Required", "NotRequired"):
                if not self.in_typeddict:
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="FORM001",
                        pep=655,
                        message=f"`{name}` can only be used inside TypedDict definitions",
                        expected_behavior="error",
                        confidence=0.90,
                    ))

            elif name in ("TypeGuard", "TypeIs"):
                if context != "return":
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="FORM001",
                        pep=647 if name == "TypeGuard" else 742,
                        message=f"`{name}` can only be used as a function return annotation",
                        expected_behavior="error",
                        confidence=0.90,
                    ))
                if n_args != 1:
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="FORM001",
                        pep=647 if name == "TypeGuard" else 742,
                        message=f"`{name}` takes exactly 1 type argument, got {n_args}",
                        expected_behavior="error",
                        confidence=0.90,
                    ))

            elif name == "Literal":
                for arg in args:
                    if not _is_valid_literal_arg(arg):
                        findings.append(SourceFinding(
                            line=node.lineno,
                            rule_id="FORM001",
                            pep=586,
                            message=f"`Literal` argument `{ast.unparse(arg)}` is not a valid literal value",
                            expected_behavior="error",
                            confidence=0.90,
                        ))

            elif name == "Annotated":
                if n_args < 2:
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="FORM001",
                        pep=593,
                        message=f"`Annotated` requires at least 2 arguments, got {n_args}",
                        expected_behavior="error",
                        confidence=0.90,
                    ))

            if isinstance(node.slice, ast.Subscript):
                self._check_annotation(node.slice, context, line)
            elif isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Subscript):
                        self._check_annotation(elt, context, line)

    FormVisitor().visit(tree)
    return findings


def _get_subscript_args(node: ast.Subscript) -> list[ast.expr]:
    if isinstance(node.slice, ast.Tuple):
        return list(node.slice.elts)
    return [node.slice]


def _is_valid_literal_arg(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    if isinstance(node, ast.Attribute):
        return True
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "Literal":
            return True
    return False


def _analyze_overload(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """OVERLOAD001: @overload structural correctness (PEP 484)."""
    findings: list[SourceFinding] = []

    _check_overload_scope(list(ast.iter_child_nodes(tree)), findings)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _check_overload_scope(node.body, findings)

    return findings


def _check_overload_scope(stmts: list[ast.stmt], findings: list[SourceFinding]):
    overload_groups: dict[str, list[ast.FunctionDef]] = {}
    impl_names: set[str] = set()

    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = _get_decorator_names(stmt)
            if "overload" in decorators:
                overload_groups.setdefault(stmt.name, []).append(stmt)
            else:
                if stmt.name in overload_groups:
                    impl_names.add(stmt.name)

    for name, overloads in overload_groups.items():
        if name not in impl_names:
            findings.append(SourceFinding(
                line=overloads[-1].lineno,
                rule_id="OVERLOAD001",
                pep=484,
                message=(
                    f"@overload group `{name}` has {len(overloads)} overload(s) "
                    f"but no implementation"
                ),
                expected_behavior="error",
                confidence=0.90,
            ))


def _analyze_runtime_checkable_non_protocol(symbols: ModuleSymbols) -> list[SourceFinding]:
    """PROTO002: @runtime_checkable on non-Protocol class (PEP 544)."""
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        if cls.is_runtime_checkable and not cls.is_protocol:
            findings.append(SourceFinding(
                line=cls.line,
                rule_id="PROTO002",
                pep=544,
                message=f"`@runtime_checkable` applied to `{cls_name}` which is not a Protocol",
                expected_behavior="error",
                confidence=0.95,
            ))

    return findings


def _analyze_decorator_targets(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """DECO001: @override/@final decorator target validation (PEP 591/698)."""
    findings: list[SourceFinding] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = _get_decorator_names(node)
            if "override" in decorators:
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id="DECO001",
                    pep=698,
                    message=f"`@override` applied to module-level function `{node.name}` (not a method)",
                    expected_behavior="error",
                    confidence=0.95,
                ))

    return findings


def _analyze_classvar_in_function(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """CLASSVAR001: ClassVar used in function scope (PEP 526)."""
    findings: list[SourceFinding] = []

    class ClassVarVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_function = False
            self.in_class = False

        def visit_FunctionDef(self, node):
            old = self.in_function
            self.in_function = True
            for child in ast.iter_child_nodes(node):
                self.visit(child)
            self.in_function = old

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            old_class = self.in_class
            self.in_class = True
            self.generic_visit(node)
            self.in_class = old_class

        def visit_AnnAssign(self, node):
            if self.in_function and node.annotation:
                ann_str = ast.unparse(node.annotation)
                if "ClassVar" in ann_str:
                    if isinstance(node.target, ast.Attribute):
                        findings.append(SourceFinding(
                            line=node.lineno,
                            rule_id="CLASSVAR001",
                            pep=526,
                            message=f"`ClassVar` used on instance attribute `{ast.unparse(node.target)}`",
                            expected_behavior="error",
                            confidence=0.90,
                        ))

    ClassVarVisitor().visit(tree)
    return findings


def _analyze_method_override_types(symbols: ModuleSymbols) -> list[SourceFinding]:
    """LSP002: Method override with incompatible parameter/return types (PEP 484)."""
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        for base_name in cls.base_names:
            base = symbols.classes.get(base_name)
            if base is None:
                continue

            for method_name, method in cls.methods.items():
                base_method = base.methods.get(method_name)
                if base_method is None:
                    continue
                if method_name in ("__init__", "__new__"):
                    continue

                sub_params = [p for p in method.params if p.name not in ("self", "cls") and p.kind == "positional"]
                base_params = [p for p in base_method.params if p.name not in ("self", "cls") and p.kind == "positional"]

                if sub_params and base_params and len(sub_params) == len(base_params):
                    for sp, bp in zip(sub_params, base_params):
                        if sp.annotation and bp.annotation:
                            if sp.annotation != bp.annotation:
                                sp_ann = sp.annotation
                                bp_ann = bp.annotation
                                if not _is_supertype_of(sp_ann, bp_ann):
                                    findings.append(SourceFinding(
                                        line=method.line,
                                        rule_id="LSP002",
                                        pep=484,
                                        message=(
                                            f"`{cls_name}.{method_name}` parameter `{sp.name}` "
                                            f"has type `{sp_ann}` but base `{base_name}.{method_name}` "
                                            f"has `{bp_ann}` — incompatible override"
                                        ),
                                        expected_behavior="error",
                                        confidence=0.85,
                                    ))

    return findings


def _is_supertype_of(sub_ann: str, base_ann: str) -> bool:
    if sub_ann == base_ann:
        return True
    if "Union[" in sub_ann and base_ann in sub_ann:
        return True
    if " | " in sub_ann and base_ann in sub_ann:
        return True
    return False


def _analyze_variance(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """VARIANCE001: Covariant TypeVar in parameter / contravariant in return (PEP 484).

    PEP 484 requires:
    - Covariant TypeVars may only appear in return types (output positions)
    - Contravariant TypeVars may only appear in parameter types (input positions)
    Using them in the wrong position is unsound.
    """
    findings: list[SourceFinding] = []

    covariant_tvars = {
        name for name, var in symbols.typevar_variance.items()
        if var == "covariant"
    }
    contravariant_tvars = {
        name for name, var in symbols.typevar_variance.items()
        if var == "contravariant"
    }

    if not covariant_tvars and not contravariant_tvars:
        return findings

    def _annotation_names(ann_str: str) -> set[str]:
        return {token for token in re.findall(r'\b([A-Za-z_]\w*)\b', ann_str)}

    all_methods: list[tuple[str, MethodInfo]] = []
    for cls_name, cls in symbols.classes.items():
        for method_name, method in cls.methods.items():
            all_methods.append((f"{cls_name}.{method_name}", method))
    for func_name, func in symbols.functions.items():
        all_methods.append((func_name, func))

    for qual_name, method in all_methods:
        for param in method.params:
            if param.name in ("self", "cls") or not param.annotation:
                continue
            used_names = _annotation_names(param.annotation)
            for tvar in covariant_tvars & used_names:
                findings.append(SourceFinding(
                    line=method.line,
                    rule_id="VARIANCE001",
                    pep=484,
                    message=(
                        f"Covariant TypeVar `{tvar}` used in parameter `{param.name}` "
                        f"of `{qual_name}` — covariant TypeVars are only valid in "
                        f"return/output positions"
                    ),
                    expected_behavior="error",
                    confidence=0.90,
                ))

        if method.return_annotation:
            used_names = _annotation_names(method.return_annotation)
            for tvar in contravariant_tvars & used_names:
                findings.append(SourceFinding(
                    line=method.line,
                    rule_id="VARIANCE001",
                    pep=484,
                    message=(
                        f"Contravariant TypeVar `{tvar}` used in return type "
                        f"of `{qual_name}` — contravariant TypeVars are only valid "
                        f"in parameter/input positions"
                    ),
                    expected_behavior="error",
                    confidence=0.90,
                ))

    return findings


def _analyze_noreturn(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """NORETURN001: Function annotated -> NoReturn/Never that contains a return statement (PEP 484).

    A function annotated with NoReturn or Never must never return normally.
    If it contains a reachable `return` statement, that is a violation.
    """
    findings: list[SourceFinding] = []
    noreturn_names = {"NoReturn", "Never"}

    class ReturnFinder(ast.NodeVisitor):
        def __init__(self):
            self.has_return = False
            self.return_line = 0

        def visit_Return(self, node):
            self.has_return = True
            self.return_line = node.lineno

        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.returns:
            continue
        ret_str = ast.unparse(node.returns)
        if ret_str not in noreturn_names:
            continue

        finder = ReturnFinder()
        for child in node.body:
            finder.visit(child)

        if finder.has_return:
            findings.append(SourceFinding(
                line=finder.return_line,
                rule_id="NORETURN001",
                pep=484,
                message=(
                    f"Function `{node.name}` is annotated `-> {ret_str}` but "
                    f"contains a return statement at line {finder.return_line}"
                ),
                expected_behavior="error",
                confidence=0.95,
            ))

    return findings


def _analyze_protocol_instantiation(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """PROTO003: Direct instantiation of a Protocol class (PEP 544).
    Protocol classes define structural interfaces and cannot be instantiated
    directly. Only concrete classes that satisfy the protocol may be used.
    """
    findings: list[SourceFinding] = []

    protocol_classes = {
        name for name, cls in symbols.classes.items()
        if cls.is_protocol
    }

    if not protocol_classes:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in protocol_classes:
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="PROTO003",
                pep=544,
                message=(
                    f"Direct instantiation of Protocol class `{node.func.id}` — "
                    f"Protocols cannot be instantiated (PEP 544)"
                ),
                expected_behavior="error",
                confidence=0.90,
            ))

    return findings


def _analyze_typeddict_inheritance(symbols: ModuleSymbols) -> list[SourceFinding]:
    """TDICT001: TypedDict inheriting from non-TypedDict class (PEP 589).

    TypedDict classes can only inherit from other TypedDict classes.
    Inheriting from a regular class is not allowed.
    """
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        if not cls.is_typeddict:
            continue
        for base_name in cls.base_names:
            if base_name in ("TypedDict", "typing.TypedDict", "typing_extensions.TypedDict"):
                continue
            base = symbols.classes.get(base_name)
            if base is not None and not base.is_typeddict:
                findings.append(SourceFinding(
                    line=cls.line,
                    rule_id="TDICT001",
                    pep=589,
                    message=(
                        f"TypedDict `{cls_name}` inherits from non-TypedDict "
                        f"class `{base_name}` — TypedDict can only extend other "
                        f"TypedDict classes"
                    ),
                    expected_behavior="error",
                    confidence=0.90,
                ))

    return findings


def _analyze_typeddict_field_conflict(symbols: ModuleSymbols) -> list[SourceFinding]:
    """TDICT002: TypedDict field type conflict in multiple inheritance (PEP 589).

    When a TypedDict inherits from multiple TypedDict parents, all parents
    must agree on the types of shared field names. A field defined with
    different types in two parents is a violation.
    """
    findings: list[SourceFinding] = []

    def _collect_fields(cls: ClassInfo) -> dict[str, tuple[str, str]]:
        """Collect all fields: name -> (type, declaring_class)."""
        fields: dict[str, tuple[str, str]] = {}
        for base_name in cls.base_names:
            base = symbols.classes.get(base_name)
            if base is not None and base.is_typeddict:
                for fname, ftype in base.attributes.items():
                    fields.setdefault(fname, (ftype, base.name))
        for fname, ftype in cls.attributes.items():
            fields[fname] = (ftype, cls.name)
        return fields

    for cls_name, cls in symbols.classes.items():
        if not cls.is_typeddict:
            continue
        typeddict_bases = [
            b for b in cls.base_names
            if b not in ("TypedDict", "typing.TypedDict", "typing_extensions.TypedDict")
            and symbols.classes.get(b) is not None
            and symbols.classes[b].is_typeddict
        ]
        if len(typeddict_bases) < 2:
            continue

        base_fields: dict[str, list[tuple[str, str]]] = {}
        for base_name in typeddict_bases:
            base = symbols.classes[base_name]
            for fname, ftype in base.attributes.items():
                base_fields.setdefault(fname, []).append((ftype, base_name))

        for fname, declarations in base_fields.items():
            types_seen = {ftype for ftype, _ in declarations}
            if len(types_seen) > 1:
                sources = ", ".join(f"`{src}` ({t})" for t, src in declarations)
                findings.append(SourceFinding(
                    line=cls.line,
                    rule_id="TDICT002",
                    pep=589,
                    message=(
                        f"TypedDict `{cls_name}` inherits conflicting types for "
                        f"field `{fname}`: {sources}"
                    ),
                    expected_behavior="error",
                    confidence=0.85,
                ))

    return findings


def _analyze_typeddict_notrequired_access(
    tree: ast.Module, symbols: ModuleSymbols,
) -> list[SourceFinding]:
    """TDICT003: Unsafe [] access on NotRequired TypedDict field (PEPs 589, 655).

    When a TypedDict field is NotRequired — either explicitly via
    ``NotRequired[T]`` or implicitly because it was declared in a
    ``total=False`` TypedDict — accessing it with ``td["key"]`` may raise
    ``KeyError`` at runtime if the key is absent.  Safe alternatives are
    ``td.get("key")`` or guarding with ``"key" in td``.

    We flag ``td["key"]`` on a NotRequired field only when the access is
    NOT protected by an ``if "key" in td`` guard on the same variable.
    """
    findings: list[SourceFinding] = []

    # Build a map: TypedDict name → set of NotRequired field names
    notrequired_fields: dict[str, set[str]] = {}

    for cls_name, cls in symbols.classes.items():
        if not cls.is_typeddict:
            continue
        nr: set[str] = set()

        # Collect NotRequired fields from this class and its bases
        for td in _typeddict_mro(cls, symbols):
            for fname, ann in td.attributes.items():
                if _field_is_notrequired(fname, ann, td):
                    nr.add(fname)

        if nr:
            notrequired_fields[cls_name] = nr

    if not notrequired_fields:
        return findings

    # Build a map: parameter name → TypedDict class name
    param_types: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation:
                ann_str = ast.unparse(arg.annotation)
                if ann_str in notrequired_fields:
                    param_types[arg.arg] = ann_str

    # Also track variable annotations at module/function level
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            ann_str = ast.unparse(node.annotation) if node.annotation else ""
            if ann_str in notrequired_fields:
                param_types[node.target.id] = ann_str

    if not param_types:
        return findings

    # Collect "key in var" guards per scope
    guarded: set[tuple[str, str]] = set()  # (var_name, key_str)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # if "key" in var or if "key" in var:
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            if isinstance(test.ops[0], ast.In):
                left = test.comparators[0] if isinstance(test.ops[0], ast.In) else None
                key_node = test.left
                var_node = test.comparators[0]
                if (isinstance(key_node, ast.Constant)
                        and isinstance(key_node.value, str)
                        and isinstance(var_node, ast.Name)):
                    guarded.add((var_node.id, key_node.value))

    # Find unguarded td["key"] accesses on NotRequired fields
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # Match: var["literal_key"]
        if not isinstance(node.value, ast.Name):
            continue
        var_name = node.value.id
        td_cls_name = param_types.get(var_name)
        if td_cls_name is None:
            continue
        nr_keys = notrequired_fields.get(td_cls_name, set())
        if not nr_keys:
            continue

        # Extract the key string
        key_str: str | None = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key_str = node.slice.value

        if key_str is None or key_str not in nr_keys:
            continue

        # Skip if inside a TYPE_CHECKING block (reveal_type calls are diagnostic)
        if _is_inside_type_checking(node, tree):
            continue

        # Skip if guarded by "key" in var
        if (var_name, key_str) in guarded:
            continue

        findings.append(SourceFinding(
            line=node.lineno,
            rule_id="TDICT003",
            pep=589,
            message=(
                f"Unsafe access `{var_name}[\"{key_str}\"]` on NotRequired "
                f"field of TypedDict `{td_cls_name}` — field may be absent "
                f"at runtime (use `.get(\"{key_str}\")` or guard with "
                f"`\"{key_str}\" in {var_name}`)"
            ),
            expected_behavior="error",
            confidence=0.90,
        ))

    return findings


def _analyze_typeddict_missing_required_keys(
    tree: ast.Module, symbols: ModuleSymbols,
) -> list[SourceFinding]:
    """TDICT004: Missing required key in TypedDict literal construction (PEPs 589, 655).

    When a dict literal is annotated as a specific TypedDict, all Required
    fields must be present.  A field is Required when it is NOT NotRequired
    (i.e., it is in a ``total=True`` class without ``NotRequired`` wrapper,
    or it is explicitly ``Required[T]``).

    We flag dict literals annotated with a TypedDict type that are missing
    one or more required keys.
    """
    findings: list[SourceFinding] = []

    # Build field maps per TypedDict
    td_fields: dict[str, dict[str, bool]] = {}  # cls_name -> {field: is_required}

    for cls_name, cls in symbols.classes.items():
        if not cls.is_typeddict:
            continue
        fields: dict[str, bool] = {}
        for td in _typeddict_mro(cls, symbols):
            for fname, ann in td.attributes.items():
                if _field_is_notrequired(fname, ann, td):
                    fields.setdefault(fname, False)
                else:
                    fields.setdefault(fname, True)
        td_fields[cls_name] = fields

    if not td_fields:
        return findings

    # Find annotated dict literals: var: TDName = {...}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if node.value is None or not isinstance(node.value, ast.Dict):
            continue
        ann_str = ast.unparse(node.annotation) if node.annotation else ""
        field_map = td_fields.get(ann_str)
        if field_map is None:
            continue

        # Collect keys present in the dict literal
        present_keys: set[str] = set()
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                present_keys.add(key.value)

        # Check for missing required keys
        required_keys = {k for k, is_req in field_map.items() if is_req}
        missing = required_keys - present_keys
        if missing:
            missing_sorted = sorted(missing)
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="TDICT004",
                pep=589,
                message=(
                    f"TypedDict `{ann_str}` literal is missing required "
                    f"key(s): {', '.join(repr(k) for k in missing_sorted)}"
                ),
                expected_behavior="error",
                confidence=0.90,
            ))

    return findings


def _typeddict_mro(cls: ClassInfo, symbols: ModuleSymbols) -> list[ClassInfo]:
    """Return the TypedDict MRO: the class itself plus all TypedDict bases (recursive)."""
    result = [cls]
    for base_name in cls.base_names:
        base = symbols.classes.get(base_name)
        if base is not None and base.is_typeddict:
            result.extend(_typeddict_mro(base, symbols))
    return result


def _field_is_notrequired(fname: str, ann: str, cls: ClassInfo) -> bool:
    """Determine if a TypedDict field is NotRequired.
    A field is NotRequired when:
      (a) It is explicitly wrapped: ``NotRequired[T]``
      (b) It is declared in a TypedDict with ``total=False`` and NOT wrapped
          in ``Required[T]``
    """
    if "NotRequired" in ann:
        return True
    if not cls.typeddict_total and "Required" not in ann:
        return True
    return False


def _is_inside_type_checking(node: ast.AST, tree: ast.Module) -> bool:
    """Return True if *node* is inside an ``if TYPE_CHECKING:`` block."""
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.If):
            continue
        test = parent.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            # Check if node's line is within this if-block body
            body_start = parent.body[0].lineno if parent.body else 0
            body_end = parent.body[-1].end_lineno or parent.body[-1].lineno if parent.body else 0
            if hasattr(node, "lineno") and body_start <= node.lineno <= body_end:
                return True
    return False


_LITERAL_TYPE_MAP: dict[type, set[str]] = {
    int: {"int", "float", "complex"},
    float: {"float", "complex"},
    str: {"str"},
    bool: {"bool", "int", "float", "complex"},
    bytes: {"bytes"},
    type(None): {"None", "type[None]"},
}

_ANNOTATION_TO_PYTHON_TYPES: dict[str, set[type]] = {
    "int": {int, bool},
    "float": {int, float, bool},
    "complex": {int, float, complex, bool},
    "str": {str},
    "bytes": {bytes},
    "bool": {bool},
}


def _analyze_incompatible_assignment(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """ASSIGN001: Obvious literal type mismatch in annotated assignment (PEP 484).
    Detects cases like `x: int = "hello"` where a literal value is clearly
    incompatible with the declared type annotation. Only fires for simple
    literal-to-builtin-type mismatches to avoid false positives.
    """
    findings: list[SourceFinding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if node.value is None or node.annotation is None:
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.annotation, ast.Name):
            continue

        ann_name = node.annotation.id
        value = node.value.value
        value_type = type(value)

        accepted_types = _ANNOTATION_TO_PYTHON_TYPES.get(ann_name)
        if accepted_types is None:
            continue

        if value_type not in accepted_types:
            if value is None and ann_name != "None":
                msg = f"Assigning `None` to variable annotated as `{ann_name}`"
            else:
                msg = (
                    f"Assigning `{value_type.__name__}` literal `{repr(value)[:50]}` "
                    f"to variable annotated as `{ann_name}`"
                )
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="ASSIGN001",
                pep=484,
                message=msg,
                expected_behavior="error",
                confidence=0.95,
            ))

    return findings


def _analyze_overload_count(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """OVERLOAD002: Single @overload without a peer (PEP 484).

    PEP 484 requires at least two @overload signatures plus an implementation.
    A single @overload is meaningless — it provides no discrimination.
    """
    findings: list[SourceFinding] = []

    _check_overload_count_scope(list(ast.iter_child_nodes(tree)), findings)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _check_overload_count_scope(node.body, findings)

    return findings


def _check_overload_count_scope(stmts: list[ast.stmt], findings: list[SourceFinding]):
    overload_groups: dict[str, list[ast.FunctionDef]] = {}

    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = _get_decorator_names(stmt)
            if "overload" in decorators:
                overload_groups.setdefault(stmt.name, []).append(stmt)

    for name, overloads in overload_groups.items():
        if len(overloads) == 1:
            findings.append(SourceFinding(
                line=overloads[0].lineno,
                rule_id="OVERLOAD002",
                pep=484,
                message=(
                    f"Only one @overload for `{name}` — at least two overload "
                    f"signatures are required"
                ),
                expected_behavior="error",
                confidence=0.85,
            ))


def _analyze_return_type_none(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """RETURN001: Non-None return annotation but function body only returns None (PEP 484).
    A function annotated with a concrete return type (not None, NoReturn, Never,
    or Any) whose body only contains `pass`, `...`, or bare `return`/`return None`
    — and is NOT an @abstractmethod, @overload, or Protocol method — violates
    PEP 484 since it will return None at runtime.
    """
    findings: list[SourceFinding] = []

    exempt_returns = {"None", "NoReturn", "Never", "Any", "type[None]"}
    protocol_classes = {
        name for name, cls in symbols.classes.items() if cls.is_protocol
    }

    def _body_is_stub(body: list[ast.stmt]) -> bool:
        if not body:
            return True
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value is ...:
                    return True
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return True
                if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                    return True
        return False

    def _only_returns_none(body: list[ast.stmt]) -> bool:
        has_return = False
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Return):
                has_return = True
                if node.value is not None:
                    if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                        return False
        return has_return or _body_is_stub(body)

    class ReturnChecker(ast.NodeVisitor):
        def __init__(self):
            self.in_class: Optional[str] = None

        def visit_ClassDef(self, node):
            old = self.in_class
            self.in_class = node.name
            self.generic_visit(node)
            self.in_class = old

        def visit_FunctionDef(self, node):
            self._check(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def _check(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.returns:
                return
            ret_str = ast.unparse(node.returns)
            if ret_str in exempt_returns:
                return
            if "Optional" in ret_str or "None" in ret_str:
                return

            decorators = _get_decorator_names(node)
            if "abstractmethod" in decorators:
                return
            if "overload" in decorators:
                return

            if self.in_class and self.in_class in protocol_classes:
                return

            if _body_is_stub(node.body) or _only_returns_none(node.body):
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id="RETURN001",
                    pep=484,
                    message=(
                        f"Function `{node.name}` is annotated `-> {ret_str}` "
                        f"but body only returns None"
                    ),
                    expected_behavior="error",
                    confidence=0.85,
                ))

    ReturnChecker().visit(tree)
    return findings


def _analyze_final_property_override(symbols: ModuleSymbols) -> list[SourceFinding]:
    """FINAL004: Property overriding a Final attribute in a base class (PEP 591).
    PEP 591 §Class-level: "Final names cannot be overridden in subclasses."
    This applies regardless of whether the override is an attribute or a
    property. The existing FINAL003 covers method→method overrides; this
    rule covers the attribute→property cross-kind case.
    """
    findings: list[SourceFinding] = []

    for cls_name, cls in symbols.classes.items():
        for base_name in cls.base_names:
            base = symbols.classes.get(base_name)
            if base is None:
                continue

            base_final_attrs: set[str] = set()
            for attr_name, ann in base.attributes.items():
                if "Final" in ann:
                    base_final_attrs.add(attr_name)

            for method_name, method in cls.methods.items():
                if method.is_property and method_name in base_final_attrs:
                    findings.append(SourceFinding(
                        line=method.line,
                        rule_id="FINAL004",
                        pep=591,
                        message=(
                            f"`{cls_name}.{method_name}` is a property that overrides "
                            f"`Final` attribute `{method_name}` in `{base_name}`"
                        ),
                        expected_behavior="error",
                        confidence=0.90,
                    ))

            for attr_name, ann in cls.attributes.items():
                if attr_name in base_final_attrs and "Final" not in ann:
                    findings.append(SourceFinding(
                        line=cls.line,
                        rule_id="FINAL004",
                        pep=591,
                        message=(
                            f"`{cls_name}.{attr_name}` overrides `Final` attribute "
                            f"`{attr_name}` in `{base_name}`"
                        ),
                        expected_behavior="error",
                        confidence=0.90,
                    ))

    return findings


def _analyze_invariant_typevar_in_protocol(symbols: ModuleSymbols) -> list[SourceFinding]:
    """VARIANCE002: Invariant TypeVar used in Protocol where variance is required (PEP 544).
    When a Protocol uses a TypeVar in ONLY contravariant positions (method
    parameters) or ONLY covariant positions (return types), the TypeVar should
    be declared with the matching variance. An invariant TypeVar in such a
    position is technically accepted but mypy (and others) flag it because
    the Protocol's structural subtyping is unsound without correct variance.

    We only flag the case where ALL usages of an invariant TypeVar within
    a Protocol are in parameter (contravariant) positions — the most common
    real-world case (e.g. callback protocols).
    """
    findings: list[SourceFinding] = []

    invariant_tvars = {
        name for name, var in symbols.typevar_variance.items()
        if var == "invariant" and symbols.typevars.get(name) == "TypeVar"
    }
    if not invariant_tvars:
        return findings

    for cls_name, cls in symbols.classes.items():
        if not cls.is_protocol:
            continue

        for tvar_name in invariant_tvars:
            if tvar_name not in cls.type_params:
                continue

            in_params = False
            in_returns = False

            for method_name, method in cls.methods.items():
                for param in method.params:
                    if param.name in ("self", "cls") or not param.annotation:
                        continue
                    if tvar_name in re.findall(r'\b([A-Za-z_]\w*)\b', param.annotation):
                        in_params = True
                if method.return_annotation:
                    if tvar_name in re.findall(r'\b([A-Za-z_]\w*)\b', method.return_annotation):
                        in_returns = True

            if in_params and not in_returns:
                line = symbols.typevar_lines.get(tvar_name, cls.line)
                findings.append(SourceFinding(
                    line=line,
                    rule_id="VARIANCE002",
                    pep=544,
                    message=(
                        f"Invariant TypeVar `{tvar_name}` used in Protocol "
                        f"`{cls_name}` only in parameter positions — should be "
                        f"declared contravariant"
                    ),
                    expected_behavior="error",
                    confidence=0.85,
                ))

    return findings


_VALID_NEWTYPE_BASES = {
    "int", "str", "bytes", "bool", "object",
    "list", "dict", "set", "frozenset", "tuple",
}


def _analyze_newtype_base(symbols: ModuleSymbols) -> list[SourceFinding]:
    """NEWTYPE002: NewType with non-class base type (PEP 484).
    The second argument to NewType must be a class or another NewType.
    Builtin types like float are technically classes at runtime, but some
    checkers (ty) flag float because it is a union type (int | float).
    We only flag clearly invalid bases: non-Name expressions like
    subscripts, binary ops, or literal values.
    """
    findings: list[SourceFinding] = []

    for name, base_str in symbols.newtype_bases.items():
        line = symbols.newtype_lines.get(name, 0)

        if base_str in symbols.classes or base_str in symbols.newtypes:
            continue
        if base_str in _VALID_NEWTYPE_BASES or base_str == "float":
            continue
        if base_str in symbols.typing_imports or base_str in symbols.all_imports:
            continue
        if re.match(r'^[A-Za-z_]\w*$', base_str):
            continue

        findings.append(SourceFinding(
            line=line,
            rule_id="NEWTYPE002",
            pep=484,
            message=(
                f"NewType `{name}` has base `{base_str}` which is not a "
                f"simple class name — NewType base must be a class or "
                f"another NewType"
            ),
            expected_behavior="error",
            confidence=0.85,
        ))

    return findings


def _analyze_generic_param_count(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """GENERIC002: Wrong number of type arguments for a generic class (PEP 484).
    When a locally-defined generic class specifies N type parameters,
    subscripting it with a different count is an error.
    """
    findings: list[SourceFinding] = []

    generic_param_counts: dict[str, int] = {}
    for name, cls in symbols.classes.items():
        if cls.type_params:
            generic_param_counts[name] = len(cls.type_params)

    if not generic_param_counts:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        cls_name = node.value.id
        expected = generic_param_counts.get(cls_name)
        if expected is None:
            continue

        args = _get_subscript_args(node)
        actual = len(args)

        if actual != expected:
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="GENERIC002",
                pep=484,
                message=(
                    f"`{cls_name}` expects {expected} type argument(s) "
                    f"but got {actual}"
                ),
                expected_behavior="error",
                confidence=0.85,
            ))

    return findings


def _analyze_overload_return_consistency(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """OVERLOAD003: Overload implementation return type incompatible with overloads (PEP 484).
    When all @overload signatures return the same type but the implementation
    returns a clearly different type, this is a violation.
    """
    findings: list[SourceFinding] = []

    def _check_scope(stmts: list[ast.stmt]):
        overload_groups: dict[str, list[ast.FunctionDef]] = {}
        impl_map: dict[str, ast.FunctionDef] = {}

        for stmt in stmts:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = _get_decorator_names(stmt)
            if "overload" in decorators:
                overload_groups.setdefault(stmt.name, []).append(stmt)
            elif stmt.name in overload_groups:
                impl_map[stmt.name] = stmt

        for name, overloads in overload_groups.items():
            impl = impl_map.get(name)
            if impl is None or not impl.returns:
                continue

            overload_returns = set()
            for ol in overloads:
                if ol.returns:
                    overload_returns.add(ast.unparse(ol.returns))

            if len(overload_returns) != 1:
                continue

            expected_ret = overload_returns.pop()
            impl_ret = ast.unparse(impl.returns)

            if impl_ret == "Any" or expected_ret == "Any":
                continue
            if impl_ret == expected_ret:
                continue
            if expected_ret in impl_ret:
                continue

            builtin_types = {"int", "str", "float", "bool", "bytes", "None"}
            if expected_ret in builtin_types and impl_ret in builtin_types and expected_ret != impl_ret:
                findings.append(SourceFinding(
                    line=impl.lineno,
                    rule_id="OVERLOAD003",
                    pep=484,
                    message=(
                        f"All overloads of `{name}` return `{expected_ret}` but "
                        f"implementation returns `{impl_ret}`"
                    ),
                    expected_behavior="error",
                    confidence=0.85,
                ))

    _check_scope(list(ast.iter_child_nodes(tree)))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _check_scope(node.body)

    return findings


def _analyze_paramspec_misuse(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """PARAMSPEC001: ParamSpec.args/kwargs used outside a function signature (PEP 612).
    P.args and P.kwargs may only appear in *args and **kwargs of a function
    that uses Concatenate[..., P] or Callable[P, R]. Using P.args or P.kwargs
    as a standalone annotation (not on *args/**kwargs) is invalid.
    """
    findings: list[SourceFinding] = []

    paramspec_names = {
        name for name, kind in symbols.typevars.items()
        if kind == "ParamSpec"
    }
    if not paramspec_names:
        return findings

    class ParamSpecVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                if arg.annotation:
                    ann = ast.unparse(arg.annotation)
                    for ps_name in paramspec_names:
                        if ann == f"{ps_name}.args" or ann == f"{ps_name}.kwargs":
                            findings.append(SourceFinding(
                                line=arg.annotation.lineno if hasattr(arg.annotation, 'lineno') else node.lineno,
                                rule_id="PARAMSPEC001",
                                pep=612,
                                message=(
                                    f"`{ann}` used as annotation for regular parameter "
                                    f"`{arg.arg}` — `{ps_name}.args` must annotate "
                                    f"`*args` and `{ps_name}.kwargs` must annotate "
                                    f"`**kwargs`"
                                ),
                                expected_behavior="error",
                                confidence=0.90,
                            ))

            if node.args.vararg and node.args.vararg.annotation:
                ann = ast.unparse(node.args.vararg.annotation)
                for ps_name in paramspec_names:
                    if ann == f"{ps_name}.kwargs":
                        findings.append(SourceFinding(
                            line=node.args.vararg.annotation.lineno if hasattr(node.args.vararg.annotation, 'lineno') else node.lineno,
                            rule_id="PARAMSPEC001",
                            pep=612,
                            message=(
                                f"`{ps_name}.kwargs` used to annotate `*args` — "
                                f"`*args` must use `{ps_name}.args`"
                            ),
                            expected_behavior="error",
                            confidence=0.90,
                        ))

            if node.args.kwarg and node.args.kwarg.annotation:
                ann = ast.unparse(node.args.kwarg.annotation)
                for ps_name in paramspec_names:
                    if ann == f"{ps_name}.args":
                        findings.append(SourceFinding(
                            line=node.args.kwarg.annotation.lineno if hasattr(node.args.kwarg.annotation, 'lineno') else node.lineno,
                            rule_id="PARAMSPEC001",
                            pep=612,
                            message=(
                                f"`{ps_name}.args` used to annotate `**kwargs` — "
                                f"`**kwargs` must use `{ps_name}.kwargs`"
                            ),
                            expected_behavior="error",
                            confidence=0.90,
                        ))

            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    ParamSpecVisitor().visit(tree)
    return findings


def _analyze_paramspec_constructs(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """Extended ParamSpec / Concatenate rules (PEP 612).
    PARAMSPEC002: Bare ParamSpec used where a normal type is expected.
    PARAMSPEC003: P.args/P.kwargs swapped (already partially in PARAMSPEC001).
    PARAMSPEC004: P.args and P.kwargs must appear together from the same ParamSpec.
    CONCAT001:    Concatenate used outside Callable or with wrong shape.
    """
    findings: list[SourceFinding] = []

    paramspec_names = {
        name for name, kind in symbols.typevars.items()
        if kind == "ParamSpec"
    }
    if not paramspec_names and "Concatenate" not in symbols.typing_imports:
        return findings

    # PARAMSPEC004: args/kwargs must appear together from same ParamSpec
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        vararg_ps = None
        kwarg_ps = None

        if node.args.vararg and node.args.vararg.annotation:
            ann = ast.unparse(node.args.vararg.annotation)
            for ps in paramspec_names:
                if ann == f"{ps}.args":
                    vararg_ps = ps
                    break

        if node.args.kwarg and node.args.kwarg.annotation:
            ann = ast.unparse(node.args.kwarg.annotation)
            for ps in paramspec_names:
                if ann == f"{ps}.kwargs":
                    kwarg_ps = ps
                    break

        if vararg_ps and not kwarg_ps:
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="PARAMSPEC004",
                pep=612,
                message=(
                    f"Function `{node.name}` uses `{vararg_ps}.args` on `*args` "
                    f"but is missing `{vararg_ps}.kwargs` on `**kwargs` — "
                    f"both must appear together"
                ),
                expected_behavior="error",
                confidence=0.95,
            ))
        elif kwarg_ps and not vararg_ps:
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="PARAMSPEC004",
                pep=612,
                message=(
                    f"Function `{node.name}` uses `{kwarg_ps}.kwargs` on `**kwargs` "
                    f"but is missing `{kwarg_ps}.args` on `*args` — "
                    f"both must appear together"
                ),
                expected_behavior="error",
                confidence=0.95,
            ))
        elif vararg_ps and kwarg_ps and vararg_ps != kwarg_ps:
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="PARAMSPEC004",
                pep=612,
                message=(
                    f"Function `{node.name}` mixes `{vararg_ps}.args` with "
                    f"`{kwarg_ps}.kwargs` — both must come from the same ParamSpec"
                ),
                expected_behavior="error",
                confidence=0.95,
            ))

    # PARAMSPEC002: Bare ParamSpec as a regular type annotation
    _VALID_PARAMSPEC_CONTEXTS = {"Callable", "Concatenate"}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        all_args = (
            node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        )
        for arg in all_args:
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                if arg.annotation.id in paramspec_names:
                    findings.append(SourceFinding(
                        line=arg.annotation.lineno
                        if hasattr(arg.annotation, "lineno") else node.lineno,
                        rule_id="PARAMSPEC002",
                        pep=612,
                        message=(
                            f"ParamSpec `{arg.annotation.id}` used as annotation "
                            f"for parameter `{arg.arg}` — ParamSpec can only appear "
                            f"in `Callable[P, R]` or `Concatenate[..., P]`"
                        ),
                        expected_behavior="error",
                        confidence=0.95,
                    ))

        if node.returns and isinstance(node.returns, ast.Name):
            if node.returns.id in paramspec_names:
                findings.append(SourceFinding(
                    line=node.returns.lineno
                    if hasattr(node.returns, "lineno") else node.lineno,
                    rule_id="PARAMSPEC002",
                    pep=612,
                    message=(
                        f"ParamSpec `{node.returns.id}` used as return type — "
                        f"ParamSpec can only appear in `Callable[P, R]` or "
                        f"`Concatenate[..., P]`"
                    ),
                    expected_behavior="error",
                    confidence=0.95,
                ))

    # CONCAT001: Concatenate used outside Callable or with wrong shape
    if "Concatenate" in symbols.typing_imports:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            if not (isinstance(node.value, ast.Name) and node.value.id == "Concatenate"):
                continue

            # must have at least 2 args
            if isinstance(node.slice, ast.Tuple):
                n_args = len(node.slice.elts)
            else:
                n_args = 1

            if n_args < 2:
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id="CONCAT001",
                    pep=612,
                    message=(
                        f"`Concatenate` requires at least two arguments "
                        f"(types + ParamSpec), got {n_args}"
                    ),
                    expected_behavior="error",
                    confidence=0.95,
                ))
                continue

            # Check if last arg is a ParamSpec
            if isinstance(node.slice, ast.Tuple):
                last = node.slice.elts[-1]
            else:
                last = node.slice
            last_name = None
            if isinstance(last, ast.Name):
                last_name = last.id

            if last_name and last_name not in paramspec_names:
                if not (isinstance(last, ast.Constant) and last.value is ...):
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="CONCAT001",
                        pep=612,
                        message=(
                            f"Last argument of `Concatenate` must be a ParamSpec, "
                            f"got `{last_name}`"
                        ),
                        expected_behavior="error",
                        confidence=0.90,
                    ))

    return findings


def _analyze_type_narrowing_functions(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """TypeGuard / TypeIs rules (PEP 647, PEP 742).
    TYPEGUARD001: TypeGuard function with no argument to narrow.
    TYPEGUARD002: TypeGuard function with definitely non-bool return.
    TYPEIS001:    TypeIs function with no positional argument to narrow.
    TYPEIS002:    TypeIs function with definitely non-bool return.
    """
    findings: list[SourceFinding] = []

    has_typeguard = "TypeGuard" in symbols.typing_imports
    has_typeis = "TypeIs" in symbols.typing_imports
    if not has_typeguard and not has_typeis:
        return findings

    _NON_BOOL_LITERALS = (int, float, str, bytes, type(None))

    def _returns_narrowing(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        """Return 'TypeGuard' or 'TypeIs' if the function returns one, else None."""
        if not node.returns:
            return None
        ret_str = ast.unparse(node.returns)
        if has_typeguard and "TypeGuard" in ret_str:
            return "TypeGuard"
        if has_typeis and "TypeIs" in ret_str:
            return "TypeIs"
        return None

    def _has_non_bool_return(body: list[ast.stmt]) -> tuple[bool, int, str]:
        """Check if the function body has an obvious non-bool return.
        Returns (found, line, description)."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if not isinstance(node, ast.Return):
                continue
            if node.value is None:
                return True, node.lineno, "bare return (None)"
            if isinstance(node.value, ast.Constant):
                val = node.value.value
                if val is None:
                    return True, node.lineno, "return None"
                if isinstance(val, bool):
                    continue
                if isinstance(val, _NON_BOOL_LITERALS):
                    return True, node.lineno, f"return {repr(val)}"
            if isinstance(node.value, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
                return True, node.lineno, f"return {type(node.value).__name__.lower()}"
        return False, 0, ""

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        narrowing_type = _returns_narrowing(node)
        if narrowing_type is None:
            continue

        pep = 647 if narrowing_type == "TypeGuard" else 742
        rule_no_arg = "TYPEGUARD001" if narrowing_type == "TypeGuard" else "TYPEIS001"
        rule_bad_ret = "TYPEGUARD002" if narrowing_type == "TypeGuard" else "TYPEIS002"

        # Must have at least one non-self/cls argument
        real_args = [
            a for a in (node.args.posonlyargs + node.args.args)
            if a.arg not in ("self", "cls")
        ]

        if narrowing_type == "TypeIs":
            # PEP 742: must have at least one positional arg
            if not real_args:
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id=rule_no_arg,
                    pep=pep,
                    message=(
                        f"Function `{node.name}` returns `TypeIs` but has no "
                        f"positional argument to narrow"
                    ),
                    expected_behavior="error",
                    confidence=0.95,
                ))
        else:
            # PEP 647: must have at least one argument
            all_args = real_args + list(node.args.kwonlyargs)
            if not all_args:
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id=rule_no_arg,
                    pep=pep,
                    message=(
                        f"Function `{node.name}` returns `TypeGuard` but has no "
                        f"argument to narrow"
                    ),
                    expected_behavior="error",
                    confidence=0.90,
                ))

        # Return value must be bool
        has_bad, bad_line, bad_desc = _has_non_bool_return(node.body)
        if has_bad:
            findings.append(SourceFinding(
                line=bad_line,
                rule_id=rule_bad_ret,
                pep=pep,
                message=(
                    f"Function `{node.name}` returns `{narrowing_type}` but "
                    f"has {bad_desc} — must return a boolean"
                ),
                expected_behavior="error",
                confidence=0.90,
            ))

    return findings


def _analyze_typevartuple_constructs(tree: ast.Module, symbols: ModuleSymbols) -> list[SourceFinding]:
    """TypeVarTuple rules (PEP 646).
    TVTUPLE001: TypeVarTuple used without unpacking (bare Ts instead of *Ts).
    TVTUPLE002: Multiple TypeVarTuples unpacked in same generic parameter list.
    TVTUPLE003: Multiple unpackings in a single tuple type.
    """
    findings: list[SourceFinding] = []

    tvt_names = {
        name for name, kind in symbols.typevars.items()
        if kind == "TypeVarTuple"
    }
    if not tvt_names:
        return findings

    # TVTUPLE001: Bare TypeVarTuple without unpacking
    # Valid uses: *Ts (ast.Starred), Unpack[Ts], tuple[*Ts], Generic[*Ts]
    # Invalid: x: Ts, def f(x: Ts), Generic[Ts] (without *)

    for node in ast.walk(tree):
        # Check annotations: x: Ts (bare) is invalid
        if isinstance(node, ast.AnnAssign) and node.annotation:
            if isinstance(node.annotation, ast.Name) and node.annotation.id in tvt_names:
                target_name = ast.unparse(node.target) if node.target else "?"
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id="TVTUPLE001",
                    pep=646,
                    message=(
                        f"TypeVarTuple `{node.annotation.id}` used without "
                        f"unpacking on `{target_name}` — use `*{node.annotation.id}` "
                        f"or `Unpack[{node.annotation.id}]`"
                    ),
                    expected_behavior="error",
                    confidence=0.95,
                ))

        # Check function params: def f(x: Ts) is invalid
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    if arg.annotation.id in tvt_names:
                        findings.append(SourceFinding(
                            line=arg.annotation.lineno
                            if hasattr(arg.annotation, "lineno") else node.lineno,
                            rule_id="TVTUPLE001",
                            pep=646,
                            message=(
                                f"TypeVarTuple `{arg.annotation.id}` used without "
                                f"unpacking on parameter `{arg.arg}` — use "
                                f"`*{arg.annotation.id}` or `Unpack[{arg.annotation.id}]`"
                            ),
                            expected_behavior="error",
                            confidence=0.95,
                        ))

        # Check Generic[Ts] (without *) in class bases
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if not isinstance(base, ast.Subscript):
                    continue
                if not isinstance(base.value, ast.Name):
                    continue
                if base.value.id not in ("Generic", "Protocol"):
                    continue
                # Check if any bare TVT name appears without starring
                if isinstance(base.slice, ast.Tuple):
                    for elt in base.slice.elts:
                        if isinstance(elt, ast.Name) and elt.id in tvt_names:
                            findings.append(SourceFinding(
                                line=node.lineno,
                                rule_id="TVTUPLE001",
                                pep=646,
                                message=(
                                    f"TypeVarTuple `{elt.id}` used without unpacking "
                                    f"in `{base.value.id}[...]` — use `*{elt.id}`"
                                ),
                                expected_behavior="error",
                                confidence=0.95,
                            ))
                elif isinstance(base.slice, ast.Name) and base.slice.id in tvt_names:
                    findings.append(SourceFinding(
                        line=node.lineno,
                        rule_id="TVTUPLE001",
                        pep=646,
                        message=(
                            f"TypeVarTuple `{base.slice.id}` used without unpacking "
                            f"in `{base.value.id}[...]` — use `*{base.slice.id}`"
                        ),
                        expected_behavior="error",
                        confidence=0.95,
                    ))

    # TVTUPLE002: Multiple TypeVarTuples in same generic parameter list
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if not isinstance(base, ast.Subscript):
                continue
            if not isinstance(base.value, ast.Name):
                continue
            if base.value.id not in ("Generic", "Protocol"):
                continue
            unpacked_tvts = []
            if isinstance(base.slice, ast.Tuple):
                for elt in base.slice.elts:
                    if isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
                        if elt.value.id in tvt_names:
                            unpacked_tvts.append(elt.value.id)
            if len(unpacked_tvts) > 1:
                findings.append(SourceFinding(
                    line=node.lineno,
                    rule_id="TVTUPLE002",
                    pep=646,
                    message=(
                        f"Multiple TypeVarTuples unpacked in `{base.value.id}`: "
                        f"{', '.join(unpacked_tvts)} — only one is allowed"
                    ),
                    expected_behavior="error",
                    confidence=0.95,
                ))

    # TVTUPLE003: Multiple unpackings in a single tuple type
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in ("tuple", "Tuple"):
            continue
        if not isinstance(node.slice, ast.Tuple):
            continue

        unpack_count = 0
        for elt in node.slice.elts:
            if isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
                if elt.value.id in tvt_names:
                    unpack_count += 1
            elif isinstance(elt, ast.Subscript) and isinstance(elt.value, ast.Name):
                if elt.value.id == "Unpack":
                    unpack_count += 1

        if unpack_count > 1:
            findings.append(SourceFinding(
                line=node.lineno,
                rule_id="TVTUPLE003",
                pep=646,
                message=(
                    "Multiple TypeVarTuple unpackings in `tuple[...]` — "
                    "only one unpacking is allowed per tuple type"
                ),
                expected_behavior="error",
                confidence=0.95,
            ))

    return findings


# MAIN ENTRY POINT

def analyze_source(source_code: str) -> list[SourceFinding]:
    """
    Analyze source code against PEP typing rules, independently of checker output.
    Returns high-confidence findings for definitive PEP violations only.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    symbols = build_module_symbols(tree)
    findings: list[SourceFinding] = []

    findings.extend(_analyze_lsp(symbols))
    findings.extend(_analyze_method_override_types(symbols))
    findings.extend(_analyze_self_context(tree, symbols))
    findings.extend(_analyze_isinstance_protocol(tree, symbols))
    findings.extend(_analyze_abstract_not_implemented(symbols))
    findings.extend(_analyze_override(symbols))
    findings.extend(_analyze_final(tree, symbols))
    findings.extend(_analyze_generic_subscript(tree, symbols))
    findings.extend(_analyze_typevar_names(symbols))
    findings.extend(_analyze_newtype_names(symbols))
    findings.extend(_analyze_typing_forms(tree, symbols))
    findings.extend(_analyze_overload(tree, symbols))
    findings.extend(_analyze_runtime_checkable_non_protocol(symbols))
    findings.extend(_analyze_decorator_targets(tree, symbols))
    findings.extend(_analyze_classvar_in_function(tree, symbols))
    findings.extend(_analyze_variance(tree, symbols))
    findings.extend(_analyze_noreturn(tree, symbols))
    findings.extend(_analyze_protocol_instantiation(tree, symbols))
    findings.extend(_analyze_typeddict_inheritance(symbols))
    findings.extend(_analyze_typeddict_field_conflict(symbols))
    findings.extend(_analyze_typeddict_notrequired_access(tree, symbols))
    findings.extend(_analyze_typeddict_missing_required_keys(tree, symbols))
    findings.extend(_analyze_incompatible_assignment(tree, symbols))
    findings.extend(_analyze_overload_count(tree, symbols))
    findings.extend(_analyze_return_type_none(tree, symbols))
    findings.extend(_analyze_final_property_override(symbols))
    findings.extend(_analyze_invariant_typevar_in_protocol(symbols))
    findings.extend(_analyze_newtype_base(symbols))
    findings.extend(_analyze_generic_param_count(tree, symbols))
    findings.extend(_analyze_overload_return_consistency(tree, symbols))
    findings.extend(_analyze_paramspec_misuse(tree, symbols))
    findings.extend(_analyze_paramspec_constructs(tree, symbols))
    findings.extend(_analyze_type_narrowing_functions(tree, symbols))
    findings.extend(_analyze_typevartuple_constructs(tree, symbols))

    return [f for f in findings if f.confidence >= 0.85]


# This is the summary of agent only categories currently based on my current understanding
# of types in Python. I got most of these cases from either cases where the tool could not
# detect which TC is correct, so it got back UNCERTAIN or some cases I have found in GitHub
# issues and the type conformance test suite. My understanding can be wrong in some cases,
# so it would be good to review each case with Professor Kapfhammer and/or a professional
# in the typing field of Python. It might be possible that some of these cases can be solved
# by annual testing methods or with the current phase testing that I have implemented.
# AGENT-ONLY: asyncio callback type violations — requires resolving imported asyncio types
#     (e.g. add_done_callback signature matching against bound methods)
# AGENT-ONLY: ParamSpec propagation through decorators — requires full ParamSpec inference
#     across call sites and decorator chains
# AGENT-ONLY: ParamSpec + Concatenate deep inference — requires tracking P through
#     Concatenate[..., P] across function boundaries
# AGENT-ONLY: ParamSpec + functools.partial inference — requires resolving partial()
#     argument binding against ParamSpec-typed callables
# AGENT-ONLY: Deep generic interactions — requires type inference across call sites
#     (e.g. Generic[T] → Callable[..., T] resolution)
# AGENT-ONLY: TypeGuard/TypeIs narrowing soundness — requires validating that the
#     narrowed type is compatible with the input type across branches
# AGENT-ONLY: TypeGuard + TypedDict soundness — requires checking that TypeGuard
#     narrowing preserves TypedDict required/optional key constraints
# AGENT-ONLY: TypedDict ReadOnly field mutation — requires tracking value flow from
#     ReadOnly[T] fields through function calls to detect aliased mutation
# AGENT-ONLY: TypedDict literal key assignment — requires matching dict literal keys
#     against TypedDict field definitions with type inference
# AGENT-ONLY: Overload resolution with ParamSpec — requires full overload dispatch
#     inference when ParamSpec is involved in overloaded classmethods
# AGENT-ONLY: Overload + factory + TypeGuard — requires resolving which overload
#     is selected and whether TypeGuard return type is correctly inferred
# AGENT-ONLY: ClassVar + Protocol + Self — requires resolving Self type and
#     ClassVar access patterns through Protocol structural subtyping
# AGENT-ONLY: Cyclic TypeVar with forward references — requires resolving forward
#     string references in TypeVar bounds/defaults
# AGENT-ONLY: NewType + TypeGuard interaction — requires tracking NewType wrapper
#     preservation through isinstance narrowing
# AGENT-ONLY: match/case exhaustiveness with generics — requires full pattern
#     matching type narrowing across union and generic types
# AGENT-ONLY: Literal type narrowing in comprehensions — requires tracking Literal
#     value preservation through dict/set comprehension type inference
# AGENT-ONLY: dataclass/pydantic transform inference — requires resolving
#     __dataclass_transform__ and field() type generation
# AGENT-ONLY: TypedDict literal dict construction type mismatch — requires resolving
#     which TypedDict variant a dict literal matches and comparing field types
# AGENT-ONLY: Final ClassVar access via subclass — accessing (not reassigning) a
#     Final ClassVar through a subclass name is valid Python; checkers disagree
#     on whether this is an error (design difference, not a PEP violation)
# AGENT-ONLY: TypeGuard/TypeIs return type constraint — determining whether the
#     narrowed type is a subtype of the input type requires full type inference

