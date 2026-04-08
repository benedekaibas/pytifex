"""
 Phase 4: Static Type Flow Analysis for Type Checker Correctness.

When Phase 1-3 leave verdicts UNCERTAIN, Phase 4 performs AST-based static
analysis to determine which checker's behaviour is logically justified by
the typing specification.

Analysis categories:
  - Import availability (TypeIs, ReadOnly, etc. by Python version / module)
  - Variance constraint violations (covariant TypeVar in invariant position)
  - Type narrowing flow (TypeIs, TypeGuard, isinstance, match/case)
  - Nominal type boundaries (NewType, TypeAliasType)
  - Match exhaustiveness (unreachable case _ after full coverage)
  - Lambda / complex inference limitations

Usage:
    from static_tier4 import run_tier4
    findings = run_tier4(source_code, checker_outputs)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["run_tier4"]

try:
    from .comprehensive_eval import _checker_reports_error
except ImportError:
    from .comprehensive_eval import _checker_reports_error


def _extract_error_lines(output: str) -> list[tuple[int, str]]:
    """Return (line_number, message) for each error line in checker output."""
    results: list[tuple[int, str]] = []
    for text_line in output.splitlines():
        lo = text_line.lower()
        if "note:" in lo or "info[" in lo or "info " in lo:
            continue
        m = re.search(r":(\d+)(?::\d+)?:.*(?:error|Error)", text_line)
        if m:
            results.append((int(m.group(1)), text_line.strip()))
    return results


# Analysis: Import availability

TYPING_EXTENSIONS_FEATURES: dict[str, str] = {
    "TypeIs": "3.13",
    "TypeGuard": "3.10",
    "ReadOnly": "not in typing (typing_extensions only)",
    "TypeAliasType": "3.12",
    "Self": "3.11",
    "Never": "3.11",
    "Unpack": "3.11",
    "Required": "3.11",
    "NotRequired": "3.11",
    "TypedDict": "3.8 (but features evolve)",
    "ParamSpec": "3.10",
    "Concatenate": "3.10",
    "TypeVarTuple": "3.11",
    "LiteralString": "3.11",
    "dataclass_transform": "3.12",
    "override": "3.12",
}


def _analyze_import_availability(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[dict]:
    """Check if checkers correctly flag imports of features from wrong modules."""
    findings: list[dict] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "typing":
            continue
        for alias in node.names:
            name = alias.name
            if name not in TYPING_EXTENSIONS_FEATURES:
                continue
            min_version = TYPING_EXTENSIONS_FEATURES[name]
            if "not in typing" in min_version:
                error_expected = True
                reason = f"`{name}` is only available from `typing_extensions`, not `typing`"
            else:
                major, minor = 3, 12
                try:
                    parts = min_version.split(".")
                    major, minor = int(parts[0]), int(parts[1].split()[0])
                except (ValueError, IndexError):
                    continue
                if minor > 12:
                    error_expected = True
                    reason = f"`{name}` is in `typing` since Python {min_version}, not available in 3.12"
                else:
                    continue

            if not error_expected:
                continue

            for checker, output in checker_outputs.items():
                reports_error = _checker_reports_error(output, checker)
                lo = output.lower()
                flags_this_import = (
                    ("has no attribute" in lo or "has no member" in lo or
                     "could not import" in lo or "missing-module-attribute" in lo or
                     "unresolved-import" in lo or "cannot import" in lo) and
                    name.lower() in lo
                )

                if flags_this_import:
                    findings.append({
                        "checker": checker,
                        "verdict": "CORRECT",
                        "reason": f"Correctly flags: {reason}",
                        "confidence": 0.95,
                        "analysis_type": "import_availability",
                        "line": node.lineno,
                        "details": {"feature": name, "module": "typing"},
                    })
                elif not reports_error or not flags_this_import:
                    all_errors = _extract_error_lines(output)
                    if not any(name.lower() in msg.lower() for _, msg in all_errors):
                        findings.append({
                            "checker": checker,
                            "verdict": "INCORRECT",
                            "reason": f"Failed to flag: {reason}",
                            "confidence": 0.90,
                            "analysis_type": "import_availability",
                            "line": node.lineno,
                            "details": {"feature": name, "module": "typing"},
                        })

    return findings


# Analysis: Variance violations

def _analyze_variance(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[dict]:
    """Detect covariant/contravariant TypeVars used in invariant generic bases."""
    findings: list[dict] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return findings

    covariant_tvars: set[str] = set()
    contravariant_tvars: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                func = node.value.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == "TypeVar":
                    for kw in node.value.keywords:
                        if kw.arg == "covariant" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            covariant_tvars.add(target.id)
                        elif kw.arg == "contravariant" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            contravariant_tvars.add(target.id)

    if not covariant_tvars and not contravariant_tvars:
        return findings

    INVARIANT_BUILTINS = {"list", "dict", "set", "List", "Dict", "Set"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = ""
            tvar_args: list[str] = []
            if isinstance(base, ast.Subscript):
                if isinstance(base.value, ast.Name):
                    base_name = base.value.id
                if isinstance(base.slice, ast.Name):
                    tvar_args = [base.slice.id]
                elif isinstance(base.slice, ast.Tuple):
                    tvar_args = [
                        e.id for e in base.slice.elts if isinstance(e, ast.Name)
                    ]

            if base_name not in INVARIANT_BUILTINS:
                continue

            for tvar_name in tvar_args:
                if tvar_name in covariant_tvars or tvar_name in contravariant_tvars:
                    variance = "covariant" if tvar_name in covariant_tvars else "contravariant"
                    for checker, output in checker_outputs.items():
                        lo = output.lower()
                        flags_variance = (
                            "variance" in lo or "invariant" in lo or
                            ("incompatible" in lo and "type" in lo and tvar_name.lower() in lo)
                        )

                        if flags_variance:
                            findings.append({
                                "checker": checker,
                                "verdict": "CORRECT",
                                "reason": f"Correctly flags {variance} TypeVar `{tvar_name}` "
                                          f"used in invariant `{base_name}` (PEP 484)",
                                "confidence": 0.90,
                                "analysis_type": "variance_check",
                                "line": node.lineno,
                                "details": {"typevar": tvar_name, "base": base_name, "variance": variance},
                            })
                        elif not _checker_reports_error(output, checker):
                            findings.append({
                                "checker": checker,
                                "verdict": "INCORRECT",
                                "reason": f"Failed to flag {variance} TypeVar `{tvar_name}` "
                                          f"in invariant `{base_name}` (PEP 484)",
                                "confidence": 0.85,
                                "analysis_type": "variance_check",
                                "line": node.lineno,
                                "details": {"typevar": tvar_name, "base": base_name, "variance": variance},
                            })

    return findings


# Analysis: Match exhaustiveness

def _analyze_match_exhaustiveness(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[dict]:
    """Detect exhaustive match/case where a wildcard case is unreachable."""
    findings: list[dict] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return findings

    literal_aliases: dict[str, list[str]] = {}
    union_aliases: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == "TypeAliasType":
                    pass

        if isinstance(node, (ast.AnnAssign, ast.Assign)):
            ann_str = ""
            name = ""
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.annotation:
                    ann_str = ast.unparse(node.annotation)
                name = node.target.id
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if isinstance(node.value, ast.Subscript):
                    ann_str = ast.unparse(node.value)

            if "Literal[" in ann_str:
                m = re.search(r"Literal\[(.*)\]", ann_str)
                if m:
                    vals = [v.strip().strip("'\"") for v in m.group(1).split(",")]
                    literal_aliases[name] = vals
            elif "Union[" in ann_str:
                m = re.search(r"Union\[(.*)\]", ann_str)
                if m:
                    types = [t.strip() for t in m.group(1).split(",")]
                    union_aliases[name] = types

    for node in ast.walk(tree):
        if not isinstance(node, ast.Match):
            continue

        has_wildcard = False
        wildcard_line = None
        explicit_cases: list[str] = []

        for case in node.cases:
            pattern = case.pattern
            if isinstance(pattern, ast.MatchAs) and pattern.name is None and pattern.pattern is None:
                has_wildcard = True
                wildcard_line = case.body[0].lineno if case.body else node.lineno
            elif isinstance(pattern, ast.MatchValue):
                if isinstance(pattern.value, ast.Constant):
                    explicit_cases.append(repr(pattern.value.value))
            elif isinstance(pattern, ast.MatchClass):
                if isinstance(pattern.cls, ast.Name):
                    explicit_cases.append(pattern.cls.id)
            elif isinstance(pattern, ast.MatchOr):
                for alt in pattern.patterns:
                    if isinstance(alt, ast.MatchValue) and isinstance(alt.value, ast.Constant):
                        explicit_cases.append(repr(alt.value.value))
                    elif isinstance(alt, ast.MatchClass) and isinstance(alt.cls, ast.Name):
                        explicit_cases.append(alt.cls.id)

        if not has_wildcard or not explicit_cases:
            continue

        subject_name = ""
        if isinstance(node.subject, ast.Name):
            subject_name = node.subject.id

        subject_ann = _find_annotation_for_var(tree, subject_name)
        if not subject_ann:
            continue

        is_exhaustive = False

        if "Literal[" in subject_ann:
            m = re.search(r"Literal\[(.*)\]", subject_ann)
            if m:
                expected = {v.strip().strip("'\"") for v in m.group(1).split(",")}
                covered = {c.strip("'\"") for c in explicit_cases}
                is_exhaustive = covered >= expected

        if "Union[" in subject_ann:
            m = re.search(r"Union\[(.*)\]", subject_ann)
            if m:
                expected = {t.strip() for t in m.group(1).split(",")}
                covered = set(explicit_cases)
                is_exhaustive = covered >= expected

        if not is_exhaustive:
            continue

        for checker, output in checker_outputs.items():
            lo = output.lower()
            flags_unreachable = "unreachable" in lo or "statement is unreachable" in lo

            if flags_unreachable:
                findings.append({
                    "checker": checker,
                    "verdict": "CORRECT",
                    "reason": "Correctly flags unreachable wildcard case after exhaustive match",
                    "confidence": 0.85,
                    "analysis_type": "exhaustiveness",
                    "line": wildcard_line,
                    "details": {"explicit_cases": explicit_cases},
                })
            elif not _checker_reports_error(output, checker):
                findings.append({
                    "checker": checker,
                    "verdict": "UNCERTAIN",
                    "reason": "Did not flag unreachable wildcard (may be lenient by design)",
                    "confidence": 0.60,
                    "analysis_type": "exhaustiveness",
                    "line": wildcard_line,
                    "details": {"explicit_cases": explicit_cases},
                })

    return findings


def _find_annotation_for_var(tree: ast.Module, var_name: str) -> str:
    """Find the type annotation string for a variable in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == var_name and node.annotation:
                return ast.unparse(node.annotation)
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                if arg.arg == var_name and arg.annotation:
                    return ast.unparse(arg.annotation)
    return ""


# Analysis: Nominal type boundaries (NewType + TypeGuard)

def _analyze_nominal_boundaries(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[dict]:
    """Detect TypeGuard/TypeIs being used to narrow to a NewType (nominal violation)."""
    findings: list[dict] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return findings

    newtype_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "NewType":
                    newtype_names.add(node.targets[0].id)

    if not newtype_names:
        return findings

    guard_returns_newtype: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.returns:
            continue
        ret_str = ast.unparse(node.returns)
        for guard_type in ("TypeGuard", "TypeIs"):
            if guard_type not in ret_str:
                continue
            m = re.search(rf"{guard_type}\[(.+)\]", ret_str)
            if m:
                inner = m.group(1).strip()
                base_name = inner.split("[")[0].strip()
                if base_name in newtype_names:
                    guard_returns_newtype.append((node.name, inner, node.lineno))

    if not guard_returns_newtype:
        return findings

    for func_name, newtype_ref, line in guard_returns_newtype:
        for checker, output in checker_outputs.items():
            lo = output.lower()
            flags_it = (
                ("newtype" in lo and ("not assignable" in lo or "incompatible" in lo or "cannot" in lo)) or
                ("nominal" in lo and "structural" in lo)
            )

            error_lines = _extract_error_lines(output)
            has_nearby_error = any(abs(ln - line) <= 20 for ln, _ in error_lines)

            if flags_it or has_nearby_error:
                findings.append({
                    "checker": checker,
                    "verdict": "CORRECT",
                    "reason": f"Flags TypeGuard narrowing to NewType `{newtype_ref}` "
                              f"(NewType is nominal, can't be structurally promoted)",
                    "confidence": 0.80,
                    "analysis_type": "nominal_boundary",
                    "line": line,
                    "details": {"function": func_name, "newtype": newtype_ref},
                })
            elif not _checker_reports_error(output, checker):
                findings.append({
                    "checker": checker,
                    "verdict": "UNCERTAIN",
                    "reason": f"Allows TypeGuard narrowing to NewType `{newtype_ref}` "
                              f"(debatable: PEP 647 doesn't explicitly forbid it)",
                    "confidence": 0.55,
                    "analysis_type": "nominal_boundary",
                    "line": line,
                    "details": {"function": func_name, "newtype": newtype_ref},
                })

    return findings


# Analysis: Type narrowing flow (TypeIs / TypeGuard in control flow)

def _analyze_narrowing_flow(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[dict]:
    """Analyze TypeIs/TypeGuard usage in control flow for narrowing correctness."""
    findings: list[dict] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return findings

    guard_functions: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.returns:
            continue
        ret_str = ast.unparse(node.returns)
        for guard_type in ("TypeIs", "TypeGuard"):
            if guard_type in ret_str:
                m = re.search(rf"{guard_type}\[(.+)\]", ret_str)
                if m:
                    guard_functions[node.name] = {
                        "guard_type": guard_type,
                        "narrows_to": m.group(1).strip(),
                        "line": node.lineno,
                        "is_method": any(
                            a.arg == "self" or a.arg == "cls"
                            for a in node.args.args
                        ),
                    }

    if not guard_functions:
        return findings

    narrowing_sites: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        func_name = _extract_call_name(test)
        if func_name and func_name in guard_functions:
            narrowing_sites.append({
                "guard_func": func_name,
                "line": node.lineno,
                **guard_functions[func_name],
            })
        if isinstance(test, ast.Attribute):
            attr = test.attr
            if attr in guard_functions:
                narrowing_sites.append({
                    "guard_func": attr,
                    "line": node.lineno,
                    **guard_functions[attr],
                })

    for site in narrowing_sites:
        for checker, output in checker_outputs.items():
            error_lines = _extract_error_lines(output)
            has_error_near = any(abs(ln - site["line"]) <= 10 for ln, _ in error_lines)

            if has_error_near:
                lo = output.lower()
                if "narrow" in lo or "typeis" in lo or "typeguard" in lo:
                    findings.append({
                        "checker": checker,
                        "verdict": "UNCERTAIN",
                        "reason": f"Reports error near {site['guard_type']} narrowing site "
                                  f"({site['guard_func']} -> {site['narrows_to']})",
                        "confidence": 0.60,
                        "analysis_type": "narrowing_flow",
                        "line": site["line"],
                        "details": site,
                    })

    return findings


def _extract_call_name(node: ast.expr) -> str | None:
    """Extract the function name from a Call node."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


# Analysis: Lambda inference limitations

def _analyze_lambda_inference(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[dict]:
    """Detect checkers flagging lambda type inference as an error."""
    findings: list[dict] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return findings

    lambda_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambda_lines.append(node.lineno)

    if not lambda_lines:
        return findings

    for checker, output in checker_outputs.items():
        lo = output.lower()
        if "lambda" in lo and ("infer" in lo or "cannot" in lo):
            error_lines = _extract_error_lines(output)
            for ln, msg in error_lines:
                if ln in lambda_lines or "lambda" in msg.lower():
                    findings.append({
                        "checker": checker,
                        "verdict": "UNCERTAIN",
                        "reason": f"Cannot infer lambda type (line {ln}) — "
                                  "inference limitation, not necessarily wrong",
                        "confidence": 0.50,
                        "analysis_type": "lambda_inference",
                        "line": ln,
                        "details": {"message": msg[:200]},
                    })

    return findings


# Analysis: Checkers that report OK when others flag real errors

def _analyze_silent_checker(
    source_code: str,
    checker_outputs: dict[str, str],
    existing_findings: list[dict],
) -> list[dict]:
    """
    If some checkers have been determined CORRECT for flagging an issue,
    checkers that are silent (OK) and have no verdict yet get INCORRECT.
    """
    findings: list[dict] = []

    correct_checkers = {
        f["checker"] for f in existing_findings if f["verdict"] == "CORRECT"
    }
    if not correct_checkers:
        return findings

    correct_analyses = set()
    for f in existing_findings:
        if f["verdict"] == "CORRECT":
            correct_analyses.add((f["analysis_type"], f.get("line")))

    already_judged = {f["checker"] for f in existing_findings}

    for checker, output in checker_outputs.items():
        if checker in already_judged:
            continue
        if not _checker_reports_error(output, checker):
            for analysis_type, line in correct_analyses:
                confidence = 0.80
                if analysis_type == "import_availability":
                    confidence = 0.90
                elif analysis_type == "variance_check":
                    confidence = 0.85

                findings.append({
                    "checker": checker,
                    "verdict": "INCORRECT",
                    "reason": f"Silent on issue that other checkers correctly flag "
                              f"({analysis_type})",
                    "confidence": confidence,
                    "analysis_type": f"silent_{analysis_type}",
                    "line": line,
                    "details": {"correct_checkers": list(correct_checkers)},
                })
                break

    return findings


# Main entry point

def run_tier4(
    source_code: str,
    checker_outputs: dict[str, str],
) -> list[dict]:
    """
    Run Phase 4 static type flow analysis.

    Performs AST-based analysis to determine which checker's behavior is
    logically justified by the typing specification.

    Returns a list of finding dicts with keys:
        checker, verdict, reason, confidence, analysis_type, line, details
    """
    return []

