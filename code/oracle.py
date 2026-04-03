"""
Oracle-based evaluation architecture for type checker correctness.

Architecture: analyze source → find violations → check if checkers flagged them.

1. Parse source code via AST to identify definitive PEP violations (oracle findings).
2. Parse each type checker's raw output into structured diagnostics.
3. Match oracle findings against checker diagnostics to determine whether
   each checker correctly reported the violation.

This module serves as the ground truth layer: it only emits findings where
a typing error MUST exist according to the PEP specifications then evaluates
each checker on whether it flagged those violations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from .source_analysis import SourceFinding, analyze_source
except ImportError:
    from source_analysis import SourceFinding, analyze_source


# DATA CLASSES

@dataclass(frozen=True)
class OracleFinding:
    line: int
    category: str
    rule_id: str
    pep: int | None
    message: str
    confidence: float
    expected: str = "error"


@dataclass(frozen=True)
class CheckerDiag:
    line: int
    code: str | None
    message: str
    severity: str


@dataclass
class OracleVerdict:
    verdict: str
    reason: str
    confidence: float
    findings_hit: list[OracleFinding]
    findings_missed: list[OracleFinding]


# CATEGORY MAPPING

CATEGORY_FROM_RULE_ID: dict[str, str] = {
    "LSP001": "METHOD_OVERRIDE_INCOMPATIBLE",
    "LSP002": "METHOD_OVERRIDE_INCOMPATIBLE",
    "SELF001": "SELF_OUTSIDE_CLASS",
    "PROTO001": "ISINSTANCE_NON_RUNTIME_PROTOCOL",
    "PROTO002": "RUNTIME_CHECKABLE_NON_PROTOCOL",
    "PROTO003": "PROTOCOL_INSTANTIATION",
    "ABC001": "ABSTRACT_NOT_IMPLEMENTED",
    "OVERRIDE001": "OVERRIDE_NO_BASE_METHOD",
    "FINAL001": "FINAL_REASSIGNMENT",
    "FINAL002": "FINAL_SUBCLASS",
    "FINAL003": "FINAL_METHOD_OVERRIDE",
    "GENERIC001": "NON_GENERIC_SUBSCRIPT",
    "TVAR001": "TYPEVAR_NAME_MISMATCH",
    "NEWTYPE001": "NEWTYPE_NAME_MISMATCH",
    "FORM001": "TYPING_FORM_VIOLATION",
    "OVERLOAD001": "OVERLOAD_NO_IMPLEMENTATION",
    "OVERLOAD002": "SINGLE_OVERLOAD",
    "RETURN001": "RETURN_TYPE_MISMATCH",
    "ASSIGN001": "INCOMPATIBLE_ASSIGNMENT",
    "DECO001": "DECORATOR_TARGET_VIOLATION",
    "CLASSVAR001": "CLASSVAR_IN_FUNCTION",
    "VARIANCE001": "VARIANCE_VIOLATION",
    "NORETURN001": "NORETURN_WITH_RETURN",
    "TDICT001": "TYPEDDICT_BAD_INHERITANCE",
    "TDICT002": "TYPEDDICT_FIELD_CONFLICT",
    "TDICT003": "TYPEDDICT_NOTREQUIRED_ACCESS",
    "TDICT004": "TYPEDDICT_MISSING_REQUIRED_KEY",
    "FINAL004": "FINAL_PROPERTY_OVERRIDE",
    "VARIANCE002": "INVARIANT_TYPEVAR_IN_PROTOCOL",
    "NEWTYPE002": "NEWTYPE_INVALID_BASE",
    "GENERIC002": "GENERIC_PARAM_COUNT_MISMATCH",
    "OVERLOAD003": "OVERLOAD_RETURN_INCONSISTENCY",
    "PARAMSPEC001": "PARAMSPEC_MISUSE",
    "PARAMSPEC002": "PARAMSPEC_MISUSE",
    "PARAMSPEC004": "PARAMSPEC_MISUSE",
    "CONCAT001": "CONCATENATE_VIOLATION",
    "TYPEGUARD001": "TYPEGUARD_MISUSE",
    "TYPEGUARD002": "TYPEGUARD_MISUSE",
    "TYPEIS001": "TYPEIS_MISUSE",
    "TYPEIS002": "TYPEIS_MISUSE",
    "TVTUPLE001": "TYPEVARTUPLE_MISUSE",
    "TVTUPLE002": "TYPEVARTUPLE_MISUSE",
    "TVTUPLE003": "TYPEVARTUPLE_MISUSE",
}


# CHECKER ERROR CODES

CHECKER_ERROR_CODES: dict[str, dict[str, list[str]]] = {
    "RETURN_TYPE_MISMATCH": {
        "mypy": ["return", "return-value", "empty-body", "misc"],
        "pyrefly": ["bad-return"],
        "ty": ["invalid-return-type", "empty-body"],
        "zuban": ["return", "return-value", "misc"],
    },
    "METHOD_OVERRIDE_INCOMPATIBLE": {
        "mypy": ["override", "arg-type", "misc"],
        "pyrefly": ["bad-override", "bad-param-name-override"],
        "ty": ["invalid-method-override"],
        "zuban": ["override", "arg-type", "misc"],
    },
    "INCOMPATIBLE_ASSIGNMENT": {
        "mypy": ["assignment", "arg-type", "misc"],
        "pyrefly": ["bad-assignment"],
        "ty": ["invalid-assignment"],
        "zuban": ["assignment", "arg-type", "misc"],
    },
    "SELF_OUTSIDE_CLASS": {
        "mypy": ["valid-type", "misc"],
        "pyrefly": ["invalid-self-type"],
        "ty": ["invalid-type-form"],
        "zuban": ["valid-type", "misc"],
    },
    "ISINSTANCE_NON_RUNTIME_PROTOCOL": {
        "mypy": ["type-var", "misc"],
        "pyrefly": ["unsafe-overlap"],
        "ty": ["invalid-argument-type"],
        "zuban": ["type-var", "misc"],
    },
    "ABSTRACT_NOT_IMPLEMENTED": {
        "mypy": ["abstract", "misc"],
        "pyrefly": ["bad-instantiation", "implicit-abstract-class"],
        "ty": ["abstract-class"],
        "zuban": ["abstract", "misc"],
    },
    "OVERRIDE_NO_BASE_METHOD": {
        "mypy": ["override", "misc"],
        "pyrefly": ["bad-override"],
        "ty": ["invalid-explicit-override"],
        "zuban": ["override", "misc"],
    },
    "FINAL_REASSIGNMENT": {
        "mypy": ["assignment", "misc"],
        "pyrefly": ["bad-assignment"],
        "ty": ["invalid-assignment"],
        "zuban": ["assignment", "misc"],
    },
    "FINAL_SUBCLASS": {
        "mypy": ["final", "misc"],
        "pyrefly": ["invalid-inheritance"],
        "ty": ["invalid-base"],
        "zuban": ["final", "misc"],
    },
    "FINAL_METHOD_OVERRIDE": {
        "mypy": ["override", "misc"],
        "pyrefly": ["bad-override"],
        "ty": ["invalid-method-override"],
        "zuban": ["override", "misc"],
    },
    "NON_GENERIC_SUBSCRIPT": {
        "mypy": ["type-arg", "misc"],
        "pyrefly": ["bad-specialization"],
        "ty": ["non-subscriptable"],
        "zuban": ["type-arg", "misc"],
    },
    "TYPEVAR_NAME_MISMATCH": {
        "mypy": ["name-match", "misc"],
        "pyrefly": ["invalid-type-var"],
        "ty": ["invalid-typevar-constraints"],
        "zuban": ["name-match", "misc"],
    },
    "NEWTYPE_NAME_MISMATCH": {
        "mypy": ["name-match", "misc"],
        "pyrefly": ["invalid-argument"],
        "ty": ["invalid-type-alias"],
        "zuban": ["name-match", "misc"],
    },
    "SINGLE_OVERLOAD": {
        "mypy": ["no-overload-impl", "misc"],
        "pyrefly": ["invalid-overload"],
        "ty": ["invalid-overload"],
        "zuban": ["no-overload-impl", "misc"],
    },
    "RUNTIME_CHECKABLE_NON_PROTOCOL": {
        "mypy": ["type-var", "misc"],
        "pyrefly": ["invalid-decorator"],
        "ty": ["invalid-argument-type"],
        "zuban": ["type-var", "misc"],
    },
    "PROTOCOL_INSTANTIATION": {
        "mypy": ["abstract", "attr-defined", "return-value", "misc"],
        "pyrefly": ["bad-instantiation"],
        "ty": ["abstract-class"],
        "zuban": ["abstract", "attr-defined", "return-value", "misc"],
    },
    "TYPING_FORM_VIOLATION": {
        "mypy": ["valid-type", "type-arg", "misc"],
        "pyrefly": ["invalid-annotation"],
        "ty": ["invalid-type-form"],
        "zuban": ["valid-type", "type-arg", "misc"],
    },
    "OVERLOAD_NO_IMPLEMENTATION": {
        "mypy": ["no-overload-impl", "misc"],
        "pyrefly": ["invalid-overload"],
        "ty": ["invalid-overload"],
        "zuban": ["no-overload-impl", "misc"],
    },
    "DECORATOR_TARGET_VIOLATION": {
        "mypy": ["override", "misc"],
        "pyrefly": ["invalid-decorator"],
        "ty": ["invalid-explicit-override"],
        "zuban": ["override", "misc"],
    },
    "CLASSVAR_IN_FUNCTION": {
        "mypy": ["valid-type", "misc"],
        "pyrefly": ["invalid-annotation"],
        "ty": ["invalid-type-form"],
        "zuban": ["valid-type", "misc"],
    },
    "VARIANCE_VIOLATION": {
        "mypy": ["type-var", "misc"],
        "pyrefly": ["invalid-variance", "variance-mismatch"],
        "ty": ["invalid-type-variable-default", "invalid-variance"],
        "zuban": ["type-var", "misc"],
    },
    "NORETURN_WITH_RETURN": {
        "mypy": ["return", "return-value", "misc"],
        "pyrefly": ["bad-return"],
        "ty": ["invalid-return-type"],
        "zuban": ["return", "return-value", "misc"],
    },
    "TYPEDDICT_BAD_INHERITANCE": {
        "mypy": ["typeddict-item", "misc"],
        "pyrefly": ["invalid-inheritance"],
        "ty": ["invalid-base"],
        "zuban": ["typeddict-item", "misc"],
    },
    "TYPEDDICT_FIELD_CONFLICT": {
        "mypy": ["typeddict-item", "misc"],
        "pyrefly": ["bad-typed-dict"],
        "ty": ["invalid-assignment"],
        "zuban": ["typeddict-item", "misc"],
    },
    "TYPEDDICT_NOTREQUIRED_ACCESS": {
        "mypy": ["typeddict-item", "misc"],
        "pyrefly": ["bad-typed-dict-key", "bad-typed-dict"],
        "ty": ["invalid-key", "possibly-unresolved-reference"],
        "zuban": ["typeddict-item", "misc"],
    },
    "TYPEDDICT_MISSING_REQUIRED_KEY": {
        "mypy": ["typeddict-item", "typeddict-unknown-key", "misc"],
        "pyrefly": ["bad-typed-dict", "missing-argument"],
        "ty": ["missing-key", "invalid-argument-type"],
        "zuban": ["typeddict-item", "typeddict-unknown-key", "misc"],
    },
    "FINAL_PROPERTY_OVERRIDE": {
        "mypy": ["override", "final", "misc"],
        "pyrefly": ["bad-override", "bad-assignment"],
        "ty": ["invalid-assignment", "invalid-method-override"],
        "zuban": ["override", "final", "misc"],
    },
    "INVARIANT_TYPEVAR_IN_PROTOCOL": {
        "mypy": ["misc"],
        "pyrefly": ["invalid-type-var", "invalid-variance"],
        "ty": ["invalid-type-variable-default", "invalid-variance"],
        "zuban": ["misc"],
    },
    "NEWTYPE_INVALID_BASE": {
        "mypy": ["valid-type", "misc"],
        "pyrefly": ["invalid-argument", "invalid-newtype"],
        "ty": ["invalid-newtype"],
        "zuban": ["valid-type", "misc"],
    },
    "GENERIC_PARAM_COUNT_MISMATCH": {
        "mypy": ["type-arg", "misc"],
        "pyrefly": ["bad-specialization"],
        "ty": ["too-many-type-params", "missing-type-params"],
        "zuban": ["type-arg", "misc"],
    },
    "OVERLOAD_RETURN_INCONSISTENCY": {
        "mypy": ["override", "misc"],
        "pyrefly": ["invalid-overload"],
        "ty": ["invalid-overload"],
        "zuban": ["override", "misc"],
    },
    "PARAMSPEC_MISUSE": {
        "mypy": ["valid-type", "misc", "arg-type"],
        "pyrefly": ["invalid-annotation", "invalid-argument"],
        "ty": ["invalid-type-form", "invalid-parameter-default"],
        "zuban": ["valid-type", "misc", "arg-type"],
    },
    "CONCATENATE_VIOLATION": {
        "mypy": ["valid-type", "misc"],
        "pyrefly": ["invalid-annotation", "invalid-argument"],
        "ty": ["invalid-type-form"],
        "zuban": ["valid-type", "misc"],
    },
    "TYPEGUARD_MISUSE": {
        "mypy": ["type-var", "valid-type", "misc", "return-value"],
        "pyrefly": ["invalid-annotation", "bad-return"],
        "ty": ["invalid-type-form", "invalid-return-type"],
        "zuban": ["type-var", "valid-type", "misc", "return-value"],
    },
    "TYPEIS_MISUSE": {
        "mypy": ["type-var", "valid-type", "misc", "return-value"],
        "pyrefly": ["invalid-annotation", "bad-return"],
        "ty": ["invalid-type-form", "invalid-return-type"],
        "zuban": ["type-var", "valid-type", "misc", "return-value"],
    },
    "TYPEVARTUPLE_MISUSE": {
        "mypy": ["type-var", "valid-type", "misc", "type-arg"],
        "pyrefly": ["invalid-type-var", "invalid-annotation"],
        "ty": ["invalid-type-form", "invalid-type-variable-default"],
        "zuban": ["type-var", "valid-type", "misc", "type-arg"],
    },
}


# CHECKER OUTPUT PARSING

_MYPY_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+)(?::\d+)?:\s*(?P<severity>error|warning|note):\s*(?P<message>.+?)(?:\s*\[(?P<code>[^\]]+)\])?\s*$"
)

_PYREFLY_LINE_RE = re.compile(
    r"^(?P<severity>ERROR|WARNING)\s+(?P<file>[^:]+):(?P<line>\d+):\d+\s+(?P<code>\S+)\s+(?P<message>.+)$"
)
_PYREFLY_BLOCK_HEADER_RE = re.compile(
    r"^(?P<severity>ERROR|WARNING)\s+(?P<message>.+?)\s+\[(?P<code>[^\]]+)\]\s*$"
)
_PYREFLY_BLOCK_LOCATION_RE = re.compile(
    r"^\s*-->\s*(?P<file>[^:]+):(?P<line>\d+):\d+"
)

_TY_BLOCK_HEADER_RE = re.compile(
    r"^(?P<severity>error|warning)\[(?P<code>[^\]]+)\]"
)
_TY_LOCATION_RE = re.compile(
    r"^\s*-->\s*(?P<file>[^:]+):(?P<line>\d+):\d+"
)

_TY_SINGLE_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):\d+:\s*(?P<severity>error|warning)\[(?P<code>[^\]]+)\]:\s*(?P<message>.+)$"
)

_NOISE_PATTERNS = re.compile(
    r"reveal_type|undefined-reveal|Revealed type", re.IGNORECASE
)


def parse_checker_diagnostics(output: str, checker: str) -> list[CheckerDiag]:
    lines = output.splitlines()
    diags: list[CheckerDiag] = []

    if checker in ("mypy", "zuban"):
        for raw in lines:
            m = _MYPY_LINE_RE.match(raw)
            if not m:
                continue
            severity = m.group("severity")
            if severity == "note":
                continue
            message = m.group("message")
            if _NOISE_PATTERNS.search(message):
                continue
            diags.append(CheckerDiag(
                line=int(m.group("line")),
                code=m.group("code"),
                message=message,
                severity=severity,
            ))

    elif checker == "pyrefly":
        i = 0
        while i < len(lines):
            raw = lines[i]

            sm = _PYREFLY_LINE_RE.match(raw)
            if sm:
                severity = sm.group("severity").lower()
                if severity != "note":
                    message = sm.group("message")
                    if not _NOISE_PATTERNS.search(message):
                        diags.append(CheckerDiag(
                            line=int(sm.group("line")),
                            code=sm.group("code"),
                            message=message,
                            severity=severity,
                        ))
                i += 1
                continue

            hm = _PYREFLY_BLOCK_HEADER_RE.match(raw)
            if hm:
                severity = hm.group("severity").lower()
                code = hm.group("code")
                message = hm.group("message")
                line_num = 0
                j = i + 1
                while j < len(lines):
                    lm = _PYREFLY_BLOCK_LOCATION_RE.match(lines[j])
                    if lm:
                        line_num = int(lm.group("line"))
                        break
                    if lines[j].strip() and not lines[j].startswith(" "):
                        break
                    j += 1
                if severity != "note" and not _NOISE_PATTERNS.search(message):
                    diags.append(CheckerDiag(
                        line=line_num,
                        code=code,
                        message=message,
                        severity=severity,
                    ))
                i = j + 1
                continue

            i += 1

    elif checker == "ty":
        i = 0
        while i < len(lines):
            raw = lines[i]

            sm = _TY_SINGLE_LINE_RE.match(raw)
            if sm:
                message = sm.group("message")
                if not _NOISE_PATTERNS.search(message):
                    diags.append(CheckerDiag(
                        line=int(sm.group("line")),
                        code=sm.group("code"),
                        message=message,
                        severity=sm.group("severity"),
                    ))
                i += 1
                continue

            hm = _TY_BLOCK_HEADER_RE.match(raw)
            if hm:
                severity = hm.group("severity")
                code = hm.group("code")
                header_rest = raw[hm.end():].strip().lstrip(":").strip()
                line_num = 0
                j = i + 1
                while j < len(lines):
                    lm = _TY_LOCATION_RE.match(lines[j])
                    if lm:
                        line_num = int(lm.group("line"))
                        break
                    if lines[j].strip() and not lines[j].startswith(" "):
                        break
                    j += 1
                message = header_rest if header_rest else code
                if not _NOISE_PATTERNS.search(message):
                    diags.append(CheckerDiag(
                        line=line_num,
                        code=code,
                        message=message,
                        severity=severity,
                    ))
                i = j + 1
                continue

            i += 1

    else:
        for raw in lines:
            m = _MYPY_LINE_RE.match(raw)
            if m:
                severity = m.group("severity")
                if severity == "note":
                    continue
                message = m.group("message")
                if _NOISE_PATTERNS.search(message):
                    continue
                diags.append(CheckerDiag(
                    line=int(m.group("line")),
                    code=m.group("code"),
                    message=message,
                    severity=severity,
                ))

    return diags


# MATCHING

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "RETURN_TYPE_MISMATCH": ["return type", "return value", "incompatible return"],
    "METHOD_OVERRIDE_INCOMPATIBLE": ["override", "incompatible override", "overridden"],
    "INCOMPATIBLE_ASSIGNMENT": ["incompatible type", "assignment", "cannot assign"],
    "SELF_OUTSIDE_CLASS": ["Self", "outside class", "self type"],
    "ISINSTANCE_NON_RUNTIME_PROTOCOL": ["isinstance", "runtime_checkable", "runtime checkable"],
    "ABSTRACT_NOT_IMPLEMENTED": ["abstract", "instantiate", "not implemented"],
    "OVERRIDE_NO_BASE_METHOD": ["override", "no base", "does not override"],
    "FINAL_REASSIGNMENT": ["Final", "reassign", "cannot assign to final"],
    "FINAL_SUBCLASS": ["Final", "final class", "cannot subclass"],
    "FINAL_METHOD_OVERRIDE": ["final", "override", "cannot override final"],
    "NON_GENERIC_SUBSCRIPT": ["not generic", "not subscriptable", "type-arg"],
    "TYPEVAR_NAME_MISMATCH": ["TypeVar", "name", "must match"],
    "NEWTYPE_NAME_MISMATCH": ["NewType", "name", "must match"],
    "SINGLE_OVERLOAD": ["overload", "single", "at least two"],
    "RUNTIME_CHECKABLE_NON_PROTOCOL": ["runtime_checkable", "not a protocol"],
    "PROTOCOL_INSTANTIATION": ["Protocol", "instantiat", "abstract"],
    "TYPING_FORM_VIOLATION": ["invalid type", "type form", "annotation"],
    "OVERLOAD_NO_IMPLEMENTATION": ["overload", "implementation", "no implementation"],
    "DECORATOR_TARGET_VIOLATION": ["override", "decorator", "no base"],
    "CLASSVAR_IN_FUNCTION": ["ClassVar", "function", "class body"],
    "VARIANCE_VIOLATION": ["covariant", "contravariant", "variance"],
    "NORETURN_WITH_RETURN": ["NoReturn", "return", "no return"],
    "TYPEDDICT_BAD_INHERITANCE": ["TypedDict", "inherit", "base"],
    "TYPEDDICT_FIELD_CONFLICT": ["TypedDict", "field", "conflict", "key"],
    "TYPEDDICT_NOTREQUIRED_ACCESS": ["TypedDict", "key", "missing", "NotRequired", "not required", "does not have key", "optional"],
    "TYPEDDICT_MISSING_REQUIRED_KEY": ["TypedDict", "missing", "required", "key", "Missing key"],
    "FINAL_PROPERTY_OVERRIDE": ["Final", "override", "property", "cannot override"],
    "INVARIANT_TYPEVAR_IN_PROTOCOL": ["invariant", "protocol", "contravariant", "covariant"],
    "NEWTYPE_INVALID_BASE": ["NewType", "invalid base", "not a class"],
    "GENERIC_PARAM_COUNT_MISMATCH": ["type argument", "type parameter", "too many", "too few", "expects"],
    "OVERLOAD_RETURN_INCONSISTENCY": ["overload", "return type", "incompatible"],
    "PARAMSPEC_MISUSE": ["ParamSpec", "P.args", "P.kwargs", "invalid annotation"],
    "CONCATENATE_VIOLATION": ["Concatenate", "ParamSpec", "invalid type", "type form"],
    "TYPEGUARD_MISUSE": ["TypeGuard", "type guard", "narrowing", "return type"],
    "TYPEIS_MISUSE": ["TypeIs", "type narrowing", "narrowing", "return type"],
    "TYPEVARTUPLE_MISUSE": ["TypeVarTuple", "Unpack", "unpack", "type variable tuple"],
}


def _finding_matches_diag(
    finding: OracleFinding,
    diag: CheckerDiag,
    checker: str,
    line_tolerance: int = 5,
) -> bool:
    if abs(finding.line - diag.line) > line_tolerance:
        return False

    if diag.severity != "error":
        return False

    category_map = CHECKER_ERROR_CODES.get(finding.category, {})
    codes = category_map.get(checker, [])

    if codes:
        if diag.code and diag.code in codes:
            return True
        for code in codes:
            if code in diag.message:
                return True

    if diag.code:
        all_codes = set()
        for checker_codes in category_map.values():
            all_codes.update(checker_codes)
        if diag.code in all_codes:
            return True

    keywords = CATEGORY_KEYWORDS.get(finding.category, [])
    if keywords:
        msg_lower = diag.message.lower()
        if any(kw.lower() in msg_lower for kw in keywords):
            return True

    return False


# ORACLE

def run_oracle(source_code: str) -> list[OracleFinding]:
    source_findings = analyze_source(source_code)
    oracle_findings: list[OracleFinding] = []

    for sf in source_findings:
        if sf.confidence < 0.85:
            continue
        category = CATEGORY_FROM_RULE_ID.get(sf.rule_id, sf.rule_id)
        oracle_findings.append(OracleFinding(
            line=sf.line,
            category=category,
            rule_id=sf.rule_id,
            pep=sf.pep,
            message=sf.message,
            confidence=sf.confidence,
        ))

    return oracle_findings

 # DELETE FROM HERE
def debug_matching(source_code: str, checker_outputs: dict[str, str]) -> None:
    """Print oracle findings and checker diagnostics side by side to debug matching."""
    findings = run_oracle(source_code)

    print(f"\n=== ORACLE FINDINGS ({len(findings)}) ===")
    for f in findings:
        print(f"  L{f.line} [{f.rule_id}] {f.category} (conf={f.confidence:.2f})")
        print(f"    {f.message}")

    for checker, output in checker_outputs.items():
        diags = parse_checker_diagnostics(output, checker)
        print(f"\n=== {checker.upper()} DIAGNOSTICS ({len(diags)}) ===")
        for d in diags:
            print(f"  L{d.line} [{d.code}] {d.severity}: {d.message[:80]}")

        print(f"\n=== MATCHING: {checker.upper()} ===")
        for f in findings:
            codes = CHECKER_ERROR_CODES.get(f.category, {}).get(checker, [])
            keywords = CATEGORY_KEYWORDS.get(f.category, [])
            print(f"  Finding L{f.line} [{f.rule_id}] {f.category}")
            print(f"    expected codes: {codes}")
            print(f"    expected keywords: {keywords}")
            matched = False
            for d in diags:
                hit = _finding_matches_diag(f, d, checker)
                if hit:
                    print(f"    MATCH: L{d.line} [{d.code}] {d.message[:60]}")
                    matched = True
            if not matched:
                print("    NO MATCH — closest diagnostics:")
                closest = sorted(diags, key=lambda d: abs(d.line - f.line))[:3]
                for d in closest:
                    print(f"      L{d.line} [{d.code}] {d.message[:60]}")

# DELETE UNTIL HERE

# EVALUATION

def _has_upstream_errors(
    finding: OracleFinding,
    diags: list[CheckerDiag],
) -> bool:
    """Check if the checker reports errors upstream of the finding.

    When a checker reports errors on lines *before* the oracle finding
    (e.g. cyclic TypeVar resolution failures at the declaration site),
    those upstream errors may prevent the checker from ever reaching the
    finding's line.  In that case penalising the checker for a "miss" is
    unfair — we should return UNCERTAIN instead.
    """
    for d in diags:
        if d.severity == "error" and d.line < finding.line:
            return True
    return False


def evaluate_checker(
    findings: list[OracleFinding],
    checker_output: str,
    checker: str,
) -> OracleVerdict:
    diags = parse_checker_diagnostics(checker_output, checker)
    hits: list[OracleFinding] = []
    misses: list[OracleFinding] = []
    blocked: list[OracleFinding] = []

    for finding in findings:
        matched = any(
            _finding_matches_diag(finding, d, checker)
            for d in diags
        )
        if matched:
            hits.append(finding)
        elif _has_upstream_errors(finding, diags):
            blocked.append(finding)
        else:
            misses.append(finding)

    if not findings:
        return OracleVerdict(
            verdict="UNCERTAIN",
            reason="No oracle findings to evaluate against",
            confidence=0.0,
            findings_hit=[],
            findings_missed=[],
        )

    if not misses and not blocked:
        return OracleVerdict(
            verdict="CORRECT",
            reason=f"All {len(hits)} oracle finding(s) matched by checker diagnostics",
            confidence=min(f.confidence for f in hits),
            findings_hit=hits,
            findings_missed=[],
        )

    if misses:
        return OracleVerdict(
            verdict="INCORRECT",
            reason=f"{len(misses)} of {len(findings)} oracle finding(s) not reported by checker",
            confidence=max(f.confidence for f in misses),
            findings_hit=hits,
            findings_missed=misses,
        )

    return OracleVerdict(
        verdict="UNCERTAIN",
        reason=f"{len(blocked)} oracle finding(s) blocked by upstream checker errors",
        confidence=0.0,
        findings_hit=hits,
        findings_missed=[],
    )


def run_oracle_evaluation(
    source_code: str,
    checker_outputs: dict[str, str],
) -> dict[str, OracleVerdict]:
    findings = run_oracle(source_code)
    return {
        checker: evaluate_checker(findings, output, checker)
        for checker, output in checker_outputs.items()
    }

