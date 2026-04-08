# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hypothesis",
#     "beartype",
#     "typeguard",
# ]
# ///

"""
Comprehensive Tiered Evaluation System for Type Checker Correctness (V2).
This module implements a multi-tiered evaluation strategy:

Phase 1: Runtime Crash Detection (~10% of cases)
    - Execute code and catch type-related exceptions
    - Highest confidence - proves actual bugs exist

Phase 2: Hypothesis Property-Based Testing (~40% of cases)
    - AST-driven call-site extraction and signature introspection
    - Hypothesis-generated inputs exercising real code paths
    - Proves type constraints matter in practice

Phase 3: PEP Specification Compliance (~40% of cases)
    - Check against official Python typing PEPs
    - Pattern-based rules for common disagreement types
    - Authoritative ground truth

Phase 4: Design Differences (~10% of cases)
    - Accept that some disagreements are philosophical
    - Document as legitimate design choices
"""

import ast
import sys
import re
import os
import json
import copy
import traceback
import io
import contextlib
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from pathlib import Path
from enum import Enum


try:
    from .oracle import run_oracle_evaluation, OracleVerdict, OracleFinding
    from .code_metrics import compute_metrics, metrics_to_dict
except ImportError:
    from oracle import run_oracle_evaluation, OracleVerdict, OracleFinding
    from code_metrics import compute_metrics, metrics_to_dict


class Verdict(Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class TypeBug:
    """A confirmed type-related bug found through testing."""
    line: int
    bug_type: str
    message: str
    source: str  # "tier1_runtime", "tier2_mutation", "tier3_pep"
    confidence: float
    details: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a file."""
    filename: str
    tier1_bugs: list[TypeBug]
    tier2_bugs: list[TypeBug]
    tier3_findings: list[dict]
    checker_verdicts: dict[str, dict]
    tier_reached: int
    oracle_verdicts: dict[str, OracleVerdict] = field(default_factory=dict)


class DebugArtifactCollector:
    """Collects ephemeral test snippets generated during evaluation for later inspection."""

    def __init__(self) -> None:
        self.tier1_snippets: list[dict[str, str]] = []
        self.tier2_snippets: list[dict[str, str]] = []

    def add_tier1(self, label: str, code: str) -> None:
        self.tier1_snippets.append({"label": label, "code": code})

    def add_tier2(self, annotation: str, violation: str, code: str) -> None:
        self.tier2_snippets.append({
            "annotation": annotation,
            "violation": violation,
            "code": code,
        })

    def save(self, directory: str, filename: str) -> None:
        stem = filename.removesuffix(".py")
        base = os.path.join(directory, stem)
        os.makedirs(base, exist_ok=True)

        for i, s in enumerate(self.tier1_snippets):
            path = os.path.join(base, f"tier1_{i}_{s['label']}.py")
            with open(path, "w") as f:
                f.write(s["code"])

        for i, s in enumerate(self.tier2_snippets):
            safe_ann = re.sub(r"[^\w]", "_", s["annotation"])[:40]
            path = os.path.join(base, f"tier2_{i}_{safe_ann}.py")
            with open(path, "w") as f:
                f.write(f"# annotation: {s['annotation']}\n")
                f.write(f"# violation:  {s['violation']}\n\n")
                f.write(s["code"])


# =============================================================================
# TIER 1: RUNTIME CRASH DETECTION
# =============================================================================

TYPE_ERROR_EXCEPTIONS = (TypeError, KeyError, AttributeError)


def _extract_all_source_lines(tb_list: list, source_tag: str = "<tier1>") -> list[int]:
    """Extract all line numbers from traceback frames that belong to our source."""
    return [frame.lineno for frame in tb_list if frame.filename == source_tag]


def _collect_chained_exceptions(exc: BaseException) -> list[BaseException]:
    """Walk __cause__ and __context__ chains to find all related exceptions.
    
    Returns exceptions root-cause-first: the deepest chained exception comes
    first so that _bugs_from_exception attributes bugs to the original fault
    rather than to re-raise sites in except blocks.
    """
    seen: set[int] = set()
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    chain.reverse()
    return chain


def _extract_try_bodies(source_code: str) -> list[tuple[int, int, str]]:
    """AST-scan for try/except blocks and return (start_line, end_line, body_source)."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    source_lines = source_code.splitlines()
    bodies: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            body_start = node.body[0].lineno
            body_end = node.body[-1].end_lineno or node.body[-1].lineno
            body_source = "\n".join(source_lines[body_start - 1 : body_end])
            bodies.append((body_start, body_end, body_source))

    return bodies


def _run_isolated_code(code: str, source_tag: str = "<tier1_isolated>") -> list[TypeBug]:
    """Execute code and collect type-related bugs with full traceback info."""
    bugs: list[TypeBug] = []
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(compile(code, source_tag, "exec"), {"__name__": "__main__"})
    except TYPE_ERROR_EXCEPTIONS as e:
        bugs.extend(_bugs_from_exception(e, source_tag))
    except Exception:
        pass
    return bugs


def _bugs_from_exception(exc: BaseException, source_tag: str) -> list[TypeBug]:
    """Create TypeBug entries from an exception and its chain."""
    bugs: list[TypeBug] = []
    seen_lines: set[int] = set()

    chain = _collect_chained_exceptions(exc)
    all_chain_lines: list[int] = []
    for chained_exc in chain:
        tb_info = traceback.extract_tb(chained_exc.__traceback__)
        all_chain_lines.extend(_extract_all_source_lines(tb_info, source_tag))
    if not all_chain_lines:
        last_exc = chain[-1] if chain else exc
        tb_info = traceback.extract_tb(last_exc.__traceback__)
        all_chain_lines = [tb_info[-1].lineno] if tb_info else [0]

    for chained_exc in chain:
        if not isinstance(chained_exc, TYPE_ERROR_EXCEPTIONS):
            continue

        bug_type = type(chained_exc).__name__
        message = str(chained_exc)[:200]
        if isinstance(chained_exc, KeyError):
            message = f"KeyError: {chained_exc}"

        tb_info = traceback.extract_tb(chained_exc.__traceback__)
        exc_source_lines = _extract_all_source_lines(tb_info, source_tag)

        if not exc_source_lines:
            exc_source_lines = [tb_info[-1].lineno] if tb_info else [0]

        primary_line = exc_source_lines[-1]

        if primary_line in seen_lines:
            continue
        seen_lines.add(primary_line)

        bugs.append(TypeBug(
            line=primary_line,
            bug_type=bug_type,
            message=message,
            source="tier1_runtime",
            confidence=1.0,
            details={"all_traceback_lines": list(dict.fromkeys(all_chain_lines))},
        ))

    return bugs


def run_tier1(
    source_code: str,
    debug: DebugArtifactCollector | None = None,
) -> list[TypeBug]:
    """
    Tier 1: Execute code and catch type-related runtime exceptions.
    
    V2 improvements:
    - Walks the full traceback to find the root cause line
    - Inspects exception chains (__cause__ / __context__)
    - Isolates try/except bodies to surface swallowed type errors
    """
    bugs = _run_isolated_code(source_code, "<tier1>")
    if debug:
        debug.add_tier1("full_source", source_code)

    try_bodies = _extract_try_bodies(source_code)
    seen_lines = {b.line for b in bugs}

    for idx, (start_line, end_line, body_source) in enumerate(try_bodies):
        if debug:
            debug.add_tier1(f"try_body_{idx}_L{start_line}", body_source)
        isolated_bugs = _run_isolated_code(body_source, "<tier1_isolated>")
        for bug in isolated_bugs:
            adjusted_line = bug.line + start_line - 1
            if adjusted_line not in seen_lines:
                seen_lines.add(adjusted_line)
                bug.line = adjusted_line
                bug.details["isolated_from_try"] = True
                bug.details["all_traceback_lines"] = [
                    ln + start_line - 1 for ln in bug.details.get("all_traceback_lines", [])
                ]
                bug.confidence = 0.95
                bugs.append(bug)

    return bugs



# =============================================================================
# TIER 3: PEP SPECIFICATION COMPLIANCE
# =============================================================================

@dataclass
class PEPRule:
    """A rule derived from Python typing PEPs."""
    pep_number: int
    pattern: str  # regex pattern to match in checker output or code
    rule_description: str
    correct_behavior: str  # "error" or "ok"


PEP_RULES = [
    # ── PEP 484: Type Hints (core) ──────────────────────────────────────
    PEPRule(
        pep_number=484,
        pattern=r"(?:override|LSP|Liskov|incompatible).*method",
        rule_description="Method override must be compatible (PEP 484 LSP)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"NewType.*float",
        rule_description="float is a valid base for NewType (PEP 484)",
        correct_behavior="ok",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"not assignable to.*NewType|NewType.*not assignable",
        rule_description="NewType creates a distinct nominal type (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"invalid.argument.type|incompatible type.*expected",
        rule_description="Argument type must match parameter annotation (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"invalid.assignment|not assignable to declared type",
        rule_description="Assigned value must be compatible with declared type (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"return type.*incompatible|incompatible return",
        rule_description="Return value must match return type annotation (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"not subscriptable|not (?:a )?generic|cannot subscript",
        rule_description="Only generic classes can be subscripted (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"type application.*only supported for generic",
        rule_description="Type application is only supported for generic classes (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"has no attribute|unresolved.reference",
        rule_description="Attribute access must resolve on the declared type (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"(?:Missing return statement|Function.*implicitly return.*None|bad-return|invalid-return-type)",
        rule_description="Non-Optional return type must return a value on all code paths (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=484,
        pattern=r"[Ii]ncompatible types? in assignment",
        rule_description="Assigned value must be compatible with the annotated variable type (PEP 484)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=544,
        pattern=r"[Ii]nvariant type variable.*protocol.*covariant|protocol.*[Ii]nvariant.*covariant",
        rule_description="Protocol type parameters must use appropriate variance (PEP 544)",
        correct_behavior="error",
    ),

    # ── PEP 526: Variable Annotations ───────────────────────────────────
    PEPRule(
        pep_number=526,
        pattern=r"ClassVar.*(?:instance|self)|instance.*ClassVar",
        rule_description="ClassVar cannot be set on instances (PEP 526)",
        correct_behavior="error",
    ),

    # ── PEP 544: Protocols ──────────────────────────────────────────────
    PEPRule(
        pep_number=544,
        pattern=r"[Pp]rotocol.*cannot be instantiated|instantiate.*[Pp]rotocol",
        rule_description="Protocol classes cannot be instantiated directly (PEP 544)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=544,
        pattern=r"does not (?:implement|satisfy|conform).*[Pp]rotocol",
        rule_description="Type must implement all Protocol members (PEP 544)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=544,
        pattern=r"incompatible.*[Pp]rotocol|not compatible with.*[Pp]rotocol",
        rule_description="Type is incompatible with Protocol (PEP 544)",
        correct_behavior="error",
    ),

    # ── PEP 586: Literal Types ──────────────────────────────────────────
    PEPRule(
        pep_number=586,
        pattern=r"str.*(?:to|→|->).*Literal\[",
        rule_description="str is not assignable to Literal[...] (PEP 586)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=586,
        pattern=r"Literal\[.*\].*(?:to|→|->).*str",
        rule_description="Literal[...] is assignable to str (PEP 586)",
        correct_behavior="ok",
    ),
    PEPRule(
        pep_number=586,
        pattern=r"invalid.*Literal|Literal.*invalid",
        rule_description="Literal parameters must be valid literal values (PEP 586)",
        correct_behavior="error",
    ),

    # ── PEP 589: TypedDict ──────────────────────────────────────────────
    PEPRule(
        pep_number=589,
        pattern=r"[Mm]issing.*(?:required|key).*TypedDict",
        rule_description="Missing required key in TypedDict (PEP 589)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=589,
        pattern=r"TypedDict.*[Mm]issing.*key",
        rule_description="Missing required key in TypedDict (PEP 589)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=589,
        pattern=r"TypedDict.*(?:extra|unexpected).*key|(?:extra|unexpected).*key.*TypedDict",
        rule_description="Extra keys not allowed in TypedDict (PEP 589)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=589,
        pattern=r"[Oo]verwriting TypedDict field",
        rule_description="Overwriting TypedDict field while extending (PEP 589)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=589,
        pattern=r"does not have key|not (?:a )?valid.*key.*TypedDict|bad.typed.dict.key",
        rule_description="Accessing invalid key on TypedDict (PEP 589)",
        correct_behavior="error",
    ),

    # ── PEP 591: Final ──────────────────────────────────────────────────
    PEPRule(
        pep_number=591,
        pattern=r"[Cc]annot (?:assign|override|overwrite).*Final|Final.*reassign",
        rule_description="Final variables cannot be reassigned (PEP 591)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=591,
        pattern=r"[Cc]annot override.*[Ff]inal.*method|[Ff]inal.*method.*override",
        rule_description="Final methods cannot be overridden (PEP 591)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=591,
        pattern=r"[Cc]annot (?:subclass|inherit|extend).*[Ff]inal",
        rule_description="Final classes cannot be subclassed (PEP 591)",
        correct_behavior="error",
    ),

    # ── PEP 604: X | Y Union Syntax ────────────────────────────────────
    PEPRule(
        pep_number=604,
        pattern=r"unsupported operand.*\|.*type",
        rule_description="X | Y union syntax requires Python 3.10+ (PEP 604)",
        correct_behavior="error",
    ),

    # ── PEP 612: ParamSpec ──────────────────────────────────────────────
    PEPRule(
        pep_number=612,
        pattern=r"ParamSpec.*(?:invalid|incorrect|misuse)|invalid.*ParamSpec",
        rule_description="ParamSpec must be used correctly (PEP 612)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=612,
        pattern=r"Concatenate.*(?:invalid|incorrect)|invalid.*Concatenate",
        rule_description="Concatenate must be used with ParamSpec (PEP 612)",
        correct_behavior="error",
    ),

    # ── PEP 613: Explicit Type Aliases ──────────────────────────────────
    PEPRule(
        pep_number=613,
        pattern=r"TypeAlias.*(?:invalid|incorrect)|invalid.*TypeAlias",
        rule_description="TypeAlias must be a valid type expression (PEP 613)",
        correct_behavior="error",
    ),

    # ── PEP 634: Structural Pattern Matching ────────────────────────────
    PEPRule(
        pep_number=634,
        pattern=r"[Ss]tatement is unreachable|unreachable code",
        rule_description="Unreachable code after exhaustive match (PEP 634)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=634,
        pattern=r"__match_args__|bad.match|cannot match positional",
        rule_description="Class must define __match_args__ for positional patterns (PEP 634)",
        correct_behavior="error",
    ),

    # ── PEP 646: Variadic Generics ──────────────────────────────────────
    PEPRule(
        pep_number=646,
        pattern=r"TypeVarTuple.*(?:invalid|incorrect)|invalid.*TypeVarTuple",
        rule_description="TypeVarTuple must be used correctly (PEP 646)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=646,
        pattern=r"[Uu]npack.*(?:invalid|only|must)|invalid.*[Uu]npack",
        rule_description="Unpack must be used with TypeVarTuple (PEP 646)",
        correct_behavior="error",
    ),

    # ── PEP 647: TypeGuard ──────────────────────────────────────────────
    PEPRule(
        pep_number=647,
        pattern=r"TypeGuard.*narrow",
        rule_description="TypeGuard narrows to specified type (PEP 647)",
        correct_behavior="ok",
    ),
    PEPRule(
        pep_number=647,
        pattern=r"TypeGuard.*positional argument|[Tt]ype guard.*positional argument",
        rule_description="TypeGuard function must accept at least one positional argument (PEP 647)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=647,
        pattern=r"TypeGuard.*[Bb]ool|TypeGuard.*return",
        rule_description="TypeGuard function must return bool (PEP 647)",
        correct_behavior="error",
    ),

    # ── PEP 655: Required / NotRequired ─────────────────────────────────
    PEPRule(
        pep_number=655,
        pattern=r"Required\[.*\].*missing|missing.*Required",
        rule_description="Required[] TypedDict keys must be present (PEP 655)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=655,
        pattern=r"NotRequired.*(?:invalid|outside TypedDict)|invalid.*NotRequired",
        rule_description="NotRequired can only be used in TypedDict (PEP 655)",
        correct_behavior="error",
    ),

    # ── PEP 673: Self Type ──────────────────────────────────────────────
    PEPRule(
        pep_number=673,
        pattern=r"Self.*outside.*class|Self.*(?:invalid|not allowed).*(?:function|module)",
        rule_description="Self can only be used inside class methods (PEP 673)",
        correct_behavior="error",
    ),

    # ── PEP 675: LiteralString ──────────────────────────────────────────
    PEPRule(
        pep_number=675,
        pattern=r"not.*LiteralString|LiteralString.*expected",
        rule_description="Non-literal string not assignable to LiteralString (PEP 675)",
        correct_behavior="error",
    ),

    # ── PEP 681: Data Class Transforms ──────────────────────────────────
    PEPRule(
        pep_number=681,
        pattern=r"dataclass_transform.*(?:invalid|incorrect)|invalid.*dataclass_transform",
        rule_description="dataclass_transform must be used correctly (PEP 681)",
        correct_behavior="error",
    ),

    # ── PEP 692: Unpack for **kwargs ────────────────────────────────────
    PEPRule(
        pep_number=692,
        pattern=r"Unpack.*kwargs|kwargs.*Unpack.*TypedDict",
        rule_description="**kwargs Unpack must use TypedDict (PEP 692)",
        correct_behavior="error",
    ),

    # ── PEP 695: Type Parameter Syntax ──────────────────────────────────
    PEPRule(
        pep_number=695,
        pattern=r"type.*statement.*invalid|invalid.*type alias.*statement",
        rule_description="Type alias statement must be valid (PEP 695)",
        correct_behavior="error",
    ),

    # ── PEP 696: Type Defaults for Type Parameters ──────────────────────
    PEPRule(
        pep_number=696,
        pattern=r"default.*TypeVar.*invalid|TypeVar.*default.*(?:invalid|not allowed)",
        rule_description="TypeVar default must be valid (PEP 696)",
        correct_behavior="error",
    ),

    # ── PEP 698: @override ──────────────────────────────────────────────
    PEPRule(
        pep_number=698,
        pattern=r"@override.*no base.*method|override.*does not override",
        rule_description="@override method must override a base class method (PEP 698)",
        correct_behavior="error",
    ),

    # ── PEP 705: TypedDict ReadOnly ─────────────────────────────────────
    PEPRule(
        pep_number=705,
        pattern=r"ReadOnly.*(?:only|must).*TypedDict|ReadOnly.*(?:invalid|cannot)",
        rule_description="ReadOnly can only be used in TypedDict (PEP 705)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=705,
        pattern=r"[Cc]annot (?:assign|write|mutate).*ReadOnly|ReadOnly.*(?:assign|mutate|write)",
        rule_description="ReadOnly TypedDict fields cannot be mutated (PEP 705)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=705,
        pattern=r"read.?only.*mutable|mutable.*read.?only|read.?write.*ReadOnly",
        rule_description="ReadOnly field incompatible with mutable parent (PEP 705)",
        correct_behavior="error",
    ),

    # ── PEP 742: TypeIs ─────────────────────────────────────────────────
    PEPRule(
        pep_number=742,
        pattern=r"TypeIs.*positional argument|[Tt]ype.?is.*positional argument",
        rule_description="TypeIs function must accept at least one positional argument (PEP 742)",
        correct_behavior="error",
    ),
    PEPRule(
        pep_number=742,
        pattern=r"TypeIs.*not.*subtype|TypeIs.*narrowing.*invalid",
        rule_description="TypeIs narrowed type must be subtype of input (PEP 742)",
        correct_behavior="error",
    ),
]


MODULE_IMPORT_RE = re.compile(
    r"[Mm]odule.*has no (?:attribute|member)|[Cc]ould not import|"
    r"unresolved.import|missing.module",
    re.IGNORECASE,
)


def run_tier3(source_code: str, checker_outputs: dict[str, str]) -> list[dict]:
    """
    Tier 3: Check against PEP specifications.
    
    Analyzes the code and checker outputs to determine which checker
    follows the official Python typing specifications.
    
    Matches PEP rules line-by-line against checker output to avoid
    false matches from note/info lines or module import errors.
    """
    findings = []
    
    for rule in PEP_RULES:
        for checker, output in checker_outputs.items():
            matched = False
            for text_line in output.splitlines():
                lower = text_line.lower()
                if "note:" in lower or "info[" in lower or lower.startswith("info "):
                    continue
                if "unknown-name" in lower and "reveal_type" in text_line:
                    continue
                if "undefined-reveal" in lower:
                    continue
                if MODULE_IMPORT_RE.search(text_line):
                    continue
                if re.search(rule.pattern, text_line, re.IGNORECASE):
                    matched = True
                    break
            
            if not matched:
                continue

            checker_says_error = _checker_reports_error(output, checker)
            
            is_correct = (
                (rule.correct_behavior == "error" and checker_says_error) or
                (rule.correct_behavior == "ok" and not checker_says_error)
            )
            
            findings.append({
                "checker": checker,
                "pep": rule.pep_number,
                "rule": rule.rule_description,
                "checker_behavior": "error" if checker_says_error else "ok",
                "correct_behavior": rule.correct_behavior,
                "is_correct": is_correct,
                "confidence": 0.85,
            })
    
    # Also check code patterns that should trigger specific rules
    code_findings = _analyze_code_patterns(source_code, checker_outputs)
    findings.extend(code_findings)
    
    # Source-aware analysis: analyze AST independently, then judge each checker
    source_findings = _run_source_analysis(source_code, checker_outputs)
    findings.extend(source_findings)
    
    return findings


def _run_source_analysis(source_code: str, checker_outputs: dict[str, str]) -> list[dict]:
    """
    Run AST-level source analysis independently of checker output, then
    judge each checker by whether it reported errors near the violations found.
    """
    try:
        from .source_analysis import analyze_source
    except ImportError:
        from source_analysis import analyze_source

    source_findings = analyze_source(source_code)
    if not source_findings:
        return []

    results: list[dict] = []
    high_confidence_findings = [f for f in source_findings if f.confidence >= 0.85]

    for finding in high_confidence_findings:
        for checker, output in checker_outputs.items():
            checker_error_lines = extract_checker_error_lines(output)
            checker_reports = _checker_reports_error(output, checker)

            has_error_near = any(
                abs(ln - finding.line) <= 10 for ln in checker_error_lines
            )

            if has_error_near:
                results.append({
                    "checker": checker,
                    "pep": finding.pep,
                    "rule": f"[{finding.rule_id}] {finding.message}",
                    "checker_behavior": "error",
                    "correct_behavior": "error",
                    "is_correct": True,
                    "confidence": finding.confidence,
                    "source": "source_analysis",
                })
            else:
                is_wrong_location = checker_reports and checker_error_lines
                results.append({
                    "checker": checker,
                    "pep": finding.pep,
                    "rule": f"[{finding.rule_id}] {finding.message}",
                    "checker_behavior": "ok" if not checker_reports else "error_wrong_location",
                    "correct_behavior": "error",
                    "is_correct": False,
                    "confidence": 0.5 if is_wrong_location else finding.confidence,
                    "source": "source_analysis",
                })

    return results


def _checker_reports_error(output: str, checker: str = "") -> bool:
    """Determine if a checker output indicates an error.

    Uses checker-specific parsing when *checker* is provided so that
    summary lines like ``"Found 0 errors"`` or ``"INFO 0 errors"`` are
    not misclassified as errors.
    """
    checker = checker.lower()

    if checker == "mypy" or checker == "zuban":
        # mypy / zuban share the same output format.
        # Success → "Success: no issues found in 1 source file"
        # Error   → "Found N errors in M file (checked …)" where N > 0
        if "success: no issues found" in output.lower():
            return False
        m = re.search(r"Found\s+(\d+)\s+errors?\s+in", output)
        if m:
            return int(m.group(1)) > 0
        # Fallback: look for individual error lines
        for line in output.splitlines():
            if re.search(r":\s*error\b", line, re.IGNORECASE):
                return True
        return False

    if checker == "pyrefly":
        real_errors = 0
        for line in output.splitlines():
            s = line.strip()
            if not s.startswith("ERROR"):
                continue
            if "unknown-name" in s.lower() and "reveal_type" in s:
                continue
            real_errors += 1
        return real_errors > 0

    if checker == "ty":
        # ty uses "All checks passed!" for clean runs.
        # Errors are "error[rule-name]:" lines; warnings/infos are not errors.
        # Summary: "Found N diagnostics" (includes warnings/infos).
        if "all checks passed" in output.lower():
            return False
        for line in output.splitlines():
            if re.match(r"\s*error\[", line, re.IGNORECASE):
                return True
        return False

    # Generic fallback (unknown checker) — preserve old heuristic
    output_lower = output.lower()
    return (
        "error" in output_lower and
        "0 error" not in output_lower and
        "success" not in output_lower
    )


def _analyze_code_patterns(source_code: str, checker_outputs: dict[str, str]) -> list[dict]:
    """Analyze code for patterns that have clear PEP-defined behavior."""
    findings = []
    
    # Pattern 1: str assigned to Literal (PEP 586)
    if re.search(r':\s*str\s*=', source_code) and re.search(r'Literal\[', source_code):
        # Check if any assignment is str -> Literal
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):
                    ann = ast.unparse(node.annotation) if node.annotation else ""
                    if "Literal[" in ann:
                        # This is a Literal annotation - checkers should flag str -> Literal
                        for checker, output in checker_outputs.items():
                            checker_says_error = _checker_reports_error(output, checker)
                            # Per PEP 586, str -> Literal should be an error
                            # But we need to check if the SOURCE is str, not the value
                            if "str" in output.lower() and "literal" in output.lower():
                                findings.append({
                                    "checker": checker,
                                    "pep": 586,
                                    "rule": "str is not assignable to Literal (PEP 586)",
                                    "line": node.lineno,
                                    "checker_behavior": "error" if checker_says_error else "ok",
                                    "correct_behavior": "error",
                                    "is_correct": checker_says_error,
                                    "confidence": 0.8,
                                })
        except SyntaxError:
            pass
    
    return findings


# =============================================================================
# VERDICT DETERMINATION
# =============================================================================

@dataclass 
class FunctionSpan:
    """Represents a function's location in source code."""
    name: str
    start_line: int
    end_line: int
    class_name: Optional[str] = None


def extract_function_spans(source_code: str) -> list[FunctionSpan]:
    """Extract all function spans from source code."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    
    spans = []
    
    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None
        
        def visit_ClassDef(self, node):
            old = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old
        
        def visit_FunctionDef(self, node):
            end = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 20
            spans.append(FunctionSpan(node.name, node.lineno, end, self.current_class))
            self.generic_visit(node)
    
    FunctionVisitor().visit(tree)
    return spans


def extract_checker_error_lines(output: str) -> list[int]:
    """Extract line numbers from checker error output (excludes notes/info)."""
    lines = []
    output_lines = output.splitlines()
    awaiting_arrow = False
    for text_line in output_lines:
        lower = text_line.lower()
        if "note:" in lower or "info[" in lower or "info " in lower:
            awaiting_arrow = False
            continue

        if text_line.startswith("ERROR ") or re.match(r'error\[', text_line):
            if "unknown-name" in lower and "reveal_type" in text_line:
                awaiting_arrow = False
                continue
            if "undefined-reveal" in lower:
                awaiting_arrow = False
                continue
            awaiting_arrow = True
            m = re.search(r'\.py:(\d+)(?::\d+)?:', text_line)
            if m:
                try:
                    lines.append(int(m.group(1)))
                except (ValueError, IndexError):
                    pass
                awaiting_arrow = False
            continue

        if awaiting_arrow:
            m = re.search(r'-->\s+\S+?:(\d+)(?::\d+)?', text_line)
            if m:
                try:
                    lines.append(int(m.group(1)))
                except (ValueError, IndexError):
                    pass
                awaiting_arrow = False
            continue

        m = re.search(r'\.py:(\d+)(?::\d+)?:', text_line)
        if m and ("error" in lower or "Error" in text_line):
            try:
                lines.append(int(m.group(1)))
            except (ValueError, IndexError):
                pass
            continue
        m = re.search(r':(\d+):.*(?:error|Error)', text_line)
        if m:
            try:
                lines.append(int(m.group(1)))
            except (ValueError, IndexError):
                pass
    return list(set(lines))


# Typing constructs the Oracle (source_analysis.py) cannot evaluate.
# If a source file uses any of these, the Oracle's silence does NOT mean
# the file is violation-free — it means the Oracle is blind to those
# constructs.  Derived from COVERAGE_MATRIX.md Section 3 (AGENT-ONLY).
_UNCOVERED_TYPING_NAMES = frozenset({
    "TypeGuard", "TypeIs",
    "ParamSpec", "ParamSpecArgs", "ParamSpecKwargs", "Concatenate",
    "TypeVarTuple", "Unpack",
    "TypeAliasType",
    "ReadOnly",
    "dataclass_transform",
    "Required", "NotRequired",
})

_UNCOVERED_CONSTRUCT_RE = re.compile(
    r"|".join([
        r"\bParamSpec\b",
        r"\bTypeGuard\b",
        r"\bTypeIs\b",
        r"\bConcatenate\b",
        r"\bTypeVarTuple\b",
        r"\bUnpack\b",
        r"\bTypeAliasType\b",
        r"\bReadOnly\b",
        r"\bdataclass_transform\b",
        r"\bRequired\b",
        r"\bNotRequired\b",
    ])
)


def _source_has_uncovered_constructs(source_code: str) -> bool:
    """Return True if the source uses typing constructs the Oracle cannot analyze."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module in ("typing", "typing_extensions")
            or node.module.startswith("typing.")
        ):
            for alias in node.names:
                if alias.name in _UNCOVERED_TYPING_NAMES:
                    return True

    if _UNCOVERED_CONSTRUCT_RE.search(source_code):
        return True

    return False


def _check_bugs_against_checker(
    bugs: list[TypeBug],
    checker_error_lines: list[int],
    function_spans: list[FunctionSpan],
) -> tuple[list[TypeBug], list[TypeBug]]:
    """Check which bugs a checker caught vs missed based on error line proximity."""
    caught = []
    missed = []
    for bug in bugs:
        found = False
        bug_lines = bug.details.get("all_traceback_lines", [bug.line])
        for error_line in checker_error_lines:
            for bl in bug_lines:
                if abs(bl - error_line) <= 5:
                    found = True
                    break
            if found:
                break
            bug_func = _get_function_at_line(function_spans, bug.line)
            err_func = _get_function_at_line(function_spans, error_line)
            if bug_func and err_func and bug_func.name == err_func.name:
                found = True
                break
        if found:
            caught.append(bug)
        else:
            missed.append(bug)
    return caught, missed


def determine_verdicts(
    tier1_bugs: list[TypeBug],
    tier2_bugs: list[TypeBug],
    tier3_findings: list[dict],
    checker_outputs: dict[str, str],
    source_code: str,
    tier4_findings: list[dict] | None = None,
    oracle_verdicts: dict[str, OracleVerdict] | None = None,
) -> dict[str, dict]:
    """
    Determine final verdict for each checker using all available evidence.

    Priority:
      1. Tier 1 runtime crashes (highest confidence — runtime proof)
      2. Tier 0 Oracle (AST-based ground truth)
      3. Tier 2 Hypothesis bugs (high confidence — runtime proof via fuzzing)
      4. Tier 3 PEP specification compliance (checker-specific evidence)
      5. Tier 4 static flow analysis (checker-specific evidence)
      6. False positive detection (oracle clean + checker reports errors)
    """
    if tier4_findings is None:
        tier4_findings = []
    if oracle_verdicts is None:
        oracle_verdicts = {}

    verdicts = {}
    function_spans = extract_function_spans(source_code)

    tier1_bugs_only = [b for b in tier1_bugs if b.confidence >= 0.85 and b.source == "tier1_runtime"]
    tier2_bugs_high = [b for b in tier2_bugs if b.confidence >= 0.85]

    checker_error_status = {
        c: _checker_reports_error(o, c) for c, o in checker_outputs.items()
    }

    oracle_has_no_findings = all(
        ov.verdict == "UNCERTAIN" and not ov.findings_hit and not ov.findings_missed
        for ov in oracle_verdicts.values()
    ) if oracle_verdicts else True

    for checker, output in checker_outputs.items():
        checker_reported_error = checker_error_status[checker]
        checker_error_lines = extract_checker_error_lines(output)

        # --- Tier 1: runtime crashes (runtime proof outranks everything) ---
        if tier1_bugs_only:
            bugs_caught, bugs_missed = _check_bugs_against_checker(
                tier1_bugs_only, checker_error_lines, function_spans,
            )

            if bugs_caught and not bugs_missed:
                verdicts[checker] = {
                    "verdict": Verdict.CORRECT.value,
                    "reason": f"Caught {len(bugs_caught)} proven runtime bug(s)",
                    "confidence": 0.9,
                    "tier": 1,
                }
                continue
            elif bugs_missed:
                verdicts[checker] = {
                    "verdict": Verdict.INCORRECT.value,
                    "reason": f"Missed {len(bugs_missed)} proven runtime bug(s)",
                    "confidence": 0.95,
                    "tier": 1,
                    "missed_bugs": [{"line": b.line, "type": b.bug_type} for b in bugs_missed],
                }
                continue

        # --- Tier 0: Oracle (AST-based ground truth) ---
        ov = oracle_verdicts.get(checker)
        if ov and ov.verdict != "UNCERTAIN":
            # Before trusting oracle INCORRECT, check if tier3 has evidence
            # that this checker is actually correct. The oracle uses line-level
            # matching which can fail even when the checker caught the right
            # issue (e.g., different error code or line offset).
            if ov.verdict == "INCORRECT":
                checker_t3_override = [
                    f for f in tier3_findings
                    if f.get("checker") == checker and f.get("is_correct") is True
                ]
                if checker_t3_override:
                    peps = sorted({f.get("pep", 0) for f in checker_t3_override})
                    verdicts[checker] = {
                        "verdict": Verdict.CORRECT.value,
                        "reason": (
                            f"Oracle matching failed but tier3 confirms checker is correct "
                            f"for {len(checker_t3_override)} PEP rule(s) (PEPs {peps})"
                        ),
                        "confidence": 0.80,
                        "tier": 3,
                        "oracle_overridden": True,
                        "oracle_hit": len(ov.findings_hit),
                        "oracle_missed": len(ov.findings_missed),
                    }
                    continue

            missed_details = [
                {"line": f.line, "rule": f.rule_id, "msg": f.message}
                for f in ov.findings_missed
            ]
            verdicts[checker] = {
                "verdict": ov.verdict,
                "reason": ov.reason,
                "confidence": ov.confidence,
                "tier": 0,
                "oracle_hit": len(ov.findings_hit),
                "oracle_missed": len(ov.findings_missed),
                "missed_findings": missed_details if missed_details else None,
            }
            continue

        # --- Tier 2: Hypothesis property-based testing ---
        if tier2_bugs_high:
            bugs_caught, bugs_missed = _check_bugs_against_checker(
                tier2_bugs_high, checker_error_lines, function_spans,
            )

            if bugs_caught and not bugs_missed:
                verdicts[checker] = {
                    "verdict": Verdict.CORRECT.value,
                    "reason": f"Caught {len(bugs_caught)} Hypothesis-proven bug(s)",
                    "confidence": 0.85,
                    "tier": 2,
                }
                continue
            elif bugs_missed and not bugs_caught:
                verdicts[checker] = {
                    "verdict": Verdict.INCORRECT.value,
                    "reason": f"Missed {len(bugs_missed)} Hypothesis-proven bug(s)",
                    "confidence": 0.85,
                    "tier": 2,
                    "missed_bugs": [{"line": b.line, "type": b.bug_type} for b in bugs_missed],
                }
                continue

        # --- Tier 3: PEP specification compliance ---
        # First, check findings tagged to this specific checker
        checker_t3 = [f for f in tier3_findings if f.get("checker") == checker]
        if not checker_t3:
            checker_t3 = [
                f for f in tier3_findings
                if "checker" not in f and f.get("checker_behavior") is not None
            ]
        if checker_t3:
            correct_count = sum(1 for f in checker_t3 if f.get("is_correct") is True)
            incorrect_count = sum(1 for f in checker_t3 if f.get("is_correct") is False)

            if correct_count > 0 and incorrect_count == 0:
                peps = sorted({f.get("pep", 0) for f in checker_t3 if f.get("is_correct")})
                verdicts[checker] = {
                    "verdict": Verdict.CORRECT.value,
                    "reason": f"Matches {correct_count} PEP rule(s) (PEPs {peps})",
                    "confidence": 0.80,
                    "tier": 3,
                }
                continue
            elif incorrect_count > 0 and correct_count == 0:
                peps = sorted({f.get("pep", 0) for f in checker_t3 if f.get("is_correct") is False})
                verdicts[checker] = {
                    "verdict": Verdict.INCORRECT.value,
                    "reason": f"Violates {incorrect_count} PEP rule(s) (PEPs {peps})",
                    "confidence": 0.80,
                    "tier": 3,
                }
                continue

        # If no checker-specific tier3 finding, check if ANY tier3 finding
        # established ground truth (a confirmed bug exists). If so, evaluate
        # this checker by whether it reported errors or not.
        if not checker_t3:
            any_confirmed_bug = [
                f for f in tier3_findings
                if f.get("is_correct") is True and f.get("correct_behavior") == "error"
            ]
            if not any_confirmed_bug:
                # Also check: any finding where a checker was correct AND reported error
                any_confirmed_bug = [
                    f for f in tier3_findings
                    if f.get("is_correct") is True
                    and f.get("checker_behavior") == "error"
                ]
            if any_confirmed_bug:
                peps = sorted({f.get("pep", 0) for f in any_confirmed_bug})
                if checker_reported_error:
                    verdicts[checker] = {
                        "verdict": Verdict.CORRECT.value,
                        "reason": f"Caught PEP-confirmed bug (PEPs {peps}, confirmed by other checker tier3 findings)",
                        "confidence": 0.80,
                        "tier": 3,
                    }
                    continue
                else:
                    verdicts[checker] = {
                        "verdict": Verdict.INCORRECT.value,
                        "reason": f"Missed PEP-confirmed bug (PEPs {peps}, confirmed by other checker tier3 findings)",
                        "confidence": 0.80,
                        "tier": 3,
                    }
                    continue

        # --- Tier 4: Static flow analysis ---
        checker_t4 = [f for f in tier4_findings if f.get("checker") == checker]
        if checker_t4:
            best = max(checker_t4, key=lambda f: f.get("confidence", 0))
            if best.get("verdict") in ("CORRECT", "INCORRECT") and best.get("confidence", 0) >= 0.80:
                verdicts[checker] = {
                    "verdict": best["verdict"],
                    "reason": best.get("reason", "Tier 4 static analysis"),
                    "confidence": best["confidence"],
                    "tier": 4,
                }
                continue

        # --- False positive detection ---
        if oracle_has_no_findings and not tier1_bugs_only and not tier2_bugs_high and checker_reported_error:
            if not _source_has_uncovered_constructs(source_code):
                try:
                    from .source_analysis import analyze_source as _analyze_source
                except ImportError:
                    from source_analysis import analyze_source as _analyze_source
                unfiltered = _analyze_source(source_code)
                if len(unfiltered) == 0:
                    verdicts[checker] = {
                        "verdict": Verdict.INCORRECT.value,
                        "reason": "Reported errors on violation-free source",
                        "confidence": 0.80,
                        "tier": 0,
                    }
                    continue

        verdicts[checker] = {
            "verdict": Verdict.UNCERTAIN.value,
            "reason": "No definitive evidence from any tier",
            "confidence": 0.5,
            "tier": 4,
        }

    return verdicts


def _get_function_at_line(spans: list[FunctionSpan], line: int) -> Optional[FunctionSpan]:
    """Find which function contains a line."""
    for span in spans:
        if span.start_line <= line <= span.end_line:
            return span
    return None


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_comprehensive(
    source_code: str,
    checker_outputs: dict[str, str],
    filename: str = "unknown.py",
    debug: DebugArtifactCollector | None = None,
    debug_dir: str | None = None,
) -> EvaluationResult:
    """
    Run comprehensive tiered evaluation on a code example.

    All tiers always run and contribute evidence to the final verdict:
      Tier 1: Runtime crash detection (highest confidence)
      Tier 2: Hypothesis property-based testing (high confidence)
      Tier 3: PEP specification compliance (medium-high confidence)
    """
    try:
        from .hypothesis_tier2 import run_hypothesis_tier2
    except ImportError:
        from hypothesis_tier2 import run_hypothesis_tier2
    
    try:
        from .targeted_tests import run_targeted_tests
    except ImportError:
        from targeted_tests import run_targeted_tests

    try:
        from .static_tier4 import run_tier4
    except ImportError:
        from static_tier4 import run_tier4

    oracle_verdicts = run_oracle_evaluation(source_code, checker_outputs)

    tier1_bugs = run_tier1(source_code, debug=debug)

    hypothesis_output_dir = None
    if debug_dir:
        hypothesis_output_dir = os.path.join(debug_dir, filename.replace(".py", ""))

    tier2_bugs = run_hypothesis_tier2(
        source_code,
        checker_outputs=checker_outputs,
        output_dir=hypothesis_output_dir,
    )

    targeted_bugs = run_targeted_tests(
        source_code,
        output_dir=hypothesis_output_dir,
        filename=filename,
    )
    tier2_bugs = tier2_bugs + targeted_bugs

    tier3_findings = run_tier3(source_code, checker_outputs)
    tier4_findings = run_tier4(source_code, checker_outputs)

    tier_reached = 0
    has_oracle_findings = any(
        ov.verdict != "UNCERTAIN" or ov.findings_hit or ov.findings_missed
        for ov in oracle_verdicts.values()
    )
    if not has_oracle_findings:
        tier_reached = 1
    if not has_oracle_findings and not tier1_bugs:
        tier_reached = 2
    if not has_oracle_findings and not tier1_bugs and not tier2_bugs:
        tier_reached = 3
    if not has_oracle_findings and not tier1_bugs and not tier2_bugs and not tier3_findings:
        tier_reached = 4

    verdicts = determine_verdicts(
        tier1_bugs, tier2_bugs, tier3_findings,
        checker_outputs, source_code,
        tier4_findings=tier4_findings,
        oracle_verdicts=oracle_verdicts,
    )

    return EvaluationResult(
        filename=filename,
        tier1_bugs=tier1_bugs,
        tier2_bugs=tier2_bugs,
        tier3_findings=tier3_findings + tier4_findings,
        checker_verdicts=verdicts,
        tier_reached=tier_reached,
        oracle_verdicts=oracle_verdicts,
    )


def _call_gemini_agent(
    source_code: str,
    filename: str,
    checker_name: str,
    checker_output: str,
    other_outputs: dict[str, str],
) -> dict:
    """Call Google Gemini API to resolve an uncertain verdict."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {
            "verdict": "UNCERTAIN",
            "reason": "GEMINI_API_KEY not set",
            "pep_citation": None,
            "confidence": 0.0,
        }

    others_text = ""
    for name, out in other_outputs.items():
        others_text += f"\n--- {name} ---\n{out}\n"

    prompt = (
        f"Source file: {filename}\n"
        f"Source code:\n```python\n{source_code}\n```\n\n"
        f"Checker: {checker_name}\n"
        f"Checker output:\n```\n{checker_output}\n```\n\n"
        f"Other checkers' outputs:\n{others_text}\n\n"
        "Determine whether this type checker's behavior is CORRECT, INCORRECT, or UNCERTAIN.\n"
        "Cite the specific PEP section that supports your verdict.\n"
        "If you cannot determine a definitive answer, return UNCERTAIN with explanation.\n\n"
        'Respond in JSON: {"verdict": "CORRECT"|"INCORRECT"|"UNCERTAIN", '
        '"reason": "...", "pep_citation": "..."|null, "confidence": 0.0-1.0}'
    )

    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    try:
        resp = httpx.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()

        try:
            candidate = data.get("candidates", [{}])[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [{}])
            text = parts[0].get("text", "")
        except (IndexError, AttributeError):
            text = ""

        if not text:
            return {
                "verdict": "UNCERTAIN",
                "reason": f"Empty Gemini response: {data}",
                "pep_citation": None,
                "confidence": 0.0,
            }

        import re as _re
        m = _re.search(r'\{[^}]*"verdict"\s*:', text)
        if m:
            json_str = text[m.start():]
            brace_count = 0
            end = 0
            for i, ch in enumerate(json_str):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            if end:
                parsed = json.loads(json_str[:end])
                verdict = parsed.get("verdict", "UNCERTAIN").upper()
                if verdict not in ("CORRECT", "INCORRECT", "UNCERTAIN"):
                    verdict = "UNCERTAIN"
                return {
                    "verdict": verdict,
                    "reason": parsed.get("reason", ""),
                    "pep_citation": parsed.get("pep_citation"),
                    "confidence": float(parsed.get("confidence", 0.5)),
                }

        prose_match = _re.search(
            r"\b(?:is|behavior is|behaviour is|verdict[:\s]+)\s*\*{0,2}"
            r"(CORRECT|INCORRECT|UNCERTAIN)\*{0,2}",
            text,
            _re.IGNORECASE,
        )
        if prose_match:
            verdict = prose_match.group(1).upper()
            reason = text[:500].replace("\n", " ").strip()
            return {
                "verdict": verdict,
                "reason": reason,
                "pep_citation": None,
                "confidence": 0.7,
            }

        return {
            "verdict": "UNCERTAIN",
            "reason": f"Could not parse agent response: {text[:200]}",
            "pep_citation": None,
            "confidence": 0.0,
        }
    except Exception as e:
        return {
            "verdict": "UNCERTAIN",
            "reason": f"Agent call failed: {str(e)[:200]}",
            "pep_citation": None,
            "confidence": 0.0,
        }


def _resolve_uncertain_via_agent(
    uncertain_cases: list[dict],
    file_entries: list[dict],
    checkers: list[str],
) -> list[dict]:
    """Resolve uncertain verdicts by calling an LLM agent via the Gemini API."""
    agent_verdicts: list[dict] = []
    file_data: dict[str, dict] = {}
    for entry in file_entries:
        fn = entry.get("filename", "")
        file_data[fn] = entry

    print(f"\nResolving {len(uncertain_cases)} uncertain case(s) via agent...")

    for i, case in enumerate(uncertain_cases, 1):
        filename = case["filename"]
        checker = case["checker"]
        result = case["result"]

        entry = file_data.get(filename, {})
        filepath = entry.get("filepath", "")
        outputs = entry.get("outputs", {})

        source_code = entry.get("source_code", "")
        if not source_code:
            if not filepath:
                print(f"  WARNING: no filepath in results for {filename}, skipping agent resolution")
                agent_verdicts.append({
                    "filename": filename,
                    "checker": checker,
                    "verdict": "UNCERTAIN",
                    "reason": "No source code or filepath available for agent resolution",
                    "pep_citation": None,
                    "confidence": 0.0,
                })
                continue
            try:
                with open(filepath) as f:
                    source_code = f.read()
            except (FileNotFoundError, KeyError):
                print(f"  WARNING: source file not found for {filename}, skipping agent resolution")
                agent_verdicts.append({
                    "filename": filename,
                    "checker": checker,
                    "verdict": "UNCERTAIN",
                    "reason": "Source file not found for agent resolution",
                    "pep_citation": None,
                    "confidence": 0.0,
                })
                continue

        checker_output = outputs.get(checker, "")
        other_outputs = {k: v for k, v in outputs.items() if k != checker}

        print(f"  [{i}/{len(uncertain_cases)}] {filename} / {checker}...", end=" ", flush=True)
        av = _call_gemini_agent(source_code, filename, checker, checker_output, other_outputs)
        av["filename"] = filename
        av["checker"] = checker
        agent_verdicts.append(av)
        print(f"{av['verdict']}")

    return agent_verdicts

def evaluate_results_comprehensive(
    results_path: str,
    save_tests_dir: str | None = None,
) -> dict:
    """
    Evaluate all files using the comprehensive tiered system.
    
    Args:
        results_path: Path to results.json from the pipeline.
        save_tests_dir: If set, save ephemeral Tier 1/2 test snippets to this directory.
                        If None, automatically saves to a 'tests/' directory next to results.json.
    """
    if save_tests_dir is None:
        save_tests_dir = os.path.join(os.path.dirname(results_path), "tests")
    os.makedirs(save_tests_dir, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)
    
    results = data.get("results", [])
    checkers = data.get("checkers_used", ["mypy", "pyrefly", "zuban", "ty"])
    
    all_results = []
    summary_stats = {
        checker: {"correct": 0, "incorrect": 0, "uncertain": 0}
        for checker in checkers
    }
    tier_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    print("=" * 70)
    print("COMPREHENSIVE TIERED EVALUATION")
    print("=" * 70)
    print("Tier 0: Oracle (AST-based ground truth)")
    print("Tier 1: Runtime crash detection")
    print("Tier 2: Hypothesis property-based testing")
    print("Tier 3: PEP specification compliance")
    print("Tier 4: Design differences (uncertain)")
    print(f"Files to evaluate: {len(results)}")
    print("=" * 70)
    print()
    
    for i, file_entry in enumerate(results, 1):
        filepath = file_entry.get("filepath", "")
        filename = file_entry.get("filename", "")
        outputs = file_entry.get("outputs", {})
        
        print(f"[{i}/{len(results)}] {filename}")
        
        try:
            with open(filepath) as f:
                source_code = f.read()
        except FileNotFoundError:
            print("  [SKIP] File not found")
            continue
        
        file_metrics = metrics_to_dict(compute_metrics(source_code))
        file_entry["metrics"] = file_metrics

        collector = DebugArtifactCollector()
        result = evaluate_comprehensive(
            source_code, outputs, filename,
            debug=collector, debug_dir=save_tests_dir,
        )
        all_results.append((result, file_metrics))
        tier_distribution[result.tier_reached] += 1
        collector.save(save_tests_dir, filename)
        
        # Print summary
        print(f"  Tier reached: {result.tier_reached}")
        print(f"  Bugs: T1={len(result.tier1_bugs)}, T2={len(result.tier2_bugs)}, T3={len(result.tier3_findings)}")
        
        for checker, verdict in result.checker_verdicts.items():
            v = verdict["verdict"]
            tier = verdict.get("tier", "?")
            if v == "CORRECT":
                print(f"  ✓ {checker}: CORRECT (tier {tier})")
                summary_stats[checker]["correct"] += 1
            elif v == "INCORRECT":
                print(f"  ✗ {checker}: INCORRECT (tier {tier})")
                summary_stats[checker]["incorrect"] += 1
            else:
                print(f"  ? {checker}: UNCERTAIN")
                summary_stats[checker]["uncertain"] += 1
        
        print()
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTier distribution:")
    for tier, count in tier_distribution.items():
        print(f"  Tier {tier}: {count} files")
    
    print(f"\n{'Checker':<12} {'Correct':>10} {'Incorrect':>10} {'Uncertain':>10}")
    print("-" * 44)
    
    for checker in checkers:
        stats = summary_stats[checker]
        print(f"{checker:<12} {stats['correct']:>10} {stats['incorrect']:>10} {stats['uncertain']:>10}")
    
    print("=" * 70)
    
    # Collect uncertain cases for agent resolution
    uncertain_cases: list[dict] = []
    for result, _ in all_results:
        for checker, verdict in result.checker_verdicts.items():
            if verdict["verdict"] == "UNCERTAIN":
                uncertain_cases.append({
                    "filename": result.filename,
                    "checker": checker,
                    "result": result,
                })

    agent_verdicts: list[dict] = []
    agent_stats = {
        checker: {"correct": 0, "incorrect": 0, "uncertain": 0}
        for checker in checkers
    }

    if uncertain_cases:
        agent_verdicts = _resolve_uncertain_via_agent(uncertain_cases, results, checkers)
        for av in agent_verdicts:
            v = av.get("verdict", "UNCERTAIN")
            c = av.get("checker", "")
            if c in agent_stats:
                if v == "CORRECT":
                    agent_stats[c]["correct"] += 1
                elif v == "INCORRECT":
                    agent_stats[c]["incorrect"] += 1
                else:
                    agent_stats[c]["uncertain"] += 1

        print()
        print("=" * 70)
        print("AGENT-RESOLVED UNCERTAIN CASES")
        print("=" * 70)
        print(f"\n{'Checker':<12} {'Correct':>10} {'Incorrect':>10} {'Uncertain':>10}")
        print("-" * 44)
        for checker in checkers:
            s = agent_stats[checker]
            print(f"{checker:<12} {s['correct']:>10} {s['incorrect']:>10} {s['uncertain']:>10}")
        print("=" * 70)

    # Write metrics back into results.json
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save results
    output_dir = os.path.dirname(results_path)
    eval_path = os.path.join(output_dir, "evaluation_comprehensive.json")

    with open(eval_path, "w") as f:
        json.dump({
            "method": "comprehensive_tiered",
            "tier_distribution": tier_distribution,
            "summary": summary_stats,
            "oracle_verdicts": [
                {
                    "filename": r.filename,
                    "metrics": m,
                    "tier_reached": r.tier_reached,
                    "tier1_bugs": [{"line": b.line, "type": b.bug_type, "msg": b.message} for b in r.tier1_bugs],
                    "tier2_bugs": [{"line": b.line, "type": b.bug_type, "msg": b.message} for b in r.tier2_bugs],
                    "tier3_findings": r.tier3_findings,
                    "verdicts": r.checker_verdicts,
                    "oracle": {
                        checker: {
                            "verdict": ov.verdict,
                            "reason": ov.reason,
                            "confidence": ov.confidence,
                            "findings_hit": len(ov.findings_hit),
                            "findings_missed": len(ov.findings_missed),
                        }
                        for checker, ov in r.oracle_verdicts.items()
                    } if r.oracle_verdicts else None,
                }
                for r, m in all_results
            ],
            "agent_verdicts": agent_verdicts,
        }, f, indent=2)

    print(f"\nResults saved to: {eval_path}")
    print(f"Generated tests saved to: {save_tests_dir}/")

    return summary_stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_eval.py <results.json> [--save-tests <dir>]")
        print()
        print("Comprehensive tiered evaluation system:")
        print("  Tier 1: Runtime crash detection (highest confidence)")
        print("  Tier 2: Mutation + Typeguard testing")
        print("  Tier 3: PEP specification compliance")
        print("  Tier 4: Design differences (uncertain)")
        print()
        print("Options:")
        print("  --save-tests <dir>  Save ephemeral test snippets for debugging")
        sys.exit(1)
    
    save_tests = None
    if "--save-tests" in sys.argv:
        idx = sys.argv.index("--save-tests")
        if idx + 1 < len(sys.argv):
            save_tests = sys.argv[idx + 1]
        else:
            print("Error: --save-tests requires a directory argument")
            sys.exit(1)
    
    evaluate_results_comprehensive(sys.argv[1], save_tests_dir=save_tests)
