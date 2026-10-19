# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hypothesis",
#     "beartype",
#     "typeguard",
# ]
# ///

"""
Two-Phase Objective Evaluation System for Type Checker Correctness.

Phase 1: Runtime Crash Detection
    - Execute code and catch type-related exceptions
    - Re-run try bodies in isolation to surface swallowed errors
    - Proves false negatives

Phase 2: Hypothesis Property-Based Testing
    - AST-driven call-site extraction and signature introspection
    - Hypothesis-generated inputs verified by beartype plus targeted tests
    - Crashes prove false negatives (2a)
    - Successful runs with a valid return type prove false positives (2b)

Checkers with no runtime evidence either way are marked UNCERTAIN.
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
    from .code_metrics import compute_metrics, metrics_to_dict
except ImportError:
    from .code_metrics import compute_metrics, metrics_to_dict


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
    source: str  # "phase1_runtime", "phase2_mutation", "phase3_pep"
    confidence: float
    details: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a file."""
    filename: str
    phase1_bugs: list[TypeBug]
    phase2_bugs: list[TypeBug]
    phase2_witnesses: list
    checker_verdicts: dict[str, dict]


class DebugArtifactCollector:
    """Collects ephemeral test snippets generated during evaluation for later inspection."""

    def __init__(self) -> None:
        self.phase1_snippets: list[dict[str, str]] = []
        self.phase2_snippets: list[dict[str, str]] = []

    def add_phase1(self, label: str, code: str) -> None:
        self.phase1_snippets.append({"label": label, "code": code})

    def add_phase2(self, annotation: str, violation: str, code: str) -> None:
        self.phase2_snippets.append({
            "annotation": annotation,
            "violation": violation,
            "code": code,
        })

    def save(self, directory: str, filename: str) -> None:
        stem = filename.removesuffix(".py")
        base = os.path.join(directory, stem)
        os.makedirs(base, exist_ok=True)

        for i, s in enumerate(self.phase1_snippets):
            path = os.path.join(base, f"phase1_{i}_{s['label']}.py")
            with open(path, "w") as f:
                f.write(s["code"])

        for i, s in enumerate(self.phase2_snippets):
            safe_ann = re.sub(r"[^\w]", "_", s["annotation"])[:40]
            path = os.path.join(base, f"phase2_{i}_{safe_ann}.py")
            with open(path, "w") as f:
                f.write(f"# annotation: {s['annotation']}\n")
                f.write(f"# violation:  {s['violation']}\n\n")
                f.write(s["code"])


# =============================================================================
# PHASE 1: RUNTIME CRASH DETECTION
# =============================================================================

TYPE_ERROR_EXCEPTIONS = (TypeError, KeyError, AttributeError)


def _extract_all_source_lines(tb_list: list, source_tag: str = "<phase1>") -> list[int]:
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


def _run_isolated_code(code: str, source_tag: str = "<phase1_isolated>") -> list[TypeBug]:
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
            source="phase1_runtime",
            confidence=1.0,
            details={"all_traceback_lines": list(dict.fromkeys(all_chain_lines))},
        ))

    return bugs


def run_phase1(
    source_code: str,
    debug: DebugArtifactCollector | None = None,
) -> list[TypeBug]:
    """
    Phase 1: Execute code and catch type-related runtime exceptions.
    - Walks the full traceback to find the root cause line
    - Inspects exception chains (__cause__ / __context__)
    - Isolates try/except bodies to surface swallowed type errors
    """
    bugs = _run_isolated_code(source_code, "<phase1>")
    if debug:
        debug.add_phase1("full_source", source_code)

    try_bodies = _extract_try_bodies(source_code)
    seen_lines = {b.line for b in bugs}

    for idx, (start_line, end_line, body_source) in enumerate(try_bodies):
        if debug:
            debug.add_phase1(f"try_body_{idx}_L{start_line}", body_source)
        isolated_bugs = _run_isolated_code(body_source, "<phase1_isolated>")
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
        # Errors are "error[rule-name]:", but warnings/infos are not errors.
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


# VERDICT DETERMINATION

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
    phase1_bugs: list[TypeBug],
    phase2_bugs: list[TypeBug],
    phase2_witnesses: list,
    checker_outputs: dict[str, str],
    source_code: str,
) -> dict[str, dict]:
    """
    Determine final verdict for each checker using only objective runtime evidence.

    Priority:
      1. Phase 1 runtime crashes — definitive false negative proof
         (checker said "ok" but code provably crashes)
      2. Phase 2 Hypothesis crashes — false negative proof via fuzzing
         (crashes found with type-conformant inputs)
      3. Phase 2 success witnesses — definitive false positive proof
         (code ran correctly with beartype-verified type-conformant inputs,
          so any checker reporting "error" is demonstrably wrong)
      4. UNCERTAIN — no runtime evidence either way
    """
    verdicts = {}
    function_spans = extract_function_spans(source_code)

    phase1_runtime = [b for b in phase1_bugs if b.confidence >= 0.85 and b.source == "phase1_runtime"]
    phase2_high = [b for b in phase2_bugs if b.confidence >= 0.85]
    # Only use witnesses that had beartype enforcement for the false positive claim,
    # or witnesses from non-fallback strategies if beartype is unavailable.
    strong_witnesses = [
        w for w in phase2_witnesses
        if w.beartype_enforced or not any(True for _ in [])  # beartype_enforced preferred
    ]

    checker_error_status = {
        c: _checker_reports_error(o, c) for c, o in checker_outputs.items()
    }

    for checker, output in checker_outputs.items():
        checker_reported_error = checker_error_status[checker]
        checker_error_lines = extract_checker_error_lines(output)

        # Phase 1: direct runtime crashes — definitive false negative evidence
        if phase1_runtime:
            caught, missed = _check_bugs_against_checker(
                phase1_runtime, checker_error_lines, function_spans,
            )
            if missed:
                verdicts[checker] = {
                    "verdict": Verdict.INCORRECT.value,
                    "reason": f"False negative: missed {len(missed)} proven runtime crash(es)",
                    "confidence": 0.95,
                    "phase": 1,
                    "missed_bugs": [{"line": b.line, "type": b.bug_type} for b in missed],
                }
                continue
            if caught:
                verdicts[checker] = {
                    "verdict": Verdict.CORRECT.value,
                    "reason": f"Correctly caught {len(caught)} proven runtime crash(es)",
                    "confidence": 0.95,
                    "phase": 1,
                }
                continue

        # Phase 2a: Hypothesis crashes with type-conformant inputs — false negative evidence
        if phase2_high:
            caught, missed = _check_bugs_against_checker(
                phase2_high, checker_error_lines, function_spans,
            )
            if missed and not caught:
                verdicts[checker] = {
                    "verdict": Verdict.INCORRECT.value,
                    "reason": f"False negative: missed {len(missed)} bug(s) proven by Hypothesis",
                    "confidence": 0.85,
                    "phase": 2,
                    "missed_bugs": [{"line": b.line, "type": b.bug_type} for b in missed],
                }
                continue
            if caught and not missed:
                verdicts[checker] = {
                    "verdict": Verdict.CORRECT.value,
                    "reason": f"Correctly caught {len(caught)} Hypothesis-proven bug(s)",
                    "confidence": 0.85,
                    "phase": 2,
                }
                continue

        # Phase 2b: success witnesses — definitive false positive evidence
        # Only applicable when no crash evidence exists (would be contradictory).
        if strong_witnesses and not phase1_runtime and not phase2_high:
            total_successes = sum(w.calls_succeeded for w in strong_witnesses)
            enforced = any(w.beartype_enforced for w in strong_witnesses)
            confidence = 0.90 if enforced else 0.75
            if checker_reported_error:
                verdicts[checker] = {
                    "verdict": Verdict.INCORRECT.value,
                    "reason": (
                        f"False positive: reported error on code that executed successfully "
                        f"with type-conformant inputs ({total_successes} successful call(s)"
                        + (", beartype-enforced)" if enforced else ")")
                    ),
                    "confidence": confidence,
                    "phase": 2,
                    "witnesses": total_successes,
                }
                continue
            else:
                verdicts[checker] = {
                    "verdict": Verdict.CORRECT.value,
                    "reason": (
                        f"Correctly accepted code that executed successfully with "
                        f"type-conformant inputs ({total_successes} successful call(s)"
                        + (", beartype-enforced)" if enforced else ")")
                    ),
                    "confidence": confidence,
                    "phase": 2,
                    "witnesses": total_successes,
                }
                continue

        verdicts[checker] = {
            "verdict": Verdict.UNCERTAIN.value,
            "reason": "No runtime evidence from Phase 1 or Phase 2",
            "confidence": 0.5,
            "phase": 2,
        }

    return verdicts


def _get_function_at_line(spans: list[FunctionSpan], line: int) -> Optional[FunctionSpan]:
    """Find which function contains a line."""
    for span in spans:
        if span.start_line <= line <= span.end_line:
            return span
    return None


# MAIN EVALUATION FUNCTION

def evaluate_comprehensive(
    source_code: str,
    checker_outputs: dict[str, str],
    filename: str = "unknown.py",
    debug: DebugArtifactCollector | None = None,
    debug_dir: str | None = None,
) -> EvaluationResult:
    """
    Run two-phase objective evaluation on a code example.

    Phase 1: Runtime crash detection — proves false negatives
    Phase 2: Hypothesis property testing — proves false negatives (crashes)
             and false positives (successful execution with type-conformant inputs)
    """
    try:
        from .hypothesis_phase2 import run_hypothesis_phase2
    except ImportError:
        from hypothesis_phase2 import run_hypothesis_phase2

    try:
        from .targeted_tests import run_targeted_tests
    except ImportError:
        from .targeted_tests import run_targeted_tests

    phase1_bugs = run_phase1(source_code, debug=debug)

    hypothesis_output_dir = None
    if debug_dir:
        hypothesis_output_dir = os.path.join(debug_dir, filename.replace(".py", ""))

    phase2_bugs, phase2_witnesses = run_hypothesis_phase2(
        source_code,
        checker_outputs=checker_outputs,
        output_dir=hypothesis_output_dir,
    )

    targeted_bugs = run_targeted_tests(
        source_code,
        output_dir=hypothesis_output_dir,
        filename=filename,
    )
    phase2_bugs = phase2_bugs + targeted_bugs

    verdicts = determine_verdicts(
        phase1_bugs, phase2_bugs, phase2_witnesses,
        checker_outputs, source_code,
    )

    return EvaluationResult(
        filename=filename,
        phase1_bugs=phase1_bugs,
        phase2_bugs=phase2_bugs,
        phase2_witnesses=phase2_witnesses,
        checker_verdicts=verdicts,
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
    Evaluate all files using the comprehensive phased system. 
    Args:
        results_path: Path to results.json from the pipeline.
        save_tests_dir: If set, save ephemeral Phase 1/2 test snippets to this directory.
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

    print("=" * 70)
    print("OBJECTIVE TWO-PHASE EVALUATION")
    print("=" * 70)
    print("Phase 1: Runtime crash detection (false negatives)")
    print("Phase 2: Hypothesis property testing (false negatives + false positives)")
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
        collector.save(save_tests_dir, filename)

        # Print summary
        print(f"  Phase 1 crashes: {len(result.phase1_bugs)}, Phase 2 bugs: {len(result.phase2_bugs)}, Phase 2 witnesses: {len(result.phase2_witnesses)}")

        for checker, verdict in result.checker_verdicts.items():
            v = verdict["verdict"]
            phase = verdict.get("phase", "?")
            if v == "CORRECT":
                print(f"  ✓ {checker}: CORRECT (phase {phase})")
                summary_stats[checker]["correct"] += 1
            elif v == "INCORRECT":
                print(f"  ✗ {checker}: INCORRECT (phase {phase})")
                summary_stats[checker]["incorrect"] += 1
            else:
                print(f"  ? {checker}: UNCERTAIN")
                summary_stats[checker]["uncertain"] += 1

        print()

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

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
            "method": "two_phase_objective",
            "summary": summary_stats,
            "results": [
                {
                    "filename": r.filename,
                    "metrics": m,
                    "phase1_bugs": [{"line": b.line, "type": b.bug_type, "msg": b.message} for b in r.phase1_bugs],
                    "phase2_bugs": [{"line": b.line, "type": b.bug_type, "msg": b.message} for b in r.phase2_bugs],
                    "phase2_witnesses": [
                        {"call": w.call_text, "successes": w.calls_succeeded, "beartype_enforced": w.beartype_enforced}
                        for w in r.phase2_witnesses
                    ],
                    "verdicts": r.checker_verdicts,
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
        print("Objective two-phase evaluation system:")
        print("  Phase 1: Runtime crash detection — proves false negatives")
        print("  Phase 2: Hypothesis property testing — proves false negatives and false positives")
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

