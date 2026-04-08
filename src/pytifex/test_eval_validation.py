"""
Validation test for the comprehensive evaluation pipeline.

Creates source files with KNOWN type bugs, runs the evaluation,
and checks that verdicts match expected ground truth.
"""

import json
import os
import tempfile
from comprehensive_eval import evaluate_comprehensive, Verdict


# TEST FILES — each has documented known bugs

# File 1: Runtime crash (Phase 1) — TypeError at runtime
FILE_RUNTIME_CRASH = '''
from typing import Protocol

class Addable(Protocol):
    def __add__(self, other: int) -> int: ...

def add_values(a: Addable, b: int) -> int:
    return a + b

if __name__ == "__main__":
    result = add_values("hello", 5)  # str + int works, but...
    result2 = add_values(None, 5)    # TypeError: None has no __add__
'''
# Known: TypeError at runtime on None + 5. A checker that reports no errors
# is INCORRECT because this crashes.

# File 2: Obvious assignment mismatch (Phase 3 source analysis)
FILE_ASSIGN_MISMATCH = '''
from typing import Final

x: int = "hello"
y: str = 42
z: Final[int] = 10
z = 20
'''
# Known bugs:
#   - line 4: str assigned to int annotation (ASSIGN001)
#   - line 5: int assigned to str annotation (ASSIGN001)
#   - line 7: reassignment to Final variable (FINAL001)
# A checker that reports no errors is INCORRECT.

# File 3: Protocol misuse (Phase 3 source analysis)
FILE_PROTOCOL_MISUSE = '''
from typing import Protocol, runtime_checkable

class Drawable(Protocol):
    def draw(self) -> None: ...

class NotAProtocol:
    def draw(self) -> None:
        print("drawing")

d = Drawable()  # PROTO003: cannot instantiate Protocol
isinstance(NotAProtocol(), Drawable)  # PROTO001: not @runtime_checkable
'''
# Known bugs:
#   - line 10: Protocol instantiation (PROTO003)
#   - line 11: isinstance with non-runtime_checkable Protocol (PROTO001)

# File 4: No bugs — clean code
FILE_CLEAN = '''
from typing import TypedDict

class Point(TypedDict):
    x: int
    y: int

def make_point(x: int, y: int) -> Point:
    return {"x": x, "y": y}

if __name__ == "__main__":
    p = make_point(1, 2)
    print(p["x"], p["y"])
'''
# Known: No type bugs. All checkers should either be CORRECT or UNCERTAIN.
# A checker that reports errors here would be a false positive.

# File 5: Override and Final violations
FILE_OVERRIDE_FINAL = '''
from typing import final, override

class Base:
    @final
    def locked(self) -> int:
        return 1

    def flexible(self) -> int:
        return 2

class Child(Base):
    def locked(self) -> int:  # FINAL003: overrides @final method
        return 99

    @override
    def nonexistent(self) -> int:  # OVERRIDE001: no base method
        return 0
'''
# Known bugs:
#   - line 13: overrides @final method (FINAL003)
#   - line 17: @override but no parent method (OVERRIDE001)

# File 6: NoReturn that returns
FILE_NORETURN_RETURNS = '''
from typing import NoReturn

def should_never_return() -> NoReturn:
    return 42  # NORETURN001: returns from NoReturn function

def actually_never_returns() -> NoReturn:
    raise RuntimeError("boom")
'''
# Known bugs:
#   - line 4: return statement in NoReturn function (NORETURN001)
#   - line 7 is fine (raises, never returns)


def make_checker_output(
    has_errors: bool,
    error_lines: list[int] | None = None,
    error_codes: list[str] | None = None,
) -> str:
    if not has_errors:
        return "Success: no issues found in 1 source file"
    lines = []
    for i, ln in enumerate(error_lines or []):
        code = error_codes[i] if error_codes and i < len(error_codes) else "assignment"
        lines.append(f"test.py:{ln}:1: error: Type error detected  [{code}]")
    lines.append(f"Found {len(lines)} error(s) in 1 source file")
    return "\n".join(lines)


def run_test(name: str, source: str, checker_configs: dict[str, dict], expected: dict[str, str]):
    """
    Run evaluation on a source file with synthetic checker outputs and
    compare verdicts against expected ground truth.

    checker_configs: {checker_name: {"has_errors": bool, "error_lines": [int, ...], "error_codes": [str, ...]}}
    expected: {checker_name: "CORRECT" | "INCORRECT" | "UNCERTAIN"}
    """
    checker_outputs = {}
    for checker, config in checker_configs.items():
        checker_outputs[checker] = make_checker_output(
            config["has_errors"],
            config.get("error_lines"),
            config.get("error_codes"),
        )

    result = evaluate_comprehensive(source, checker_outputs, f"{name}.py")

    passed = True
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Tier reached: {result.tier_reached}")
    print(f"  T1 bugs: {len(result.tier1_bugs)}, T2 bugs: {len(result.tier2_bugs)}, T3 findings: {len(result.tier3_findings)}")

    for checker, verdict_info in result.checker_verdicts.items():
        actual = verdict_info["verdict"]
        exp = expected.get(checker, "ANY")
        match = "PASS" if (exp == "ANY" or actual == exp) else "FAIL"
        if match == "FAIL":
            passed = False
        print(f"  {match}: {checker}: got {actual}, expected {exp}")
        print(f"        reason: {verdict_info['reason']}")

    return passed


def main():
    all_passed = True

    # Test 1: Runtime crash — checker that says OK is INCORRECT
    ok = run_test(
        "runtime_crash",
        FILE_RUNTIME_CRASH,
        {
            "checker_catches": {"has_errors": True, "error_lines": [11]},
            "checker_misses": {"has_errors": False},
        },
        {
            "checker_catches": "CORRECT",
            "checker_misses": "INCORRECT",
        },
    )
    all_passed &= ok

    # Test 2: Assignment mismatch — checker that says OK is INCORRECT
    ok = run_test(
        "assign_mismatch",
        FILE_ASSIGN_MISMATCH,
        {
            "checker_catches": {"has_errors": True, "error_lines": [4, 5, 7], "error_codes": ["assignment", "assignment", "assignment"]},
            "checker_misses": {"has_errors": False},
            "checker_wrong_lines": {"has_errors": True, "error_lines": [99], "error_codes": ["assignment"]},
        },
        {
            "checker_catches": "CORRECT",
            "checker_misses": "INCORRECT",
            "checker_wrong_lines": "INCORRECT",
        },
    )
    all_passed &= ok

    # Test 3: Protocol misuse
    ok = run_test(
        "protocol_misuse",
        FILE_PROTOCOL_MISUSE,
        {
            "checker_catches": {"has_errors": True, "error_lines": [10, 11], "error_codes": ["abstract", "type-var"]},
            "checker_misses": {"has_errors": False},
        },
        {
            "checker_catches": "CORRECT",
            "checker_misses": "INCORRECT",
        },
    )
    all_passed &= ok

    # Test 4: Clean code — no bugs, checker reporting errors is wrong
    ok = run_test(
        "clean_code",
        FILE_CLEAN,
        {
            "checker_ok": {"has_errors": False},
            "checker_false_positive": {"has_errors": True, "error_lines": [8]},
        },
        {
            "checker_ok": "UNCERTAIN",
            "checker_false_positive": "INCORRECT",
        },
    )
    all_passed &= ok

    # Test 5: Override and Final violations
    ok = run_test(
        "override_final",
        FILE_OVERRIDE_FINAL,
        {
            "checker_catches": {"has_errors": True, "error_lines": [13, 17], "error_codes": ["override", "override"]},
            "checker_misses": {"has_errors": False},
        },
        {
            "checker_catches": "CORRECT",
            "checker_misses": "INCORRECT",
        },
    )
    all_passed &= ok

    # Test 6: NoReturn that returns
    ok = run_test(
        "noreturn_returns",
        FILE_NORETURN_RETURNS,
        {
            "checker_catches": {"has_errors": True, "error_lines": [4], "error_codes": ["return"]},
            "checker_misses": {"has_errors": False},
        },
        {
            "checker_catches": "CORRECT",
            "checker_misses": "INCORRECT",
        },
    )
    all_passed &= ok

    # Test 7: Errors on wrong lines (the verdict gap fix)
    ok = run_test(
        "wrong_lines_tier2",
        FILE_ASSIGN_MISMATCH,
        {
            "checker_near": {"has_errors": True, "error_lines": [4], "error_codes": ["assignment"]},
            "checker_far": {"has_errors": True, "error_lines": [50, 60], "error_codes": ["assignment", "assignment"]},
        },
        {
            "checker_near": "CORRECT",
            "checker_far": "INCORRECT",
        },
    )
    all_passed &= ok

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

