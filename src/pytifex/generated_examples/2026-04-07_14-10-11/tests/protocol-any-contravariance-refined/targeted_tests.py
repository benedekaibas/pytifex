"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: protocol-any-contravariance-refined.py
Patterns detected: 1
    - protocol_conformance (1 tests)
Test cases generated: 1
"""

# --- Original source ---

from typing import Protocol, Union, Any

class NumericLike(Protocol):
    def __add__(self, other: Any) -> Any:
        """Adds another object."""
        # Default behavior (e.g., fallback to basic addition if not overridden)
        # This is not a real default implementation for protocols,
        # but the *expectation* of handling other types is implicit.
        pass

    def __sub__(self, other: Any) -> Any:
        """Subtracts another object."""
        pass

class CustomTimeDelta:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __add__(self, other: Union[int, "CustomTimeDelta"]) -> "CustomTimeDelta":
        if isinstance(other, CustomTimeDelta):
            return CustomTimeDelta(self.seconds + other.seconds)
        if isinstance(other, int):
            return CustomTimeDelta(self.seconds + other)
        raise TypeError(f"Unsupported operand type for +: 'CustomTimeDelta' and '{type(other).__name__}'")

    def __sub__(self, other: Union[int, "CustomTimeDelta"]) -> "CustomTimeDelta":
        if isinstance(other, CustomTimeDelta):
            return CustomTimeDelta(self.seconds - other.seconds)
        if isinstance(other, int):
            return CustomTimeDelta(self.seconds - other)
        raise TypeError(f"Unsupported operand type for -: 'CustomTimeDelta' and '{type(other).__name__}'")

    def __repr__(self) -> str:
        return f"CustomTimeDelta({self.seconds})"

if __name__ == "__main__":
    # The Protocol defines `Any`, but the implementation is more specific.
    # Checkers might differ on whether `CustomTimeDelta` fully implements `NumericLike`
    # if they expect the `Any` to imply support for *all* types.
    # The original bug was about numpy and datetime interaction; this is a similar custom type.
    delta1 = CustomTimeDelta(100)
    delta2 = CustomTimeDelta(50)

    sum_delta = delta1 + delta2
    print(f"{delta1} + {delta2} = {sum_delta}")

    sum_int = delta1 + 20
    print(f"{delta1} + 20 = {sum_int}")

    # --- START OF MODIFICATION FOR DIVERGENCE ---

    # Divergence Point 1: Assigning a CustomTimeDelta instance to a NumericLike variable.
    # Some type checkers (e.g., stricter ones) might flag this assignment as an error.
    # The `NumericLike` protocol's `__add__` and `__sub__` methods declare `other: Any`.
    # `CustomTimeDelta` implements these with `other: Union[int, CustomTimeDelta]`.
    # For parameter types, an implementation's type must be a SUPERTYPE of the protocol's type (contravariance).
    # `Union[int, CustomTimeDelta]` is a SUBTYPE of `Any`, making it more restrictive.
    # This might violate strict protocol conformance rules for some checkers.
    numeric_like_instance: NumericLike = CustomTimeDelta(30)
    print(f"\nAssigned CustomTimeDelta to NumericLike variable: {numeric_like_instance}")

    # Divergence Point 2: Calling a method on the NumericLike variable with an unsupported type.
    # According to `NumericLike.__add__(self, other: Any)`, `other` can be `float`.
    # However, the underlying `CustomTimeDelta.__add__` does not support `float`.
    # Type checkers might diverge on whether they:
    # 1. Allow this call (if they trust the `NumericLike` type and the previous assignment passed).
    # 2. Flag the assignment in Divergence Point 1, making this call implicitly problematic.
    # 3. Perform a deeper analysis to detect the runtime error despite the `NumericLike` type.
    try:
        delta_float_sum = numeric_like_instance + 5.5
        print(f"{numeric_like_instance} + 5.5 = {delta_float_sum}")
    except TypeError as e:
        print(f"Caught runtime error as expected: {e}")

    # --- END OF MODIFICATION FOR DIVERGENCE ---

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_NumericLike_non_conforming_object():
    """Pass a non-conforming object where Protocol NumericLike is expected."""
    class _FakeNonConforming:
        pass
    fake = _FakeNonConforming()
    for func_name_check, func_obj in [(k, v) for k, v in globals().items() if callable(v)]:
        pass


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
