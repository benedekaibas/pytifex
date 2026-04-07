"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: union-callable-list-none.py
Patterns detected: 1
    - callable_param (3 tests)
Test cases generated: 3
"""

# --- Original source ---

from collections.abc import Callable
from typing import Union, List, Optional

def get_data_source_factory(has_data: bool) -> Callable[[], List[str]] | Callable[[], List[None]]:
    """
    Returns a callable that produces either a list of strings or a list of Nones.
    """
    if has_data:
        return lambda: ["apple", "banana", "cherry"]
    return lambda: [None, None, None]

def analyze_data(source_fn: Callable[[], List[str]] | Callable[[], List[None]]) -> int:
    data_list = source_fn()
    # 'data_list' type should be List[str] | List[None].
    # Iterating over it means 'item' is str | None.
    # Checkers might incorrectly assume 'item' is always str or always None.
    valid_items = [item for item in data_list if item is not None]
    if valid_items and isinstance(valid_items[0], str):
        print(f"First valid item: {valid_items[0].upper()}") # Should be safe if `item` is str
    return len(valid_items)

if __name__ == "__main__":
    len_with_data = analyze_data(get_data_source_factory(True))
    print(f"Valid items count from data source: {len_with_data}")
    len_without_data = analyze_data(get_data_source_factory(False))
    print(f"Valid items count from empty source: {len_without_data}")

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_analyze_data_none_callable():
    """Call analyze_data with None for Callable param 'source_fn'."""
    try:
        analyze_data(source_fn=None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 12, "type": type(e).__name__, "error": str(e)[:200], "test": "none_callable"})


def test_analyze_data_string_callable():
    """Call analyze_data with a string for Callable param 'source_fn'."""
    try:
        analyze_data(source_fn="not_callable")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 12, "type": type(e).__name__, "error": str(e)[:200], "test": "string_callable"})


def test_analyze_data_wrong_arity_callable():
    """Call analyze_data with a zero-arg callable for param 'source_fn'."""
    try:
        analyze_data(source_fn=lambda: None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 12, "type": type(e).__name__, "error": str(e)[:200], "test": "wrong_arity_callable"})


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
