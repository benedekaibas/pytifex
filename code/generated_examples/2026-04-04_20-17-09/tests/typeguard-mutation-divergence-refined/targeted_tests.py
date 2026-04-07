"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: typeguard-mutation-divergence-refined.py
Patterns detected: 1
    - typeguard_narrowing (5 tests)
Test cases generated: 5
"""

# --- Original source ---

from typing import Any, TypeGuard, List, Union

def is_homogenous_str_list(val: List[Any]) -> TypeGuard[List[str]]:
    """
    A TypeGuard that attempts to narrow a List[Any] to a List[str].
    Checkers might struggle with:
    1. The `all(isinstance(item, str))` logic when `item` is of type `Any`.
    2. Correctly applying the narrowing to `data` after the check, especially with `Any`.
    """
    return all(isinstance(item, str) for item in val)

def process_mixed_data(data: List[Any]) -> None:
    if is_homogenous_str_list(data):
        # 'data' should be narrowed to List[str] here according to TypeGuard rules.
        print("Data is a homogenous string list:")
        
        # --- MODIFIED LINE FOR DIVERGENCE ---
        # This line attempts to append an integer to a list that is
        # currently narrowed to `List[str]` by the TypeGuard.
        # Strict type checkers should flag this as an error because you
        # cannot add an int to a List[str].
        # However, some checkers might be more lenient, considering the
        # original type of 'data' (List[Any]) or might fail to propagate
        # the TypeGuard's effect fully to mutable operations like 'append'.
        data.append(123) 
        # ------------------------------------
        
        for item in data:
            # If the `data.append(123)` was allowed, 'item' could be an int here,
            # leading to a runtime AttributeError for `item.upper()`.
            # A robust type checker should either flag the `append` operation
            # or detect the potential AttributeError here if the `append` was permitted.
            print(f"- {item.upper()}") 
    else:
        print("Data is not a homogenous string list (or is empty):")
        for item in data:
            if isinstance(item, str):
                print(f"- Found string: {item}")
            else:
                print(f"- Found non-string: {item} (type: {type(item).__name__})")

if __name__ == "__main__":
    list_str_int: List[Any] = ["apple", 123, "banana"]
    process_mixed_data(list_str_int) # This will go to the 'else' branch
    print("-" * 20)

    list_only_str: List[Any] = ["cat", "dog"]
    # This call will hit the 'if' branch and the problematic 'data.append(123)' line.
    process_mixed_data(list_only_str) 
    print("-" * 20)

    list_any_bool: List[Any] = [True, False, "bool string"]
    process_mixed_data(list_any_bool) # This will go to the 'else' branch

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_is_homogenous_str_list_returns_bool():
    """Verify is_homogenous_str_list returns a boolean."""
    try:
        result = is_homogenous_str_list([])
        if not isinstance(result, bool):
            BUGS.append({"line": 3, "type": "ReturnTypeMismatch", "error": f"TypeGuard is_homogenous_str_list returned {type(result).__name__}, expected bool", "test": "typeguard_returns_bool"})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_empty_list"})


def test_is_homogenous_str_list_with_ints():
    """Call is_homogenous_str_list with list of ints."""
    try:
        result = is_homogenous_str_list([1, 2, 3])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_ints"})


def test_is_homogenous_str_list_with_strings():
    """Call is_homogenous_str_list with list of strings."""
    try:
        result = is_homogenous_str_list(["a", "b", "c"])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_strings"})


def test_is_homogenous_str_list_with_mixed():
    """Call is_homogenous_str_list with mixed type list."""
    try:
        result = is_homogenous_str_list([1, "hello", True, 3.14])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_mixed"})


def test_is_homogenous_str_list_with_bools():
    """Call is_homogenous_str_list with list of booleans."""
    try:
        result = is_homogenous_str_list([True, False, True])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_bools"})


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
