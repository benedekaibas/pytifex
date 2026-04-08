"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: match-typevartuple-never.py
Patterns detected: 2
    - tuple_length (5 tests)
  - main_block_replay (4 tests)
Test cases generated: 9
"""

# --- Original source ---

from typing import Literal, assert_never, Tuple, TypeVarTuple, Unpack

Ts = TypeVarTuple('Ts')

def handle_config_directive(directive: Tuple[Literal['set', 'get', 'log'], Unpack[Ts]]) -> None:
    """
    Handles configuration directives using TypeVarTuple for flexible arguments.
    Exhaustiveness checking with `TypeVarTuple` is notoriously difficult for checkers.
    A checker might fail to determine if all possible tuple forms (e.g., ('set', value), ('set',))
    are covered, leading to false positives or false negatives for `assert_never`.
    """
    match directive:
        case 'set', key, value:
            print(f"SET command: key={key}, value={value}")
        case 'get', key:
            print(f"GET command: key={key}")
        case 'log', message:
            print(f"LOG message: {message}")
        case 'log', *parts: # Catch all log messages (single or multiple parts)
            print(f"LOG multi-part: {parts}")
        case _:
            # This branch should be unreachable if all 'set', 'get', 'log' variations
            # are covered by the preceding patterns.
            # However, `TypeVarTuple` makes exhaustiveness hard.
            assert_never(directive) # Type checker might fail to flag this.
            print(f"Unhandled directive: {directive}")

if __name__ == "__main__":
    handle_config_directive(('set', 'theme', 'dark'))
    handle_config_directive(('get', 'version'))
    handle_config_directive(('log', 'System started'))
    handle_config_directive(('log', 'User', 'admin', 'logged in'))
    # What if a directive of type ('set',) or ('get',) is passed?
    # This might implicitly match a `case _, *parts` if it were present, or fall to `_`.
    # A checker might struggle to see that 'set', 'get', 'log' are the only Literals,
    # and all their potential argument counts are (hopefully) covered.

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 12

# --- Test cases ---

def test_handle_config_directive_empty_tuple():
    """Call handle_config_directive with empty tuple for param 'directive'."""
    try:
        handle_config_directive(directive=())
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 5, "type": type(e).__name__, "error": str(e)[:200], "test": "empty_tuple"})


def test_handle_config_directive_single_element_tuple():
    """Call handle_config_directive with single-element tuple for param 'directive'."""
    try:
        handle_config_directive(directive=(1,))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 5, "type": type(e).__name__, "error": str(e)[:200], "test": "single_element_tuple"})


def test_handle_config_directive_none_in_tuple():
    """Call handle_config_directive with None element in tuple for param 'directive'."""
    try:
        handle_config_directive(directive=(None,))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 5, "type": type(e).__name__, "error": str(e)[:200], "test": "none_in_tuple"})


def test_handle_config_directive_wrong_types_in_tuple():
    """Call handle_config_directive with wrong types in tuple for param 'directive'."""
    try:
        handle_config_directive(directive=(123, 456, 789))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 5, "type": type(e).__name__, "error": str(e)[:200], "test": "wrong_types_in_tuple"})


def test_handle_config_directive_string_instead_of_tuple():
    """Call handle_config_directive with a string instead of tuple for param 'directive'."""
    try:
        handle_config_directive(directive="not a tuple")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 5, "type": type(e).__name__, "error": str(e)[:200], "test": "string_instead_of_tuple"})


def test_main_call_handle_config_directive___set____theme__():
    """Execute main block call: handle_config_directive(('set', 'theme', 'dark'))"""
    import traceback as _tb, sys as _sys
    try:
        handle_config_directive(('set', 'theme', 'dark'))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        _fault_line = 29
        _root = e
        while getattr(_root, '__cause__', None) or getattr(_root, '__context__', None):
            _root = _root.__cause__ or _root.__context__
        _frames = _tb.extract_tb(_root.__traceback__)
        if _frames:
            _fault_line = _frames[-1].lineno - _SOURCE_LINE_OFFSET
        BUGS.append({"line": _fault_line, "type": type(e).__name__, "error": str(e)[:200], "test": "main_block_call"})


def test_main_call_handle_config_directive___get____version():
    """Execute main block call: handle_config_directive(('get', 'version'))"""
    import traceback as _tb, sys as _sys
    try:
        handle_config_directive(('get', 'version'))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        _fault_line = 30
        _root = e
        while getattr(_root, '__cause__', None) or getattr(_root, '__context__', None):
            _root = _root.__cause__ or _root.__context__
        _frames = _tb.extract_tb(_root.__traceback__)
        if _frames:
            _fault_line = _frames[-1].lineno - _SOURCE_LINE_OFFSET
        BUGS.append({"line": _fault_line, "type": type(e).__name__, "error": str(e)[:200], "test": "main_block_call"})


def test_main_call_handle_config_directive___log____System_():
    """Execute main block call: handle_config_directive(('log', 'System started'))"""
    import traceback as _tb, sys as _sys
    try:
        handle_config_directive(('log', 'System started'))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        _fault_line = 31
        _root = e
        while getattr(_root, '__cause__', None) or getattr(_root, '__context__', None):
            _root = _root.__cause__ or _root.__context__
        _frames = _tb.extract_tb(_root.__traceback__)
        if _frames:
            _fault_line = _frames[-1].lineno - _SOURCE_LINE_OFFSET
        BUGS.append({"line": _fault_line, "type": type(e).__name__, "error": str(e)[:200], "test": "main_block_call"})


def test_main_call_handle_config_directive___log____User___():
    """Execute main block call: handle_config_directive(('log', 'User', 'admin', 'logged in'))"""
    import traceback as _tb, sys as _sys
    try:
        handle_config_directive(('log', 'User', 'admin', 'logged in'))
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        _fault_line = 32
        _root = e
        while getattr(_root, '__cause__', None) or getattr(_root, '__context__', None):
            _root = _root.__cause__ or _root.__context__
        _frames = _tb.extract_tb(_root.__traceback__)
        if _frames:
            _fault_line = _frames[-1].lineno - _SOURCE_LINE_OFFSET
        BUGS.append({"line": _fault_line, "type": type(e).__name__, "error": str(e)[:200], "test": "main_block_call"})


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
