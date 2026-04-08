"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: typeguard-list-complex-init.py
Patterns detected: 1
    - typeguard_narrowing (5 tests)
Test cases generated: 5
"""

# --- Original source ---

from typing import TypeGuard, Union, overload, Any

class BasicRecord:
    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name

class ComplexRecord:
    @overload
    def __init__(self, id: int, data: dict[str, Any]) -> None: ...
    @overload
    def __init__(self, id: int, value: float, unit: str) -> None: ...
    def __init__(self, id: int, *args, **kwargs) -> None:
        self.id = id
        if len(args) == 1 and isinstance(args[0], dict):
            self.data = args[0]
            self.value = None
            self.unit = None
        elif len(args) == 2 and isinstance(args[0], float) and isinstance(args[1], str):
            self.value = args[0]
            self.unit = args[1]
            self.data = {}
        else:
            raise TypeError("Invalid arguments for ComplexRecord initialization")

def is_complex_record_list(items: list[Union[BasicRecord, ComplexRecord]]) -> TypeGuard[list[ComplexRecord]]:
    # A TypeGuard trying to narrow a list containing classes with overloaded __init__.
    # Type checkers might struggle to fully understand the implications for list elements.
    if not items:
        return True
    return all(isinstance(item, ComplexRecord) for item in items)

def process_records(records: list[Union[BasicRecord, ComplexRecord]]):
    if is_complex_record_list(records):
        # Type should be list[ComplexRecord] here.
        # Checkers might still flag access to specific attributes of ComplexRecord
        # if they don't fully narrow or struggle with overloaded __init__ for instantiation checks.
        for rec in records:
            if rec.value is not None:
                print(f"Complex Record (Value): ID={rec.id}, Value={rec.value} {rec.unit}")
            elif rec.data:
                print(f"Complex Record (Data): ID={rec.id}, Data={rec.data}")
            else:
                print(f"Complex Record (Unknown): ID={rec.id}")
    else:
        for rec in records:
            print(f"Basic Record: ID={rec.id}, Name={rec.name}")

if __name__ == "__main__":
    basic_list = [BasicRecord(1, "Alpha"), BasicRecord(2, "Beta")]
    complex_list = [ComplexRecord(101, {"key": "val"}), ComplexRecord(102, 12.3, "m")]
    mixed_list = [BasicRecord(3, "Gamma"), ComplexRecord(201, 5.0, "s")]

    process_records(basic_list)
    process_records(complex_list)
    process_records(mixed_list) # This should fall into the 'else' block.

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_is_complex_record_list_returns_bool():
    """Verify is_complex_record_list returns a boolean."""
    try:
        result = is_complex_record_list([])
        if not isinstance(result, bool):
            BUGS.append({"line": 26, "type": "ReturnTypeMismatch", "error": f"TypeGuard is_complex_record_list returned {type(result).__name__}, expected bool", "test": "typeguard_returns_bool"})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 26, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_empty_list"})


def test_is_complex_record_list_with_ints():
    """Call is_complex_record_list with list of ints."""
    try:
        result = is_complex_record_list([1, 2, 3])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 26, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_ints"})


def test_is_complex_record_list_with_strings():
    """Call is_complex_record_list with list of strings."""
    try:
        result = is_complex_record_list(["a", "b", "c"])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 26, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_strings"})


def test_is_complex_record_list_with_mixed():
    """Call is_complex_record_list with mixed type list."""
    try:
        result = is_complex_record_list([1, "hello", True, 3.14])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 26, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_mixed"})


def test_is_complex_record_list_with_bools():
    """Call is_complex_record_list with list of booleans."""
    try:
        result = is_complex_record_list([True, False, True])
        assert isinstance(result, bool)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 26, "type": type(e).__name__, "error": str(e)[:200], "test": "typeguard_bools"})


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
