"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: func-name-on-callable-divergence.py
Patterns detected: 1
    - callable_param (3 tests)
Test cases generated: 3
"""

# --- Original source ---

from typing import TypeVar, ParamSpec, Callable, Generic, Self
import functools

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

def log_call(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # The Callable[P, R] type does not guarantee the existence of a __name__ attribute.
        # While actual functions/methods have it, a generic Callable could be an object
        # with a __call__ method but no __name__.
        # Ty is stricter about this, while others often infer it works in typical decorator contexts.
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

class DataContainer(Generic[T]):
    value: T

    def __init__(self, value: T):
        self.value = value

    @classmethod
    @log_call
    def create_from_tuple(cls: type[Self], data: tuple[T, ...]) -> Self:
        # Expects a tuple of T, uses the first element to create an instance
        if not data:
            raise ValueError("Tuple cannot be empty")
        print(f"Creating {cls.__name__} from tuple: {data}")
        return cls(data[0])

    @log_call
    def get_value(self) -> T:
        print(f"Getting value from instance: {self.value}")
        return self.value

if __name__ == "__main__":
    container_int = DataContainer[int].create_from_tuple((10, 20, 30))
    print(f"Created container value: {container_int.get_value()}")

    container_str = DataContainer[str].create_from_tuple(("hello", "world"))
    print(f"Created container value: {container_str.get_value()}")

    # This call previously caused a type mismatch error.
    # By fixing that error, we expose a divergence where `ty` flags the
    # access of `func.__name__` in `log_call` because a generic `Callable[P, R]`
    # does not formally guarantee the `__name__` attribute, even though
    # in practice, decorated functions/methods typically have it.
    container_fail = DataContainer[int].create_from_tuple((1, 2)) # Now type-correct
    print(f"Created container value: {container_fail.get_value()}")

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_log_call_none_callable():
    """Call log_call with None for Callable param 'func'."""
    try:
        log_call(func=None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 8, "type": type(e).__name__, "error": str(e)[:200], "test": "none_callable"})


def test_log_call_string_callable():
    """Call log_call with a string for Callable param 'func'."""
    try:
        log_call(func="not_callable")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 8, "type": type(e).__name__, "error": str(e)[:200], "test": "string_callable"})


def test_log_call_wrong_arity_callable():
    """Call log_call with a zero-arg callable for param 'func'."""
    try:
        log_call(func=lambda: None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 8, "type": type(e).__name__, "error": str(e)[:200], "test": "wrong_arity_callable"})


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
