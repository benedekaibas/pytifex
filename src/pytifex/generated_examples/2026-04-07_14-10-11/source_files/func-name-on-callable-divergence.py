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