from collections.abc import Callable
from typing import Protocol, TypeVar, SupportsFloat, Type

T = TypeVar('T')

class SupportsInitFromFloat(Protocol[T]):
    """
    A protocol for types that can be initialized with a single float argument.
    """
    def __init__(self, value: float, /) -> None: ... # Constructor must take float

class Metric(SupportsInitFromFloat[float]):
    def __init__(self, value: float, /) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"Metric({self.value})"

def construct_from_supports_float(
    # 'factory_type' is expected to be a callable that takes a float and returns
    # an instance of a type that implements SupportsInitFromFloat.
    # The actual argument will be `Type[Metric]`.
    # The core issue from pyright#11343 was `type(init_val)` returning `Callable[[int], SupportsInt]`
    # but the call expecting 0 arguments. Here, we are explicitly saying the factory
    # should take a float.
    factory_type: Callable[[float], SupportsInitFromFloat[float]],
    initial_val: SupportsFloat
) -> SupportsInitFromFloat[float]:
    # `Type[Metric]` is assignable to `Callable[[float], Metric]`.
    # And `Metric` implements `SupportsInitFromFloat[float]`.
    # So `Type[Metric]` should be assignable to `Callable[[float], SupportsInitFromFloat[float]]`.
    # The call `factory_type(float(initial_val))` should be valid.
    # Disagreements can arise if a checker doesn't fully understand the variance
    # or inheritance of Callable when `Type[T]` is involved, or if it struggles
    # with the implicit conversion of `Type[Concrete]` to `Callable`.
    return factory_type(float(initial_val))

if __name__ == "__main__":
    metric_obj = construct_from_supports_float(Metric, 123.45)
    print(f"Constructed metric: {metric_obj}")

    # This should result in a type error because 'str' cannot be passed to a factory expecting float.
    # construct_from_supports_float(Metric, "not_a_float")