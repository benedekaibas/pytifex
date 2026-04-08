from typing import Generic, TypeVar, Self, Union, reveal_type

V = TypeVar("V", bound=Union[int, float]) # Value type for arithmetic

class Scalar(Generic[V]):
    def __init__(self, value: V) -> None:
        self.value = value

    def __add__(self, other: Union[V, Self]) -> Self:
        if isinstance(other, Scalar):
            return type(self)(self.value + other.value)
        return type(self)(self.value + other)

    def __sub__(self, other: Union[V, Self]) -> Self:
        if isinstance(other, Scalar):
            return type(self)(self.value - other.value)
        return type(self)(self.value - other)

    def __repr__(self) -> str:
        return f"Scalar({self.value})"

class CustomInt(Scalar[int]):
    # This class explicitly specifies int, but the generic Self return
    # might cause issues when interacting with raw ints or floats.
    pass

class CustomFloat(Scalar[float]):
    pass

if __name__ == "__main__":
    s1 = CustomInt(10)
    s2 = CustomInt(5)
    s_sum = s1 + s2
    reveal_type(s_sum)
    print(f"{s1} + {s2} = {s_sum}, type: {type(s_sum)}")

    s_int_sum = s1 + 3
    reveal_type(s_int_sum)
    print(f"{s1} + 3 = {s_int_sum}, type: {type(s_int_sum)}")

    s_float_diff = CustomFloat(10.5) - CustomFloat(2.5)
    reveal_type(s_float_diff)
    print(f"{CustomFloat(10.5)} - {CustomFloat(2.5)} = {s_float_diff}, type: {type(s_float_diff)}")

    # This might be tricky if V is constrained, and Self needs to inherit that constraint
    # Adding a float to an CustomInt instance
    s_mixed_add = s1 + CustomFloat(2.5) # This should error as CustomInt can't hold float
    # pyright might accept due to general Self, mypy might catch it.
    print(f"{s1} + {CustomFloat(2.5)} = {s_mixed_add}, type: {type(s_mixed_add)}")