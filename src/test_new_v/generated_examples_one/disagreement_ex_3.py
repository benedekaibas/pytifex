# Strategy: Shows a complex overload. The call `my_func(Union[int, str])` is designed to be ambiguous. Some checkers may try to resolve the Union, others may not find a matching overload.
from typing import overload, Union

@overload
def my_func(arg: int) -> str:
    ...

@overload
def my_func(arg: str) -> int:
    ...


def my_func(arg: Union[int, str]) -> Union[str, int]:
    if isinstance(arg, int):
        return "int"
    else:
        return 123

result = my_func(Union[int, str]) # Passing the Union type itself
print(result)
