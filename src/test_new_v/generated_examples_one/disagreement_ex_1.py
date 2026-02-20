# Strategy: Demonstrates container invariance. Mypy/Pyright treat List[int] as incompatible with List[Union[int, str]], even though the runtime behavior is safe. Pytype is more lenient.

from typing import List, Union

def append_string(items: List[Union[int, str]]) -> None:
    items.append("hello")

my_list: List[int] = [1, 2, 3]
append_string(my_list)

print(my_list)
