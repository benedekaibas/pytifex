from typing import List, Union

def foo(x: List[Union[int, str]]) -> None:
    pass

y: List[int] = [1, 2, 3]
foo(y)
