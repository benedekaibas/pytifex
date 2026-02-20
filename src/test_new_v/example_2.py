
from typing import TypeVar

T = TypeVar('T')

def process_item(item: T) -> T:
    if isinstance(item, int):
        return "Not an int anymore" # Returns str, but T is int. Type checker disagreement expected.
    return item

result_int = process_item(10)
result_str = process_item("hello")
