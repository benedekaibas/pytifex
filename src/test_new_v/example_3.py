
from typing import Union, List, Dict

def process_data(data: Union[List[int], Dict[str, str]]) -> str:
    if isinstance(data, list):
        return str(sum(data))
    else:
        return ", ".join(data.values())

result1 = process_data([1, 2, 3])
result2 = process_data({"a": "apple", "b": "banana"})
