from collections.abc import Callable
from typing import Union, List, Optional

def get_data_source_factory(has_data: bool) -> Callable[[], List[str]] | Callable[[], List[None]]:
    """
    Returns a callable that produces either a list of strings or a list of Nones.
    """
    if has_data:
        return lambda: ["apple", "banana", "cherry"]
    return lambda: [None, None, None]

def analyze_data(source_fn: Callable[[], List[str]] | Callable[[], List[None]]) -> int:
    data_list = source_fn()
    # 'data_list' type should be List[str] | List[None].
    # Iterating over it means 'item' is str | None.
    # Checkers might incorrectly assume 'item' is always str or always None.
    valid_items = [item for item in data_list if item is not None]
    if valid_items and isinstance(valid_items[0], str):
        print(f"First valid item: {valid_items[0].upper()}") # Should be safe if `item` is str
    return len(valid_items)

if __name__ == "__main__":
    len_with_data = analyze_data(get_data_source_factory(True))
    print(f"Valid items count from data source: {len_with_data}")
    len_without_data = analyze_data(get_data_source_factory(False))
    print(f"Valid items count from empty source: {len_without_data}")