from typing import TypeVar, ParamSpec, Callable, Self, Type
from functools import wraps

P = ParamSpec('P')
R = TypeVar('R')

def logging_factory_decorator(cls_method: Callable[P, R]) -> Callable[P, R]:
    """
    A decorator that logs calls to a class method (factory).
    It uses ParamSpec to preserve the signature.
    """
    @wraps(cls_method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"LOG: Calling factory method '{cls_method.__name__}' with args={args}, kwargs={kwargs}")
        return cls_method(*args, **kwargs)
    return wrapper

class Item:
    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name

    def __str__(self) -> str:
        return f"Item(id={self.id}, name='{self.name}')"

    @classmethod
    @logging_factory_decorator
    def create_item(cls: Type[Self], item_id: int, item_name: str) -> Self:
        """
        A class method acting as a factory.
        `cls` is `Type[Self]`, and it's called to create an instance of `Self`.
        Checkers must correctly infer `P` for `logging_factory_decorator` from `create_item`'s signature
        and verify that `cls(item_id, item_name)` aligns with `Item.__init__(self, id, name)`.
        Disagreements can arise from `ParamSpec` interaction with `Type[Self]` in `classmethod`.
        """
        return cls(item_id, item_name)

if __name__ == "__main__":
    my_item = Item.create_item(101, "Widget")
    print(my_item)

    # This should be a type error (wrong argument type for item_name)
    # Item.create_item(102, 123)