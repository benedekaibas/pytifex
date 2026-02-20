# Strategy: Illustrates Protocol privacy. `_private_attr` suggests it should not be accessible, but Python allows it at runtime.  Checkers might disagree if `MyClass` truly implements `MyProto` given the attribute name.
from typing import Protocol, runtime_checkable

class MyProto(Protocol):
    _private_attr: int  # Marked as private

class MyClass:
    _private_attr: int = 5

    def __init__(self) -> None:
        pass


def accept_proto(arg: MyProto) -> None:
    print(arg._private_attr)  # Accessing supposedly private attribute.

instance = MyClass()
accept_proto(instance)
