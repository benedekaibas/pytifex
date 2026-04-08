from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Self

T = TypeVar("T")

class AbstractProcessor(Generic[T], ABC):
    @abstractmethod
    def process_items(self, data: list[T]) -> list[Self]:
        """Processes items and returns a list of Self instances."""
        pass

class ConcreteProcessor(AbstractProcessor[T]):
    def process_items(self, data: list[T]) -> list[Self]:
        # A real implementation would process data. For type checking,
        # we focus on the return type compatibility.
        print(f"Processing data of type {type(data)} with {type(self)}")
        # Simulate returning instances of Self (ConcreteProcessor[T])
        return [self, self] # Intentionally returning two to test list[Self]

if __name__ == "__main__":
    cp_int = ConcreteProcessor[int]()
    result = cp_int.process_items([1, 2, 3])
    print(f"Result type: {type(result)} containing {type(result[0])}")
    # mypy might complain about Self being incompatible with ConcreteProcessor[T] in the return
    # pyright might be fine
    # zuban might have issues with Generic T in closure/method context