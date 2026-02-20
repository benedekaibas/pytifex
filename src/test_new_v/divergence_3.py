
from typing import Generic, TypeVar

T = TypeVar('T', contravariant=True)

class Handler(Generic[T]):
    def handle(self, x: T) -> None:
        pass

class Animal:
    pass

class Dog(Animal):
    pass

class DogHandler(Handler[Dog]):
    # This method should accept Dog, but we widen it to Animal.
    # Some checkers might accept this, others might flag it.
    def handle(self, x: Animal) -> None:
        print("Handling an animal, even though this is a DogHandler")

# This should be fine, passing a Dog to a DogHandler's handle method.
dog_handler = DogHandler()
dog_handler.handle(Dog())

# This should be an error for a strict checker, as Animal is wider than Dog
# for a contravariant T in Handler[Dog].
dog_handler.handle(Animal())
