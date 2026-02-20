
from typing_extensions import Self

class Builder:
    def set_x(self, x: int) -> Self:
        return self

class Advanced(Builder):
    def set_y(self, y: str) -> Self:
        return self

def chain(b: Builder) -> Builder:
    return b.set_x(1)

result = chain(Advanced())
# This line should ideally be an error for some checkers if 'Self' is lost
# as result is typed as Builder, but it's an Advanced instance.
result.set_y("hi")
