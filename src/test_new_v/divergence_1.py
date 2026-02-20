
from typing import Protocol, List

class Reader(Protocol):
    def read(self, n: int = -1) -> bytes: ...

class File:
    def read(self, n: int = 100) -> bytes:  # Different default, mypy may error, pyright ok
        return b"data"

def use(r: Reader):
    r.read()

use(File())
