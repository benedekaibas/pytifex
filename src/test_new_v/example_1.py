
from typing import Self, TypeVar

T = TypeVar("T")

class BaseProcessor(object):
    def process(self, value: T) -> Self:
        return self

class IntProcessor(BaseProcessor[int]):
    def process(self, value: int) -> Self:
        return self

class StringProcessor(BaseProcessor[str]):
    def process(self, value: str) -> Self:
        return self

def handle_processor(p: BaseProcessor[str]):
    pass

int_proc = IntProcessor()
handle_processor(int_proc.process(10))
