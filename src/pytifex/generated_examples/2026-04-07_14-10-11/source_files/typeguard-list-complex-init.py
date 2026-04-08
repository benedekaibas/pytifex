from typing import TypeGuard, Union, overload, Any

class BasicRecord:
    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name

class ComplexRecord:
    @overload
    def __init__(self, id: int, data: dict[str, Any]) -> None: ...
    @overload
    def __init__(self, id: int, value: float, unit: str) -> None: ...
    def __init__(self, id: int, *args, **kwargs) -> None:
        self.id = id
        if len(args) == 1 and isinstance(args[0], dict):
            self.data = args[0]
            self.value = None
            self.unit = None
        elif len(args) == 2 and isinstance(args[0], float) and isinstance(args[1], str):
            self.value = args[0]
            self.unit = args[1]
            self.data = {}
        else:
            raise TypeError("Invalid arguments for ComplexRecord initialization")

def is_complex_record_list(items: list[Union[BasicRecord, ComplexRecord]]) -> TypeGuard[list[ComplexRecord]]:
    # A TypeGuard trying to narrow a list containing classes with overloaded __init__.
    # Type checkers might struggle to fully understand the implications for list elements.
    if not items:
        return True
    return all(isinstance(item, ComplexRecord) for item in items)

def process_records(records: list[Union[BasicRecord, ComplexRecord]]):
    if is_complex_record_list(records):
        # Type should be list[ComplexRecord] here.
        # Checkers might still flag access to specific attributes of ComplexRecord
        # if they don't fully narrow or struggle with overloaded __init__ for instantiation checks.
        for rec in records:
            if rec.value is not None:
                print(f"Complex Record (Value): ID={rec.id}, Value={rec.value} {rec.unit}")
            elif rec.data:
                print(f"Complex Record (Data): ID={rec.id}, Data={rec.data}")
            else:
                print(f"Complex Record (Unknown): ID={rec.id}")
    else:
        for rec in records:
            print(f"Basic Record: ID={rec.id}, Name={rec.name}")

if __name__ == "__main__":
    basic_list = [BasicRecord(1, "Alpha"), BasicRecord(2, "Beta")]
    complex_list = [ComplexRecord(101, {"key": "val"}), ComplexRecord(102, 12.3, "m")]
    mixed_list = [BasicRecord(3, "Gamma"), ComplexRecord(201, 5.0, "s")]

    process_records(basic_list)
    process_records(complex_list)
    process_records(mixed_list) # This should fall into the 'else' block.