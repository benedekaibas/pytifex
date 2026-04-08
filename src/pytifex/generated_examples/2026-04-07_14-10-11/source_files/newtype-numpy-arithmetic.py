import numpy as np
from typing import NewType, reveal_type

# NewType for a custom numeric ID
EntityId = NewType("EntityId", int)

def calculate_offset(base_id: EntityId, array: np.ndarray) -> np.ndarray:
    # Adding a NewType to a numpy array.
    # Some checkers might treat NewType strictly as its base type and allow arithmetic.
    # Others might treat it as a distinct type that's incompatible with numpy's scalar arithmetic rules.
    # The original issue was about datetime; this is numeric.
    offset_array = array + base_id
    reveal_type(offset_array)
    return offset_array

if __name__ == "__main__":
    my_id = EntityId(100)
    data_array = np.array([1, 2, 3])

    result_array = calculate_offset(my_id, data_array)
    print(f"Result array: {result_array}, type: {type(result_array)}")

    # Test with a different base type that NewType wraps
    AnotherId = NewType("AnotherId", float)
    another_id = AnotherId(5.5)
    float_array = np.array([1.1, 2.2])
    result_float_array = calculate_offset(EntityId(int(another_id)), float_array) # Implicit conversion
    print(f"Result float array: {result_float_array}, type: {type(result_float_array)}")

    # What if the NewType is used in a context where its base type's methods are called?
    # e.g., my_id.bit_length() is valid at runtime but type checkers might differ.