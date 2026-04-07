from typing import Self, List, Type

class DataBatch:
    def __init__(self, data_slice: List[str], batch_index: int) -> None:
        self.data_slice = data_slice
        self.batch_index = batch_index

    def __str__(self) -> str:
        return f"Batch({self.batch_index}): {len(self.data_slice)} items"

    @classmethod
    def create_batches(cls: Type[Self], all_data: List[str], batch_size: int) -> List[Self]:
        """
        Creates a list of DataBatch instances from input data.
        A walrus operator tracks the `current_batch_index` within the comprehension.
        This tests walrus operator's scope and update behavior inside a class method,
        interacting with `cls` and `Self`.
        """
        current_batch_index = 0
        batches = [
            cls(all_data[i:i + batch_size], (current_batch_index := current_batch_index + 1))
            for i in range(0, len(all_data), batch_size)
        ]
        # 'current_batch_index' should hold the total number of batches created.
        # Checkers might fail to correctly track the final value of 'current_batch_index' here.
        print(f"Total batches created (from walrus): {current_batch_index}")
        return batches

if __name__ == "__main__":
    all_words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
    batch_list = DataBatch.create_batches(all_words, 3)
    for batch in batch_list:
        print(batch)
    # Expected output:
    # Total batches created (from walrus): 3
    # Batch(1): 3 items
    # Batch(2): 3 items
    # Batch(3): 2 items