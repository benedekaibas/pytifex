from typing import NewType, List, Optional, Tuple

TransactionId = NewType('TransactionId', str)

def filter_and_count_transactions(tx_ids: List[Optional[TransactionId]]) -> Tuple[List[TransactionId], int]:
    """
    Filters a list of optional NewType values and counts them using a walrus operator
    within a conditional list comprehension.
    This tests:
    1. Correct scope and update of `valid_count` by walrus.
    2. Correct type inference for `tx_id_val` and `TransactionId` within the comprehension.
    3. Accessing `valid_count` correctly in the outer scope.
    """
    valid_count = 0
    # Using walrus in a conditional within a list comprehension.
    # 'tx_id_val' is assigned and used. 'valid_count' is incremented.
    # Checkers can struggle with these intertwined assignments and conditions.
    processed_transactions = [
        tx_id_val
        for item in tx_ids
        if (tx_id_val := item) is not None
        and (valid_count := valid_count + 1) # This increments valid_count for each valid transaction.
    ]
    # 'valid_count' should hold the total number of non-None TransactionIds.
    return processed_transactions, valid_count

if __name__ == "__main__":
    input_transactions = [
        TransactionId("tx1001"), None, TransactionId("tx1002"),
        None, TransactionId("tx1003"), TransactionId("tx1004")
    ]
    filtered_list, count = filter_and_count_transactions(input_transactions)
    print(f"Filtered transactions: {filtered_list}") # Expected: [tx1001, tx1002, tx1003, tx1004]
    print(f"Total valid count: {count}")             # Expected: 4