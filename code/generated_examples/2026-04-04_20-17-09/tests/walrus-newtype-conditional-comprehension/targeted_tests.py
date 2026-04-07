"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: walrus-newtype-conditional-comprehension.py
Patterns detected: 1
    - newtype (3 tests)
Test cases generated: 3
"""

# --- Original source ---

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

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_TransactionId_from_string():
    """Create TransactionId from a plain string."""
    try:
        val = TransactionId("test_value")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_from_base"})


def test_TransactionId_from_int():
    """Create TransactionId from an int (wrong base type)."""
    try:
        val = TransactionId(42)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_wrong_base"})


def test_TransactionId_from_none():
    """Create TransactionId from None."""
    try:
        val = TransactionId(None)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 3, "type": type(e).__name__, "error": str(e)[:200], "test": "newtype_none"})


# --- Runner ---
if __name__ == "__main__":
    import sys
    _test_fns = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    print(f"Running {len(_test_fns)} targeted tests...")
    _passed = 0
    _failed = 0
    for _name, _fn in _test_fns:
        try:
            _fn()
            _passed += 1
        except Exception as _e:
            _failed += 1
    print(f"Passed: {_passed}, Failed: {_failed}, Bugs found: {len(BUGS)}")
    for _bug in BUGS:
        print(f"  BUG L{_bug['line']} [{_bug['type']}] {_bug['error']}")
