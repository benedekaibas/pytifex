from typing import Iterator, List, Tuple

def calculate_running_sum_and_final(data: List[int]) -> Tuple[List[int], int]:
    """
    Calculates running sums using a walrus operator within a generator expression.
    The variable updated by walrus (`current_sum`) is then accessed *after* the
    generator expression has been fully evaluated.
    Checkers like 'ty' had issues correctly identifying the scope and final value
    of such variables.
    """
    current_sum = 0
    # 'current_sum' is updated inside the generator expression.
    # The variable 'total_sums_gen' holds the generator.
    total_sums_gen: Iterator[int] = (current_sum := current_sum + x for x in data)
    
    # At this point, `current_sum` still holds its initial value (0) if the generator hasn't been consumed.
    # The crucial part is that `current_sum` here IS correctly scoped to the outer function.

    final_sums_list = list(total_sums_gen) # Consuming the generator updates 'current_sum' progressively.

    # After consumption, 'current_sum' should hold the total sum of all elements.
    # A checker might incorrectly flag 'current_sum' as undefined or having its initial value here.
    return final_sums_list, current_sum

if __name__ == "__main__":
    numbers = [10, 20, 30, 40]
    sums_list, final_total = calculate_running_sum_and_final(numbers)
    print(f"Running sums: {sums_list}")    # Expected: [10, 30, 60, 100]
    print(f"Final total from walrus: {final_total}") # Expected: 100
    # The output confirms the runtime behavior, testing static analysis's understanding.