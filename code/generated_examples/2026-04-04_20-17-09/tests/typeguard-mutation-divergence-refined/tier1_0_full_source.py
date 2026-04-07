from typing import Any, TypeGuard, List, Union

def is_homogenous_str_list(val: List[Any]) -> TypeGuard[List[str]]:
    """
    A TypeGuard that attempts to narrow a List[Any] to a List[str].
    Checkers might struggle with:
    1. The `all(isinstance(item, str))` logic when `item` is of type `Any`.
    2. Correctly applying the narrowing to `data` after the check, especially with `Any`.
    """
    return all(isinstance(item, str) for item in val)

def process_mixed_data(data: List[Any]) -> None:
    if is_homogenous_str_list(data):
        # 'data' should be narrowed to List[str] here according to TypeGuard rules.
        print("Data is a homogenous string list:")
        
        # --- MODIFIED LINE FOR DIVERGENCE ---
        # This line attempts to append an integer to a list that is
        # currently narrowed to `List[str]` by the TypeGuard.
        # Strict type checkers should flag this as an error because you
        # cannot add an int to a List[str].
        # However, some checkers might be more lenient, considering the
        # original type of 'data' (List[Any]) or might fail to propagate
        # the TypeGuard's effect fully to mutable operations like 'append'.
        data.append(123) 
        # ------------------------------------
        
        for item in data:
            # If the `data.append(123)` was allowed, 'item' could be an int here,
            # leading to a runtime AttributeError for `item.upper()`.
            # A robust type checker should either flag the `append` operation
            # or detect the potential AttributeError here if the `append` was permitted.
            print(f"- {item.upper()}") 
    else:
        print("Data is not a homogenous string list (or is empty):")
        for item in data:
            if isinstance(item, str):
                print(f"- Found string: {item}")
            else:
                print(f"- Found non-string: {item} (type: {type(item).__name__})")

if __name__ == "__main__":
    list_str_int: List[Any] = ["apple", 123, "banana"]
    process_mixed_data(list_str_int) # This will go to the 'else' branch
    print("-" * 20)

    list_only_str: List[Any] = ["cat", "dog"]
    # This call will hit the 'if' branch and the problematic 'data.append(123)' line.
    process_mixed_data(list_only_str) 
    print("-" * 20)

    list_any_bool: List[Any] = [True, False, "bool string"]
    process_mixed_data(list_any_bool) # This will go to the 'else' branch