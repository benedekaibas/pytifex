from typing import Any, Literal, Union, assert_never

def describe_resource_state(state: Union[Literal["created", "running", "stopped"], Any]) -> str:
    """
    Function to describe a resource state. The input can also be Any.
    The `assert_never` call at the end is meant to catch unhandled states.
    However, if `state` can be `Any`, then `Any` is not `Never`.
    Some checkers might correctly flag `assert_never(state)` as an error
    because `Any` represents a type that *could* be `created`, `running`, or `stopped`,
    meaning the earlier cases are not truly exhaustive against `Any`.
    Others might let it pass, treating `Any` as implicitly handled or ignoring its specific type.
    """
    match state:
        case "created":
            return "Resource is freshly created."
        case "running":
            return "Resource is actively running."
        case "stopped":
            return "Resource has been gracefully stopped."
        case _:
            # If 'state' is `Any`, this 'case _' will match it.
            # 'assert_never' should complain that `Any` cannot be narrowed to `Never`.
            # This is a key point of disagreement for how `Any` interacts with exhaustiveness.
            assert_never(state) # Type checker should report an error here.
            return "Unknown resource state encountered."

if __name__ == "__main__":
    print(describe_resource_state("created"))
    print(describe_resource_state("running"))
    print(describe_resource_state("stopped"))
    # These calls should trigger the 'Any' path and potentially an assert_never type error.
    print(describe_resource_state(123))
    print(describe_resource_state("paused"))