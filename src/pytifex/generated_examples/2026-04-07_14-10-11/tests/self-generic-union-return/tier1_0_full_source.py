from typing import Generic, TypeVar, Self, Union, Optional

T = TypeVar("T")

class Node(Generic[T]):
    def __init__(self, value: T, children: Optional[list[Self]] = None):
        self.value = value
        self.children = children if children is not None else []

    def add_child(self, child_value: T) -> Self:
        new_child = type(self)(child_value)
        self.children.append(new_child)
        return new_child

    def find_first(self, predicate: Callable[[T], bool]) -> Optional[Union[Self, "Node[Any]"]]:
        # Return type is a union of Self or a Node with a different type var.
        # This complexity can lead to non-deterministic type resolution for different checkers.
        # The original issue was about non-deterministic union ordering in error messages.
        if predicate(self.value):
            return self
        for child in self.children:
            found = child.find_first(predicate)
            if found:
                return found
        return None

    def __repr__(self) -> str:
        return f"Node({self.value})"

if __name__ == "__main__":
    from typing import Callable, Any

    node_int = Node[int](10)
    node_int.add_child(20).add_child(30)
    node_int.add_child(40)

    # Search for an even number
    found_node = node_int.find_first(lambda x: x % 2 == 0)
    if found_node:
        print(f"Found even node: {found_node.value}, type: {type(found_node)}")
        # Checkers might struggle to consistently infer the exact type of `found_node.value` here
        # given the `Union[Self, "Node[Any]"]` return.

    node_str = Node[str]("root")
    node_str.add_child("child1").add_child("grandchild1")
    node_str.add_child("child2")

    found_str_node = node_str.find_first(lambda s: "grand" in s)
    if found_str_node:
        print(f"Found 'grand' node: {found_str_node.value}, type: {type(found_str_node)}")

    # What if a predicate is incompatible with T?
    # This should be caught, but the complex union return might affect error clarity.
    # bad_found = node_int.find_first(lambda s: "test" in s)