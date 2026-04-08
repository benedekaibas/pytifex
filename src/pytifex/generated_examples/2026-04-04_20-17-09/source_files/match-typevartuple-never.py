from typing import Literal, assert_never, Tuple, TypeVarTuple, Unpack

Ts = TypeVarTuple('Ts')

def handle_config_directive(directive: Tuple[Literal['set', 'get', 'log'], Unpack[Ts]]) -> None:
    """
    Handles configuration directives using TypeVarTuple for flexible arguments.
    Exhaustiveness checking with `TypeVarTuple` is notoriously difficult for checkers.
    A checker might fail to determine if all possible tuple forms (e.g., ('set', value), ('set',))
    are covered, leading to false positives or false negatives for `assert_never`.
    """
    match directive:
        case 'set', key, value:
            print(f"SET command: key={key}, value={value}")
        case 'get', key:
            print(f"GET command: key={key}")
        case 'log', message:
            print(f"LOG message: {message}")
        case 'log', *parts: # Catch all log messages (single or multiple parts)
            print(f"LOG multi-part: {parts}")
        case _:
            # This branch should be unreachable if all 'set', 'get', 'log' variations
            # are covered by the preceding patterns.
            # However, `TypeVarTuple` makes exhaustiveness hard.
            assert_never(directive) # Type checker might fail to flag this.
            print(f"Unhandled directive: {directive}")

if __name__ == "__main__":
    handle_config_directive(('set', 'theme', 'dark'))
    handle_config_directive(('get', 'version'))
    handle_config_directive(('log', 'System started'))
    handle_config_directive(('log', 'User', 'admin', 'logged in'))
    # What if a directive of type ('set',) or ('get',) is passed?
    # This might implicitly match a `case _, *parts` if it were present, or fall to `_`.
    # A checker might struggle to see that 'set', 'get', 'log' are the only Literals,
    # and all their potential argument counts are (hopefully) covered.