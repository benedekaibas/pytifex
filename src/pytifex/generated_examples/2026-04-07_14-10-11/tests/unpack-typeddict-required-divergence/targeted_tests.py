"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: unpack-typeddict-required-divergence.py
Patterns detected: 1
    - protocol_conformance (2 tests)
Test cases generated: 2
"""

# --- Original source ---

from typing import Protocol, TypedDict, Unpack, Any, Literal, NotRequired # Added NotRequired

class RenderSettings(TypedDict): # Removed extra_items=Any
    theme: Literal["dark", "light"] # This is a Required field by default
    font_size: NotRequired[int]
    debug_mode: NotRequired[bool]

class Renderer(Protocol):
    def render(self, content: str, **kwargs: Unpack[RenderSettings]):
        """Renders content using provided settings.
        Note: 'theme' is a Required key in RenderSettings, so it must be passed by callers."""
        # The default handling in the protocol's docstring is *not* part of the signature.
        # The protocol's definition of `render` states `**kwargs: Unpack[RenderSettings]`,
        # implying all *required* keys in `RenderSettings` must be present at the call site.
        print(f"Rendering '{content}' with settings: {kwargs}")

class HTMLRenderer:
    # This implementation's signature matches the protocol precisely.
    def render(self, content: str, **kwargs: Unpack[RenderSettings]):
        # The implementation's internal default argument handling makes it robust
        # to 'theme' being potentially absent if its own signature were more lenient.
        # However, because it implements the Renderer protocol, the external contract
        # still requires 'theme' to be passed.
        settings: RenderSettings = {'theme': 'light', 'font_size': 14, **kwargs}
        print(f"HTML Rendering '{content}' with explicit settings: {settings}")

if __name__ == "__main__":
    html_renderer: Renderer = HTMLRenderer() # This assignment should now pass for all checkers.

    html_renderer.render("Hello", theme="dark", debug_mode=True)

    # This call is the intended point of divergence.
    # 'theme' is a Required field in RenderSettings, but it's omitted here.
    # Strict type checkers (like mypy, pyright) should flag this as an error.
    # Other type checkers might fail to catch this, leading to divergence.
    html_renderer.render("World")

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_HTMLRenderer_has_render():
    """Verify HTMLRenderer has required protocol method 'render'."""
    try:
        obj = HTMLRenderer()
        method = getattr(obj, "render", None)
        if method is None:
            BUGS.append({"line": 8, "type": "AttributeError", "error": "HTMLRenderer missing protocol method render", "test": "protocol_method_exists"})
        elif not callable(method):
            BUGS.append({"line": 8, "type": "TypeError", "error": "HTMLRenderer.render is not callable", "test": "protocol_method_callable"})
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        BUGS.append({"line": 8, "type": type(e).__name__, "error": str(e)[:200], "test": "protocol_check"})


def test_Renderer_non_conforming_object():
    """Pass a non-conforming object where Protocol Renderer is expected."""
    class _FakeNonConforming:
        pass
    fake = _FakeNonConforming()
    for func_name_check, func_obj in [(k, v) for k, v in globals().items() if callable(v)]:
        pass


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
