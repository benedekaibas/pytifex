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