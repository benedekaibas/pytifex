from dataclasses import dataclass


@dataclass
class DivergencePattern:
    id: str
    category: str
    description: str
    pep_refs: list[str]


PATTERNS = [
    DivergencePattern(
        id="protocol-defaults",
        category="protocols",
        description="Protocol methods with default argument values may be checked differently when implementations use different defaults",
        pep_refs=["PEP 544"],
    ),
    DivergencePattern(
        id="typed-dict-total",
        category="typed-dict",
        description="TypedDict with mixed total=True/False inheritance and Required/NotRequired fields",
        pep_refs=["PEP 589", "PEP 655"],
    ),
    DivergencePattern(
        id="typeguard-narrowing",
        category="type-narrowing",
        description="TypeGuard and TypeIs functions with generic type parameters and list narrowing",
        pep_refs=["PEP 647", "PEP 742"],
    ),
    DivergencePattern(
        id="param-spec-decorator",
        category="callable",
        description="ParamSpec used in decorators applied to classmethods or staticmethods",
        pep_refs=["PEP 612"],
    ),
    DivergencePattern(
        id="self-generic",
        category="generics",
        description="Self type used in generic classes, especially with abstract methods",
        pep_refs=["PEP 673"],
    ),
    DivergencePattern(
        id="newtype-containers",
        category="newtypes",
        description="NewType values in generic containers and covariance/contravariance handling",
        pep_refs=["PEP 484"],
    ),
    DivergencePattern(
        id="overload-literals",
        category="overloads",
        description="Overloaded functions with Literal types and overlapping signatures",
        pep_refs=["PEP 484", "PEP 586"],
    ),
    DivergencePattern(
        id="final-override",
        category="inheritance",
        description="Final class attributes overridden by properties or descriptors in subclasses",
        pep_refs=["PEP 591"],
    ),
    DivergencePattern(
        id="keyword-vs-positional",
        category="callable",
        description="Protocol callable signatures with keyword-only vs positional-or-keyword parameters",
        pep_refs=["PEP 544", "PEP 570"],
    ),
    DivergencePattern(
        id="bounded-typevars",
        category="generics",
        description="TypeVar bounds with nested generic types and multiple inheritance",
        pep_refs=["PEP 484"],
    ),
]

