# Pytifex

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Automated mutation based differential testing for Python type checkers. Pytifex discovers disagreements between type checkers by mining historical bug reports, generating targeted
test cases with an LLM, and establishing ground truth through runtime validation testing.

For more information, see [the Pytifex documentation](https://benedekaibas.github.io/pytifex/docs).

> **Note:** Pytifex implements a bug-seeded mutation methodology for proactively finding type checker bugs before users encounter them.

## Type Checkers

Pytifex tests the following type checkers:

| Checker | Version | Repository |
|---------|---------|------------|
| mypy | 1.19.0 | [python/mypy](https://github.com/python/mypy) |
| pyrefly | 0.44.2 | [facebook/pyrefly](https://github.com/facebook/pyrefly) |
| zuban | 0.3.0 | [zubanls/zuban](https://github.com/zubanls/zuban) |
| ty | 0.0.1-alpha.32 | [astral-sh/ty](https://github.com/astral-sh/ty) |

## Divergence Patterns

| Pattern | Description | PEPs |
|---------|-------------|------|
| `protocol-defaults` | Protocol methods with different default argument values | 544 |
| `typed-dict-total` | TypedDict with mixed `total`/`Required`/`NotRequired` inheritance | 589, 655 |
| `typeguard-narrowing` | TypeGuard/TypeIs with generic type parameters | 647, 742 |
| `param-spec-decorator` | ParamSpec decorators on classmethods/staticmethods | 612 |
| `self-generic` | Self type in generic classes with abstract methods | 673 |
| `newtype-containers` | NewType in containers (covariance/contravariance) | 484 |
| `overload-literals` | Overloaded functions with Literal type discrimination | 484, 586 |
| `final-override` | Final attributes overridden by properties | 591 |
| `keyword-vs-positional` | Protocol callables with keyword-only parameters | 544, 570 |
| `bounded-typevars` | TypeVar bounds with nested generics | 484 |

## Installation

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

To download and run Pytifex, please follow the following commands:

```bash
pip install pytifex

*or*

pip3 install pytifex
```

**NOTE**: Type checkers are automatically installed by `uv` when you run the tool.

## Usage

```bash
# You have to set your Google Gemini API key in your terminal window
export GEMINI_API_KEY="your-api-key-here"

# Run the full pipeline: mine → generate → filter → evaluate
uv run main.py

# Generate until N disagreements are found
uv run main.py --num-examples 10

# Use a different model
uv run main.py --model gemini-2.5-pro

# Skip GitHub seed fetching
uv run main.py --no-github
```

### Commands

| Command | Description |
|---------|-------------|
| `uv run main.py` | Full pipeline (generate + evaluate) |
| `uv run main.py generate` | Generate disagreements only |
| `uv run main.py check` | Run type checkers on existing examples |
| `uv run main.py eval` | Evaluate existing results |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-examples N` | 5 | Target disagreement count |
| `--batch-size N` | 15 | Examples per LLM batch |
| `--max-attempts N` | 5 | Max generation attempts |
| `--max-refinements N` | 2 | Refinement attempts per example |
| `--model MODEL` | `gemini-2.5-flash` | Gemini model |
| `--eval-method METHOD` | `comprehensive` | Evaluation method |
| `--no-github` | — | Skip GitHub seed fetching |
| `-v, --verbose` | — | Show all examples |

## Evaluation

Pytifex uses a multi-phase evaluation oracle to determine which checker is correct:

| Phase | Method | Confidence |
|-------|--------|------------|
| 0 | AST-based PEP specification oracle | 0.85–0.95 |
| 1 | Runtime crash detection | 0.95–1.0 |
| 2 | Hypothesis property-based testing | 0.85 |
| 3 | PEP specification compliance matching | 0.80 |
| 4 | Static flow analysis | 0.80 |

**Key insight:** Runtime behavior is the ultimate ground truth. If code raises `TypeError` at runtime, any checker that reported "OK" is definitively wrong.

