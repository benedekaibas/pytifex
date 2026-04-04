"""
Prompt builders for Pytifex.
Use REAL bug reports from type checker GitHub issues as seeds,
then ask the LLM to generate VARIATIONS that might cause divergences.
"""

from patterns import PATTERNS
from github_issues import IssueExample, format_example_for_prompt


def build_seed_based_prompt(
    seed_examples: list[IssueExample],
    num_variations: int = 10,
) -> str:
    """
    Build a prompt using real GitHub issue examples as seeds.

    This is the PRIMARY generation strategy:
    1. Fetch real bug reports from mypy/pyrefly/ty repos
    2. Show them to the LLM as examples of real type checker bugs
    3. Ask it to generate VARIATIONS that might cause divergences
    """

    # Format seed examples
    seeds_text = "\n\n".join(
        format_example_for_prompt(ex) for ex in seed_examples[:5]
    )

    # Pattern descriptions for context
    patterns_text = "\n".join(
        f"- {p.id}: {p.description}"
        for p in PATTERNS[:6]
    )

    return f"""You are an expert in Python type systems. Your task is to generate Python code
that causes DISAGREEMENTS between type checkers (mypy, pyrefly, zuban, ty).

Below are REAL code examples extracted from closed GitHub issues in type checker repositories.
These represent actual bugs - false positives (checker reports error when code is correct) or
false negatives (checker misses a real error).

## REAL BUG EXAMPLES FROM TYPE CHECKER ISSUES:

{seeds_text}

## YOUR TASK:

Using these real bugs as inspiration, generate {num_variations} NEW Python code examples that:

1. Are VARIATIONS or EXTENSIONS of the patterns shown above
2. Target subtle type system edge cases likely to cause checker disagreements
3. Are self-contained and runnable (include all imports)
4. Have a minimal `if __name__ == "__main__":` block

## KNOWN DIVERGENCE PATTERNS TO TARGET:

{patterns_text}

## STRATEGY FOR GENERATING DIVERGENCES:

- If a seed shows a false positive in mypy, try to create a similar case that other checkers also get wrong
- If a seed shows a false negative, create variations that test the boundaries of what gets caught
- Combine patterns: e.g., TypedDict + Protocol, ParamSpec + classmethod
- Modify the seeds slightly: change types, add generics, wrap in decorators

## OUTPUT FORMAT:

For each example, use this exact format:

# id: <short-kebab-name>
# category: <pattern category>
# seed_issue: <repo>#<issue_number> (REQUIRED - must reference a real seed issue from above)

```python
<your code here>
```

IMPORTANT RULES:
- Generate exactly {num_variations} examples
- EVERY example MUST have a valid seed_issue referencing one of the GitHub issues shown above
- Do NOT use "original" - all examples must be inspired by a real issue
- Focus on NOVEL variations, not copies of the seeds
- Distribute examples across different seed issues (not all from the same one)
"""


def build_expert_prompt(num_examples: int = 10) -> str:
    """
    Fallback prompt when no GitHub examples are available.
    Uses pattern descriptions to guide generation.
    """
    patterns_text = "\n".join(
        f"- **{p.id}** ({p.category}): {p.description} [refs: {', '.join(p.pep_refs)}]"
        for p in PATTERNS
    )

    return f"""You are an expert in Python type systems and type checker implementation differences.
Generate exactly {num_examples} Python code snippets that demonstrate REAL divergences between
mypy, pyrefly, zuban, and ty type checkers.

CRITICAL REQUIREMENTS:
1. Each snippet must be SELF-CONTAINED and RUNNABLE
2. Use ONLY valid Python 3.12+ syntax and REAL typing features
3. NO forward reference issues - use string annotations if needed: `-> "ClassName"`
4. Each snippet must target a SPECIFIC type checker divergence area
5. Include imports from `typing` and `typing_extensions` as needed
6. Include a minimal `if __name__ == "__main__":` block

TARGET THESE DIVERGENCE PATTERNS:
{patterns_text}

OUTPUT FORMAT:
For each example, use this exact format:

# id: <short-kebab-case-name>
# category: <category from list above>

```python
<your code here>
```

Generate exactly {num_examples} examples covering different patterns from the list above.
Ensure all examples have valid Python syntax.
Focus on subtle type system edge cases where checkers genuinely disagree.
"""


def build_refinement_prompt(
    code: str,
    checker_results: dict[str, str],
    seed_example: IssueExample | None = None,
) -> str:
    """
    Build a prompt to refine code that didn't cause a disagreement.
    Args:
        code: The current code that all checkers agreed on
        checker_results: Dict mapping checker name to its output summary
        seed_example: Optional original seed example for context
    """

    results_text = "\n".join(
        f"- {name}: {output[:200]}"
        for name, output in checker_results.items()
    )

    seed_context = ""
    if seed_example:
        seed_context = f"""
This was inspired by a real bug from {seed_example.repo} Issue #{seed_example.issue_number}:
- Title: {seed_example.issue_title}
- Was labeled as: {'false positive' if seed_example.is_false_positive else 'false negative' if seed_example.is_false_negative else 'bug'}
"""

    return f"""The following Python code was tested with all 4 type checkers (mypy, pyrefly, zuban, ty)
but they ALL AGREED - meaning this is NOT a useful divergence example.
{seed_context}
## CURRENT CODE:
```python
{code}
```

## ACTUAL CHECKER RESULTS (all agree):
{results_text}

## YOUR TASK:

Modify this code MINIMALLY to create a REAL divergence where at least one checker
disagrees with the others.

STRATEGIES:
1. If all passed: Add a subtle type error that only some checkers catch
2. If all failed: Fix the obvious error but keep a subtle edge case
3. Change the typing pattern slightly (add Protocol, use TypeGuard, add overloads)
4. Wrap in a decorator with ParamSpec
5. Use TypedDict with Required/NotRequired inheritance

## REQUIREMENTS:
- Keep it self-contained and runnable
- Use valid Python 3.12+ syntax
- Target a real type system ambiguity, not just syntax tricks

## OUTPUT:
Provide ONLY the modified code in this format:

# id: <name>-refined
# category: refined

```python
<your modified code>
```
"""

