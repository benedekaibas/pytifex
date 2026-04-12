"""
The pipeline uses type checkers as ground truth.

How it works: Fetch REAL bug reports from type checker GitHub issues,
use them as seeds, and generate variations that might cause divergences.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from pydantic import HttpUrl

from .agent import GetAccessToGemini
from .config import BASE_GEN_DIR, CHECKERS
from .prompts import build_seed_based_prompt, build_expert_prompt, build_refinement_prompt
from .github_issues import fetch_random_examples, IssueExample
from .code_metrics import compute_metrics, metrics_to_dict
from . import generate_json


@dataclass
class CheckerResult:
    status: str  # "ok" or "error"
    output: str


@dataclass
class Example:
    id: str
    code: str
    metadata: str
    results: dict[str, CheckerResult] | None = None
    seed_issue: str | None = None  # GitHub issue URL or "original"


def extract_seed_issue(metadata: str) -> str | None:
    """Extract seed_issue URL or reference from metadata."""
    # Look for "# seed_issue: ..." line
    match = re.search(r'#\s*seed_issue:\s*(.+)', metadata, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        # If it's a repo/issue reference like "python/mypy#12345", convert to URL
        repo_issue_match = re.match(r'(\w+/\w+)#(\d+)', value)
        if repo_issue_match:
            repo, issue_num = repo_issue_match.groups()
            return f"https://github.com/{repo}/issues/{issue_num}"
        # If it's already a URL then return as-is
        return value if value.lower() != "original" else None
    return None


def run_checker_on_code(code: str, checker_name: str, command: list[str]) -> CheckerResult:
    """Run a single type checker on code and return the result."""
    # Use current directory for temp files since zuban doesn't work with /tmp/ paths
    temp_filename = f"_pytifex_temp_{os.getpid()}.py"
    temp_path = os.path.join(os.getcwd(), temp_filename)

    with open(temp_path, "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            command + [temp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr

        # Determine status based on return code and output
        if result.returncode == 0:
            status = "ok"
        else:
            status = "error"

        # Some checkers return 0 but still report errors
        output_lower = output.lower()
        if "error" in output_lower and "0 error" not in output_lower:
            status = "error"

        return CheckerResult(status=status, output=output.strip())
    except subprocess.TimeoutExpired:
        return CheckerResult(status="error", output="Timeout")
    except Exception as e:
        return CheckerResult(status="error", output=str(e))
    finally:
        os.unlink(temp_path)


def run_all_checkers(code: str) -> dict[str, CheckerResult]:
    """Run all type checkers on code and return results."""
    results = {}
    for name, command in CHECKERS.items():
        results[name] = run_checker_on_code(code, name, command)
    return results


def has_disagreement(results: dict[str, CheckerResult]) -> bool:
    """Check if at least one checker disagrees with the others."""
    statuses = [r.status for r in results.values()]
    return len(set(statuses)) > 1


def summarize_results(results: dict[str, CheckerResult]) -> str:
    """Create a summary of checker results to regenerate prompt."""
    lines = []
    for name, result in results.items():
        first_line = result.output.split("\n")[0][:100] if result.output else "no output"
        lines.append(f"{name}: {result.status} - {first_line}")
    return "\n".join(lines)


def build_refinement_prompt_for_example(example: Example) -> str:
    """Build a prompt to regenerate a non-divergent example."""
    results_dict = {
        name: result.output for name, result in (example.results or {}).items()
    }
    return build_refinement_prompt(example.code, results_dict, None)


def generate_with_filtering(
    model: str = "gemini-2.5-flash",
    target_count: int = 10,
    max_attempts: int = 5,
    batch_size: int = 15,
    max_refinements: int = 2,
    verbose: bool = False,
    use_github_seeds: bool = True,
) -> tuple[list[Example], str]:
    """
    Generate examples until we have `target_count` actual disagreements.
    Strategy:
    1. Fetch real bug reports from type checker GitHub issues
    2. Use them as seeds to generate variations
    3. Run all 4 type checkers on each variation
    4. Keep only examples where checkers disagree
    5. Refine non-divergent examples with feedback
    Returns:
        Tuple of (list of examples with disagreements, output directory path)
    """
    token = os.environ.get("GEMINI_API_KEY")
    if not token:
        raise ValueError("Please set GEMINI_API_KEY environment variable")

    agent = GetAccessToGemini(
        model=model,
        token=token,
        api_base=HttpUrl("https://generativelanguage.googleapis.com/v1beta"),
        timeout=320.0,
    )

    collected: list[Example] = []
    all_generated: list[Example] = []
    seed_examples: list[IssueExample] = []
    attempt = 0

    print(f"Target: {target_count} disagreement examples")
    print(f"Using model: {model}")
    print("=" * 60)

    # Fetch seed examples from GitHub
    if use_github_seeds:
        print("\n[STEP 0] Fetching seed examples from GitHub issues...")
        try:
            seed_examples = fetch_random_examples(max_per_repo=5)
        except Exception as e:
            print(f"  Warning: Could not fetch GitHub issues: {e}")
            print("  Falling back to pattern-based generation")
            seed_examples = []

    while len(collected) < target_count and attempt < max_attempts:
        attempt += 1
        print(f"\n[Attempt {attempt}/{max_attempts}] Generating batch of {batch_size}...")

        # Build prompt (prefer seed-based if we have seeds)
        if seed_examples:
            # Rotate through seeds for variety
            start_idx = (attempt - 1) * 3 % len(seed_examples)
            batch_seeds = seed_examples[start_idx:start_idx + 5]
            if len(batch_seeds) < 3:
                batch_seeds = seed_examples[:5]
            prompt = build_seed_based_prompt(batch_seeds, batch_size)
            print(f"  Using {len(batch_seeds)} GitHub issue seeds")
        else:
            prompt = build_expert_prompt(batch_size)
            print("  Using pattern-based generation (no seeds)")

        response = agent.predict(prompt)

        parsed = generate_json.parse_generated_content(response)
        if not parsed:
            print("  ⚠️  No examples parsed from response")
            continue

        print(f"  Parsed {len(parsed)} examples, running type checkers...")

        # Test each example
        for item in parsed:
            metadata = item.get("metadata", "")
            seed_issue = extract_seed_issue(metadata)

            # Skip examples without a valid seed_issue (only for guided runs)
            if seed_issue is None and use_github_seeds:
                if verbose:
                    print(f"  ⚠️  {item['id']}: SKIPPED (no seed_issue)")
                continue

            example = Example(
                id=item["id"],
                code=item["code"],
                metadata=metadata,
                seed_issue=seed_issue,
            )
            all_generated.append(example)

            # Run checkers
            example.results = run_all_checkers(example.code)

            if has_disagreement(example.results):
                collected.append(example)
                statuses = {k: v.status for k, v in example.results.items()}
                print(f"  ✓ {example.id}: DISAGREEMENT {statuses}")
            else:
                if verbose:
                    statuses = {k: v.status for k, v in example.results.items()}
                    print(f"  ✗ {example.id}: all agree {statuses}")

                # Try refinement
                if max_refinements > 0:
                    refined = refine_example(agent, example, max_refinements, verbose)
                    if refined:
                        collected.append(refined)
                        statuses = {k: v.status for k, v in refined.results.items()}
                        print(f"  ✓ {refined.id}: DISAGREEMENT (refined) {statuses}")

        print(f"  Progress: {len(collected)}/{target_count} disagreements found")

    print("\n" + "=" * 60)
    print(f"GENERATION COMPLETE: {len(collected)} disagreements from {len(all_generated)} total examples")

    # Save results
    output_dir = save_disagreements(collected, all_generated, model)

    return collected, output_dir


def refine_example(
    agent: GetAccessToGemini,
    example: Example,
    max_attempts: int,
    verbose: bool = False,
) -> Optional[Example]:
    """Try to refine an example to create a disagreement."""
    current = example

    for i in range(max_attempts):
        if verbose:
            print(f"    Regenerating {example.id} (attempt {i+1}/{max_attempts})...")

        prompt = build_refinement_prompt_for_example(current)

        try:
            response = agent.predict(prompt)
        except Exception as e:
            if verbose:
                print(f"    Regeneration failed: {e}")
            return None

        # Parse the regenerated code
        parsed = generate_json.parse_generated_content(response)
        if not parsed:
            # Try to extract code directly from markdown
            import re
            match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                parsed = [{"id": f"{example.id}-refined", "code": code, "metadata": ""}]

        if not parsed:
            continue

        refined = Example(
            id=parsed[0]["id"],
            code=parsed[0]["code"],
            metadata=parsed[0].get("metadata", ""),
            seed_issue=example.seed_issue,
        )
        refined.results = run_all_checkers(refined.code)

        if has_disagreement(refined.results):
            return refined

        current = refined

    return None


def save_disagreements(
    disagreements: list[Example],
    all_examples: list[Example],
    model: str,
) -> str:
    """Save the disagreement examples to disk."""
    import datetime
    import json

    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(BASE_GEN_DIR, folder_name)
    source_files_path = os.path.join(base_path, "source_files")
    os.makedirs(source_files_path, exist_ok=True)

    print(f"\n[INFO] Saving to: {base_path}")

    # Save only disagreement examples as .py files
    for ex in disagreements:
        filename = f"{ex.id}.py"
        file_path = os.path.join(source_files_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(ex.code)
        print(f"  -> Saved {filename}")

    # Save detailed results JSON
    results_data = {
        "timestamp": now.isoformat(),
        "model_used": model,
        "total_generated": len(all_examples),
        "disagreements_found": len(disagreements),
        "success_rate": f"{len(disagreements)/max(len(all_examples),1)*100:.1f}%",
        "checkers_used": list(CHECKERS.keys()),
        "results": [
            {
                "filename": f"{ex.id}.py",
                "filepath": os.path.join(source_files_path, f"{ex.id}.py"),
                "seed_issue": ex.seed_issue,
                "metrics": metrics_to_dict(compute_metrics(ex.code)),
                "outputs": {
                    name: result.output
                    for name, result in (ex.results or {}).items()
                },
                "statuses": {
                    name: result.status
                    for name, result in (ex.results or {}).items()
                },
            }
            for ex in disagreements
        ],
    }

    results_path = os.path.join(base_path, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"[INFO] Saved results.json with {len(disagreements)} disagreements")

    return base_path

