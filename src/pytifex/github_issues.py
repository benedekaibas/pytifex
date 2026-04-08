"""
Fetching GitHub issues for type checker repositories.
Extracts code examples from closed issues (especially false positives/negatives).
"""

import os
import re
import random
import httpx
from dataclasses import dataclass
from typing import Optional


@dataclass
class IssueExample:
    repo: str
    issue_number: int
    issue_title: str
    issue_url: str
    code: str
    labels: list[str]
    is_false_positive: bool
    is_false_negative: bool


# Note: We fetch from multiple repos to get diverse bug patterns
REPOS = {
    "mypy": "python/mypy",
    "pyrefly": "facebook/pyrefly",
    "ty": "astral-sh/ty",
    "pyright": "microsoft/pyright",  # Additional source for diverse patterns
    "zuban": "zubanls/zuban",
}

# Labels that indicate false positive/negative bugs
FALSE_POSITIVE_LABELS = [
    "false-positive",
    "false positive",
    "false-alarm",
    "spurious",
    "incorrect-error",
    "bug: false positive",
]

FALSE_NEGATIVE_LABELS = [
    "false-negative",
    "false negative",
    "missed-error",
    "should-error",
    "bug: false negative",
]

BUG_LABELS = [
    "bug",
    "topic-type-inference",
    "topic-protocols",
    "topic-generics",
    "topic-typeddict",
    "topic-overloads",
    "topic-paramspec",
    "topic-typeguard",
]


def extract_pyrefly_sandbox_code(text: str) -> list[str]:
    """
    Extract Python code from pyrefly sandbox links.
    Pyrefly issues often contain links like:
    https://pyrefly.org/sandbox/?project=<base64_encoded_data>
    The project parameter contains base64-encoded JSON with the code.
    """
    import base64
    import json
    import urllib.parse

    codes = []

    # Find pyrefly sandbox URLs (Only Pyrefly has sandbox URLs for code snippets)
    sandbox_pattern = r'https://pyrefly\.org/sandbox/\?project=([A-Za-z0-9%+/=]+)'
    matches = re.findall(sandbox_pattern, text)

    for encoded in matches:
        try:
            # Decoding URL first
            decoded_url = urllib.parse.unquote(encoded)
            # Decoding Base64
            decoded_bytes = base64.b64decode(decoded_url)
            decoded_str = decoded_bytes.decode('utf-8')

            # Try to parse as JSON (pyrefly format)
            try:
                data = json.loads(decoded_str)
                # Extract code from various possible structures
                if isinstance(data, dict):
                    for key in ['code', 'source', 'content', 'files']:
                        if key in data:
                            value = data[key]
                            if isinstance(value, str) and len(value) > 50:
                                codes.append(value)
                            elif isinstance(value, dict):
                                # Files dict: {filename: content}
                                for content in value.values():
                                    if isinstance(content, str) and len(content) > 50:
                                        codes.append(content)
            except json.JSONDecodeError:
                # Not JSON, so it might be plain code
                if len(decoded_str) > 50 and ('def ' in decoded_str or 'class ' in decoded_str):
                    codes.append(decoded_str)
        except Exception:
            # Failed to decode, skip this sandbox link and issue.
            continue

    return codes


def extract_python_code(text: str) -> list[str]:
    """Extract Python code blocks from markdown text."""
    codes = []

    # Try to extract from pyrefly sandbox links
    sandbox_codes = extract_pyrefly_sandbox_code(text)
    codes.extend(sandbox_codes)

    pattern = r"```(?:python|py)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    codes.extend(matches)

    # Try to find indented code blocks after "Example:" or similar
    if not codes:
        pattern = r"(?:Example|Code|Reproduce|MRE|Minimal).*?:\n```\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        codes.extend(matches)

    # Filter out very short snippets (likely not useful) and deduplicates
    seen = set()
    result = []
    for code in codes:
        code = code.strip()
        if len(code) > 50 and code not in seen:
            seen.add(code)
            result.append(code)

    return result


def fetch_issues(
    repo: str,
    labels: list[str] | None = None,
    state: str = "closed",
    per_page: int = 30,
    max_pages: int = 3,
) -> list[dict]:
    """Fetch issues from a GitHub repository."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_issues = []

    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "direction": "desc",
        }
        if labels:
            params["labels"] = ",".join(labels)

        try:
            resp = httpx.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            issues = resp.json()

            if not issues:
                break

            all_issues.extend(issues)
        except Exception as e:
            print(f"  Warning: Failed to fetch from {repo}: {e}")
            break

    return all_issues


def get_issue_body(repo: str, issue_number: int) -> str:
    """Fetch the full body of a specific issue."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

    try:
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get("body", "")
    except Exception:
        return ""


def classify_issue(labels: list[str]) -> tuple[bool, bool]:
    """Classify if an issue is a false positive or false negative."""
    label_names = [l.get("name", "").lower() for l in labels]

    is_fp = any(
        any(fp in name for fp in FALSE_POSITIVE_LABELS)
        for name in label_names
    )
    is_fn = any(
        any(fn in name for fn in FALSE_NEGATIVE_LABELS)
        for name in label_names
    )

    return is_fp, is_fn


def is_confirmed_bug(issue: dict) -> bool:
    """
    Check if an issue is a confirmed, fixed bug.
    Filters out:
    - Issues closed as "not_planned" (won't fix, not a bug, etc.)
    - Issues that are still open
    Only accepts:
    - Issues with state_reason: "completed" (confirmed and fixed)
    """
    state = issue.get("state", "")
    state_reason = issue.get("state_reason", "")

    # Must be closed because it means it is resolved
    if state != "closed":
        return False

    # Must be completed (not "not_planned", it is important)
    # NOTE: state_reason can be "completed", "not_planned", or null
    # We accept "completed" and null (older issues don't have state_reason)
    if state_reason == "not_planned":
        return False

    return True


def fetch_examples_from_repo(
    checker_name: str,
    repo: str,
    max_examples: int = 10,
    prefer_bugs: bool = True,
) -> list[IssueExample]:
    """Fetch code examples from a type checker's GitHub issues."""
    examples = []

    print(f"  Fetching issues from {repo}...")

    # Get bug-labeled issues first since they are the interesting cases
    if prefer_bugs:
        issues = fetch_issues(repo, labels=["bug"], per_page=50, max_pages=2)
    else:
        issues = fetch_issues(repo, per_page=50, max_pages=2)

    if not issues:
        print(f"  No issues found in {repo}")
        return []

    # Filter to only confirmed bugs (not "won't fix" or "not planned" since those do not matter)
    confirmed_issues = [i for i in issues if is_confirmed_bug(i)]
    skipped_count = len(issues) - len(confirmed_issues)
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} issues (not_planned/won't fix)")

    # Shuffle to get variety
    random.shuffle(confirmed_issues)

    for issue in confirmed_issues:
        if len(examples) >= max_examples:
            break

        issue_labels = issue.get("labels", [])
        is_fp, is_fn = classify_issue(issue_labels)

        # Get the issue body
        body = issue.get("body", "") or ""
        if len(body) < 100:
            # Might be shortened, so fetch full body
            body = get_issue_body(repo, issue["number"])

        # Extract code examples
        code_blocks = extract_python_code(body)

        for code in code_blocks:
            if len(examples) >= max_examples:
                break

            examples.append(IssueExample(
                repo=repo,
                issue_number=issue["number"],
                issue_title=issue.get("title", ""),
                issue_url=issue.get("html_url", ""),
                code=code,
                labels=[l.get("name", "") for l in issue_labels],
                is_false_positive=is_fp,
                is_false_negative=is_fn,
            ))

    print(f"  Found {len(examples)} code examples from {repo}")
    return examples


def fetch_random_examples(
    max_per_repo: int = 5,
    checkers: list[str] | None = None,
) -> list[IssueExample]:
    """
    Fetch random code examples from type checker GitHub issues.
    Args:
        max_per_repo: Maximum examples to fetch per repository
        checkers: List of checker names to fetch from (default: all)
    Returns:
        List of IssueExample objects with code from real bug reports
    """
    all_examples = []

    repos_to_check = REPOS
    if checkers:
        repos_to_check = {k: v for k, v in REPOS.items() if k in checkers}

    print("Fetching examples from type checker GitHub issues...")

    for checker_name, repo in repos_to_check.items():
        examples = fetch_examples_from_repo(
            checker_name, repo, max_examples=max_per_repo
        )
        all_examples.extend(examples)

    random.shuffle(all_examples)

    print(f"Total: {len(all_examples)} examples from GitHub issues")
    return all_examples


def format_example_for_prompt(example: IssueExample) -> str:
    """Format a GitHub issue example for use in a prompt."""
    labels_str = ", ".join(example.labels[:5]) if example.labels else "none"

    return f"""### Example from {example.repo} (Issue #{example.issue_number})
Title: {example.issue_title}
Labels: {labels_str}
False Positive: {example.is_false_positive} | False Negative: {example.is_false_negative}

```python
{example.code}
```
"""

