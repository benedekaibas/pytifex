import asyncio
import os
import json
import httpx
import random
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List, Set

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic import BaseModel

# --- 1. AUTHENTICATION ---
gemini_key = os.getenv("GEMINI_API_KEY")
github_pat = os.getenv("GITHUB_PAT")

if not gemini_key:
    print("❌ ERROR: GEMINI_API_KEY is missing. Run: export GEMINI_API_KEY='AIzaSy...'")
    exit(1)
if not github_pat:
    print("❌ ERROR: GITHUB_PAT is missing. Run: export GITHUB_PAT='ghp_...'")
    exit(1)

provider = GoogleProvider(api_key=gemini_key)
model = GoogleModel('gemini-2.0-flash', provider=provider)

OUTPUT_FILE = "differential_test_cases.json"

TARGET_REPOS = [
    "https://github.com/python/mypy",          
    "https://github.com/microsoft/pyright",    
    "https://github.com/facebook/pyre-check",  
    "https://github.com/zubanls/zuban",        
    "https://github.com/astral-sh/ty"          
]

SEARCH_KEYWORDS = [
    "Protocol", "TypeGuard", "TypedDict", "ParamSpec", 
    "Generic", "Variance", "Overload", "Recursive"
]

# --- 2. DATA STRUCTURES ---
class DifferentialCase(BaseModel):
    id: str
    source_issue: str
    original_code: str
    tweaked_code: str          
    mutation_strategy: str     
    checker_outputs: dict = field(default_factory=dict) 
    analysis: str = "Pending"

@dataclass
class FuzzerState:
    cases: List[DifferentialCase] = field(default_factory=list)
    visited_issues: Set[str] = field(default_factory=set)

# --- 3. HELPERS ---
def extract_slug(url: str) -> str:
    if "github.com/" not in url: return url.strip()
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else url

def save_results_to_disk(state: FuzzerState):
    data = [c.model_dump() for c in state.cases]
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    # We don't print "Saved" here anymore to keep the terminal clean for code display

# --- 4. AGENT & TOOLS ---

def system_prompt() -> str:
    return """You are an **Advanced Python Type System Fuzzer**.

YOUR GOAL: 
1. Use `search_seeds` to find a bug in the requested repo.
2. Read the reproduction code from the tool output.
3. Apply a **Mutation Strategy** to make it syntactically valid but confusing for type checkers.
4. Call `generate_mutation`.

### ⛔ STRICT RULES
* **DO NOT** search repositories other than the one requested.
* **DO NOT** create toy examples (like simple Dicts). Use complex features (Generics, Protocols).

### MUTATION STRATEGIES (Pick one)
1. **Contravariant Protocol:** Mix Protocol + contravariant TypeVars.
2. **Self Type Paradox:** `def method(self) -> Self` conflicts.
3. **Overload Explosion:** `@overload` with Unions.
4. **Recursive Generics:** Infinite type recursion.
"""

agent = Agent(model, deps_type=FuzzerState, system_prompt=system_prompt())

async def fetch_issue_comments(client: httpx.AsyncClient, issue_url: str, headers: dict) -> str:
    """Fetches top comments to find hidden reproduction code."""
    try:
        # Convert HTML URL to API URL
        if "github.com" in issue_url and "api.github.com" not in issue_url:
            parts = urlparse(issue_url).path.strip("/").split("/")
            if len(parts) == 4:
                comments_url = f"https://api.github.com/repos/{parts[0]}/{parts[1]}/issues/{parts[3]}/comments"
            else:
                return ""
        elif "api.github.com" in issue_url:
            comments_url = f"{issue_url}/comments"
        else:
            return ""

        resp = await client.get(comments_url, headers=headers)
        if resp.status_code == 200:
            comments = resp.json()
            if not comments: return ""
            # Pick a random comment from top 3 to ensure variety
            selected = random.choice(comments[:3])
            return f"Comment by {selected['user']['login']}:\n{selected['body'][:1000]}"
    except:
        return ""
    return ""

@agent.tool
async def search_seeds(ctx: RunContext[FuzzerState], repo_slug: str) -> str:
    """
    Searches for bugs, picks a RANDOM one, and fetches its content.
    """
    cleaned_slug = extract_slug(repo_slug)
    keyword = random.choice(SEARCH_KEYWORDS)
    print(f"  🔍 Mining {cleaned_slug} for '{keyword}'...")
    
    headers = {"Authorization": f"token {github_pat}", "Accept": "application/vnd.github.v3+json"}
    
    # We fetch 10 results so we can pick a random one
    query = f"repo:{cleaned_slug} is:issue is:closed label:bug {keyword} code:python"
    url = f"https://api.github.com/search/issues?q={query}&per_page=10&sort=updated"

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200: return f"API Error: {resp.status_code}"
            data = resp.json()
            
            items = data.get("items", [])
            if not items: return f"No bugs found for '{keyword}'. Stop."

            # RANDOM SELECTION: Pick one random issue from the results
            item = random.choice(items)
            
            url = item["html_url"]
            body = item["body"] or ""
            comments = await fetch_issue_comments(client, url, headers)
            
            full_content = f"DESCRIPTION:\n{body[:1500]}\n\nCOMMENTS:\n{comments}"
            return f"URL: {url}\nCONTENT:\n{full_content}"

        except Exception as e: return f"Network Error: {e}"

@agent.tool
def generate_mutation(ctx: RunContext[FuzzerState], 
                      source_url: str, original_code: str, 
                      mutated_code: str, strategy_name: str) -> str:
    """Saves the mutation and PRINTS it to the console."""
    try:
        parts = source_url.split('/')
        if "issues" in parts:
            idx = parts.index("issues")
            uid = f"{parts[idx-1]}-{parts[idx+1]}-mutated"
        else:
            uid = f"synthetic-{len(ctx.deps.cases)}"
    except: uid = f"unknown-{len(ctx.deps.cases)}"
    
    case = DifferentialCase(
        id=uid, source_issue=source_url, original_code=original_code,
        tweaked_code=mutated_code, mutation_strategy=strategy_name
    )
    ctx.deps.cases.append(case)
    
    # --- VISIBILITY: PRINT THE CODE ---
    print("\n" + "█"*60)
    print(f"🐍 GENERATED CODE ({uid})")
    print(f"📝 Strategy: {strategy_name}")
    print("█"*60 + "\n")
    print(mutated_code)
    print("\n" + "█"*60 + "\n")
    
    return "Saved mutation."

# --- 5. EXECUTION LOOP ---

async def run_with_retry(prompt: str, state: FuzzerState, retries=3):
    for attempt in range(retries):
        try:
            await agent.run(prompt, deps=state)
            return
        except Exception as e:
            if "429" in str(e):
                wait_time = 30 * (attempt + 1)
                print(f"   🛑 Rate Limit (429). Sleeping {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                # If error is not 429, we just print and skip to next repo
                print(f"   ⚠️ Agent Error: {e}")
                return

async def main():
    state = FuzzerState()
    print("="*60)
    print("🧬 STARTING VISIBLE FUZZER (3 Examples)")
    print("="*60)
    
    # RANDOM SELECTION: Pick exactly 3 repositories to target
    selected_repos = random.sample(TARGET_REPOS, k=3)
    
    for i, repo_url in enumerate(selected_repos, 1):
        slug = extract_slug(repo_url)
        print(f"\n👉 Target {i}/3: {slug}")
        
        prompt = f"""
        TASK:
        1. Call `search_seeds('{slug}')`.
        2. Analyze the code found.
        3. Apply a 'Complex Mutation Strategy'.
        4. Call `generate_mutation`.
        """
        
        await run_with_retry(prompt, state)
        save_results_to_disk(state)

        # Pause to prevent rate limits
        if i < 3:
            print("   💤 Cooling down (3s)...")
            await asyncio.sleep(3)
    
    print("\n✅ DONE. File saved: " + OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())
