# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mypy==1.19.0",
#     "pyrefly==0.44.2",
#     "zuban==0.3.0",
#     "ty==0.0.1-alpha.32",
#     "pydantic",
#     "httpx",
# ]
# ///

"""
Pytifex - Type Checker Disagreement Analysis Pipeline

This tool generates Python code examples that cause disagreements between Python
type checkers (mypy, pyrefly, zuban, ty) and evaluates which checker is correct.

Key improvement: Uses type checkers as ground truth, not LLM predictions.
Only keeps examples where checkers actually disagree.
"""

import argparse
import sys

from run_checkers import run_checkers
from eval import evaluate_results
from pipeline import generate_with_filtering


def main():
    parser = argparse.ArgumentParser(
        description="Pytifex - Type Checker Disagreement Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py                    Run full pipeline (generate + filter + eval)
  uv run main.py full               Run full pipeline
  uv run main.py generate           Only generate examples (with filtering)
  uv run main.py check              Only run type checkers on existing examples
  uv run main.py eval               Only evaluate existing results

Options:
  uv run main.py --num-examples 5   Generate until 5 disagreements found
  uv run main.py --model gemini-2.5-pro --num-examples 20 full
  uv run main.py -v full            Verbose output showing all examples
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="full",
        choices=["full", "generate", "check", "eval"],
        help="Command to run (default: full)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        choices=["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"],
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Target number of DISAGREEMENT examples (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Examples to generate per batch (default: 15)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum generation attempts (default: 5)",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=2,
        help="Max refinement attempts per non-divergent example (default: 2)",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Skip fetching seeds from GitHub issues (use pattern-based generation only)",
    )
    parser.add_argument(
        "--eval-method",
        choices=["comprehensive", "multi_step", "consensus", "runtime", "all"],
        default="comprehensive",
        help="Evaluation method (default: comprehensive = 4-tier runtime/mutation/PEP evaluation)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    try:
        if args.command == "full":
            print("=" * 60)
            print("PYTIFEX - Full Pipeline (with disagreement filtering)")
            print("=" * 60)

            print("\n[STEP 1/2] Generating examples with disagreement filtering...")
            disagreements, base_path = generate_with_filtering(
                model=args.model,
                target_count=args.num_examples,
                max_attempts=args.max_attempts,
                batch_size=args.batch_size,
                max_refinements=args.max_refinements,
                verbose=args.verbose,
                use_github_seeds=not args.no_github,
            )

            if not disagreements:
                print("[WARNING] No disagreements found. Try increasing --max-attempts or --batch-size")
                sys.exit(0)

            print(f"\n[STEP 2/2] Evaluating {len(disagreements)} disagreements...")
            results_path = f"{base_path}/results.json"

            if args.eval_method == "comprehensive":
                from comprehensive_eval import evaluate_results_comprehensive
                evaluate_results_comprehensive(results_path)
                eval_path = f"{base_path}/evaluation_comprehensive.json"
            else:
                eval_path = evaluate_results(
                    results_path, method=args.eval_method, verbose=args.verbose
                )

            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print(f"Disagreements found: {len(disagreements)}")
            print(f"Output directory: {base_path}")
            print(f"Evaluation: {eval_path}")

        elif args.command == "generate":
            disagreements, base_path = generate_with_filtering(
                model=args.model,
                target_count=args.num_examples,
                max_attempts=args.max_attempts,
                batch_size=args.batch_size,
                max_refinements=args.max_refinements,
                verbose=args.verbose,
                use_github_seeds=not args.no_github,
            )
            print(f"\n[SUCCESS] {len(disagreements)} disagreements saved to: {base_path}")

        elif args.command == "check":
            results_path = run_checkers()
            print(f"\n[SUCCESS] Results saved to: {results_path}")

        elif args.command == "eval":
            # Find the latest results file
            from config import BASE_GEN_DIR
            import glob
            results_files = sorted(glob.glob(f"{BASE_GEN_DIR}/*/results.json"))
            if not results_files:
                print("[ERROR] No results.json found. Run 'generate' first.")
                sys.exit(1)
            results_path = results_files[-1]

            if args.eval_method == "comprehensive":
                from comprehensive_eval import evaluate_results_comprehensive
                evaluate_results_comprehensive(results_path)
                eval_path = results_path.replace("results.json", "evaluation_comprehensive.json")
            else:
                eval_path = evaluate_results(
                    results_path, method=args.eval_method, verbose=args.verbose
                )
            print(f"\n[SUCCESS] Evaluation saved to: {eval_path}")

    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

