"""
Test _checker_reports_error against real checker outputs from
generated_examples/2026-02-20_13-08-02/results.json.

For each example we compare our parser result against the ground-truth
`statuses` recorded at generation time.
"""
import json, pathlib, sys, re

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from comprehensive_eval import _checker_reports_error as comp_reports_error
from static_tier4 import _checker_reports_error as tier4_reports_error

RESULTS_FILE = ROOT / "generated_examples" / "2026-02-20_13-08-02" / "results.json"

def main():
    data = json.loads(RESULTS_FILE.read_text())
    results = data["results"]

    total = 0
    pass_count = 0
    fail_count = 0
    failures = []

    for entry in results:
        filename = entry["filename"]
        for checker, output in entry["outputs"].items():
            expected_error = entry["statuses"][checker] == "error"

            comp_result = comp_reports_error(output, checker)
            tier4_result = tier4_reports_error(output, checker)

            for label, got in [("comprehensive_eval", comp_result), ("static_tier4", tier4_result)]:
                total += 1
                if got == expected_error:
                    pass_count += 1
                else:
                    fail_count += 1
                    failures.append({
                        "file": filename,
                        "checker": checker,
                        "module": label,
                        "expected_error": expected_error,
                        "got_error": got,
                        "output_preview": output[:120],
                    })

    print(f"\n{'='*60}")
    print("Checker output parsing test results")
    print(f"{'='*60}")
    print(f"Total checks : {total}")
    print(f"Passed        : {pass_count}")
    print(f"Failed        : {fail_count}")

    if failures:
        print("\n--- FAILURES ---")
        for f in failures:
            print(f"\n  File    : {f['file']}")
            print(f"  Checker : {f['checker']}")
            print(f"  Module  : {f['module']}")
            print(f"  Expected: error={f['expected_error']}")
            print(f"  Got     : error={f['got_error']}")
            print(f"  Output  : {f['output_preview']}...")
    else:
        print("\nAll checks passed!")

    return fail_count == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

