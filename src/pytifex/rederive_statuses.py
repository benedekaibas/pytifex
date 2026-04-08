"""
Re-derive `statuses` in all results.json files using the new
checker-specific parsers.

Modes:
  --dry-run   (default) Report misclassifications without modifying files.
  --apply     Overwrite results.json files with corrected statuses.
"""
import json, pathlib, re, sys, glob as globmod

ROOT = pathlib.Path(__file__).resolve().parent


def checker_reports_error(output: str, checker: str) -> bool:
    checker = checker.lower()

    if checker in ("mypy", "zuban"):
        if "success: no issues found" in output.lower():
            return False
        m = re.search(r"Found\s+(\d+)\s+errors?\s+in", output)
        if m:
            return int(m.group(1)) > 0
        for line in output.splitlines():
            if re.search(r":\s*error\b", line, re.IGNORECASE):
                return True
        return False

    if checker == "pyrefly":
        m = re.search(r"INFO\s+(\d+)\s+errors?", output)
        if m:
            return int(m.group(1)) > 0
        for line in output.splitlines():
            if line.strip().startswith("ERROR"):
                return True
        return False

    if checker == "ty":
        if "all checks passed" in output.lower():
            return False
        for line in output.splitlines():
            if re.match(r"\s*error\[", line, re.IGNORECASE):
                return True
        return False

    lo = output.lower()
    return "error" in lo and "0 error" not in lo and "success" not in lo


def process_file(path: pathlib.Path, apply: bool) -> dict:
    data = json.loads(path.read_text())
    mismatches = []

    for entry in data.get("results", []):
        for checker, output in entry.get("outputs", {}).items():
            old_status = entry.get("statuses", {}).get(checker)
            new_error = checker_reports_error(output, checker)
            new_status = "error" if new_error else "ok"

            if old_status != new_status:
                mismatches.append({
                    "file": entry.get("filename", "?"),
                    "checker": checker,
                    "old": old_status,
                    "new": new_status,
                    "output_preview": output[:120],
                })
                if apply:
                    if "statuses" not in entry:
                        entry["statuses"] = {}
                    entry["statuses"][checker] = new_status

    if apply and mismatches:
        path.write_text(json.dumps(data, indent=2) + "\n")

    return {"path": str(path.relative_to(ROOT)), "mismatches": mismatches}


def main():
    apply = "--apply" in sys.argv
    mode = "APPLY" if apply else "DRY-RUN"

    pattern = str(ROOT / "generated_examples" / "*" / "results.json")
    files = sorted(globmod.glob(pattern))

    total_mismatches = 0
    total_files_affected = 0

    print(f"{'='*70}")
    print(f"Re-derive statuses ({mode})")
    print(f"{'='*70}")
    print(f"Scanning {len(files)} results.json files...\n")

    for fpath in files:
        result = process_file(pathlib.Path(fpath), apply=apply)
        mm = result["mismatches"]
        if mm:
            total_files_affected += 1
            total_mismatches += len(mm)
            print(f"  {result['path']}  ({len(mm)} mismatch(es))")
            for m in mm:
                print(f"    {m['file']} / {m['checker']}: {m['old']} -> {m['new']}")
                print(f"      output: {m['output_preview']}...")
            print()

    print(f"{'='*70}")
    print(f"Total files scanned   : {len(files)}")
    print(f"Files with mismatches : {total_files_affected}")
    print(f"Total mismatches      : {total_mismatches}")
    if apply:
        print("Status: APPLIED — files have been updated.")
    else:
        print("Status: DRY-RUN — no files modified. Use --apply to write changes.")
    print(f"{'='*70}")

    return total_mismatches


if __name__ == "__main__":
    main()

