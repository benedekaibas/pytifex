import os
import json
import subprocess
import sys
import glob as glob_module

from config import BASE_GEN_DIR, CHECKERS
from rederive_statuses import checker_reports_error
from code_metrics import compute_metrics, metrics_to_dict


def get_latest_generation_dir() -> str:
    """Finds the most recent timestamped folder in generated_examples."""
    if not os.path.exists(BASE_GEN_DIR):
        raise FileNotFoundError(f"Directory '{BASE_GEN_DIR}' does not exist.")

    subdirs = [
        os.path.join(BASE_GEN_DIR, d)
        for d in os.listdir(BASE_GEN_DIR)
        if os.path.isdir(os.path.join(BASE_GEN_DIR, d))
    ]

    if not subdirs:
        raise FileNotFoundError(f"No generated examples found in '{BASE_GEN_DIR}'.")

    latest = max(subdirs, key=os.path.basename)
    return latest


def run_tool(command: list[str], filepath: str) -> str:
    """Runs a single type checker command on a file."""
    try:
        full_cmd = command + [filepath]

        result = subprocess.run(full_cmd, capture_output=True, text=True, check=False)

        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr

        return output.strip() if output.strip() else "Success (No Output)"

    except FileNotFoundError:
        return f"Error: Command '{command[0]}' not found in PATH."
    except Exception as e:
        return f"Execution Error: {str(e)}"


def run_checkers(target_dir: str | None = None) -> str:
    """
    Run type checkers on Python files in the target directory.
    Returns the path to the results.json file.
    """
    if target_dir is None:
        target_dir = get_latest_generation_dir()

    source_files_dir = os.path.join(target_dir, "source_files")

    if not os.path.exists(source_files_dir):
        raise FileNotFoundError(
            f"No 'source_files' directory found in {target_dir}"
        )

    py_files = glob_module.glob(os.path.join(source_files_dir, "*.py"))
    if not py_files:
        raise FileNotFoundError("No .py files found to check.")

    print(f"--- Running Type Checkers on {len(py_files)} files ---")
    print(f"Directory: {target_dir}\n")

    all_results = []

    for filepath in py_files:
        filename = os.path.basename(filepath)
        print(f"Checking {filename}...")

        with open(filepath, "r", encoding="utf-8") as src:
            source_code = src.read()

        file_result = {"filename": filename, "filepath": filepath, "metrics": metrics_to_dict(compute_metrics(source_code)), "outputs": {}}

        for tool_name, command in CHECKERS.items():
            output = run_tool(command, filepath)
            file_result["outputs"][tool_name] = output

        file_result["statuses"] = {}
        for tool_name, output in file_result["outputs"].items():
            file_result["statuses"][tool_name] = "error" if checker_reports_error(output, tool_name) else "ok"

        all_results.append(file_result)

    results_json_path = os.path.join(target_dir, "results.json")

    final_output = {
        "timestamp": os.path.basename(target_dir),
        "checkers_used": list(CHECKERS.keys()),
        "results": all_results,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    print(f"\n[SUCCESS] Results saved to: {results_json_path}")

    return results_json_path


def main():
    """CLI entry point."""
    try:
        run_checkers()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

