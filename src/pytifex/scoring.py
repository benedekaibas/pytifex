# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typeguard",
#     "beartype",
# ]
# ///

"""
Formal scoring framework for evaluating type checker correctness.

Computes per-file detection matrices, detection rates per tier,
precision, recall, F1, and weighted scores for each type checker.

Usage:
    python scoring.py <results.json>
    python scoring.py <results.json> --html report.html
"""

import json
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path

from comprehensive_eval import evaluate_comprehensive, EvaluationResult


TIER_WEIGHTS = {1: 1.00, 2: 0.90, 3: 0.75}


@dataclass
class FileScore:
    filename: str
    has_tier1_bug: bool
    has_tier2_bug: bool
    has_tier3_finding: bool
    checker_detections: dict[str, dict[str, int]]
    checker_reported_error: dict[str, bool]


@dataclass
class CheckerMetrics:
    checker: str
    tier1_detection_rate: float
    tier2_detection_rate: float
    tier3_detection_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    weighted_score: float


def compute_file_score(
    source_code: str,
    checker_outputs: dict[str, str],
    filename: str,
) -> FileScore:
    result = evaluate_comprehensive(source_code, checker_outputs, filename)

    has_t1 = len(result.tier1_bugs) > 0
    has_t2 = len(result.tier2_bugs) > 0
    has_t3 = len(result.tier3_findings) > 0

    checkers = list(checker_outputs.keys())
    detections: dict[str, dict[str, int]] = {}
    reported_error: dict[str, bool] = {}

    t3_checkers_involved = {
        f.get("checker") for f in result.tier3_findings
    }

    for checker in checkers:
        verdict_info = result.checker_verdicts.get(checker, {})
        v = verdict_info.get("verdict", "UNCERTAIN")
        tier = verdict_info.get("tier", 4)

        t1 = 0
        t2 = 0
        t3 = 0

        if v == "CORRECT":
            if tier == 1:
                t1 = 1
            elif tier == 2:
                t2 = 1
            elif tier == 3:
                t3 = 1

        checker_has_t3 = has_t3 and checker in t3_checkers_involved

        detections[checker] = {
            "t1": t1 if has_t1 else -1,
            "t2": t2 if has_t2 else -1,
            "t3": t3 if checker_has_t3 else -1,
        }

        output = checker_outputs.get(checker, "")
        output_lower = output.lower()
        has_error = (
            "error" in output_lower
            and "0 error" not in output_lower
            and "success" not in output_lower
        )
        reported_error[checker] = has_error

    return FileScore(
        filename=filename,
        has_tier1_bug=has_t1,
        has_tier2_bug=has_t2,
        has_tier3_finding=has_t3,
        checker_detections=detections,
        checker_reported_error=reported_error,
    )


def compute_metrics(
    file_scores: list[FileScore],
    checkers: list[str],
) -> dict[str, CheckerMetrics]:
    metrics: dict[str, CheckerMetrics] = {}

    for checker in checkers:
        t1_total = sum(1 for f in file_scores if f.has_tier1_bug)
        t2_total = sum(1 for f in file_scores if f.has_tier2_bug)
        t3_total = sum(
            1 for f in file_scores
            if f.checker_detections[checker]["t3"] >= 0
        )

        t1_caught = sum(
            1 for f in file_scores
            if f.has_tier1_bug and f.checker_detections[checker]["t1"] == 1
        )
        t2_caught = sum(
            1 for f in file_scores
            if f.has_tier2_bug and f.checker_detections[checker]["t2"] == 1
        )
        t3_caught = sum(
            1 for f in file_scores
            if f.checker_detections[checker]["t3"] == 1
        )

        t1_dr = t1_caught / t1_total if t1_total > 0 else 0.0
        t2_dr = t2_caught / t2_total if t2_total > 0 else 0.0
        t3_dr = t3_caught / t3_total if t3_total > 0 else 0.0

        has_any_bug = [
            f for f in file_scores
            if f.has_tier1_bug or f.has_tier2_bug or f.has_tier3_finding
        ]
        no_bug = [
            f for f in file_scores
            if not f.has_tier1_bug and not f.has_tier2_bug and not f.has_tier3_finding
        ]

        tp = sum(
            1 for f in has_any_bug
            if f.checker_reported_error[checker]
        )
        fn = sum(
            1 for f in has_any_bug
            if not f.checker_reported_error[checker]
        )
        fp = sum(
            1 for f in no_bug
            if f.checker_reported_error[checker]
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        weighted = (
            TIER_WEIGHTS[1] * t1_dr
            + TIER_WEIGHTS[2] * t2_dr
            + TIER_WEIGHTS[3] * t3_dr
        )

        metrics[checker] = CheckerMetrics(
            checker=checker,
            tier1_detection_rate=t1_dr,
            tier2_detection_rate=t2_dr,
            tier3_detection_rate=t3_dr,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            weighted_score=weighted,
        )

    return metrics


def print_results(
    file_scores: list[FileScore],
    metrics: dict[str, CheckerMetrics],
    checkers: list[str],
) -> None:
    print("=" * 90)
    print("DETECTION MATRIX (1=caught, 0=missed, -=no bug at tier)")
    print("=" * 90)

    header = f"{'File':<50} {'Bug':>9}  "
    for c in checkers:
        header += f" {c:>12}"
    print(header)

    sub = f"{'':50} {'T1 T2 T3':>9}  "
    for _ in checkers:
        sub += f" {'T1 T2 T3':>12}"
    print(sub)
    print("-" * len(header))

    for f in file_scores:
        t1 = "Y" if f.has_tier1_bug else "-"
        t2 = "Y" if f.has_tier2_bug else "-"
        t3 = "Y" if f.has_tier3_finding else "-"
        row = f"{f.filename:<50} {t1:>2} {t2:>2} {t3:>2}  "
        for c in checkers:
            d = f.checker_detections[c]
            v1 = str(d["t1"]) if d["t1"] >= 0 else "-"
            v2 = str(d["t2"]) if d["t2"] >= 0 else "-"
            v3 = str(d["t3"]) if d["t3"] >= 0 else "-"
            row += f" {v1:>3} {v2:>2} {v3:>2}"
        print(row)

    print()
    print("=" * 90)
    print("SCORING SUMMARY")
    print("=" * 90)
    print(f"{'Checker':<12} {'T1 DR':>7} {'T2 DR':>7} {'T3 DR':>7}"
          f" {'Prec':>7} {'Rec':>7} {'F1':>7} {'WScore':>8} {'Rank':>6}")
    print("-" * 76)

    ranked = sorted(metrics.values(), key=lambda m: m.weighted_score, reverse=True)
    for rank, m in enumerate(ranked, 1):
        print(f"{m.checker:<12}"
              f" {m.tier1_detection_rate:>6.2f}"
              f" {m.tier2_detection_rate:>6.2f}"
              f" {m.tier3_detection_rate:>6.2f}"
              f" {m.precision:>6.2f}"
              f" {m.recall:>6.2f}"
              f" {m.f1:>6.2f}"
              f" {m.weighted_score:>7.2f}"
              f" {rank:>5}")

    print("=" * 90)


def generate_html_report(
    file_scores: list[FileScore],
    metrics: dict[str, CheckerMetrics],
    checkers: list[str],
    output_path: str,
) -> None:
    def cell(val: int) -> str:
        if val < 0:
            return '<td class="na">—</td>'
        if val == 1:
            return '<td class="yes">1</td>'
        return '<td class="no">0</td>'

    def mcell(val: float, thresholds=(0.8, 0.6)) -> str:
        cls = "metric-good" if val >= thresholds[0] else "metric-mid" if val >= thresholds[1] else "metric-bad"
        return f'<td class="{cls}">{val:.2f}</td>'

    rows_t1 = ""
    for f in file_scores:
        t1 = '<td class="yes">✓</td>' if f.has_tier1_bug else '<td class="na">—</td>'
        t2 = '<td class="yes">✓</td>' if f.has_tier2_bug else '<td class="na">—</td>'
        t3 = '<td class="yes">✓</td>' if f.has_tier3_finding else '<td class="na">—</td>'
        checker_cells = ""
        for c in checkers:
            d = f.checker_detections[c]
            checker_cells += cell(d["t1"]) + cell(d["t2"]) + cell(d["t3"])
        rows_t1 += f'<tr><td style="text-align:left;">{f.filename}</td>{t1}{t2}{t3}{checker_cells}</tr>\n'

    t1_total = sum(1 for f in file_scores if f.has_tier1_bug)
    t2_total = sum(1 for f in file_scores if f.has_tier2_bug)
    t3_total = sum(1 for f in file_scores if f.has_tier3_finding)

    rows_dr = ""
    for c in checkers:
        m = metrics[c]
        rows_dr += f"<tr><td><b>{c}</b></td>{mcell(m.tier1_detection_rate)}{mcell(m.tier2_detection_rate)}{mcell(m.tier3_detection_rate)}</tr>\n"

    rows_prf = ""
    for c in checkers:
        m = metrics[c]
        rows_prf += (f"<tr><td><b>{c}</b></td>"
                     f"<td>{m.true_positives}</td><td>{m.false_positives}</td><td>{m.false_negatives}</td>"
                     f"{mcell(m.precision)}{mcell(m.recall)}{mcell(m.f1)}</tr>\n")

    ranked = sorted(metrics.values(), key=lambda m: m.weighted_score, reverse=True)
    medals = ["🥇", "🥈", "🥉"] + [""] * 10
    rows_ws = ""
    for rank, m in enumerate(ranked):
        rows_ws += (f"<tr><td><b>{m.checker}</b></td>"
                    f"<td>{m.tier1_detection_rate:.2f}</td>"
                    f"<td>{m.tier2_detection_rate:.2f}</td>"
                    f"<td>{m.tier3_detection_rate:.2f}</td>"
                    f'{mcell(m.weighted_score, (2.0, 1.5))}'
                    f"<td>{medals[rank]} #{rank+1}</td></tr>\n")

    rows_summary = ""
    for rank, m in enumerate(ranked):
        rows_summary += (f"<tr><td><b>{m.checker}</b></td>"
                         f"{mcell(m.precision)}{mcell(m.recall)}{mcell(m.f1)}"
                         f"<td>{m.tier1_detection_rate:.2f}</td>"
                         f"<td>{m.tier2_detection_rate:.2f}</td>"
                         f"<td>{m.tier3_detection_rate:.2f}</td>"
                         f'{mcell(m.weighted_score, (2.0, 1.5))}'
                         f"<td>{medals[rank]} #{rank+1}</td></tr>\n")

    checker_headers = ""
    checker_sub = ""
    for c in checkers:
        checker_headers += f'<th colspan="3" class="checker-header">{c}</th>'
        checker_sub += "<th>T1</th><th>T2</th><th>T3</th>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pytifex Scoring Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e6edf3; padding: 40px; }}
  h1 {{ color: #58a6ff; font-size: 1.6em; }}
  h2 {{ color: #79c0ff; font-size: 1.2em; margin-top: 40px; }}
  table {{ border-collapse: collapse; margin: 16px 0; font-size: 0.85em; }}
  th, td {{ border: 1px solid #30363d; padding: 8px 14px; text-align: center; }}
  th {{ background: #161b22; color: #79c0ff; font-weight: 600; }}
  td {{ background: #0d1117; }}
  .yes {{ color: #3fb950; font-weight: bold; }}
  .no {{ color: #f85149; font-weight: bold; }}
  .na {{ color: #484f58; }}
  .section {{ margin-top: 50px; border-top: 1px solid #30363d; padding-top: 20px; }}
  .checker-header {{ background: #1c2333; }}
  .metric-good {{ color: #3fb950; }}
  .metric-mid {{ color: #d29922; }}
  .metric-bad {{ color: #f85149; }}
  .formula {{ background: #161b22; border: 1px solid #30363d; padding: 12px 20px; border-radius: 6px; margin: 10px 0; font-family: monospace; color: #e6edf3; font-size: 0.9em; }}
  .note {{ color: #8b949e; font-size: 0.85em; margin-top: 6px; }}
</style>
</head>
<body>

<h1>Pytifex Scoring Report</h1>
<p class="note">Generated from real evaluation data. Files evaluated: {len(file_scores)}</p>

<h2>Table 1: Per-File Detection Matrix</h2>
<p>For each file and tier: does a proven bug exist, and did each checker catch it?</p>
<table>
<thead>
<tr><th rowspan="2">File</th><th colspan="3">Proven Bugs</th>{checker_headers}</tr>
<tr><th>T1</th><th>T2</th><th>T3</th>{checker_sub}</tr>
</thead>
<tbody>{rows_t1}</tbody>
</table>

<div class="section">
<h2>Table 2: Detection Rate Per Tier</h2>
<div class="formula">Detection Rate = files where checker caught tier bug / files that have a tier bug</div>
<table>
<thead><tr><th>Checker</th>
<th>Tier 1 DR<br><span class="note">{t1_total} files</span></th>
<th>Tier 2 DR<br><span class="note">{t2_total} files</span></th>
<th>Tier 3 DR<br><span class="note">{t3_total} files</span></th></tr></thead>
<tbody>{rows_dr}</tbody>
</table>
</div>

<div class="section">
<h2>Table 3: Precision, Recall, F1</h2>
<div class="formula">Precision = TP / (TP + FP)&nbsp;&nbsp;&nbsp;Recall = TP / (TP + FN)&nbsp;&nbsp;&nbsp;F1 = 2 * P * R / (P + R)</div>
<table>
<thead><tr><th>Checker</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
<tbody>{rows_prf}</tbody>
</table>
</div>

<div class="section">
<h2>Table 4: Weighted Score</h2>
<div class="formula">Score = (1.00 × T1 DR) + (0.90 × T2 DR) + (0.75 × T3 DR)</div>
<table>
<thead><tr><th>Checker</th>
<th>T1 DR<br><span class="note">w=1.00</span></th>
<th>T2 DR<br><span class="note">w=0.90</span></th>
<th>T3 DR<br><span class="note">w=0.75</span></th>
<th>Weighted Score</th><th>Rank</th></tr></thead>
<tbody>{rows_ws}</tbody>
</table>
</div>

<div class="section">
<h2>Table 5: Combined Summary</h2>
<table>
<thead><tr><th>Checker</th><th>Precision</th><th>Recall</th><th>F1</th><th>T1 DR</th><th>T2 DR</th><th>T3 DR</th><th>Weighted Score</th><th>Rank</th></tr></thead>
<tbody>{rows_summary}</tbody>
</table>
</div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


def save_json_report(
    file_scores: list[FileScore],
    metrics: dict[str, CheckerMetrics],
    checkers: list[str],
    output_path: str,
) -> None:
    data = {
        "method": "pytifex_scoring_framework",
        "tier_weights": TIER_WEIGHTS,
        "files_evaluated": len(file_scores),
        "detection_matrix": [
            {
                "filename": f.filename,
                "proven_bugs": {"t1": f.has_tier1_bug, "t2": f.has_tier2_bug, "t3": f.has_tier3_finding},
                "checker_detections": f.checker_detections,
            }
            for f in file_scores
        ],
        "metrics": {
            c: {
                "tier1_detection_rate": m.tier1_detection_rate,
                "tier2_detection_rate": m.tier2_detection_rate,
                "tier3_detection_rate": m.tier3_detection_rate,
                "true_positives": m.true_positives,
                "false_positives": m.false_positives,
                "false_negatives": m.false_negatives,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "weighted_score": m.weighted_score,
            }
            for c, m in metrics.items()
        },
        "ranking": [
            m.checker
            for m in sorted(metrics.values(), key=lambda x: x.weighted_score, reverse=True)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def run_scoring(results_path: str, html_path: str | None = None) -> dict[str, CheckerMetrics]:
    tc_dir = Path(__file__).resolve().parent

    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    checkers = data.get("checkers_used", ["mypy", "pyrefly", "zuban", "ty"])

    file_scores: list[FileScore] = []

    for i, entry in enumerate(results, 1):
        filepath = entry.get("filepath", "")
        filename = entry.get("filename", "")
        outputs = entry.get("outputs", {})

        resolved = (tc_dir / filepath).resolve()
        if not resolved.exists():
            continue

        source_code = resolved.read_text()
        score = compute_file_score(source_code, outputs, filename)
        file_scores.append(score)

    if not file_scores:
        print("No files could be evaluated.")
        return {}

    metrics = compute_metrics(file_scores, checkers)
    print_results(file_scores, metrics, checkers)

    output_dir = os.path.dirname(results_path)
    json_path = os.path.join(output_dir, "scoring_report.json")
    save_json_report(file_scores, metrics, checkers, json_path)
    print(f"\nJSON report: {json_path}")

    if html_path is None:
        html_path = os.path.join(output_dir, "scoring_report.html")
    generate_html_report(file_scores, metrics, checkers, html_path)
    print(f"HTML report: {html_path}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scoring.py <results.json> [--html output.html]")
        sys.exit(1)

    results_arg = sys.argv[1]
    tc_dir = Path(__file__).resolve().parent
    results_path = (tc_dir / results_arg).resolve()
    if not results_path.exists():
        results_path = Path(results_arg).resolve()

    html_out = None
    if "--html" in sys.argv:
        idx = sys.argv.index("--html")
        if idx + 1 < len(sys.argv):
            html_out = sys.argv[idx + 1]

    run_scoring(str(results_path), html_out)

