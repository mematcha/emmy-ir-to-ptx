
"""Run the official Emmy correctness evaluator on every public problem.

Usage:
    python3.12 tests/run_baseline.py

Each problem is evaluated independently through submit.py.
Results are saved to baseline_results.json and baseline_results.log.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "problems"
SUBMIT = ROOT / "submit.py"
COMPILER = ROOT / "compile.py"

RESULT_JSON = ROOT / "baseline_results.json"
RESULT_LOG = ROOT / "baseline_results.log"

# Allow time for remote compilation, execution, and queue delays.
TIMEOUT_SECONDS = 900


def discover_problems():
    """Discover all official problems with metadata and expected outputs."""
    return sorted(
        p for p in PROBLEMS.iterdir()
        if p.is_dir()
        and (p / "meta.json").is_file()
        and (p / "expected.npz").is_file()
    )


def run_problem(problem):
    command = [
        sys.executable,
        str(SUBMIT),
        problem.name,
        "--compiler",
        str(COMPILER),
    ]

    started = time.monotonic()

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )

        output = completed.stdout
        if completed.stderr:
            output += "\n[stderr]\n" + completed.stderr

        status = "PASS" if completed.returncode == 0 else "FAIL"
        return_code = completed.returncode

    except subprocess.TimeoutExpired as exc:
        status = "TIMEOUT"
        return_code = None

        def decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value or ""

        output = (
            decode(exc.stdout)
            + "\n"
            + decode(exc.stderr)
            + "\nOfficial runner timed out."
        )

    elapsed = round(time.monotonic() - started, 2)

    return {
        "problem": problem.name,
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "output": output.strip(),
    }


def main():
    if not SUBMIT.is_file() or not COMPILER.is_file():
        print("Missing submit.py or compile.py", file=sys.stderr)
        return 2

    if not PROBLEMS.is_dir():
        print("Missing problems directory", file=sys.stderr)
        return 2

    problems = discover_problems()

    if not problems:
        print("No valid public problems found.", file=sys.stderr)
        return 2

    print(f"Discovered {len(problems)} official problems.\n", flush=True)

    results = []

    for index, problem in enumerate(problems, 1):
        print(
            f"[{index}/{len(problems)}] Testing {problem.name} ...",
            flush=True,
        )

        result = run_problem(problem)
        results.append(result)

        print(
            f"  {result['status']} "
            f"({result['elapsed_seconds']}s)",
            flush=True,
        )

        if result["output"]:
            print(result["output"], flush=True)

        print("-" * 65, flush=True)

    counts = {
        status: sum(r["status"] == status for r in results)
        for status in ("PASS", "FAIL", "TIMEOUT")
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "summary": counts,
        "results": results,
    }

    RESULT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    RESULT_LOG.write_text(
        "\n\n".join(
            f"=== {r['problem']} : {r['status']} ===\n"
            f"{r['output']}"
            for r in results
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("EMMY BASELINE CORRECTNESS REPORT")
    print("=" * 65)

    print(f"{'Problem':<32} {'Status':<10} {'Seconds':>8}")
    print("-" * 55)

    for result in results:
        print(
            f"{result['problem']:<32} "
            f"{result['status']:<10} "
            f"{result['elapsed_seconds']:>8.2f}"
        )

    print("-" * 55)
    print(f"Total:     {len(results)}")
    print(f"Passed:    {counts['PASS']}")
    print(f"Failed:    {counts['FAIL']}")
    print(f"Timed out: {counts['TIMEOUT']}")
    print(f"\nJSON report: {RESULT_JSON}")
    print(f"Full log:    {RESULT_LOG}")

    return 0 if counts["PASS"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
