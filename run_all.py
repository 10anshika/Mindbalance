"""
MindBalance v2 | run_all.py
Runs the full pipeline in sequence with section headers and timing.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time
import os

# Resolve paths relative to this script so it works from any working directory.
ROOT    = os.path.dirname(os.path.abspath(__file__))
PYTHON  = sys.executable  # use the same interpreter that launched run_all.py
SRC     = os.path.join(ROOT, "src")

STEPS = [
    ("Exploratory Data Analysis",  [PYTHON, os.path.join(SRC, "eda.py")]),
    ("Model Training",             [PYTHON, os.path.join(SRC, "train.py")]),
    ("Model Evaluation",           [PYTHON, os.path.join(SRC, "evaluate.py")]),
    ("SHAP Explainability",        [PYTHON, os.path.join(SRC, "shap_explainer.py")]),
    ("IKS Engine (demo mode)",     [PYTHON, os.path.join(SRC, "iks_engine.py"), "--demo"]),
    ("Lifestyle Recommender (batch)", [PYTHON, os.path.join(SRC, "recommender.py"), "--batch"]),
]

W = 70  # banner width


def banner(title: str, index: int, total: int):
    step_tag = f"STEP {index}/{total}"
    print()
    print("█" * W)
    print(f"█  {step_tag:<10}  {title}")
    print("█" * W)


def run_step(cmd: list[str]) -> tuple[int, float]:
    """Run a subprocess, stream its output, return (returncode, elapsed_seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=ROOT,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    return proc.returncode, time.perf_counter() - t0


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def main():
    overall_start = time.perf_counter()
    total         = len(STEPS)
    results       = []

    print()
    print("=" * W)
    print("  MINDBALANCE v2 — FULL PIPELINE RUN")
    print("=" * W)

    for i, (title, cmd) in enumerate(STEPS, start=1):
        banner(title, i, total)
        rc, elapsed = run_step(cmd)
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"\n  [{status}]  {title}  —  {fmt_time(elapsed)}")
        results.append((title, rc, elapsed))

        if rc != 0:
            print(f"\n  Pipeline aborted at step {i}: {title}")
            print("  Fix the error above and re-run.\n")
            break

    # ── Summary table ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - overall_start
    print()
    print("=" * W)
    print("  PIPELINE SUMMARY")
    print("=" * W)
    print(f"  {'Step':<38} {'Time':>8}   Status")
    print("  " + "─" * (W - 2))
    for title, rc, elapsed in results:
        status = "✓ OK" if rc == 0 else "✗ FAILED"
        print(f"  {title:<38} {fmt_time(elapsed):>8}   {status}")
    print("  " + "─" * (W - 2))
    print(f"  {'TOTAL':<38} {fmt_time(total_elapsed):>8}")
    print("=" * W)
    print()

    all_ok = all(rc == 0 for _, rc, _ in results)
    if all_ok:
        print("  All steps completed successfully.")
        print(f"  Outputs written to:  {os.path.join(ROOT, 'outputs')}")
        print(f"  Models saved to:     {os.path.join(ROOT, 'models')}")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
