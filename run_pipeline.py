"""Run the Veltrix data science workflow (d02 onward): all steps in order, or named steps.

    python run_pipeline.py                 # d02 -> d07 (d99 is never run by default)
    python run_pipeline.py select profiles # named steps
    python run_pipeline.py --list
"""
import importlib
import os
import sys
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

STEPS = {
    "baseline": "src.ds.d02_baseline",
    "candidates": "src.ds.d03_candidates",
    "select": "src.ds.d04_select_validate",
    "profiles": "src.ds.d05_segment_stats",
    "test_design": "src.ds.d06_test_design",
    "score": "src.ds.d07_score",
    "external_eval": "src.ds.d99_external_eval",   # reads the restricted answer key; run once, last
}
DEFAULT = [s for s in STEPS if s != "external_eval"]


def run(name: str) -> None:
    print(f"\n=== {name} ({STEPS[name]}) ===")
    t0 = time.perf_counter()
    importlib.import_module(STEPS[name]).main()
    print(f"=== {name} done in {time.perf_counter() - t0:.0f}s ===")


def main(args: list[str]) -> None:
    if "--list" in args:
        print("\n".join(STEPS))
        return
    unknown = [a for a in args if a not in STEPS]
    if unknown:
        sys.exit(f"Unknown step(s): {', '.join(unknown)}. Available: {', '.join(STEPS)}")
    for name in args or DEFAULT:
        run(name)


if __name__ == "__main__":
    main(sys.argv[1:])
