"""Run end-to-end EV desert pipeline stages in order.

Usage:
  python3 run_pipeline.py
  python3 run_pipeline.py --from-stage 06
  python3 run_pipeline.py --to-stage 10
  python3 run_pipeline.py --skip 05
  python3 run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGES = [
    ("04b", "notebooks/04b_modeling_features.py"),
    ("05", "notebooks/05_eda_visualizations.py"),
    ("06", "notebooks/06_xgboost_classifier.py"),
    ("07", "notebooks/07_candidate_site_generation.py"),
    ("08", "notebooks/08_site_ranking_topN.py"),
    ("10", "notebooks/10_installation_forecasting.py"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EV desert pipeline stages.")
    parser.add_argument("--from-stage", choices=[s for s, _ in STAGES], default=STAGES[0][0])
    parser.add_argument("--to-stage", choices=[s for s, _ in STAGES], default=STAGES[-1][0])
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=[s for s, _ in STAGES],
        help="Stage IDs to skip, e.g. --skip 05",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stage commands only.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    order = [s for s, _ in STAGES]
    i0, i1 = order.index(args.from_stage), order.index(args.to_stage)
    if i0 > i1:
        raise ValueError("--from-stage must come before --to-stage")

    selected = STAGES[i0 : i1 + 1]
    selected = [(sid, script) for sid, script in selected if sid not in set(args.skip)]
    if not selected:
        print("No stages selected.")
        return 0

    print("Pipeline root:", ROOT)
    for sid, script in selected:
        cmd = [sys.executable, str(ROOT / script)]
        print(f"[{sid}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("Pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
