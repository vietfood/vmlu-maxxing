"""
VMLU Maxxing: Full Pipeline Runner
Runs all phases sequentially (Phase 0 -> 5).

Usage:
    uv run python scripts/run_all.py
    uv run python scripts/run_all.py --start-from 2        # Resume from Phase 2
    uv run python scripts/run_all.py --start-from 3 --translate  # Phase 3+ with translation
"""

import argparse

from run_phase0 import run as run_phase0
from run_phase1 import run as run_phase1
from run_phase2 import run as run_phase2
from run_phase3 import run as run_phase3
from run_phase4 import run as run_phase4
from run_phase5 import run as run_phase5


def main():
    parser = argparse.ArgumentParser(description="VMLU Maxxing Full Pipeline")
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Phase to start from (default: 0)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable English dataset translation in Phase 1",
    )
    args = parser.parse_args()

    phases = {
        0: lambda: run_phase0(),
        1: lambda: run_phase1(do_translation=args.translate),
        2: lambda: run_phase2(),
        3: lambda: run_phase3(),
        4: lambda: run_phase4(),
        5: lambda: run_phase5(),
    }

    for phase_num in range(args.start_from, 6):
        phases[phase_num]()
        print()

    print("🎉 ALL PHASES COMPLETE! 🎉")


if __name__ == "__main__":
    main()
