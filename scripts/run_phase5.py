"""
Phase 5: Evaluation & Final Merge
- Evaluates with 5-shot logit extraction on VMLU valid/test
- Merges all LoRA adapters into a standalone HuggingFace model
"""

import argparse

from vmlu_maxxing.evaluate import main as evaluate
from vmlu_maxxing.export import merge_adapters


def run(skip_eval: bool = False, skip_merge: bool = False):
    print("=" * 60)
    print("PHASE 5: Evaluation & Final Merge")
    print("=" * 60)

    if not skip_eval:
        print("\n[Step 5.1] Running 5-shot logit extraction evaluation...")
        evaluate()

    if not skip_merge:
        print("\n[Step 5.2] Merging all adapters into standalone model...")
        merge_adapters()

    print("\n✅ Phase 5 Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Eval & Merge")
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation, only merge"
    )
    parser.add_argument(
        "--skip-merge", action="store_true", help="Skip merge, only evaluate"
    )
    args = parser.parse_args()
    run(skip_eval=args.skip_eval, skip_merge=args.skip_merge)
