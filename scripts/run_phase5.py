"""
Phase 5: Evaluation & Final Merge
- Evaluates with 5-shot logit extraction on VMLU valid/test
- Merges all LoRA adapters into a standalone HuggingFace model
"""

import argparse

from vmlu_maxxing.evaluate import evaluate_model as evaluate
from vmlu_maxxing.export import merge_adapters


def run(skip_eval: bool = False, skip_merge: bool = False, base_only: bool = False, load_in_4bit: bool = False, output_prefix: str = "vmlu_eval", zero_shot: bool = False):
    print("=" * 60)
    print("PHASE 5: Evaluation & Final Merge")
    print("=" * 60)

    if not skip_eval:
        shot_text = "0-shot" if zero_shot else "5-shot"
        print(f"\n[Step 5.1] Running {shot_text} logit extraction evaluation...")
        evaluate(load_adapters=not base_only, use_4bit=load_in_4bit, output_prefix=output_prefix, use_few_shot=not zero_shot)

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
    parser.add_argument(
        "--base-only", action="store_true", help="Evaluate only the raw base model (skip loading any LoRA adapters)"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Evaluate the model in 4-bit precision instead of bfloat16"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="vmlu_eval", help="Prefix for the saved CSV and JSON results"
    )
    parser.add_argument(
        "--zero-shot", action="store_true", help="Run 0-shot evaluation instead of 5-shot"
    )
    args = parser.parse_args()
    run(
        skip_eval=args.skip_eval, 
        skip_merge=args.skip_merge,
        base_only=args.base_only,
        load_in_4bit=args.load_in_4bit,
        output_prefix=args.output_prefix,
        zero_shot=args.zero_shot
    )
