"""
Phase 0: Vietnamese Continued Pre-Training (CPT)
- Step 1: Prepare CPT data (filter, sample, tokenize)
- Step 2: Train with QLoRA
- Step 3: Sanity check (Vietnamese generation + MMLU subset)
"""

from vmlu_maxxing.prepare_cpt import main as prepare_cpt
from vmlu_maxxing.train_cpt import main as train_cpt
from vmlu_maxxing.sanity_check import main as sanity_check


def run():
    print("=" * 60)
    print("PHASE 0: Vietnamese Continued Pre-Training")
    print("=" * 60)

    print("\n[Step 0.1] Preparing CPT data...")
    prepare_cpt()

    print("\n[Step 0.2] Training CPT with QLoRA...")
    train_cpt()

    print("\n[Step 0.3] Running sanity checks...")
    sanity_check()

    print("\n✅ Phase 0 Complete!")


if __name__ == "__main__":
    run()
