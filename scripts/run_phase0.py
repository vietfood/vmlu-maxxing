"""
Phase 0: Vietnamese Continued Pre-Training (CPT)
- Step 1: Prepare CPT data (filter, sample, tokenize)
- Step 2: Train with QLoRA
- Step 3: Sanity check (Vietnamese generation + MMLU subset)
"""

from vmlu_maxxing.prepare_cpt import prepare_cpt_data
from vmlu_maxxing.train_cpt import train_cpt_model
from vmlu_maxxing.sanity_check import sanity_check


def run(dataset_path: str = None):
    print("=" * 60)
    print("PHASE 0: Vietnamese Continued Pre-Training")
    print("=" * 60)

    if dataset_path is None:
        print("\n[Step 0.1] Preparing CPT data...")
        prepare_cpt_data()
    else:
        print(f"\n[Step 0.1] Skipping CPT preparation. Using customized dataset: {dataset_path}")

    print("\n[Step 0.2] Training CPT with QLoRA...")
    train_cpt_model(dataset_path)

    print("\n[Step 0.3] Running sanity checks...")
    sanity_check()

    print("\n✅ Phase 0 Complete!")


if __name__ == "__main__":
    run()
