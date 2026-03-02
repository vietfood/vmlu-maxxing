"""
Phase 2: Vietnamese SFT with QLoRA
- Loads the Phase 1 SFT dataset
- Trains with masked answer-only loss using DataCollatorForMCQ
- Stacks SFT adapter on top of Phase 0 CPT adapter (if present)
"""

from vmlu_maxxing.train_sft import main as train_sft


def run():
    print("=" * 60)
    print("PHASE 2: Vietnamese SFT with QLoRA")
    print("=" * 60)

    print("\n[Step 2.1-2.3] Training SFT...")
    train_sft()

    print("\n✅ Phase 2 Complete!")


if __name__ == "__main__":
    run()
