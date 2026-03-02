"""
Phase 4: Soft Label Distillation Training
- Trains student model using mixed CE + KL-Divergence loss
- Stacks KD adapter on top of Phase 2 SFT adapter
"""

from vmlu_maxxing.train_kd import main as train_kd


def run():
    print("=" * 60)
    print("PHASE 4: Soft Label Distillation Training")
    print("=" * 60)

    print("\n[Step 3.2] Training with KD loss (alpha=0.7, tau=2.0)...")
    train_kd()

    print("\n✅ Phase 4 Complete!")


if __name__ == "__main__":
    run()
