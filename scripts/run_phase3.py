"""
Phase 3: Knowledge Distillation from Strong Teacher
- Requires SGLang server running (see launch_sglang.sh)
- Extracts logprobs from Qwen3.5-27B for each MCQ question
- Saves distilled soft labels to data/distilled_sft/
"""

from vmlu_maxxing.distill_teacher import main as distill_teacher


def run():
    print("=" * 60)
    print("PHASE 3: Knowledge Distillation from Strong Teacher")
    print("=" * 60)

    print("\n⚠️  Make sure SGLang server is running!")
    print("   Run: bash launch_sglang.sh")
    print()

    print("[Step 3.1] Extracting teacher logprobs via SGLang...")
    distill_teacher()

    print("\n✅ Phase 3 Complete!")


if __name__ == "__main__":
    run()
