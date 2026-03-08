"""
Phase 1: Data Curation & Vietnamese MCQ Corpus
- Ingests VMLU, MMLU, ARC, SciQ, ViMMRC, VSEC
- Optionally translates English sources via LangChain
- Saves the unified SFT dataset to data/sft/
"""

from vmlu_maxxing.prepare_sft import prepare_sft_data as prepare_sft


def run(do_translation: bool = False):
    print("=" * 60)
    print("PHASE 1: Data Curation & Vietnamese MCQ Corpus")
    print("=" * 60)

    print("\n[Step 1.1-1.3] Preparing SFT dataset...")
    prepare_sft(do_translation=do_translation)

    print("\n✅ Phase 1 Complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Data Curation")
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable translation of English datasets (requires OPENAI_API_KEY or GOOGLE_API_KEY)",
    )
    args = parser.parse_args()
    run(do_translation=args.translate)
