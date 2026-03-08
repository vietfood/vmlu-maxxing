import argparse

from vmlu_maxxing import prepare_cpt
from vmlu_maxxing import train_cpt
from vmlu_maxxing import prepare_sft
from vmlu_maxxing import train_sft
from vmlu_maxxing import distill_teacher
from vmlu_maxxing import train_kd
from vmlu_maxxing import evaluate
from vmlu_maxxing import export

def run_phase0(dataset_path: str = None):
    print("=" * 60)
    print("PHASE 0: CPT Pre-training")
    print("=" * 60)
    
    if dataset_path is None:
        prepare_cpt.prepare_cpt_data()
    else:
        print(f"Skipping local data preparation, using dataset: {dataset_path}")

    train_cpt.train_cpt_model(dataset_path)
    print("\n✅ Phase 0 Complete!")

def run_phase1(do_translation: bool = False):
    print("=" * 60)
    print("PHASE 1: SFT Data Preparation")
    print("=" * 60)
    prepare_sft.prepare_sft_data(do_translation=do_translation)
    print("\n✅ Phase 1 Complete!")

def run_phase2(dataset_path: str = None):
    print("=" * 60)
    print("PHASE 2: SFT Training")
    print("=" * 60)
    train_sft.train_sft_model(dataset_path)
    print("\n✅ Phase 2 Complete!")

def run_phase3(dataset_path: str = None):
    print("=" * 60)
    print("PHASE 3: Logit Distillation Extraction")
    print("=" * 60)
    distill_teacher.extract_teacher_logits(dataset_path)
    print("\n✅ Phase 3 Complete!")

def run_phase4(dataset_path: str = None):
    print("=" * 60)
    print("PHASE 4: Knowledge Distillation Training")
    print("=" * 60)
    train_kd.train_kd_model(dataset_path)
    print("\n✅ Phase 4 Complete!")

def run_phase5(skip_eval: bool = False, skip_merge: bool = False, base_only: bool = False, load_in_4bit: bool = False, output_prefix: str = "vmlu_eval", zero_shot: bool = False):
    print("=" * 60)
    print("PHASE 5: Evaluation & Final Merge")
    print("=" * 60)

    if not skip_eval:
        shot_text = "0-shot" if zero_shot else "5-shot"
        print(f"\n[Step 5.1] Running {shot_text} logit extraction evaluation...")
        evaluate.evaluate_model(load_adapters=not base_only, use_4bit=load_in_4bit, output_prefix=output_prefix, use_few_shot=not zero_shot)

    if not skip_merge:
        print("\n[Step 5.2] Merging all adapters into standalone model...")
        export.merge_adapters()

    print("\n✅ Phase 5 Complete!")

def main():
    parser = argparse.ArgumentParser(description="VMLU Maxxing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run-all
    parser_all = subparsers.add_parser("run-all", help="Run all phases sequentially")
    parser_all.add_argument("--start-from", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="Phase to start from")
    parser_all.add_argument("--translate", action="store_true", help="Enable English dataset translation in Phase 1")
    parser_all.add_argument("--dataset-path", type=str, default=None, help="Custom dataset path (local or HF HuggingFace ID) to inject to Phase 0, 2, 3 or 4")
    
    # Command: run-phase
    parser_phase = subparsers.add_parser("run-phase", help="Run a specific phase")
    parser_phase.add_argument("phase", type=int, choices=[0, 1, 2, 3, 4, 5], help="Phase number to run")
    parser_phase.add_argument("--translate", action="store_true", help="Enable English dataset translation in Phase 1")
    parser_phase.add_argument("--dataset-path", type=str, default=None, help="Custom dataset path (local or HF HuggingFace ID)")
    
    # Phase 5 specific args for run-phase
    parser_phase.add_argument("--skip-eval", action="store_true", help="Phase 5: Skip evaluation")
    parser_phase.add_argument("--skip-merge", action="store_true", help="Phase 5: Skip merge")
    parser_phase.add_argument("--base-only", action="store_true", help="Phase 5: Evaluate base model only")
    parser_phase.add_argument("--load-in-4bit", action="store_true", help="Phase 5: Eval in 4-bit instead of bfloat16")
    parser_phase.add_argument("--output-prefix", type=str, default="vmlu_eval", help="Phase 5: Output prefix")
    parser_phase.add_argument("--zero-shot", action="store_true", help="Phase 5: Run 0-shot eval")

    args = parser.parse_args()

    if args.command == "run-all":
        phases = {
            0: lambda: run_phase0(dataset_path=args.dataset_path),
            1: lambda: run_phase1(do_translation=args.translate),
            2: lambda: run_phase2(dataset_path=args.dataset_path),
            3: lambda: run_phase3(dataset_path=args.dataset_path),
            4: lambda: run_phase4(dataset_path=args.dataset_path),
            5: lambda: run_phase5(),
        }
        for phase_num in range(args.start_from, 6):
            phases[phase_num]()
            print()
        print("🎉 ALL PHASES COMPLETE! 🎉")
        
    elif args.command == "run-phase":
        if args.phase == 0:
            run_phase0(dataset_path=args.dataset_path)
        elif args.phase == 1:
            run_phase1(do_translation=args.translate)
        elif args.phase == 2:
            run_phase2(dataset_path=args.dataset_path)
        elif args.phase == 3:
            run_phase3(dataset_path=args.dataset_path)
        elif args.phase == 4:
            run_phase4(dataset_path=args.dataset_path)
        elif args.phase == 5:
            run_phase5(
                skip_eval=args.skip_eval,
                skip_merge=args.skip_merge,
                base_only=args.base_only,
                load_in_4bit=args.load_in_4bit,
                output_prefix=args.output_prefix,
                zero_shot=args.zero_shot
            )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
