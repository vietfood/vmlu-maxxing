import json
import os

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from vmlu_maxxing.consts import (
    BASE_MODEL,
    CPT_OUTPUT_DIR,
    EVAL_RESULTS_DIR,
    FEW_SHOT_BANK_PATH,
    KD_OUTPUT_DIR,
    SFT_OUTPUT_DIR,
    VMLU_RAW_DIR,
)
from vmlu_maxxing.prepare_sft import format_mcq, load_jsonl


def get_eval_model_and_tokenizer(use_4bit: bool = False, load_adapters: bool = True):
    print(f"Loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if use_4bit:
        print("Loading base model in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        print("Loading base model in bfloat16 full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()

    if load_adapters:
        # Sequentially merge adapters into the base model.
        # PEFT's set_adapter() only activates ONE adapter at a time, so we must
        # merge each adapter into the weights before loading the next one.
        # Order: CPT -> SFT -> KD (each builds on the previous)
        if os.path.exists(CPT_OUTPUT_DIR):
            print(f"Loading and merging Phase 0 CPT adapter from {CPT_OUTPUT_DIR}...")
            model = PeftModel.from_pretrained(model, CPT_OUTPUT_DIR, is_trainable=False)
            model = model.merge_and_unload()

        if os.path.exists(SFT_OUTPUT_DIR):
            print(f"Loading and merging Phase 2 SFT adapter from {SFT_OUTPUT_DIR}...")
            model = PeftModel.from_pretrained(model, SFT_OUTPUT_DIR, is_trainable=False)
            model = model.merge_and_unload()

        if os.path.exists(KD_OUTPUT_DIR):
            print(f"Loading and merging Phase 4 KD adapter from {KD_OUTPUT_DIR}...")
            model = PeftModel.from_pretrained(model, KD_OUTPUT_DIR, is_trainable=False)
            model = model.merge_and_unload()

    return model, tokenizer


def build_few_shot_prompt(question_row, few_shot_bank):
    """
    Constructs the context (with optional few-shot examples).
    """
    prompt = ""
    
    if few_shot_bank is not None:
        subject = question_row.get("subject", "")
        examples = few_shot_bank.get(subject, [])

        # Append the formatted examples
        for ex in examples:
            prompt += ex["formatted"] + "\n\n"

    # Format the actual question (without providing the answer)
    # We use format_mcq but strip the trailing Answer out so the model can predict it
    full_mcq = format_mcq(question_row["question"], question_row["choices"], "")
    # Trim the trailing newline that format_mcq might leave
    full_mcq = full_mcq.rstrip()

    prompt += full_mcq
    return prompt


def evaluate_predictions(predictions, ground_truths, subjects):
    """
    Calculates overall and per-subject accuracy.
    """
    correct = 0
    total = len(predictions)

    subject_correct = {}
    subject_total = {}

    for pred, truth, subj in zip(predictions, ground_truths, subjects):
        if subj not in subject_total:
            subject_total[subj] = 0
            subject_correct[subj] = 0

        subject_total[subj] += 1
        if pred == truth:
            correct += 1
            subject_correct[subj] += 1

    print(
        f"\\n--- Overall Accuracy: {correct}/{total} ({(correct / total) * 100:.2f}%) ---"
    )

    print("\\n--- Lowest 10 Subjects ---")
    sub_acc = []
    for subj in subject_total:
        acc = subject_correct[subj] / subject_total[subj]
        sub_acc.append((subj, acc, subject_total[subj]))

    sub_acc.sort(key=lambda x: x[1])

    for subj, acc, count in sub_acc[:10]:
        print(f"{subj}: {acc * 100:.2f}% ({count} questions)")

    return {"overall": correct / total, "subjects": sub_acc}


def main(load_adapters: bool = True, use_4bit: bool = False, output_prefix: str = "vmlu_eval", use_few_shot: bool = True):
    few_shot_bank = None
    if use_few_shot:
        if not os.path.exists(FEW_SHOT_BANK_PATH):
            raise FileNotFoundError(
                f"Few shot bank missing at {FEW_SHOT_BANK_PATH}. Run prepare_sft.py first."
            )

        with open(FEW_SHOT_BANK_PATH, "r", encoding="utf-8") as f:
            few_shot_bank = json.load(f)

    # Prefer test split if available, otherwise fallback to valid
    test_file = os.path.join(VMLU_RAW_DIR, "test.jsonl")
    valid_file = os.path.join(VMLU_RAW_DIR, "valid.jsonl")

    if os.path.exists(test_file):
        eval_data = load_jsonl(test_file)
        print(f"Evaluating on {test_file} ({len(eval_data)} rows)...")
    elif os.path.exists(valid_file):
        eval_data = load_jsonl(valid_file)
        print(f"Evaluating on {valid_file} ({len(eval_data)} rows)...")
    else:
        raise FileNotFoundError(
            "Neither test.jsonl nor valid.jsonl found in VMLU directory."
        )

    model, tokenizer = get_eval_model_and_tokenizer(
        use_4bit=use_4bit, load_adapters=load_adapters
    )

    # 5 choices map to these exact letter tokens.
    # Qwen tokenization variations (leading spaces, newlines, etc).
    # Since we pad Answer: with a space, it could be ' A' or strictly 'A' inside a byte pair.
    # We examine all 3 common variants.
    vocab_mapping = {
        "A": [
            tokenizer.encode("A", add_special_tokens=False)[-1],
            tokenizer.encode(" A", add_special_tokens=False)[-1],
            tokenizer.encode("\nA", add_special_tokens=False)[-1],
        ],
        "B": [
            tokenizer.encode("B", add_special_tokens=False)[-1],
            tokenizer.encode(" B", add_special_tokens=False)[-1],
            tokenizer.encode("\nB", add_special_tokens=False)[-1],
        ],
        "C": [
            tokenizer.encode("C", add_special_tokens=False)[-1],
            tokenizer.encode(" C", add_special_tokens=False)[-1],
            tokenizer.encode("\nC", add_special_tokens=False)[-1],
        ],
        "D": [
            tokenizer.encode("D", add_special_tokens=False)[-1],
            tokenizer.encode(" D", add_special_tokens=False)[-1],
            tokenizer.encode("\nD", add_special_tokens=False)[-1],
        ],
        "E": [
            tokenizer.encode("E", add_special_tokens=False)[-1],
            tokenizer.encode(" E", add_special_tokens=False)[-1],
            tokenizer.encode("\nE", add_special_tokens=False)[-1],
        ],
    }

    valid_keys = list(vocab_mapping.keys())  # ["A", "B", "C", "D", "E"]

    predictions = []
    ground_truths = []
    subjects = []
    ids = []

    print("Beginning Evaluation...")
    with torch.no_grad():
        for i, row in enumerate(tqdm(eval_data)):
            prompt = build_few_shot_prompt(row, few_shot_bank)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Forward pass: extract logits of the exact final output token
            outputs = model(**inputs)
            final_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]

            num_choices = len(row["choices"])
            if num_choices > 5:
                num_choices = 5

            allowed_letters = valid_keys[:num_choices]

            # Sum probability mass across token variants
            choice_probs = {}
            for letter in allowed_letters:
                token_ids = vocab_mapping[letter]
                # Logit sum or Probability sum?
                # According to DEVELOPMENT.md: sum over Variants before Argmax.
                # Since log probabilities are exponential, summing probabilities is technically the truest "mass".
                # To avoid precision implosion, we extract raw logits, exponentiate them to get relative probs (softmax denominator cancels out in argmax), and sum them.
                raw_logits_for_variant = final_logits[token_ids]
                probs = torch.exp(raw_logits_for_variant)
                choice_probs[letter] = probs.sum().item()

            # Argmax over available choices
            best_choice = max(choice_probs, key=choice_probs.get)

            predictions.append(best_choice)
            ground_truths.append(row["answer"])
            subjects.append(row.get("subject", "Unknown"))
            
            # VMLU standardizes on custom IDs, fallback to index if missing
            row_id = row.get("id", f"unknown-{i}")
            ids.append(row_id)

    # Compute Metrics
    results_data = evaluate_predictions(predictions, ground_truths, subjects)

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    
    # Save metrics JSON
    json_path = os.path.join(EVAL_RESULTS_DIR, f"{output_prefix}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
        
    # Save CSV Submission format: id,answer (lowercase)
    import csv
    csv_path = os.path.join(EVAL_RESULTS_DIR, f"{output_prefix}_submission.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for row_id, pred in zip(ids, predictions):
            writer.writerow([row_id, pred.lower()])

    print(f"Metrics saved to {json_path}")
    print(f"Submission CSV saved to {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true", help="Evaluate the raw base model (no adapters)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit instead of bfloat16")
    parser.add_argument("--output-prefix", type=str, default="vmlu_eval")
    parser.add_argument("--zero-shot", action="store_true", help="Run 0-shot evaluation instead of 5-shot")
    args = parser.parse_args()
    
    main(
        load_adapters=not args.base_only, 
        use_4bit=args.load_in_4bit,
        output_prefix=args.output_prefix,
        use_few_shot=not args.zero_shot
    )
