import os

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from vmlu_maxxing.consts import (
    BASE_MODEL,
    CPT_OUTPUT_DIR,
    MMLU_TASKS,
    NUM_MMLU_SHOTS,
    NUM_VN_PARAGRAPHS,
)

# CPT model path (leave empty to test original base model)
CPT_LORA_PATH = CPT_OUTPUT_DIR


def load_model(use_4bit: bool = False):
    if use_4bit:
        print(f"Loading base model {BASE_MODEL} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        print(f"Loading base model {BASE_MODEL} in full precision (bfloat16)...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    if os.path.exists(CPT_LORA_PATH):
        print(f"Loading CPT LoRA adapters from {CPT_LORA_PATH}...")
        model = PeftModel.from_pretrained(model, CPT_LORA_PATH)
    else:
        print("CPT LoRA path not found. Testing BASE MODEL performance.")

    return model, tokenizer


def generate_vietnamese(model, tokenizer):
    print("\n--- Generating Vietnamese Paragraphs ---")

    # We use chat template here because we're testing the model's actual generation ability post-CPT
    prompt = "Hãy viết một đoạn văn ngắn khoảng 100 chữ bằng tiếng Việt về vẻ đẹp của thiên nhiên."
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generation config based on DEVELOPMENT.md Qwen recommendations
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        )

    # Standard Qwen parsing
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("\n[Thinking]:", thinking_content)
    print("\n[Content]:", content)


def format_mmlu_question(question_data, with_answer=False):
    """Formats MMLU questions strictly into A, B, C, D format."""
    q = question_data["question"]
    choices = question_data["choices"]

    formatted = f"Question: {q}\n"
    for i, choice in enumerate(choices):
        formatted += f"{chr(ord('A') + i)}. {choice}\n"

    formatted += "Answer:"

    if with_answer:
        answer_idx = question_data["answer"]  # 0 for A, 1 for B...
        formatted += f" {chr(ord('A') + answer_idx)}\n\n"

    return formatted


def run_mmlu_subset(model, tokenizer):
    print("\n--- Running MMLU 5-shot Eval (subset) ---")

    total_correct = 0
    total_questions = 0

    for task in MMLU_TASKS:
        print(f"Loading MMLU task: {task}...")
        try:
            # cais/mmlu load requires 'all' or specific task name
            dataset = load_dataset("cais/mmlu", task)
        except Exception as e:
            print(f"Failed to load MMLU task {task}: {e}")
            continue

        dev_set = list(dataset["dev"])
        test_set = list(dataset["test"])[:50]  # Run on 50 questions per task for speed

        # Build 5-shot prompt from dev set
        few_shot_prompt = ""
        for i in range(min(NUM_MMLU_SHOTS, len(dev_set))):
            few_shot_prompt += format_mmlu_question(dev_set[i], with_answer=True)

        task_correct = 0
        pbar = tqdm(total=len(test_set), desc=f"Eval {task}")

        for row in test_set:
            prompt = few_shot_prompt + format_mmlu_question(row, with_answer=False)

            # We don't use chat templates for exact MMLU evaluation, just direct completion
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            # Extract logit logic from DEVELOPMENT.md Phase 5 guidelines
            with torch.no_grad():
                outputs = model(**inputs)

            # Logits of last token
            final_logits = outputs.logits[0, -1, :]

            # Get IDs for A, B, C, D variants
            # Check for multiple spacing variants to capture probability mass
            vocab_variants = {
                0: [" A", "A", "\nA"],
                1: [" B", "B", "\nB"],
                2: [" C", "C", "\nC"],
                3: [" D", "D", "\nD"],
            }

            choice_probs = []
            for choice_idx in range(4):
                variants = vocab_variants[choice_idx]
                prob_sum = 0.0
                for variant in variants:
                    token_ids = tokenizer.encode(variant, add_special_tokens=False)
                    token_id = token_ids[-1] if token_ids else None
                    if token_id is not None and token_id != tokenizer.unk_token_id:
                        # For exact precision, we sum probabilities (exp(logit)), not sum raw logits directly
                        prob_sum += torch.exp(final_logits[token_id]).item()
                choice_probs.append(prob_sum)

            predicted_answer = choice_probs.index(max(choice_probs))
            actual_answer = row["answer"]

            if predicted_answer == actual_answer:
                task_correct += 1

            pbar.update(1)

        pbar.close()
        acc = task_correct / len(test_set)
        print(f"{task} Accuracy: {acc:.2f} ({task_correct}/{len(test_set)})")

        total_correct += task_correct
        total_questions += len(test_set)

    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    print(
        f"\nOverall MMLU Subset Accuracy: {overall_acc:.2f} ({total_correct}/{total_questions})"
    )


def main(use_4bit: bool = False):
    model, tokenizer = load_model(use_4bit)

    generate_vietnamese(model, tokenizer)
    run_mmlu_subset(model, tokenizer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit instead of bfloat16")
    args = parser.parse_args()
    main(use_4bit=args.load_in_4bit)
