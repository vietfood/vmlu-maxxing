import json
import os
from collections import defaultdict

from datasets import Dataset
from transformers import AutoTokenizer

from vmlu_maxxing.consts import (
    BASE_MODEL,
    FEW_SHOT_BANK_PATH,
    SFT_PACKED_DATA_DIR,
    VMLU_RAW_DIR,
)
from vmlu_maxxing.ingest_sources import (
    ingest_arc_split,
    ingest_mmlu_split,
    ingest_sciq_split,
    ingest_vimmrc_split,
    ingest_vsec,
)
from vmlu_maxxing.translate_pipeline import translate_sync


def format_mcq(question: str, choices: list[str], answer: str = None) -> str:
    """
    Câu hỏi: {question}
    A. {choice_A}
    B. {choice_B}
    C. {choice_C}
    D. {choice_D}
    [E. {choice_E}]
    Đáp án:
    """
    formatted = f"Câu hỏi: {question}\n"
    for choice in choices:
        formatted += f"{choice}\n"

    formatted += "Đáp án:"
    if answer:
        formatted += f" {answer}"
    return formatted


def build_few_shot_bank(dev_data: list[dict]):
    """
    Extracts 5 high quality examples per subject to use as a few-shot bank.
    Using dev split (304 questions).
    """
    print("Building few-shot example bank from dev split...")
    bank = defaultdict(list)

    for row in dev_data:
        subject = row.get("subject", "general")
        if len(bank[subject]) < 5:
            formatted_example = format_mcq(
                row["question"], row["choices"], row["answer"]
            )
            bank[subject].append(
                {
                    "id": row["id"],
                    "formatted": formatted_example,
                    "num_choices": len(row["choices"]),
                }
            )

    os.makedirs(os.path.dirname(FEW_SHOT_BANK_PATH), exist_ok=True)
    with open(FEW_SHOT_BANK_PATH, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print(f"Saved few-shot bank for {len(bank)} subjects to {FEW_SHOT_BANK_PATH}")


def load_jsonl(filepath: str) -> list[dict]:
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return data

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_sft_dataset(train_data: list[dict], tokenizer):
    """
    Converts raw training dictionary data into a Hugging Face Dataset with columns:
    [text, answer_token_id, subject, language, source, num_choices]
    """
    print(f"Processing {len(train_data)} training examples...")

    processed = {
        "text": [],
        "answer_token_id": [],
        "target_token": [],  # for debugging mapping
        "subject": [],
        "language": [],
        "source": [],
        "num_choices": [],
    }

    skipped = 0
    for row in train_data:
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]  # e.g. 'A'
        subject = row.get("subject", "general")
        source = row.get("source", "vmlu")

        # --- Validation checks (DEVELOPMENT.md) ---
        # 1. Answer must match one of the available choices
        valid_letters = [chr(ord("A") + i) for i in range(len(choices))]
        if answer not in valid_letters:
            skipped += 1
            continue

        # 2. All choices must be non-empty
        if any(not c.strip() for c in choices):
            skipped += 1
            continue

        # 3. Warn on duplicate choices (don't crash)
        choice_texts = [
            c.split(".", 1)[-1].strip() if "." in c else c.strip() for c in choices
        ]
        if len(set(choice_texts)) < len(choice_texts):
            print(f"Warning: duplicate choices in question: {question[:80]}...")

        # 4. Check for answer leakage in question text
        if f"Đáp án: {answer}" in question or f"Answer: {answer}" in question:
            skipped += 1
            continue

        formatted_text = format_mcq(question, choices, answer)

        ans_token_str = f" {answer}"
        ans_token_ids = tokenizer.encode(ans_token_str)

        ans_token_id = ans_token_ids[-1] if ans_token_ids else -100

        processed["text"].append(formatted_text)
        processed["answer_token_id"].append(ans_token_id)
        processed["target_token"].append(ans_token_str)
        processed["subject"].append(subject)
        processed["language"].append("vi")
        processed["source"].append(source)
        processed["num_choices"].append(len(choices))

    if skipped > 0:
        print(f"Skipped {skipped} invalid examples during validation.")

    dataset = Dataset.from_dict(processed)

    print(f"Saving SFT dataset ({len(dataset)} examples) to {SFT_PACKED_DATA_DIR}...")
    os.makedirs(SFT_PACKED_DATA_DIR, exist_ok=True)
    dataset.save_to_disk(SFT_PACKED_DATA_DIR)


def prepare_sft_data(do_translation=False, provider="sglang", model_name=None):
    print("Initializing Phase 1 SFT Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Paths for VMLU
    dev_path = os.path.join(VMLU_RAW_DIR, "dev.jsonl")
    valid_path = os.path.join(VMLU_RAW_DIR, "valid.jsonl")
    train_path = os.path.join(VMLU_RAW_DIR, "train.jsonl")

    # 1. Load VMLU Dev and build few-shot bank
    dev_data = load_jsonl(dev_path)
    if dev_data:
        build_few_shot_bank(dev_data)

    # 2. VMLU Final Aggregation
    sft_data = []

    vmlu_train = load_jsonl(train_path)
    if not vmlu_train:
        print(f"VMLU Train set missing! Defaulting to valid_path.")
        vmlu_train = load_jsonl(valid_path)

    if vmlu_train:
        print(f"Loaded {len(vmlu_train)} VMLU examples.")
        # Tag VMLU data with source='vmlu'
        for row in vmlu_train:
            row["source"] = "vmlu"
        sft_data.extend(vmlu_train)

    # 3. Load Auxiliary Vietnamese Datasets (No translation needed)
    try:
        vimmrc_data = ingest_vimmrc_split("train")
        print(f"Loaded {len(vimmrc_data)} ViMMRC examples.")
        for row in vimmrc_data:
            row["source"] = "vimmrc"
        sft_data.extend(vimmrc_data)
    except Exception as e:
        print(f"Failed to load ViMMRC: {e}")

    # To load VSEC, provide a path if downloaded locally, otherwise skipped
    try:
        vsec_data = ingest_vsec("train")
        print(f"Loaded {len(vsec_data)} VSEC examples.")
        for row in vsec_data:
            row["source"] = "vsec"
        sft_data.extend(vsec_data)
    except Exception as e:
        print(f"Failed to load VSEC: {e}")

    # 4. Load & Translate English Datasets
    if do_translation:
        print("\n--- Running Auxiliary Translation Pipelines ---")
        try:
            # For speed/demo, we use small splits.
            mmlu_data = ingest_mmlu_split("dev")
            arc_data = ingest_arc_split("ARC-Challenge", "validation")
            sciq_data = ingest_sciq_split("validation")

            english_data = mmlu_data + arc_data + sciq_data
            print(f"Total English examples to translate: {len(english_data)}")

            translated_data = translate_sync(
                english_data, provider=provider, model_name=model_name
            )
            sft_data.extend(translated_data)

        except Exception as e:
            print(f"Failed to load or translate English datasets: {e}")
    else:
        print(
            "\nSkipping translation pipeline. Use main(do_translation=True) to translate MMLU/ARC/SciQ locally via SGLang."
        )

    if not sft_data:
        raise RuntimeError("No training data found to process.")

    print(f"\nTotal SFT corpus size: {len(sft_data)}")
    build_sft_dataset(sft_data, tokenizer)
