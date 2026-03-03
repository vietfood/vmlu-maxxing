import hashlib
import json
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from underthesea import classify

from vmlu_maxxing.consts import (
    BASE_MODEL,
    CPT_DATASET_NAME,
    CPT_MAX_SEQ_LEN,
    CPT_PACKED_DATA_DIR,
    CPT_TARGET_SAMPLES,
    EDUCATIONAL_TOPICS,
)


def get_hash(text: str) -> str:
    """Returns MD5 hash for exact string deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def contains_educational_content(text: str) -> bool:
    """Use underthesea to aggressively check if text is educational."""
    try:
        topics = classify(text)
        if not topics:
            return False
        # Topics might be a string or a list of strings
        if isinstance(topics, str):
            topics = [topics]

        for topic in topics:
            if topic in EDUCATIONAL_TOPICS:
                return True
        return False
    except Exception:
        # Fallback if classify fails (sometimes happens on extremely weird text)
        return False


def build_dataset():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Dataset streams, no need to download 12.2M rows locally
    print("Streaming dataset...")
    dataset_stream = load_dataset(CPT_DATASET_NAME, split="train", streaming=True)

    seen_hashes = set()
    collected_texts = []

    print(f"Collecting {CPT_TARGET_SAMPLES} high-quality samples...")
    pbar = tqdm(total=CPT_TARGET_SAMPLES)

    for row in dataset_stream:
        text = row.get("text", "")

        # 1. Length filter
        if not (200 <= len(text) <= 20000):
            continue

        # 2. Exact Deduplication
        text_hash = get_hash(text)
        if text_hash in seen_hashes:
            continue

        # 3. Content filtering / Classification
        # This is the slowest part.
        if not contains_educational_content(text):
            continue

        seen_hashes.add(text_hash)
        collected_texts.append(text)
        pbar.update(1)

        if len(collected_texts) >= CPT_TARGET_SAMPLES:
            break

    pbar.close()

    print(
        f"Collected {len(collected_texts)} texts. Starting tokenization and packing..."
    )

    # 4. Tokenization and Packing
    # Group texts together, tokenize, and pack to MAX_SEQ_LEN chunks
    packed_input_ids = []
    current_chunk = []
    current_length = 0

    # Iterate and pack tokens
    for i in tqdm(range(len(collected_texts)), desc="Tokenizing"):
        text = collected_texts[i]

        # We are doing Continuous Pretraining (CPT), so we use raw text instead of chat format
        # as specified in DEVELOPMENT.md Phase 0
        tokens = tokenizer.encode(text)
        tokens.append(tokenizer.eos_token_id)

        for token in tokens:
            current_chunk.append(token)
            current_length += 1
            if current_length == CPT_MAX_SEQ_LEN:
                packed_input_ids.append(current_chunk)
                current_chunk = []
                current_length = 0

    # Create the final HF Dataset
    print(f"Total packed sequences (length {CPT_MAX_SEQ_LEN}): {len(packed_input_ids)}")
    final_dataset = Dataset.from_dict({"input_ids": packed_input_ids})

    print(f"Saving to {CPT_PACKED_DATA_DIR}...")
    os.makedirs(CPT_PACKED_DATA_DIR, exist_ok=True)
    final_dataset.save_to_disk(CPT_PACKED_DATA_DIR)

    print("Done!")
