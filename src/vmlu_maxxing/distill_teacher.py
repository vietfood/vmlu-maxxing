import asyncio
import math
import os
from typing import Dict, List

from datasets import Dataset, load_from_disk
from openai import AsyncOpenAI

from vmlu_maxxing.consts import (
    DISTILLED_SFT_DIR,
    FEW_SHOT_BANK_PATH,
    SFT_PACKED_DATA_DIR,
    SGLANG_HOST,
    TEACHER_MODEL_ID,
)

# SGLang uses the standard OpenAI API specification for its `/v1` endpoint.
client = AsyncOpenAI(
    base_url=SGLANG_HOST,
    api_key="EMPTY",  # SGLang doesn't require an API key by default
)


def _normalize_logprobs(
    logprobs_dict: Dict[str, float], num_choices: int
) -> List[float]:
    """
    Extracts logprobs for the exactly available choices (A, B, C, D, [E]).
    Applies Softmax to normalize them into a probability distribution summing to 1.
    """
    valid_letters = ["A", "B", "C", "D", "E"][:num_choices]

    # Sometimes tokenizers include a leading space like " A" instead of "A"
    # We check for exact matches or matches with leading spaces.
    raw_probs = []

    for letter in valid_letters:
        val = None
        # Try exact
        if letter in logprobs_dict:
            val = logprobs_dict[letter]
        # Try with leading space
        elif f" {letter}" in logprobs_dict:
            val = logprobs_dict[f" {letter}"]
        else:
            # Token not in top_logprobs (teacher highly confident it's wrong)
            # Assign a very low logprob
            val = -100.0

        raw_probs.append(val)

    # Softmax conversion: e^x_i / sum(e^x_j)
    # Using log-sum-exp trick for numerical stability
    max_logprob = max(raw_probs)
    scaled_probs = [math.exp(p - max_logprob) for p in raw_probs]
    sum_scaled = sum(scaled_probs)

    normalized_probs = [p / sum_scaled for p in scaled_probs]
    return normalized_probs


async def fetch_teacher_logprobs(item: dict) -> dict:
    """
    Calls the SGLang teacher model to get the logprobs for the next token after the prompt.
    """
    prompt = item["text"]
    num_choices = item["num_choices"]

    try:
        # We use standard Completions API instead of ChatCompletions because we
        # want the exact next token probability following "Đáp án: " without chat template injection.
        response = await client.completions.create(
            model=TEACHER_MODEL_ID,
            prompt=prompt,
            max_tokens=1,  # We only need the very next token (A, B, C, etc.)
            temperature=0.0,  # Greedy, though technically irrelevant for logprobs
            logprobs=10,  # Get top 10 token logprobs
            echo=False,  # Don't return the prompt
        )

        # Parse the OpenAI-compatible Logprobs object
        # SGLang populates this mimicking OpenAI's structure
        choice = response.choices[0]
        teacher_answer = choice.text.strip()

        # The logprobs for the first (and only) generated token
        # structure: top_logprobs is a List of dicts mapping token strings to logprob floats
        # openai.types.completion_choice.Logprobs
        top_logprobs_dict = choice.logprobs.top_logprobs[
            0
        ]  # Returns dictionary of str -> float

        if hasattr(top_logprobs_dict, "model_dump"):
            top_logprobs_dict = top_logprobs_dict.model_dump()

        normalized_distribution = _normalize_logprobs(top_logprobs_dict, num_choices)

        # Add to item
        item["teacher_logprobs"] = normalized_distribution
        item["teacher_answer"] = teacher_answer
        return item

    except Exception as e:
        print(f"Teacher distillation failed for prompt: {e}")
        return None


async def distill_dataset(dataset: Dataset, batch_size: int = 200) -> Dataset:
    """
    Distills an entire HuggingFace dataset iteratively through SGLang.
    """
    print(f"Connecting to SGLang Teacher: {TEACHER_MODEL_ID} at {SGLANG_HOST}...")

    items = dataset.to_list()
    processed_items = []

    # Process in async batches
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        tasks = [fetch_teacher_logprobs(item) for item in batch]

        results = await asyncio.gather(*tasks)

        valid_results = [res for res in results if res is not None]
        processed_items.extend(valid_results)

        print(
            f"Distilled batch {i // batch_size + 1}, Successful: {len(valid_results)}/{len(batch)}"
        )

    return Dataset.from_list(processed_items)


def main():
    if not os.path.exists(SFT_PACKED_DATA_DIR):
        raise FileNotFoundError(
            f"SFT Dataset missing at {SFT_PACKED_DATA_DIR}. Run prepare_sft.py first."
        )

    print(f"Loading Student SFT Dataset from {SFT_PACKED_DATA_DIR}...")
    dataset = load_from_disk(SFT_PACKED_DATA_DIR)

    # Run async loop
    distilled_dataset = asyncio.run(distill_dataset(dataset))

    print(f"Saving Distilled Dataset to {DISTILLED_SFT_DIR}...")
    os.makedirs(DISTILLED_SFT_DIR, exist_ok=True)
    distilled_dataset.save_to_disk(DISTILLED_SFT_DIR)
    print("Phase 3 Distillation Complete!")
