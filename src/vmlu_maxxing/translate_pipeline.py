import asyncio
import json
from typing import Dict, List

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from vmlu_maxxing.consts import SGLANG_HOST, TEACHER_MODEL_ID


# We enforce exactly the same formatting required by VMLU via Pydantic
class TranslatedMCQ(BaseModel):
    question: str = Field(description="The translated Vietnamese question")
    choices: List[str] = Field(
        description="The translated Vietnamese choices. Preserve the exact A., B., C., D. prefixes from the source text."
    )
    answer: str = Field(
        description="The exact choice letter of the answer (e.g. A, B, C, D). Do not translate the letter."
    )


def get_async_client(provider="sglang"):
    """
    Setup AsyncOpenAI client pointing to local SGLang Server.
    """
    if provider == "sglang":
        return AsyncOpenAI(
            base_url=SGLANG_HOST,
            api_key="EMPTY",
        )
    else:
        raise ValueError("Unsupported provider! Only 'sglang' is supported locally.")


async def _translate_single(client: AsyncOpenAI, model_name: str, item: Dict) -> Dict:
    """Async wrapper for a single translation"""
    choices_str = "\n".join(item["choices"])

    system_prompt = (
        "You are an expert English-to-Vietnamese translator specialized in maintaining "
        "exact multiple-choice exam formatting. Your output MUST be valid JSON fitting the schema:\n"
        "{\n"
        '  "question": "The translated Vietnamese question",\n'
        '  "choices": ["The translated choices...", "Preserving exactly A., B., C., D. prefixes"],\n'
        '  "answer": "The exact choice letter of the answer (e.g. A, B, C, D)."\n'
        "}"
    )

    user_prompt = (
        "Translate the following multiple choice question into Vietnamese. Keep any LaTeX or code blocks intact.\n\n"
        f"Question: {item['question']}\n"
        f"Choices:\n{choices_str}\n"
        f"Answer Key: {item['answer']}"
    )

    try:
        response = await client.chat.completions.create(
            model=model_name or TEACHER_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = response.choices[0].message.content
        result_dict = json.loads(content)

        # Validate against our Pydantic schema to ensure correctness
        result = TranslatedMCQ(**result_dict)

        return {
            "question": result.question,
            "choices": result.choices,
            "answer": result.answer,
            "subject": item["subject"],  # Pass through original metadata
        }
    except Exception as e:
        print(f"Translation failed for item: {e}")
        return None


async def translate_dataset_batch(
    dataset: List[Dict], provider="sglang", model_name=None, batch_size=50
) -> List[Dict]:
    """
    Translates an entire ingested list of multiple-choice dictionaries asynchronously.
    """
    client = get_async_client(provider)
    model_id = model_name or TEACHER_MODEL_ID
    translated_results = []

    print(f"Starting {provider} ({model_id}) translation for {len(dataset)} items...")

    # We batch async calls to avoid rate limiting
    for i in tqdm(range(0, len(dataset), batch_size), desc="Translating batches"):
        batch = dataset[i : i + batch_size]
        tasks = [_translate_single(client, model_id, item) for item in batch]

        # Execute batch and wait
        results = await asyncio.gather(*tasks)

        # Filter out failures
        valid_results = [res for res in results if res is not None]
        translated_results.extend(valid_results)

        # Small delay between batches
        await asyncio.sleep(1.0)

    return translated_results


def translate_sync(
    dataset: List[Dict], provider="sglang", model_name=None, batch_size=50
):
    """Synchronous wrapper for script execution."""
    return asyncio.run(
        translate_dataset_batch(dataset, provider, model_name, batch_size)
    )
