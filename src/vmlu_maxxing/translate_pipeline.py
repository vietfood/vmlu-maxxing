import asyncio
import json
import os
from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Support both OpenAI and Google GenAI via LangChain as requested
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm


# We enforce exactly the same formatting required by VMLU via Pydantic
class TranslatedMCQ(BaseModel):
    question: str = Field(description="The translated Vietnamese question")
    choices: List[str] = Field(
        description="The translated Vietnamese choices. Preserve the exact A., B., C., D. prefixes from the source text."
    )
    answer: str = Field(
        description="The exact choice letter of the answer (e.g. A, B, C, D). Do not translate the letter."
    )


def get_llm_chain(provider="openai", model_name="gpt-4o-mini"):
    """
    Setup Langchain model with Pydantic structured output parser to strictly enforce JSON output format
    """
    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=0.1)
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
    else:
        raise ValueError("Unsupported provider! Use 'openai' or 'google'")

    # Setup prompt
    parser = PydanticOutputParser(pydantic_object=TranslatedMCQ)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert English-to-Vietnamese translator specialized in maintaining exact multiple-choice exam formatting. Your output MUST be valid JSON fitting the provided schema.\n{format_instructions}",
            ),
            (
                "user",
                "Translate the following multiple choice question into Vietnamese. Keep any LaTeX or code blocks intact.\n\nQuestion: {question}\nChoices:\n{choices}\nAnswer Key: {answer}",
            ),
        ]
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # LangChain expression language (LCEL) chain
    chain = prompt | llm | parser
    return chain


async def _translate_single(chain, item: Dict) -> Dict:
    """Async wrapper for a single translation"""
    choices_str = "\n".join(item["choices"])
    try:
        # LangChain invoke
        result: TranslatedMCQ = await chain.ainvoke(
            {
                "question": item["question"],
                "choices": choices_str,
                "answer": item["answer"],
            }
        )

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
    dataset: List[Dict], provider="openai", model_name="gpt-4o-mini", batch_size=50
) -> List[Dict]:
    """
    Translates an entire ingested list of multiple-choice dictionaries asynchronously.
    """
    chain = get_llm_chain(provider, model_name)
    translated_results = []

    print(f"Starting {provider} ({model_name}) translation for {len(dataset)} items...")

    # We batch async calls to avoid rate limiting
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        tasks = [_translate_single(chain, item) for item in batch]

        # Execute batch and wait
        results = await asyncio.gather(*tasks)

        # Filter out failures
        valid_results = [res for res in results if res is not None]
        translated_results.extend(valid_results)

        print(
            f"Processed batch {i // batch_size + 1}, successful: {len(valid_results)}/{len(batch)}"
        )

        # Small delay between batches
        await asyncio.sleep(1.0)

    return translated_results


def translate_sync(
    dataset: List[Dict], provider="openai", model_name="gpt-4o-mini", batch_size=50
):
    """Synchronous wrapper for script execution."""
    return asyncio.run(
        translate_dataset_batch(dataset, provider, model_name, batch_size)
    )
