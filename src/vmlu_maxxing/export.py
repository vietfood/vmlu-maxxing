import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .consts import (
    BASE_MODEL,
    CPT_OUTPUT_DIR,
    FINAL_MODEL_DIR,
    KD_OUTPUT_DIR,
    SFT_OUTPUT_DIR,
)


def merge_adapters():
    """
    Export script designed to merge any existing Phase 0, Phase 2, and Phase 4
    LoRA adapters sequentially down into a single portable Float16 model.
    """
    print(f"Loading Base Model ({BASE_MODEL}) in full precision (bfloat16)...")

    # We load in bfloat16, NOT 4-bit, because merge_and_unload requires
    # dequantized native weights to correctly merge the adapter tensors.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Doing this on CPU to avoid massive VRAM usage during merge
        low_cpu_mem_usage=True,
    )

    # Merge Phase 0 CPT
    if os.path.exists(CPT_OUTPUT_DIR):
        print(f"Found Phase 0 adapter ({CPT_OUTPUT_DIR}). Loading and merging...")
        model = PeftModel.from_pretrained(model, CPT_OUTPUT_DIR)
        model = model.merge_and_unload()

    # Merge Phase 2 SFT
    if os.path.exists(SFT_OUTPUT_DIR):
        print(f"Found Phase 2 adapter ({SFT_OUTPUT_DIR}). Loading and merging...")
        model = PeftModel.from_pretrained(model, SFT_OUTPUT_DIR)
        model = model.merge_and_unload()

    # Merge Phase 4 Soft Label KD
    if os.path.exists(KD_OUTPUT_DIR):
        print(f"Found Phase 4 adapter ({KD_OUTPUT_DIR}). Loading and merging...")
        model = PeftModel.from_pretrained(model, KD_OUTPUT_DIR)
        model = model.merge_and_unload()

    print(
        f"All adapters merged. Saving final unquantized HF Model to {FINAL_MODEL_DIR}..."
    )

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("\\n🎉 VMLU Maxxing Complete: Model Exported Successfully! 🎉")


if __name__ == "__main__":
    merge_adapters()
