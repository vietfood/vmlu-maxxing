import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from .collators import DataCollatorForMCQ
from .consts import (
    BASE_MODEL,
    CPT_OUTPUT_DIR,
    SFT_EPOCHS,
    SFT_GRAD_ACCUM_STEPS,
    SFT_LOG_DIR,
    SFT_LR,
    SFT_MAX_SEQ_LEN,
    SFT_OUTPUT_DIR,
    SFT_PACKED_DATA_DIR,
    SFT_PER_DEVICE_BATCH_SIZE,
    SFT_WEIGHT_DECAY,
    VMLU_RAW_DIR,
)
from .prepare_sft import load_jsonl


def get_sft_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Qwen standardizes on eos_token for padding during pretraining if not specified
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model in 4-bit...")
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
        attn_implementation="flash_attention_2",  # Optimized for A6000/5090 (Ampere/Ada/Blackwell)
    )

    # Enable gradient checkpointing to save VRAM and allow potentially larger batches
    model.gradient_checkpointing_enable()

    # BEST APPROACH FOR QLORA ADAPTER STACKING
    # Merging 4-bit bases dynamically degrades precision and speed.
    # We load the Phase 0 CPT adapter (if it exists) as a non-trainable base.
    # Then we add a new SFT adapter on top of it.

    print("Applying QLoRA config...")
    model = prepare_model_for_kbit_training(model)

    if os.path.exists(CPT_OUTPUT_DIR):
        print(
            f"Discovered Phase 0 CPT Adapter at {CPT_OUTPUT_DIR}. Loading it as non-trainable base..."
        )
        # Load CPT adapter
        model = PeftModel.from_pretrained(
            model, CPT_OUTPUT_DIR, adapter_name="cpt_adapter", is_trainable=False
        )
        model.train()  # Make sure peft model is in train mode overall

        # Define new SFT adapter
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        print("Stacking new SFT adapter on top of CPT...")
        model.add_adapter("sft_adapter", lora_config)
        model.set_adapter("sft_adapter")  # Set as actively trainable
    else:
        print(
            f"No CPT Adapter found at {CPT_OUTPUT_DIR}. Proceeding with base model only."
        )
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer


def main():
    if not os.path.exists(SFT_PACKED_DATA_DIR):
        raise FileNotFoundError(
            f"Prepared SFT dataset not found at {SFT_PACKED_DATA_DIR}. Run prepare_sft.py first."
        )

    model, tokenizer = get_sft_model_and_tokenizer()

    print(f"Loading packed SFT dataset from {SFT_PACKED_DATA_DIR}...")
    train_dataset = load_from_disk(SFT_PACKED_DATA_DIR)

    # Load validation split for checkpoint selection
    valid_path = os.path.join(VMLU_RAW_DIR, "valid.jsonl")
    valid_dataset = None
    if os.path.exists(valid_path):
        print("Prepping validation split...")
        from .prepare_sft import format_mcq

        valid_raw = load_jsonl(valid_path)
        valid_processed = {
            "text": [],
            "answer_token_id": [],
            "subject": [],
            "num_choices": [],
        }
        for row in valid_raw:
            answer = row["answer"]
            valid_processed["text"].append(
                format_mcq(row["question"], row["choices"], answer)
            )
            ans_token_str = f" {answer}"
            ans_token_ids = tokenizer.encode(ans_token_str)
            valid_processed["answer_token_id"].append(
                ans_token_ids[-1] if ans_token_ids else -100
            )
            valid_processed["subject"].append(row.get("subject", "general"))
            valid_processed["num_choices"].append(len(row["choices"]))
        from datasets import Dataset

        valid_dataset = Dataset.from_dict(valid_processed)

    # Setup custom masked collator
    collator = DataCollatorForMCQ(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        logging_dir=SFT_LOG_DIR,
        learning_rate=SFT_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=SFT_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUM_STEPS,
        weight_decay=SFT_WEIGHT_DECAY,
        gradient_checkpointing=True,  # Saves VRAM significantly
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if valid_dataset else "no",
        eval_steps=500 if valid_dataset else None,
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        max_seq_length=SFT_MAX_SEQ_LEN,
        data_collator=collator,
        dataset_text_field="text",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    print("Starting Phase 2 SFT training...")
    trainer.train()

    print(f"Training complete. Saving model to {SFT_OUTPUT_DIR}...")
    trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
    print("Done!")
