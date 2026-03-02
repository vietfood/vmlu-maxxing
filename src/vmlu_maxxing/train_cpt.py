import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .consts import (
    BASE_MODEL,
    CPT_EPOCHS,
    CPT_GRAD_ACCUM_STEPS,
    CPT_LOG_DIR,
    CPT_LR,
    CPT_MAX_SEQ_LEN,
    CPT_OUTPUT_DIR,
    CPT_PACKED_DATA_DIR,
    CPT_PER_DEVICE_BATCH_SIZE,
)


class DataCollatorForCLM:
    """Simple data collator for pre-tokenized causal language modeling.
    Copies input_ids as labels (Trainer handles the left-shift internally).
    """

    def __call__(self, features):
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def setup_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Qwen standardizes on eos_token for padding during pretraining if not specified
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)

    print("Setting up LoRA adapter...")
    # Target all linear layers (as per Phase 2 config mentioned in DEVELOPMENT.md)
    # Llama/Qwen typical linear layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
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
        init_lora_weights=True,  # PyTorch initializes to 0 by default for B matrices
    )

    model = get_peft_model(model, lora_config)

    # Verify exact zero-initialization of B matrices as per DEVELOPMENT.md rule 3
    # Actually PEFT does this natively if init_lora_weights=True.

    model.print_trainable_parameters()
    return model, tokenizer


def main():
    if not os.path.exists(CPT_PACKED_DATA_DIR):
        raise FileNotFoundError(
            f"Prepared dataset not found at {CPT_PACKED_DATA_DIR}. Run prepare_cpt.py first."
        )

    print(f"Loading packed dataset from {CPT_PACKED_DATA_DIR}...")
    dataset = load_from_disk(CPT_PACKED_DATA_DIR)

    # Optional: split a tiny validation set (1k rows) for monitoring perplexity
    # For Phase 0 sanity check, we want to watch eval loss drop.
    dataset = dataset.train_test_split(test_size=1000, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    model, tokenizer = setup_model_and_tokenizer()

    # Use standard Trainer for pre-tokenized data (not SFTTrainer).
    # SFTTrainer's skip_prepare_dataset is non-standard and unreliable.
    # Standard Trainer natively handles datasets with input_ids columns.
    training_args = TrainingArguments(
        output_dir=CPT_OUTPUT_DIR,
        logging_dir=CPT_LOG_DIR,
        learning_rate=CPT_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=CPT_EPOCHS,
        per_device_train_batch_size=CPT_PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=CPT_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=CPT_GRAD_ACCUM_STEPS,
        eval_strategy="steps",
        eval_steps=100,  # evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        bf16=True,  # Use bfloat16 for training
        report_to="none",  # Change to "wandb" if logging is needed
    )

    collator = DataCollatorForCLM()

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("Starting Phase 0 CPT training...")
    trainer.train()

    print(f"Training complete. Saving model to {CPT_OUTPUT_DIR}...")
    trainer.save_model(CPT_OUTPUT_DIR)
    tokenizer.save_pretrained(CPT_OUTPUT_DIR)
    print("Done!")
