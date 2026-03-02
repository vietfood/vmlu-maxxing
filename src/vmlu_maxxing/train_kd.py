import os

import torch
import torch.nn.functional as F
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
    DISTILLED_SFT_DIR,
    KD_ALPHA,
    KD_EPOCHS,
    KD_LOG_DIR,
    KD_LR,
    KD_OUTPUT_DIR,
    KD_TEMPERATURE,
    SFT_GRAD_ACCUM_STEPS,
    SFT_MAX_SEQ_LEN,
    SFT_OUTPUT_DIR,
    SFT_PER_DEVICE_BATCH_SIZE,
    SFT_WEIGHT_DECAY,
    VMLU_RAW_DIR,
)


class KDTrainer(SFTTrainer):
    def __init__(self, *args, alpha=0.7, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom Knowledge Distillation Loss
        Computes the standard Cross Entropy on the hard label + KL Divergence
        on the teacher's soft probability distribution for the available choices.
        """
        # 1. Pop the teacher distribution provided by distill_teacher.jsonl
        # The teacher_logprobs array is added to the batch parallel to our standard SFT dataset variables.
        # But we need to ensure the custom dataset columns are available in the trainer collator
        # (transformers strips non-model columns by default, so we must tell it not to).
        teacher_probs = inputs.pop("teacher_logprobs", None)

        # We also pop the answer mask, since standard CausalLM just does shifted cross-entropy on labels.
        # It's computationally tricky to intercept the HF loss pipeline directly, so we run the forward pass
        # and calculate loss ourselves.
        labels = inputs.pop("labels", None)

        # 2. Forward pass Student
        outputs = model(**inputs)
        logits = outputs.get("logits")  # Shape: (batch, seq_len, vocab)

        # 3. Standard CausalLM CE Loss (calculated just on the Unmasked tokens - i.e. our Answer labels)
        ce_loss_fct = torch.nn.CrossEntropyLoss()

        # Shift logits and labels by 1 for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate standard SFT Loss
        student_ce_loss = ce_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        loss = student_ce_loss

        # 4. KD Loss calculation (If teacher distribution is present)
        # It requires isolating the exactly selected student tokens that map to the answer
        if teacher_probs is not None and self.alpha > 0.0:
            # Find the index where the label is not -100 (this is the answer token index)
            # Shape of labels: (batch, seq_len)
            # Get the indices across the batch dimension
            batch_size = shift_labels.size(0)

            kd_loss_sum = 0.0
            valid_kd_samples = 0

            for b in range(batch_size):
                # Find where this sequence's answer token lies
                valid_indices = torch.where(shift_labels[b] != -100)[0]
                if len(valid_indices) == 0:
                    continue

                # Usually there's only 1 answer token we care about the distribution over.
                # It's the first non-masked prediction token (e.g. 'A' or ' B' etc).
                target_idx = valid_indices[0]

                # Student logits for the next-token prediction
                student_vocab_logits = shift_logits[
                    b, target_idx
                ]  # Shape: (vocab_size)

                # We only want to compute KL divergence over the choices A, B, C, D (E).
                # We need to map the teacher's 4/5 probabilities to the student's equivalent token IDs.
                # Since teacher_probs is already normalized to 1.0, we just need to isolate the exact 5
                # tokens the student uses.
                # Since Qwen uses space delimiters for multiple choice (" A", " B"), we look those up.
                # Alternatively we can compute full-vocab KL divergence, but the teacher only gave us top 5.

                # For robust implementation, if we don't have exactly mapped token indexes, we
                # compute KLD purely on the dimension of num_choices.
                # Dynamically build letter list based on this question's actual choice count.
                num_choices = len(teacher_probs[b])
                if num_choices > 5:
                    num_choices = 5  # Safety cap

                letters = [" A", " B", " C", " D", " E"][:num_choices]
                choice_token_ids = [
                    self.tokenizer.encode(letter)[-1] for letter in letters
                ]

                # Extract student logits for JUST these tokens
                student_choice_logits = student_vocab_logits[choice_token_ids]

                # KLDivergence Input: Log Softmax of Student
                q_student = F.log_softmax(
                    student_choice_logits / self.temperature, dim=-1
                )

                # Target: Softmax of Teacher (teacher_probs is already normalized softmax, so just scale log of it?)
                # The developer formula: D_KL(P || Q).
                # teacher_probs is already a softmax distribution over the choices.
                # PyTorch kl_div expects `input` to be log-probabilities and `target` to be probabilities
                p_teacher = teacher_probs[b][:num_choices].to(q_student.device)

                # Add a tiny epsilon to prevent log(0) if teacher is 100% / 0%
                p_teacher = torch.clamp(p_teacher, min=1e-8)
                p_teacher = p_teacher / p_teacher.sum()  # renormalize

                # Apply formula
                # Use reduction='sum' because q_student/p_teacher are 1D (not batched).
                # 'batchmean' would divide by num_choices, deflating KD loss.
                # We manually average across the batch below.
                kd = F.kl_div(q_student, p_teacher, reduction="sum") * (
                    self.temperature**2
                )

                kd_loss_sum += kd
                valid_kd_samples += 1

            if valid_kd_samples > 0:
                avg_kd_loss = kd_loss_sum / valid_kd_samples
                loss = (self.alpha * avg_kd_loss) + ((1 - self.alpha) * student_ce_loss)

        return (loss, outputs) if return_outputs else loss


def get_kd_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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
        attn_implementation="flash_attention_2",
    )

    model.gradient_checkpointing_enable()

    print("Applying QLoRA config...")
    model = prepare_model_for_kbit_training(model)

    # KD depends on the student already knowing SFT.
    # Wrap base model in Phase 2 SFT adapter and freeze it to preserve SFT.
    if os.path.exists(SFT_OUTPUT_DIR):
        print(
            f"Loading Phase 2 SFT adapter from {SFT_OUTPUT_DIR} as non-trainable base..."
        )
        model = PeftModel.from_pretrained(
            model, SFT_OUTPUT_DIR, adapter_name="sft_adapter", is_trainable=False
        )
        model.train()

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
        print("Stacking Phase 4 KD adapter...")
        model.add_adapter("kd_adapter", lora_config)
        model.set_adapter("kd_adapter")
    else:
        print(
            f"SFT Adapter not found at {SFT_OUTPUT_DIR}. Falling back to clean QLoRA..."
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
    if not os.path.exists(DISTILLED_SFT_DIR):
        raise FileNotFoundError(
            f"Distilled SFT dataset missing at {DISTILLED_SFT_DIR}. Run distill_teacher.py first."
        )

    model, tokenizer = get_kd_model_and_tokenizer()

    print(f"Loading distilled dataset from {DISTILLED_SFT_DIR}...")
    train_dataset = load_from_disk(DISTILLED_SFT_DIR)

    collator = DataCollatorForMCQ(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = SFTConfig(
        output_dir=KD_OUTPUT_DIR,
        logging_dir=KD_LOG_DIR,
        learning_rate=KD_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=KD_EPOCHS,
        per_device_train_batch_size=SFT_PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=SFT_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUM_STEPS,
        weight_decay=SFT_WEIGHT_DECAY,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=SFT_MAX_SEQ_LEN,
        data_collator=collator,
        dataset_text_field="text",
        remove_unused_columns=False,  # CRUCIAL: Retain teacher_logprobs array during collation
        bf16=True,
        report_to="none",
    )

    trainer = KDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        alpha=KD_ALPHA,
        temperature=KD_TEMPERATURE,
    )

    print("Starting Phase 4 Distillation Training...")
    trainer.train()

    print(f"Training complete. Saving model to {KD_OUTPUT_DIR}...")
    trainer.save_model(KD_OUTPUT_DIR)
    tokenizer.save_pretrained(KD_OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
