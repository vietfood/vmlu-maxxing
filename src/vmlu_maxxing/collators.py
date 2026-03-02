from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForMCQ:
    """
    Data collator that masks the question so the model is only penalized
    on the answer tokens.

    Expects input features to contain 'input_ids' and 'attention_mask'.
    Finds the separator token ("Đáp án:" or "Answer:") and masks everything before it
    with -100 in the `labels`.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None
    label_pad_token_id: int = -100

    # Supported answer separators in the preprocessed SFT dataset
    separators: tuple = ("Đáp án:", "Answer:")

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle padding using standard fast tokenizer logic
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # We clone input_ids as the initial labels
        labels = batch["input_ids"].clone()

        # Tokenize the separators without adding special tokens (like <s>)
        # So we can search for the sequence of tokens within each row
        sep_token_sequences = [
            self.tokenizer.encode(sep, add_special_tokens=False)
            for sep in self.separators
        ]

        for i in range(len(labels)):
            # By default, mask the entire sequence
            answer_idx = -1
            seq = labels[i].tolist()

            # Find the separator
            for sep_seq in sep_token_sequences:
                sep_len = len(sep_seq)
                for j in range(len(seq) - sep_len + 1):
                    if seq[j : j + sep_len] == sep_seq:
                        # Found the 'Đáp án:' sequence! The answer tokens start immediately AFTER this separator.
                        answer_idx = j + sep_len
                        break
                if answer_idx != -1:
                    break

            if answer_idx != -1:
                # Mask everything up to and including the separator
                labels[i, :answer_idx] = self.label_pad_token_id
            else:
                # Fallback: if no separator found, mask the whole thing to avoid training on garbage
                labels[i, :] = self.label_pad_token_id

            # Always mask padding tokens
            labels[i, batch["attention_mask"][i] == 0] = self.label_pad_token_id

        batch["labels"] = labels
        return batch
