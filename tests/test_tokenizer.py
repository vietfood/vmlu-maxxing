import unittest

from transformers import AutoTokenizer


class TestTokenizerMasking(unittest.TestCase):
    def setUp(self):
        # Mocks the behavior we expect from the pipeline without needing the downloaded model
        # if run in a CI pipeline, but here we assume Qwen3 is available locally.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        except Exception:
            self.skipTest("Qwen3-1.7B not downloaded locally yet")

    def test_separator_tokenization(self):
        """
        Byte-Pair Encoding (BPE) tokenizers like Qwen/Llama often absorb leading spaces or newlines.
        We need to ensure that our search for 'Đáp án:' isn't broken by a preceding newline.
        """
        text_with_newline = "Câu hỏi: 1 + 1 = ?\nA. 1\nB. 2\nĐáp án: B"
        tokens = self.tokenizer.encode(text_with_newline)

        # Test 1: Does encode('Đáp án:', add_special_tokens=False) correctly match a sub-slice of the full text?
        sep_tokens_strict = self.tokenizer.encode("Đáp án:", add_special_tokens=False)
        sep_tokens_newline = self.tokenizer.encode(
            "\nĐáp án:", add_special_tokens=False
        )

        found_strict = self._find_sublist(tokens, sep_tokens_strict)
        found_newline = self._find_sublist(tokens, sep_tokens_newline)

        # We expect at least the strict one to be found, or the newline one.
        # If neither is found, our naive masking logic in collators.py is flawed!
        self.assertTrue(
            found_strict != -1 or found_newline != -1,
            "DataCollator separator masking will fail! Tokenizer absorbed the boundary differently.",
        )

    def _find_sublist(self, sequence, sublist):
        for i in range(len(sequence) - len(sublist) + 1):
            if sequence[i : i + len(sublist)] == sublist:
                return i
        return -1


if __name__ == "__main__":
    unittest.main()
