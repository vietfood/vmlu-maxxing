import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.vmlu_maxxing.distill_teacher import _normalize_logprobs, fetch_teacher_logprobs


class TestDistillTeacher(unittest.IsolatedAsyncioTestCase):
    def test_normalize_logprobs_exact(self):
        # Teacher is 100% sure it's A
        raw = {"A": 0.0, "B": -10.0, "C": -10.0, "D": -10.0}
        probs = _normalize_logprobs(raw, num_choices=4)

        self.assertAlmostEqual(probs[0], 1.0, places=3)
        self.assertAlmostEqual(probs[1], 0.0, places=3)
        self.assertEqual(len(probs), 4)

    def test_normalize_logprobs_with_spaces(self):
        # Qwen tokenizer often predicts ' A'
        raw = {" A": -0.1, " B": -2.3, " C": -10.0, " D": -10.0}
        probs = _normalize_logprobs(raw, num_choices=4)

        # ' A' should command majority ~90%+
        self.assertTrue(probs[0] > 0.8)
        self.assertTrue(probs[1] > 0.05)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_normalize_logprobs_missing_tokens(self):
        # Teacher only outputs top 10, maybe C and D aren't even in the top 10
        raw = {"A": -0.5, "B": -1.2}
        probs = _normalize_logprobs(raw, num_choices=4)

        # Softmax should still work, C and D get -100 (essentially 0 prob)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(probs[2] < 1e-10)
        self.assertTrue(probs[3] < 1e-10)


if __name__ == "__main__":
    unittest.main()
