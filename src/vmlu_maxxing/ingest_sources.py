import random

from datasets import load_dataset


def ingest_mmlu_split(split="dev"):
    """
    Ingest cais/mmlu task splits.
    Returns format: [{'question': str, 'choices': [str], 'answer': str (e.g. 'A'), 'subject': str}]
    """
    print(f"Loading MMLU split: {split}")
    # 'all' loads everything. It will take a bit on first download.
    dataset = load_dataset("cais/mmlu", "all", split=split)

    results = []
    # Map label (int 0..3) to A, B, C, D
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for row in dataset:
        results.append(
            {
                "question": row["question"],
                "choices": [
                    f"{label_map[i]}. {c}" for i, c in enumerate(row["choices"])
                ],
                "answer": label_map[row["answer"]],
                "subject": row.get("subject", "mmlu_general"),
            }
        )
    return results


def ingest_arc_split(arc_type="ARC-Challenge", split="train"):
    """
    Ingest ARC Challenge/Easy splits.
    arc_type: 'ARC-Challenge' or 'ARC-Easy'
    """
    print(f"Loading ARC {arc_type} split: {split}")
    dataset = load_dataset("allenai/ai2_arc", arc_type, split=split)

    results = []
    # answerKey is usually typically "A", "B", "C", "D" or "1", "2", "3", "4"
    # To keep aligned with VMLU, we map 1,2,3,4 to A,B,C,D where needed.
    for row in dataset:
        raw_choices = row["choices"]["text"]
        raw_labels = row["choices"]["label"]
        raw_answer = row["answerKey"]

        # Determine correct answer index
        try:
            ans_idx = raw_labels.index(raw_answer)
        except ValueError:
            # Skip malformed row
            continue

        # Enforce A,B,C,D labelling layout always
        label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

        # Only process if 1-5 choices
        if len(raw_choices) > 5 or len(raw_choices) == 0:
            continue

        choices = [f"{label_map[i]}. {text}" for i, text in enumerate(raw_choices)]

        results.append(
            {
                "question": row["question"],
                "choices": choices,
                "answer": label_map[ans_idx],
                "subject": "science_arc",
            }
        )
    return results


def ingest_sciq_split(split="train"):
    """
    Ingest SciQ splits. SciQ has 1 correct answer and 3 distractors.
    We randomize the position of the correct answer.
    """
    print(f"Loading SciQ split: {split}")
    dataset = load_dataset("allenai/sciq", split=split)

    results = []
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for row in dataset:
        distractors = [row["distractor1"], row["distractor2"], row["distractor3"]]
        correct_ans = row["correct_answer"]

        # Build choices
        all_options = distractors + [correct_ans]
        random.shuffle(all_options)

        ans_idx = all_options.index(correct_ans)

        choices = [f"{label_map[i]}. {text}" for i, text in enumerate(all_options)]

        # For translation context, sometimes `support` helps, but we want just Q/A for MCQ prompt
        results.append(
            {
                "question": row["question"],
                "choices": choices,
                "answer": label_map[ans_idx],
                "subject": "science_sciq",
            }
        )

    return results


def ingest_vimmrc_split(split="train"):
    """
    Ingest ViMMRC Reading Comprehension.
    ViMMRC has articles mapping to multiple questions. We construct questions containing the passage context.
    """
    print(f"Loading ViMMRC 2.0 split: {split}")
    dataset = load_dataset("sonlam1102/vimmrc2.0", split=split)

    results = []
    label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    for row in dataset:
        article = row["article"]
        questions = row["questions"]
        options_list = row["options"]  # list of lists
        answers = row["answers"]

        # Usually length of questions == length of options_list == length of answers
        for i in range(len(questions)):
            if i >= len(options_list) or i >= len(answers):
                break

            q_text = questions[i]
            raw_opts = options_list[i]
            ans_text = answers[i]

            # Construct complete question text
            full_question = f"Đoạn văn:\n{article}\n\nDựa vào đoạn văn, trả lời câu hỏi sau:\n{q_text}"

            try:
                ans_idx = raw_opts.index(ans_text)
            except ValueError:
                continue

            choices = [f"{label_map[j]}. {opt}" for j, opt in enumerate(raw_opts)]

            results.append(
                {
                    "question": full_question,
                    "choices": choices,
                    "answer": label_map[ans_idx],
                    "subject": "reading_comprehension",
                }
            )

    return results


def _generate_synthetic_distractor(text: str) -> str:
    """
    Generate a simple synthetic typo by randomly replacing or dropping a character.
    Very basic heuristic for generating distractors.
    """
    if len(text) < 5:
        return text

    chars = list(text)
    # Pick a random index, avoid first and last few chars to not break structure obviously
    idx = random.randint(2, len(chars) - 3)

    # Common Vietnamese typo replacements (vowels, tones)
    replacements = ["a", "e", "i", "o", "u", "y", "h", "g", "n", "m", "t", "c"]

    # 50% chance to replace, 50% chance to drop
    if random.random() < 0.5:
        chars[idx] = random.choice(replacements)
    else:
        chars.pop(idx)

    return "".join(chars)


def ingest_vsec(split="train"):
    """
    Ingest VSEC from HF.
    Option 2: Identify the correctly spelled sentence.
    Generates synthetic distractors to ensure 4 choices.
    """
    print(f"Loading VSEC split: {split}")
    dataset = load_dataset(
        "nguyenthanhasia/vsec-vietnamese-spell-correction", split=split
    )

    results = []
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for row in dataset:
        if not row.get("has_errors", False):
            continue

        original = row["text"]  # contains actual error
        corrected = row["corrected_text"]

        # Distractors formulation
        # 1 correct option: `corrected`
        # 1 real error option: `original`
        # 2 synthetic error options
        synth1 = _generate_synthetic_distractor(corrected)
        synth2 = _generate_synthetic_distractor(corrected)

        # If synth happens to be identical (rare, but possible), it's fine for now, or we can use set()
        options = list(set([corrected, original, synth1, synth2]))

        # Pad with more synthetics if set deduplication dropped choices
        while len(options) < 4:
            options.append(_generate_synthetic_distractor(corrected))
            options = list(set(options))

        # If it has more than 4, trim it down
        options = options[:4]
        random.shuffle(options)

        ans_idx = options.index(corrected)
        choices = [f"{label_map[j]}. {opt}" for j, opt in enumerate(options)]

        results.append(
            {
                "question": "Hãy chọn câu viết đúng chính tả tiếng Việt:",
                "choices": choices,
                "answer": label_map[ans_idx],
                "subject": "vietnamese_spelling",
            }
        )

    return results
