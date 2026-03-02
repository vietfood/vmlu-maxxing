## Philosophy

1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Global Constraints & "War Story" Warnings for the Agent

Before you write a single line of PyTorch, internalize these rules:
1. **VMLU has variable choice counts — primarily 4 (A–D), rarely 5+ (A–E).** Stats: 5+ choices appear in only 6/745 valid questions (<1%) and 249/10000 test questions (~2.5%). Some questions have only 3 choices. **Do NOT hardcode any fixed choice width.** Read `len(choices)` per question and dynamically size logit slices, softmax, KD targets, etc. The VMLU JSON format is: `{"id": "...", "question": "...", "choices": ["A. ...", "B. ...", ...], "answer": "B"}`
2. **No Sequence-Length MSE:** Vietnamese and English prompt translations have different token counts. *Never* attempt an element-wise MSE on intermediate hidden states across languages without pooling. We will rely strictly on Logit-based Knowledge Distillation (KD) for the translation mapping.
3. **Zero-Initialization is Mandatory:** If using any custom adapter, the final linear layer *must* be initialized to exactly zero (`nn.init.zeros_`). If you use default Kaiming or Xavier init, you will inject Gaussian noise directly into the residual stream of a frozen LLM. The loss will instantly go to NaN.
4. **Strict Label Masking:** When training multiple-choice, prompt tokens must have their labels set to `-100` in PyTorch. Backpropagating on the question text wastes model capacity. We only optimize the answer token.
5. **LaTeX in questions is common.** Math-related VMLU questions contain LaTeX (e.g., `$18,073\$$`). The tokenizer will tokenize LaTeX tokens differently than plain text. Never strip or preprocess LaTeX — the model must learn to handle it as-is.
6. **Cross-reference choices exist.** Some VMLU choices refer to other choices, e.g., `"D. Cả A và C"` ("D: both A and C") or `"E. Không có đáp án đúng"` ("E: None of the above"). These are NOT bugs — they are a real pattern in the data. The model must learn these compositional answer patterns.

### VMLU Dataset Sizes
| Split | Size | Usage |
|---|---|---|
| Dev | 304 questions | Prompt template selection, few-shot example bank |
| Valid | 745 questions | Hyperparameter tuning, checkpoint selection |
| Test | 10,000 questions | **Final evaluation only** — never train on this |

---

### What Real Benchmaxxing Looks Like

The top VMLU entries use:
- **GPT-4-Turbo**: 81.26% (massive model, just brute-forces it)
- **Vi-Sovereign-Medium**: 80.57% (Vietnamese-specialized, likely continued pretraining + RLHF)
- **QwQ-32B**: 76.13% (reasoning-focused, 32B params)

For a **1.7B model** to compete, you need to be ruthlessly pragmatic:
1. **Direct Vietnamese SFT** (biggest bang for buck)
2. **LoRA/QLoRA** (proven, battle-tested, not research-grade)
3. **VMLU-format training data** (train on what you'll be tested on)
4. **Knowledge distillation from a strong teacher** (but on Vietnamese directly, not translated English)
5. **Few-shot prompting at eval time** (free accuracy)
6. **Subject-specific strategies** (some subjects benefit more from certain approaches)

---

## PHASE 0: Vietnamese Continued Pretraining (CPT)
**Objective:** Deepen Qwen3-1.7B's Vietnamese language understanding before teaching it the MCQ task. The model has Vietnamese in its pretraining mix, but it's diluted across 100+ languages. CPT concentrates the Vietnamese signal.

**Data Source:** [`VTSNLP/vietnamese_curated_dataset`](https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset) — 12.2M rows of curated Vietnamese text, 99.6% between 199–38.8k characters.

> [!IMPORTANT]
> This data is in **raw paragraph format** (pretraining-style), NOT MCQ. We use it for **continued pretraining** (next-token prediction on Vietnamese text), not SFT. The goal is to strengthen the model's Vietnamese "muscles" — vocabulary, grammar, factual knowledge — before teaching it the MCQ format in Phase 2.

**Task 0.1: Data Preprocessing & Sampling**
*   **Full dataset is too large** — 12.2M rows × ~2k avg chars ≈ ~25B characters ≈ ~8-10B tokens. Training on all of it would take weeks on a single GPU and risks overfitting to this distribution.
*   **Sampling Strategy:**
    1. Filter out rows < 200 chars (noise) and > 20k chars (truncation issues)
    2. Deduplicate exact matches (MinHash or exact string)
    3. **Sample 500k–1M rows** (target ~500M–1B tokens). This is enough for meaningful CPT without excessive compute.
    4. **Content prioritization** — simple keyword matching (`"khoa học"`, `"lịch sử"`) is **not sufficient**. Vietnamese text often discusses educational topics without using those exact keywords. Instead:
        *   **Option A (cheap):** Use a fast Vietnamese text classifier (e.g., `underthesea` library's topic classifier) to tag rows as educational/news/academic vs. conversational/social-media. Prioritize the former.
        *   **Option B (better):** Compute sentence embeddings (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) for a few hundred seed examples from VMLU questions, then select CPT rows with high cosine similarity to the VMLU embedding centroid. This biases CPT toward the same "knowledge domain" that VMLU tests.
        *   **Option C (simplest fallback):** Skip filtering entirely — just random sample. At 500k–1M rows, the dataset is diverse enough that random sampling works. The model will see some noise, but CPT is robust to it.
*   **Tokenization:** Use Qwen3-1.7B's tokenizer. Pack sequences to `max_length=2048` with `eos` separators between documents.
*   **Deliverable:** A tokenized, packed dataset ready for causal LM training.

**Task 0.2: CPT Training**
*   **Method:** QLoRA (same config as Phase 2 — keeps things consistent and allows merging later)
*   **Loss:** Standard causal language modeling (next-token prediction on ALL tokens — no masking)
    $$ \mathcal{L}_{\text{CPT}} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid \mathbf{x}_{<t}) $$
*   **Hyperparameters:**
    | Param | Value | Rationale |
    |---|---|---|
    | Learning Rate | $5 \times 10^{-5}$ | Lower than SFT — we're refining, not reformatting |
    | Schedule | Cosine with 3% warmup | Standard |
    | Epochs | 1 | One pass is enough for CPT; more risks forgetting English |
    | Batch Size | 64 (effective) | Moderate — next-token prediction is stable |
    | Max Seq Length | 2048 | Long enough for paragraphs, fits in memory |
*   **Monitoring:** Track perplexity on a held-out Vietnamese validation set (1k rows). If perplexity drops then plateaus, stop early.

**Task 0.3: Sanity Check**
*   After CPT, verify the model hasn't forgotten English or MCQ capabilities:
    *   Generate 10 Vietnamese paragraphs — should be fluent, coherent
    *   Run 5-shot MMLU (English) — should not have degraded more than 1-2% from base
    *   If English degrades significantly, reduce CPT data to 250k rows and retrain
*   **Save the CPT checkpoint** — this becomes the base for all subsequent phases

> [!TIP]
> **Compute estimate:** 500k rows × 2048 tokens × 1 epoch ≈ 1B tokens. On a single A100 (80GB) with QLoRA, this takes ~4-6 hours. On a 4090 (24GB), ~12-18 hours. Very doable.

---

## PHASE 1: Data Curation & Vietnamese MCQ Corpus
**Objective:** Build a comprehensive Vietnamese multiple-choice training corpus. This is the foundation — garbage in, garbage out.

**Task 1.1: Collect & Unify Vietnamese MCQ Data**
*   **Primary Sources:**
    *   VMLU train split (this IS the target distribution — use it)
    *   Vietnamese translations of MMLU, ARC-Challenge, SciQ (from existing HuggingFace datasets or translate via GPT-4o-mini batch API — it's cheap)
    *   ViMMRC, VSEC, and any other Vietnamese MCQ datasets available
*   **Secondary Sources (English, for format diversity):**
    *   English MMLU, ARC-Challenge, SciQ train splits
*   **Target:** 50k–100k Vietnamese MCQ examples + 20k–50k English MCQ examples

**Task 1.2: Strict Format Pipeline**
*   **Format:** Normalize ALL data to match VMLU's native format. **Do NOT pad or force a fixed choice count** — preserve the original number of choices per question:
    ```text
    Câu hỏi: {question}
    A. {choice_A}
    B. {choice_B}
    C. {choice_C}
    D. {choice_D}
    [E. {choice_E}]  ← only if present in source data
    Đáp án:
    ```
    For English data, use `Question:` / `Answer:` equivalents but same structural template.
*   **Variable choice handling:** Each question stores its `num_choices` (3, 4, or 5). At training/eval time, logits are sliced to `[:num_choices]`. This avoids artificially inflating the answer space for 4-choice questions.
*   **Cross-reference choices:** Preserve choices like `"D. Cả A và C"` exactly as-is. Do NOT attempt to resolve them — the model needs to learn this pattern.
*   **LaTeX preservation:** Keep all LaTeX notation intact (e.g., `$x^2 + y^2$`). Do not strip, escape, or convert to Unicode.
*   **Validation:**
    *   Assert final answer token matches one of the available choices for that question
    *   Check tokenizer spacing variants: `' A'`, `'A'`, `'\nA'`
    *   Assert all choices are non-empty
    *   Assert no answer leakage in the question text
    *   Warn (don't crash) on duplicate choices — some VMLU questions legitimately have similar-looking options
*   **Deliverable:** A HuggingFace `Dataset` object saved to disk with columns: `[text, answer_token_id, subject, language, source, num_choices]`

**Task 1.3: Build Few-Shot Example Bank**
*   For each VMLU subject, select 5 high-quality examples from the **dev split** (304 questions = ~5 per subject across 58 subjects — nearly perfect fit)
*   **Selection criteria:** prefer examples with all 5 real choices, no cross-reference answers, moderate length
*   Store as a dict: `{subject: [list of 5 formatted examples with answers]}`
*   These will be prepended at eval time (Phase 5)
*   Use the **valid split** (745 questions) exclusively for checkpoint selection and hyperparameter tuning — never leak into training

---

## PHASE 2: Vietnamese SFT with QLoRA
**Objective:** Fine-tune `Qwen/Qwen3-1.7B` on the Vietnamese MCQ corpus using QLoRA. This is the single highest-impact step.

**Task 2.1: QLoRA Setup**
*   **Base Model:** `Qwen/Qwen3-1.7B` loaded in 4-bit (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`)
*   **LoRA Config:**
    ```python
    LoraConfig(
        r=64,                          # rank — go high, we have the budget on 1.7B
        lora_alpha=128,                # alpha = 2*r is standard
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # all linear layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ```
*   **Why QLoRA over full fine-tune:** 1.7B is small enough for full fine-tune on a good GPU, but QLoRA lets us iterate faster, checkpoint more, and experiment with different configs without OOM risk. We can always do a full fine-tune as a final step if QLoRA plateaus.

**Task 2.2: Masked SFT Training**
*   **Loss Formulation:** Standard cross-entropy, but masked to answer tokens only:
    $$ \mathcal{L}_{\text{SFT}} = - \log P_\theta(x_{t_{\text{ans}}} \mid \mathbf{x}_{< t_{\text{ans}}}) $$
*   **Implementation:** Custom collator that:
    1. Tokenizes the full prompt + answer
    2. Sets `labels = input_ids.clone()`
    3. Finds the index of the answer separator token (`Đáp án:` or `Answer:`)
    4. Sets all `labels[:answer_start_idx]` to `-100`
*   **Hyperparameters:**
    | Param | Value | Rationale |
    |---|---|---|
    | Learning Rate | $2 \times 10^{-4}$ | Higher than full FT because LoRA adapters are small |
    | Schedule | Cosine with 5% warmup | Standard |
    | Epochs | 3 | More than 2 because LoRA underfits on small epoch counts |
    | Batch Size | 128 (effective via gradient accumulation) | Large batch stabilizes MCQ training |
    | Max Seq Length | 512 | MCQ questions are short |
    | Weight Decay | 0.01 | Mild regularization |
*   **Curriculum:** Train Vietnamese data first (2 epochs), then mix in English data (1 epoch). Vietnamese is the target distribution; English provides format robustness.

**Task 2.3: Checkpoint Evaluation Loop**
*   After every 500 steps, run the Phase 5 eval script on the **VMLU valid split** (745 questions)
*   Track `accuracy_by_subject` and `overall_accuracy`
*   Save the best checkpoint by overall accuracy
*   **Stop early** if accuracy plateaus for 2000 steps
*   ⚠️ 745 questions is small — each question is worth ~0.13%. Don't over-interpret fluctuations < 1%.

---

## PHASE 3: Knowledge Distillation from Strong Teacher
**Objective:** Use a powerful teacher model to generate soft labels on **Vietnamese** MCQ data directly. No translation needed — the teacher reads Vietnamese.

> [!IMPORTANT]
> The key insight vs. the original plan: modern teachers (GPT-4o, Claude, Gemini) handle Vietnamese natively. Translating to English and back introduces noise for zero benefit.

**Task 3.1: Teacher Logit Generation**
*   **Teacher Options (pick best available):**
    *   `GPT-4o` via API (best Vietnamese performance, ~81% on VMLU)
    *   `Claude 3.5 Sonnet` via API
    *   `Qwen2.5-72B-Instruct` (open-source, can run locally with vLLM on multi-GPU)
*   **Process:**
    1. Take the full VMLU train split + any additional Vietnamese MCQ data
    2. Send each question to the teacher WITH 5-shot examples (same bank from Phase 1)
    3. Extract `log_prob` for each available choice (A–D, or A–E if 5 choices) from the teacher's output
    4. Save as: `{question_id, subject, num_choices: int, teacher_logprobs: [num_choices], teacher_answer: str}`
*   **Fallback for API models:** If the API doesn't expose logprobs for specific tokens, use the teacher's hard label + a heuristic soft distribution:
    ```python
    # If teacher says "B" with no logprobs available (e.g., 4-choice question):
    n = num_choices  # 4 or 5
    soft_label = [0.84 if i == answer_idx else (1 - 0.84) / (n - 1) for i in range(n)]
    ```
*   **Budget:** ~$20-50 for GPT-4o on 50k questions via batch API. Worth it.

**Task 3.2: Distillation Training**
*   **Starting Point:** Best checkpoint from Phase 2 (NOT from scratch)
*   **Loss:** Mixed loss combining SFT and KD:
    $$ \mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{\text{SFT}} + \alpha \cdot \tau^2 \cdot D_{\text{KL}}(p_{\text{teacher}} \| q_{\text{student}}) $$
    Where:
    *   $\alpha = 0.7$ (lean toward teacher signal)
    *   $\tau = 2.0$ (temperature scaling)
    *   $p_{\text{teacher}}$ = teacher's softmax distribution over available choices at temperature $\tau$
    *   $q_{\text{student}}$ = student's softmax distribution over available choices at temperature $\tau$
*   **Implementation:**
    ```python
    def kd_loss(student_logits, teacher_logprobs, hard_labels, alpha=0.7, tau=2.0):
        # student_logits: [batch, N] — raw logits for the N choices in this batch
        # teacher_logprobs: [batch, N] — teacher's log-probs
        # N is typically 4, sometimes 5. Batch questions by num_choices for efficiency.
        p_teacher = F.softmax(teacher_logprobs / tau, dim=-1)
        q_student = F.log_softmax(student_logits / tau, dim=-1)
        kd = F.kl_div(q_student, p_teacher, reduction='batchmean') * (tau ** 2)
        ce = F.cross_entropy(student_logits, hard_labels)
        return alpha * kd + (1 - alpha) * ce
    ```
*   **Batching note:** Group questions by `num_choices` (separate 4-choice and 5-choice batches) so tensor shapes are uniform within each batch.
*   **Hyperparameters:** LR = $5 \times 10^{-5}$ (lower than Phase 2 — we're refining, not learning from scratch), 1-2 epochs, same LoRA config.

---

## PHASE 4: Subject-Specific Boosting (Optional but High ROI)
**Objective:** Identify weak subjects and apply targeted interventions.

**Task 4.1: Per-Subject Accuracy Analysis**
*   Run full eval on all 58 VMLU subjects
*   Sort by accuracy ascending — the bottom 10 subjects are your targets
*   Categorize weaknesses:
    *   **Knowledge gap:** model doesn't know the facts → need more training data
    *   **Reasoning gap:** model can't chain logic → need CoT-augmented examples
    *   **Format confusion:** model outputs wrong format → need more format examples

**Task 4.2: Targeted Data Augmentation**
*   For knowledge-gap subjects:
    *   Use GPT-4o to generate 500-2000 additional MCQ questions per weak subject
    *   Include Vietnamese Wikipedia passages as context where relevant
*   For reasoning-gap subjects:
    *   Generate chain-of-thought explanations from the teacher, then train with:
        ```text
        Câu hỏi: {question}
        A. ... B. ... C. ... D. ... [E. ...]  ← if present
        Giải thích: {CoT from teacher}
        Đáp án: {answer}
        ```
    *   Still mask loss to answer token only, but the CoT in the prompt helps the model learn reasoning patterns

**Task 4.3: Targeted Fine-Tuning**
*   Fine-tune the best Phase 3 checkpoint on the augmented data
*   Use a very low LR ($1 \times 10^{-5}$) to avoid catastrophic forgetting
*   1 epoch only

---

## PHASE 5: Evaluation & The "Benchmaxxing" Logit Game
**Objective:** Maximize score on the official VMLU test set using every legal trick available.

**Task 5.1: Few-Shot Logit Extraction Eval Script**
*   Do *not* use `.generate()` and regex parsing. Small models are too brittle for that.
*   **Implementation:**
    1. For each test question, prepend 5 few-shot examples from the same subject (from Phase 1's example bank)
    2. Format the full prompt: `[5 examples] + [test question]`
    3. Pass into the model
    4. Extract logits at the final token position
    5. Read `num_choices` for this question. Slice logits to vocabulary IDs for the first `num_choices` letters (`A`–`D` for 4-choice, `A`–`E` for 5-choice). Check ALL tokenizer variants: with/without space prefix, newline prefix, etc.
    6. Apply argmax over only the available choices:
    $$ \hat{y} = \arg\max_{c \in \text{choices}} \sum_{v \in \text{variants}(c)} \mathbf{z}_{\text{final}}[v] $$
    Note: sum over token variants (e.g., `' A'` and `'A'`) before argmax to capture all probability mass.
*   **Compute accuracy per subject and overall.**

**Task 5.2: Prompt Engineering Sweep**
*   Try multiple prompt templates and pick the best:
    *   Vietnamese-only prompt (as shown above)
    *   Bilingual prompt (Vietnamese question + English instruction wrapper)
    *   With/without subject header
    *   With/without "Hãy chọn đáp án đúng nhất:" instruction
*   This is free accuracy — just a for-loop over templates.

**Task 5.3: Ensemble (If Compute Allows)**
*   If you have multiple good checkpoints from different phases/configs:
    *   Average their logits for available choices before argmax
    *   Even 2-model ensemble typically adds 1-2% accuracy
    *   $$ \hat{y} = \arg\max_{c} \frac{1}{K}\sum_{k=1}^{K} \mathbf{z}^{(k)}_{\text{final}}[c] $$

**Task 5.4: Calibration (Advanced)**
*   Some models have systematic biases (e.g., always favoring option A)
*   On the VMLU valid split (4-choice questions only — sufficient sample size), compute: `P(predict=X|true=Y)` for all X,Y ∈ {A,B,C,D}
*   Build a **4×4 confusion matrix** and apply a simple affine correction to logits at eval time
*   For the rare 5-choice questions, skip calibration (too few samples to estimate)
*   This often adds 0.5-1% accuracy for free
*   ⚠️ With only 745 valid questions, the confusion matrix will be noisy. Use Laplace smoothing.

**Task 5.5: Alternative Evaluation via lm-eval-harness + SGLang**
*   For highly standardized, multi-threaded evaluation (especially when testing our 1.7B model against the 27B teacher baseline), you can use `lm-eval-harness`.
*   **Step 1:** Create `vmlu.yaml` in your lm-eval tasks directory:
    ```yaml
    task: vmlu
    dataset_path: json
    dataset_name: null
    dataset_kwargs:
      data_files: data/sft/test.jsonl # Assuming we parsed the VMLU test split
    output_type: multiple_choice
    training_split: train
    test_split: train # If HF dataset parsing loaded data into the 'train' split
    fewshot_split: train
    doc_to_text: "Câu hỏi: {{question}}\nĐáp án:"
    doc_to_target: "{{['A', 'B', 'C', 'D'].index(answer)}}" 
    doc_to_choice:
      - "{{choices[0]}}"
      - "{{choices[1]}}"
      - "{{choices[2]}}"
      - "{{choices[3]}}"
    metric_list:
      - metric: acc
        aggregation: mean
        higher_is_better: true
    ```
    *(Note: VMLU has 3-5 choice questions, so `vmlu.yaml` may require a custom Python `utils.py` function mapped via `!function` for dynamic length arrays, but the YAML above serves as the generic template).*
*   **Step 2:** Launch the SGLang HTTP API locally containing our distilled merged model (`checkpoints/vmlu-qwen3-1.7b-maxxing`).
*   **Step 3:** Run `lm-eval`:
    ```bash
    lm_eval --model local-completions \
        --model_args '{"base_url": "http://localhost:8000/v1/completions", "model": "vmlu-qwen3-1.7b-maxxing", "num_concurrent": 256, "max_retries": 10, "max_gen_toks": 2048}' \
        --tasks vmlu \
        --batch_size auto \
        --num_fewshot 5 \
        --trust_remote_code
    ```

---

### Final Verification Checklist for the Agent:
- [x] Are logit slices dynamically sized per question's `num_choices` (not hardcoded to 4 or 5)?
- [x] Did you mask the cross-entropy loss `labels` to `-100` for all non-answer tokens?
- [x] Did you verify tokenizer behavior for answer tokens (`'A'`, `' A'`, `'\nA'`)? *Yes, unit mapped variants*
- [x] Are you using QLoRA with `r=64` targeting all linear layers?
- [x] Did you freeze the base model and only train LoRA adapters? *Yes, using PeftModel stacking*
- [x] Are you evaluating with 5-shot examples prepended (from dev split, not valid)?
- [x] Did you sum logit variants before argmax in the eval script?
- [x] Are you scaling the KL-Divergence loss by $\tau^2$?
- [x] Did you train on Vietnamese data first, English second? *SFT Pipeline combines them, so they mix*
- [x] Did you run per-subject accuracy analysis to find weak spots? *Yes, `evaluate.py` isolates bottom 10 subjects*
- [x] Did you preserve LaTeX in questions as-is (no stripping)?
- [x] Did you handle cross-reference choices (`"Cả A và C"`) without modification?
- [x] Are 4-choice and 5-choice questions batched separately for KD training? *Handled dynamically item-by-item in KDTrainer*

### Expected Accuracy Range (Realistic)
| Phase | Expected Accuracy | Notes |
|---|---|---|
| Base Qwen3-1.7B (0-shot) | ~30-35% | Random-ish on Vietnamese MCQ |
| After Phase 0 (Vietnamese CPT) | ~35-42% | Better Vietnamese comprehension, not yet MCQ-aware |
| After Phase 2 (Vietnamese SFT) | ~48-58% | Biggest single jump — format + knowledge |
| After Phase 3 (KD from teacher) | ~58-65% | Teacher signal refines decision boundary |
| After Phase 4 (Subject boosting) | ~60-67% | Targeted improvements on weak subjects |
| After Phase 5 (Few-shot + tricks) | ~63-70% | Prompt engineering + calibration + ensemble |

A 1.7B model scoring 63-70% on VMLU would be **extremely competitive** for its parameter class. For reference, QwQ-32B (18x larger) scores 76%.

Agent, begin execution with Phase 0. Set up the data pipeline for `VTSNLP/vietnamese_curated_dataset` — download, filter, sample, and tokenize. I will review the preprocessing before we start CPT training.