# Model Constants
BASE_MODEL = "Qwen/Qwen3-1.7B"

# Phase 0 Preprocessing Constants
CPT_DATASET_NAME = "VTSNLP/vietnamese_curated_dataset"
CPT_TARGET_SAMPLES = 500000
CPT_MAX_SEQ_LEN = 2048
CPT_PACKED_DATA_DIR = "data/cpt_packed"

# Educational/Factual topics categorized by `underthesea`
EDUCATIONAL_TOPICS = {
    "Khoa hoc",
    "Cong nghe",
    "Giao duc",
    "The gioi",
    "Kinh doanh",
    "Phap luat",
    "Suc khoe",
    "Lich su",
}

# Phase 0 Training Constants
CPT_OUTPUT_DIR = "checkpoints/cpt_qwen3_1.7b"
CPT_LOG_DIR = "logs/cpt_qwen3_1.7b"
CPT_LR = 5e-5
CPT_EPOCHS = 1
CPT_PER_DEVICE_BATCH_SIZE = 4
CPT_GRAD_ACCUM_STEPS = 16

# Phase 0 Sanity Check Constants
MMLU_TASKS = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]
NUM_MMLU_SHOTS = 5
NUM_VN_PARAGRAPHS = 10

# Phase 1 Data Curation Constants
VMLU_RAW_DIR = "vmlu_mqa_v1.5"
SFT_PACKED_DATA_DIR = "data/sft"
FEW_SHOT_BANK_PATH = "data/few_shot_bank.json"

# Phase 2 SFT Training Constants
SFT_OUTPUT_DIR = "checkpoints/sft_qwen3_1.7b"
SFT_LOG_DIR = "logs/sft_qwen3_1.7b"
SFT_LR = 2e-4
SFT_EPOCHS = 3
SFT_PER_DEVICE_BATCH_SIZE = 8  # Will be adjusted to hit effective 128 (e.g. 16 grad accum steps * 8 = 128 on 1 GPU)
SFT_GRAD_ACCUM_STEPS = 16
SFT_MAX_SEQ_LEN = 512
SFT_WEIGHT_DECAY = 0.01

# Phase 3 Knowledge Distillation Constants
TEACHER_MODEL_ID = "Qwen/Qwen3.5-27B"
DISTILLED_SFT_DIR = "data/distilled_sft"
SGLANG_HOST = "http://localhost:8000/v1"

# Phase 4 Distillation Training Constants
KD_OUTPUT_DIR = "checkpoints/kd_qwen3_1.7b"
KD_LOG_DIR = "logs/kd_qwen3_1.7b"
KD_LR = 5e-5
KD_EPOCHS = 2
KD_ALPHA = 0.7
KD_TEMPERATURE = 2.0

# Phase 5 Evaluation & Export Constants
EVAL_RESULTS_DIR = "results"
FINAL_MODEL_DIR = "checkpoints/vmlu-qwen3-1.7b-maxxing"
