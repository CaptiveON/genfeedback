"""Central configuration: paths, model names, and hyperparameters.

This is the single source of truth. Override any value via environment
variables (e.g. GENFEEDBACK_BERT_DIR) so the same code runs in dev and prod.
"""

import os
from pathlib import Path

# --- Base directories -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.getenv("GENFEEDBACK_ARTIFACTS", PROJECT_ROOT / "artifacts"))

# --- Model artifact paths ---------------------------------------------------
# NOTE: Original code saved Gemma to the BERT path and loaded it from a
# different path. These are now distinct and consistent.
BERT_DIR = Path(os.getenv("GENFEEDBACK_BERT_DIR", ARTIFACTS_DIR / "bert_ner"))
GEMMA_DIR = Path(os.getenv("GENFEEDBACK_GEMMA_DIR", ARTIFACTS_DIR / "gemma_feedback"))

# --- Base (pretrained) model identifiers ------------------------------------
BERT_BASE_MODEL = os.getenv("GENFEEDBACK_BERT_BASE", "bert-base-uncased")
# Feedback LLM. Default is Gemma 4 E2B-it (gated, multimodal — needs HF auth).
# For a lighter, text-only option that runs with less memory, override with:
#   export GENFEEDBACK_GEMMA_BASE=google/gemma-2-2b-it
GEMMA_BASE_MODEL = os.getenv("GENFEEDBACK_GEMMA_BASE", "google/gemma-4-E2B-it")

# Optional URL of the decoupled Gemma feedback microservice (gemma_api.py).
# When set (e.g. http://gemma-api:8000 in Kubernetes), the app calls it over
# HTTP instead of loading the model in-process. Empty = run the model locally.
GEMMA_API_URL = os.getenv("GENFEEDBACK_GEMMA_API_URL", "")

# --- Data -------------------------------------------------------------------
DATA_PATH = Path(os.getenv("GENFEEDBACK_DATA", PROJECT_ROOT / "data" / "resume.txt"))

# --- NER training hyperparameters -------------------------------------------
MAX_LEN_CAP = 512
MAX_LEN_PERCENTILE = 95
# LoRA trains only ~0.5% of params, so it needs a much higher LR than the
# 2e-5 used for full fine-tuning (which left the adapters barely updated).
NER_LEARNING_RATE = 3e-4
NER_TRAIN_BATCH_SIZE = 8
NER_EVAL_BATCH_SIZE = 8
NER_EPOCHS = 20
NER_WEIGHT_DECAY = 0.01
NER_SEED = int(os.getenv("GENFEEDBACK_NER_SEED", 42))
NER_PATIENCE = int(os.getenv("GENFEEDBACK_NER_PATIENCE", 3))

# --- NER LoRA hyperparameters -----------------------------------------------
NER_LORA_R = int(os.getenv("GENFEEDBACK_NER_LORA_R", 16))
NER_LORA_ALPHA = int(os.getenv("GENFEEDBACK_NER_LORA_ALPHA", 32))
NER_LORA_DROPOUT = float(os.getenv("GENFEEDBACK_NER_LORA_DROPOUT", 0.1))

# --- Data split ratios ------------------------------------------------------
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- Feedback / Gemma hyperparameters ---------------------------------------
GEMMA_MAX_NEW_TOKENS = 512
GEMMA_LEARNING_RATE = 2e-5
GEMMA_TRAIN_BATCH_SIZE = 4
GEMMA_EPOCHS = 3
GEMMA_WEIGHT_DECAY = 0.01
GEMMA_WARMUP_STEPS = 500

# --- Company / recruiter defaults for feedback emails -----------------------
COMPANY_NAME = os.getenv("GENFEEDBACK_COMPANY", "Acme Corp")
RECRUITER_NAME = os.getenv("GENFEEDBACK_RECRUITER", "Sam")
ROLE_TITLE = os.getenv("GENFEEDBACK_ROLE", "Software Engineer")
