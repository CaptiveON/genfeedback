"""Load / save the Gemma feedback model and tokenizer.

Two ways to get a generator:
  * `load_gemma()`       — a *fine-tuned* model from a local dir (GEMMA_DIR).
  * `load_base_gemma()`  — the *base* instruction-tuned model from the HF Hub
    (gated; needs an HF token). This is the quickest way to test feedback in the
    UI without fine-tuning. Handles both standard CausalLM Gemma (2-2b-it) and
    the multimodal E2B/E4B Gemma (e.g. gemma-4-E2B-it), which loads via the
    generic `AutoProcessor` + `AutoModelForImageTextToText` classes.
"""

import logging
import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import config

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hf_token():
    """Return the effective HF token, or None.

    Checks env vars first, then the token saved by `hf auth login`
    (~/.cache/huggingface/token) via `huggingface_hub.get_token()` — so a CLI
    login is detected just like an env var.
    """
    env = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if env:
        return env
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001 - any hub error -> treat as no token
        return None


def _is_multimodal(model_id):
    """E2B/E4B (and 3n) Gemma are image-text-to-text; plain *-2b-it are CausalLM."""
    low = model_id.lower()
    return any(tag in low for tag in ("e2b", "e4b", "3n"))


class GemmaBundle:
    """Holds a loaded generator + its tokenizer/processor and device."""

    def __init__(self, model, tokenizer, device, multimodal):
        self.model = model
        self.tokenizer = tokenizer  # AutoTokenizer or AutoProcessor
        self.device = device
        self.multimodal = multimodal


@lru_cache(maxsize=2)
def load_base_gemma(model_id=None):
    """Load a base instruction-tuned Gemma from the HF Hub for generation.

    Cached so repeated calls reuse the loaded weights. Gemma is gated: set an
    HF token (HF_TOKEN) and accept the model license on its HF page first.
    """
    model_id = model_id or config.GEMMA_BASE_MODEL
    token = hf_token()
    device = get_device()
    # bf16 on accelerators (Gemma's preferred dtype), fp32 on CPU.
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    logger.info("Loading base Gemma %r on %s (%s)...", model_id, device, dtype)

    if _is_multimodal(model_id):
        # E2B/E4B Gemma (e.g. gemma-4-E2B-it) are image-text-to-text models;
        # load with the generic Auto classes (version-agnostic).
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                f"{model_id} needs a newer transformers. Upgrade it:\n"
                "  .venv/bin/pip install -U transformers accelerate timm"
            ) from exc
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, token=token, dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(device).eval()
        return GemmaBundle(model, processor, device, multimodal=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return GemmaBundle(model, tokenizer, device, multimodal=False)


def save_gemma(model, tokenizer, path=None):
    path = str(path or config.GEMMA_DIR)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info("Saved Gemma model + tokenizer to %s", path)


def load_gemma(path=None):
    # Fixed: original saved to BERT path but loaded from a different path.
    path = str(path or config.GEMMA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    device = get_device()
    model.to(device)
    return model, tokenizer, device


def gemma_artifacts_exist(path=None):
    path = config.GEMMA_DIR if path is None else path
    return (path / "config.json").exists()
