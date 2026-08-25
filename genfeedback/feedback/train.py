"""Fine-tune Gemma for email-style generation on the AESLC dataset.

Fixes from the original monolith:
  * Uses `datasets.load_dataset` directly (no longer shadowed by a local
    function of the same name).
  * Saves to GEMMA_DIR, the same path it is later loaded from.
  * Uses `eval_strategy` (consistent with the NER trainer).

NOTE: google/gemma-2-2b-it is a gated model requiring HuggingFace auth.
Per project scope, training is wired up but not run here. For real
fine-tuning of a quantized base model you should add LoRA/PEFT adapters;
full fine-tuning of 4-bit weights will not train correctly.
"""

import logging

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .. import config
from .model_io import save_gemma

logger = logging.getLogger(__name__)


def fine_tune_gemma(output_dir=None):
    """Fine-tune Gemma on AESLC and save to GEMMA_DIR."""
    output_dir = str(output_dir or config.GEMMA_DIR)

    dataset = load_dataset("Yale-LILY/aeslc")
    tokenizer = AutoTokenizer.from_pretrained(config.GEMMA_BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.GEMMA_BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    def tokenize(examples):
        return tokenizer(
            examples["subject_line"],
            text_target=examples["email_body"],
            max_length=512,
            truncation=True,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["subject_line", "email_body"],
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=config.GEMMA_LEARNING_RATE,
        per_device_train_batch_size=config.GEMMA_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.GEMMA_TRAIN_BATCH_SIZE,
        num_train_epochs=config.GEMMA_EPOCHS,
        weight_decay=config.GEMMA_WEIGHT_DECAY,
        warmup_steps=config.GEMMA_WARMUP_STEPS,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    logger.info("Starting Gemma training...")
    trainer.train()
    save_gemma(model, tokenizer, output_dir)
    logger.info("Gemma fine-tuned and saved.")
    return model, tokenizer
