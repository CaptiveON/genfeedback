"""Fine-tune BERT for token classification with LoRA, class-weighted loss,
early stopping, and seqeval entity-level metrics.

Upgrades from v1:
  * LoRA via peft (parameter-efficient; only query/value + classifier trained).
  * seqeval entity-level P/R/F1 replaces token-level sklearn metrics during
    Trainer evaluation; final reports are entity-level too.
  * set_seed() + TrainingArguments.seed for reproducibility.
  * EarlyStoppingCallback + load_best_model_at_end to restore best checkpoint.
  * LoRA adapters are merged into the base model before returning, so the
    save/load path and inference code are unchanged.
"""

import logging

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from seqeval.metrics import (
    classification_report as seqeval_cls_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from sklearn.utils import class_weight as sk_class_weight
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from transformers import (
    BertForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .. import config
from ..labels import ID_TO_LABEL, LABEL_TO_ID, NUM_LABELS
from .dataset import ResumeDataset

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Trainer that applies a fixed class-weight vector to the CE loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = None
        if self._class_weights is not None:
            weights = torch.tensor(
                self._class_weights, device=logits.device, dtype=torch.float
            )
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _compute_class_weights(train_labels):
    """Balanced weights over the full label set; absent classes get weight 1.0."""
    flat = [lab for sub in train_labels for lab in sub]
    present = np.unique(flat)
    weights_present = sk_class_weight.compute_class_weight(
        class_weight="balanced", classes=present, y=np.array(flat)
    )
    weight_map = dict(zip(present.tolist(), weights_present.tolist()))
    return np.array([weight_map.get(i, 1.0) for i in range(NUM_LABELS)], dtype=float)


def _to_label_seqs(label_ids, pred_ids):
    """Convert parallel int arrays to lists-of-string-lists for seqeval.

    Positions where label_id == -100 (subword / padding) are skipped.
    """
    true_seqs, pred_seqs = [], []
    for lab_seq, pred_seq in zip(label_ids, pred_ids):
        true_row, pred_row = [], []
        for l, p in zip(lab_seq, pred_seq):
            if l != -100:
                true_row.append(ID_TO_LABEL[l])
                pred_row.append(ID_TO_LABEL[p])
        true_seqs.append(true_row)
        pred_seqs.append(pred_row)
    return true_seqs, pred_seqs


def _compute_metrics(pred):
    """Entity-level P/R/F1 via seqeval (used by Trainer during eval)."""
    label_ids = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)
    true_seqs, pred_seqs = _to_label_seqs(label_ids, preds)
    return {
        "f1": seqeval_f1(true_seqs, pred_seqs),
        "precision": seqeval_precision(true_seqs, pred_seqs),
        "recall": seqeval_recall(true_seqs, pred_seqs),
    }


def _entity_report(label_ids_iter, pred_argmax_iter):
    """Return a seqeval entity-level classification report string."""
    true_seqs, pred_seqs = _to_label_seqs(label_ids_iter, pred_argmax_iter)
    return seqeval_cls_report(true_seqs, pred_seqs)


def fine_tune_bert(
    train_texts,
    train_labels,
    tokenizer,
    device,
    val_texts=None,
    val_labels=None,
    test_texts=None,
    test_labels=None,
    output_dir=None,
    num_train_epochs=None,
):
    """Fine-tune BERT NER with LoRA and return the merged (non-adapter) model."""
    set_seed(config.NER_SEED)
    device = torch.device(device)
    output_dir = str(output_dir or config.BERT_DIR)
    num_train_epochs = num_train_epochs or config.NER_EPOCHS

    token_lengths = [len(tokenizer(t).input_ids) for t in train_texts]
    max_len = min(
        config.MAX_LEN_CAP,
        int(np.percentile(token_lengths, config.MAX_LEN_PERCENTILE)),
    )
    logger.info("Computed max_len=%d", max_len)

    class_weights = _compute_class_weights(train_labels)
    logger.info("Class weights: %s", class_weights)

    train_dataset = ResumeDataset(train_texts, train_labels, tokenizer, max_len)

    has_val = val_texts is not None and val_labels is not None
    if has_val:
        val_dataset = ResumeDataset(val_texts, val_labels, tokenizer, max_len)
    else:
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        logger.info("Auto-split: %d train / %d val", train_size, val_size)

    test_dataset = None
    if test_texts and test_labels:
        test_dataset = ResumeDataset(test_texts, test_labels, tokenizer, max_len)

    base_model = BertForTokenClassification.from_pretrained(
        config.BERT_BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=config.NER_LORA_R,
        lora_alpha=config.NER_LORA_ALPHA,
        lora_dropout=config.NER_LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=config.NER_LEARNING_RATE,
        per_device_train_batch_size=config.NER_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.NER_EVAL_BATCH_SIZE,
        num_train_epochs=num_train_epochs,
        weight_decay=config.NER_WEIGHT_DECAY,
        logging_steps=50,
        seed=config.NER_SEED,
    )

    def collator(data):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data]),
        }

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        data_collator=collator,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.NER_PATIENCE)],
    )

    logger.info("Starting BERT+LoRA training...")
    trainer.train()
    logger.info("Training complete.")

    val_pred = trainer.predict(val_dataset)
    logger.info("Validation metrics: %s", val_pred.metrics)
    print("\nValidation Entity-Level Report:")
    print(_entity_report(val_pred.label_ids, np.argmax(val_pred.predictions, axis=2)))

    if test_dataset is not None:
        test_pred = trainer.predict(test_dataset)
        logger.info("Test metrics: %s", test_pred.metrics)
        print("\nTest Entity-Level Report:")
        print(_entity_report(test_pred.label_ids, np.argmax(test_pred.predictions, axis=2)))

    # Merge LoRA adapters back into the base weights; inference/save path unchanged.
    merged_model = trainer.model.merge_and_unload()
    logger.info("LoRA adapters merged into base model.")
    return merged_model
