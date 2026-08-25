"""CLI entrypoint to fine-tune and save the BERT NER model.

Usage:
    python -m genfeedback.train_bert
"""

import logging

from transformers import BertTokenizerFast

from . import config
from .data import load_conll_dataset, split_dataset
from .ner.model_io import get_device, save_bert
from .ner.train import fine_tune_bert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    texts, labels = load_conll_dataset(config.DATA_PATH)
    tr_t, tr_l, va_t, va_l, te_t, te_l = split_dataset(
        texts, labels,
        config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO,
    )
    tokenizer = BertTokenizerFast.from_pretrained(config.BERT_BASE_MODEL)
    device = get_device()

    model = fine_tune_bert(
        train_texts=tr_t, train_labels=tr_l,
        val_texts=va_t, val_labels=va_l,
        test_texts=te_t, test_labels=te_l,
        tokenizer=tokenizer, device=device,
    )
    save_bert(model, tokenizer)
    logger.info("BERT fine-tuned and saved.")


if __name__ == "__main__":
    main()
