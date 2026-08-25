"""Load / save the BERT NER model and tokenizer."""

import logging

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

from .. import config

logger = logging.getLogger(__name__)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_bert(model, tokenizer, path=None):
    path = str(path or config.BERT_DIR)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info("Saved BERT model + tokenizer to %s", path)


def load_bert(path=None):
    path = str(path or config.BERT_DIR)
    tokenizer = BertTokenizerFast.from_pretrained(path)
    model = BertForTokenClassification.from_pretrained(path)
    device = get_device()
    model.to(device)
    return model, tokenizer, device


def bert_artifacts_exist(path=None):
    path = config.BERT_DIR if path is None else path
    return (path / "config.json").exists()
