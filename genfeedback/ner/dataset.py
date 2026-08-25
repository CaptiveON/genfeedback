"""Torch Dataset that aligns BIO labels to BERT subword tokens.

The text is fed to the tokenizer as a pre-split list of words
(``is_split_into_words=True``) so that ``word_ids()`` line up *exactly* with
the per-word label list. This is essential: passing a raw string instead lets
BERT's punctuation splitting (e.g. "C++" -> "c","+","+") desync word_ids from
the labels, silently corrupting every sample that contains such a token.
"""

import torch
from torch.utils.data import Dataset


class ResumeDataset(Dataset):
    """Tokenizes pre-split words and aligns word-level labels to subwords.

    Only the first subword of each word carries the label; continuation
    subwords and special/padding tokens get -100 so they are ignored by the
    loss and metrics (standard HF token-classification alignment).
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        # texts hold space-joined CoNLL tokens; split() recovers them exactly,
        # aligned 1:1 with target_labels.
        words = self.texts[item].split()
        target_labels = self.labels[item]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        aligned = []
        previous_word = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != previous_word:
                aligned.append(
                    target_labels[word_id] if word_id < len(target_labels) else -100
                )
            else:
                aligned.append(-100)
            previous_word = word_id

        # Pad/truncate label sequence to match input length exactly
        aligned += [-100] * (len(input_ids) - len(aligned))
        aligned = aligned[: self.max_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(aligned, dtype=torch.long),
        }
