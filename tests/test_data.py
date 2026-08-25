"""The CoNLL loader must fail loudly on bad data — a silently corrupted corpus
trains a do-nothing model that still reports high token accuracy."""

import pytest

from genfeedback.data import load_conll_dataset
from genfeedback.labels import LABEL_TO_ID


def _write(tmp_path, name, text, binary=False):
    p = tmp_path / name
    if binary:
        p.write_bytes(text)
    else:
        p.write_text(text)
    return p


def test_basic_parse_and_sample_split(tmp_path):
    p = _write(
        tmp_path,
        "ok.txt",
        "Python B-TECHNICALSKILL\ndev O\n\n5 B-YEAROFEXPERIENCE\nyears I-YEAROFEXPERIENCE\n",
    )
    texts, labels = load_conll_dataset(p)
    assert texts == ["Python dev", "5 years"]
    assert labels == [
        [LABEL_TO_ID["B-TECHNICALSKILL"], LABEL_TO_ID["O"]],
        [LABEL_TO_ID["B-YEAROFEXPERIENCE"], LABEL_TO_ID["I-YEAROFEXPERIENCE"]],
    ]


def test_unknown_tag_raises_in_strict_mode(tmp_path):
    p = _write(tmp_path, "bad.txt", "Python B-BOGUSTAG\n")
    with pytest.raises(ValueError):
        load_conll_dataset(p)


def test_lenient_mode_drops_instead_of_raising(tmp_path):
    p = _write(tmp_path, "bad.txt", "Python B-BOGUSTAG\n")
    texts, labels = load_conll_dataset(p, strict=False)
    assert texts == ["Python"]
    assert labels == [[LABEL_TO_ID["O"]]]


def test_bom_does_not_corrupt_first_token(tmp_path):
    p = _write(
        tmp_path,
        "bom.txt",
        "﻿Python B-TECHNICALSKILL\nrocks O\n".encode("utf-8"),
        binary=True,
    )
    texts, _ = load_conll_dataset(p)
    assert texts[0].split()[0] == "Python"


def test_words_and_labels_stay_aligned(tmp_path):
    p = _write(tmp_path, "ok.txt", "a O\nb O\nc B-TOOLS\n")
    texts, labels = load_conll_dataset(p)
    assert len(texts[0].split()) == len(labels[0])
