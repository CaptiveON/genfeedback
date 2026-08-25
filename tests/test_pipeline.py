"""End-to-end checks against the trained NER model. Skipped when the
artifact is absent (fresh clone) — train with `python -m genfeedback.train_bert`."""

import pytest

from genfeedback.ner.model_io import bert_artifacts_exist

pytestmark = pytest.mark.skipif(
    not bert_artifacts_exist(),
    reason="NER artifact not trained yet (python -m genfeedback.train_bert)",
)


@pytest.fixture(scope="module")
def bert():
    from genfeedback.ner.model_io import load_bert

    return load_bert()


def _extract(bert, text):
    from genfeedback.ner.inference import extract_entities, post_process_entities

    model, tokenizer, device = bert
    return post_process_entities(extract_entities([text], model, tokenizer, device)[0])


def test_demo_jd_extraction(bert):
    ents = _extract(
        bert,
        "We are hiring a Senior Machine Learning Engineer with 5 years of "
        "experience, proficient in Python and PyTorch, with Docker and AWS.",
    )
    assert "Python" in ents.get("technicalskill", [])
    assert "5 years" in ents.get("yearofexperience", [])


def test_long_document_is_not_truncated(bert):
    # The skills sit at the very end of a >1200-word document; naive
    # 512-token truncation would lose them entirely.
    filler = "The team collaborates on planning and delivery across offices. " * 120
    ents = _extract(bert, filler + "Proficient in Python and PyTorch.")
    assert "Python" in ents.get("technicalskill", [])
    assert "PyTorch" in ents.get("technicalskill", [])


def test_perfect_match_end_to_end(bert):
    from genfeedback.matching import find_missing

    jd = _extract(
        bert,
        "Requires Python, PyTorch, TensorFlow, Docker, Kubernetes, AWS and "
        "5 years of experience.",
    )
    resume = _extract(
        bert,
        "Senior engineer with 6 years of experience. Skills: Python, PyTorch, "
        "TensorFlow, Docker, Kubernetes, AWS.",
    )
    assert find_missing(jd, resume) == {}
