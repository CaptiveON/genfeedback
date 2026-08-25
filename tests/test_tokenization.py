"""The shared tokenizer is the keystone of label alignment — if these break,
training labels and inference predictions silently desync."""

from genfeedback.tokenization import tokenize_words


def test_tech_tokens_stay_whole():
    words = tokenize_words("C++ and C# with Node.js, scikit-learn (3+ years).")
    assert "C++" in words
    assert "C#" in words
    assert "Node.js" in words
    assert "scikit-learn" in words
    assert "3+" in words


def test_punctuation_is_isolated():
    words = tokenize_words("Python, Java.")
    assert words == ["Python", ",", "Java", "."]


def test_casing_is_preserved():
    assert tokenize_words("PyTorch") == ["PyTorch"]


def test_empty_and_whitespace():
    assert tokenize_words("") == []
    assert tokenize_words("   \n\t ") == []
