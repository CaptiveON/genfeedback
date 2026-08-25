"""Shared word-level tokenizer used by BOTH data generation and inference.

Using one tokenizer everywhere is what keeps training labels and inference
predictions aligned. It keeps real tech tokens intact as single words —
``C++``, ``C#``, ``Node.js``, ``scikit-learn``, ``3+`` — while splitting
surrounding punctuation (commas, periods, parentheses) into their own tokens.

The downstream BERT model is then always fed with ``is_split_into_words=True``,
so each word maps to exactly one label and reconstruction is just
``" ".join(words)`` — preserving original surface form and casing.
"""

import re

# A "word" is an alphanumeric run that may contain internal + # . - joined to
# more alphanumerics (Node.js, scikit-learn) and may end in + or # (C++, C#, 3+).
# Anything else (a comma, period, paren) is emitted as its own single token.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[+#.\-][A-Za-z0-9]+)*[+#]*|[^\sA-Za-z0-9]")


def tokenize_words(text):
    """Split text into word tokens, preserving tech symbols, isolating punctuation."""
    return _WORD_RE.findall(text)
