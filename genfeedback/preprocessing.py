"""Text preprocessing utilities.

NOTE: stopword removal is kept available but is *not* applied before NER
by default. Removing stopwords shifts token positions and can break the
word-to-label alignment that the NER model relies on. Use `clean_for_display`
for cosmetic cleaning and reserve `remove_stopwords` for non-NER contexts.
"""

import logging
import re
import ssl

import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

_STOPWORDS = None


def _ensure_nltk_stopwords():
    """Download NLTK stopwords once, tolerating restricted SSL environments."""
    global _STOPWORDS
    if _STOPWORDS is not None:
        return _STOPWORDS
    try:
        _ = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = ssl._create_unverified_context
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def normalize_whitespace(text):
    """Collapse whitespace and strip; keep all word content for NER."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(texts):
    """Remove English stopwords. For non-NER text (e.g. JD keyword matching)."""
    stop = _ensure_nltk_stopwords()
    cleaned = []
    for text in texts:
        text = re.sub(r"[^\w\s.]", "", text)
        words = [w for w in text.split() if w.lower() not in stop]
        cleaned.append(" ".join(words))
    return cleaned


def prepare_for_ner(texts):
    """Light cleaning safe for NER: normalize whitespace only."""
    return [normalize_whitespace(t) for t in texts]
