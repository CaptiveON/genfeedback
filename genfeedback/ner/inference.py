"""Inference: run the NER model and group words into entities.

Reconstruction is word-level, not subword-level: the input is split with the
shared :func:`tokenize_words`, fed to BERT as pre-split words
(``is_split_into_words=True``), and each word takes the prediction of its first
subword. Entity text is the original words joined back together, so surface
forms like ``C++``, ``C#``, ``Node.js`` and full qualification titles survive
intact and casing is preserved. No noise-word heuristics are needed. The one
form that tokenizer does not keep whole is a name-initial dot — ``.NET`` splits
into ``.`` and ``NET`` — so :func:`_prefix_dot_glue` reads the source text to
put it back together at grouping time.

Long documents are WINDOWED, never truncated. BERT reads at most 512 tokens,
but a real resume is a few thousand words. Feeding one truncated sequence threw
away everything past the cut — usually the skills section, which sits at the
bottom of most resumes — and every skill in it was then reported as a gap the
candidate had to close. That was silent: no exception, no warning, just wrong
feedback. So the document is packed into overlapping windows that each fit the
model and the predictions are merged before entities are grouped:

* Windows are packed by EXACT subword count, not by a guessed word count. A
  word costs 1.25 subwords on average but up to 80 in the worst case, so any
  fixed word width is only a probabilistic bound — a 384-word window overflows
  512 tokens on ~6% of real windows. One cheap tokenizer pass per document buys
  the exact per-word lengths instead (~36 ms per 6000 words, against ~1.8 s of
  forward passes), which turns "probably fits" into "cannot overflow".
* Merging happens at the WORD-LABEL level, not the entity-string level: every
  window writes into one document-length label array, and :func:`_group_words`
  runs once over the whole document. An entity straddling a window edge is
  therefore contiguous in that array and re-forms as a single entity, with no
  string stitching. De-duplication is likewise structural — each word index has
  exactly one owning window, so a span two windows both saw is emitted once,
  while genuine repeats far apart in the document stay distinct.
* ``max_length`` is the hard cap per WINDOW. It is no longer a document
  truncation length, and nothing in this module truncates.
"""

import logging

import torch

from ..labels import ID_TO_LABEL
from ..tokenization import tokenize_words

logger = logging.getLogger(__name__)

# Words of overlap between consecutive windows. 64 covers 99.98% of gold entity
# spans in data/ (p99 is 13 words, p99.9 is 38), so every word is labelled by a
# window that had at least half the overlap as context on the side facing the
# seam. Costs ~1.16x redundant compute on real resumes.
OVERLAP_WORDS = 64

# How far from the middle of an overlap we may move the seam to land on a word
# both windows call "O". 16 words escapes 99.3% of spans while still leaving
# real context on the weak side of the cut.
SEAM_SNAP_RADIUS = 16


def extract_entities(texts, model, tokenizer, device, max_length=512):
    """Run NER over a list of texts; return a list of {label: [spans]} dicts.

    ``max_length`` bounds each window, not the document: longer text is split
    into overlapping windows and merged, so no input is ever truncated.
    """
    model.to(device)
    model.eval()
    entities_list = []

    # [CLS]/[SEP] are added once per window, so they come off the budget once
    # per window — never inside the packing loop. Derived, not hardcoded to 2,
    # so the number cannot drift away from the tokenizer.
    budget = max_length - tokenizer.num_special_tokens_to_add(False)

    for text in texts:
        words = tokenize_words(text)
        if not words:
            entities_list.append({})
            continue

        glue = _prefix_dot_glue(text, words)
        lengths = _word_subword_lengths(words, tokenizer)
        windows = _plan_windows(lengths, budget)
        logger.debug("%d words -> %d windows", len(words), len(windows))
        if len(windows) > 1:
            # Logged at INFO so a truncation regression is visible in the logs
            # instead of showing up as phantom skill gaps.
            logger.info(
                "%d words exceed one %d-token window; running %d overlapping windows",
                len(words),
                max_length,
                len(windows),
            )

        window_labels = [
            _predict_window(words, start, end, lengths, model, tokenizer, device)
            for start, end in windows
        ]
        labels = _merge_window_labels(windows, window_labels, len(words))
        entities_list.append(_group_words(words, labels, glue))

    return entities_list


def _word_subword_lengths(words, tokenizer):
    """Return the exact subword count of each word, as one list per document.

    WordPiece pre-tokenizes on word boundaries, so a word's subword count does
    not depend on its neighbours; that is what makes this table sliceable per
    window and makes window planning exact rather than statistical.
    """
    # verbose=False: this pass is a deliberate length probe over the whole
    # document, so the tokenizer's "longer than model_max_length" warning is
    # expected and would only be noise. Nothing is fed to the model here.
    encoding = tokenizer(
        words, is_split_into_words=True, add_special_tokens=False, verbose=False
    )
    lengths = [0] * len(words)
    for wid in encoding.word_ids(batch_index=0):
        if wid is not None:
            lengths[wid] += 1
    # Words can legitimately be 0 subwords long: BERT's text cleaner deletes
    # zero-width spaces, BOMs and combining marks, so those words never appear
    # in word_ids and never reach the model.
    return lengths


def _pack_from(lengths, start, budget):
    """Take words greedily from `start` while they fit; return the exclusive end.

    Always consumes at least one word, so callers cannot loop forever. A real
    WordPiece tokenizer caps a word at 80 subwords (anything over 100
    characters collapses to a single [UNK]), so that fallback is unreachable
    with a 510-subword budget; :func:`_plan_windows` logs it if it ever fires.
    """
    n = len(lengths)
    total = 0
    end = start
    while end < n and total + lengths[end] <= budget:
        total += lengths[end]
        end += 1
    return end + 1 if end == start and start < n else end


def _plan_windows(lengths, budget):
    """Pack word indices into overlapping [start, end) windows that fit budget."""
    if not lengths:
        return []

    n = len(lengths)
    windows = []
    start = 0

    while True:
        end = _pack_from(lengths, start, budget)
        if end - start == 1 and lengths[start] > budget:
            logger.warning(
                "word %d alone exceeds the %d-subword window budget", start, budget
            )
        windows.append((start, end))
        if end >= n:
            # Breaking on coverage, not on `start < n`, is what stops a final
            # window that would only repeat the tail of this one.
            break

        # Capping the overlap at half the window keeps the stride positive: a
        # URL-dense document packs only ~28 words per window, and a flat
        # 64-word overlap would step backwards and never advance.
        overlap = min(OVERLAP_WORDS, (end - start) // 2)
        # Where cheap text runs straight into dense text (a paragraph followed
        # by a list of long URLs), the next window can be so much narrower than
        # this one that it ends no further along — pure repeated work. Shrink
        # the overlap until it buys new coverage; at 0 the next window starts
        # exactly here and must advance, since one word always fits.
        while overlap > 0 and _pack_from(lengths, end - overlap, budget) <= end:
            overlap //= 2
        start = end - overlap

    return windows


def _predict_window(words, start, end, lengths, model, tokenizer, device):
    """Predict one window; return {document word index: label}."""
    if sum(lengths[start:end]) == 0:
        # Only zero-subword words here, so the encoding would be [CLS] [SEP]
        # with no word to label at all — skip the forward pass entirely.
        return {}

    encoding = tokenizer(
        words[start:end],
        is_split_into_words=True,
        return_tensors="pt",
        truncation=False,
    )
    word_ids = encoding.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = logits.argmax(dim=-1)[0].tolist()

    # Word-level label = prediction on that word's FIRST subword. word_ids are
    # window-local, so `start` shifts them back to document word indices —
    # forgetting that offset mislabels the whole document with no error.
    labels_here = {}
    previous_word = None
    for idx, wid in enumerate(word_ids):
        if wid is not None and wid != previous_word:
            labels_here[start + wid] = ID_TO_LABEL[pred_ids[idx]]
        previous_word = wid
    return labels_here


def _snap_seam(left_labels, right_labels, lo, hi):
    """Choose the word index where two overlapping windows hand over.

    The midpoint of the overlap [lo, hi) is the max-context choice: it gives
    every word to the window in which it sits furthest from an edge. From there
    we search outward for a word both windows call "O", because cutting where
    neither window sees an entity means no entity can be split by the cut.
    Candidates are tried nearest-to-midpoint first, and `mid - d` before
    `mid + d`, so two runs on the same input always produce the same gaps.
    """
    mid = (lo + hi) // 2
    radius = min(SEAM_SNAP_RADIUS, (hi - lo) // 4)
    for distance in range(radius + 1):
        for candidate in (mid - distance, mid + distance):
            if lo <= candidate < hi:
                if (
                    left_labels.get(candidate, "O") == "O"
                    and right_labels.get(candidate, "O") == "O"
                ):
                    return candidate
    # The midpoint fallback is the invariant; snapping is only an optimization.
    return mid


def _merge_window_labels(windows, window_labels, word_count):
    """Fold per-window predictions into one label per document word."""
    cuts = []
    for k in range(len(windows) - 1):
        lo = windows[k + 1][0]  # overlap region [lo, hi) shared by k and k+1
        hi = windows[k][1]
        cut = _snap_seam(window_labels[k], window_labels[k + 1], lo, hi)
        if cuts and cut < cuts[-1]:
            # Seams must not go backwards, or a word would be claimed by two
            # windows. They can invert only when a very dense window follows a
            # much wider one, which snapping can then pull further left.
            cut = cuts[-1]
        cuts.append(cut)

    labels = ["O"] * word_count
    for k, (start, end) in enumerate(windows):
        lo = start if k == 0 else cuts[k - 1]
        hi = end if k == len(windows) - 1 else cuts[k]
        for i in range(lo, hi):
            # .get covers zero-subword words, which the model never saw.
            labels[i] = window_labels[k].get(i, "O")
    return labels


def _prefix_dot_glue(text, words):
    """Flag each lone "." that is written flush against the word after it.

    :func:`tokenize_words` splits the leading dot of ``.NET`` into its own word,
    so the model can only ever label it separately and the span reconstructs as
    ``". NET"`` — a name no operator recognises and no candidate wrote. The
    model cannot resolve this: it is fed pre-split words, so ``"in .NET and"``
    and ``"in ML. NLP and"`` are the same token sequence to it, and it does tag
    both dots (verified: ``"Trained in ML. NLP ..."`` groups as ``"ML . NLP"``).
    Only the source text separates the two — a name-initial dot has no space
    after it, a sentence period always does — which is why this reads the text
    rather than the span string.

    Each word is a literal substring of `text`, the words are in order and do
    not overlap, and every non-whitespace character belongs to some word, so
    scanning forward from the end of the previous word finds each word at its
    true offset and equality of the two offsets means "no space between".
    """
    glue = [False] * len(words)
    cursor = 0
    for index, word in enumerate(words):
        start = text.find(word, cursor)
        if start < 0:
            # Unreachable while `words` comes from this `text`. Claiming no
            # glue at all degrades to the old ". NET" spans, whereas guessing
            # would fuse words that are not adjacent in the source.
            logger.warning(
                "word %r not found in its own text; skipping dot glue", word
            )
            return [False] * len(words)
        if index and start == cursor and words[index - 1] == ".":
            glue[index - 1] = True
        cursor = start + len(word)
    return glue


def _group_words(words, labels, glue):
    """Group BIO-tagged words into {category: [entity text]} using original words."""
    entities = {}
    current_words = []
    current_label = None

    def flush():
        nonlocal current_words, current_label
        if current_words and current_label:
            entities.setdefault(current_label.lower(), []).append(
                " ".join(current_words)
            )
        current_words, current_label = [], None

    for index, (word, label) in enumerate(zip(words, labels)):
        if label == "O":
            flush()
        elif label.startswith("B-"):
            flush()
            current_label = label[2:]
            current_words = [word]
        elif label.startswith("I-"):
            label_type = label[2:]
            if current_label == label_type:
                if glue[index - 1]:
                    # ".", "NET" were ".NET" in the source: re-fuse them into
                    # one word instead of joining them with a space. Only the
                    # continuation branch can do this — across a B- the model
                    # is claiming two separate entities.
                    current_words[-1] += word
                else:
                    current_words.append(word)
            else:
                # Treat a stray I- as the start of a new entity (robust to noise).
                flush()
                current_label = label_type
                current_words = [word]

    flush()
    return entities


def _trim_edge_punctuation(span):
    """Drop punctuation-only words from both ends of a span.

    :func:`tokenize_words` emits a separator as its own word, so when the model
    sweeps one into an entity it lands in the span as a standalone token:
    ``"C++ /"``. Only whole words with no alphanumeric character are dropped,
    which is why ``"C++ 11"`` keeps its ``11`` and single words such as
    ``"Node.js"``, ``"scikit-learn"``, ``"C#"`` or ``"3+"`` are never touched —
    they are one word each and always contain a letter or digit. ``".NET"`` is
    in that protected group too: :func:`_prefix_dot_glue` has already fused it
    into one word, so what arrives here is never a bare leading ``"."`` unless
    the source really did put a space after the dot.
    """
    parts = span.split()
    start, end = 0, len(parts)
    while start < end and not any(c.isalnum() for c in parts[start]):
        start += 1
    while end > start and not any(c.isalnum() for c in parts[end - 1]):
        end -= 1
    return " ".join(parts[start:end])


def post_process_entities(entities):
    """Minimal cleanup: trim edge punctuation, drop empties, de-duplicate.

    Reconstruction already yields clean surface forms, so this does no
    word-list filtering — it only strips punctuation-only words off the ends of
    a span, removes empties, and drops case-insensitive dupes.
    """
    cleaned = {}
    for key, values in entities.items():
        out, seen = [], set()
        for value in values:
            value = _trim_edge_punctuation(value)
            if not value:
                continue
            low = value.lower()
            if low not in seen:
                seen.add(low)
                out.append(value)
        if out:
            cleaned[key] = out
    return cleaned
