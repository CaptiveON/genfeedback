"""Dataset loading (CoNLL-style) and splitting.

The original code named a function `load_dataset`, which shadowed
`datasets.load_dataset`. Here the reader is named `load_conll_dataset`
to remove that collision entirely.

Tags are now validated instead of being coerced. The previous reader mapped
every unrecognised tag to O (`LABEL_TO_ID.get(tag, 0)`), so a file written in
a different taxonomy — data/azzFinal.txt, whose tags are lowercase and named
differently — loaded as 2150 samples of pure O with no exception and no
warning. That trains a model which never predicts an entity while still
reporting ~99% token accuracy, because O is ~99% of the tokens. A typo
(B-TOOL for B-TOOLS), a lowercase tag, or taxonomy drift all failed the same
silent way. Refusing to load is strictly better than training on discarded
supervision, so `strict=True` is the default; `strict=False` keeps the old
lenient behaviour for deliberate exploration but reports the damage at ERROR
level.

Encoding is read as `utf-8-sig`, and every field is then screened for invisible
characters. `str.strip()` removes whitespace only, so a Unicode format (Cf) or
control (Cc) character glues itself to the neighbouring token and survives every
downstream check: `'<U+200B>Python'` carries a valid tag, renders as `Python` in
every editor, and trains the model on a token no resume will ever contain. The
byte-order mark that Windows tooling prepends is only the most common member of
that family — PDF and Word copy-paste routinely contribute U+00AD soft hyphen,
U+200B zero-width space, U+2060 word joiner and U+200D zero-width joiner — so
the screen is by Unicode category rather than by a list of codepoints, which is
the only version of it that closes for characters nobody has hit yet.

Where the character sits decides what happens to it. At the edge of a field it
cannot mean anything — a joiner with nothing to join, a soft hyphen with no word
to break, a BOM that is encoding metadata rather than content — so it is removed
and the removal is reported. Between two visible characters it may be genuine
orthography (U+200C is load-bearing in Persian and Devanagari) or paste damage,
and nothing available here can tell those apart; guessing is the exact failure
this module exists to prevent, so strict mode refuses the file and lenient mode
keeps the field byte-for-byte and reports it. Neither mode ever changes a token
without saying so.

Expected file format (one token + tag per line, blank line between samples):

    John B-DESIGNATION
    Doe I-DESIGNATION

    Python B-TECHNICALSKILL
    developer O
"""

import logging
import unicodedata

from .labels import LABEL_TO_ID

logger = logging.getLogger(__name__)

# A file in the wrong taxonomy complains once per token (30k+ times for
# azzFinal.txt), which buries the summary that actually matters. Cap the
# per-line noise; the end-of-load summary carries the real counts.
MAX_LINE_WARNINGS = 20

# Characters that render as nothing yet are not whitespace, so neither
# str.strip() nor str.split() gets rid of them: Cf (format — U+FEFF, U+200B,
# U+00AD, U+2060, U+200D, U+200C, ...) and Cc (control — a stray NUL or BEL from
# a truncated binary paste). Invisible characters that ARE whitespace to Python
# (U+00A0, U+2028) need no entry: split() already separates on them, so they
# surface as a loud field-count error, not a silent graft onto a token.
# Deliberately no wider. Codepoints that merely render blank — U+3164 HANGUL
# FILLER (Lo), U+2800 BRAILLE PATTERN BLANK (So), unassigned ones (Cn) — are
# content, or may become content in a later Unicode revision, so removing them
# would be the guess this module refuses to make; they reach the caller exactly
# as the file wrote them.
INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc"})


def _scan_invisible(field):
    """Trim invisible characters off both ends of `field`.

    Returns `(trimmed, interior)`, where `interior` is the first invisible
    character still standing between two visible ones, or None. Edge and
    interior are reported apart because only the edge is safe to repair on our
    own authority; this function therefore never drops an interior character,
    it hands it to the caller to refuse or record.
    """
    start, end = 0, len(field)
    while start < end and unicodedata.category(field[start]) in INVISIBLE_CATEGORIES:
        start += 1
    while end > start and unicodedata.category(field[end - 1]) in INVISIBLE_CATEGORIES:
        end -= 1
    trimmed = field[start:end]
    for char in trimmed:
        if unicodedata.category(char) in INVISIBLE_CATEGORIES:
            return trimmed, char
    return trimmed, None


def load_conll_dataset(data_path, strict=True):
    """Read a CoNLL-style file into (texts, label_id_lists).

    Args:
        data_path: path to the CoNLL-style file
        strict:    when True (the default), an unrecognised tag, a malformed
                   line, or an invisible character inside a field aborts the
                   load with an error naming file, line and offender. When
                   False, unknown tags fall back to O, malformed lines are
                   skipped, and fields with an interior invisible character are
                   kept unchanged — all three reported at ERROR level.
                   Invisible characters at the EDGE of a field are removed and
                   reported in both modes; see the module docstring.

    Returns:
        texts:  list[str]      - whitespace-joined tokens per sample
        labels: list[list[int]] - label IDs aligned to whitespace tokens

    Raises:
        ValueError: on a bad tag, a malformed line, or an invisible character
            inside a field when strict, and always if a sample's token count
            and label count disagree.
        UnicodeDecodeError: if the file is not UTF-8. Never suppressed in
            either mode — a fallback codec would mangle characters instead of
            failing, which is the exact outcome this reader exists to prevent.
    """
    texts, labels = [], []
    current_text, current_labels = [], []
    unknown_tags = {}
    malformed_lines = 0
    invisible_edge_lines = 0
    invisible_interior_fields = 0
    warnings_emitted = 0
    line_no = 0

    def flush(line_no):
        """Append the accumulated sample and reset the buffers."""
        text = " ".join(current_text)
        # The trainer aligns labels to text.split(); if those ever disagree,
        # every label after the divergence is attached to the wrong token, so
        # refuse rather than train on a silently shifted sample.
        if len(text.split()) != len(current_labels):
            raise ValueError(
                f"{data_path}: sample ending at line {line_no} has "
                f"{len(text.split())} whitespace tokens but "
                f"{len(current_labels)} labels — token/label alignment is "
                f"broken for this sample."
            )
        texts.append(text)
        labels.append([LABEL_TO_ID.get(tag, 0) for tag in current_labels])
        current_text.clear()
        current_labels.clear()

    try:
        # utf-8-sig drops a leading BOM; plain utf-8 would hand it to us as
        # U+FEFF, which strip() does not remove. newline is left at the default
        # so universal-newline translation handles CRLF and lone-CR files.
        with open(data_path, "r", encoding="utf-8-sig") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                # Screen every field, not just the first token of the file:
                # utf-8-sig drops a BOM at offset 0 only, and a file
                # concatenated from several exports carries one mid-file, where
                # it lands on whichever token or tag follows it. A field made of
                # nothing but invisible characters trims away to empty and is
                # dropped, which is what it renders as: a line of them is the
                # blank separator it looks like, and one standing where a token
                # belongs leaves a one-field line for the check below to refuse.
                parts, hidden, altered = [], None, False
                for field in line.split():
                    clean, interior = _scan_invisible(field)
                    altered = altered or clean != field
                    if interior and hidden is None:
                        hidden = (clean, interior)
                    if clean:
                        parts.append(clean)

                if altered:
                    invisible_edge_lines += 1
                    if warnings_emitted < MAX_LINE_WARNINGS:
                        warnings_emitted += 1
                        logger.warning(
                            "%s:%d: removed invisible character(s) from line "
                            "%r.", data_path, line_no, line,
                        )

                if not parts:
                    if current_text:
                        flush(line_no)
                    continue

                if len(parts) != 2:
                    reason = (
                        f"expected 'token TAG' but found {len(parts)} field(s)"
                    )
                    if strict:
                        raise ValueError(
                            f"{data_path}:{line_no}: malformed line {line!r} — "
                            f"{reason}. Pass strict=False to skip such lines."
                        )
                    malformed_lines += 1
                    if warnings_emitted < MAX_LINE_WARNINGS:
                        warnings_emitted += 1
                        logger.warning(
                            "%s:%d: skipping malformed line %r (%s)",
                            data_path, line_no, line, reason,
                        )
                    continue

                token, tag = parts
                if hidden:
                    bad_field, char = hidden
                    detail = (
                        f"U+{ord(char):04X} "
                        f"({unicodedata.name(char, 'unnamed control character')})"
                    )
                    if strict:
                        raise ValueError(
                            f"{data_path}:{line_no}: {bad_field!r} contains the "
                            f"invisible character {detail} between visible "
                            f"characters, so this field is not the word it "
                            f"appears to be. Whether that character is real "
                            f"orthography (U+200C carries meaning in Persian "
                            f"and Devanagari) or paste damage cannot be decided "
                            f"from the file, so the load refuses instead of "
                            f"guessing. Remove it, or pass strict=False to keep "
                            f"the field exactly as written and have it reported."
                        )
                    invisible_interior_fields += 1
                    if warnings_emitted < MAX_LINE_WARNINGS:
                        warnings_emitted += 1
                        logger.warning(
                            "%s:%d: %r contains the invisible character %s "
                            "between visible characters — keeping the field "
                            "exactly as written.",
                            data_path, line_no, bad_field, detail,
                        )

                if tag not in LABEL_TO_ID:
                    if strict:
                        raise ValueError(
                            f"{data_path}:{line_no}: unrecognised tag {tag!r} "
                            f"on line {line!r}. Expected one of: "
                            f"{', '.join(sorted(LABEL_TO_ID))}. If the token "
                            f"itself contains a space, this line is malformed "
                            f"and its second word was read as the tag. Pass "
                            f"strict=False to map unknown tags to O instead "
                            f"(that silently discards the supervision)."
                        )
                    unknown_tags[tag] = unknown_tags.get(tag, 0) + 1
                    if warnings_emitted < MAX_LINE_WARNINGS:
                        warnings_emitted += 1
                        logger.warning(
                            "%s:%d: unrecognised tag %r on line %r — labelling "
                            "the token O.", data_path, line_no, tag, line,
                        )

                current_text.append(token)
                current_labels.append(tag)

            # Flush trailing sample with no terminating blank line
            if current_text:
                flush(line_no)

    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise
    except PermissionError:
        logger.error("Permission denied reading: %s", data_path)
        raise
    except UnicodeDecodeError:
        # Decoding with the wrong codec would substitute or mangle characters
        # rather than fail, so the codec is never relaxed here; re-encode the
        # file instead. Name the file, since the default message does not.
        logger.error(
            "%s: not valid UTF-8 — the file is in some other encoding "
            "(a Windows-1252 or UTF-16 export is the usual cause). Re-encode "
            "it as UTF-8; loading it with a fallback codec would corrupt "
            "tokens instead of failing.", data_path,
        )
        raise

    if invisible_edge_lines:
        # Not folded into the strict=False summaries below: this happens in
        # both modes, and it is a repaired file, not a damaged load.
        logger.warning(
            "%s: removed invisible format/control characters from the edge of "
            "a field on %d line(s) — usually a byte-order mark from a "
            "concatenated export, or a soft hyphen or zero-width space from a "
            "PDF or Word paste. Left in place each one would have become part "
            "of the neighbouring token or tag while staying invisible in every "
            "editor.", data_path, invisible_edge_lines,
        )

    if unknown_tags or malformed_lines:
        worst = sorted(unknown_tags.items(), key=lambda kv: -kv[1])
        logger.error(
            "%s: loaded with strict=False — %d token(s) carrying %d "
            "unrecognised tag(s) were forced to O and %d malformed line(s) "
            "were dropped. Most frequent unrecognised tags: %s",
            data_path, sum(unknown_tags.values()), len(unknown_tags),
            malformed_lines,
            # repr, not the bare tag: an invisible character inside a tag would
            # otherwise print identically to the valid tag it is not.
            ", ".join(f"{tag!r} x{count}" for tag, count in worst[:10]) or "none",
        )

    if invisible_interior_fields:
        logger.error(
            "%s: loaded with strict=False — %d field(s) hold an invisible "
            "character between visible characters and were kept exactly as "
            "written, because removing it could destroy real orthography. They "
            "will not compare equal to the same word typed normally, and a tag "
            "with one in it fell through to the unrecognised-tag report above.",
            data_path, invisible_interior_fields,
        )

    if texts and not any(any(sample) for sample in labels):
        # Every ID is 0 (O). Training on this yields a model that predicts
        # nothing yet scores ~99% token accuracy, so say it out loud.
        logger.error(
            "%s: every label in all %d samples is O — a model trained on this "
            "can never predict an entity.", data_path, len(texts),
        )
    if not texts:
        # A file whose every line was rejected is not an empty file, and saying
        # so sends the operator to check the wrong thing. Any well-formed line
        # would have produced a sample, so if none did, all of them failed.
        if malformed_lines:
            logger.error(
                "%s: no samples found — the file is not empty, but every one "
                "of its %d non-blank line(s) was dropped as malformed. Check "
                "the field separator and the encoding before concluding the "
                "data is missing.", data_path, malformed_lines,
            )
        else:
            logger.error("%s: no samples found — the file is empty or blank.",
                         data_path)

    logger.info("Loaded %d samples from %s", len(texts), data_path)
    return texts, labels


def split_dataset(texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Sequentially split into train/val/test. Ratios should sum to ~1.0."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    total = len(texts)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return (
        texts[:train_end], labels[:train_end],
        texts[train_end:val_end], labels[train_end:val_end],
        texts[val_end:], labels[val_end:],
    )
