"""Confidence gate on NER extraction, run BEFORE any gap matching.

Extraction failure is silent: when the NER model finds nothing, it returns an
empty dict, and every downstream stage reads that emptiness as good news.
`find_missing({}, {})` is `{}`, an empty gap set renders as "no skill gaps
detected", and the template writer turns that into a warm email. So an empty
resume, a scanned PDF that pasted as gibberish, or a resume in a language the
model never saw all arrive at the candidate as a flawless review — the pipeline
cannot tell "we compared and found nothing missing" apart from "we could not
read this at all".

This module makes the difference explicit. Callers ask `extraction_problem`
first and refuse to produce feedback when it returns a reason, so a failure is
reported as a failure instead of being laundered into a compliment.

The question it asks is NOT "do these spans look like words". Two earlier
rounds asked that, scoring spans for readability, and it failed in both
directions at once: a header block of a name and a street address scored ten
perfectly readable "terms" and passed, while `C++`, `C#` and `R&D` scored as
unreadable debris and a real posting built on them was refused with a message
claiming its text was garbled.

The question it asks instead is "did we recognise anything we can actually
compare", and `lexicon.recognise` answers it. A span only counts when it names
a skill, tool or qualification the lexicon knows. That single rule replaces
every span-plausibility heuristic this module used to carry, because garbage
does not produce recognised terms: a base64 blob, an RTF header, a cid-garbled
paste, a name and a street address all recognise as nothing at all, with no
scoring involved. It also fixes the false rejects for free — `C++` is a known
skill, so it passes by construction — and closes the misfiled-noun hole the
scoring approach could never reach: the model tags the city `Leeds` as a tool,
but `Leeds` is not a known skill, so a posting whose only content is `Leeds`
has nothing to compare and is refused rather than silently cleared.

The two sides get different rules, because a thin JD and a thin resume fail in
opposite directions.

A thin JD is safe but a requirement-less JD is not. One requirement is one
honest question ("does this resume show Kubernetes?") with a truthful answer
and nothing invented, so a terse posting needs only ONE recognised requirement.
What a JD must never do is contribute ZERO and have that read as agreement: an
empty gap dict renders as "no skill gaps detected" for a candidate who matches
nothing about the role. The count is taken over `matching.COMPARABLE_CATEGORIES`
so a posting that extracts only as a job title still refuses.

Durations do not count towards that floor even though `yearofexperience` is
comparable. "5 years" states no requirement a candidate can be told they lack
except in combination with something else, and a bare number is the one shape
digit noise reliably produces — a cid dump alone yields twelve of them.

A thin RESUME is the dangerous one: fewer resume terms mean MORE gaps, because
every JD requirement the resume cannot be shown to cover is written into the
email as something the candidate lacks. But refusing a resume that recognises
as nothing would refuse the tool's own reason for existing — a career changer
with no relevant skills is not a broken input, they are the candidate who most
needs the letter, and `lexicon.recognise` finds nothing in a genuine retail
supervisor's CV. So a resume with nothing recognised is not refused outright;
it is checked against its SOURCE TEXT, and the check is the honest one:

  * readable text naming nothing known -> PROCEED, with a warning logged.
    Every gap the letter then lists is TRUE of the text that was read.
  * unreadable input -> REFUSE. The gaps would be true of the encoding rather
    than of the document, so "your resume does not show Python" is a false
    claim about a candidate whose resume does.

Readability is measured on the text, not on the spans, as the share of its
chunks that are ordinary written words (`_is_word`). Measured over the corpus
in the self-test the separation is total and needs no tuning: markup, encodings
and non-Latin scripts score 0.00-0.43 (base64, CJK, Arabic and a cid dump 0.00,
RTF 0.21, PDF binary 0.42, an HTML page source 0.43) while every real document
scores 0.89-1.00 — accented European, `C++`-heavy, and an ASCII-art resume of
rules and bullets included. Applied to a whole document this is safe in a way
that scoring individual spans never was: `C++` is one word among hundreds of
ordinary English ones, so a document built on it is never in doubt.

That test also subsumes the separate script test this module used to run, and
catches what the lexicon alone cannot. An HTML page source contains the literal
words "html" and "css", which ARE known skills, so recognition passes it and
only the readability of the text refuses it.

The remaining length check exists for one shape: a paste that stopped after the
letterhead. A name, a street and an email address are readable and recognise as
nothing, which is also true of a career changer's CV — length is what separates
them, and the margin is wide (7 words against 141 in the self-test corpus).

Refusal reasons are separated by cause and each states only what was observed.
"We read this and there is too little of it" and "we could not read this" are
different failures with different fixes, and telling an operator whose posting
is merely terse that their text was "not in English" is simply false.
"""

import logging

from .lexicon import recognise
from .matching import COMPARABLE_CATEGORIES

logger = logging.getLogger(__name__)

# Recognised terms a RESUME must contribute before it counts as comparable
# evidence. One, not two: under the old gate a "term" was any readable span and
# two was a hedge against a single debris token counting as a profile. A
# lexicon term is not debris, and a resume whose one skill is the one the
# posting asks for must not be refused.
MIN_SPANS = 1

# Recognised requirements a JD must yield. One is enough, and one is required.
MIN_JD_REQUIREMENTS = 1

# Share of a document's chunks that must be ordinary words before the model can
# be said to have read it. Half sits in a wide empty band: the worst real
# document measured scores 0.89 and the best garbled one 0.43. Deliberately set
# nearer the garbled end — a false reject tells a user their good text is
# unreadable, which is the failure the previous two rounds shipped.
MIN_PROSE_RATIO = 0.5

# Words a resume must hold when nothing in it was recognised, to be a thin
# profile rather than a letterhead. Applied to the RESUME only, and only in
# that case: a JD may legitimately be one line ("Must know Kubernetes.").
MIN_PROSE_WORDS = 40

# Longer than any written word, so a base64 line or a minified script counts as
# markup however many letters it holds.
_MAX_WORD_LENGTH = 20

# Punctuation that lives INSIDE a word, all of it load-bearing: without it
# "node.js", "scikit-learn", "CI/CD", "R&D" and "C++11" would be read as
# markup, which is how the previous gate came to refuse a perfectly readable
# posting. Trailing punctuation needs no entry here — "C++" and "C#" have
# already reduced to their "C" core by the time this is consulted.
_IN_WORD_PUNCTUATION = "+#.-'&/"

# Hints are carried alongside the fragment that earned them and are never
# merged into one generic sentence, because that is how the old gate came to
# tell an operator with a one-line JD that their text was unreadable.
_HINT_UNREADABLE = (
    "That is what a scanned or badly-encoded PDF, a page of markup, or text in "
    "an alphabet the model was not trained on looks like. Please paste the "
    "text as plain readable English and try again."
)
_HINT_THIN = (
    "The text itself was read; there is simply not enough of it to compare "
    "against. Please supply the full text and try again."
)
_HINT_NO_REQUIREMENT = (
    "Please paste the full posting, including the skills, tools and "
    "qualifications it asks for."
)
_HINT_EMPTY = "Please paste the text and try again."
# Nothing here is wrong with the user's input, so this must not tell them to
# fix it. The two-argument call is the only way to reach this, and the fix
# belongs to whoever made it.
_HINT_NO_TEXT = (
    "Nothing is necessarily wrong with the resume itself. Pass resume_text to "
    "extraction_problem and run the analysis again."
)


def _recognised(entities):
    """Distinct lexicon terms an entity dict contributes to a comparison.

    Distinct, because one repeated span must not read as several findings, and
    limited to `COMPARABLE_CATEGORIES` because a span the matcher discards is
    not evidence of anything — a JD extracting only as a job title has no
    requirement in it however well the title is recognised.

    Recruiting boilerplate needs no separate test here: "plus", "related
    field" and "teams" are not in the lexicon, so they never arrive.
    """
    terms = set()
    for category, spans in (entities or {}).items():
        if category not in COMPARABLE_CATEGORIES:
            continue
        for span in spans or ():
            term = recognise(span)
            if term:
                terms.add(term)
    return terms


def _is_word(token):
    """True when a whitespace chunk is an ordinary written word.

    Punctuation on either end is ignored, so "(Python)," and "/Type" reduce to
    their core. What is left must be short enough to be a word, must carry a
    letter of the alphabet the model was trained on, and must be built only
    from letters, digits and `_IN_WORD_PUNCTUATION`.

    The ASCII-letter requirement is what replaces the old separate script
    test: a CJK or Arabic chunk is alphanumeric but carries no such letter, so
    a document in either scores near zero, while "Zürich", "José" and
    "Ingénieur" carry plenty and score as the ordinary words they are.
    """
    start, end = 0, len(token)
    while start < end and not token[start].isalnum():
        start += 1
    while end > start and not token[end - 1].isalnum():
        end -= 1
    core = token[start:end]
    if not core or len(core) > _MAX_WORD_LENGTH:
        return False
    if not any(c.isascii() and c.isalpha() for c in core):
        return False
    return all(c.isalnum() or c in _IN_WORD_PUNCTUATION for c in core)


def _words(text):
    """The ordinary written words of a text."""
    return [chunk for chunk in text.split() if _is_word(chunk)]


def _unreadable(text, side):
    """Reason a source text could not be read at all, or None.

    Only reachable when a caller supplies the text; with entity dicts alone
    recognition is the whole gate.
    """
    if text is None:
        return None
    if not text.strip():
        return ("the {} was empty".format(side), _HINT_EMPTY)
    # Chunks carrying no letter or digit at all are decoration — a rule of
    # dashes, a bullet, a pipe — and are neither readable nor unreadable, so
    # they are left out of the measurement entirely rather than counted
    # against a document for being laid out. A text that is ALL decoration has
    # no chunks left and is unreadable by the same rule.
    chunks = [c for c in text.split() if any(ch.isalnum() for ch in c)]
    words = _words(text)
    if chunks and len(words) >= len(chunks) * MIN_PROSE_RATIO:
        return None
    return (
        "the {} could not be read: it holds {} ordinary words among {} chunks "
        "of text, so most of it is not readable prose — markup, an encoding, "
        "symbols, or an alphabet this model was not trained on".format(
            side, len(words), len(chunks)
        ),
        _HINT_UNREADABLE,
    )


def extraction_ok(entities):
    """True when an entity dict names enough recognised terms to be compared.

    The resume floor. A job description is held to `MIN_JD_REQUIREMENTS`
    instead, which `extraction_problem` applies.
    """
    return len(_recognised(entities)) >= MIN_SPANS


def _jd_problem(entities, text):
    """Reason the job description cannot be compared against, or None."""
    unreadable = _unreadable(text, "job description")
    if unreadable:
        return unreadable

    requirements = _recognised(entities)
    if len(requirements) >= MIN_JD_REQUIREMENTS:
        return None
    # Read, but naming nothing comparable. This is the dangerous shape, not a
    # thin one: `find_missing` would compare against an empty requirement set
    # and its empty gap dict renders as "no skill gaps detected" for a
    # candidate who matches nothing about the role.
    logger.warning(
        "No recognised requirement in the job description; the model returned %r",
        entities,
    )
    return (
        "the job description names no skill, tool or qualification this tool "
        "recognises, so there is no requirement for a resume to be compared "
        "against",
        _HINT_NO_REQUIREMENT,
    )


def _resume_problem(entities, text):
    """Reason the resume cannot be compared, or None.

    Nothing recognised is not on its own a refusal — see the module docstring.
    It is a refusal only when the text cannot back it up.
    """
    unreadable = _unreadable(text, "resume")
    if unreadable:
        return unreadable
    if extraction_ok(entities):
        return None

    if text is None:
        return (
            "nothing in the resume was recognised as a skill, tool or "
            "qualification, and the resume text was not supplied, so a "
            "candidate who simply has none cannot be told apart from a "
            "document that never got read",
            _HINT_NO_TEXT,
        )

    words = _words(text)
    if len(words) < MIN_PROSE_WORDS:
        return (
            "nothing in the resume was recognised as a skill, tool or "
            "qualification, and it holds only {} words — too little to read "
            "as a profile rather than a letterhead or a truncated "
            "paste".format(len(words)),
            _HINT_THIN,
        )

    # Readable, and it genuinely names nothing the lexicon knows. Every gap the
    # letter goes on to list is true of this text, so the analysis proceeds;
    # the warning is for the operator, who will want to know that the whole
    # posting came back as a gap because the candidate matched none of it.
    logger.warning(
        "No recognised skill or qualification in the resume, but its %d words "
        "read as ordinary text; proceeding, so every requirement will be "
        "reported as a gap. Model returned %r",
        len(words),
        entities,
    )
    return None


def extraction_problem(jd_entities, resume_entities, jd_text=None, resume_text=None):
    """Reason the pipeline must refuse to write feedback, or None if it may proceed.

    The reason names the side that failed (job description, resume, or both) so
    the user can fix the input that is actually broken, and states only what was
    observed: a document that was read but names nothing gets a different
    message from one that could not be read.

    Args:
        jd_entities:     dict {category: [spans]} for the job description
        resume_entities: dict {category: [spans]} for the resume
        jd_text:         optional JD source text
        resume_text:     optional resume source text

    The texts are optional so the two-argument call keeps working, but passing
    them is strongly preferred and changes the verdict in both directions. They
    are the only way to catch an input the model read only part of — a page of
    markup whose stray "html" and "css" recognise as real skills — and the only
    way to tell an unqualified candidate apart from an unread document, which
    the two-argument call has to refuse rather than guess at.
    """
    findings = [
        finding
        for finding in (
            _jd_problem(jd_entities, jd_text),
            _resume_problem(resume_entities, resume_text),
        )
        if finding
    ]
    if not findings:
        return None

    detail = "; ".join(fragment for fragment, _ in findings)
    hints = []
    for _, hint in findings:
        if hint not in hints:               # both sides often fail the same way
            hints.append(hint)
    reason = "{}{}. {}".format(detail[0].upper(), detail[1:], " ".join(hints))
    logger.warning(
        "Refusing to generate feedback (jd: %d recognised requirements; "
        "resume: %d recognised terms): %s",
        len(_recognised(jd_entities)),
        len(_recognised(resume_entities)),
        reason,
    )
    return reason
