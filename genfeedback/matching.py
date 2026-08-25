"""Compare entities extracted from a JD vs a resume.

Every gap this module returns is written verbatim into a rejection email, so a
fabricated gap ("you lack Jira" to a candidate whose resume lists Jira) is far
worse than a missed one.

The comparison is anchored on :mod:`genfeedback.lexicon`. A job-description span
becomes a candidate-facing gap ONLY when it names a term the lexicon
recognises, and the comparison is then plain equality between the recognised
terms on the two sides. Everything this module used to do instead — n-gram
subset tolerance, per-category coverage lanes, manufactured-short-word
suppression, degree-subject ranking — existed to guess which of the extractor's
nouns were requirements and which were noise. The lexicon answers that outright,
so the guesses are gone rather than refined.

What follows from that anchor:

  * The category axis is IGNORED on the resume side. 14.6% of the gold spans in
    data/ have a surface filed under more than one skill-like category (figma,
    jira, selenium, azure and github are all tagged both `tools` and
    `technicalskill`), so a term named on both sides has a large chance of being
    filed differently on each. A recognised term is evidence the resume names
    it, whatever bucket the extractor dropped it in — `qualification` and
    `designation` included. A candidate whose degree is in machine learning, or
    whose job title is "Kubernetes Platform Lead", has named those things, and
    telling them otherwise is the fabrication this module exists to prevent.
  * Span boundaries are context-dependent — the same prose gives a JD
    "production ML systems" and a resume "ML" — so recognition scans INSIDE a
    span rather than comparing whole spans.
  * Requirements the lexicon does not recognise do NOT vanish. They are reported
    by `find_missing_notes` as requirements this tool could not verify, because
    silently dropping a requirement is how a real one disappears.
  * Qualifications are compared on DEGREE LEVEL alone. Subject is deliberately
    not compared: the extractor cannot be relied on to deliver one ("Education
    Computer Science", "Bachelor degree , 2019"), and the previous attempt to
    match subjects dropped a genuine bachelor's-vs-master's gap. Level is the
    only qualification question that can be answered honestly, so it is the only
    one asked; when either side states no rankable degree the comparison abstains
    and says so, and never waives silently.
  * Experience is compared numerically and abstains unless both sides parse, as
    a third of the experience spans in data/ are employment dates rather than
    durations.

`designation` is the one category excluded on the JD side: a posting's own job
title describes the opening, not a candidate shortcoming.

Caveats about comparisons that could not be made are returned by
`find_missing_notes`, deliberately NOT inside the gap dict: the gap dict is fed
straight into the feedback email, and a parser caveat must never reach a
candidate.
"""

import logging

from .labels import LABEL_TO_ID
from .lexicon import degree_level, recognised_terms
from .normalize import canonical, is_generic, parse_years

logger = logging.getLogger(__name__)

# A JD's own job title describes the opening, not a candidate shortcoming.
EXCLUDED_CATEGORIES = ("designation",)
EXPERIENCE_CATEGORY = "yearofexperience"
QUALIFICATION_CATEGORY = "qualification"

# Every category the NER can emit, taken from the label scheme so the two
# cannot drift apart.
ALL_CATEGORIES = frozenset(
    name.split("-", 1)[1].lower() for name in LABEL_TO_ID if name != "O"
)

# The categories `find_missing` actually compares. Callers (quality.py) count
# spans in these to decide whether there is anything comparable at all: a
# document holding nothing but a job title has nothing to compare.
COMPARABLE_CATEGORIES = ALL_CATEGORIES - frozenset(EXCLUDED_CATEGORIES)


def _resume_terms(entities):
    """Every known term the resume names, pooled across all categories."""
    terms = set()
    for spans in (entities or {}).values():
        for span in spans or ():
            terms |= recognised_terms(span)
    return terms


def _skill_gaps(jd_items, resume_terms):
    """JD requirements naming a known term the resume never names.

    Reports the ORIGINAL JD surface string, never the canonical key, and reports
    each missing term once: two JD spans naming the same skill are one gap.
    """
    gaps, reported = [], set()
    for item in jd_items or ():
        missing = recognised_terms(item) - resume_terms - reported
        if missing:
            reported |= missing
            gaps.append(item)
    return gaps


def _resume_degree_level(resume_entities):
    """Highest degree level the resume states, or None when it states none.

    Read from `qualification` only. Degrees stay in that lane in data/ — 1358
    spans rank as a degree there against 5 in every other category combined, and
    those 5 are mis-hits ("Scrum Master", "GPON Master") rather than degrees
    filed elsewhere, so widening the lane would waive real gaps.
    """
    levels = [
        degree_level(span)
        for span in resume_entities.get(QUALIFICATION_CATEGORY) or ()
    ]
    levels = [level for level in levels if level is not None]
    return max(levels) if levels else None


def _qualification_gaps(jd_items, resume_entities):
    """JD degree requirements the resume's highest degree falls short of.

    Empty unless BOTH sides state a rankable degree; `find_missing_notes`
    reports every requirement this leaves unchecked.
    """
    held = _resume_degree_level(resume_entities)
    if held is None:
        return []
    gaps, reported = [], set()
    for item in jd_items or ():
        wanted = degree_level(item)
        if wanted is not None and wanted > held and wanted not in reported:
            reported.add(wanted)
            gaps.append(item)
    return gaps


def _experience_pairs(entities):
    """(original span, parsed years or None) for one side's experience spans."""
    return [(span, parse_years(span)) for span in entities.get(EXPERIENCE_CATEGORY, [])]


def _experience_gaps(jd_entities, resume_entities):
    """JD experience spans the resume falls short of.

    Empty unless BOTH sides parse to numbers. The resume is read as the max of
    its parseable spans, which is what makes the model's split spans work: it
    tags "Total years of experience 9" as ["years", "9"], and only "9" parses.
    """
    jd_pairs = _experience_pairs(jd_entities)
    resume_pairs = _experience_pairs(resume_entities)
    jd_numbers = [v for _, v in jd_pairs if v is not None]
    resume_numbers = [v for _, v in resume_pairs if v is not None]
    if not jd_numbers or not resume_numbers:
        logger.info(
            "Skipping experience comparison, no duration on one side: jd=%r resume=%r",
            [s for s, _ in jd_pairs],
            [s for s, _ in resume_pairs],
        )
        return []
    resume_best = max(resume_numbers)
    return [s for s, v in jd_pairs if v is not None and v > resume_best]


def find_missing(jd_entities, resume_entities):
    """Per category, return the JD requirements the resume does not cover.

    Args:
        jd_entities:     dict {category: [spans]} for the job description
        resume_entities: dict {category: [spans]} for the resume

    Returns {category: [spans]} holding the ORIGINAL JD surface strings.
    Categories with no gaps are absent, as are `designation`, every requirement
    the lexicon does not recognise, and every term the resume names under any
    category at all.
    """
    resume_terms = _resume_terms(resume_entities)
    missing = {}
    for category, jd_items in jd_entities.items():
        if category not in COMPARABLE_CATEGORIES:
            if category not in EXCLUDED_CATEGORIES:
                logger.warning(
                    "Ignoring unknown entity category %r with %d span(s); it is "
                    "not in the label scheme, so there is no rule for comparing it",
                    category,
                    len(jd_items or ()),
                )
            continue
        if category == EXPERIENCE_CATEGORY:
            gaps = _experience_gaps(jd_entities, resume_entities)
        elif category == QUALIFICATION_CATEGORY:
            gaps = _qualification_gaps(jd_items, resume_entities)
        else:
            gaps = _skill_gaps(jd_items, resume_terms)
        if gaps:
            missing[category] = gaps
    return missing


def _quote_spans(spans):
    return ", ".join("'{}'".format(s) for s in spans)


def _unverified_note(jd_entities):
    """JD requirements that name nothing the lexicon knows.

    These are the price of anchoring on a curated vocabulary, and they are said
    out loud rather than dropped. Most are the extractor's noise — a city, a
    section heading, a candidate's name — but a genuine skill the lexicon has
    simply never heard of lands here too, and an operator has to be able to see
    the difference. Recruiting boilerplate is left out: "a plus" and "related
    field" are not requirements anybody failed to verify.
    """
    unverified = []
    for category, spans in (jd_entities or {}).items():
        if category not in COMPARABLE_CATEGORIES or category in (
                EXPERIENCE_CATEGORY, QUALIFICATION_CATEGORY):
            continue                        # both have a note of their own
        for span in spans or ():
            if not recognised_terms(span) and not is_generic(canonical(span)):
                unverified.append(span)
    if not unverified:
        return []
    return [
        "Could not check these job-description items against the resume: {}. "
        "They match no skill, tool or qualification this tool knows, so they "
        "were left out of the gap list rather than reported as something the "
        "candidate lacks.".format(_quote_spans(unverified))
    ]


def _qualification_notes(jd_entities, resume_entities):
    """Degree requirements `_qualification_gaps` could not rank on one side."""
    jd_spans = jd_entities.get(QUALIFICATION_CATEGORY) or ()
    wanted = [span for span in jd_spans if degree_level(span) is not None]
    unrankable = [
        span
        for span in jd_spans
        if degree_level(span) is None and not is_generic(canonical(span))
    ]
    notes = []
    if unrankable:
        notes.append(
            "Could not compare qualifications: no degree level this tool can "
            "rank appears in {}, so the comparison was skipped rather than "
            "resolved either way.".format(_quote_spans(unrankable))
        )
    if wanted and _resume_degree_level(resume_entities) is None:
        notes.append(
            "Could not compare qualifications: the job description asks for {} "
            "but no degree was recognised in the resume, so no qualification "
            "gap was reported either way.".format(_quote_spans(wanted))
        )
    return notes


def _experience_notes(jd_entities, resume_entities):
    """Duration requirements `_experience_gaps` had to abstain from."""
    jd_pairs = _experience_pairs(jd_entities)
    if not jd_pairs:
        return []                           # the JD asked for no duration at all
    resume_pairs = _experience_pairs(resume_entities)
    jd_spans = [s for s, _ in jd_pairs]
    resume_spans = [s for s, _ in resume_pairs]

    if not any(v is not None for _, v in jd_pairs):
        return [
            "Could not compare years of experience: the job description states "
            "{} rather than a duration this tool can read, and durations are "
            "never inferred from dates.".format(_quote_spans(jd_spans))
        ]
    if any(v is not None for _, v in resume_pairs):
        return []
    if resume_spans:
        return [
            "Could not compare years of experience: the resume states {} rather "
            "than a duration, and this tool does not infer durations from "
            "dates.".format(_quote_spans(resume_spans))
        ]
    return [
        "Could not compare years of experience: no duration was extracted from "
        "the resume, so the job description's {} was left unchecked."
        .format(_quote_spans(jd_spans))
    ]


def find_missing_notes(jd_entities, resume_entities):
    """Caveats about comparisons `find_missing` could NOT make.

    Returned separately from the gap dict on purpose: the gap dict is fed
    verbatim into the feedback email, and a parser caveat must never reach a
    candidate. These are for the operator / UI only.
    """
    return (
        _unverified_note(jd_entities)
        + _qualification_notes(jd_entities, resume_entities)
        + _experience_notes(jd_entities, resume_entities)
    )
