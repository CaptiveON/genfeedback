"""The vocabulary this tool is willing to talk to a candidate about.

Why a lexicon at all
--------------------
The NER model is an unreliable extractor, and every heuristic that tried to
absorb that unreliability inside the matcher made it worse. It tags the city
"Leeds" and a candidate's own name as tools, "Education" as a qualification, and
"plus" / "teams" / "related field" as requirements. A matcher that treats every
JD span as a requirement then has to decide, span by span, which noun is a skill
— a judgement no amount of n-gram tolerance, alias expansion or plausibility
scoring can make, because the information is not in the span.

So the decision moves somewhere it CAN be made: a curated list. A requirement
that is not a recognised skill, tool or qualification can never become a
candidate-facing gap. That one rule replaces, at the root, every rule that used
to guess — "Leeds", "plus", "related field", "teams", "product", "junior
engineers", section headings, PDF debris and HTML tags are simply absent from
the list, and no rule has to be written for any of them.

Where the vocabulary comes from
-------------------------------
:mod:`genfeedback.datagen` is IMPORTED, not copied. Those lists are what the NER
model was trained on, so they are the vocabulary it can be relied on to emit; a
copy would drift from the model's own world silently and in both directions —
the matcher recognising terms the model cannot tag, and failing to recognise
ones it was trained to. The dependency therefore runs matcher -> generator,
which is the direction the truth flows, and the import is pure data (`datagen`
pulls in only `config` and `tokenization` and generates nothing at import time).

`_EXTRA_SKILLS` adds vocabulary real resumes use and the synthetic generator
never learned. It is deliberately short and high-precision, because the two
errors are not symmetric: a wrong entry here becomes a sentence in a rejection
email, a missing one only an operator note.

Recognition, not equality
-------------------------
Span boundaries are context-dependent — the same prose gives "production ML
systems" in a JD and "ML" in a resume, and the tokenizer hands over ". NET" for
".NET" — so :func:`recognised_terms` scans INSIDE a span instead of comparing
whole spans. It windows the raw written words and canonicalises each window,
which is what lets :data:`normalize.ALIASES` fire on a term sitting inside a
longer span ("ML" -> "machine learning") where a whole-span lookup never could.
"""

import logging

from . import datagen
from .normalize import canonical

logger = logging.getLogger(__name__)

# Vocabulary the synthetic generator never learned but real resumes and postings
# use constantly. Admission rule: the term must be unambiguous as a WHOLE span in
# a resume or JD, because anything admitted here can be printed to a candidate as
# something they lack. Deliberately absent, so nobody "helpfully" adds them:
#   C          : canonicalisation splits "Objective-C" and "C/C++" apart, and a
#                bare "C" requirement is worth less than the fabrications it
#                risks. A JD asking for C now abstains instead.
#   Spring     : "Spring Boot" is in the generator's list; the bare word is a
#                season.
#   Ray, Excel : a person's name and a verb.
#   web services : would let "RESTful Web Services" answer an AWS requirement.
# A surface VARIANT of a term already here belongs in `normalize.ALIASES`, never
# in this list: two entries for one thing are two DIFFERENT canonical terms, and
# a JD naming one against a resume naming the other is then a fabricated gap.
_EXTRA_SKILLS = (
    # Languages and frameworks
    "Rust", "Kotlin", "Swift", "PHP", "Perl", "Julia", "Svelte", "Next.js",
    "jQuery", "Tailwind CSS", "Redux", ".NET", "ASP.NET", "Laravel",
    "Ruby on Rails", "Hibernate", "gRPC", "OAuth",
    # Data platforms and stores
    "Snowflake", "dbt", "Databricks", "Redshift", "BigQuery", "Cassandra",
    "DynamoDB", "Neo4j", "Hive", "ClickHouse", "PySpark", "relational database",
    "data modeling", "data warehousing", "data visualization", "data pipelines",
    # Infrastructure and operations
    "Helm", "Pulumi", "Argo CD", "Splunk", "Kibana", "OpenShift",
    "CloudFormation", "PowerShell", "CI/CD", "MLOps", "cloud computing",
    "distributed systems", "system design",
    # Testing and ways of working
    "pytest", "JUnit", "Cypress", "Playwright", "Jest", "unit testing",
    "integration testing", "quality assurance", "code review", "Agile",
    "Scrum", "Kanban", "A/B testing",
    # Modelling and analysis
    "artificial intelligence", "generative AI", "large language models",
    "prompt engineering", "LangChain", "Hugging Face", "MLflow", "ONNX",
    "Jupyter", "Matplotlib", "Seaborn", "Plotly", "SciPy",
)

def _canonical_set(*groups):
    """Fold raw vocabulary lists into one set of comparison keys."""
    terms = {canonical(raw) for group in groups for raw in group}
    terms.discard("")
    return frozenset(terms)


# Everything that both states and satisfies a skill requirement. The generator's
# three skill-like lists are one set here on purpose: the boundary between a
# "technical skill", a "tool" and a "soft skill" is noise the extractor does not
# reproduce (figma, jira, selenium, azure and github are tagged as both tools and
# technicalskill in data/), so nothing may depend on it.
SKILL_TERMS = _canonical_set(
    datagen.TECHNICALSKILL, datagen.TOOLS, datagen.SOFTSKILL, _EXTRA_SKILLS
)

# Degrees, fields and the composed "{degree} in {field}" forms. The two private
# generator lists are read directly because `datagen.QUALIFICATION` holds only
# the COMPOSED forms, while a resume routinely names the parts on their own
# ("Masters degree", "Computer Science").
QUALIFICATION_TERMS = _canonical_set(
    datagen._DEGREES, datagen._FIELDS, datagen._STANDALONE_DEGREES,
    datagen.QUALIFICATION,
)

# Degree ranks, used to answer the only qualification question this tool can
# answer honestly: is the candidate's highest degree below the level asked for?
# Keys are CANONICAL words, so most abbreviations arrive here already expanded by
# normalize.ALIASES ("MSc" -> "master of science"); the abbreviations are listed
# anyway because an alias only fires on a whole span, and inside "M.Sc in
# Physics" the word form is all there is.
#
# Deliberately absent, because they are not degree levels in a resume:
#   bare "ms"           : "MS Office" and "MS SQL Server" are tagged in data/, a
#                         master's is not. The dotted "m.s" is kept — nobody
#                         writes Microsoft that way.
#   bare "graduate"     : "graduate experience" is tagged in data/.
#   associate           : "Cisco Certified Network Associate", "Associate
#                         Engineer" — a certification or a job title, and reading
#                         it as a degree would let any bachelor's waive a CCNA.
#   diploma, high school: a "Post Graduate Diploma" is a master's in India and a
#                         school-leaving certificate elsewhere. Ranking it either
#                         way invents a gap for one of them; abstaining does not.
# Known false positive, accepted: "Scrum Master" reads as a master's. It only
# matters for spans the extractor files under `qualification`, which it does not
# in data/ (2 occurrences, both `designation`).
_DEGREE_RANKS = {
    "bachelor": 1, "bsc": 1, "b.sc": 1, "b.s": 1, "b.e": 1, "btech": 1,
    "b.tech": 1, "bca": 1, "bba": 1, "undergraduate": 1,
    "master": 2, "msc": 2, "m.sc": 2, "m.s": 2, "m.e": 2, "mtech": 2,
    "m.tech": 2, "mba": 2, "mca": 2, "postgraduate": 2,
    "doctor": 3, "doctorate": 3, "doctoral": 3, "phd": 3, "ph.d": 3,
}

# Longest known term, in words. Bounds the scan window in `recognised_terms`, so
# a paragraph the extractor mis-tagged as one span costs time linear in its
# length rather than quadratic. Derived, so a longer term cannot outgrow it.
_MAX_TERM_WORDS = max(len(t.split()) for t in SKILL_TERMS | QUALIFICATION_TERMS)


def is_known_skill(canonical_term):
    """Is this canonical term a skill, tool or soft skill the lexicon knows?"""
    return canonical_term in SKILL_TERMS


def is_known_qualification(canonical_term):
    """Is this canonical term a degree, field or qualification the lexicon knows?"""
    return canonical_term in QUALIFICATION_TERMS


def _is_known(canonical_term):
    return canonical_term in SKILL_TERMS or canonical_term in QUALIFICATION_TERMS


def recognised_terms(span):
    """Every known term a raw span names, as canonical strings.

    Windows of written words are tried longest-first and consumed once matched,
    so "Master of Science in Computer Science" is read as the qualification it
    is rather than as a pile of shorter fragments, and a span that is mostly
    noise still gives up the one real term inside it ("Summary Software
    Engineer" -> "software engineer" is not a skill, but "production ML systems"
    -> "machine learning" is).

    Words carrying no alphanumeric character are dropped before windowing, so a
    separator the tokenizer emitted as its own word ("natural , language ,
    processing") cannot break a term in half.

    A ONE-CHARACTER term must have been WRITTEN as its own word. Canonicalisation
    flattens punctuation, so "(C)" and "R&D" would otherwise read as the
    languages C and R — a skill the document never claimed. "C / C++" still names
    C, because the tokenizer emits that as three separate words and the source
    really did write one of them as "C".
    """
    words = [w for w in span.split() if any(c.isalnum() for c in w)]
    written = {w.lower() for w in words if w.isalnum()}
    terms = set()
    start = 0
    while start < len(words):
        for end in range(min(start + _MAX_TERM_WORDS, len(words)), start, -1):
            term = canonical(" ".join(words[start:end]))
            if not _is_known(term) or (len(term) == 1 and term not in written):
                continue
            terms.add(term)
            start = end
            break
        else:
            start += 1
    return terms


def recognise(span):
    """The most specific known term a raw span names, or None when it names none.

    Single-answer form of :func:`recognised_terms`, for callers that need one
    label for one span. The longest match wins, on word count then length, so
    the answer does not depend on set iteration order.
    """
    return max(
        recognised_terms(span),
        key=lambda term: (len(term.split()), len(term)),
        default=None,
    )


def degree_level(span):
    """Rank of the highest degree a span states, or None when it states none.

    Ranks are ordered and comparable (bachelor 1 < master 2 < doctorate 3) and
    mean nothing else; the numbers are not a score.
    """
    ranks = [_DEGREE_RANKS[w] for w in canonical(span).split() if w in _DEGREE_RANKS]
    return max(ranks) if ranks else None
