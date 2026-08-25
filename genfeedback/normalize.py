"""Canonical comparison keys for NER spans.

The NER model is unstable in two ways that matter when a JD span is compared
against a resume span. Its surface forms vary ("AWS" / "Amazon Web Services",
"PyTorch" / "torch", "data pipelines" / "data pipeline"), and it tags plain
recruiting boilerplate ("plus", "related field", "teams") as if it were a
requirement. Both make the matcher invent skill gaps, and an invented gap is a
factually wrong sentence in a rejection email a candidate actually reads.

Everything here reduces a span to one key both sides can be compared on:
`canonical()` for the key itself, `ALIASES` for the vocabulary that key folds
together, and `is_generic()` for spans that carry no advice at all.

The canonical form is a comparison key and nothing else. It is allowed to be
ugly — over-stemmed, alias-expanded, punctuation-flattened — because both sides
fold identically, and callers always report the ORIGINAL surface span. Never
show a canonical form to a candidate.
"""

import re

# Whole-string equivalences only: an alias fires when it is the entire span, not
# when it appears as a word inside one. Both key and value run through
# canonical(), so a single entry matches in both directions (JD "AWS" vs resume
# "Amazon Web Services" and the reverse). Adding the reverse entry creates a
# cycle - don't.
#
# Admission rule for new entries: the key must be unambiguous as a COMPLETE span
# in a resume or JD. Deliberately excluded, so nobody "helpfully" adds them:
#   cv  -> computer vision   : on a resume "CV" is usually curriculum vitae, and
#                              the alias would silently hide a real CV-skill gap.
#   ms  -> master of science : collides with Microsoft. msc / m.sc / m.s are safe.
#   it, cs, be, db, api, rest api : too short or too common; "REST API" is not
#                              interchangeable with "API".
# Watch-list (kept, but re-check in review if false matches show up): ts, node,
# golang, qa.
ALIASES = {
    "ai": "artificial intelligence",
    "aws": "amazon web services",
    "b.e": "bachelor of engineering",
    "b.s": "bachelor of science",
    "b.sc": "bachelor of science",
    "b.tech": "bachelor of technology",
    "bsc": "bachelor of science",
    "btech": "bachelor of technology",
    "ci cd pipeline": "ci cd",
    "cicd": "ci cd",
    "dl": "deep learning",
    "gcp": "google cloud platform",
    "genai": "generative ai",
    "golang": "go",
    "hadoop": "apache hadoop",
    "js": "javascript",
    "kafka": "apache kafka",
    "k8": "kubernetes",
    "k8s": "kubernetes",
    "k8s cluster": "kubernetes",
    "m.s": "master of science",
    "m.sc": "master of science",
    "m.tech": "master of technology",
    "mba": "master of business administration",
    "ml": "machine learning",
    "msc": "master of science",
    "mtech": "master of technology",
    "natural language processing nlp": "natural language processing",
    "nlp": "natural language processing",
    "node": "node.js",
    "nodejs": "node.js",
    "oop": "object oriented programming",
    "ph.d": "doctor of philosophy",
    "phd": "doctor of philosophy",
    "postgres": "postgresql",
    "psql": "postgresql",
    "py torch": "pytorch",
    "qa": "quality assurance",
    "rdbms": "relational database",
    "react.js": "react",
    "reactjs": "react",
    "scikit": "scikit learn",
    "scikitlearn": "scikit learn",
    "sklearn": "scikit learn",
    "spark": "apache spark",
    "tensor flow": "tensorflow",
    "torch": "pytorch",
    "ts": "typescript",
    "vue": "vue.js",
}

# Recruiting boilerplate the NER tags as a requirement. Entries are canonical
# (singular) forms, so "teams" -> "team" and "junior engineers" -> "junior
# engineer" are caught too. Must not intersect ALIASES keys.
GENERIC_TERMS = {
    "a", "an", "the", "and", "or", "we", "you", "us", "our", "your", "etc",
    "ability", "background", "bonus", "business", "candidate", "check",
    "client", "company", "customer", "degree", "domain", "education",
    "engineer", "environment", "excellent", "expectation", "experience",
    "expertise", "external", "field", "good", "great", "ideal", "industry",
    "internal", "is a plus", "job", "junior engineer", "knowledge", "member",
    "month", "must", "other", "platform", "plus", "a plus", "plus point",
    "position", "preferred", "process", "product", "proficiency", "project",
    "qualification", "related", "related field", "requirement", "required",
    "responsibility", "role", "senior engineer", "service", "should", "skill",
    "solid", "solution", "stakeholder", "strong", "summary", "system", "task",
    "team", "technology", "tool", "understanding", "various", "work",
    "working", "year",
}

# Words whose trailing "s" is part of the name, not a plural.
_NO_FOLD = {"aws", "kubernetes", "devops", "ops", "analysis", "analytics",
            "redis", "jenkins", "pandas", "windows", "rails", "status",
            "its", "cs"}


def _collapse_dotted_acronym(word):
    """"a.w.s" -> "aws", while "node.js" and "b.sc" keep their dot.

    Only single-letter segments collapse, and only from three segments up, so
    the dotted ALIASES keys ("b.e", "m.s", "ph.d") survive intact.
    """
    parts = word.split(".")
    if len(parts) >= 3 and all(len(p) == 1 and p.isalnum() for p in parts):
        return "".join(parts)
    return word


def _normalize(term):
    t = term.lower().strip()
    t = re.sub(r"[^a-z0-9+#.]+", " ", t)      # keeps c++, c#, node.js; ci/cd -> "ci cd"
    t = re.sub(r"\s+", " ", t).strip()
    words = [w.strip(".") or w for w in t.split()]
    return " ".join(_collapse_dotted_acronym(w) for w in words)


def _fold_word(w):
    if w in _NO_FOLD or not w.isalpha() or len(w) <= 3:
        return w                               # protects R, C, Go, C++, C#, node.js
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us")):
        return w[:-1]
    return w


def canonical(term):
    """Comparison key for a span: lowercased, depunctuated, singular, aliased."""
    t = _normalize(term)
    # Two alias lookups, before and after folding, so the table can be written
    # naturally (plural, dotted, cased) on both sides. Do not collapse to one.
    if t in ALIASES:
        t = _normalize(ALIASES[t])
    t = " ".join(_fold_word(w) for w in t.split())
    if t in ALIASES:
        t = " ".join(_fold_word(w) for w in _normalize(ALIASES[t]).split())
    return t


def is_generic(canon):
    """Reject spans that are recruiting boilerplate rather than requirements.

    Membership only, never a length threshold: the shortest spans here are the
    most valuable ones ("R", "C", "AWS", "NLP") while every boilerplate span is
    long ("related field", "junior engineers").
    """
    if not canon or not any(c.isalnum() for c in canon):
        return True
    return canon in GENERIC_TERMS


_MAX_PLAUSIBLE_YEARS = 60

# "may" is deliberately absent, as it always was from the abbreviations. Here it
# is a modal verb far more often than a month ("5 years experience may be
# required"), and reading it as one threw the whole span away. Nothing is lost:
# a real employment period carries a four-digit year or an open-ended end
# marker, and both are matched in their own right.
_MONTHS = (
    "january|february|march|april|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)

# A calendar year, a month name, or an open-ended end marker means the span is
# an employment period, not a duration. " to " on its own is not a marker -
# "3 to 5 years" is a duration.
_DATE_RANGE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b"
    r"|\b(?:" + _MONTHS + r")\b"
    r"|\b(?:present|current|till date|to date|ongoing|onwards)\b"
)

# A thousands-separated number, matched whole and FIRST. Left to the pattern
# below it broke apart and "1,500 years" was read as its leading 1 - a plausible
# duration invented out of a typo. Read whole it is 1500, which the plausibility
# bound then rejects, and the span abstains as it should.
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?![\d,])")

# A number written with a decimal point OR a decimal comma ("1,5 years" is how
# much of the world writes 1.5), and with the integer part optional so ".5
# years" reads as half a year instead of five. At most two fractional digits,
# which is what separates a decimal comma from a thousands separator.
_NUMBER_RE = re.compile(
    _THOUSANDS_RE.pattern + r"|\d+(?:[.,]\d{1,2}(?!\d))?|[.,]\d{1,2}(?!\d)"
)

# An ordinal is a position, never a duration: "10th grade" is a school class and
# "1st year" is a point in time. Suffix only, so "10 years" is untouched.
_ORDINAL_SUFFIX_RE = re.compile(r"\s*(?:st|nd|rd|th)\b")

# Schooling vocabulary. Guards the other half of the ordinal problem, where the
# number leads ("grade 10", "semester 6") and nothing marks it as an ordinal.
_SCHOOLING_RE = re.compile(
    r"\b(?:grade|grades|class|classes|standard|semester|semesters|percentile|"
    r"percentage|cgpa|gpa|marks|matriculation)\b"
)

# A minus glued to a digit ("-3 years", "over -3 years") is a malformed
# duration, not a range. Genuine ranges stay out of it without a lookbehind on
# whitespace: in "3-5 years" the minus follows a digit, and in "3 - 5 years" a
# space follows the minus, so nothing is glued to it. That half of the old
# lookbehind only ever exempted the malformed form it was meant to catch.
_NEGATIVE_RE = re.compile(r"(?<!\d)-(?=[\d.,])")

_MONTH_UNIT_RE = re.compile(
    r"(" + _NUMBER_RE.pattern + r")\s*(?:months?|mos?)\b"
)

# Ordered largest first so "twenty one years" reads as 20, not 1.
_WORD_NUM = {
    "twenty": 20, "fifteen": 15, "twelve": 12, "eleven": 11, "ten": 10,
    "nine": 9, "eight": 8, "seven": 7, "six": 6, "five": 5, "four": 4,
    "three": 3, "two": 2, "one": 1,
}


def _looks_like_date_range(s):
    return bool(_DATE_RANGE_RE.search(s))


def _quantities(s):
    """(value, start, end) for every number in `s` that states a quantity.

    Ordinals are dropped rather than read, because a rank is not an amount:
    without this "10th grade" parses as a decade of experience, and the
    candidate is told they are short of years they were never asked for.
    """
    found = []
    for match in _NUMBER_RE.finditer(s):
        if _ORDINAL_SUFFIX_RE.match(s, match.end()):
            continue
        text = match.group()
        # A comma is a decimal point in half the world and a thousands separator
        # in the other half. Only the SHAPE of the number says which, so the
        # shape decides rather than a locale nobody supplied.
        digits = (text.replace(",", "") if _THOUSANDS_RE.fullmatch(text)
                  else text.replace(",", "."))
        found.append((float(digits), match.start(), match.end()))
    return found


def parse_years(span):
    """Duration in years stated by a span, or None when it states no duration.

    Returns None for employment date ranges. Deriving a duration from
    "X to Present" needs a reference date, which would make identical inputs
    produce different feedback on different days; and a bounded range is one
    job's period, not a career total, which would need overlap detection the
    NER cannot provide. A wrong number here becomes a specific false claim
    about the candidate's career, so this abstains instead of guessing.

    Same reasoning applied to the shapes that used to parse as something they
    are not: school grades and ordinals ("10th grade" -> 10 years), negative
    numbers ("-3 years" -> 3), and a non-string input, which raised. Each now
    abstains. Months are added to years rather than discarded, so "2 years 6
    months" is 2.5 and not 2.
    """
    if not isinstance(span, str) or not span.strip():
        return None                      # callers pass whatever the NER emitted
    s = span.lower().strip()
    if _looks_like_date_range(s) or _SCHOOLING_RE.search(s) or _NEGATIVE_RE.search(s):
        return None

    months_match = _MONTH_UNIT_RE.search(s)
    numbers = _quantities(s)
    total = 0.0
    counted = False

    if months_match:
        months = _quantities(months_match.group(1))
        if months:
            total += months[0][0] / 12.0
            counted = True

    # The YEARS figure is the first number that is not the month count, never
    # the number sitting against the word "years": in "3-5 years" the first
    # number is the lower bound the candidate must clear, and in "6 months to 2
    # years" skipping the month count is what stops 6 being read as 6 years.
    month_start = months_match.start(1) if months_match else -1
    for value, start, _ in numbers:
        if start == month_start:
            continue
        total += value
        counted = True
        break

    if counted:
        return total if 0 < total <= _MAX_PLAUSIBLE_YEARS else None

    for word, value in _WORD_NUM.items():
        if re.search(r"\b" + word + r"\b", s):
            return float(value)
    return None
