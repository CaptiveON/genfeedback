"""The gap engine's core promises: never fabricate a gap, never silently
drop a requirement it could not check."""

from genfeedback.matching import (
    COMPARABLE_CATEGORIES,
    find_missing,
    find_missing_notes,
)


def test_perfect_match_yields_zero_gaps():
    ents = {"technicalskill": ["Python", "PyTorch"], "tools": ["Docker"]}
    assert find_missing(ents, ents) == {}


def test_true_gaps_are_reported():
    jd = {"technicalskill": ["Python", "PyTorch"], "tools": ["Kubernetes"]}
    resume = {"technicalskill": ["Python"], "tools": ["Docker"]}
    missing = find_missing(jd, resume)
    assert missing == {"technicalskill": ["PyTorch"], "tools": ["Kubernetes"]}


def test_cross_category_evidence_counts():
    # The extractor may file the same skill under different categories on
    # each side; that must never fabricate a gap.
    jd = {"tools": ["PyTorch"]}
    resume = {"technicalskill": ["PyTorch"]}
    assert find_missing(jd, resume) == {}


def test_alias_evidence_counts():
    jd = {"tools": ["AWS"], "technicalskill": ["Python"]}
    resume = {"technicalskill": ["Amazon Web Services", "Python"]}
    assert find_missing(jd, resume) == {}


def test_years_compared_numerically():
    jd = {"yearofexperience": ["5 years"], "technicalskill": ["Python"]}
    short = {"yearofexperience": ["3 years"], "technicalskill": ["Python"]}
    more = {"yearofexperience": ["6 years"], "technicalskill": ["Python"]}
    assert "yearofexperience" in find_missing(jd, short)
    assert find_missing(jd, more) == {}


def test_designation_is_never_a_gap():
    jd = {"designation": ["Senior Engineer"], "technicalskill": ["Python"]}
    resume = {"technicalskill": ["Python"]}
    assert find_missing(jd, resume) == {}
    assert "designation" not in COMPARABLE_CATEGORIES


def test_generic_noise_never_reaches_a_candidate():
    jd = {"technicalskill": ["plus", "related field", "Python"]}
    resume = {"technicalskill": ["Python"]}
    assert find_missing(jd, resume) == {}


def test_unrecognised_requirements_surface_as_notes():
    # An unverifiable requirement must not vanish silently — the operator
    # is told what could not be checked.
    jd = {"technicalskill": ["Frobnicator", "Python"]}
    resume = {"technicalskill": ["Python"]}
    assert find_missing(jd, resume) == {}
    notes = find_missing_notes(jd, resume)
    assert any("Frobnicator" in n for n in notes)


def test_empty_inputs():
    assert find_missing({}, {}) == {}


def test_date_range_experience_abstains_with_a_note():
    # An employment period is not a duration: the comparison abstains
    # (no fabricated gap) and tells the operator why.
    jd = {"yearofexperience": ["5 years"], "technicalskill": ["Python"]}
    resume = {
        "yearofexperience": ["October 2018 to Present"],
        "technicalskill": ["Python"],
    }
    assert find_missing(jd, resume) == {}
    notes = find_missing_notes(jd, resume)
    assert any("October 2018 to Present" in n for n in notes)
