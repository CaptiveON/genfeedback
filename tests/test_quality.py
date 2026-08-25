"""The extraction-confidence gate: a failed extraction must never be
indistinguishable from a favourable result."""

import logging

from genfeedback.quality import extraction_ok, extraction_problem

logging.disable(logging.WARNING)


def test_empty_extraction_refuses():
    assert extraction_problem({}, {}) is not None


def test_rich_extraction_passes():
    jd = {"technicalskill": ["Python", "Java"], "tools": ["Docker"]}
    resume = {"technicalskill": ["Python"], "tools": ["Git"]}
    assert extraction_problem(jd, resume) is None


def test_designation_only_jd_refuses():
    # Job titles are not comparable requirements; a JD that yields nothing
    # else must refuse rather than report a favourable "no gaps".
    jd = {"designation": ["Senior Dev", "Engineer"]}
    resume = {"technicalskill": ["Python"], "tools": ["Git"]}
    assert extraction_problem(jd, resume) is not None


def test_extraction_ok_counts_spans():
    assert not extraction_ok({})
    assert extraction_ok({"technicalskill": ["Python", "Docker"]})


def test_unreadable_source_text_refuses():
    # Garbage spans backed by non-Latin source text must refuse — a failed
    # read must never come back as a favourable comparison.
    jd = {"technicalskill": ["Python", "Java"], "tools": ["Docker"]}
    junk = {"tools": ["五 年 以 上", "目"], "designation": ["学 学 士"]}
    problem = extraction_problem(
        jd,
        junk,
        jd_text="Requires Python, Java and Docker.",
        resume_text="五年以上工作经验，目前在一家公司担任工程师。",
    )
    assert problem is not None
