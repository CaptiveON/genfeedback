"""Canonicalisation and years parsing — the difference between a fair
comparison and telling a six-year engineer to go get five years."""

from genfeedback.normalize import canonical, parse_years


def test_alias_expansion_both_directions():
    assert canonical("AWS") == canonical("Amazon Web Services")
    assert canonical("ML") == canonical("machine learning")


def test_case_insensitive():
    assert canonical("PyTorch") == canonical("pytorch")


def test_parse_plain_years():
    assert parse_years("5 years") == 5.0
    assert parse_years("five years") == 5.0


def test_parse_ranges_take_lower_bound():
    assert parse_years("3-5 years") == 3.0


def test_parse_years_and_months():
    assert parse_years("2 years 6 months") == 2.5


def test_employment_date_ranges_abstain():
    # "October 2018 to Present" is an employment period, not a duration.
    assert parse_years("October 2018 to Present") is None


def test_ordinals_are_not_experience():
    assert parse_years("10th grade") is None


def test_nonsense_returns_none():
    assert parse_years("-3 years") is None
    assert parse_years("") is None
