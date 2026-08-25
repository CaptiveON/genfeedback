"""BIO label scheme for the resume NER task."""

LABEL_TO_ID = {
    "O": 0,
    "B-TECHNICALSKILL": 1, "I-TECHNICALSKILL": 2,
    "B-SOFTSKILL": 3, "I-SOFTSKILL": 4,
    "B-DESIGNATION": 5, "I-DESIGNATION": 6,
    "B-TOOLS": 7, "I-TOOLS": 8,
    "B-YEAROFEXPERIENCE": 9, "I-YEAROFEXPERIENCE": 10,
    "B-QUALIFICATION": 11, "I-QUALIFICATION": 12,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

NUM_LABELS = len(LABEL_TO_ID)


def label_names_sorted_by_id():
    """Return label names ordered by their integer ID (for reports)."""
    return [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)]
