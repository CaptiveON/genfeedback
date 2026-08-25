"""Test the trained NER model on a job description and a resume.

This is a Gemma-free way to inspect what the BERT NER model extracts and what
skill gaps the matcher finds. Useful for evaluating extraction quality before
the feedback-generation step.

Three silent-failure modes are closed here, because all three produce output
that looks like good news:

* An empty extraction makes `find_missing` return `{}`, which used to print
  "None — resume covers everything the JD asks for." The gate in
  :mod:`genfeedback.quality` is consulted first, and the command refuses with a
  non-zero exit instead of congratulating the candidate on an unread resume.
* `--jd ""` used to fall through to the built-in DEMO_JD, and because the
  "running on the demo pair" notice was computed by the same truthiness test,
  the notice was suppressed too — a real resume was scored against a
  fabricated job description with nothing on screen to say so.
* A comparison the matcher had to abstain from (a resume that states employment
  dates instead of a duration) dropped the JD's requirement out of the gap list
  entirely, which reads as "met". `find_missing_notes` says so out loud, in the
  report and in the JSON payload.
* A refused analysis used to publish its gap list anyway through `--json`: the
  report withheld it, while the payload documented "for piping into other
  tools" carried the same fabricated list of things the candidate supposedly
  lacks. The gate now runs inside `analyze`, so on a refusal there is no gap
  list in the result for ANY output format to hand on.

Usage:
    # Run on the built-in demo JD/resume:
    python -m genfeedback.extract

    # Provide text inline:
    python -m genfeedback.extract --jd "Senior Python developer ..." \\
                                  --resume "Java engineer with 3 years ..."

    # Provide files:
    python -m genfeedback.extract --jd-file jd.txt --resume-file resume.txt

    # JSON output (for piping into other tools):
    python -m genfeedback.extract --json
"""

import argparse
import json
import logging
import sys

from .matching import find_missing, find_missing_notes
from .ner.inference import extract_entities, post_process_entities
from .ner.model_io import bert_artifacts_exist, load_bert
from .preprocessing import prepare_for_ner
from .quality import extraction_problem

logger = logging.getLogger(__name__)

# Pretty display order + human-readable names for the BIO categories.
CATEGORY_LABELS = [
    ("technicalskill", "Technical Skills"),
    ("tools", "Tools"),
    ("softskill", "Soft Skills"),
    ("designation", "Designation / Role"),
    ("yearofexperience", "Years of Experience"),
    ("qualification", "Qualification"),
]

# A realistic demo pair used when no JD/resume is supplied.
DEMO_JD = (
    "We are hiring a Senior Machine Learning Engineer with 5 years of experience. "
    "The ideal candidate is proficient in Python and PyTorch, has experience with "
    "TensorFlow and scikit-learn, and is skilled in natural language processing and "
    "computer vision. Strong communication and leadership skills are required. "
    "Experience with Docker, Kubernetes, and AWS is a plus. A Master of Science in "
    "Computer Science is preferred."
)
DEMO_RESUME = (
    "Experienced Data Scientist with 4 years of experience. Proficient in Python and "
    "scikit-learn, with strong skills in machine learning and statistical modeling. "
    "Used Git and Jenkins for version control and CI/CD. Excellent communication "
    "skills and a Bachelor of Science in Computer Science."
)


class SourceError(Exception):
    """A JD/resume source was supplied but could not be turned into text."""


def analyze(jd_text, resume_text, model, tokenizer, device):
    """Extract entities from a JD and resume and compute skill gaps.

    Returns a dict with cleaned `jd_entities`, `resume_entities`, `problem`
    (the extraction gate's refusal reason, or None), `missing` (JD items absent
    from the resume, per category) and `notes` (comparisons the matcher had to
    abstain from).

    The gate runs HERE, ahead of the matching, so that a refused analysis never
    holds a gap list at all. Computing the gaps first and withholding them at
    print time is what let `--json` publish the very list the report refuses to
    print: one output format remembered to withhold it, the other did not, and
    every format added later would have had to remember again.

    `missing` and `notes` are None on a refusal — not `{}` and not `[]`,
    because an empty gap dict is exactly what "the resume covers everything"
    looks like, and that mistake is the reason this module has a gate. None is
    not iterable, so a consumer that ignores `problem` fails loudly instead of
    quietly reporting a clean bill of health.

    `notes` is kept out of `missing` and stays out of any candidate-facing
    text: it is an operator diagnostic about what could NOT be checked.
    """
    jd_clean = prepare_for_ner([jd_text])
    resume_clean = prepare_for_ner([resume_text])

    jd_entities = post_process_entities(
        extract_entities(jd_clean, model, tokenizer, device)[0]
    )
    resume_entities = post_process_entities(
        extract_entities(resume_clean, model, tokenizer, device)[0]
    )

    # The source texts go with the entities — a document the model could not
    # read (another script, PDF debris) is only visible in the text, not in the
    # spans it produced from it.
    problem = extraction_problem(
        jd_entities, resume_entities, jd_text=jd_text, resume_text=resume_text
    )
    result = {
        "jd_entities": jd_entities,
        "resume_entities": resume_entities,
        "problem": problem,
        "missing": None,
        "notes": None,
    }
    if problem:
        return result

    result["missing"] = find_missing(jd_entities, resume_entities)
    result["notes"] = find_missing_notes(jd_entities, resume_entities)
    return result


def _format_block(title, entities):
    lines = [f"=== {title} ==="]
    any_found = False
    for key, label in CATEGORY_LABELS:
        spans = entities.get(key)
        if spans:
            any_found = True
            lines.append(f"  {label}:")
            for span in spans:
                lines.append(f"    - {span}")
    if not any_found:
        lines.append("  (no entities extracted)")
    return "\n".join(lines)


def _format_missing(missing, notes):
    """Render the gap list; `notes` only qualifies the empty case.

    An empty gap list means "covers everything" only when every requirement was
    actually checked. With an abstention outstanding it means "nothing found
    among the ones we could check", which is a different claim.
    """
    lines = ["=== Skill Gaps (in JD, missing from Resume) ==="]
    if not missing:
        if notes:
            lines.append(
                "  None found — but not every requirement could be checked; "
                "see the notes below."
            )
        else:
            lines.append("  None — resume covers everything the JD asks for.")
        return "\n".join(lines)
    for key, label in CATEGORY_LABELS:
        gaps = missing.get(key)
        if gaps:
            lines.append(f"  {label}:")
            for gap in gaps:
                lines.append(f"    - {gap}")
    return "\n".join(lines)


def _format_notes(notes):
    """Render the matcher's abstentions, or None when there were none.

    Printed next to the gap list because an abstention is invisible there: a
    requirement the matcher could not check is simply absent from the gaps,
    which is exactly how a requirement the candidate meets looks.
    """
    if not notes:
        return None
    lines = ["=== Notes — comparisons this tool could NOT make ==="]
    for note in notes:
        lines.append(f"  - {note}")
    lines.append(
        "  (operator diagnostics — these are not findings about the candidate "
        "and must not be sent to them.)"
    )
    return "\n".join(lines)


def _read_text_file(path, what):
    """Read a UTF-8 text file, turning the usual OS failures into SourceError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise SourceError(f"no such file for {what}: {path}") from None
    except IsADirectoryError:
        raise SourceError(
            f"{what} points at a directory, not a text file: {path}"
        ) from None
    except PermissionError:
        raise SourceError(f"permission denied reading {what}: {path}") from None
    except UnicodeDecodeError:
        raise SourceError(
            f"{what} is not UTF-8 text: {path} — a PDF, DOCX or other binary "
            f"will do this. Convert it to plain text first."
        ) from None
    except OSError as exc:
        raise SourceError(f"could not read {what} ({path}): {exc}") from None


def _read_source(inline, file_path, default, what="the input"):
    """Resolve text from an inline string, a file path, or the default.

    argparse hands back None for a flag that was never given and "" for a flag
    given as empty, so absence is tested with `is None`, never truthiness: the
    old truthiness test let `--jd ""` fall through to the demo JD.

    Raises:
        SourceError: the source was supplied but yields no usable text.
    """
    if inline is not None:
        if not inline.strip():
            raise SourceError(f"{what} was supplied but is empty.")
        return inline
    if file_path is not None:
        text = _read_text_file(file_path, what)
        if not text.strip():
            raise SourceError(f"{what} file is empty: {file_path}")
        return text
    return default


def main():
    parser = argparse.ArgumentParser(
        description="Extract NER entities from a JD and resume and show skill gaps."
    )
    jd_source = parser.add_mutually_exclusive_group()
    jd_source.add_argument("--jd", help="Job description text (inline).")
    jd_source.add_argument("--jd-file", help="Path to a job description text file.")
    resume_source = parser.add_mutually_exclusive_group()
    resume_source.add_argument("--resume", help="Resume text (inline).")
    resume_source.add_argument("--resume-file", help="Path to a resume text file.")
    parser.add_argument(
        "--json", action="store_true", help="Emit raw JSON instead of a report."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )

    if not bert_artifacts_exist():
        from . import config

        parser.error(
            f"No trained BERT model at {config.BERT_DIR}. "
            "Train it first: python -m genfeedback.train_bert"
        )

    try:
        jd_text = _read_source(
            args.jd, args.jd_file, DEMO_JD, "the job description (--jd/--jd-file)"
        )
        resume_text = _read_source(
            args.resume, args.resume_file, DEMO_RESUME,
            "the resume (--resume/--resume-file)",
        )
    except SourceError as exc:
        parser.error(str(exc))

    # Track each side separately: supplying only a resume still scores it
    # against the invented demo JD, which must be announced just as loudly.
    used_demo_jd = args.jd is None and args.jd_file is None
    used_demo_resume = args.resume is None and args.resume_file is None

    model, tokenizer, device = load_bert()
    # analyze() consults the extraction gate before it matches anything, so a
    # refused result carries no gap list for either branch below to publish.
    result = analyze(jd_text, resume_text, model, tokenizer, device)
    problem = result["problem"]

    if args.json:
        payload = dict(result)
        payload["used_demo_jd"] = used_demo_jd
        payload["used_demo_resume"] = used_demo_resume
        print(json.dumps(payload, indent=2))
        return 1 if problem else 0

    if used_demo_jd and used_demo_resume:
        print("(no JD/resume provided — running on the built-in demo pair)\n")
    elif used_demo_jd:
        print("(no JD provided — scoring against the built-in demo job description)\n")
    elif used_demo_resume:
        print("(no resume provided — using the built-in demo resume)\n")

    print(_format_block("Entities from Job Description", result["jd_entities"]))
    print()
    print(_format_block("Entities from Resume", result["resume_entities"]))
    print()

    if problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        print(
            "Refusing to report skill gaps: a gap list computed from this "
            "extraction describes the failure above, not the candidate.",
            file=sys.stderr,
        )
        return 1

    print(_format_missing(result["missing"], result["notes"]))
    notes_block = _format_notes(result["notes"])
    if notes_block:
        print()
        print(notes_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
