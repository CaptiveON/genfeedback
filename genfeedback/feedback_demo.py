"""End-to-end feedback test from the terminal (NER extract -> Gemma feedback).

Runs the full pipeline: extract entities from a JD and a resume with the trained
NER model, then hand BOTH entity sets to the base Gemma model and let it find the
gaps and write the candidate feedback. No fine-tuning involved.

This is the second command that emits candidate-facing text, so it carries the
same guards as :mod:`genfeedback.extract` rather than a looser version of them:

* The `quality.extraction_problem` gate runs BEFORE the Gemma load. An
  unreadable resume (empty, a PDF pasted as gibberish, a language the model
  never saw) extracts to nothing, and Gemma asked to compare nothing against
  nothing still writes a confident, fluent rejection email. Refusing early also
  means the failing case never pays for a multi-GB model load.
* `--jd ""` is a user error, not a request for the built-in demo JD, and the
  demo pair is announced whenever it is used — the same real-resume-against-a-
  fabricated-JD trap `extract` closes.
* `find_missing_notes` caveats are printed for the operator only, outside the
  email block, and are never handed to the model: they describe what this tool
  could not check, and a parser's limitation must not reach a candidate.

Usage:
    python -m genfeedback.feedback_demo                       # built-in demo pair
    python -m genfeedback.feedback_demo --jd "..." --resume "..."
    python -m genfeedback.feedback_demo --jd-file jd.txt --resume-file resume.txt

Requires HF auth (Gemma is gated). The model id comes from config.GEMMA_BASE_MODEL
(override with GENFEEDBACK_GEMMA_BASE, e.g. google/gemma-2-2b-it for a lighter run).
"""

import argparse
import logging
import sys

from . import config
from .extract import (
    DEMO_JD,
    DEMO_RESUME,
    SourceError,
    _format_block,
    _format_notes,
    _read_source,
    analyze,
)
from .feedback.generate import generate_feedback_from_entities
from .feedback.model_io import hf_token, load_base_gemma
from .ner.model_io import bert_artifacts_exist, load_bert
from .quality import extraction_problem

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Gemma feedback end-to-end.")
    # Grouped so that --jd with --jd-file is refused rather than silently
    # resolved in favour of one of them.
    jd_source = parser.add_mutually_exclusive_group()
    jd_source.add_argument("--jd", help="Job description text (inline).")
    jd_source.add_argument("--jd-file", help="Path to a job description file.")
    resume_source = parser.add_mutually_exclusive_group()
    resume_source.add_argument("--resume", help="Resume text (inline).")
    resume_source.add_argument("--resume-file", help="Path to a resume file.")
    parser.add_argument("--role", default=config.ROLE_TITLE)
    parser.add_argument("--company", default=config.COMPANY_NAME)
    parser.add_argument("--recruiter", default=config.RECRUITER_NAME)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not bert_artifacts_exist():
        parser.error(
            f"No trained NER model at {config.BERT_DIR}. "
            "Train it first: python -m genfeedback.train_bert"
        )
    if not hf_token():
        logger.warning(
            "No HF token detected — Gemma is gated and the load will likely fail. "
            "Run `hf auth login` or set HF_TOKEN first."
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

    # Each side is tracked separately: supplying only a resume still scores it
    # against the invented demo JD, which must be announced just as loudly.
    used_demo_jd = args.jd is None and args.jd_file is None
    used_demo_resume = args.resume is None and args.resume_file is None
    if used_demo_jd and used_demo_resume:
        print("(no JD/resume provided — running on the built-in demo pair)\n")
    elif used_demo_jd:
        print("(no JD provided — scoring against the built-in demo job description)\n")
    elif used_demo_resume:
        print("(no resume provided — using the built-in demo resume)\n")

    # 1) NER extraction
    print("Loading NER model and extracting entities...")
    bert_model, bert_tokenizer, device = load_bert()
    result = analyze(jd_text, resume_text, bert_model, bert_tokenizer, device)
    print()
    print(_format_block("Entities from Job Description", result["jd_entities"]))
    print()
    print(_format_block("Entities from Resume", result["resume_entities"]))

    # 2) Gate, before Gemma is loaded: with nothing extracted there is nothing
    # to compare, and the model would write a fluent rejection email anyway.
    # The source texts go with the entities — a document the model could not
    # read is only visible in the text, not in the spans it produced from it.
    problem = extraction_problem(
        result["jd_entities"], result["resume_entities"],
        jd_text=jd_text, resume_text=resume_text,
    )
    if problem:
        print(f"\nERROR: {problem}", file=sys.stderr)
        print(
            "Refusing to generate candidate feedback: an email written from "
            "this extraction would describe the failure above, not the "
            "candidate.",
            file=sys.stderr,
        )
        return 1

    # Operator diagnostics about comparisons the deterministic matcher could
    # not make. Gemma reasons about the gaps itself here, so these name exactly
    # the requirements whose treatment in the email below nothing has checked.
    # Printed outside the email block and deliberately NOT passed to the model.
    notes_block = _format_notes(result["notes"])
    if notes_block:
        print()
        print(notes_block)

    # 3) Gemma feedback (model does its own gap reasoning)
    print(f"\nLoading {config.GEMMA_BASE_MODEL} (first run downloads weights)...")
    bundle = load_base_gemma(config.GEMMA_BASE_MODEL)
    print("Generating feedback...\n")
    feedback = generate_feedback_from_entities(
        result["jd_entities"], result["resume_entities"], bundle,
        company=args.company, recruiter=args.recruiter, role=args.role,
    )

    print("=" * 60)
    print("CANDIDATE FEEDBACK")
    print("=" * 60)
    print(feedback)
    return 0


if __name__ == "__main__":
    sys.exit(main())
