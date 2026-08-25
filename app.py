"""GenFeedback Streamlit app.

Streamlit re-executes this entire script on *every* widget interaction, so an
analysis must outlive the run that produced it: `run_analysis` computes and
returns, `main` stores the result in `st.session_state`, and `render_results`
draws only from there. Without that split, ticking the review checkbox — or
clicking Download — would re-enter the script, find no button press, and wipe
the entities and email it was meant to release, forcing a full NER + Gemma
re-run.

The draft is an adverse-action email, so two gates stand between extraction and
a downloadable file: the pipeline refuses to write anything when the extraction
is too thin to trust (`extraction_problem`), and a human must confirm they read
the draft before the download appears.

Outliving its run is also what makes the stored analysis dangerous: the input
boxes can be refilled with a second candidate without pressing Analyze, and a
rerun would then redraw the first candidate's entities, gaps and rejection
email — download still live — beside the second candidate's resume. Every
analysis therefore carries a fingerprint of the exact text it was computed
from, and is withdrawn on sight the moment the boxes stop matching it.

Run with:
    streamlit run app.py
"""

import hashlib
import inspect
import logging

import streamlit as st

from genfeedback import config
from genfeedback.feedback.client import request_feedback
from genfeedback.feedback.generate import (
    compose_feedback,
    generate_feedback_from_entities,
)
from genfeedback.feedback.model_io import hf_token, load_base_gemma
from genfeedback.matching import find_missing, find_missing_notes
from genfeedback.ner.inference import extract_entities, post_process_entities
from genfeedback.ner.model_io import bert_artifacts_exist, load_bert
from genfeedback.preprocessing import prepare_for_ner
from genfeedback.quality import extraction_problem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPLATE_WRITER = "Template (instant, no model)"
GEMMA_WRITER = "Gemma LLM (Hugging Face)"

# session_state keys. RESULTS_KEY holds one whole analysis so any rerun redraws
# it instead of recomputing it; REVIEWED_KEY is the review checkbox's widget key
# so a fresh analysis can reset the tick before the widget is created.
RESULTS_KEY = "analysis"
REVIEWED_KEY = "draft_reviewed"

REVIEW_PROMPT = "I have reviewed this draft"

# Zero-width and soft-hyphen characters are not str.isspace(), so a text box
# holding only these passes `.strip()` and sends empty input down the pipeline.
# Written as escapes on purpose: as literals they would be invisible in source.
_INVISIBLE_CHARS = (
    "\u200b"  # zero-width space
    "\u200c"  # zero-width non-joiner
    "\u200d"  # zero-width joiner
    "\u2060"  # word joiner
    "\ufeff"  # zero-width no-break space / BOM
    "\u00ad"  # soft hyphen
    "\u180e"  # mongolian vowel separator
)
_INVISIBLE_TABLE = {ord(c): None for c in _INVISIBLE_CHARS}

# `extraction_problem` is gaining optional source-text parameters in a parallel
# change to quality.py, and handing a function an argument it does not accept
# is a hard crash on every analysis. Ask the signature once instead of guessing;
# this can collapse into a plain keyword call once the parameters are permanent.
_PROBLEM_PARAMS = inspect.signature(extraction_problem).parameters
_PROBLEM_TAKES_TEXT = "jd_text" in _PROBLEM_PARAMS and "resume_text" in _PROBLEM_PARAMS


@st.cache_resource(show_spinner="Loading NER model...")
def get_bert():
    return load_bert()


@st.cache_resource(show_spinner=False)
def get_base_gemma(model_id):
    return load_base_gemma(model_id)


def visible_text(text):
    """Return `text` with invisible characters and whitespace stripped.

    Used only to decide whether a box is empty. The model still receives the
    original string: zero-width characters *inside* real words are the
    tokenizer's to normalise, and deleting them here would glue words together.
    """
    return text.translate(_INVISIBLE_TABLE).strip()


def input_fingerprint(jd_text, resume_text):
    """Identify the exact pair of texts an analysis was computed from.

    Compared, never shown, so any digest would do; the lengths are folded in
    ahead of each text so that moving characters across the JD/resume boundary
    cannot collide with the original pair. Hashes the raw text rather than the
    cleaned-up `prepare_for_ner` form on purpose: an edit wrongly called
    significant costs one re-run, while an edit wrongly called harmless costs
    a candidate someone else's rejection email.
    """
    digest = hashlib.sha256()
    for text in (jd_text, resume_text):
        digest.update(f"{len(text)}\0".encode("utf-8"))
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def extraction_refusal(jd_entities, resume_entities, jd_text, resume_text):
    """The extraction gate's verdict, given the source texts if it wants them."""
    if _PROBLEM_TAKES_TEXT:
        return extraction_problem(
            jd_entities, resume_entities, jd_text=jd_text, resume_text=resume_text
        )
    return extraction_problem(jd_entities, resume_entities)


def feedback_filename(results):
    """Name the downloaded draft after the analysis that produced it.

    Every candidate's rejection email downloading as `feedback.txt` is how the
    wrong draft ends up sent: on a reviewer's disk the files are
    indistinguishable. The role plus the input fingerprint tie one file to one
    analysis of one pair of documents.
    """
    words = "".join(
        c if c.isalnum() else " " for c in results["settings"]["role"].lower()
    ).split()
    slug = "-".join(words) or "role"
    stamp = results["fingerprint"][:8]
    return f"feedback_{slug}_{stamp}.txt"


def write_feedback(
    jd_entities, resume_entities, missing, mode, role, company, recruiter
):
    """Draft the email; return (text, notices).

    Notices are returned rather than drawn so that a later rerun (checkbox,
    download) still shows why the template fallback was used — a message drawn
    inline here would vanish the moment the user touched anything.
    """
    notices = []
    if mode == GEMMA_WRITER:
        try:
            if config.GEMMA_API_URL:
                # Decoupled path: call the gemma-api microservice over HTTP.
                with st.spinner("Requesting feedback from the Gemma service..."):
                    feedback = request_feedback(
                        config.GEMMA_API_URL, jd_entities, resume_entities,
                        role=role, company=company, recruiter=recruiter,
                    )
            else:
                # In-process path: load and run the model locally (dev fallback).
                with st.spinner(
                    f"Loading {config.GEMMA_BASE_MODEL} "
                    "(first run downloads weights; this can take a while)..."
                ):
                    bundle = get_base_gemma(config.GEMMA_BASE_MODEL)
                with st.spinner("Gemma is comparing the entities and writing feedback..."):
                    # Gemma gets BOTH entity sets and finds the gaps itself.
                    feedback = generate_feedback_from_entities(
                        jd_entities, resume_entities, bundle,
                        company=company, recruiter=recruiter, role=role,
                    )
            return feedback, notices
        except Exception as exc:  # noqa: BLE001 - surface any load/gen error
            logger.exception("Gemma generation failed; falling back to template")
            notices.append(("error", f"Gemma generation failed: {exc}"))
            msg = str(exc).lower()
            if "does not recognize this architecture" in msg or "out of date" in msg:
                notices.append((
                    "warning",
                    "This usually means the app is running under the **system** "
                    "Python (old transformers), not the venv. Launch it with the "
                    "venv instead:\n\n"
                    "`./run_ui.sh`  or  "
                    "`.venv/bin/python -m streamlit run app.py`",
                ))
            notices.append(("info", "Falling back to the template writer."))

    feedback = compose_feedback(
        missing, company=company, recruiter=recruiter, role=role
    )
    return feedback, notices


def run_analysis(jd_input, resume_input, mode, role, company, recruiter):
    """Run the pipeline once and return everything the page needs to redraw.

    Draws spinners but no results: the caller stashes the returned dict in
    session_state and `render_results` paints it on this and every later rerun.
    """
    model, bert_tokenizer, device = get_bert()

    with st.spinner("Extracting entities..."):
        jd_clean = prepare_for_ner([jd_input])
        resume_clean = prepare_for_ner([resume_input])
        # extract_entities returns one dict per text; clean for display + matching.
        jd_entities = post_process_entities(
            extract_entities(jd_clean, model, bert_tokenizer, device)[0]
        )
        resume_entities = post_process_entities(
            extract_entities(resume_clean, model, bert_tokenizer, device)[0]
        )

    results = {
        "jd_entities": jd_entities,
        "resume_entities": resume_entities,
        "problem": None,
        "missing": None,
        "notes": [],
        "feedback": None,
        "notices": [],
        "settings": {
            "role": role, "company": company,
            "recruiter": recruiter, "mode": mode,
        },
        # Pins this result to the text it came from; render_results refuses to
        # draw it once the boxes hold anything else.
        "fingerprint": input_fingerprint(jd_input, resume_input),
    }

    # Extraction gate: if the entity sets are too thin to reason about, no gap
    # is meaningful, so refuse the whole downstream step rather than draft a
    # rejection email out of noise. extraction_problem() logs the detail.
    problem = extraction_refusal(
        jd_entities, resume_entities, jd_input, resume_input
    )
    if problem:
        results["problem"] = problem
        return results

    # Rule-based gap view (deterministic) — shown for reference. The Gemma
    # writer does its OWN gap reasoning from both entity sets below.
    results["missing"] = find_missing(jd_entities, resume_entities)
    # find_missing abstains from a comparison it cannot make (experience stated
    # as employment dates, say) by returning no gap for it, which is
    # indistinguishable on screen from a requirement the candidate meets. The
    # notes are the only record that a JD requirement went unchecked, so they
    # are stored beside the gaps — never merged into them, as they must not
    # reach the candidate through the email.
    results["notes"] = find_missing_notes(jd_entities, resume_entities)
    results["feedback"], results["notices"] = write_feedback(
        jd_entities, resume_entities, results["missing"],
        mode, role, company, recruiter,
    )
    return results


def render_notices(notices):
    for level, message in notices:
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)


def render_results(results, jd_input, resume_input, mode, role, company, recruiter):
    """Draw a stored analysis. Runs on every rerun and computes nothing."""
    # Streamlit reruns on blur, so the boxes can hold a second candidate while
    # session_state still holds the first candidate's analysis. Redrawing it
    # would put one person's entities, gaps and rejection email beside another
    # person's resume with the download live, and nothing on screen would say
    # the analysed text had changed. Withdraw the whole result instead: every
    # part of it describes text that is no longer in the boxes.
    if results["fingerprint"] != input_fingerprint(jd_input, resume_input):
        # Assigning the checkbox's key is only legal while the widget does not
        # exist on this run — this branch returns before creating it.
        st.session_state[REVIEWED_KEY] = False
        st.error(
            "**The input changed — these results were withdrawn.**\n\n"
            "The job description or resume above is no longer the text this "
            "analysis was run on, so its entities, gaps and draft email "
            "described different input and are not shown. Press "
            "**Analyze & generate feedback** to analyse what is in the boxes "
            "now."
        )
        st.caption(
            "The download has been withdrawn and the review confirmation "
            "cleared, so no draft written for earlier text can be saved "
            "against this one."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Job Description entities")
        st.json(results["jd_entities"])
    with col2:
        st.write("### Resume entities")
        st.json(results["resume_entities"])

    # The entities above stay on screen for debugging, but everything derived
    # from them is withheld: no gap list, no email, no download.
    if results["problem"]:
        st.error(
            "**No feedback was generated — the extraction cannot be trusted.**"
            f"\n\n{results['problem']}"
        )
        st.caption(
            "The entities above are shown only so you can see what the model "
            "did find. They are not a result: no gap list, email, or download "
            "is produced from an extraction this thin."
        )
        return

    st.write("### Skill gaps (rule-based: in JD, missing from resume)")
    if results["missing"]:
        st.json(results["missing"])
    elif results["notes"]:
        # "No gaps" would claim the whole JD was checked, which is exactly what
        # the notes below say did not happen.
        st.info(
            "No skill gaps were found among the requirements that could be "
            "compared — see what could not be compared below."
        )
    else:
        st.success("No skill gaps detected against the job description.")

    # Abstentions are invisible in the gap list: a requirement the tool could
    # not check looks the same as one the candidate meets. These say which.
    for note in results["notes"]:
        st.warning(note)

    # --- Feedback -----------------------------------------------------------
    st.write("### Candidate feedback")

    # Results outlive the run that made them, so the sidebar can now disagree
    # with the draft on screen. Say so instead of showing an email signed by
    # the wrong recruiter as if it were current.
    current = {"role": role, "company": company, "recruiter": recruiter, "mode": mode}
    if results["settings"] != current:
        st.info(
            "This draft was written with the earlier settings "
            f"(role: {results['settings']['role']}, "
            f"company: {results['settings']['company']}, "
            f"recruiter: {results['settings']['recruiter']}, "
            f"writer: {results['settings']['mode']}). "
            "Run the analysis again to apply the current ones."
        )

    render_notices(results["notices"])

    st.warning(
        "**Unreviewed AI-generated draft — do not send as-is.**\n\n"
        "A person must read this email before any of it reaches a candidate. "
        "The skills, tools and qualifications it argues from were extracted "
        "automatically and can be wrong, incomplete, or misattributed, so the "
        "gaps it names may not be real. Correct anything inaccurate or unfair, "
        "and make sure the underlying decision was made by a human."
    )
    st.text(results["feedback"])

    # The checkbox gates the download: until it is ticked, no download widget
    # exists to click. Ticking it only reruns the script, which redraws this
    # same stored draft.
    reviewed = st.checkbox(REVIEW_PROMPT, key=REVIEWED_KEY)
    if reviewed:
        st.download_button(
            "Download feedback",
            results["feedback"],
            file_name=feedback_filename(results),
        )
    else:
        st.caption(
            f"Tick **{REVIEW_PROMPT}** to enable the download."
        )


def main():
    st.title("GenFeedback")
    st.caption(
        "Extracts skills/qualifications from a job description and a resume, "
        "finds the gaps, and drafts constructive candidate feedback."
    )

    # NER is the core; Gemma (the LLM feedback writer) is optional.
    if not bert_artifacts_exist():
        st.error(
            f"NER model not found at {config.BERT_DIR}. "
            "Train it first: `python -m genfeedback.train_bert`"
        )
        return

    # --- Sidebar: feedback email settings -----------------------------------
    st.sidebar.header("Feedback settings")
    role = st.sidebar.text_input("Role title", value=config.ROLE_TITLE)
    company = st.sidebar.text_input("Company", value=config.COMPANY_NAME)
    recruiter = st.sidebar.text_input("Recruiter", value=config.RECRUITER_NAME)

    mode = st.sidebar.radio("Feedback writer", [TEMPLATE_WRITER, GEMMA_WRITER])
    if mode == GEMMA_WRITER:
        st.sidebar.caption(f"Model: `{config.GEMMA_BASE_MODEL}`")
        if hf_token():
            st.sidebar.success("Hugging Face token detected.")
        else:
            st.sidebar.warning(
                "No HF token found. Gemma is gated — set `HF_TOKEN` and accept "
                "the model license, or use the template writer."
            )

    # --- Inputs -------------------------------------------------------------
    st.header("Input data")
    jd_input = st.text_area("Job Description", height=200, key="jd_input")
    resume_input = st.text_area("Candidate Resume", height=200, key="resume_input")

    if st.button("Analyze & generate feedback"):
        # Drop the old analysis before starting: if this one fails or is
        # rejected, the page must not keep showing the previous email as if it
        # belonged to the text now in the boxes.
        st.session_state.pop(RESULTS_KEY, None)
        # A new draft has not been reviewed. Assigning the checkbox's key is
        # only legal before the widget is created — render_results() creates it
        # further down this same run.
        st.session_state[REVIEWED_KEY] = False

        if not visible_text(jd_input) or not visible_text(resume_input):
            st.warning("Please provide both a job description and a resume.")
        else:
            st.session_state[RESULTS_KEY] = run_analysis(
                jd_input, resume_input, mode, role, company, recruiter
            )

    # Every rerun renders from session_state, so widget clicks never recompute.
    results = st.session_state.get(RESULTS_KEY)
    if results is None:
        return
    render_results(
        results, jd_input, resume_input, mode, role, company, recruiter
    )


if __name__ == "__main__":
    main()
