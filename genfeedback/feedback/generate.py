"""Generate a constructive rejection email from a skills-gap summary.

Two paths:
  * `generate_feedback(...)`  — uses the fine-tuned Gemma model (gated; deferred).
  * `compose_feedback(...)`   — a deterministic template that needs no model, so
    the full pipeline (extract -> gaps -> email) works in the UI today.
"""

import logging

import torch

from .. import config

logger = logging.getLogger(__name__)

# Human-readable names + email phrasing for each BIO category.
_CATEGORY_PHRASING = [
    ("technicalskill", "Technical skills"),
    ("tools", "Tools & platforms"),
    ("softskill", "Soft skills"),
    ("qualification", "Qualifications"),
    ("yearofexperience", "Experience level"),
    ("designation", "Role alignment"),
]


def _format_entities(entities):
    lines = ["Missing skills / qualifications:"]
    for category, items in entities.items():
        lines.append(f"{category.capitalize()}:")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def build_prompt(missing_entities, company=None, recruiter=None, role=None):
    """Construct the instruction prompt for the feedback model."""
    company = company or config.COMPANY_NAME
    recruiter = recruiter or config.RECRUITER_NAME
    role = role or config.ROLE_TITLE

    instruction = (
        f"Write a concise, respectful rejection email for an applicant to the "
        f"{role} role. Reference the skill gaps listed above as constructive, "
        f"specific areas for growth. Be encouraging and professional. "
        f"Company name is {company}; recruiter name is {recruiter}. "
        f"Output only the email body, ready to send."
    )
    return f"{_format_entities(missing_entities)}\n\n{instruction}"


def generate_feedback(
    missing_entities, model, tokenizer, device, max_new_tokens=None
):
    """Generate feedback text from the missing-skills summary."""
    max_new_tokens = max_new_tokens or config.GEMMA_MAX_NEW_TOKENS
    prompt = build_prompt(missing_entities)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Decode only the newly generated tokens, not the echoed prompt
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _format_entity_dict(entities):
    """Render a {category: [items]} dict as readable 'Label: a, b, c' lines."""
    lines = []
    for key, label in _CATEGORY_PHRASING:
        items = entities.get(key)
        if items:
            lines.append(f"- {label}: {', '.join(items)}")
    return "\n".join(lines) if lines else "- (none extracted)"


def build_comparison_prompt(
    jd_entities, resume_entities, company=None, recruiter=None, role=None
):
    """Prompt that gives Gemma BOTH entity sets and asks it to find the gaps.

    The model does the comparison itself (rather than us precomputing gaps),
    then writes the feedback email.
    """
    company = company or config.COMPANY_NAME
    recruiter = recruiter or config.RECRUITER_NAME
    role = role or config.ROLE_TITLE

    return (
        f"You are {recruiter}, a recruiter at {company}, reviewing a candidate "
        f"for the {role} role. Below are the skills, tools, qualifications and "
        f"experience extracted from the JOB DESCRIPTION and from the CANDIDATE'S "
        f"RESUME, grouped by category.\n\n"
        f"JOB DESCRIPTION requires:\n{_format_entity_dict(jd_entities)}\n\n"
        f"CANDIDATE'S RESUME offers:\n{_format_entity_dict(resume_entities)}\n\n"
        f"Compare the two and determine where the candidate falls short of the "
        f"role's requirements — the skills, tools, qualifications or experience "
        f"the job needs that the resume does not show. Then write a concise, "
        f"warm, professional feedback email to the candidate that thanks them for "
        f"applying, explains those specific gaps as constructive areas to grow, "
        f"and stays encouraging and respectful. Sign off as {recruiter}, "
        f"{company} Recruiting Team. Output only the email body, ready to send."
    )


def _run_chat(bundle, prompt, max_new_tokens):
    """Run a single user prompt through a base Gemma bundle's chat template."""
    if bundle.multimodal:
        processor = bundle.tokenizer
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(bundle.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = bundle.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return processor.batch_decode(
            out[:, input_len:], skip_special_tokens=True
        )[0].strip()

    tokenizer = bundle.tokenizer
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True: transformers 5.x returns a BatchEncoding by default, so
    # request it explicitly and unpack — mirroring the multimodal branch above.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(bundle.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = bundle.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def generate_feedback_from_entities(
    jd_entities, resume_entities, bundle, max_new_tokens=None,
    company=None, recruiter=None, role=None,
):
    """Hand Gemma both entity sets; it finds the gaps and writes the feedback.

    `bundle` is a `GemmaBundle` from `load_base_gemma()`.
    """
    max_new_tokens = max_new_tokens or config.GEMMA_MAX_NEW_TOKENS
    prompt = build_comparison_prompt(
        jd_entities, resume_entities, company, recruiter, role
    )
    return _run_chat(bundle, prompt, max_new_tokens)


def generate_feedback_llm(
    missing_entities, bundle, max_new_tokens=None,
    company=None, recruiter=None, role=None,
):
    """Generate a feedback email from a precomputed gap summary (legacy path)."""
    max_new_tokens = max_new_tokens or config.GEMMA_MAX_NEW_TOKENS
    prompt = build_prompt(missing_entities, company, recruiter, role)
    return _run_chat(bundle, prompt, max_new_tokens)


def compose_feedback(missing_entities, company=None, recruiter=None, role=None):
    """Build a constructive rejection email from gaps — no model required.

    Deterministic fallback so the feedback step works without the gated Gemma
    model. Lists each non-empty gap category as a concrete growth area.

    With no gaps the email stays neutral: an empty gap set can equally mean the
    extractor read nothing, so it must never be phrased as a favourable review
    of the candidate. Callers screen that case with `quality.extraction_problem`
    before reaching this function.
    """
    company = company or config.COMPANY_NAME
    recruiter = recruiter or config.RECRUITER_NAME
    role = role or config.ROLE_TITLE

    gap_lines = []
    for key, label in _CATEGORY_PHRASING:
        items = missing_entities.get(key)
        if items:
            gap_lines.append(f"  • {label}: {', '.join(items)}")

    para = [
        "Dear Applicant,",
        "",
        f"Thank you for taking the time to apply for the {role} position at "
        f"{company}. We truly appreciate your interest and the effort you put "
        "into your application.",
        "",
    ]

    if gap_lines:
        para.append(
            "After carefully reviewing your background against the requirements "
            "for this role, we identified a few areas that we'd encourage you to "
            "strengthen to be a closer match for similar positions:"
        )
        para.append("")
        para.extend(gap_lines)
        para.append("")
        para.append(
            "These are meant as constructive, specific suggestions rather than a "
            "reflection of your potential. Building depth in the areas above would "
            "make your profile notably more competitive."
        )
    else:
        para.append(
            "Comparing your application against the requirements listed for this "
            "role, we did not identify any specific gaps to raise with you. That "
            "is not an assessment of your candidacy or a decision on your "
            "application — it means only that this comparison found no particular "
            "area to point you to against the requirements as listed."
        )

    para.extend([
        "",
        "We encourage you to apply again in the future, and we wish you every "
        "success in your job search.",
        "",
        "Warm regards,",
        recruiter,
        f"{company} Recruiting Team",
    ])
    return "\n".join(para)
