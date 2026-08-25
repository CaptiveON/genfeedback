"""Generate a synthetic, realistically-structured CoNLL NER dataset.

Why this exists
---------------
Hand-written fragment samples (v1) were too short and inconsistent: the model
overfit to "the first few tokens are an entity" and tagged connective words
("and", "with", "the") as entities on real prose. This generator builds
sentence- and paragraph-level samples that match the inference distribution
(flowing JD / resume text) and labels every token programmatically, so filler
words are *always* O and multi-word entities get correct B-/I- spans.

Output: a CoNLL file (one `token TAG` per line, blank line between samples) at
`config.DATA_PATH` by default.

Usage:
    python -m genfeedback.datagen                 # 1200 samples, seed 42
    python -m genfeedback.datagen --n 2000 --seed 7 --out data/resume.txt
"""

import argparse
import random

from . import config
from .tokenization import tokenize_words

# --- Entity vocabularies ----------------------------------------------------
# Multi-word values are fine; they are whitespace-split into B-/I- spans.

TECHNICALSKILL = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Ruby",
    "Scala", "R", "MATLAB", "SQL", "NoSQL", "HTML", "CSS", "Flask", "Django",
    "FastAPI", "Spring Boot", "React", "Angular", "Vue.js", "Node.js",
    "Express.js", "MongoDB", "PostgreSQL", "MySQL", "Redis", "TensorFlow",
    "PyTorch", "Keras", "scikit-learn", "pandas", "NumPy", "XGBoost", "OpenCV",
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data analysis", "statistical modeling",
    "reinforcement learning", "time series analysis", "feature engineering",
    "REST APIs", "GraphQL", "microservices", "data structures", "algorithms",
    "object-oriented programming", "test-driven development", "Apache Spark",
    "Apache Kafka", "Apache Hadoop", "ETL pipelines", "CUDA", "Linux", "Bash",
]

TOOLS = [
    "Git", "GitHub", "GitLab", "Bitbucket", "Docker", "Kubernetes", "Jenkins",
    "Terraform", "Ansible", "Prometheus", "Grafana", "Tableau", "Power BI",
    "JIRA", "Confluence", "AWS", "GCP", "Azure", "Postman", "Figma",
    "Selenium", "Nginx", "RabbitMQ", "Datadog", "Airflow", "CircleCI",
    "Heroku", "Vercel", "Elasticsearch", "VS Code", "IntelliJ IDEA",
]

SOFTSKILL = [
    "communication", "leadership", "teamwork", "problem-solving",
    "critical thinking", "time management", "adaptability", "collaboration",
    "creativity", "mentorship", "presentation", "public speaking",
    "negotiation", "emotional intelligence", "attention to detail",
    "interpersonal", "organizational", "analytical thinking",
]

DESIGNATION = [
    "Software Engineer", "Senior Software Engineer", "Data Scientist",
    "Machine Learning Engineer", "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "DevOps Engineer", "Data Engineer", "Data Analyst",
    "Cloud Architect", "Tech Lead", "Engineering Manager", "Product Manager",
    "Site Reliability Engineer", "QA Engineer", "Research Scientist",
    "AI Engineer", "Security Engineer", "Database Administrator",
    "Systems Administrator", "Principal Engineer", "Junior Developer",
]

# Qualifications are generated as {degree} in {field} plus standalone degrees,
# so many real surface forms are covered (e.g. "Masters in Computer Science",
# "PhD in Statistics", "B.Tech in Information Technology"). Apostrophes and
# trailing-dot abbreviations are avoided so the word tokenizer round-trips them.
_DEGREES = [
    "Bachelor of Science", "Master of Science", "Bachelor of Arts",
    "Master of Arts", "Bachelor of Engineering", "Master of Engineering",
    "Bachelors", "Masters", "Bachelors degree", "Masters degree",
    "B.Tech", "M.Tech", "PhD", "Doctorate", "Associate degree",
]
_FIELDS = [
    "Computer Science", "Data Science", "Information Technology",
    "Artificial Intelligence", "Machine Learning", "Software Engineering",
    "Electrical Engineering", "Mathematics", "Statistics", "Physics",
]
_STANDALONE_DEGREES = [
    "MBA", "PhD", "Bachelors degree", "Masters degree", "Associate degree",
]

QUALIFICATION = (
    [f"{deg} in {field}" for deg in _DEGREES for field in _FIELDS]
    + _STANDALONE_DEGREES
)

YEAROFEXPERIENCE = [
    "1 year", "2 years", "3 years", "4 years", "5 years", "6 years",
    "7 years", "8 years", "10 years", "12 years", "3+ years", "5+ years",
    "6+ years", "2+ years",
]

VOCAB = {
    "TECHNICALSKILL": TECHNICALSKILL,
    "TOOLS": TOOLS,
    "SOFTSKILL": SOFTSKILL,
    "DESIGNATION": DESIGNATION,
    "YEAROFEXPERIENCE": YEAROFEXPERIENCE,
    "QUALIFICATION": QUALIFICATION,
}

# --- Sentence templates -----------------------------------------------------
# A template is a list of segments. A str segment is literal filler (all O).
# A ("LIST", "TYPE", n) segment expands to n entities of TYPE joined by commas
# and a trailing "and". A ("SLOT", "TYPE") segment is a single entity.

TEMPLATES = [
    ["Proficient in ", ("LIST", "TECHNICALSKILL", 2), "."],
    ["Skilled in ", ("LIST", "TECHNICALSKILL", 3), "."],
    ["Strong experience with ", ("LIST", "TECHNICALSKILL", 2), "and ",
     ("LIST", "TOOLS", 2), "."],
    ["Experienced ", ("SLOT", "DESIGNATION"), " with ",
     ("SLOT", "YEAROFEXPERIENCE"), " of experience."],
    ["Worked as a ", ("SLOT", "DESIGNATION"), " for ",
     ("SLOT", "YEAROFEXPERIENCE"), "."],
    ["We are hiring a ", ("SLOT", "DESIGNATION"), " proficient in ",
     ("LIST", "TECHNICALSKILL", 2), "."],
    ["The ideal candidate has ", ("SLOT", "YEAROFEXPERIENCE"),
     " of experience in ", ("LIST", "TECHNICALSKILL", 2), "."],
    ["Looking for a ", ("SLOT", "DESIGNATION"), " with strong ",
     ("LIST", "SOFTSKILL", 2), "skills."],
    ["Excellent ", ("LIST", "SOFTSKILL", 2), "skills."],
    ["Demonstrated ", ("LIST", "SOFTSKILL", 2), "in fast-paced environments."],
    ["Holds a ", ("SLOT", "QUALIFICATION"), "."],
    ["Completed a ", ("SLOT", "QUALIFICATION"), " with strong ",
     ("SLOT", "SOFTSKILL"), " skills."],
    ["Requires a ", ("SLOT", "QUALIFICATION"), " and ",
     ("SLOT", "YEAROFEXPERIENCE"), " of relevant experience."],
    ["Used ", ("LIST", "TOOLS", 2), "for deployment and CI/CD."],
    ["Experience with ", ("LIST", "TOOLS", 3), "is required."],
    ["Deployed applications on ", ("LIST", "TOOLS", 2), "."],
    ["Built scalable systems using ", ("LIST", "TECHNICALSKILL", 2), "and ",
     ("SLOT", "TOOLS"), "."],
    [("SLOT", "DESIGNATION"), " skilled in ", ("LIST", "TECHNICALSKILL", 2),
     "with ", ("SLOT", "YEAROFEXPERIENCE"), " of experience."],
    [("SLOT", "DESIGNATION"), " with a ", ("SLOT", "QUALIFICATION"),
     " and expertise in ", ("LIST", "TECHNICALSKILL", 2), "."],
    ["Seeking a ", ("SLOT", "DESIGNATION"), " with ",
     ("SLOT", "YEAROFEXPERIENCE"), " of experience in ",
     ("LIST", "TECHNICALSKILL", 2), "and ", ("SLOT", "TOOLS"), "."],
    ["Strong background in ", ("LIST", "TECHNICALSKILL", 2),
     "and excellent ", ("SLOT", "SOFTSKILL"), " skills."],
    ["Proficient with ", ("LIST", "TOOLS", 2), "and ",
     ("LIST", "TECHNICALSKILL", 2), "."],
    ["Candidate should have ", ("SLOT", "YEAROFEXPERIENCE"),
     " of hands-on experience with ", ("LIST", "TECHNICALSKILL", 2), "."],
    ["Responsible for building ", ("LIST", "TECHNICALSKILL", 2),
     "using ", ("LIST", "TOOLS", 2), "."],
    ["A ", ("SLOT", "QUALIFICATION"), " is preferred for this role."],
    ["Hands-on experience deploying ", ("SLOT", "TECHNICALSKILL"),
     " models with ", ("LIST", "TOOLS", 2), "."],
]


def _tag_value(value, etype):
    """Return list of (token, tag) for an entity value, tokenized like inference."""
    toks = tokenize_words(value)
    return [(toks[0], f"B-{etype}")] + [(t, f"I-{etype}") for t in toks[1:]]


def _tag_literal(text):
    """Return list of (token, 'O') for filler text, tokenized like inference."""
    return [(t, "O") for t in tokenize_words(text)]


def _expand_list(rng, etype, n):
    """Expand n entities into a natural list with connectives tagged O.

    2 items  -> "a and b"            (no comma; matches real prose)
    3+ items -> "a , b , and c"      (comma-separated with a final 'and')
    """
    values = rng.sample(VOCAB[etype], min(n, len(VOCAB[etype])))
    out = []
    for i, val in enumerate(values):
        if i > 0:
            if len(values) == 2:
                out.append(("and", "O"))
            else:
                out.append((",", "O"))
                if i == len(values) - 1:
                    out.append(("and", "O"))
        out.extend(_tag_value(val, etype))
    return out


def _build_sample(rng):
    """Generate one (token, tag) sequence from a random template."""
    template = rng.choice(TEMPLATES)
    tagged = []
    for seg in template:
        if isinstance(seg, str):
            tagged.extend(_tag_literal(seg))
        elif seg[0] == "SLOT":
            _, etype = seg
            tagged.extend(_tag_value(rng.choice(VOCAB[etype]), etype))
        elif seg[0] == "LIST":
            _, etype, n = seg
            tagged.extend(_expand_list(rng, etype, n))
    return tagged


def generate(n_samples, seed):
    """Generate n samples; ~40% are 2-3 sentence multi-template documents."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        if rng.random() < 0.4:
            k = rng.randint(2, 3)
            combined = []
            for _ in range(k):
                combined.extend(_build_sample(rng))
            samples.append(combined)
        else:
            samples.append(_build_sample(rng))
    return samples


def write_conll(samples, out_path):
    lines = []
    for sample in samples:
        for token, tag in sample:
            lines.append(f"{token} {tag}")
        lines.append("")  # blank line between samples
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NER data.")
    parser.add_argument("--n", type=int, default=1200, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=config.NER_SEED, help="RNG seed.")
    parser.add_argument("--out", default=str(config.DATA_PATH), help="Output path.")
    args = parser.parse_args()

    samples = generate(args.n, args.seed)
    write_conll(samples, args.out)

    total_tokens = sum(len(s) for s in samples)
    print(f"Wrote {len(samples)} samples ({total_tokens} tokens) to {args.out}")


if __name__ == "__main__":
    main()
