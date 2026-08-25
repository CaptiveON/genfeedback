"""HTTP client for the decoupled Gemma feedback service (gemma_api).

The Streamlit app uses this to POST extracted entities to the model service and
get back a feedback email, instead of loading the ~10GB model in-process. This
small module is the seam that keeps the app light and lets the model scale on
its own node/service.
"""

import requests

# Generation can be slow (especially on CPU), so give the request real headroom.
DEFAULT_TIMEOUT = 180


def request_feedback(
    api_url, jd_entities, resume_entities,
    role=None, company=None, recruiter=None, timeout=DEFAULT_TIMEOUT,
):
    """POST entities to the gemma-api `/feedback` endpoint; return the email text."""
    resp = requests.post(
        f"{api_url.rstrip('/')}/feedback",
        json={
            "jd_entities": jd_entities,
            "resume_entities": resume_entities,
            "role": role,
            "company": company,
            "recruiter": recruiter,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["feedback"]


def api_healthy(api_url, timeout=5):
    """Return True if the gemma-api is reachable and reports ready."""
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False
