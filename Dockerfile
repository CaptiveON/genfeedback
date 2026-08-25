# syntax=docker/dockerfile:1
#
# One image, two roles: the same image runs the Streamlit app OR the gemma-api
# (the command is overridden per service). Multi-stage build keeps the final
# image free of compilers and pip caches.

############################################################
# Stage 1: builder — install Python deps into an isolated venv.
############################################################
FROM python:3.12-slim AS builder

# Build tools some wheels need. They live ONLY in this stage, not the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# A self-contained venv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY requirements first. Docker caches layers top-to-bottom, so as long
# as requirements.txt is unchanged, this slow pip layer is reused even when you
# edit app code below. (Order matters for fast rebuilds.)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

############################################################
# Stage 2: runtime — slim image with just the venv + app code.
############################################################
FROM python:3.12-slim AS runtime

# Bring over the prebuilt venv (no build tools => smaller, safer image).
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

WORKDIR /app

# App code + the small first-party BERT NER artifacts (~435MB, fine to bake in).
# The big Gemma weights are deliberately NOT baked: they download at runtime into
# HF_HOME, which is mounted as a volume so they persist across restarts.
COPY genfeedback/ ./genfeedback/
COPY app.py gemma_api.py ./
COPY artifacts/bert_ner/ ./artifacts/bert_ner/

# Run as a non-root user — a basic but important container security practice.
RUN useradd --create-home appuser \
    && mkdir -p /models/hf && chown -R appuser:appuser /app /models
USER appuser

# Default role = the Streamlit app. docker-compose / k8s override this for the API.
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
