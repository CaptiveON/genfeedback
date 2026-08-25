#!/usr/bin/env bash
# Launch the GenFeedback Streamlit UI with the PROJECT VENV's Python.
#
# Why this script: a bare `streamlit run app.py` uses the system `streamlit`
# binary, whose shebang points at the system Python (transformers 4.46, which
# does NOT know Gemma 4). The venv has transformers 5.x. Running via
# `.venv/bin/python -m streamlit` guarantees the app loads under the venv.
set -e
cd "$(dirname "$0")"
PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false \
  exec .venv/bin/python -m streamlit run app.py "$@"
