#!/usr/bin/env bash
# Reproduce the Python environment for the Atlas RAG support agent.
#
#   bash tools/setup_wsl.sh
#
# Creates ./.venv, installs requirements.txt + JupyterLab, and registers the
# `ragenv313` kernel that tools/run_notebook.py executes against.
#
# This only prepares the interpreter. The agent itself needs an OpenAI-compatible
# endpoint, configured in .env - see the final message printed below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "project : $ROOT"

if [ ! -d .venv ]; then
    echo "creating .venv ..."
    python3 -m venv .venv
fi

PY="$ROOT/.venv/bin/python"
"$PY" -V

echo "installing dependencies ..."
"$PY" -m pip install --upgrade pip -q

# torch comes FIRST and from the CPU-only index, on purpose. requirements.txt
# lists sentence-transformers, which depends on torch; if torch is not already
# present pip resolves it from PyPI, which serves the CUDA build (~2.5 GB) and
# is useless without an NVIDIA GPU. Installing the CPU build up front costs
# ~200 MB and makes the later resolution a no-op.
if "$PY" -c "import torch" 2>/dev/null; then
    echo "  torch already present: $("$PY" -c 'import torch; print(torch.__version__)')"
else
    echo "  installing CPU-only torch ..."
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

"$PY" -m pip install -r requirements.txt
# JupyterLab is the notebook UI; notebook code execution itself only needs the
# ipykernel/nbclient already listed in requirements.txt.
"$PY" -m pip install jupyterlab

echo "registering kernel 'ragenv313' ..."
"$PY" -m ipykernel install --user --name ragenv313 \
      --display-name "Python 3 (RAG agent)" >/dev/null
echo "  -> $(jupyter kernelspec list 2>/dev/null | grep ragenv313 || echo 'check manually')"

# The interpreter is ready; the credential is not. Say so loudly rather than
# letting the first run fail with a kernel traceback.
if [ -f .env ]; then
    echo "found .env"
else
    echo "no .env yet - creating one from .env.example"
    cp .env.example .env
    echo "  ACTION REQUIRED: open .env and set OPENAI_API_KEY before running."
fi

cat <<EOF

done. next steps:

  1. set OPENAI_API_KEY in .env   (optional: OPENAI_BASE_URL, model names)

     EMBED_BACKEND=api    embed through the endpoint  (default)
     EMBED_BACKEND=local  embed with LOCAL_EMBED_MODEL in-process

     With EMBED_BACKEND=local the endpoint only has to serve chat, so a
     chat-only relay becomes usable. The key stays required either way.

  2. then run any of:
       $PY tools/run_notebook.py                  # headless re-run -> expect errors=0
       $PY tools/ask.py -v "how do I reset the office router"
       $PY tools/webui.py --port 8777             # http://127.0.0.1:8777
       $PY -m jupyter lab --no-browser --port 8888 --IdentityProvider.token=atlas

  3. to check an endpoint before trusting it:
       $PY tools/check_endpoint.py

EOF
