# -*- coding: utf-8 -*-
"""Execute the RAG notebook headlessly and report any failure.

The notebook performs a fatal endpoint preflight in section 3, so a real
credential is required. This script checks for one *before* starting a kernel,
which turns a 30-second kernel boot into an immediate, readable message.

Configuration comes from `.env` at the project root (see `.env.example`), or
from the process environment.

Usage:
    python run_notebook.py            # execute, assert zero errors
    RAG_PUBLISH=1 python run_notebook.py   # also write the results back
"""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent   # .../rag-tech-support-agent
SRC = ROOT / "Technical_Support_AI_Agent_RAG.ipynb"
KERNEL = "ragenv313"
ENV_PATH = ROOT / ".env"
OUT_PATH = ROOT / "_executed.ipynb"


PLACEHOLDERS = {"sk-your-key-here", "sk-xxx", "your-api-key", "changeme"}


def _real(value: str) -> bool:
    """True when *value* looks like an actual credential, not a template stub."""
    value = value.strip()
    return bool(value) and value.lower() not in PLACEHOLDERS \
        and not value.startswith("sk-your-")


def _has_credential() -> bool:
    """True when OPENAI_API_KEY is set in the environment or in .env."""
    if _real(os.environ.get("OPENAI_API_KEY") or ""):
        return True
    if not ENV_PATH.exists():
        return False
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY" and _real(value.strip().strip("'\"")):
            return True
    return False


if not _has_credential():
    print("[run] NO CREDENTIAL FOUND - refusing to start a kernel.", file=sys.stderr)
    print("", file=sys.stderr)
    print("The notebook has no offline mode: section 3 fails fast without a key,", file=sys.stderr)
    print("and the kernel would only report that after a slow boot.", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  1. cp .env.example .env      (in {ROOT})", file=sys.stderr)
    print("  2. set OPENAI_API_KEY in .env", file=sys.stderr)
    print("  3. re-run this script", file=sys.stderr)
    sys.exit(3)

nb = nbformat.read(str(SRC), as_version=4)
nb.metadata.setdefault("kernelspec", {})
nb.metadata["kernelspec"]["name"] = KERNEL

client = NotebookClient(
    nb,
    timeout=900,
    kernel_name=KERNEL,
    allow_errors=False,
    resources={"metadata": {"path": str(ROOT)}},
)

print(f"[run] executing {SRC.name} with kernel {KERNEL!r} ...")
try:
    client.execute()
except Exception as exc:
    print(f"[run] EXECUTION FAILED: {type(exc).__name__}: {exc}")
    # Persist whatever was produced so the failing cell can be inspected.
    nbformat.write(nb, str(ROOT / "_failed.ipynb"))
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                print(f"\n--- error in cell #{i} ---")
                print("".join(out.get("traceback", []))[-3000:])
    sys.exit(1)

nbformat.write(nb, str(OUT_PATH))
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_err = sum(
    1
    for c in nb.cells
    if c.cell_type == "code"
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
)
print(f"[run] OK - {len(nb.cells)} cells ({n_code} code), errors={n_err}")
print(f"[run] written to {OUT_PATH.name}")

# Optionally publish this execution as the shipped notebook, so the delivered
# .ipynb already contains its results. The kernelspec is reset to the portable
# "python3" name because the local "ragenv313" spec will not exist elsewhere.
if os.environ.get("RAG_PUBLISH") == "1":
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.write(nb, str(SRC))
    print(f"[run] published over {SRC.name} (kernelspec reset to python3)")

# Print the last code cell's text output so the run can be verified at a glance.
for cell in reversed(nb.cells):
    if cell.cell_type == "code":
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                print("\n--- final report ---")
                print("".join(out.get("text", "")))
                sys.exit(0)
