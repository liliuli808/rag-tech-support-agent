"""Command-line interface to the Atlas technical-support agent.

Why this file exists
--------------------
The notebook is the single source of truth for the pipeline. Rather than
re-implementing retrieval and grounding here (which would silently drift from
the notebook), this script *extracts the notebook's own code cells and executes
them* into one shared namespace. Whatever the notebook does, the CLI does.

Usage
-----
    python tools/ask.py                          # interactive session
    python tools/ask.py "how do I reset the router"
    python tools/ask.py -v "what does 429 mean"  # show coverage and sources
    python tools/ask.py --k 6 "vpn errors"
    python tools/ask.py --json "..."             # machine-readable output

Configuration
-------------
Credentials come from `.env` at the project root (loaded by the notebook's own
section 3, which this script executes):

    OPENAI_API_KEY      required
    OPENAI_BASE_URL     optional - any OpenAI-compatible gateway
    OPENAI_EMBED_MODEL  optional
    OPENAI_CHAT_MODEL   optional

There is no offline mode. If the configuration is missing or the endpoint is
unreachable, the script prints the notebook's diagnostic and exits with code 3
rather than answering from a degraded local model.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

import nbformat


class SetupError(RuntimeError):
    """A configuration or connectivity failure that is already self-explained."""


def _is_setup_error(exc: BaseException) -> bool:
    """True when *exc* carries one of the notebook's diagnostic banners."""
    message = str(exc)
    return ("CONFIGURATION ERROR" in message) or ("PREFLIGHT FAILED" in message)


# --------------------------------------------------------------------------
# Locate the project and the notebook
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "Technical_Support_AI_Agent_RAG.ipynb"

# The notebook's rendering/side effects must not leak into the terminal.
MAGIC_RE = re.compile(r"^\s*[%!]")

BANNER = r"""
  _   _   _         _
 / \ | |_| | __ _ ___
/ _ \| __| |/ _` / __|   Atlas - technical-support AI agent (RAG)
/_/ \_\__|_|\__,_\___|   grounded retrieval + LLM generation, .env configured
"""


def _prepare_cell_source(source: str) -> str:
    """Drop IPython magics/shell escapes so the cell is valid plain Python."""
    kept = [ln for ln in source.splitlines() if not MAGIC_RE.match(ln)]
    return "\n".join(kept)


def build_agent(verbose: bool = False) -> dict:
    """Execute the notebook's code cells and return the resulting namespace.

    Execution stops as soon as the public entry point ``ask`` has been defined,
    so the evaluation and plotting cells at the end of the notebook are skipped.
    """
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"notebook not found: {NOTEBOOK}")

    nb = nbformat.read(str(NOTEBOOK), as_version=4)

    # The notebook resolves the knowledge base and artifacts/ relative to the
    # current working directory, so pin it before running anything.
    os.chdir(PROJECT_ROOT)

    # `%matplotlib inline` is a Jupyter magic; force a headless backend instead.
    import warnings

    import matplotlib

    matplotlib.use("Agg")

    # The notebook calls plt.show() after saving each figure, which means
    # nothing outside a notebook. Silence it so the CLI stays clean.
    import matplotlib.pyplot as _plt

    _plt.show = lambda *args, **kwargs: None
    warnings.filterwarnings("ignore", message=".*non-interactive.*")

    namespace: dict = {"__name__": "__atlas__"}
    buffer = io.StringIO()
    ran = 0

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = _prepare_cell_source(cell.source).strip()
        if not source:
            continue
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, f"<cell {index}>", "exec"), namespace)
        except RuntimeError as exc:
            # Configuration / preflight failures already carry a full
            # diagnosis; re-raising them verbatim beats a stack trace.
            if _is_setup_error(exc):
                raise SetupError(str(exc)) from exc
            print(f"[atlas] cell {index} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            raise
        except Exception as exc:                      # pragma: no cover
            print(f"[atlas] cell {index} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            raise
        ran += 1
        if "ask" in namespace:
            break

    namespace["_cells_executed"] = ran
    if verbose:
        # stdout, not stderr: PowerShell renders native stderr as a red
        # NativeCommandError block, which makes a normal run look like a crash.
        print(f"[atlas] initialised from {NOTEBOOK.name}: "
              f"{ran} cells, {len(namespace['chunks'])} chunks indexed, "
              f"backend = {namespace.get('EMBED_BACKEND')}")
    return namespace


def render(result: dict, show_trace: bool) -> str:
    lines = [result["answer"]]
    if show_trace:
        lines += [
            "",
            f"  coverage        : {result['coverage']}   "
            f"(abstention threshold {result['threshold']})",
            f"  top similarity  : {result['top_similarity']}",
            f"  citations       : {', '.join(result['sources']) or '-'}",
            f"  context chars   : {result['context_chars']}",
            f"  abstained       : {result['abstained']}",
            f"  gated locally   : {result.get('gated')}"
            f"   (True = refused before calling the model)",
        ]
    return "\n".join(lines)


def answer(ns: dict, question: str, k: int, show_trace: bool, as_json: bool) -> str:
    ask = ns["ask"]
    top_k = ns["TOP_K"]
    threshold = ns["COVERAGE_THRESHOLD"]

    result = ask(question, k=k)
    result["threshold"] = threshold

    if as_json:
        return json.dumps(result, ensure_ascii=False, indent=2)
    out = render(result, show_trace)
    if not as_json and k != top_k and show_trace:
        out += f"\n  (top-k overridden: {k}, notebook default {top_k})"
    return out


def repl(ns: dict, k: int, show_trace: bool) -> int:
    print(BANNER)
    print(f"  endpoint  : {ns.get('RESOLVED_BASE_URL')}")
    print(f"  {len(ns['chunks'])} chunks indexed | embedder: {ns['EMBED_BACKEND']}")
    print(f"  generator : {ns['GENERATOR_BACKEND']}")
    print("  Type a question, or 'exit' / Ctrl-C to quit.")
    print("  'trace on' / 'trace off' toggles the diagnostics block.\n")

    while True:
        try:
            question = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return 0
        if question.lower() in {"trace on", "trace off"}:
            show_trace = question.endswith("on")
            print(f"  diagnostics {'on' if show_trace else 'off'}\n")
            continue
        try:
            print()
            print(answer(ns, question, k, show_trace, as_json=False))
            print()
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the Atlas technical-support agent from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               '  python tools/ask.py "how do I reset the office router"\n'
               '  python tools/ask.py -v "what does a 429 error mean"\n'
               '  python tools/ask.py -k 6 "vpn tunnel establishment timeout"\n',
    )
    parser.add_argument("question", nargs="*", help="question; omit for a REPL")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show coverage, similarity and citations")
    parser.add_argument("-k", "--top-k", type=int, default=None,
                        help="number of chunks to retrieve (default: notebook's TOP_K)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    try:
        ns = build_agent(verbose=args.verbose)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except SetupError as exc:
        # The message already explains what to fix (see .env.example).
        print(exc, file=sys.stderr)
        return 3

    k = args.top_k or ns["TOP_K"]

    if not args.question:
        return repl(ns, k, args.verbose)

    question = " ".join(args.question)
    print(answer(ns, question, k, args.verbose, args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
