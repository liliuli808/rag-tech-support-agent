"""Verify that an OpenAI-compatible endpoint can actually serve this pipeline.

The RAG agent needs **two** routes. A relay that only proxies chat completions
looks perfectly healthy until §6 tries to embed and fails with an opaque stack
trace. This script answers the two questions that matter, before anything is
indexed:

  1. Does `/embeddings` exist, and which embedding model does it accept?
  2. Does `/chat/completions` exist, and which chat model does it accept?
  3. What is the embedding dimension? (it decides the FAISS index shape and must
     match what the notebook reads at runtime)

Usage
-----
    python tools/check_endpoint.py                      # read .env
    python tools/check_endpoint.py --embed-model bge-m3
    python tools/check_endpoint.py --base-url https://relay.example/v1 --api-key sk-...
    python tools/check_endpoint.py --list-models        # just dump /models

Exit codes
----------
    0  both routes work with the configured models
    1  the endpoint is reachable but a route or model is unusable
    2  the configuration itself is missing or incomplete
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Model names worth trying when the configured one is rejected. Ordered by how
# likely a relay is to carry them.
EMBED_FALLBACKS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "bge-m3",
    "bge-large-zh-v1.5",
    "text-embedding-v3",
    "text-embedding-v2",
    "embedding-3",
    "nomic-embed-text",
]
CHAT_FALLBACKS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"]

PLACEHOLDERS = {"sk-your-key-here", "your-key-here", "sk-xxx", "changeme", ""}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def load_dotenv_file(path: Path) -> dict:
    """Minimal .env reader (no dependency on python-dotenv, so this works even
    before the environment is set up)."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def mask(secret: str) -> str:
    if not secret:
        return "(empty)"
    if len(secret) <= 12:
        return secret[:3] + "..." + secret[-2:]
    return f"{secret[:8]}...{secret[-4:]} (len={len(secret)})"


def resolve_config(args) -> dict:
    """Environment variables win over .env, matching the notebook's policy."""
    dotenv = load_dotenv_file(ENV_PATH)
    cfg = {
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY") or dotenv.get("OPENAI_API_KEY", ""),
        "base_url": (args.base_url or os.environ.get("OPENAI_BASE_URL")
                     or dotenv.get("OPENAI_BASE_URL", "")).rstrip("/"),
        "embed_model": (args.embed_model or os.environ.get("OPENAI_EMBED_MODEL")
                        or dotenv.get("OPENAI_EMBED_MODEL", "")),
        "chat_model": (args.chat_model or os.environ.get("OPENAI_CHAT_MODEL")
                       or dotenv.get("OPENAI_CHAT_MODEL", "")),
        # Where the vectors come from. With EMBED_BACKEND=local the endpoint is
        # never asked for an embedding, so probing /embeddings would report a
        # failure that says nothing about whether the pipeline can actually run.
        "embed_backend": (os.environ.get("EMBED_BACKEND")
                          or dotenv.get("EMBED_BACKEND", "api")).strip().lower(),
        "local_embed_model": (os.environ.get("LOCAL_EMBED_MODEL")
                              or dotenv.get("LOCAL_EMBED_MODEL", "")),
    }
    return cfg


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def probe_models(client, base_url: str) -> list:
    try:
        r = client.get(f"{base_url}/models")
        if r.status_code != 200:
            return []
        return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
    except Exception:                                       # noqa: BLE001
        return []


def probe_embeddings(client, base_url: str, model: str):
    """Return (ok, dim, detail)."""
    r = client.post(f"{base_url}/embeddings", json={"model": model, "input": "hello world"})
    if r.status_code == 200:
        try:
            payload = r.json()
            dim = len(payload["data"][0]["embedding"])
            return True, dim, f"dim={dim}, reported model={payload.get('model')}"
        except Exception as exc:                            # noqa: BLE001
            return False, 0, f"200 but malformed body: {exc}"
    return False, 0, f"HTTP {r.status_code}: {_short(r)}"


def probe_chat(client, base_url: str, model: str):
    """Return (ok, detail).

    A 200 alone does not prove that the model we asked for is the model that
    answered: a relay is free to route elsewhere and label the reply with your
    requested id. The response body's own `model` field is the only
    client-visible evidence, so it is compared against the request. A mismatch
    is reported as a warning rather than a failure - the route does work, but
    what served it is not what you asked for.
    """
    r = client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK"}],
            # Reasoning models spend the first few tokens before emitting any
            # visible text, so a very small budget yields an empty string and
            # the probe reads as a failure when the route is in fact healthy.
            "max_tokens": 32,
        },
    )
    if r.status_code == 200:
        try:
            body = r.json()
            content = body["choices"][0]["message"].get("content", "")
            served = body.get("model")
        except Exception as exc:                            # noqa: BLE001
            return False, f"200 but malformed body: {exc}"
        if str(content).strip():
            detail = f"replied {str(content)[:40]!r}"
        else:
            detail = "empty content despite HTTP 200 (raise max_tokens)"
        if served and served != model:
            detail += f"  [WARNING: served as {served!r}]"
        elif served:
            detail += f"  [served as {served}]"
        return True, detail
    return False, f"HTTP {r.status_code}: {_short(r)}"


def _short(response, limit: int = 200) -> str:
    text = (response.text or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def diagnose(status_code) -> str:
    return {
        401: "credential rejected - the key is wrong, expired or revoked",
        402: "payment required - the account has no credit",
        403: "authenticated but not authorised - often an out-of-quota account, "
             "or the key's group has no channel for this model",
        404: "no such route on this host - the base URL prefix is probably wrong "
             "(the SDK appends /embeddings and /chat/completions)",
        422: "the request shape was rejected - the gateway is not fully "
             "OpenAI-compatible",
        429: "rate limited or out of quota",
        502: "bad gateway - the relay's upstream failed",
        503: "service unavailable - no channel is configured for this model "
             "(the model name is often simply not offered)",
        504: "gateway timeout - the upstream did not answer in time",
    }.get(status_code, "unexpected status")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", help="override OPENAI_BASE_URL")
    parser.add_argument("--api-key", help="override OPENAI_API_KEY")
    parser.add_argument("--embed-model", help="override OPENAI_EMBED_MODEL")
    parser.add_argument("--chat-model", help="override OPENAI_CHAT_MODEL")
    parser.add_argument("--list-models", action="store_true",
                        help="only fetch and print the /models listing")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    try:
        import httpx
    except ImportError:
        print("httpx is required. It is listed in requirements.txt.", file=sys.stderr)
        return 2

    cfg = resolve_config(args)
    base_url = cfg["base_url"] or "https://api.openai.com/v1"

    print("=" * 74)
    print("ENDPOINT VERIFICATION")
    print("=" * 74)
    print(f"env file        : {ENV_PATH}  ({'found' if ENV_PATH.exists() else 'NOT FOUND'})")
    print(f"base URL        : {base_url}")
    print(f"api key         : {mask(cfg['api_key'])}")
    if cfg["embed_backend"] == "local":
        print(f"embeddings      : LOCAL -> {cfg['local_embed_model'] or '(unset)'}"
              f"   (in-process; this endpoint is not asked for vectors)")
        print(f"embed model     : {cfg['embed_model'] or '(not set)'}"
              f"   (only used when EMBED_BACKEND=api)")
    else:
        print(f"embed model     : {cfg['embed_model'] or '(not set)'}")
    print(f"chat model      : {cfg['chat_model'] or '(not set)'}")
    print()

    if not cfg["api_key"] or cfg["api_key"].strip() in PLACEHOLDERS:
        print("CONFIGURATION INCOMPLETE")
        print("  OPENAI_API_KEY is missing or still the placeholder value.")
        print(f"  Set it in {ENV_PATH} (or pass --api-key).")
        print("=" * 74)
        return 2

    # A relay with an expired certificate is common; verify is relaxed because
    # this script only ever talks to a host the user explicitly configured.
    with httpx.Client(
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        timeout=args.timeout,
        follow_redirects=True,
        verify=False,
    ) as client:
        # ---- /models ---------------------------------------------------
        print("-" * 74)
        print("GET /models")
        print("-" * 74)
        models = probe_models(client, base_url)
        if models:
            embed_like = [m for m in models if "embed" in m.lower()]
            print(f"  {len(models)} models advertised")
            print(f"  embedding-capable ({len(embed_like)}): {embed_like[:20] or 'none'}")
            preview = models[:20]
            print(f"  first 20: {preview}")
            if cfg["embed_model"] and cfg["embed_model"] not in models:
                print(f"  NOTE: configured embed model {cfg['embed_model']!r} is not in this list")
            if cfg["chat_model"] and cfg["chat_model"] not in models:
                print(f"  NOTE: configured chat model {cfg['chat_model']!r} is not in this list")
        else:
            print("  no listing returned (many relays disable /models; not fatal)")

        if args.list_models:
            print("=" * 74)
            return 0

        # ---- /embeddings -----------------------------------------------
        print()
        print("-" * 74)
        print("POST /embeddings")
        print("-" * 74)
        embed_ok, embed_dim, embed_working_model = False, 0, None
        embed_local = cfg["embed_backend"] == "local"
        if embed_local:
            print("  SKIPPED - EMBED_BACKEND=local")
            print(f"  Vectors are produced in-process by "
                  f"{cfg['local_embed_model'] or 'the configured model'};")
            print("  this endpoint only has to serve /chat/completions.")
        else:
            candidates = ([cfg["embed_model"]] if cfg["embed_model"] else []) + EMBED_FALLBACKS
            seen = set()
            for model in candidates:
                if not model or model in seen:
                    continue
                seen.add(model)
                try:
                    ok, dim, detail = probe_embeddings(client, base_url, model)
                except Exception as exc:                    # noqa: BLE001
                    ok, dim, detail = False, 0, f"{type(exc).__name__}: {str(exc)[:140]}"
                print(f"  {model:26s} {'OK  ' if ok else 'FAIL'}  {detail}")
                if ok:
                    embed_ok, embed_dim, embed_working_model = True, dim, model
                    break

            if not embed_ok:
                print()
                print("  No embedding model worked.")
                print("  This endpoint cannot serve the retrieval half of the pipeline.")
                print("  Either find an embedding model it does serve, or set")
                print("  EMBED_BACKEND=local to run the embedder in-process.")

        # ---- /chat/completions -----------------------------------------
        print()
        print("-" * 74)
        print("POST /chat/completions")
        print("-" * 74)
        chat_ok, chat_working_model = False, None
        candidates = ([cfg["chat_model"]] if cfg["chat_model"] else []) + CHAT_FALLBACKS
        seen = set()
        for model in candidates:
            if not model or model in seen:
                continue
            seen.add(model)
            try:
                ok, detail = probe_chat(client, base_url, model)
            except Exception as exc:                        # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {str(exc)[:140]}"
            print(f"  {model:26s} {'OK  ' if ok else 'FAIL'}  {detail}")
            if ok:
                chat_ok, chat_working_model = True, model
                break

        if not chat_ok:
            print()
            print("  No chat model worked.")
            print("  The generator half of the pipeline cannot run on this endpoint.")

    # ---- verdict -------------------------------------------------------
    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    if embed_local:
        print(f"  embeddings : LOCAL  -> {cfg['local_embed_model'] or 'in-process model'}"
              f"   (endpoint not used)")
    else:
        print(f"  embeddings : {'USABLE' if embed_ok else 'NOT USABLE'}"
              + (f"  -> {embed_working_model}, {embed_dim}-d" if embed_ok else ""))
    print(f"  chat       : {'USABLE' if chat_ok else 'NOT USABLE'}"
          + (f"  -> {chat_working_model}" if chat_ok else ""))

    embed_satisfied = embed_ok or embed_local

    if embed_satisfied and chat_ok:
        print()
        print("  Both halves of the pipeline are covered. Put this in .env:")
        print(f"    OPENAI_BASE_URL={base_url}")
        print("    OPENAI_API_KEY=<your key>")
        print(f"    OPENAI_CHAT_MODEL={chat_working_model}")
        if embed_local:
            print("    EMBED_BACKEND=local")
            print(f"    LOCAL_EMBED_MODEL={cfg['local_embed_model'] or '<hf model id>'}")
            print("    (embeddings run in-process, so OPENAI_EMBED_MODEL is unused)")
        else:
            print(f"    OPENAI_EMBED_MODEL={embed_working_model}")
        print("=" * 74)
        return 0

    print()
    if not chat_ok:
        print("  This endpoint cannot serve the generator half of the pipeline.")
    else:
        print("  This endpoint cannot serve the retrieval half of the pipeline.")
        print("  Either point OPENAI_EMBED_MODEL at a model it does serve, or set")
        print("  EMBED_BACKEND=local to run the embedder in-process.")
    print("=" * 74)
    return 1


if __name__ == "__main__":
    # Local relays frequently use self-signed certificates.
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
