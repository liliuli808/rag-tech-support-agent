"""A tiny local web UI for the Atlas support agent.

Why this exists alongside JupyterLab
------------------------------------
JupyterLab is the right tool for reading and running the notebook itself; this
page is for *asking the agent questions* without touching a kernel. It imports
`ask.py` -- which executes the notebook's own code cells in-process -- and puts a
chat page on top of it, showing the groundedness coverage, the top similarity
and the latency of every answer. Nothing here can drift from the notebook,
because the pipeline is literally the notebook's.

Configuration
-------------
Credentials and the endpoint come from `.env` at the project root, exactly as in
the notebook. There is no offline mode: if the configuration is missing or the
endpoint is unreachable, startup prints the notebook's diagnostic and exits with
code 3 instead of serving a degraded agent.

Run
---
    python tools/webui.py                 # http://127.0.0.1:8777
    python tools/webui.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ask as atlas                                  # noqa: E402  (sibling module)

LOCK = threading.Lock()
AGENT: dict = {}

SAMPLES = [
    "How do I reset the office router?",
    "What does a 429 error mean?",
    "The VPN fails with E-2011, what should I do?",
    "How much does the Premium subscription cost?",
    "How do I restore a deleted file?",
    "What is the capital of France?",              # deliberately out of domain
]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlas - Technical Support Agent</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #1c2024; --muted: #6b7280;
    --line: #e4e7ec; --accent: #2f6feb; --accent-soft: #eaf1fe;
    --warn: #b54708; --warn-soft: #fef4e6; --ok: #0f7b4f; --ok-soft: #e8f6ef;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI",
               "Microsoft YaHei", Roboto, sans-serif; }
  header { background: var(--panel); border-bottom: 1px solid var(--line);
           padding: 16px 22px; position: sticky; top: 0; z-index: 5; }
  h1 { margin: 0 0 6px; font-size: 17px; font-weight: 650; letter-spacing: .1px; }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; }
  .badge { font-size: 11.5px; padding: 2px 8px; border-radius: 999px;
           background: #eef1f5; color: var(--muted); border: 1px solid var(--line); }
  .badge.on { background: var(--accent-soft); color: var(--accent);
              border-color: #cfdcfb; }
  main { max-width: 860px; margin: 0 auto; padding: 22px 18px 150px; }
  .samples { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .samples button { font: inherit; font-size: 13px; padding: 6px 12px;
      border: 1px solid var(--line); background: var(--panel); color: var(--ink);
      border-radius: 999px; cursor: pointer; }
  .samples button:hover { border-color: var(--accent); color: var(--accent); }
  .turn { margin-bottom: 18px; }
  .who { font-size: 11.5px; text-transform: uppercase; letter-spacing: .7px;
         color: var(--muted); margin-bottom: 5px; }
  .q { background: var(--accent); color: #fff; padding: 10px 14px;
       border-radius: 12px 12px 12px 3px; display: inline-block;
       max-width: 100%; }
  .a { background: var(--panel); border: 1px solid var(--line); padding: 14px 16px;
       border-radius: 3px 12px 12px 12px; white-space: pre-wrap; }
  .a ul { margin: 6px 0 0; padding-left: 20px; }
  .a li { margin: 2px 0; }
  .src { display: inline-block; font-size: 12px; background: var(--ok-soft);
         color: var(--ok); border: 1px solid #cbe8da; border-radius: 6px;
         padding: 1px 7px; margin-top: 10px; }
  .meta { margin-top: 10px; font-size: 12px; color: var(--muted); }
  .meta span { margin-right: 14px; }
  .abstain { background: var(--warn-soft); border-color: #f5dcc0;
             color: var(--warn); }
  footer { position: fixed; bottom: 0; left: 0; right: 0; background: var(--panel);
           border-top: 1px solid var(--line); padding: 12px 18px; }
  form { max-width: 860px; margin: 0 auto; display: flex; gap: 10px; }
  input { flex: 1; font: inherit; padding: 11px 14px; border-radius: 10px;
          border: 1px solid var(--line); background: #fff; color: var(--ink); }
  input:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
  form button { font: inherit; font-weight: 600; padding: 11px 22px; border: 0;
      border-radius: 10px; background: var(--accent); color: #fff; cursor: pointer; }
  form button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<header>
  <h1>Atlas &mdash; Technical Support Agent <span style="color:var(--muted);font-weight:400">(RAG)</span></h1>
  <div class="badges" id="badges"></div>
</header>
<main>
  <div class="samples" id="samples"></div>
  <div id="log"></div>
</main>
<footer>
  <form id="form">
    <input id="q" placeholder="Ask a support question..." autocomplete="off" autofocus>
    <button id="send" type="submit">Ask</button>
  </form>
</footer>
<script>
const log = document.getElementById('log');
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function renderAnswer(text) {
  const lines = text.split('\\n');
  let html = '', inList = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { if (inList) { html += '</ul>'; inList = false; } continue; }
    const src = line.match(/^\\[Source:\\s*(.+?)\\]$/);
    if (src) {
      if (inList) { html += '</ul>'; inList = false; }
      html += '<div><span class="src">Source: ' + esc(src[1]) + '</span></div>';
      continue;
    }
    if (line.startsWith('- ')) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + esc(line.slice(2)) + '</li>';
      continue;
    }
    if (inList) { html += '</ul>'; inList = false; }
    html += '<div>' + esc(line) + '</div>';
  }
  if (inList) html += '</ul>';
  return html;
}

function turn(question, data) {
  const el = document.createElement('div');
  el.className = 'turn';
  const elapsed = data.elapsed_ms != null ? data.elapsed_ms + ' ms' : '';
  el.innerHTML =
    '<div class="who">You</div><div class="q">' + esc(question) + '</div>' +
    '<div class="who" style="margin-top:12px">Atlas</div>' +
    '<div class="a' + (data.abstained ? ' abstain' : '') + '">' +
      renderAnswer(data.answer) +
      '<div class="meta">' +
        '<span>coverage <b>' + data.coverage + '</b> / gate ' + data.threshold + '</span>' +
        '<span>top sim <b>' + data.top_similarity + '</b></span>' +
        (data.gated ? '<span>refused before the model was called</span>' : '') +
        '<span>' + elapsed + '</span>' +
      '</div>' +
    '</div>';
  log.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

async function send(question) {
  if (!question.trim()) return;
  const btn = document.getElementById('send');
  btn.disabled = true;
  try {
    const res = await fetch('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    turn(question, data);
  } catch (e) {
    alert('Request failed: ' + e.message);
  } finally {
    btn.disabled = false;
    document.getElementById('q').focus();
  }
}

document.getElementById('form').addEventListener('submit', ev => {
  ev.preventDefault();
  const input = document.getElementById('q');
  const q = input.value;
  input.value = '';
  send(q);
});

fetch('/api/meta').then(r => r.json()).then(m => {
  document.getElementById('badges').innerHTML = [
    m.endpoint,
    m.chunks + ' chunks indexed',
    m.embed_backend,
    m.generator_backend,
    'abstention gate &lt; ' + m.threshold
  ].map(t => '<span class="badge">' + t + '</span>').join('');
  document.getElementById('samples').innerHTML = m.samples
    .map(s => '<button type="button">' + s.replace(/[&<>]/g, c =>
        ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])) + '</button>').join('');
  document.querySelectorAll('#samples button').forEach(b =>
    b.addEventListener('click', () => send(b.textContent)));
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AtlasUI"

    def log_message(self, fmt, *args):          # keep the console readable
        sys.stdout.write(f"  {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:                    # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._json({
                "chunks": len(AGENT["chunks"]),
                "endpoint": AGENT.get("RESOLVED_BASE_URL", "?"),
                "embed_backend": AGENT.get("EMBED_BACKEND", "?"),
                "generator_backend": AGENT.get("GENERATOR_BACKEND", "?"),
                "threshold": AGENT["COVERAGE_THRESHOLD"],
                "samples": SAMPLES,
            })
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:                   # noqa: N802
        if self.path != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                self._json({"error": "empty question"}, 400)
                return
            import time

            started = time.perf_counter()
            with LOCK:
                result = AGENT["ask"](question)
            result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            result["threshold"] = AGENT["COVERAGE_THRESHOLD"]
            self._json(result)
        except Exception as exc:                 # surface it in the UI, not the log
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local web UI for the Atlas support agent.")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: loopback only)")
    args = parser.parse_args(argv)

    print("[webui] initialising the agent from the notebook ...")
    try:
        AGENT.update(atlas.build_agent())
    except atlas.SetupError as exc:
        # Configuration / preflight failure: the message already says what to fix.
        print(exc, file=sys.stderr)
        print("[webui] not started - fix the configuration above and retry.",
              file=sys.stderr)
        return 3

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[webui] endpoint: {AGENT.get('RESOLVED_BASE_URL')}")
    print(f"[webui] {len(AGENT['chunks'])} chunks | embedder: {AGENT.get('EMBED_BACKEND')}")
    print(f"[webui] generator: {AGENT.get('GENERATOR_BACKEND')}")
    print(f"[webui] serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
