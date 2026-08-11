# MCP server (team Justin)

Three independent MCP servers, one per grading stage, served from a single
process on a single port:

| Path                    | Stage | Tools       |
|-------------------------|-------|-------------|
| `POST /1/evaluate/mcp`  | 1  | `get_name`, `do_arithmetic`, `resolve_whole_expr`, `classify_shape_from_base64`, `shape_total` |
| `POST /2/evaluate/mcp`  | 2  | `go`, `recall` |
| `POST /3/evaluate/mcp`  | 3  | `evaluate` |
| `GET  /health`          | -  | plain liveness check |
| `POST /event`           | -  | grader telemetry sink, logged to stdout |

Tool names and signatures come from the per-stage cheat sheets on the grader
host, served at `/{stage}/cheat-sheet` (markdown at
`/docs/stage-{n}-cheat-sheet.md`). They are the authoritative contract and
are not linked from the public briefs.

`recall` returns a JSON **string**, not a `list[str]`, and this is load
bearing: MCP serialises a list return as one text block per element, so a
list makes the grader parse a bare passage instead of a JSON array and
permanently void the question.

Registered `teamUrl` per stage is `https://<host>/1/evaluate`,
`https://<host>/2/evaluate`, etc. - the grader appends `/mcp` itself.

Stage 1 and 2 are implemented in `server/stages/stage{1,2}.py`. Stage 3 is
still a stub in `server/stages/stage3.py`, replacing the `evaluate()` stub
when that work starts. They're kept as separate FastMCP instances/modules
so a heavy import added to one stage never gets paid for by the others.

Stage 2 bundles a small (39MB) pre-trained GloVe word-vector table at
`server/stages/data/glove50_trimmed.npz`, used to blend TF-IDF with
word-embedding cosine similarity for revision-passage retrieval. Plain
TF-IDF alone was tested against the real study material and badly fails
questions with no literal keyword overlap with the source text (e.g. a
question about a "sensor grid" being "brought back into alignment" when
the source text says an "array" was "recalibrated") - see
`server/stages/stage2.py`'s module docstring.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PORT=8000 python -m server.app
# or: uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Constraints this scaffold was built against

- Must run inside 512 MiB total, across whichever single stage is graded at
  a time (the grader only ever calls one of `/1`, `/2`, `/3` per run).
- Each tool call has a 10s budget, including any first-call cold work — so
  expensive setup (loading a model, building an index) belongs in a startup
  hook, not inside `evaluate()`.
- The ML **libraries** you import cost far more than the models you load
  with them. Look hard at imports, not just model size, when a stage needs
  real ML work later. Keep heavy imports inside `evaluate()` (or a
  lazily-invoked setup function) in the specific stage module that needs
  them, not at module scope, so the other two stages never pay for it.
- DNS-rebinding host-header protection is disabled in
  `server/transport.py` (`PUBLIC_TRANSPORT_SECURITY`) since this server must
  be reachable from an arbitrary internet host, not just localhost.
