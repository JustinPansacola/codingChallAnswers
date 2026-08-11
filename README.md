# MCP server (team Justin)

Three independent MCP servers, one per grading stage, served from a single
process on a single port:

| Path            | Stage | Tools       |
|-----------------|-------|-------------|
| `POST /1/evaluate` | 1  | `get_name`, `calculate`, `classify_shape`, `shape_total` |
| `POST /2/evaluate` | 2  | `evaluate` |
| `POST /3/evaluate` | 3  | `evaluate` |
| `GET  /health`     | -  | plain liveness check |

Stage 1 is implemented in `server/stages/stage1.py`. Stages 2 and 3 are still
stubs in `server/stages/stage{2,3}.py`, replacing the `evaluate()` stub as
that work starts. They're kept as separate FastMCP instances/modules so a
heavy import added to one stage (e.g. an ML library for stage 2) never gets
paid for by the other stages.

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
