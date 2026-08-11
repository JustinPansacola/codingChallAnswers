"""Combined ASGI app: three independent MCP servers, one per stage, each
mounted at the exact path the grader calls.

    POST /1/evaluate/mcp  -> stage1 MCP server, tools: get_name, do_arithmetic,
                             resolve_whole_expr, classify_shape_from_base64,
                             shape_total
    POST /2/evaluate/mcp  -> stage2 MCP server, tools: go, recall
    POST /3/evaluate/mcp  -> stage3 MCP server, tool: find_earliest_free_window
    GET  /health           -> plain liveness check, no MCP/heavy imports involved

Each stage keeps its own FastMCP instance (see server/stages/*.py) so a
future heavy import in one stage's evaluate() never touches the others.
"""

import json
import os
from contextlib import AsyncExitStack, asynccontextmanager

from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from server.stages import stage1, stage2, stage3

STAGES = {
    "/1/evaluate/mcp": stage1.mcp,
    "/2/evaluate/mcp": stage2.mcp,
    "/3/evaluate/mcp": stage3.mcp,
}

# Force each FastMCP instance to lazily create its session manager now, at
# import time (i.e. before the port is bound), rather than on first request.
for path, server in STAGES.items():
    server.streamable_http_app()


@asynccontextmanager
async def lifespan(app: Starlette):
    async with AsyncExitStack() as stack:
        for server in STAGES.values():
            await stack.enter_async_context(server.session_manager.run())
        yield


async def health(request):
    return JSONResponse({"ok": True})


async def event(request):
    """Grader telemetry sink: one POST per tool call while a run is in
    progress, naming the problem and attempt. Purely diagnostic - it cannot
    affect scoring, and the grader swallows any error raised here - but it
    is the only channel that says *why* a run scored what it did, so it is
    logged to stdout where the platform's log viewer will show it.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {"unparseable_body": True}
    print(f"[event] {json.dumps(payload, default=str)}", flush=True)
    return JSONResponse({"ok": True})


routes = [
    Route(path, endpoint=StreamableHTTPASGIApp(server.session_manager), methods=["GET", "POST", "DELETE"])
    for path, server in STAGES.items()
]
routes.append(Route("/health", endpoint=health))
routes.append(Route("/event", endpoint=event, methods=["POST"]))

app = Starlette(lifespan=lifespan, routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
