"""Combined ASGI app: three independent MCP servers, one per stage, each
mounted at the exact path the grader calls.

    POST /mcp         -> stage1 MCP server, tools: get_name, calculate,
                          classify_shape, shape_total
    POST /2/evaluate  -> stage2 MCP server, tool "evaluate"
    POST /3/evaluate  -> stage3 MCP server, tool "evaluate"
    GET  /health       -> plain liveness check, no MCP/heavy imports involved

Each stage keeps its own FastMCP instance (see server/stages/*.py) so a
future heavy import in one stage's evaluate() never touches the others.
"""

import os
from contextlib import AsyncExitStack, asynccontextmanager

from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from server.stages import stage1, stage2, stage3

STAGES = {
    "/mcp": stage1.mcp,
    "/2/evaluate": stage2.mcp,
    "/3/evaluate": stage3.mcp,
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


routes = [
    Route(path, endpoint=StreamableHTTPASGIApp(server.session_manager), methods=["GET", "POST", "DELETE"])
    for path, server in STAGES.items()
]
routes.append(Route("/health", endpoint=health))

app = Starlette(lifespan=lifespan, routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
