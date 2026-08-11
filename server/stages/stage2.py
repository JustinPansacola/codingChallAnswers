"""Stage 2: find the right passages in a small set of documents, solve a
shortest path.

The costly stage - the doc says the ML *libraries* (10x) dominate memory
over the model itself. Whatever indexing/embedding library gets used here,
import it inside evaluate() (or a lazily-called setup function), not at
module scope, so:
  - stages 1 and 3 never pay for it, and
  - the cost is paid once at first use and can be warmed up deliberately
    (e.g. from a startup hook) rather than surprising the first graded
    request's 10-second budget.
"""

from mcp.server.fastmcp import FastMCP

from server.transport import PUBLIC_TRANSPORT_SECURITY

mcp = FastMCP("stage2", transport_security=PUBLIC_TRANSPORT_SECURITY)


@mcp.tool()
def evaluate(input: str = "") -> str:
    """Stub. Replace with real stage 2 logic (passage retrieval + shortest path)."""
    return "stage2: not implemented yet"
