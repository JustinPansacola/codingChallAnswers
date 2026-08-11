"""Stage 3: read an inbox and some calendars, and reconcile them.

Cheap stage, same rule as stage 1: keep any future imports inside evaluate().
"""

from mcp.server.fastmcp import FastMCP

from server.transport import PUBLIC_TRANSPORT_SECURITY

mcp = FastMCP("stage3", transport_security=PUBLIC_TRANSPORT_SECURITY)


@mcp.tool()
def evaluate(input: str = "") -> str:
    """Stub. Replace with real stage 3 logic (inbox/calendar reconciliation)."""
    return "stage3: not implemented yet"
