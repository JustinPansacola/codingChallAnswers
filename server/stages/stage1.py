"""Stage 1 ("The Nursery"): a name, basic arithmetic, and shape recognition.

Cheap stage - no heavy imports at module scope. PIL/numpy are only imported
inside the shape-classification helper, so importing this module (which
happens at process startup via server/app.py) stays fast even before any
shape tool is actually called.
"""

import ast
import base64
import io
import math
import operator
import re

from pydantic import BaseModel

from mcp.server.fastmcp import FastMCP

from server.transport import PUBLIC_TRANSPORT_SECURITY

mcp = FastMCP("stage1", transport_security=PUBLIC_TRANSPORT_SECURITY)

NAME = "JustinMCP"


# --- arithmetic -------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression node: {ast.dump(node)}")


def _safe_arithmetic(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse expression: {expression!r}") from e
    return _eval_node(tree)


# --- shapes -------------------------------------------------------------

_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.*)$", re.DOTALL)


def _decode_image_bytes(image_base64: str) -> bytes:
    data = image_base64.strip()
    match = _DATA_URI_RE.match(data)
    if match:
        data = match.group(1)
    try:
        return base64.b64decode(data)
    except Exception as e:
        raise ValueError("could not decode base64 image") from e


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list) -> list:
    """Andrew's monotone chain. Points are (x, y) tuples."""
    points = sorted(set(points))
    if len(points) <= 2:
        return points
    lower: list = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _perp_dist(p: tuple, a: tuple, b: tuple) -> float:
    if a == b:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    num = abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]))
    den = math.hypot(b[0] - a[0], b[1] - a[1])
    return num / den


def _rdp(points: list, epsilon: float) -> list:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    idx, dmax = max(
        ((i, _perp_dist(p, start, end)) for i, p in enumerate(points[1:-1], 1)),
        key=lambda t: t[1],
        default=(None, 0.0),
    )
    if dmax > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [start, end]


def _classify_shape(image_base64: str) -> str:
    from PIL import Image
    import numpy as np

    image_bytes = _decode_image_bytes(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.asarray(img, dtype=np.float64)

    h, w = arr.shape
    background = float(np.median([arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]]))

    diff = np.abs(arr - background)
    threshold = max(20.0, float(diff.max()) * 0.25) if diff.max() > 0 else 20.0
    mask = diff > threshold

    if not mask.any():
        raise ValueError("no shape detected in image")

    ys, xs = np.nonzero(mask)
    hull = _convex_hull(list(zip(xs.tolist(), ys.tolist())))
    if len(hull) < 3:
        raise ValueError("shape outline is degenerate")

    closed = hull + [hull[0]]
    perimeter = sum(
        math.hypot(closed[i + 1][0] - closed[i][0], closed[i + 1][1] - closed[i][1])
        for i in range(len(closed) - 1)
    )
    simplified = _rdp(closed, epsilon=0.02 * perimeter)
    num_vertices = len(simplified) - 1

    # Convex-hull vertex count, rotation-invariant: a triangle's hull has 3
    # corners, a rectangle's (incl. rotated squares/rhombi/trapezoids) has 4
    # give or take pixelation noise, and a circle's stays polygon-like under
    # simplification only when it keeps many corners.
    if num_vertices <= 3:
        return "triangle"
    if num_vertices <= 6:
        return "rectangle"
    return "circle"


class ShapeGroup(BaseModel):
    image_base64: str
    count: float
    value: float | None = None


# --- tools ----------------------------------------------------------------


@mcp.tool()
def get_name() -> str:
    """Return this android's name."""
    return NAME


@mcp.tool()
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Supports +, -, *, /, unary minus, and parentheses, with standard
    operator precedence (multiplication/division before addition/
    subtraction) - so "2 + 3 * 4" correctly evaluates to 14, not 20.
    """
    return _safe_arithmetic(expression)


@mcp.tool()
def classify_shape(image_base64: str) -> str:
    """Classify a base64-encoded PNG image of a single 2D shape.

    Returns exactly one of: "rectangle", "triangle", "circle". Any
    four-sided figure (square, rhombus, trapezoid, ...) counts as a
    "rectangle"; the only distinction made is four-sided vs three-sided
    vs round.
    """
    return _classify_shape(image_base64)


@mcp.tool()
def shape_total(shapes: list[ShapeGroup], value_by_shape: dict[str, float] | None = None) -> float:
    """Total up several groups of shapes.

    `shapes` is a list of groups, each with a base64 PNG image, a count
    of how many of that shape there are, and optionally the point value
    for that group. If a group omits `value`, its image is classified
    (rectangle/triangle/circle) and its value is looked up from
    `value_by_shape` instead. Returns sum(count * value) across all
    groups.
    """
    total = 0.0
    for group in shapes:
        value = group.value
        if value is None:
            if not value_by_shape:
                raise ValueError("group has no value and no value_by_shape table was given")
            shape = _classify_shape(group.image_base64)
            if shape not in value_by_shape:
                raise ValueError(f"no value given for shape {shape!r}")
            value = value_by_shape[shape]
        total += group.count * value
    return total
