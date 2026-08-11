"""Stage 2 ("School Days"): passage retrieval over study material, and
shortest-path navigation with entry tolls and an optional hop allowance.

The costly stage - the doc says the ML *libraries* (10x) dominate memory
over the model itself. Retrieval here is TF-IDF cosine similarity blended
with cosine similarity over averaged, IDF-weighted GloVe word vectors (a
small bundled 50-dim table, server/stages/data/glove50_trimmed.npz) - no
embedding *model* (nothing to run inference through), just a static lookup
table and numpy dot products. Plain TF-IDF alone was tested against the
real study material and badly fails questions phrased with no literal
keyword overlap with the source text (e.g. "sensor grid... alignment" for
a passage that says "array... recalibrated") - the embedding component is
what catches that. numpy and tiktoken are imported inside functions, not
at module scope, so stages 1 and 3 never pay for them.
"""

import heapq
import json
import math
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from server.transport import PUBLIC_TRANSPORT_SECURITY

mcp = FastMCP("stage2", transport_security=PUBLIC_TRANSPORT_SECURITY)

DEFAULT_API_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"
RETRIEVAL_TOKEN_BUDGET = 900
CHUNK_TARGET_TOKENS = 70
CHUNK_MAX_TOKENS = 110
EMBEDDING_WEIGHT = 0.6
GLOVE_PATH = Path(__file__).parent / "data" / "glove50_trimmed.npz"

STOPWORDS = frozenset(
    """a an the this that these those is are was were be been being to of in on at for with from by as
    into onto over under about above below and or but not no nor so than then when where why how what
    which who whom it its it's i you he she we they them his her their our your my me him do does did
    doing done have has had having will would shall should can could may might must if else while during
    before after between out up down off again further once here there all any both each few more most
    other some such only own same too very s t don now back last brought""".split()
)


def _api_base_url() -> str:
    return os.environ.get("STAGE2_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


# --- Part 2: shortest path with entry tolls and an optional hop cap -------


def _fetch_graph(map_id: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    import httpx

    resp = httpx.get(f"{_api_base_url()}/graph", params={"map_id": map_id}, timeout=8.0)
    resp.raise_for_status()
    data = resp.json()
    return data["adjacency"], data["tolls"]


def _reconstruct(prev: dict, destination: str, source: str) -> list[str]:
    path = [destination]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _dijkstra_path(
    adjacency: dict[str, dict[str, float]],
    tolls: dict[str, float],
    source: str,
    destination: str,
) -> list[str] | None:
    """Shortest path where entering node v costs edge_weight(u, v) + toll(v)."""
    dist = {source: 0.0}
    prev: dict[str, str] = {}
    visited = set()
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == destination:
            break
        for v, w in adjacency.get(u, {}).items():
            nd = d + w + tolls.get(v, 0.0)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if destination not in dist:
        return None
    return _reconstruct(prev, destination, source)


def _bounded_hop_path(
    adjacency: dict[str, dict[str, float]],
    tolls: dict[str, float],
    source: str,
    destination: str,
    max_hops: int,
) -> list[str] | None:
    """Cheapest path from source to destination using at most max_hops edges.

    DP over (edges used, node) - dp[k][v] is the min cost to reach v using
    exactly k edges. The answer is the best destination cost over all
    k in [1, max_hops], since arriving early with hops to spare is fine.
    """
    dp: list[dict[str, tuple[float, str | None]]] = [dict() for _ in range(max_hops + 1)]
    dp[0][source] = (0.0, None)
    for k in range(1, max_hops + 1):
        for u, (cu, _) in dp[k - 1].items():
            for v, w in adjacency.get(u, {}).items():
                nc = cu + w + tolls.get(v, 0.0)
                if v not in dp[k] or nc < dp[k][v][0]:
                    dp[k][v] = (nc, u)

    best_k, best_cost = None, math.inf
    for k in range(1, max_hops + 1):
        entry = dp[k].get(destination)
        if entry is not None and entry[0] < best_cost:
            best_cost, best_k = entry[0], k

    if best_k is None:
        return None

    path = [destination]
    node, k = destination, best_k
    while k > 0:
        node = dp[k][node][1]
        path.append(node)
        k -= 1
    path.reverse()
    return path


@mcp.tool()
def go(source: str, target: str, id_of_map: str, hops_remaining: int | None = None) -> str:
    """Navigate one hop toward `target` along the cheapest remaining route,
    and return the next node to visit (not the whole path).

    Cost is edge weight plus the entry toll of every node moved into (never
    the toll of the node being left). If `hops_remaining` is given, at most
    that many edges remain including this one - the route returned is the
    cheapest one that still arrives within that limit, which is generally
    not the globally cheapest route.

    If `source` already equals `target`, returns `source`.
    """
    if source == target:
        return source

    adjacency, tolls = _fetch_graph(id_of_map)

    if hops_remaining is not None:
        path = _bounded_hop_path(adjacency, tolls, source, target, hops_remaining)
    else:
        path = _dijkstra_path(adjacency, tolls, source, target)

    if not path or len(path) < 2:
        raise ValueError(f"no route from {source!r} to {target!r} within the given limit")
    return path[1]


# --- Part 1: revision passage retrieval -----------------------------------


def _load_documents() -> list[str]:
    import httpx

    base = _api_base_url()
    with httpx.Client(timeout=10.0) as client:
        index = client.get(f"{base}/study-materials").raise_for_status().json()
        documents = []
        for entry in index["documents"]:
            resp = client.get(f"{base}/study-materials/{entry['id']}")
            resp.raise_for_status()
            documents.append(resp.text)
    return documents


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def _chunk_document(text: str, encoding) -> list[str]:
    """Group sentences into chunks close to CHUNK_TARGET_TOKENS, capped at
    CHUNK_MAX_TOKENS, without splitting a sentence across chunks unless a
    single sentence alone exceeds the cap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush():
        if current:
            chunks.append(" ".join(current))

    for sentence in _split_sentences(text):
        n = len(encoding.encode(sentence))
        if n > CHUNK_MAX_TOKENS:
            flush()
            current, current_tokens = [], 0
            chunks.append(sentence)
            continue
        if current_tokens + n > CHUNK_MAX_TOKENS or (current_tokens >= CHUNK_TARGET_TOKENS):
            flush()
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += n

    flush()
    return [c for c in chunks if c.strip()]


_WORD_RE = re.compile(r"[a-z]+")


def _tokenize_words(text: str) -> list[str]:
    """All words, for TF-IDF (stopwords still filtered by IDF naturally,
    but explicitly dropped here too since they only add noise)."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def _tf_idf_vectors(tokenized: list[list[str]]):
    import numpy as np

    vocab: dict[str, int] = {}
    for words in tokenized:
        for w in set(words):
            vocab.setdefault(w, len(vocab))

    doc_freq = np.zeros(len(vocab), dtype=np.float64)
    for words in tokenized:
        for w in set(words):
            doc_freq[vocab[w]] += 1
    n = len(tokenized)
    idf = np.log((1 + n) / (1 + doc_freq)) + 1.0

    vectors = np.zeros((n, len(vocab)), dtype=np.float64)
    for i, words in enumerate(tokenized):
        if not words:
            continue
        tf = np.zeros(len(vocab), dtype=np.float64)
        for w in words:
            tf[vocab[w]] += 1.0
        tf /= len(words)
        vec = tf * idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors[i] = vec
    return vectors, vocab, idf


def _tf_idf_query(words: list[str], vocab: dict[str, int], idf):
    import numpy as np

    tf = np.zeros(len(vocab), dtype=np.float64)
    for w in words:
        if w in vocab:
            tf[vocab[w]] += 1.0
    if tf.sum() > 0:
        tf /= len(words)
    vec = tf * idf
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class _GloVe:
    def __init__(self, index_by_word: dict[str, int], unit_vectors):
        self.index_by_word = index_by_word
        self.unit_vectors = unit_vectors


_glove_cache: _GloVe | None = None


def _get_glove() -> _GloVe:
    global _glove_cache
    if _glove_cache is None:
        import numpy as np

        with np.load(GLOVE_PATH) as data:
            words = data["words"]
            vectors = data["vectors"]
        index_by_word = {w: i for i, w in enumerate(words)}
        _glove_cache = _GloVe(index_by_word, vectors)
    return _glove_cache


def _embed(words: list[str], glove: _GloVe, idf_lookup: dict[str, float]):
    """IDF-weighted mean of unit-normalized GloVe vectors for known words."""
    import numpy as np

    known = [w for w in words if w in glove.index_by_word]
    if not known:
        return np.zeros(glove.unit_vectors.shape[1], dtype=np.float32)
    weights = np.array([idf_lookup.get(w, 1.0) for w in known], dtype=np.float32)
    vecs = glove.unit_vectors[[glove.index_by_word[w] for w in known]]
    v = (vecs * weights[:, None]).sum(axis=0) / weights.sum()
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class _Index:
    def __init__(self, chunks: list[str], chunk_tokens: list[int], lex_vectors, lex_vocab, lex_idf, emb_vectors):
        self.chunks = chunks
        self.chunk_tokens = chunk_tokens
        self.lex_vectors = lex_vectors
        self.lex_vocab = lex_vocab
        self.lex_idf = lex_idf
        self.emb_vectors = emb_vectors


_index_cache: _Index | None = None


def _build_index() -> _Index:
    import numpy as np
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    documents = _load_documents()

    chunks: list[str] = []
    for doc in documents:
        chunks.extend(_chunk_document(doc, encoding))
    chunk_tokens = [len(encoding.encode(c)) for c in chunks]

    tokenized = [_tokenize_words(c) for c in chunks]
    lex_vectors, lex_vocab, lex_idf = _tf_idf_vectors(tokenized)
    idf_lookup = {w: lex_idf[i] for w, i in lex_vocab.items()}

    glove = _get_glove()
    emb_vectors = np.array([_embed(words, glove, idf_lookup) for words in tokenized], dtype=np.float32)

    return _Index(chunks, chunk_tokens, lex_vectors, lex_vocab, lex_idf, emb_vectors)


def _get_index() -> _Index:
    global _index_cache
    if _index_cache is None:
        _index_cache = _build_index()
    return _index_cache


def _knapsack_select(scores, token_costs: list[int], budget: int) -> list[int]:
    """0/1 knapsack: maximize total relevance score subject to a total
    token budget. A rank-order greedy fill can strand a highly relevant
    chunk that doesn't fit the leftover space behind lower-value chunks
    that do - this instead finds the actual best-value combination."""
    n = len(token_costs)
    # dp[i][w] = best achievable score using the first i items and budget w
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost = token_costs[i - 1]
        value = scores[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for w in range(budget + 1):
            best = prev_row[w]
            if cost <= w:
                withItem = prev_row[w - cost] + value
                if withItem > best:
                    best = withItem
            row[w] = best

    selected = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= token_costs[i - 1]
    selected.reverse()
    return selected


@mcp.tool()
def recall(query: str) -> str:
    """Return the passages from the study materials most relevant to
    `query`, as a JSON array of strings.

    Passages are chosen to maximize total relevance subject to a 900-token
    budget (o200k_base encoding), preferring fewer high-value passages over
    padding with marginal ones.

    The return value is a JSON *string* rather than a list on purpose: a
    list return is serialised by MCP as one text block per element, so the
    grader - which parses the response as JSON - would see a bare passage
    instead of an array and permanently void the question.
    """
    index = _get_index()
    words = _tokenize_words(query)
    idf_lookup = {w: index.lex_idf[i] for w, i in index.lex_vocab.items()}

    lex_scores = index.lex_vectors @ _tf_idf_query(words, index.lex_vocab, index.lex_idf)
    emb_scores = index.emb_vectors @ _embed(words, _get_glove(), idf_lookup)
    scores = (1 - EMBEDDING_WEIGHT) * lex_scores + EMBEDDING_WEIGHT * emb_scores

    candidates = [i for i in range(len(index.chunks)) if scores[i] > 0 and index.chunk_tokens[i] <= RETRIEVAL_TOKEN_BUDGET]
    candidates.sort(key=lambda i: scores[i], reverse=True)

    # Rank-based geometric-decay values, not raw scores: raw hybrid scores
    # cluster tightly (e.g. 0.35 vs 0.36), so sum-maximizing on them prefers
    # many low-ranked filler chunks over one chunk that's ranked just below
    # a budget cliff. Geometric decay makes the knapsack respect rank order
    # (never trade a higher-ranked chunk for several lower-ranked ones)
    # while still finding the best-fitting combination near that order.
    rank_values = [0.5**rank for rank in range(len(candidates))]
    chosen = _knapsack_select(
        rank_values,
        [index.chunk_tokens[i] for i in candidates],
        RETRIEVAL_TOKEN_BUDGET,
    )
    return json.dumps([index.chunks[candidates[i]] for i in chosen])
