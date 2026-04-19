"""Skynet Memory MCP server.

HTTP/JSON MCP-ish wrapper around two in-cluster diagnostic surfaces:

  * ``skynet-identity`` — ``POST /context?debug=true`` returns a
    ``score_breakdown`` (vec / access / recency / graph / strike /
    logical_decay / compression / clique / cross_clique / final) per
    retrieved point. See Phase 13 of the RAG roadmap. Used to answer
    "why didn't my memory come back?".
  * ``skynet-rag-atlas`` — ``GET /panel/compression/histogram`` and
    ``GET /panel/pressure/heatmap`` give the current shape of memory
    compression + per-clique eviction pressure.

Primary consumers are LLM agents (Claude Code sessions, the Skynet
agent itself), NOT humans. The tools prune raw responses aggressively
so the diagnostic fits in an LLM context window.

Transport: plain HTTP/JSON, identical contract to skynet-grafana-mcp
(``GET /tools`` manifest + ``POST /tools/{name}/call``). Can also be
proxied into Claude Code via stdio — see README.

Environment variables:
  IDENTITY_URL        - default http://skynet-identity.skynet-identity.svc:8080
  ATLAS_URL           - default http://skynet-rag-atlas.skynet-rag-atlas.svc:8080
  REQUEST_TIMEOUT     - upstream request timeout seconds (default: 45)
  MAX_ITEMS_PER_BUCKET - trim score_breakdown to top-N per bucket (default: 3)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from skynet_core.logging import setup_logging
from skynet_core.tracing import setup_tracing
from skynet_mcp import ToolRegistry, mount_mcp
from skynet_qdrant import AsyncQdrantClient

log = setup_logging("memory-mcp")

setup_tracing("memory-mcp", instrumentors=["fastapi", "httpx"])

IDENTITY_URL = os.getenv(
    "IDENTITY_URL", "http://skynet-identity.skynet-identity.svc:8080"
).rstrip("/")
ATLAS_URL = os.getenv(
    "ATLAS_URL", "http://skynet-rag-atlas.skynet-rag-atlas.svc:8080"
).rstrip("/")
QDRANT_URL = os.getenv(
    "QDRANT_URL", "http://qdrant-headless.qdrant.svc:6333"
).rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "45"))
MAX_ITEMS_PER_BUCKET = int(os.getenv("MAX_ITEMS_PER_BUCKET", "3"))

# Collections the temporal tools see by default. user_profile_raw holds
# imported data (gemini, phone, git history, claude sessions) — where
# date-anchored questions usually find their answer. skynet_episodic
# holds live Matrix/task memory. skynet_knowledge is excluded because
# it's deprecated (see project_skynet_knowledge_deleted.md).
DEFAULT_TEMPORAL_COLLECTIONS = ["user_profile_raw", "skynet_episodic"]
# Hard cap on points returned per tool call — temporal queries can easily
# page through thousands of points, and LLM context isn't cheap. Tune
# via env if a caller legitimately needs more.
TEMPORAL_MAX_RESULTS = int(os.getenv("TEMPORAL_MAX_RESULTS", "50"))

app = FastAPI(title="skynet-memory-mcp", version="0.1.0")
registry = ToolRegistry()

_identity_client: httpx.Client | None = None
_atlas_client: httpx.Client | None = None
_qdrant_client: AsyncQdrantClient | None = None


def _identity() -> httpx.Client:
    global _identity_client
    if _identity_client is None:
        _identity_client = httpx.Client(base_url=IDENTITY_URL, timeout=REQUEST_TIMEOUT)
    return _identity_client


def _atlas() -> httpx.Client:
    global _atlas_client
    if _atlas_client is None:
        _atlas_client = httpx.Client(base_url=ATLAS_URL, timeout=REQUEST_TIMEOUT)
    return _atlas_client


def _qdrant() -> AsyncQdrantClient:
    """Lazily-instantiated shared Qdrant client.

    Temporal tools use :meth:`AsyncQdrantClient.scroll` / ``count`` (via
    ``skynet_qdrant``) for everything they need. The lib gives us an
    httpx.AsyncClient underneath pooled across calls.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=REQUEST_TIMEOUT)
    return _qdrant_client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "identity": IDENTITY_URL,
        "atlas": ATLAS_URL,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness = identity reachable. Atlas is a bonus — if only
    compression stats are unavailable we still serve the primary
    ``retrieval_debug`` tool."""
    try:
        r = _identity().get("/health")
        if r.status_code != 200:
            raise HTTPException(status_code=503, detail=f"identity /health {r.status_code}")
        return {"status": "ok"}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"identity unreachable: {e}") from e


# ---------------------------------------------------------------------------
# Tool: retrieval_debug
# ---------------------------------------------------------------------------

class RetrievalDebugArgs(BaseModel):
    query: str = Field(..., description=(
        "The user query to replay through identity's /context endpoint. "
        "Exactly what the user typed — DON'T paraphrase, the embedding is "
        "what's being diagnosed."
    ))
    anchor: Optional[str] = Field(None, description=(
        "Optional conversational anchor: the last sentence of the previous "
        "assistant turn. Include it when diagnosing short follow-ups "
        "(\"so what?\", \"тоді як?\") — without an anchor the retrieval will "
        "look artificially bad on ambiguous queries."
    ))
    session_id: Optional[str] = Field(None, description=(
        "Optional session id to replay. Affects session_context loading "
        "only, does NOT change scoring of episodic/knowledge/raw buckets."
    ))
    top_k: int = Field(10, ge=1, le=50, description=(
        "How many candidates per bucket to request from identity. Default "
        "matches production."
    ))


def _prune_bucket(items: list[dict]) -> list[dict]:
    """Trim a score_breakdown bucket to the top MAX_ITEMS_PER_BUCKET
    candidates by final_score, dropping anything obviously redundant.
    We keep the raw component fields (vec, access_boost, recency_boost,
    graph_boost, raw_score, strike_penalty, logical_decay,
    compression_boost, clique_boost, cross_clique_boost, final_score)
    because those ARE the diagnostic — without them this tool is
    useless."""
    sorted_items = sorted(
        items, key=lambda x: float(x.get("final_score", 0.0)), reverse=True
    )
    return sorted_items[:MAX_ITEMS_PER_BUCKET]


def _previews_by_id(bucket: list[dict]) -> dict[str, str]:
    """Build {id: short_preview} lookup so we can annotate score
    breakdowns with human-readable content. Prevents the LLM from
    having to cross-reference uuids against the separate
    episodic/knowledge/raw arrays."""
    out: dict[str, str] = {}
    for item in bucket:
        pid = str(item.get("id") or item.get("point_id") or "")
        if not pid:
            continue
        txt = (
            item.get("text")
            or item.get("summary")
            or item.get("content")
            or item.get("body")
            or ""
        )
        if isinstance(txt, str) and txt:
            out[pid] = txt[:200]
    return out


@registry.tool(
    name="retrieval_debug",
    description=(
        "Replay a query through skynet-identity with debug=True and "
        "return a pruned view of per-candidate score components "
        "(vec, access, recency, graph, logical_decay, compression_boost, "
        "clique_boost, ...). Use this whenever a user reports "
        "\"Skynet doesn't remember X\" or \"the reply referenced the "
        "wrong memory\". Answers the question: why did this candidate "
        "win/lose? Returns top 3 per bucket (episodic, knowledge, raw) "
        "plus a one-paragraph diagnosis."
    ),
    schema=RetrievalDebugArgs.model_json_schema(),
)
def _tool_retrieval_debug(**kwargs: Any) -> dict[str, Any]:
    args = RetrievalDebugArgs(**kwargs)
    payload: dict[str, Any] = {
        "query": args.query,
        "debug": True,
        "top_k": args.top_k,
    }
    if args.anchor:
        payload["anchor"] = args.anchor
    if args.session_id:
        payload["session_id"] = args.session_id

    r = _identity().post("/context", json=payload)
    r.raise_for_status()
    body = r.json()

    score_breakdown = body.get("score_breakdown") or {}
    if not score_breakdown:
        return {
            "warning": (
                "identity returned no score_breakdown — either the "
                "deployed image pre-dates Phase 13 or debug=True was "
                "silently ignored. Check identity version."
            ),
            "stats": body.get("stats", {}),
        }

    # Build id->preview lookup from the full-content buckets so the
    # agent sees what each scored point actually contains.
    previews: dict[str, str] = {}
    for key in ("episodic_memories", "knowledge", "raw_memories"):
        previews.update(_previews_by_id(body.get(key, []) or []))

    def _enrich(items: list[dict]) -> list[dict]:
        out = []
        for it in _prune_bucket(items or []):
            enriched = dict(it)
            pid = str(it.get("id", ""))
            if pid in previews:
                enriched["preview"] = previews[pid]
            out.append(enriched)
        return out

    pruned = {
        "episodic": _enrich(score_breakdown.get("episodic", [])),
        "knowledge": _enrich(score_breakdown.get("knowledge", [])),
        "raw": _enrich(score_breakdown.get("raw", [])),
    }

    # Summary hints computed from the pruned view — makes diagnosis
    # scannable without the LLM having to re-digest every number.
    def _all_items() -> list[dict]:
        return pruned["episodic"] + pruned["knowledge"] + pruned["raw"]

    diagnosis: list[str] = []
    items = _all_items()
    if items:
        low_vec = [i for i in items if float(i.get("vec", 1.0)) < 0.5]
        if low_vec:
            diagnosis.append(
                f"{len(low_vec)}/{len(items)} top candidates have vec<0.5 "
                "— embedding mismatch; try different phrasing or check "
                "that the embedding model hasn't changed."
            )
        decaying = [i for i in items if float(i.get("logical_decay", 1.0)) < 0.9]
        if decaying:
            diagnosis.append(
                f"{len(decaying)}/{len(items)} candidates have "
                "logical_decay<0.9 — points are being punished for "
                "being missed; consolidation may be due."
            )
        lifted = [i for i in items if float(i.get("compression_boost", 1.0)) > 1.2]
        if lifted:
            diagnosis.append(
                f"{len(lifted)}/{len(items)} got compression_boost>1.2 "
                "— short query is pulling summaries ahead of raw points."
            )
        no_clique = [i for i in items if not i.get("clique_boost")]
        if no_clique and args.anchor:
            diagnosis.append(
                f"{len(no_clique)}/{len(items)} have no clique_boost "
                "despite an anchor — anchor may not resolve to a "
                "qualified clique in this collection."
            )

    return {
        "query": args.query,
        "anchor_supplied": bool(args.anchor),
        "candidates_used": score_breakdown.get("candidates_used"),
        "query_word_count": score_breakdown.get("query_word_count"),
        "score_breakdown": pruned,
        "diagnosis": diagnosis,
        "stats": body.get("stats", {}),
        "hint": (
            "Component glossary: vec=cosine similarity. access_boost, "
            "recency_boost, graph_boost, compression_boost, "
            "clique_boost, cross_clique_boost are multiplicative "
            "(1.0 = no effect). strike_penalty and logical_decay are "
            "multiplicative downweights (<1.0 = penalty). "
            "final_score is the product fed to top-K selection."
        ),
    }


# ---------------------------------------------------------------------------
# Tool: list_compression_stats
# ---------------------------------------------------------------------------

class ListCompressionStatsArgs(BaseModel):
    collection: str = Field(
        "user_profile_raw",
        description=(
            "Qdrant collection name. Defaults to user_profile_raw "
            "(what end-user memory lives in). Other options: "
            "skynet_episodic, skynet_knowledge (likely empty — see "
            "memory/project_skynet_knowledge_deleted.md)."
        ),
    )


@registry.tool(
    name="list_compression_stats",
    description=(
        "Return a compact summary of memory compression health for a "
        "Qdrant collection: compression_level histogram (are we "
        "consolidating enough?) + cliques under eviction pressure. "
        "Use this when asking \"is memory healthy?\" or before "
        "triggering a manual consolidation DAG."
    ),
    schema=ListCompressionStatsArgs.model_json_schema(),
)
def _tool_list_compression_stats(**kwargs: Any) -> dict[str, Any]:
    args = ListCompressionStatsArgs(**kwargs)
    out: dict[str, Any] = {"collection": args.collection}

    try:
        r = _atlas().get(
            "/panel/compression/histogram",
            params={"collection": args.collection},
        )
        r.raise_for_status()
        hist = r.json()
        # Compact — histogram bins only, drop UI rendering fields.
        bins = hist.get("bins") or hist.get("histogram") or hist
        out["compression_histogram"] = bins
    except httpx.HTTPError as e:
        out["compression_histogram_error"] = str(e)

    try:
        r = _atlas().get(
            "/panel/pressure/heatmap",
            params={"collection": args.collection},
        )
        r.raise_for_status()
        heat = r.json()
        # Keep only cliques with pressure > 0 — no-pressure rows are noise.
        cells = heat.get("cells") or heat.get("rows") or []
        if isinstance(cells, list):
            cells = [
                c for c in cells
                if float(c.get("pressure", c.get("value", 0)) or 0) > 0
            ]
            cells.sort(
                key=lambda c: float(c.get("pressure", c.get("value", 0)) or 0),
                reverse=True,
            )
            out["pressure_heatmap"] = {
                "top_cliques_under_pressure": cells[:20],
                "total_under_pressure": len(cells),
            }
        else:
            out["pressure_heatmap"] = heat
    except httpx.HTTPError as e:
        out["pressure_heatmap_error"] = str(e)

    out["hint"] = (
        "compression_histogram: distribution of compression_level values "
        "in this collection (0=raw, 1=episodic-ish, 2=knowledge-ish, "
        "3+=wiki-ish). A long raw tail means consolidation is "
        "under-running. pressure_heatmap: cliques approaching the "
        "eviction threshold under new-point pressure — these will be "
        "consolidated next."
    )
    return out


# ---------------------------------------------------------------------------
# Temporal tools: memory_by_date / memory_recent / memory_first
# ---------------------------------------------------------------------------
#
# These tools bypass identity's vector scoring and query Qdrant directly
# for time-anchored lookups. The LLM supplies absolute ISO dates it
# parsed from the user's natural language ("що було 15 березня" at
# today=2026-04-15 → after=2026-03-15, before=2026-03-16). That keeps
# language parsing in the model where it's native, and the tool stays
# dumb + deterministic.
#
# Qdrant range filter applies to numeric fields only, so we scroll with
# a coarse must-filter (event_timestamp exists, not archived) and apply
# the ISO string comparison in Python. ISO-8601 sorts lexically, so a
# direct string compare is equivalent to a datetime compare without any
# parsing — that's why event_timestamp is stored as a string everywhere.


async def _scroll_temporal(
    collection: str,
    event_after: Optional[str],
    event_before: Optional[str],
    limit: int = 200,
) -> list[dict]:
    """Scroll one collection for points whose event_timestamp falls in
    the given range. Uses Qdrant's datetime payload index — range
    filters are applied server-side in O(log n), not scanned via
    Python compare in a scroll loop.

    Two passes per collection because a point may carry event_timestamp
    (real event time) OR only the ingestion `timestamp` (for older
    imports where event time was lost). The second pass widens
    coverage for pre-backfill data at the cost of lower confidence —
    each result flags which timestamp it matched on so the LLM can
    decide whether to trust the date.

    `limit` bounds pages returned per-collection. Production callers
    set limit to the final output cap; the tool never returns more
    than that per collection so nothing is scanned needlessly.
    """
    out: list[dict] = []

    # Base filter — archived / superseded / gemini-imports are all
    # excluded from the main episodic/raw pool; we don't want to page
    # through them. Adding the range clauses makes this an indexed
    # lookup rather than a full scan.
    base_must_not: list[dict] = [
        {"key": "archived", "match": {"value": True}},
    ]
    base_must: list[dict] = [
        {"is_empty": {"key": "superseded_by"}},
    ]

    def _range_clause(field: str) -> dict:
        rng: dict[str, str] = {}
        if event_after:
            rng["gte"] = event_after
        if event_before:
            rng["lte"] = event_before
        return {"key": field, "range": rng}

    client = _qdrant()
    seen_ids: set[str] = set()

    # Pass 1: real event_timestamp — highest confidence.
    if event_after or event_before:
        points, _ = await client.scroll(
            collection,
            limit=limit,
            filter={
                "must": base_must + [_range_clause("event_timestamp")],
                "must_not": base_must_not,
            },
            with_payload=True,
            with_vector=False,
        )
        for p in points:
            pid = str(p.get("id"))
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            pl = p.get("payload") or {}
            text = pl.get("text") or pl.get("action") or pl.get("summary") or ""
            out.append({
                "id": pid,
                "collection": collection,
                "event_timestamp": pl.get("event_timestamp"),
                "is_ingestion_timestamp": False,
                "source": pl.get("source"),
                "category": pl.get("category"),
                "text": (text if isinstance(text, str) else "")[:300],
            })

        # Pass 2: fallback on ingestion `timestamp` for points lacking
        # event_timestamp (common for google_takeout_gemini — 4140
        # points where source dates were dropped by the ingestion
        # pipeline pre-fix). Limit the extra rows so a date-range query
        # doesn't get flooded with low-confidence gemini imports.
        if len(out) < limit:
            points, _ = await client.scroll(
                collection,
                limit=limit - len(out),
                filter={
                    "must": base_must + [
                        _range_clause("timestamp"),
                        {"is_empty": {"key": "event_timestamp"}},
                    ],
                    "must_not": base_must_not,
                },
                with_payload=True,
                with_vector=False,
            )
            for p in points:
                pid = str(p.get("id"))
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                pl = p.get("payload") or {}
                text = pl.get("text") or pl.get("action") or pl.get("summary") or ""
                out.append({
                    "id": pid,
                    "collection": collection,
                    "event_timestamp": pl.get("timestamp"),
                    "is_ingestion_timestamp": True,
                    "source": pl.get("source"),
                    "category": pl.get("category"),
                    "text": (text if isinstance(text, str) else "")[:300],
                })
    else:
        # No range — used by memory_recent / memory_first. Scroll up to
        # limit ordered-by-index (Qdrant doesn't sort, but scroll order
        # is stable-per-collection so the caller can scan until limit
        # and the result is still deterministic).
        points, _ = await client.scroll(
            collection,
            limit=limit,
            filter={
                "must": base_must + [
                    {"key": "event_timestamp", "range": {"gte": "1970-01-01T00:00:00Z"}},
                ],
                "must_not": base_must_not,
            },
            with_payload=True,
            with_vector=False,
        )
        for p in points:
            pl = p.get("payload") or {}
            text = pl.get("text") or pl.get("action") or pl.get("summary") or ""
            out.append({
                "id": str(p.get("id")),
                "collection": collection,
                "event_timestamp": pl.get("event_timestamp"),
                "is_ingestion_timestamp": False,
                "source": pl.get("source"),
                "category": pl.get("category"),
                "text": (text if isinstance(text, str) else "")[:300],
            })

    return out


async def _count_in_range(
    collection: str, event_after: Optional[str], event_before: Optional[str]
) -> dict[str, int]:
    """Exact count per timestamp-source (event vs ingestion) in range."""
    rng: dict[str, str] = {}
    if event_after:
        rng["gte"] = event_after
    if event_before:
        rng["lte"] = event_before

    client = _qdrant()

    by_event = await client.count(
        collection,
        filter={
            "must": [
                {"is_empty": {"key": "superseded_by"}},
                {"key": "event_timestamp", "range": rng},
            ],
            "must_not": [{"key": "archived", "match": {"value": True}}],
        },
    )
    by_ingestion = await client.count(
        collection,
        filter={
            "must": [
                {"is_empty": {"key": "superseded_by"}},
                {"is_empty": {"key": "event_timestamp"}},
                {"key": "timestamp", "range": rng},
            ],
            "must_not": [{"key": "archived", "match": {"value": True}}],
        },
    )
    return {
        "event": by_event,
        "ingestion_fallback": by_ingestion,
        "total": by_event + by_ingestion,
    }


class MemoryByDateArgs(BaseModel):
    event_after: Optional[str] = Field(
        None,
        description=(
            "Inclusive lower bound on event_timestamp, ISO-8601 "
            "(e.g. \"2024-03-15T00:00:00\" or just \"2024-03-15\"). "
            "Parse natural language yourself — current date is available "
            "in your system context. Omit for open-ended past."
        ),
    )
    event_before: Optional[str] = Field(
        None,
        description=(
            "Inclusive upper bound on event_timestamp, ISO-8601. "
            "For a single-day query use after=\"2024-03-15\" and "
            "before=\"2024-03-16\" (range is [after, before] in ISO "
            "lexical order). Omit for open-ended future."
        ),
    )
    collections: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TEMPORAL_COLLECTIONS),
        description=(
            "Which Qdrant collections to scan. Defaults cover the two "
            "that store dated memory: user_profile_raw (imported data — "
            "gemini, phone, git, claude sessions) and skynet_episodic "
            "(live matrix/task memory)."
        ),
    )
    limit: int = Field(
        20, ge=1, le=TEMPORAL_MAX_RESULTS,
        description=(
            "Max points to return, newest first. The tool already hard-caps "
            "at TEMPORAL_MAX_RESULTS (50) to keep the response LLM-friendly; "
            "raise only if you need exhaustive coverage."
        ),
    )


@registry.tool(
    name="memory_by_date",
    description=(
        "Retrieve memory points whose event_timestamp falls in a "
        "date range, newest first. Use this for \"що було 15 "
        "березня?\", \"що я робив минулого тижня?\", \"мої коміти "
        "за квітень\" — any question where the user anchors on a "
        "specific time, not a topic. Parse the natural-language date "
        "YOURSELF (current date is in your system context) and pass "
        "ISO-8601 strings. Returns only points that are actually "
        "dated — see is_ingestion_timestamp on each result to tell "
        "\"known to have happened then\" from \"known to have been "
        "imported then\"."
    ),
    schema=MemoryByDateArgs.model_json_schema(),
)
async def _tool_memory_by_date(**kwargs: Any) -> dict[str, Any]:
    args = MemoryByDateArgs(**kwargs)
    if not args.event_after and not args.event_before:
        return {
            "warning": (
                "Neither event_after nor event_before was set — this "
                "would return EVERY dated point. Use memory_recent for "
                "chronological browsing; use this tool only when you "
                "have at least one bound."
            ),
            "results": [],
        }

    merged: list[dict] = []
    errors: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for col in args.collections:
        try:
            merged.extend(
                await _scroll_temporal(
                    col, args.event_after, args.event_before, limit=args.limit
                )
            )
            counts[col] = await _count_in_range(
                col, args.event_after, args.event_before
            )
        except httpx.HTTPError as e:
            errors[col] = str(e)[:200]

    merged.sort(key=lambda p: p["event_timestamp"] or "", reverse=True)
    truncated = merged[: args.limit]

    total_event = sum(c.get("event", 0) for c in counts.values())
    total_fallback = sum(c.get("ingestion_fallback", 0) for c in counts.values())

    return {
        "event_after": args.event_after,
        "event_before": args.event_before,
        "count_total_in_range": total_event + total_fallback,
        "count_with_real_event_timestamp": total_event,
        "count_ingestion_fallback": total_fallback,
        "count_returned": len(truncated),
        "by_collection": counts,
        "results": truncated,
        "errors": errors,
        "hint": (
            "is_ingestion_timestamp=True means the date is when the "
            "point was IMPORTED, not when the described event happened "
            "— common for google_takeout_gemini (4140 points). Treat "
            "those dates as lower-confidence. If count_total_in_range "
            "> count_returned, call again with a narrower range or "
            "raise `limit` up to 50."
        ),
    }


class MemoryRecentArgs(BaseModel):
    n: int = Field(
        10, ge=1, le=TEMPORAL_MAX_RESULTS,
        description="How many most-recent points to return.",
    )
    collections: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TEMPORAL_COLLECTIONS),
        description="Which Qdrant collections to scan.",
    )


@registry.tool(
    name="memory_recent",
    description=(
        "Return the N most recent memory points across dated "
        "collections. Use for \"що я останнім часом робив?\", "
        "\"останні події\", or to check the freshness of the corpus "
        "before answering a recency-sensitive question."
    ),
    schema=MemoryRecentArgs.model_json_schema(),
)
async def _tool_memory_recent(**kwargs: Any) -> dict[str, Any]:
    args = MemoryRecentArgs(**kwargs)
    merged: list[dict] = []
    errors: dict[str, str] = {}
    # Pull generous headroom per collection (3× target) so after the
    # combined sort we still have N fresh points even if one collection
    # dominates with near-future dates.
    fetch = max(args.n * 3, args.n)
    for col in args.collections:
        try:
            merged.extend(await _scroll_temporal(col, None, None, limit=fetch))
        except httpx.HTTPError as e:
            errors[col] = str(e)[:200]
    merged.sort(key=lambda p: p["event_timestamp"] or "", reverse=True)
    return {
        "count_scanned": len(merged),
        "count_returned": min(args.n, len(merged)),
        "results": merged[: args.n],
        "errors": errors,
    }


class MemoryFirstArgs(BaseModel):
    n: int = Field(
        10, ge=1, le=TEMPORAL_MAX_RESULTS,
        description="How many oldest points to return.",
    )
    collections: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TEMPORAL_COLLECTIONS),
        description="Which Qdrant collections to scan.",
    )


@registry.tool(
    name="memory_first",
    description=(
        "Return the N oldest memory points across dated collections. "
        "Use for \"з чого все починалось?\" or to spot-check how "
        "far back the imported history actually reaches."
    ),
    schema=MemoryFirstArgs.model_json_schema(),
)
async def _tool_memory_first(**kwargs: Any) -> dict[str, Any]:
    args = MemoryFirstArgs(**kwargs)
    merged: list[dict] = []
    errors: dict[str, str] = {}
    fetch = max(args.n * 3, args.n)
    for col in args.collections:
        try:
            merged.extend(await _scroll_temporal(col, None, None, limit=fetch))
        except httpx.HTTPError as e:
            errors[col] = str(e)[:200]
    merged.sort(key=lambda p: p["event_timestamp"] or "")
    return {
        "count_scanned": len(merged),
        "count_returned": min(args.n, len(merged)),
        "results": merged[: args.n],
        "errors": errors,
    }


class MemoryCoverageArgs(BaseModel):
    event_after: Optional[str] = Field(
        None,
        description="Optional lower bound (ISO-8601). Omit to count everything below `event_before`.",
    )
    event_before: Optional[str] = Field(
        None,
        description="Optional upper bound (ISO-8601). Omit to count everything above `event_after`.",
    )
    collections: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TEMPORAL_COLLECTIONS),
    )


@registry.tool(
    name="memory_coverage",
    description=(
        "Exact count of memory points in a date range, split by "
        "real-event-timestamp vs ingestion-fallback. Call this "
        "first when you suspect \"there might be nothing from that "
        "date\" — a zero totals.total means the corpus genuinely "
        "has no data for the range, so don't hallucinate an answer. "
        "Cheap (no payload fetch, just indexed count)."
    ),
    schema=MemoryCoverageArgs.model_json_schema(),
)
async def _tool_memory_coverage(**kwargs: Any) -> dict[str, Any]:
    args = MemoryCoverageArgs(**kwargs)
    totals: dict[str, int] = {"event": 0, "ingestion_fallback": 0, "total": 0}
    by_col: dict[str, dict[str, int]] = {}
    errors: dict[str, str] = {}
    for col in args.collections:
        try:
            c = await _count_in_range(col, args.event_after, args.event_before)
            by_col[col] = c
            for k, v in c.items():
                totals[k] = totals.get(k, 0) + v
        except httpx.HTTPError as e:
            errors[col] = str(e)[:200]
    return {
        "event_after": args.event_after,
        "event_before": args.event_before,
        "totals": totals,
        "by_collection": by_col,
        "errors": errors,
        "hint": (
            "totals.event = points with a real event_timestamp in range; "
            "totals.ingestion_fallback = points without event_timestamp "
            "where the import date matches (lower confidence — date "
            "tells when it was imported, not when it happened). Zero "
            "`total` means the corpus has nothing for that range — do "
            "not invent an answer."
        ),
    }


# ---------------------------------------------------------------------------
# Mount MCP routes (GET /tools, POST /tools/{name}/call)
# ---------------------------------------------------------------------------

mount_mcp(app, registry)


@app.on_event("shutdown")
async def _close_qdrant() -> None:
    """Cleanly close the shared AsyncQdrantClient on shutdown to avoid
    'Unclosed client session' warnings in logs."""
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


# ---------------------------------------------------------------------------
# Convenience endpoints for curl debugging.
# ---------------------------------------------------------------------------

@app.post("/retrieval_debug")
def ep_retrieval_debug(args: RetrievalDebugArgs) -> dict[str, Any]:
    return {"result": _tool_retrieval_debug(**args.model_dump())}


@app.get("/compression_stats")
def ep_compression_stats(collection: str = "user_profile_raw") -> dict[str, Any]:
    return {"result": _tool_list_compression_stats(collection=collection)}
