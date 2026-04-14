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
from skynet_core.tracing import setup_tracing, get_tracer

log = setup_logging("memory-mcp")

setup_tracing("memory-mcp", instrumentors=["fastapi", "httpx"])
tracer = get_tracer("memory-mcp")

IDENTITY_URL = os.getenv(
    "IDENTITY_URL", "http://skynet-identity.skynet-identity.svc:8080"
).rstrip("/")
ATLAS_URL = os.getenv(
    "ATLAS_URL", "http://skynet-rag-atlas.skynet-rag-atlas.svc:8080"
).rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "45"))
MAX_ITEMS_PER_BUCKET = int(os.getenv("MAX_ITEMS_PER_BUCKET", "3"))

app = FastAPI(title="skynet-memory-mcp", version="0.1.0")

_identity_client: httpx.Client | None = None
_atlas_client: httpx.Client | None = None


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


def _tool_retrieval_debug(args: RetrievalDebugArgs) -> dict[str, Any]:
    payload = {
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
    previews = {}
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


def _tool_list_compression_stats(args: ListCompressionStatsArgs) -> dict[str, Any]:
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
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "retrieval_debug": {
        "description": (
            "Replay a query through skynet-identity with debug=True and "
            "return a pruned view of per-candidate score components "
            "(vec, access, recency, graph, logical_decay, compression_boost, "
            "clique_boost, ...). Use this whenever a user reports "
            "\"Skynet doesn't remember X\" or \"the reply referenced the "
            "wrong memory\". Answers the question: why did this candidate "
            "win/lose? Returns top 3 per bucket (episodic, knowledge, raw) "
            "plus a one-paragraph diagnosis."
        ),
        "model": RetrievalDebugArgs,
        "fn": _tool_retrieval_debug,
    },
    "list_compression_stats": {
        "description": (
            "Return a compact summary of memory compression health for a "
            "Qdrant collection: compression_level histogram (are we "
            "consolidating enough?) + cliques under eviction pressure. "
            "Use this when asking \"is memory healthy?\" or before "
            "triggering a manual consolidation DAG."
        ),
        "model": ListCompressionStatsArgs,
        "fn": _tool_list_compression_stats,
    },
}


def _tool_manifest() -> dict[str, Any]:
    tools = []
    for name, spec in TOOLS.items():
        model: type[BaseModel] = spec["model"]
        tools.append(
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": model.model_json_schema(),
            }
        )
    return {"tools": tools}


@app.get("/tools")
def list_tools() -> dict[str, Any]:
    return _tool_manifest()


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


@app.post("/tools/{name}/call")
def call_tool(name: str, req: ToolCallRequest) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown tool: {name}")
    model: type[BaseModel] = spec["model"]
    try:
        parsed = model(**(req.arguments or {}))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"invalid arguments: {e}") from e
    with tracer.start_as_current_span(f"tool_{name}") as span:
        span.set_attribute("tool.name", name)
        if hasattr(parsed, "query"):
            span.set_attribute("tool.query", str(parsed.query)[:200])
        if hasattr(parsed, "collection"):
            span.set_attribute("tool.collection", parsed.collection)
        try:
            result = spec["fn"](parsed)
            return {"result": result}
        except httpx.HTTPStatusError as e:
            span.record_exception(e)
            log.warning("upstream error for %s: %s", name, e.response.text[:500])
            raise HTTPException(
                status_code=502,
                detail=f"upstream {e.response.status_code}: {e.response.text[:200]}",
            ) from e
        except Exception as e:  # noqa: BLE001
            span.record_exception(e)
            log.exception("tool %s failed", name)
            raise HTTPException(status_code=500, detail=str(e)) from e


# Convenience endpoints for curl debugging.
@app.post("/retrieval_debug")
def ep_retrieval_debug(args: RetrievalDebugArgs) -> dict[str, Any]:
    return {"result": _tool_retrieval_debug(args)}


@app.get("/compression_stats")
def ep_compression_stats(collection: str = "user_profile_raw") -> dict[str, Any]:
    return {"result": _tool_list_compression_stats(ListCompressionStatsArgs(collection=collection))}
