# skynet-memory-mcp

HTTP/JSON MCP-style diagnostic server for Skynet memory retrieval.

This is a **diagnostic tool for AI agents** (Claude Code sessions + the
Skynet agent itself), not a human-facing dashboard. When a user says
"Skynet forgot about X" or "the chat quoted the wrong memory", an LLM
agent calls `retrieval_debug` here and gets back the per-candidate
score components to diagnose *why* retrieval behaved that way.

Mirror of the `/ui/retrieval-debug` atlas page, optimized for LLM
token budgets (pruned + annotated).

Part of the Skynet stack (sanscfs/infra).

## Endpoints

- `GET  /health`            liveness
- `GET  /ready`             checks identity reachability
- `GET  /tools`             MCP tool manifest (name + JSON schema)
- `POST /tools/{name}/call` invoke a tool, body: `{"arguments": {...}}`

Curl shortcuts: `POST /retrieval_debug`, `GET /compression_stats?collection=...`.

### Tools

| name                      | description                                                              |
|---------------------------|--------------------------------------------------------------------------|
| `retrieval_debug`         | Replay a query through identity `/context?debug=true`, prune the result to top 3 candidates per bucket (episodic / knowledge / raw), annotate with previews + a short diagnosis. |
| `list_compression_stats`  | Atlas compression histogram + top cliques under eviction pressure for a given Qdrant collection. |

### `retrieval_debug` — what the score components mean

For each surviving candidate:

| component           | typical | meaning                                                       |
|---------------------|---------|---------------------------------------------------------------|
| `vec`               | 0.0..1.0 | cosine similarity to the (anchor-aware) query embedding      |
| `access_boost`      | ~1.0    | upweight if the point was read/written recently                |
| `recency_boost`     | ~1.0    | wall-clock recency bonus                                       |
| `graph_boost`       | ~1.0    | 1-hop typed-edge neighbour boost                               |
| `raw_score`         |         | vec × all boosts (pre-penalty product)                         |
| `strike_penalty`    | <=1.0   | downweight for points the user has thumbs-downed               |
| `logical_decay`     | <=1.0   | *logical-time* decay (ticks on missed retrievals, not days)    |
| `compression_boost` | ~1.0    | summaries get lifted on short queries                          |
| `clique_boost`      | ~1.0    | in-clique boost when anchor resolves to a qualified clique     |
| `cross_clique_boost`| ~1.0    | cross-clique bridge boost                                      |
| `final_score`       |         | what selection actually compared                               |

Rule-of-thumb diagnoses (auto-generated into the `diagnosis` field):

- `vec < 0.5` on all top candidates -> embedding mismatch; re-phrase the
  query or check that the embedding model hasn't changed.
- `logical_decay < 0.9` -> point is being punished for misses; due for
  consolidation.
- `compression_boost > 1.2` -> short query is pulling summaries ahead
  of the raw facts the user probably wants.
- anchor supplied but no `clique_boost` anywhere -> anchor doesn't
  resolve to a qualified clique in this collection; Louvain clique
  count may be too low.

## Environment

| var                   | default                                                      |
|-----------------------|--------------------------------------------------------------|
| `IDENTITY_URL`        | `http://skynet-identity.skynet-identity.svc:8080`            |
| `ATLAS_URL`           | `http://skynet-rag-atlas.skynet-rag-atlas.svc:8080`          |
| `REQUEST_TIMEOUT`     | `45`                                                         |
| `MAX_ITEMS_PER_BUCKET`| `3`                                                          |

## Registering with Claude Code

This server speaks HTTP. Claude Code expects MCP over stdio OR an
HTTP MCP endpoint. Two options:

### Option A — stdio via `kubectl exec` (simplest)

In `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "skynet-memory": {
      "command": "kubectl",
      "args": [
        "-n", "skynet-memory-mcp",
        "exec", "-i", "deploy/skynet-memory-mcp",
        "--", "python3", "-c",
        "import sys, json, httpx\nwhile True:\n  line = sys.stdin.readline()\n  if not line: break\n  req = json.loads(line)\n  r = httpx.post(f'http://localhost:8080/tools/{req[\"name\"]}/call', json={'arguments': req.get('arguments', {})}, timeout=60)\n  sys.stdout.write(json.dumps(r.json()) + '\\n'); sys.stdout.flush()"
      ]
    }
  }
}
```

### Option B — direct HTTP (when the server is port-forwarded / Authelia-fronted)

```bash
kubectl -n skynet-memory-mcp port-forward svc/skynet-memory-mcp 8788:8080
```

Then point Claude Code's HTTP MCP client at `http://localhost:8788`.
Manifest endpoint: `GET /tools`. Invoke: `POST /tools/{name}/call`.

### Option C — one-shot invocation from a Claude Code bash tool

Simplest for ad-hoc diagnosis from a session:

```bash
kubectl -n skynet-memory-mcp exec deploy/skynet-memory-mcp -- \
  python3 -c "import httpx, json, sys
r = httpx.post('http://localhost:8080/retrieval_debug',
    json={'query': sys.argv[1], 'anchor': sys.argv[2] if len(sys.argv)>2 else None},
    timeout=60)
print(json.dumps(r.json()['result'], indent=2, ensure_ascii=False))
" "what did we decide about embeddings"
```

## Local dev

```bash
pip install -r requirements.txt
IDENTITY_URL=http://localhost:8080 ATLAS_URL=http://localhost:8081 \
  uvicorn main:app --reload --port 8788
curl http://localhost:8788/tools | jq .
```
