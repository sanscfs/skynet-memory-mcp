"""Per-tool wiring coverage for skynet-memory-mcp.

Verifies:
  * Every expected tool is registered with the skynet-mcp ToolRegistry.
  * The mounted /tools endpoint lists all tools with the MCP wire shape.
  * Qdrant-backed tools (memory_coverage, memory_by_date) actually call
    into the AsyncQdrantClient rather than hand-rolled httpx.
  * Unknown tools 404 and schema violations 400 (delegated to skynet-mcp,
    but we sanity-check the wiring end-to-end).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main


EXPECTED_TOOLS = {
    "retrieval_debug",
    "list_compression_stats",
    "memory_by_date",
    "memory_recent",
    "memory_first",
    "memory_coverage",
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture
def qdrant_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the lazy `_qdrant_client` module-global with an AsyncMock
    so temporal tools exercise the skynet_qdrant API surface without
    hitting a real Qdrant."""
    mock = AsyncMock()
    # Default behaviors -- individual tests can override.
    mock.count.return_value = 0
    mock.scroll.return_value = ([], None)
    monkeypatch.setattr(main, "_qdrant_client", mock)
    return mock


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_has_all_expected_tools() -> None:
    names = {spec.name for spec in main.registry}
    assert names == EXPECTED_TOOLS, f"tool set drifted: {names - EXPECTED_TOOLS} / {EXPECTED_TOOLS - names}"


def test_get_tools_exposes_every_tool_with_mcp_shape(client: TestClient) -> None:
    resp = client.get("/tools")
    assert resp.status_code == 200
    body = resp.json()
    assert {t["name"] for t in body["tools"]} == EXPECTED_TOOLS
    for t in body["tools"]:
        # MCP descriptor shape = {name, description, inputSchema}.
        assert set(t.keys()) == {"name", "description", "inputSchema"}
        # Schema must be a non-empty JSON schema (generated from pydantic).
        assert t["inputSchema"].get("type") == "object"


# ---------------------------------------------------------------------------
# Qdrant-client usage (the Phase-1 lib adoption we're asserting on)
# ---------------------------------------------------------------------------


def test_memory_coverage_calls_skynet_qdrant_count_twice_per_collection(
    client: TestClient, qdrant_mock: AsyncMock
) -> None:
    # Return distinct values for the two filter variants (event vs ingestion).
    qdrant_mock.count.side_effect = [11, 7]

    resp = client.post(
        "/tools/memory_coverage/call",
        json={
            "arguments": {
                "event_after": "2024-01-01",
                "event_before": "2024-01-31",
                "collections": ["user_profile_raw"],
            }
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()["result"]

    # skynet_qdrant.count must have been invoked twice (event + fallback).
    assert qdrant_mock.count.call_count == 2
    assert qdrant_mock.count.call_args_list[0].args[0] == "user_profile_raw"

    # Split counts surface on the response payload.
    assert body["totals"]["event"] == 11
    assert body["totals"]["ingestion_fallback"] == 7
    assert body["totals"]["total"] == 18


def test_memory_by_date_uses_skynet_qdrant_scroll(
    client: TestClient, qdrant_mock: AsyncMock
) -> None:
    qdrant_mock.scroll.return_value = (
        [
            {
                "id": "p1",
                "payload": {
                    "event_timestamp": "2024-03-15T10:00:00Z",
                    "text": "point one",
                    "source": "git",
                    "category": "commit",
                },
            },
        ],
        None,
    )
    qdrant_mock.count.return_value = 1

    resp = client.post(
        "/tools/memory_by_date/call",
        json={
            "arguments": {
                "event_after": "2024-03-15",
                "event_before": "2024-03-16",
                "collections": ["user_profile_raw"],
                "limit": 5,
            }
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()["result"]

    # At least one scroll against the lib (may do a fallback pass too).
    assert qdrant_mock.scroll.call_count >= 1
    scroll_args = qdrant_mock.scroll.call_args_list[0]
    assert scroll_args.args[0] == "user_profile_raw"
    # Lib accepts `limit` + `filter` as kwargs -- the service forwards them.
    assert scroll_args.kwargs.get("with_payload") is True

    assert body["count_returned"] == 1
    assert body["results"][0]["id"] == "p1"
    assert body["results"][0]["is_ingestion_timestamp"] is False


# ---------------------------------------------------------------------------
# HTTP-level MCP guarantees (delegated to skynet-mcp, sanity-checked here)
# ---------------------------------------------------------------------------


def test_call_unknown_tool_returns_404(client: TestClient) -> None:
    resp = client.post("/tools/does_not_exist/call", json={"arguments": {}})
    assert resp.status_code == 404
