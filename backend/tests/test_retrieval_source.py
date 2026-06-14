"""Tests for RAG provider tagging and status helpers."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from pathwise.learn import rag_kb


@pytest.mark.asyncio
async def test_retrieve_uses_foundry_when_configured(monkeypatch):
    monkeypatch.setenv("FOUNDRY_SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("FOUNDRY_INDEX", "prepkb-index")

    fake_matches = [{"doc": "05_rag_systems", "section": "Overview", "chunk": "RAG text", "similarity": 0.9}]

    with patch.object(rag_kb, "retrieve_kb", new=AsyncMock(return_value=[])):
        with patch("pathwise.infra.foundry_iq.foundry_is_configured", return_value=True):
            with patch("pathwise.infra.foundry_iq.retrieve", new=AsyncMock(return_value=fake_matches)):
                out = await rag_kb.retrieve("what is RAG?", top_k=3)

    assert len(out) == 1
    assert out[0]["retrieval_provider"] == "foundry_iq"
    status = rag_kb.get_retrieval_status()
    assert status["provider"] == "foundry_iq"
    assert status["match_count"] == 1


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_supabase(monkeypatch):
    monkeypatch.delenv("FOUNDRY_SEARCH_ENDPOINT", raising=False)

    supa_matches = [{"doc": "local_kb", "section": "S", "chunk": "from pgvector", "similarity": 0.7}]

    with patch.object(rag_kb, "retrieve_kb", new=AsyncMock(return_value=supa_matches)):
        out = await rag_kb.retrieve("system design", top_k=2)

    assert out[0]["retrieval_provider"] == "supabase_pgvector"
    status = rag_kb.get_retrieval_status()
    assert status["provider"] == "supabase_pgvector"
    assert status["fallback_reason"] in ("foundry_not_configured", "foundry_empty", None) or status["fallback_reason"]
