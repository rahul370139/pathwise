"""
foundry_iq.py — Microsoft Foundry IQ grounding layer for PathWise.

This is the REQUIRED Microsoft IQ integration for the Creative Apps track.
It grounds retrieval in a Foundry IQ knowledge base (backed by Azure AI Search)
and returns results in the SAME shape as `rag_kb.retrieve_kb`, so it is a
drop-in source. When Foundry env vars are not set, callers transparently fall
back to the existing Supabase KB (see `rag_kb.retrieve`).

Returned item shape (identical to the Supabase path, plus `source_uri`):
    { "doc": str, "section": str|None, "chunk": str, "similarity": float, "source_uri": str|None }

Env (set in backend/.env — never commit secrets):
    FOUNDRY_SEARCH_ENDPOINT   https://<your-search>.search.windows.net
    FOUNDRY_SEARCH_KEY        admin/query key (omit to use DefaultAzureCredential / Entra)
    FOUNDRY_INDEX             knowledge-base index name (e.g. "prepkb-index")
    FOUNDRY_SEMANTIC_CONFIG   optional semantic configuration name (enables semantic reranking)
    FOUNDRY_FIELD_CONTENT     content field name  (optional; auto-detected if unset)
    FOUNDRY_FIELD_TITLE       title field name    (optional; auto-detected if unset)
    FOUNDRY_FIELD_SOURCE      source-uri field    (optional; auto-detected if unset)

Field auto-detection: portal-created indexes vary (e.g. "snippet" +
"metadata_storage_path"). If the *_FIELD_* env vars are unset, we sniff the
returned document for the most likely content/source fields and derive a
human-readable doc title from the chunk's first markdown heading.

Install (only needed once Foundry is wired):
    pip install azure-search-documents azure-identity
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from loguru import logger

# Candidate field names, ordered by preference. Covers our push_to_foundry
# schema ("content"/"sourceUri") and common portal/indexer schemas
# ("snippet"/"chunk" + "metadata_storage_path").
_CONTENT_CANDIDATES = ["content", "snippet", "chunk", "text", "page_content", "body"]
_SOURCE_CANDIDATES = [
    "sourceUri", "source", "url", "metadata_storage_path",
    "metadata_storage_name", "parent_id", "snippet_parent_id",
]
_TITLE_CANDIDATES = ["title", "name", "document_title", "filename"]

_last_error: Optional[str] = None


def get_last_error() -> Optional[str]:
    """Human-readable reason the last Foundry call returned no results."""
    return _last_error


def normalize_endpoint(raw: Optional[str] = None) -> str:
    """Return a clean Azure Search base URL (no trailing slash, no /indexes path)."""
    value = (raw or os.getenv("FOUNDRY_SEARCH_ENDPOINT") or "").strip()
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    parsed = urlparse(value)
    if not parsed.netloc:
        raise ValueError(f"Invalid FOUNDRY_SEARCH_ENDPOINT: {raw!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def check_endpoint_dns(endpoint: Optional[str] = None) -> Tuple[bool, str]:
    """Return (dns_ok, message) for the search service hostname."""
    try:
        url = normalize_endpoint(endpoint)
    except ValueError as e:
        return False, str(e)
    if not url:
        return False, "FOUNDRY_SEARCH_ENDPOINT is not set"
    host = urlparse(url).hostname or ""
    if not host.endswith(".search.windows.net"):
        return False, f"Unexpected host {host!r} — expected *.search.windows.net"
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return True, f"DNS OK for {host}"
    except socket.gaierror as e:
        return False, (
            f"DNS lookup failed for {host}: {e}. "
            "Copy the exact URL from Azure Portal → your Search service → Overview → Url."
        )


def foundry_is_configured() -> bool:
    """True only when the minimum Foundry IQ env vars are present."""
    return bool(normalize_endpoint() and os.getenv("FOUNDRY_INDEX"))


def _pick_field(doc: Dict, configured: Optional[str], candidates: List[str]) -> Optional[str]:
    """Return the field name to read from this doc: env override, else first
    candidate that holds a non-empty string."""
    if configured and isinstance(doc.get(configured), str) and doc[configured].strip():
        return configured
    for c in candidates:
        v = doc.get(c)
        if isinstance(v, str) and v.strip():
            return c
    return None


def _first_heading(snippet: str) -> Optional[str]:
    """First markdown heading in a chunk (used as the citation 'section')."""
    for line in (snippet or "").splitlines():
        s = line.strip()
        if s.startswith("#"):
            cleaned = s.lstrip("#").strip().strip("*").strip()
            if cleaned:
                return cleaned[:90]
    return None


def _filename_stem(source_val: Optional[str]) -> Optional[str]:
    """Readable file label from a source path/URL, e.g.
    '.../26_system_design_for_ml.md' -> '26_system_design_for_ml'."""
    if not source_val:
        return None
    base = os.path.basename(source_val.rstrip("/")) or source_val
    return os.path.splitext(base)[0][:90] or None


def _search_sync(query: str, top_k: int) -> List[Dict]:
    """Blocking Azure AI Search call. Run via asyncio.to_thread from `retrieve`."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    endpoint = normalize_endpoint()
    index = os.environ["FOUNDRY_INDEX"]
    key = os.getenv("FOUNDRY_SEARCH_KEY")

    if key:
        credential = AzureKeyCredential(key)
    else:
        from azure.identity import DefaultAzureCredential  # Entra / managed identity

        credential = DefaultAzureCredential()

    client = SearchClient(endpoint=endpoint, index_name=index, credential=credential)

    cfg_content = os.getenv("FOUNDRY_FIELD_CONTENT")
    cfg_title = os.getenv("FOUNDRY_FIELD_TITLE")
    cfg_source = os.getenv("FOUNDRY_FIELD_SOURCE")

    search_kwargs: Dict = {"search_text": query, "top": top_k}

    # Hybrid retrieval: add a server-vectorized query when the index exposes a
    # vector field + vectorizer (integrated vectorization). Greatly improves
    # recall over keyword-only BM25.
    vector_field = os.getenv("FOUNDRY_VECTOR_FIELD")
    if vector_field:
        try:
            from azure.search.documents.models import VectorizableTextQuery

            search_kwargs["vector_queries"] = [
                VectorizableTextQuery(
                    text=query, k_nearest_neighbors=top_k, fields=vector_field
                )
            ]
        except Exception as e:  # noqa: BLE001 - degrade to keyword-only
            logger.warning(f"Foundry IQ vector query unavailable ({e}); keyword-only.")

    # Semantic reranking (L2) when a semantic configuration exists.
    semantic_config = os.getenv("FOUNDRY_SEMANTIC_CONFIG")
    if semantic_config:
        search_kwargs.update(
            query_type="semantic",
            semantic_configuration_name=semantic_config,
        )

    results = client.search(**search_kwargs)

    out: List[Dict] = []
    for r in results:
        doc = dict(r)  # SearchClient rows are mapping-like
        content_field = _pick_field(doc, cfg_content, _CONTENT_CANDIDATES)
        source_field = _pick_field(doc, cfg_source, _SOURCE_CANDIDATES)
        title_field = _pick_field(doc, cfg_title, _TITLE_CANDIDATES)

        chunk = (doc.get(content_field) or "").strip() if content_field else ""
        source_val = doc.get(source_field) if source_field else None
        title_val = doc.get(title_field) if title_field else None
        heading = _first_heading(chunk)

        # Citation label preference: explicit title field → source file stem →
        # first markdown heading. The heading becomes the 'section' when we
        # already have a doc label, mirroring the Supabase KB shape.
        if title_val:
            doc_label, section = title_val, (doc.get("section") or heading)
        elif source_val:
            doc_label, section = (_filename_stem(source_val) or "foundry-kb"), (doc.get("section") or heading)
        else:
            doc_label, section = (heading or "foundry-kb"), doc.get("section")

        # Semantic reranker score when present, else the BM25/vector score.
        score = doc.get("@search.reranker_score") or doc.get("@search.score") or 0.0
        out.append(
            {
                "doc": doc_label,
                "section": section,
                "chunk": chunk,
                "similarity": float(score),
                "source_uri": source_val,
            }
        )
    return out


async def retrieve(query: str, top_k: int = 6) -> List[Dict]:
    """Return grounded chunks from Foundry IQ. Empty list on any failure.

    Callers MUST handle the empty case (so `rag_kb.retrieve` can fall back).
    Inspect ``get_last_error()`` to distinguish DNS/auth errors from zero hits.
    """
    global _last_error
    _last_error = None

    if not foundry_is_configured() or not query or not query.strip():
        if not foundry_is_configured():
            _last_error = "foundry_not_configured"
        return []

    dns_ok, dns_msg = check_endpoint_dns()
    if not dns_ok:
        _last_error = dns_msg
        logger.warning(f"Foundry IQ endpoint check failed: {dns_msg}")
        return []

    try:
        matches = await asyncio.to_thread(_search_sync, query, top_k)
        # Drop rows with no retrievable text (wrong field mapping).
        matches = [m for m in matches if (m.get("chunk") or "").strip()]
        if not matches:
            _last_error = "foundry_empty"
        return matches
    except ImportError:
        _last_error = "azure-search-documents not installed"
        logger.warning(
            "azure-search-documents not installed; install it to use Foundry IQ. "
            "Falling back to Supabase KB."
        )
        return []
    except Exception as e:  # noqa: BLE001 - we always degrade gracefully
        _last_error = str(e)
        logger.warning(f"Foundry IQ retrieval failed ({e}); falling back to Supabase KB.")
        return []


if __name__ == "__main__":  # manual smoke check
    import sys

    q = " ".join(sys.argv[1:]) or "what is retrieval augmented generation"
    print("configured:", foundry_is_configured())
    for m in asyncio.run(retrieve(q, top_k=5)):
        print(f"- {m['similarity']:.3f} [{m['doc']}] {m['chunk'][:90]}...")
