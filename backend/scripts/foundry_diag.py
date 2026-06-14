"""
foundry_diag.py — read-only diagnostic for the Foundry IQ (Azure AI Search) index.

Confirms: (1) the index exists, (2) what the field names actually are, and
(3) that a query returns grounded text + scores (i.e. citations will work).

Run:  python3 scripts/foundry_diag.py "what is rag"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API = "2024-07-01"
endpoint = os.environ["FOUNDRY_SEARCH_ENDPOINT"].rstrip("/")
index = os.environ["FOUNDRY_INDEX"]
key = os.environ["FOUNDRY_SEARCH_KEY"]
headers = {"api-key": key, "Content-Type": "application/json"}
query = " ".join(sys.argv[1:]) or "what is retrieval augmented generation"

print(f"endpoint = {endpoint}")
print(f"index    = {index}\n")

# 1) schema
r = requests.get(f"{endpoint}/indexes/{index}?api-version={API}", headers=headers, timeout=30)
print(f"[schema] HTTP {r.status_code}")
if r.status_code == 200:
    fields = r.json().get("fields", [])
    print("fields:")
    for f in fields:
        flags = [k for k in ("searchable", "retrievable", "key") if f.get(k)]
        print(f"  - {f['name']:<24} {f['type']:<28} {','.join(flags)}")
else:
    print(r.text[:400])

# 2) sample query
body = {"search": query, "top": 3, "queryType": "simple"}
r = requests.post(
    f"{endpoint}/indexes/{index}/docs/search?api-version={API}",
    headers=headers, data=json.dumps(body), timeout=30,
)
print(f"\n[search '{query}'] HTTP {r.status_code}")
if r.status_code == 200:
    docs = r.json().get("value", [])
    print(f"hits: {len(docs)}")
    for i, d in enumerate(docs):
        score = d.get("@search.score")
        text_keys = {k: v for k, v in d.items() if isinstance(v, str) and not k.startswith("@")}
        longest = max(text_keys.items(), key=lambda kv: len(kv[1]), default=("?", ""))
        print(f"\n  #{i+1} score={score}")
        print(f"     non-vector string fields: {list(text_keys.keys())}")
        print(f"     longest field '{longest[0]}': {longest[1][:160]}...")
else:
    print(r.text[:400])
