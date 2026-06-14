# PathWise Backend

FastAPI service for PathWise — PDF distillation, Foundry IQ–grounded RAG, Career Simulator, career matching, and dashboard APIs.

**Live API docs:** `/docs` when running locally · **Deep dive:** [../docs/backend/ARCHITECTURE.md](../docs/backend/ARCHITECTURE.md) · **Package tree:** [../docs/backend/STRUCTURE.md](../docs/backend/STRUCTURE.md)

---

## Quick start

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Load env from `.env` at repository root (never commit)
export $(grep -v '^#' ../.env | xargs) 2>/dev/null || true

uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Package layout

```
backend/
├── main.py                 # FastAPI entry (routes → pathwise modules)
├── pathwise/
│   ├── infra/              # Supabase + Microsoft Foundry IQ
│   ├── learn/              # distiller.py, rag_kb.py
│   ├── career/             # career_simulator, resume_career, matcher
│   ├── dashboard/          # dashboard.py, mastery.py
│   └── eval/               # eval_simulator.py
├── scripts/                # foundry_diag, push_to_foundry, test_rag
├── data/                   # O*NET CSV, eval reports, career plan snapshots
├── knowledge_base/         # Grounding corpus → Foundry + Supabase
├── migrations/             # Supabase SQL (apply in Studio)
├── Dockerfile
└── docker-compose.yml
```

See [../docs/backend/STRUCTURE.md](../docs/backend/STRUCTURE.md) for CLI probes and retrieval observability.

---

## Environment variables

Copy [backend.env.example](./backend.env.example) → `backend.env` on the VPS (never commit secrets).

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Llama-3.3-70b generation |
| `COHERE_API_KEY` | embed-english-light-v3.0 (384-dim) |
| `NEXT_PUBLIC_SUPABASE_URL` | Postgres + auth |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase client key |
| `FOUNDRY_SEARCH_ENDPOINT` | Azure AI Search URL (`*.search.windows.net`) |
| `FOUNDRY_SEARCH_KEY` | Search admin/query key |
| `FOUNDRY_INDEX` | Index name (e.g. `prepkb-index`) |
| `FOUNDRY_FIELD_CONTENT` | e.g. `snippet` |
| `FOUNDRY_FIELD_SOURCE` | e.g. `metadata_storage_path` |
| `FOUNDRY_VECTOR_FIELD` | e.g. `snippet_vector` |
| `FOUNDRY_SEMANTIC_CONFIG` | optional semantic reranker |

Local dev loads from **`.env`** at the repository root.

---

## Smoke tests

```bash
# Foundry IQ (primary grounding)
python scripts/foundry_diag.py "what is RAG?"

# Full retrieve() path (Foundry → Supabase fallback)
python -m pathwise.learn.rag_kb probe "what is RAG?"

# Career Simulator eval harness
python -m pathwise.eval.eval_simulator

# HTTP (server running)
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/api/rag/status"
curl "http://127.0.0.1:8000/api/rag/probe?q=what%20is%20RAG"
```

Expected: `foundry_iq` provider when Foundry env is set; `[RAG:foundry_iq]` in logs.

---

## Key endpoints

| Surface | Paths |
|---|---|
| **Learn** | `POST /api/chat`, `/api/chat/upload`, `/api/distill` |
| **Career Simulator** | `POST /api/simulator/start`, `/answer` · `GET /stream/{id}`, `/report/{id}`, `/eval` |
| **Career** | `POST /api/career/resume/parse`, `/plan/build`, `/upgrade`, `/match` |
| **Dashboard** | `GET /api/dashboard/*`, `GET /api/agent/mastery/{user_id}` |
| **RAG debug** | `GET /api/rag/status`, `/api/rag/probe` |

Full endpoint table: [../docs/backend/ARCHITECTURE.md](../docs/backend/ARCHITECTURE.md).

---

## Microsoft Foundry IQ

PathWise satisfies the **Agents League Microsoft IQ requirement** via Foundry IQ (Azure AI Search):

- **`pathwise/infra/foundry_iq.py`** — hybrid + semantic search, citation-shaped chunks
- **`pathwise/learn/rag_kb.py`** — `retrieve()` prefers Foundry, falls back to Supabase pgvector

Push corpus: `python scripts/push_to_foundry.py --source knowledge_base`

Corpus docs: [knowledge_base/README.md](./knowledge_base/README.md)

---

## Docker deploy (Hostinger VPS)

```bash
cp backend.env.example backend.env   # fill secrets
chmod 600 backend.env
docker compose up -d --build
docker compose logs -f pathwise-backend
```

Full VPS + Vercel checklist: [../README.md](../README.md).

---

## Documentation

| Doc | Description |
|---|---|
| [../docs/SUBMISSION.md](../docs/SUBMISSION.md) | Hackathon project page copy |
| [../docs/COPILOT_NOTES.md](../docs/COPILOT_NOTES.md) | GitHub Copilot build narrative |
| [../docs/backend/ARCHITECTURE.md](../docs/backend/ARCHITECTURE.md) | Flows, modules, endpoints |
| [../docs/backend/STRUCTURE.md](../docs/backend/STRUCTURE.md) | Directory tree + CLI |
| [../docs/README.md](../docs/README.md) | Documentation index |
