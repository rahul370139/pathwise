# Built with GitHub Copilot — PathWise

Submission for **Agents League · Creative Apps (GitHub Copilot)**. PathWise was built end-to-end in VS Code with GitHub Copilot. This document is the judge-facing narrative: where Copilot accelerated the work and how **Microsoft Foundry IQ** grounds the experience.

---

## Microsoft IQ integration (required)

**Foundry IQ** (Azure AI Search) is the primary retrieval layer.

| Piece | Location |
|---|---|
| Foundry client | `backend/pathwise/infra/foundry_iq.py` |
| Retrieval indirection | `backend/pathwise/learn/rag_kb.py` → `retrieve()` |
| Observability | `GET /api/rag/status`, `GET /api/rag/probe?q=…`, `python scripts/foundry_diag.py` |

When `FOUNDRY_SEARCH_*` env vars are set, `retrieve()` queries Azure AI Search index `prepkb-index` and returns citation-bearing chunks. On DNS/API failure it **falls back** to Supabase pgvector (`interview_prep_kb`) — the demo's reliability story (kill Foundry → Supabase, no downtime).

---

## What Copilot helped build

### Multi-agent Career Simulator
- **`pathwise/career/career_simulator.py`** — Planner, Retrieval, Interviewer, Scorer, Remediation, Report agents + `SafetyGuard` + per-session SSE event bus
- Copilot scaffolded dataclass sessions and prompt templates; each agent **delegates** to existing modules (no logic rewritten)

### FastAPI surface
- **`backend/main.py`** — `/api/simulator/{start,answer,stream,report,eval}`, SSE generator for the live timeline, `/api/rag/status` and `/api/rag/probe`

### Eval harness
- **`pathwise/eval/eval_simulator.py`** — golden-set runner: groundedness, refusal correctness, answer relevance, p95 latency → `data/eval/simulator_eval_report.json`

### Frontend (Career Simulator UX)
- `frontend/app/simulator/page.tsx`
- `frontend/components/agent-thinking-panel.tsx` — EventSource timeline
- `frontend/components/readiness-report.tsx` — 30/60/90 plan + Save-as-PDF
- `frontend/components/eval-card.tsx` — dashboard eval metrics
- `frontend/lib/api.ts` — `simulatorAPI` helpers

### Foundry IQ wiring
- **`pathwise/infra/foundry_iq.py`** — Azure Search client, field mapping, hybrid + semantic retrieval
- **`scripts/foundry_diag.py`** — DNS check + schema + live query smoke test
- **`scripts/push_to_foundry.py`** — corpus upload to `prepkb-index`

---

## Reuse over rewrite (key engineering decision)

Every new agent calls battle-tested code:

| Agent need | Existing module |
|---|---|
| Resume parse, O*NET lookup, career plan | `pathwise/career/resume_career.py` |
| Groq calls, JSON parsing, flashcards/quiz | `pathwise/learn/distiller.py` |
| Grounded retrieval | `pathwise/learn/rag_kb.py` (Foundry IQ or Supabase) |
| Career matching | `pathwise/career/career_matcher.py` |

---

## Run locally

```bash
# Env: copy keys into .env at repository root (never commit)
# FOUNDRY_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
# FOUNDRY_SEARCH_KEY=...
# FOUNDRY_INDEX=prepkb-index

# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Verify Foundry IQ
python scripts/foundry_diag.py "what is RAG?"
python -m pathwise.learn.rag_kb probe "what is RAG?"
python -m pathwise.eval.eval_simulator

# Frontend
cd ../frontend && pnpm install && pnpm dev
# http://localhost:3000/simulator
```

---

## Deploy (competition demo)

| Layer | Host | Notes |
|---|---|---|
| Frontend | **Vercel** | Root dir `frontend/`; `API_PROXY_TARGET` → VPS backend |
| Backend | **Hostinger VPS** | `docker compose up -d --build`; secrets in `backend.env` |
| Grounding | **Azure AI Search** | Index `prepkb-index`; corpus via `scripts/push_to_foundry.py` |
| Auth | **Supabase** | Magic links; redirect URLs must include `/auth/callback` |

Full checklists: [../README.md](../README.md) · [SUBMISSION.md](./SUBMISSION.md)

---

## Demo intro (30 seconds)

*"PathWise is a Copilot-built learning and career-readiness platform. Every grounded answer flows through Microsoft Foundry IQ on Azure AI Search with citations. The Career Simulator runs a visible multi-agent loop — plan, retrieve with citations, interview, score, and produce a shareable readiness report — with automatic failover to Supabase if Foundry is unavailable."*
