# Built with GitHub Copilot — PathWise Career Simulator

This submission for the **Creative Apps (GitHub Copilot)** track was built end-to-end
in VS Code with GitHub Copilot. Below is where Copilot accelerated the work, and the
**Microsoft IQ** integration that grounds the experience.

## Microsoft IQ integration (required)
- **Foundry IQ** is the grounding layer. `backend/foundry_iq.py` queries a Foundry IQ
  knowledge base (Azure AI Search) and returns citation-bearing chunks. `backend/rag_kb.py`
  gains a single `retrieve()` indirection that prefers Foundry IQ when configured and
  falls back to the existing Supabase pgvector KB — which is also the demo's reliability
  story (kill Foundry → Supabase, no downtime).

## What Copilot helped build
- **Multi-agent orchestrator** (`backend/career_simulator.py`): the named agents
  (Planner, Retrieval, Interviewer, Scorer, Remediation, Report) + `SafetyGuard` and the
  per-session SSE event bus. Copilot scaffolded the dataclass session + prompt templates;
  each agent delegates to existing modules (no logic rewritten).
- **FastAPI endpoints** (`backend/main.py`): `/api/simulator/{start,answer,stream,report,eval}`,
  including the Server-Sent-Events generator for the live timeline.
- **Eval harness** (`backend/eval_simulator.py`): golden-set runner for groundedness,
  refusal correctness, answer relevance, and p95 latency.
- **Frontend** (`frontend/`): `app/simulator/page.tsx`, `components/agent-thinking-panel.tsx`
  (EventSource timeline), `components/readiness-report.tsx` (30/60/90 plan + Save-as-PDF),
  `components/eval-card.tsx`, and the `simulatorAPI` helpers in `lib/api.ts`.

## Reuse over rewrite (key engineering decision)
Every agent calls code that already existed and is battle-tested:
- `parse_resume`, `_find_onet_row`, `_onet_summary`, `build_career_plan` → `resume_career.py`
- `call_groq`, `_parse_json_safely`, `map_reduce_summary`, `gen_flashcards_quiz` → `distiller.py`
- `retrieve`, `format_kb_context` → `rag_kb.py` (Foundry IQ or Supabase)

## Run locally
```bash
# Backend
cd backend && uvicorn main:app --reload --port 8000
# (optional) ground in Foundry IQ:
#   set FOUNDRY_SEARCH_ENDPOINT / FOUNDRY_SEARCH_KEY / FOUNDRY_INDEX in backend/.env
#   python scripts/push_to_foundry.py --source ../interview_prep --create-index
#   python eval_simulator.py     # writes data/eval/simulator_eval_report.json

# Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000/simulator
```
