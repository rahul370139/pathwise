# Backend layout

Python package lives under `pathwise/`. `main.py` stays at the repo root for Docker/uvicorn (`main:app`).

**Full architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md) · **Monorepo README:** [../../README.md](../../README.md)

```
backend/
├── main.py                 # FastAPI entry (routes only — delegates to pathwise)
├── pathwise/
│   ├── schemas.py          # Pydantic contracts
│   ├── paths.py            # BACKEND_ROOT, DATA_DIR, KB_DIR (stable asset paths)
│   ├── infra/              # Supabase + Microsoft Foundry IQ
│   │   ├── supabase_helper.py
│   │   └── foundry_iq.py
│   ├── learn/              # PDF distillation, chat, RAG
│   │   ├── distiller.py
│   │   └── rag_kb.py
│   ├── career/             # Simulator, matching, resume plans
│   │   ├── career_simulator.py
│   │   ├── resume_career.py
│   │   ├── career_matcher.py
│   │   ├── unified_career_system.py
│   │   └── career_plan_storage.py
│   ├── dashboard/
│   │   ├── dashboard.py
│   │   └── mastery.py
│   └── eval/
│       └── eval_simulator.py
├── tests/                  # pytest (retrieval provider, smoke)
├── scripts/                # offline: push_to_foundry, test_rag, onet_to_markdown
├── data/                   # O*NET CSV, career plan snapshots, eval reports
├── knowledge_base/         # grounding corpus (sync to Foundry + Supabase)
└── migrations/             # Supabase SQL
```

## CLI entry points

```bash
cd backend
python -m pathwise.learn.rag_kb probe "what is RAG?"
python -m pathwise.learn.rag_kb ingest knowledge_base
python -m pathwise.eval.eval_simulator
python -m scripts.test_rag
python scripts/push_to_foundry.py --categories behavioral,onet_careers
```

## Check Foundry vs Supabase retrieval

| Method | What it shows |
|--------|----------------|
| `GET /api/rag/status` | Last provider used, whether Foundry env is set |
| `GET /api/rag/probe?q=...` | Live query + `retrieval_provider` per chunk |
| `python -m pathwise.learn.rag_kb probe "..."` | Same in the terminal |
| Server logs | Lines like `[RAG:foundry_iq]` or `[RAG:supabase_pgvector]` |

Each retrieved chunk includes `retrieval_provider`: `foundry_iq` or `supabase_pgvector`.
