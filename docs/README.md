# PathWise documentation

Documentation for the **Agents League Hackathon — Creative Apps (GitHub Copilot)** submission.

| Document | Audience | Purpose |
|---|---|---|
| [backend/ARCHITECTURE.md](./backend/ARCHITECTURE.md) | Judges & developers | End-to-end architecture, Foundry IQ, Career Simulator, endpoints |
| [COPILOT_NOTES.md](./COPILOT_NOTES.md) | Creative Apps judges | Where GitHub Copilot accelerated the build |
| [backend/ARCHITECTURE.md](./backend/ARCHITECTURE.md) | Backend developers | Module flows, endpoints, Foundry IQ, Career Simulator |
| [backend/STRUCTURE.md](./backend/STRUCTURE.md) | Backend developers | Package tree, CLI probes, retrieval observability |

## Section READMEs (stay in place)

- **[../README.md](../README.md)** — monorepo overview, architecture diagrams, deploy checklists
- **[../backend/README.md](../backend/README.md)** — FastAPI quick start, env vars, smoke tests
- **[../frontend/README.md](../frontend/README.md)** — Next.js setup, routes, Vercel config
- **[../backend/knowledge_base/README.md](../backend/knowledge_base/README.md)** — grounding corpus & ingest

## Quick verification before submit

```bash
# From repository root: cd backend (with .venv active and .env loaded)
python scripts/foundry_diag.py "what is RAG?"
python -m pathwise.learn.rag_kb probe "what is RAG?"
python -m pathwise.eval.eval_simulator

# Frontend production build
cd ../frontend && pnpm build
```

Expected: `foundry_iq` provider, DNS OK, eval report written, Next.js build succeeds.

## Submission & compliance (Agents League)

Before pushing to a **public** repository:

- [ ] No secrets in code — use `backend.env` / Vercel env only (see [SECURITY.md](../SECURITY.md))
- [ ] `backend.env` is **not** tracked by git (listed in `.gitignore`)
- [ ] Only General-level, non-confidential content
- [ ] [LICENSE](../LICENSE) included (MIT)
- [ ] Sign the **Microsoft CLA** when contributing to the official Agents League org repo (GitHub prompts on first PR)

