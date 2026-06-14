# PathWise — Agents League project page

Copy fields below into the **Projects** form. All fields can be edited until submissions close (June 14, 2026).

---

## Title *(max 140 characters)*

```
PathWise — AI Career Simulator with Microsoft Foundry IQ
```

*(58 characters)*

---

## Tagline *(max 300 characters)*

```
Drop in a PDF or job posting and get personalized learning plus a live, citation-backed interview simulator — built with GitHub Copilot and grounded by Microsoft Foundry IQ on Azure AI Search.
```

*(189 characters)*

---

## Keywords *(tags)*

Add these one at a time in the form:

```
GitHub Copilot
Microsoft Foundry IQ
Career Simulator
Retrieval-Augmented Generation
Multi-Agent AI
FastAPI
Next.js
Azure AI Search
Interview Preparation
Learning Platform
```

---

## Description *(Markdown)*

Copy everything inside the block below into the **Description** field.

```markdown
## What is PathWise?

**PathWise** is an AI-powered learning and career-readiness platform for the **Agents League · Creative Apps (GitHub Copilot)** track.

Upload a resume and target job description — or drop in any PDF — and PathWise delivers:

- **Personalized learning** — summaries, quizzes, flashcards, micro-lessons, and concept maps
- **Career Simulator** — a multi-agent, adaptive mock interview with a visible **plan → retrieve → generate → verify** loop
- **Readiness Report** — top skill gaps, a 30/60/90-day plan, and an exportable report card

**Live demo:** [pathwise001.vercel.app/simulator](https://pathwise001.vercel.app/simulator)

---

## Microsoft IQ integration *(required)*

PathWise integrates **Foundry IQ** (Azure AI Search) as the primary grounding layer:

- Citation-backed retrieval for Learn chat and the Career Simulator
- Hybrid keyword + vector search with semantic reranking
- Automatic **failover to Supabase pgvector** if Foundry is unavailable — the demo keeps running

Every grounded answer can show **source citations** in the Agent Thinking panel.

---

## Signature feature — Career Simulator

1. **Planner** — maps resume vs. job description to competency gaps (O*NET-backed)
2. **Retrieval** — Foundry IQ pulls evidence per competency
3. **Interviewer** — adaptive questions scored against a rubric
4. **Scorer** — flags grounded vs. unsupported claims
5. **Remediation** — weak areas link to targeted micro-lessons on `/learn`
6. **Report** — shareable readiness card with 30/60/90 roadmap

The UI streams agent steps over **Server-Sent Events** so judges see reasoning in real time.

---

## Built with GitHub Copilot

Developed end-to-end in **VS Code with GitHub Copilot** — orchestrator, Foundry client, FastAPI routes, SSE timeline, eval harness, and React components. See [`docs/COPILOT_NOTES.md`](docs/COPILOT_NOTES.md) in the repo.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind, shadcn/ui · **Vercel** |
| Backend | FastAPI · **Hostinger VPS (Docker)** |
| LLM | Groq (llama-3.3-70b) |
| Embeddings | Cohere embed-english-light-v3.0 |
| Grounding | **Microsoft Foundry IQ** + Supabase pgvector fallback |
| Auth & data | Supabase (magic links, progress, mastery) |
| Career data | O*NET (271 roles) + interview-prep knowledge base |

---

## Rubric alignment

| Criterion | How PathWise delivers |
|---|---|
| **Accuracy & Relevance** | Foundry IQ citation-first RAG; competency-specific retrieval |
| **Reasoning & Multi-step** | Named agents + live SSE Agent Thinking timeline |
| **Creativity & Originality** | Career Simulator chains learning, RAG, interview, and roadmap in one flow |
| **UX & Presentation** | Polished UI, streaming citations, Readiness Report export |
| **Reliability & Safety** | Grounding guard, scope refusal, provider failover, offline eval harness |

---

## Repository

Public repo with README, architecture docs, and setup instructions.

- **README:** project overview + deploy checklists
- **docs/SUBMISSION.md:** this page
- **docs/COPILOT_NOTES.md:** Copilot build narrative
- **docs/backend/ARCHITECTURE.md:** technical deep dive

---

## 2-minute demo script

1. Open [/simulator](https://pathwise001.vercel.app/simulator)
2. Upload resume + paste a target JD
3. Watch the Agent Thinking panel (plan → retrieve with citations → verify)
4. Answer 2 adaptive questions; show grounded feedback
5. Download the Readiness Report

*Built with GitHub Copilot. Grounded with Microsoft Foundry IQ.*
```

---

## Internal reference *(do not paste into form)*

### Track

**Creative Apps** — GitHub Copilot in VS Code

### Live URLs

| Surface | URL |
|---|---|
| Home | [pathwise001.vercel.app](https://pathwise001.vercel.app/) |
| Career Simulator | [pathwise001.vercel.app/simulator](https://pathwise001.vercel.app/simulator) |
| Learn | [pathwise001.vercel.app/learn](https://pathwise001.vercel.app/learn) |

### Security checklist

- [ ] No `.env` or API keys in the public repo
- [ ] Read the [Agents League Disclaimer](https://aka.ms/agentsleague)
- [ ] Repository is public with a clear README
- [ ] Production secrets only in Vercel / VPS dashboards

### Related docs

- [COPILOT_NOTES.md](./COPILOT_NOTES.md)
- [COMPETITION.md](./COMPETITION.md) — internal rubric brief
- [backend/ARCHITECTURE.md](./backend/ARCHITECTURE.md)
- [../README.md](../README.md)
