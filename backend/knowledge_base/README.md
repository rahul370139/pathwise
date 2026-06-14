# PathWise Knowledge Base

Version-controlled source for the grounding corpus. Upload these to the Foundry
index (`prepkb-index`) via `python scripts/push_to_foundry.py --source knowledge_base`,
and/or ingest into the Supabase fallback with `python -m pathwise.learn.rag_kb ingest knowledge_base`.

## Folders

- `learning/` — technical interview prep topics (`01_activation_functions.md` … `47_deep_statistical_reasoning.md`).
- `behavioral/` — behavioral / STAR interview guidance and a model-answer bank.
- `projects/` — portfolio project deep-dives (P01–P14, including PathWise P08).
- `onet_careers/` — 271 O*NET career briefs (generated from `data/onet_bls_trimmed.csv`
  by `scripts/onet_to_markdown.py`). Regenerate anytime with that script.

## "Single source" — how the system keeps things relevant

Everything lives in **one** Foundry index, but the system routes by *intent and
query*, so the right content surfaces in the right place:

1. **Career numbers are structured, not retrieved.** Salary, growth, top skills,
   and role matching come from the **O*NET CSV** via `career_matcher` +
   `_find_onet_row` / `build_career_plan`. This is deterministic and accurate —
   the markdown briefs do **not** replace it; they make the same facts
   *retrievable* as prose when chat/interview discusses a role.

2. **Interview grounding is semantic.** The simulator's `RetrievalAgent` queries
   the index with a **competency-specific** query
   (e.g. *"system design for Backend Engineer: best practices, interview
   expectations"*). Hybrid (vector + keyword) search with the semantic reranker
   naturally returns the matching technical/behavioral docs — not random career
   briefs — because relevance is driven by the query, not the upload bucket.

3. **Learn chat is topic-focused.** Deep links (`/learn?topic=...`) seed the
   tutor with a topic, so retrieval is scoped to that subject.

Because each doc has a strong `# H1` title and a `**Type:**` line, the embedder
gets a clear signal of what each chunk is, which keeps behavioral, technical, and
career content cleanly separated at query time — no per-type index required.

## Sync to Microsoft Foundry IQ

```bash
cd backend
# Requires FOUNDRY_SEARCH_ENDPOINT, FOUNDRY_SEARCH_KEY, FOUNDRY_INDEX in .env
python scripts/push_to_foundry.py --source knowledge_base --create-index
python scripts/foundry_diag.py          # smoke test a query
python eval_simulator.py                # offline reliability metrics
```

At runtime, `rag_kb.retrieve()` prefers Foundry IQ and transparently falls back to
Supabase pgvector if Foundry is unavailable — the competition failover demo.

## Tips

- Keep one topic per file with `##` sub-headings (the chunker is heading-aware;
  headings become citation `section`s).
- After pushing new files, re-run `push_to_foundry.py` or trigger reindex in the Azure portal.
