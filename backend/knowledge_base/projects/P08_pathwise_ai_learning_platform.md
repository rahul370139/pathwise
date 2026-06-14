# P08: PathWise (PathWise) - AI-Powered Career Guidance & Learning Platform

> **Project Type:** Self-Project | **Timeline:** Summer 2025
> **Live Demo:** [pathwise001.vercel.app](https://pathwise001.vercel.app/)
> **GitHub:** [github.com/rahul370139/PathWise](https://github.com/rahul370139/PathWise)

---

## 1. Project Overview

### STAR Summary

| Component | Detail |
|-----------|--------|
| **Situation** | Learners lack a unified platform that combines personalized learning content generation with AI-powered career guidance. Existing tools either do content or career advice, never both intelligently together. |
| **Task** | Build a full-stack AI platform that ingests PDFs to generate multi-format learning content (summaries, quizzes, flashcards, micro-lessons, concept maps) and provides RIASEC-based career matching with personalized roadmaps -- all powered by LLM and semantic search. |
| **Action** | Designed and built a 10,687-line Python backend (FastAPI) with RAG architecture using Groq LLM + Cohere embeddings + Supabase PostgreSQL. Implemented map-reduce summarization, LRU caching with TTL, dual API key failover, and a React/Next.js frontend. Deployed backend on Railway and frontend on Vercel. |
| **Result** | Live production platform with PDF-to-learning pipeline (upload to content in ~8s), 10-question RIASEC career quiz matching against O*NET/BLS career dataset, multi-format content generation (summaries, quizzes up to 20 Qs, flashcards, workflows, concept maps), and comprehensive career roadmaps with skill gap analysis. |

### Architecture Diagram

```
                        PathWise Architecture
 ============================================================================

   [User Browser]
        |
        | HTTPS
        v
 +------------------+         +-------------------------------------------+
 |  Vercel (CDN)    |         |        Railway (Backend)                  |
 |  React / Next.js |  REST   |  +------------------------------------+  |
 |  Frontend        |-------->|  |     FastAPI  (main.py - 2,292 L)   |  |
 |                  |<--------|  |     CORS Middleware                 |  |
 +------------------+         |  +----+--------+--------+--------+----+  |
                              |       |        |        |        |       |
                              |       v        v        v        v       |
                              |  +--------+ +------+ +------+ +-------+  |
                              |  |Distiller| |Career| |Study | |Dash-  |  |
                              |  |1,791 L  | |Matcher| |Agent | |board  |  |
                              |  |PDF+RAG  | |875 L | |584 L | |584 L  |  |
                              |  +---+-----+ +--+---+ +--+---+ +---+---+  |
                              |      |          |        |          |      |
                              +------|----------|--------|----------|------+
                                     |          |        |          |
                 +-------------------+----------+--------+----------+
                 |                   |                    |
                 v                   v                    v
          +-----------+      +-----------+        +-----------+
          | Groq API  |      | Cohere    |        | Supabase  |
          | llama-3.3 |      | embed-    |        | PostgreSQL|
          | 70b       |      | english-  |        | + pgvector|
          | temp=0.3  |      | light-v3.0|        |           |
          +-----------+      | 384-dim   |        +-----------+
                             +-----------+
```

### Full Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Backend** | FastAPI + Python | 2,292 lines (main.py), 10,687 total across 13 modules |
| **LLM** | Groq API | `llama-3.3-70b-versatile`, temperature 0.3, dual API keys |
| **Embeddings** | Cohere API | `embed-english-light-v3.0`, 384 dimensions |
| **Database** | Supabase (PostgreSQL) | Tables: `lessons`, `lesson_metadata`, `concept_maps`, `lesson_completions`, `user_roles`, `mastery`, `doc_concepts` |
| **Frontend** | React / Next.js | Deployed on Vercel CDN |
| **Backend Deploy** | Railway | Uvicorn ASGI server, `Procfile` config |
| **Frontend Deploy** | Vercel | Automatic CI/CD from GitHub |
| **PDF Parsing** | PyMuPDF (fitz) | Text extraction from uploaded PDFs |
| **ML/Data** | pandas, numpy, scikit-learn | Career data processing, cosine similarity |
| **Validation** | Pydantic v2 | 497 lines of schemas, 35+ model classes |
| **Logging** | Loguru | Structured logging across all modules |
| **HTTP Client** | httpx | Async HTTP for Groq/Cohere API calls |
| **Auth** | Supabase Auth + UUID normalization | Deterministic UUID derivation for non-UUID user IDs |

---

## 2. Deep Technical Walkthrough

### 2.1 FastAPI Backend Design (2,292 lines)

**Endpoint Structure (~45+ endpoints):**

| Category | Endpoints | Purpose |
|----------|----------|---------|
| **Learn/Distill** | `POST /api/distill`, `GET/POST /api/lesson/{id}/{action}` | PDF upload, content generation |
| **Chat** | `POST /api/chat`, `POST /api/chat/upload`, `GET /api/chat/conversations/{user_id}` | AI chatbot with RAG |
| **Career** | `GET /api/career/quiz`, `POST /api/career/match`, `GET /api/career/roadmap/{title}` | RIASEC quiz, career matching |
| **User** | `PUT /api/users/{id}/role`, `GET /api/users/{id}/progress` | Role management, progress tracking |
| **Dashboard** | `POST /api/dashboard/recommendations`, `GET /api/dashboard/analytics/{id}` | Personalized analytics |
| **Agent** | `POST /api/agent/route`, `POST /api/agent/summary`, `POST /api/agent/flashcards` | Agentic AI system |
| **Unified Career** | `POST /api/career/roadmap/unified`, `POST /api/career/comprehensive-plan` | Full career planning |

**Key Design Patterns:**

```python
# Startup hook: precompute micro-lesson embeddings at boot
@app.on_event("startup")
async def _startup():
    total, embedded = await precompute_micro_lessons_embeddings()
    logger.info(f"Micro-lessons precomputed: {embedded}/{total}")

# CORS middleware allows Vercel frontend + localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://v0-frontend-opal-nine.vercel.app", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Pydantic Schemas (497 lines, 35+ models):**

Includes `ExplanationLevel` enum (5-year-old, intern, senior), `Framework` enum (16 frameworks: fastapi, docker, python, ML, react, nextjs, etc.), request/response models for every endpoint with `Field` validators (`min_length`, `max_length`, `ge`, `le`).

```python
class CareerMatchRequest(BaseModel):
    owner_id: str
    answers: List[int] = Field(..., min_length=10, max_length=10)
    user_profile: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    type: Optional[str] = None  # chat|lesson|quiz|flashcards|workflow|summary
    quiz_data: Optional[Dict[str, Any]] = None
    flashcard_data: Optional[Dict[str, Any]] = None
    # ... 15+ optional fields for different content types
```

### 2.2 PDF Processing Pipeline

```
 Upload PDF
     |
     v
 [PyMuPDF text extraction] ── fitz.open() → page.get_text()
     |                         Min 10 chars validation, 50MB limit
     v
 [Chunking] ── 400-word chunks, 50-word overlap (sliding window)
     |          Words split → accumulate → emit chunk → keep last 50
     v
 [Embedding] ── Cohere embed-english-light-v3.0 (384-dim)
     |           Batch processing, 32 texts per API call
     |           Fallback: heuristic embedding (technical/framework/learning scores)
     v
 [Content Dedup] ── SHA-256 hash of full text
     |               Skip re-processing if hash seen before
     v
 [Parallel Generation] ── asyncio.gather()
     |
     +──> map_reduce_summary()     → Bullet-point summary (max 10)
     +──> gen_flashcards_quiz()    → Flashcards + MCQ quiz
     +──> generate_concept_map()   → Nodes + edges graph
     +──> detect_framework()       → Auto-detect (16 frameworks)
     |
     v
 [Store to Supabase] ── lessons, lesson_metadata, concept_maps tables
     |                   + in-memory LRU cache
     v
 Return { lesson_id, actions: ["summary","lesson","quiz","flashcards","workflow"], preview }
```

**Chunking Implementation (from `distiller.py`):**

```python
CHUNK_WORDS = 400
OVERLAP = 50

def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if len(cur) >= CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur = cur[-OVERLAP:]  # sliding window overlap
    if cur:
        chunks.append(" ".join(cur))
    return chunks
```

**Why 400 words with 50-word overlap?**
- 400 words fits within LLM context windows while being semantically cohesive
- 50-word overlap prevents information loss at chunk boundaries
- Trade-off: more overlap = better context continuity but more redundant embeddings

### 2.3 RAG Implementation

```
 User Query (e.g., "Generate quiz about machine learning")
     |
     v
 [Query Embedding] ── Cohere embed (input_type="search_document")
     |
     v
 [Vector Search] ── Cosine similarity against all chunk embeddings
     |                Manual implementation (dot product / magnitudes)
     |                find_similar_content(query_embed, content_embeds, top_k=6)
     v
 [Top-6 Chunks Retrieved] ── Sorted by similarity descending
     |
     v
 [Prompt Construction] ── System prompt + retrieved context + user query
     |
     v
 [Groq LLM Generation] ── llama-3.3-70b-versatile, temp=0.3
     |                      JSON output with _parse_json_safely()
     v
 [Structured Response] ── Quiz, flashcards, summary, etc.
```

**Cosine Similarity (hand-implemented):**

```python
def calculate_embedding_similarity(embedding1, embedding2) -> float:
    min_length = min(len(embedding1), len(embedding2))
    dot_product = sum(embedding1[i] * embedding2[i] for i in range(min_length))
    magnitude1 = sum(val ** 2 for val in embedding1[:min_length]) ** 0.5
    magnitude2 = sum(val ** 2 for val in embedding2[:min_length]) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return max(-1.0, min(1.0, dot_product / (magnitude1 * magnitude2)))
```

**Why hand-implemented cosine similarity instead of numpy/sklearn?**
- Avoids heavy library dependency for a simple operation
- Works with variable-length embeddings (defensive programming)
- Clamps output to `[-1, 1]` for robustness

### 2.4 Map-Reduce Summarization Pattern

```
 PDF Chunks: [Chunk₁, Chunk₂, ..., Chunkₙ]
     |
     | MAP PHASE (parallel via asyncio.gather)
     v
 [Groq LLM] "Summarize into 3 concise bullets"
     |         Runs for each chunk concurrently
     v
 Mapped Summaries: [Summary₁, Summary₂, ..., Summaryₙ]
     |
     | REDUCE PHASE
     v
 [Groq LLM] "Merge into max 10 key bullets"
     |         Single call combining all mapped results
     v
 Final Bullet-Point Summary (•-delimited)
```

```python
async def map_reduce_summary(chunks, explanation_level):
    async def summarize_chunk(chunk):
        messages = [
            {"role": "system", "content": f"Summarize into 3 concise bullets. {explanation_prompt}"},
            {"role": "user", "content": chunk},
        ]
        return await call_groq(messages)

    mapped = await asyncio.gather(*[summarize_chunk(ch) for ch in chunks])

    reduce_prompt = (
        f"Merge these bullet groups into **max 10 key bullets**:\n"
        + "\n".join(mapped)
    )
    return await call_groq([{"role": "user", "content": reduce_prompt}])
```

**Why map-reduce?**
- Individual chunks fit within Groq's context window
- Parallel map phase exploits asyncio for speed
- Reduce phase merges into cohesive, non-redundant summary
- Graceful fallback if LLM fails: returns truncated chunk previews

### 2.5 Content Generation System

| Content Type | Generation Method | Output Structure |
|-------------|-------------------|-----------------|
| **Summary** | Map-reduce → max 10 bullets | `List[str]` (bullet-delimited) |
| **Quiz** | RAG + LLM → structured JSON | `{question, options: [A,B,C,D], answer}` |
| **Flashcards** | RAG + LLM → structured JSON | `{front, back}` |
| **Concept Map** | LLM → graph structure | `{nodes: [{id, label, level}], edges: [{from, to, label}]}` |
| **Workflow** | LLM → step sequence | `{steps: [{step, title, description, duration, best_practices}]}` |
| **Lesson Plan** | RAG + LLM → learning path | `{title, overview, difficulty, learning_topics, learning_path}` |

**Variable-count generation:** Users can request "make 15 flashcards" -- the system extracts the count via regex and passes `num_items` (clamped 1-20):

```python
def _extract_desired_count(message: str) -> Optional[int]:
    m = re.search(r"(\d{1,2})\s*(flashcard|quiz|question|cards)?", message.lower())
    return int(m.group(1)) if m else None
```

### 2.6 Career Matching: RIASEC Framework

**10-Question Quiz with 6 RIASEC Dimensions:**

```
 ┌──────────────────────────────────────────────────────┐
 │  RIASEC Dimensions:                                   │
 │  R = Realistic    │ I = Investigative │ A = Artistic  │
 │  S = Social       │ E = Enterprising  │ C = Conventional│
 └──────────────────────────────────────────────────────┘

 Each question has 6 options, each mapping to weighted RIASEC scores.
 Example: "Working with my hands" → {R:5, I:1, A:1, S:1, E:1, C:1}

 User Answers (10) → RIASEC Vector (6D) → Normalize to unit length
     |
     v
 Career Matching Pipeline:
     ├── [Embedding-Based] User profile → Cohere embed → Cosine sim vs careers
     │     Weighted: 50% semantic + 30% skill match + 20% interest match
     │
     └── [Fallback: Basic RIASEC] Cosine sim of RIASEC vectors
           Uses O*NET/BLS career dataset (pandas DataFrame)
           Diversity filter to avoid similar career titles
```

**Career data source:** O*NET/BLS dataset (`onet_bls_trimmed.csv`) with fields: `title`, `salary_low`, `salary_high`, `growth_pct`, `top_skills`, `day_in_life`, and RIASEC dimension scores.

### 2.7 Caching: LRU with TTL

```python
# In distiller.py -- uses collections.OrderedDict for O(1) LRU
lesson_store: OrderedDict[str, Dict] = OrderedDict()
LESSON_CACHE_TTL_SECONDS = 60 * 60 * 2  # 2 hours
LESSON_CACHE_CAPACITY = 50

def set_lesson_cache(lesson_id: str, data: Dict):
    data["cached_at"] = datetime.utcnow().isoformat()
    if lesson_id in lesson_store:
        lesson_store.pop(lesson_id, None)  # Remove old position
    lesson_store[lesson_id] = data          # Insert at end (MRU)
    _lesson_cache_evict_if_needed()         # Evict oldest if > 50

def get_lesson_cache(lesson_id: str) -> Optional[Dict]:
    rec = lesson_store.get(lesson_id)
    if not rec: return None
    if _is_expired(rec):  # Check 2-hour TTL
        lesson_store.pop(lesson_id, None)
        return None
    # Touch: move to end (MRU position)
    lesson_store.pop(lesson_id, None)
    lesson_store[lesson_id] = rec
    return rec
```

**Three-tier content resolution:**
1. In-memory LRU cache (fastest, populated on upload)
2. Supabase database (persistent, cross-instance)
3. On-demand LLM generation (fallback, slowest)

### 2.8 Error Handling & Resilience

**Dual API Key Failover:**

```python
async def call_groq(messages):
    primary_key = os.getenv('GROQ_API_KEY')
    fallback_key = os.getenv('GROQ_API_KEY_2')

    if primary_key:
        result = await _call_groq_with_key(messages, primary_key, "primary")
        if result: return result

    if fallback_key:
        result = await _call_groq_with_key(messages, fallback_key, "fallback")
        if result: return result

    return ""
```

**Exponential Backoff with Rate Limit Detection:**

```python
# In _call_groq_with_key():
max_retries = 2
backoff = 1.5
# On 429 rate limit:
m = re.search(r"try again in ([0-9.]+)s", error_message)
wait_s = min(float(m.group(1)) + 0.5, 15.0)  # Parse server hint
await asyncio.sleep(wait_s)
backoff *= 2  # Exponential increase
```

**Concurrency Control:**

```python
_LLM_SEMAPHORE = asyncio.Semaphore(2)  # Max 2 concurrent LLM calls

async with _LLM_SEMAPHORE:
    # API call here -- prevents overwhelming Groq rate limits
```

**Graceful Degradation Chain:**

```
 Supabase available?
     ├── YES → Read from database
     └── NO → In-memory fallback
              ├── Cache hit? → Return cached
              └── Cache miss? → Generate on-demand with LLM
                                  ├── LLM success? → Return + cache
                                  └── LLM failure? → Return static fallback content
```

**Supabase graceful degradation (every function follows this pattern):**

```python
def insert_lesson(owner_id, title, summary, ...):
    global DUMMY_ID_COUNTER
    if not SUPA:  # Supabase unavailable
        DUMMY_ID_COUNTER += 1
        return DUMMY_ID_COUNTER  # Return unique ID, don't crash
    try:
        res = SUPA.table("lessons").insert(data).execute()
        return res.data[0]["id"]
    except Exception:
        DUMMY_ID_COUNTER += 1
        return DUMMY_ID_COUNTER  # Fallback on any error
```

### 2.9 Robust JSON Parsing

LLMs don't always return valid JSON. The `_parse_json_safely()` function handles:
- Markdown code fences (`\`\`\`json ... \`\`\``)
- Largest `{...}` block extraction
- Single quotes → double quotes conversion
- Trailing comma removal
- Fallback to None (caller provides defaults)

### 2.10 Frontend: React/Next.js on Vercel

- **Learn Page:** PDF upload → view summaries, quizzes, flashcards, workflows, concept maps
- **Career Page:** 10-question RIASEC quiz → top 5 career matches with roadmaps
- **Dashboard:** Progress tracking, analytics, personalized recommendations
- **Chat:** AI chatbot with PDF context (RAG-powered)
- Deployed on Vercel with automatic CI/CD from GitHub

### 2.11 Database: Supabase Schema

| Table | Key Columns | Purpose |
|-------|------------|---------|
| `lessons` | id, owner (UUID), title, summary, framework, explanation_level, full_text | Processed PDF lessons |
| `lesson_metadata` | lesson_id, card_type (bullet/flashcard/quiz), payload (JSONB), embed_vector | Learning content with embeddings |
| `concept_maps` | lesson_id, nodes (JSONB), edges (JSONB) | Graph-structured concept maps |
| `lesson_completions` | user_id, lesson_id, progress_percentage, completed_at | Progress tracking |
| `user_roles` | user_id (UUID), role, experience_level, interests | User profile preferences |
| `mastery` | user_id, skill, score | Skill mastery tracking |
| `doc_concepts` | pdf_id, user_id, concept, frequency, page_references | Extracted document concepts |

### 2.12 Agentic AI System

The project includes an agent-based architecture with:

- **AgentRouter:** Intent detection from user messages (summary, quiz, flashcards, diagnostic)
- **SummarizerAgent:** Topic-filtered summaries with concept maps and page references
- **DiagnosticAgent:** Full diagnostic cycle -- quiz generation, answer evaluation, mastery updates
- **Validators:** `qa_flashcards()`, `qa_quiz()` -- structural validation of LLM output
- **Repairs:** `repair_flashcards()`, `repair_quiz()` -- re-generation when validation fails
- **Mastery Tracking:** Per-skill scores updated after diagnostic assessments

---

## 3. Key Metrics & Results

| Metric | Value |
|--------|-------|
| **Total Backend Code** | 10,687 lines across 13 Python modules |
| **Main API** | 2,292 lines, ~45+ REST endpoints |
| **Pydantic Schemas** | 497 lines, 35+ validated model classes |
| **Live Deployment** | Frontend on Vercel, Backend on Railway |
| **Content Types** | 6 (summaries, quizzes, flashcards, workflows, concept maps, lesson plans) |
| **Career Dataset** | O*NET/BLS career data with RIASEC scores, salaries, growth rates |
| **Quiz Questions** | 10 career assessment questions, 6 options each (RIASEC-weighted) |
| **Embedding Dimensions** | 384 (Cohere embed-english-light-v3.0) |
| **Cache Capacity** | 50 lessons, 2-hour TTL |
| **LLM Model** | llama-3.3-70b-versatile via Groq (fast inference) |
| **Frameworks Supported** | 16 auto-detected (FastAPI, React, Python, Docker, ML, etc.) |
| **Error Handling** | Dual API keys, exponential backoff, 3-tier fallback, semaphore rate limiting |

---

## 4. Topics You Must Know

### 4.1 FastAPI Internals

- **Starlette foundation:** FastAPI is built on Starlette (ASGI). Uvicorn serves as the ASGI server.
- **Pydantic v2 validation:** All request/response models use Pydantic BaseModel with Field validators. Response models enforce schema at serialization time (`response_model=CareerMatchResponse`).
- **Async processing:** All LLM calls use `async/await` with `asyncio.gather()` for parallel processing. `asyncio.Semaphore(2)` limits concurrent LLM calls.
- **CORS middleware:** Configured to allow Vercel frontend origins + localhost for development.
- **Dependency injection:** Query parameters with `Query(...)`, file uploads with `File(...)`, form data with `Form(...)`.
- **Startup events:** `@app.on_event("startup")` precomputes micro-lesson embeddings at boot.

### 4.2 RAG Pipeline

- **Indexing:** PDF → text → chunks → embeddings → store in cache + Supabase
- **Retrieval:** Query → embed → cosine similarity against stored embeddings → top-k chunks
- **Generation:** Retrieved chunks + query → LLM prompt → structured output
- **Why top-6?** Balance between context richness and token budget. More chunks = more context but risk of noise.

### 4.3 Cohere Embeddings

- **Model:** `embed-english-light-v3.0` -- lightweight, fast, 384 dimensions
- **Input type:** `search_document` for content indexing
- **Batch processing:** 32 texts per API call to respect rate limits
- **Fallback:** Heuristic embeddings based on keyword density (technical, framework, learning terms)

### 4.4 Cosine Similarity

- Measures angle between vectors: `cos(theta) = (A . B) / (||A|| * ||B||)`
- Range: `[-1, 1]` where 1 = identical direction
- Used for: chunk retrieval, career matching, micro-lesson selection
- Hand-implemented for lightweight operation without numpy dependency in hot path

### 4.5 PDF Processing with PyMuPDF

- `fitz.open(path)` → iterate pages → `page.get_text()` → concatenate
- Validates minimum 10 characters of extracted text
- 50MB file size limit enforced at upload
- Temp file cleanup in `finally` block

### 4.6 LRU Caching with TTL

- `collections.OrderedDict` provides O(1) insertion-order tracking
- **LRU:** Most recently accessed item moved to end; eviction from front
- **TTL:** Each entry stamped with `cached_at`; expired entries removed on access
- **Capacity:** 50 lessons max; eviction when exceeded

### 4.7 Map-Reduce Pattern

- **Map:** Parallel LLM calls on each chunk (3 bullets each)
- **Reduce:** Single LLM call merging all bullet groups into max 10 key bullets
- **Why?** Handles documents of any length by processing chunks independently, then synthesizing
- **Analogy:** Like MapReduce in distributed computing -- partition work, process in parallel, combine results

### 4.8 Supabase Integration

- PostgreSQL database accessed via `supabase-py` client
- Tables use JSONB for flexible payload storage (lesson cards, concept maps)
- UUID normalization: non-UUID user IDs converted via `uuid5(NAMESPACE_DNS, "pathwise:{id}")`
- Every database function has graceful degradation: returns sensible defaults when Supabase unavailable

### 4.9 RIASEC Career Framework

- **Holland's RIASEC model:** 6 personality types mapped to career preferences
- **R**ealistic, **I**nvestigative, **A**rtistic, **S**ocial, **E**nterprising, **C**onventional
- 10 questions × 6 options = weighted RIASEC vector → normalized to unit length → cosine similarity against career dataset
- Enhanced with embedding-based matching: 50% semantic similarity + 30% skill match + 20% interest match

### 4.10 Vercel / Railway Deployment

- **Vercel:** React/Next.js frontend, automatic deploys from GitHub, CDN edge caching
- **Railway:** Python FastAPI backend, `Procfile: web: python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variables:** API keys (Groq, Cohere), Supabase credentials stored as Railway/Vercel env vars
- **CORS:** Frontend origin allowlisted in FastAPI middleware

---

## 5. Interview Questions & Model Answers

### Architecture & Design (Q1-Q7)

**Q1: Walk me through the system architecture of PathWise.**

> PathWise uses a decoupled architecture with a React/Next.js frontend on Vercel and a FastAPI backend on Railway. The backend is organized into specialized modules: `distiller.py` handles PDF processing and RAG-based content generation, `career_matcher.py` manages the RIASEC-based career quiz and matching, `unified_career_system.py` generates comprehensive career roadmaps, and `supabase_helper.py` abstracts all database operations.
>
> When a user uploads a PDF, it flows through PyMuPDF for text extraction, gets chunked into 400-word segments with 50-word overlap, embedded via Cohere's API, and then we run parallel content generation (summary, quiz, flashcards, concept map) using Groq's LLM. Everything is stored in Supabase PostgreSQL and cached in-memory with an LRU cache (50 lessons, 2-hour TTL). For career matching, the user's 10-question RIASEC quiz answers are converted to a 6D vector and matched against an O*NET/BLS career dataset using a combination of embedding-based semantic similarity and skill/interest matching.

**Q2: Why did you choose FastAPI over Flask or Django?**

> Three reasons: First, **native async support** -- our system makes multiple concurrent API calls to Groq and Cohere, and FastAPI's async/await integration with Starlette handles this elegantly. Second, **Pydantic integration** -- with 35+ request/response models, Pydantic's automatic validation and serialization saved enormous development time and caught bugs at the schema level. Third, **performance** -- FastAPI on Uvicorn (ASGI) handles concurrent requests much better than Flask's WSGI model, which matters when each request involves multiple LLM API calls.

**Q3: Why Groq with llama-3.3-70b instead of OpenAI GPT-4?**

> **Cost and speed.** Groq provides extremely fast inference through their LPU architecture, making it practical for a real-time learning platform where users expect content generation in seconds, not minutes. The `llama-3.3-70b-versatile` model offers strong instruction-following and JSON output capabilities at a fraction of GPT-4's cost. I set temperature to 0.3 to get more deterministic, structured JSON outputs rather than creative but unparseable responses. The dual API key system with automatic failover also provides resilience -- if one key hits rate limits, the system seamlessly switches to the backup.

**Q4: Explain your chunking strategy. Why 400 words with 50-word overlap?**

> The 400-word chunk size is a deliberate trade-off. It's large enough to capture a coherent idea or paragraph section -- most technical concepts can be explained in 300-500 words -- but small enough to fit within embedding model context windows and LLM prompt budgets. The 50-word overlap (12.5%) prevents information loss at chunk boundaries. If a concept spans two chunks, the overlap ensures both chunks retain the transitional context. I chose word-based splitting over sentence-based because it's more predictable for buffer management, and PDFs often have inconsistent sentence boundaries.

**Q5: How does your three-tier content resolution work?**

> When the frontend requests content (say, a quiz for lesson #42), the backend checks three tiers in order:
> 1. **In-memory LRU cache** -- O(1) lookup in the OrderedDict. If the lesson was recently processed, it's here.
> 2. **Supabase database** -- persistent storage. If the server restarted but the lesson exists in the DB, we read from the `lesson_metadata` table.
> 3. **On-demand LLM generation** -- if neither cache nor DB has the content, we regenerate using the lesson's stored summary + RAG retrieval against cached chunks, then cache the result for next time.
>
> This approach gives us fast responses (cache hit: <50ms), durability (DB survives restarts), and resilience (LLM generation as the ultimate fallback).

**Q6: How do you handle concurrent API calls and rate limiting?**

> I use `asyncio.Semaphore(2)` to limit concurrent Groq LLM calls to 2. This prevents overwhelming Groq's rate limits while still allowing parallelism. The map-reduce summarization uses `asyncio.gather()` to process chunks in parallel, but the semaphore ensures at most 2 are hitting the API simultaneously. For Cohere embeddings, I batch requests (32 texts per call) to minimize API round trips. If a 429 rate limit response comes back, I parse the server's "try again in Xs" header and sleep accordingly, with exponential backoff capped at 15 seconds.

**Q7: What's your approach to JSON parsing from LLM outputs?**

> LLMs are unreliable JSON generators, so I built a robust `_parse_json_safely()` function that: (1) strips markdown code fences, (2) finds the largest `{...}` block in the response, (3) replaces single quotes with double quotes if no double quotes exist, (4) removes trailing commas before `}` or `]`, and (5) returns `None` on failure so callers can use fallback content. Every content generation function has a fallback -- for example, if quiz generation fails, we return a set of generic but well-structured quiz questions. The user never sees an error; they just get slightly less personalized content.

### RAG & ML (Q8-Q12)

**Q8: Walk me through a complete RAG query.**

> Let's say a user in the chat types "Generate quiz about machine learning."
> 1. We extract the topic "machine learning" using regex-based command detection.
> 2. We embed the topic using Cohere's API (`input_type="search_document"`).
> 3. We calculate cosine similarity between this query embedding and all stored chunk embeddings from the user's uploaded PDF.
> 4. We retrieve the top 6 most similar chunks -- these contain the most relevant passages about machine learning from their document.
> 5. We construct a prompt: system instructions + retrieved chunks as context + request for quiz in specific JSON format.
> 6. Groq's LLM generates the quiz using the retrieved context as grounding.
> 7. We parse the JSON, validate the structure, cache the result, and return it.
>
> The key insight is that retrieval grounds the LLM in the user's actual document content rather than its training data, producing relevant, specific quizzes.

**Q9: Why Cohere for embeddings instead of OpenAI or a local model?**

> Cohere's `embed-english-light-v3.0` offers an excellent trade-off: 384 dimensions (compact, fast to compare), high quality for English text, and a generous free tier. Compared to OpenAI's `text-embedding-3-small` (1536 dimensions), Cohere's smaller vectors mean faster cosine similarity computation and less memory for caching. I also liked the explicit `input_type` parameter -- `search_document` for indexing vs `search_query` for retrieval -- which improves retrieval quality. For fallback, I implemented a heuristic embedding based on keyword density (technical terms, framework mentions, learning terms), so the system works even if Cohere's API is down.

**Q10: How does your career matching combine embedding similarity with RIASEC scores?**

> It's a weighted ensemble approach:
> 1. **Semantic similarity (50% weight):** We embed the user's comprehensive profile (RIASEC scores, skills, interests, work preferences, learning style, career goals) using Cohere, and compare it against embeddings of each career's description. This captures nuanced semantic relationships.
> 2. **Skill match (30% weight):** We tokenize the user's extracted skills and the career's `top_skills` field, computing overlap ratio.
> 3. **Interest match (20% weight):** We check how many of the user's interests appear in the career description text.
>
> The combined score: `0.5 * semantic + 0.3 * skill_match + 0.2 * interest_match`. If embedding similarity is too low (< 0.3), we fall back to pure RIASEC vector cosine similarity, which is simpler but still effective.

**Q11: Why did you implement cosine similarity by hand instead of using scipy/sklearn?**

> For the hot path in `distiller.py` where we do retrieval on every chat message and content generation request, I wanted zero external dependency overhead. The function is simple -- dot product divided by magnitude product -- and adding numpy/scipy just for this would add startup time and memory. In `career_matcher.py`, where we process the full career dataset, I do use numpy and sklearn for batch operations. It's about choosing the right tool for the right context.

**Q12: How do you handle embedding failures?**

> Every embedding function has a fallback path. If Cohere's API fails, I generate a heuristic 384-dimensional vector based on:
> - Dimensions 0-2: keyword density scores (technical terms, framework terms, learning terms)
> - Dimensions 3-4: text length and word density
> - Dimensions 5-383: seeded random values (deterministic based on text hash for consistency)
>
> These fallback embeddings are less semantically meaningful but still provide differentiation between different types of content, so the system remains functional.

### System Design & Production (Q13-Q18)

**Q13: How would you scale this system to handle 10,000 concurrent users?**

> Several changes needed:
> 1. **Replace in-memory cache with Redis** -- the current OrderedDict doesn't survive restarts or share across instances. Redis provides distributed caching with built-in TTL.
> 2. **Use pgvector for vector search** -- instead of loading all embeddings into memory, store them in Supabase's pgvector extension and do similarity search at the database level.
> 3. **Horizontal scaling** -- Railway supports multiple instances, but we'd need to remove all in-memory state (conversation_store, lesson_store) and move them to Redis/database.
> 4. **Message queue for PDF processing** -- heavy PDF processing should go through a task queue (Celery/Redis) instead of blocking the API server.
> 5. **CDN for static assets** -- Vercel already handles this for the frontend.

**Q14: What are the security implications of your CORS configuration?**

> Currently, the CORS config includes `"*"` (allow all origins) which is intentional for development but should be restricted in production. In a production hardening pass, I'd:
> - Remove `"*"` and only allow the specific Vercel domain
> - Add rate limiting per IP/user (e.g., using `slowapi`)
> - Validate file uploads more rigorously (magic bytes, not just extension)
> - Add authentication middleware (Supabase JWT verification)
> - Sanitize all user inputs before passing to LLM prompts (prompt injection prevention)

**Q15: How does content deduplication work?**

> When a PDF is uploaded, I compute a SHA-256 hash of the full extracted text. I maintain a `content_hash_to_lesson_id` mapping in memory. If the same PDF content is uploaded again (by the same or different user), we skip the expensive processing pipeline and return the existing `lesson_id`. This saves LLM API costs and provides instant responses for re-uploads.

**Q16: How do you ensure the LLM generates valid structured output?**

> Multi-layered approach:
> 1. **Prompt engineering:** I specify exact JSON structure in the prompt with examples and explicit "Return ONLY valid JSON (no prose, no markdown)."
> 2. **Temperature 0.3:** Lower temperature reduces creative deviations from the requested format.
> 3. **Robust parsing:** `_parse_json_safely()` handles markdown fences, single quotes, trailing commas, and extracts the largest JSON object.
> 4. **Validation:** For the agentic system, `qa_flashcards()` and `qa_quiz()` validate structural requirements.
> 5. **Repair:** If validation fails, `repair_flashcards()` and `repair_quiz()` re-prompt the LLM with error feedback.
> 6. **Fallback content:** Every generation function has a static fallback that returns well-structured default content.

**Q17: Explain your conversation management system.**

> I maintain an in-memory `conversation_store` dict keyed by conversation ID. Each conversation holds:
> - `messages`: chat history (role + content + timestamp)
> - `file_context`: full PDF text if uploaded
> - `chunks` and `chunk_embeddings`: for RAG retrieval in chat
> - `metadata`: current PDF info, framework, explanation level
>
> I also keep a `last_conversation_by_user` mapping so if the frontend forgets to pass a conversation ID (which happened), we can automatically resume the user's most recent conversation. For the chat, I send the last 10 messages as context to the LLM plus the RAG-retrieved chunks, creating a context-aware conversational experience.

**Q18: What monitoring/observability do you have in production?**

> - **Loguru** structured logging throughout every module with `logger.info()`, `logger.error()`, `logger.warning()`
> - **Debug endpoints:** `/api/debug/supabase/check` tests database connectivity, `/api/debug/lesson/{id}` tests all content generation
> - **Health check:** `/health` endpoint for Railway's health monitoring
> - **Error tracking:** Every exception is logged with context (lesson ID, user ID, API key type)
> - For improvement: I'd add Sentry for error aggregation, Prometheus metrics for latency/throughput, and structured JSON logging for ELK stack analysis.

### Behavioral / Deep Dive (Q19-Q25)

**Q19: What was the hardest technical challenge you faced?**

> **Reliable JSON extraction from LLM outputs.** The LLM would randomly wrap JSON in markdown code fences, use single quotes, add trailing commas, or include explanatory text before/after the JSON. This broke the entire content generation pipeline. I built `_parse_json_safely()` as a multi-stage parser: strip fences, find largest `{...}` block, repair common issues, then parse. Combined with the repair agents that re-prompt on validation failure, we achieved >95% successful structured output generation. This taught me that LLM integration isn't just prompt engineering -- it's building robust output processing.

**Q20: What would you do differently if you started over?**

> 1. **Use a proper vector database** (Pinecone or pgvector from day one) instead of in-memory embeddings. The current approach doesn't scale beyond a single server instance.
> 2. **Implement proper authentication** from the start. The current system uses user IDs but lacks JWT verification.
> 3. **Use a task queue** for PDF processing instead of processing synchronously in the API handler. Long PDFs can time out.
> 4. **Add comprehensive tests** -- I have test files but not comprehensive unit/integration test coverage.
> 5. **Use a streaming response** for LLM-generated content so users see incremental results instead of waiting for full generation.

**Q21: How did you decide on the map-reduce pattern for summarization?**

> I initially tried sending the entire PDF text to the LLM, but longer documents exceeded context windows and produced inconsistent summaries. I explored three approaches:
> 1. **Truncation** -- just send the first N tokens. Lost information from later sections.
> 2. **Refine** -- sequential summarization where each chunk's summary builds on the previous. Worked but was slow (sequential, not parallel).
> 3. **Map-reduce** -- parallel per-chunk summaries, then a single reduction step. Won on both quality and speed.
>
> The map phase exploits asyncio parallelism, and the reduce phase produces a cohesive final summary. The trade-off is slightly more API calls, but with Groq's fast inference, the total latency is actually lower than the sequential refine approach.

**Q22: How does the explanation level system work?**

> Three levels: `5_year_old`, `intern`, and `senior`. Each maps to a system prompt modifier:
> - **5-year-old:** "Explain like you're talking to a 5-year-old. Use simple words, analogies."
> - **Intern:** "Explain for someone learning with basic knowledge. Use clear examples."
> - **Senior:** "Explain for an experienced professional. Use technical terms."
>
> This prompt is injected into every LLM call -- summaries, quizzes, flashcards, chat responses. The user can change levels dynamically in the chat ("explain like 5"), and the preference persists per conversation. It's a simple but effective way to personalize content without changing the underlying architecture.

**Q23: Walk me through how the agentic system works.**

> The agentic system adds a layer of intelligence on top of the core content generation:
> 1. **AgentRouter** detects user intent from natural language (summary, quiz, flashcards, diagnostic).
> 2. **SummarizerAgent** generates topic-filtered summaries with concept maps.
> 3. **DiagnosticAgent** runs a full assessment cycle: generates quiz → user answers → evaluates performance → updates mastery scores in Supabase.
> 4. **Validators** (`qa_flashcards`, `qa_quiz`) check structural validity of LLM output.
> 5. **Repairs** (`repair_flashcards`, `repair_quiz`) re-prompt with error feedback if validation fails.
> 6. **Mastery** tracks per-skill scores, enabling personalized difficulty adjustment.
>
> The key pattern is: generate → validate → repair if needed → return. This gives us much more reliable output than single-shot LLM generation.

**Q24: How do you handle the cold start problem?**

> Two strategies:
> 1. **Startup precomputation:** The `@app.on_event("startup")` hook precomputes embeddings for all micro-lessons at boot time. This means career roadmap recommendations with relevant micro-lesson suggestions are instant from the first request.
> 2. **Lazy initialization with caching:** Career data (pandas DataFrame) and quiz questions (JSON) are loaded once at module import time. The Supabase client is initialized at import. Only LLM calls are lazy (on-demand), but they're fast thanks to Groq's LPU infrastructure.

**Q25: How do you prevent prompt injection?**

> Currently, user messages are passed to the LLM with minimal sanitization -- this is a known gap. In production, I would:
> 1. Use a separate system prompt that explicitly instructs the LLM to ignore user attempts to override its role
> 2. Validate and sanitize user inputs (strip control characters, limit length)
> 3. Use output validation to detect when the LLM has been manipulated
> 4. Consider a prompt injection detection classifier as a pre-filter

---

## 6. Red Flags & How to Handle

| Red Flag | How to Handle |
|----------|---------------|
| **"CORS allows `*` -- isn't that insecure?"** | Acknowledge it's for development convenience. In production, I'd restrict to the specific Vercel domain and add rate limiting. The current setup prioritizes demo accessibility. |
| **"In-memory storage doesn't scale"** | Agree. Explain the three-tier fallback (cache → DB → generate) and that the in-memory layer is a performance optimization, not the single source of truth. Supabase provides persistence. For multi-instance scaling, I'd use Redis. |
| **"No authentication on API endpoints"** | Acknowledge the gap. The system uses user IDs for data isolation but lacks JWT verification. Supabase Auth is the intended auth provider; the backend trusts the user ID from the frontend. In production, I'd add middleware to verify Supabase JWTs. |
| **"What if the LLM returns garbage?"** | Walk through the full resilience chain: robust JSON parsing → structural validation → repair re-prompting → static fallback content. The user never sees an error. |
| **"How do you handle large PDFs?"** | 50MB limit enforced at upload. Text extraction is synchronous but fast (PyMuPDF is C-based). Chunking handles any document length. For very large PDFs (100+ pages), I'd add a background processing queue. |
| **"Embedding fallbacks seem hacky"** | Explain it's a pragmatic degraded-mode solution. The heuristic embeddings provide enough differentiation for basic retrieval to work. In production, I'd add retry logic with longer timeouts before falling back. |
| **"Why not use OpenAI/Anthropic?"** | Groq's LPU gives sub-second inference, critical for real-time UX. Cost is 10-50x lower than GPT-4. For a self-funded project, this was the practical choice. The modular design means swapping to any OpenAI-compatible API is trivial. |

---

## 7. Key Takeaways

### What This Project Demonstrates

1. **Full-stack AI system design** -- from PDF ingestion to content generation to career matching, all integrated in a production deployment
2. **RAG architecture mastery** -- chunking strategies, embedding selection, vector search, context-grounded generation
3. **Production resilience patterns** -- dual API keys, exponential backoff, graceful degradation, LRU caching with TTL, robust JSON parsing
4. **LLM engineering** -- prompt design for structured output, temperature tuning, map-reduce for long documents, multi-agent validation/repair
5. **Full-stack deployment** -- React frontend on Vercel CDN, FastAPI backend on Railway, Supabase PostgreSQL, environment management

### Technical Depth Signals

- **10,687 lines of Python** -- this isn't a tutorial project; it's a comprehensive system
- **13 specialized modules** -- clean separation of concerns (distiller, career_matcher, supabase_helper, study_agent, validators, repairs, mastery)
- **Multiple AI integration points** -- Groq LLM for generation, Cohere for embeddings, RIASEC for career matching
- **Sophisticated caching** -- LRU with TTL, three-tier resolution, content deduplication via SHA-256
- **Agentic patterns** -- generate → validate → repair cycle, intent detection, mastery tracking

### One-Liner Pitch

> "I built PathWise, a full-stack AI learning platform that turns any PDF into interactive learning content using RAG with Groq and Cohere, and matches users to careers using RIASEC psychometrics enhanced with embedding similarity -- deployed live on Vercel and Railway."
