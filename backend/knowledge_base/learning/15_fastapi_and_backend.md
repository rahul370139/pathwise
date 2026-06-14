# 15. FastAPI & Backend Development

> **Rahul Sharma** -- Data Scientist who designed and shipped a production FastAPI backend (PathWise: 10,687 lines of Python across 13 modules, 2,292-line `main.py`, 50+ Pydantic models, 74 async functions, Supabase integration, Groq LLM + Cohere embeddings).

---

## Table of Contents

1. [What Is FastAPI](#1-what-is-fastapi)
2. [The Two Pillars: Starlette & Pydantic](#2-the-two-pillars-starlette--pydantic)
3. [HTTP Methods & CRUD Mapping](#3-http-methods--crud-mapping)
4. [Core Concepts](#4-core-concepts)
5. [Async Support](#5-async-support)
6. [Pydantic Models (Schemas)](#6-pydantic-models-schemas)
7. [CORS](#7-cors)
8. [Authentication](#8-authentication)
9. [Database Integration](#9-database-integration)
10. [Caching Strategies](#10-caching-strategies)
11. [Error Handling](#11-error-handling)
12. [Deployment](#12-deployment)
13. [FastAPI vs Flask vs Django](#13-fastapi-vs-flask-vs-django)
14. [ML Model Serving with FastAPI](#14-ml-model-serving-with-fastapi)
15. [Testing](#15-testing)
16. [Your Project: PathWise](#16-your-project-pathwise)
17. [Common Interview Questions](#17-common-interview-questions)
18. [Key Takeaways](#18-key-takeaways)

---

## 1. What Is FastAPI

FastAPI is a **modern, high-performance Python web framework** for building APIs. It was created by Sebastián Ramírez and first released in late 2018.

### Why It Matters

| Property | Detail |
|---|---|
| **Speed** | On par with Node.js and Go -- one of the fastest Python frameworks (Starlette + Uvicorn) |
| **Type Safety** | Leverages Python 3.6+ type hints for automatic validation and serialization |
| **Auto Docs** | Generates interactive Swagger UI (`/docs`) and ReDoc (`/redoc`) automatically |
| **Standards** | Fully based on OpenAPI (formerly Swagger) and JSON Schema |
| **Async Native** | First-class `async`/`await` support for high-concurrency workloads |

### Minimal Example

```python
from fastapi import FastAPI

app = FastAPI(title="My API")

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

Running it:

```bash
uvicorn main:app --reload
```

This gives you:
- A running server at `http://127.0.0.1:8000`
- Interactive docs at `http://127.0.0.1:8000/docs`
- ReDoc at `http://127.0.0.1:8000/redoc`
- Automatic request validation (try passing `item_id=abc` -- you get a 422)

---

## 2. The Two Pillars: Starlette & Pydantic

FastAPI is not built from scratch. It stands on two battle-tested libraries:

```
┌──────────────────────────────────────────────┐
│                  FastAPI                      │
│   (routing decorators, DI, OpenAPI gen)       │
├───────────────────────┬──────────────────────┤
│       Starlette       │      Pydantic        │
│  (ASGI, HTTP, WS,     │  (data validation,   │
│   middleware, routing) │   serialization,     │
│                       │   type coercion)     │
├───────────────────────┴──────────────────────┤
│              Uvicorn / Hypercorn              │
│            (ASGI server, event loop)          │
└──────────────────────────────────────────────┘
```

### 2a. Starlette -- The Web Layer

Starlette is a lightweight **ASGI framework** that provides:

| Feature | Description |
|---|---|
| **ASGI Protocol** | Asynchronous Server Gateway Interface -- the async successor to WSGI |
| **Routing** | URL path matching, path parameters, method-based dispatch |
| **Middleware** | Pluggable request/response processing pipeline |
| **WebSockets** | Full-duplex communication for real-time apps |
| **Background Tasks** | Fire-and-forget work after the response is sent |
| **Test Client** | Built-in test client using `httpx` |
| **Static Files** | Serve static assets |
| **Sessions/Cookies** | Built-in session and cookie handling |

**ASGI vs WSGI:**

```
WSGI (synchronous):    Request → Worker thread → Response (blocking)
ASGI (asynchronous):   Request → Event loop → await DB/API → Response (non-blocking)
```

ASGI allows a single process to handle thousands of concurrent connections because it doesn't block on I/O.

### 2b. Pydantic -- The Data Layer

Pydantic enforces **type hints at runtime** and converts raw input into validated Python objects.

| Feature | Description |
|---|---|
| **BaseModel** | Declare data shapes as classes with typed fields |
| **Auto-coercion** | `"42"` becomes `42` for an `int` field automatically |
| **Validation** | `Field(ge=0, le=100)`, `regex`, `min_length`, `max_length` |
| **Error Messages** | Returns structured JSON listing every validation failure |
| **Nested Models** | Models can contain other models for complex schemas |
| **Serialization** | `.model_dump()` / `.model_dump_json()` for output |
| **Enum Support** | Enums work as field types for constrained values |

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class CareerMatchRequest(BaseModel):
    owner_id: str
    answers: List[int] = Field(..., min_length=10, max_length=10)
    user_profile: Optional[dict] = None
```

If someone sends `{"owner_id": 123, "answers": [1,2]}`, Pydantic will:
1. Coerce `owner_id` from `123` to `"123"` (int to str)
2. Reject the request because `answers` has only 2 items (needs exactly 10)
3. Return a clear 422 error with details

---

## 3. HTTP Methods & CRUD Mapping

| HTTP Method | CRUD Operation | Idempotent? | Request Body? | Typical Use |
|---|---|---|---|---|
| `GET` | **R**ead | Yes | No | Fetch resource(s) |
| `POST` | **C**reate | No | Yes | Create new resource |
| `PUT` | **U**pdate (full) | Yes | Yes | Replace entire resource |
| `PATCH` | **U**pdate (partial) | No | Yes | Modify specific fields |
| `DELETE` | **D**elete | Yes | Rarely | Remove resource |

### FastAPI Examples

```python
from fastapi import FastAPI

app = FastAPI()
items_db = {}

# CREATE
@app.post("/items/", status_code=201)
async def create_item(item: Item):
    items_db[item.id] = item
    return item

# READ (single)
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return items_db.get(item_id)

# READ (list)
@app.get("/items/")
async def list_items(skip: int = 0, limit: int = 10):
    return list(items_db.values())[skip:skip + limit]

# UPDATE (full replace)
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    items_db[item_id] = item
    return item

# UPDATE (partial)
@app.patch("/items/{item_id}")
async def patch_item(item_id: int, updates: ItemPatch):
    stored = items_db[item_id]
    stored.update(updates.dict(exclude_unset=True))
    return stored

# DELETE
@app.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: int):
    del items_db[item_id]
```

### Idempotency Explained

**Idempotent** = calling it N times produces the same result as calling it once.

- `GET /users/5` -- always returns the same user (idempotent)
- `DELETE /users/5` -- first call deletes, subsequent calls are no-ops (idempotent)
- `POST /users` -- every call creates a new user (NOT idempotent)
- `PUT /users/5` -- replaces user 5 with the payload, always same result (idempotent)

---

## 4. Core Concepts

### 4a. Path Parameters

Extracted from the URL path. Typed and validated automatically.

```python
@app.get("/api/lesson/{lesson_id}/{action}")
async def lesson_action(lesson_id: int, action: str):
    # lesson_id is guaranteed to be an int
    # action is a string
    ...
```

FastAPI generates the OpenAPI schema showing `lesson_id` as integer and `action` as string.

### 4b. Query Parameters

Anything not in the path becomes a query parameter. Use `Query()` for validation.

```python
from fastapi import Query

@app.post("/api/distill")
async def distill_pdf(
    owner_id: str = Query(..., description="Supabase user UUID"),
    explanation_level: ExplanationLevel = Query(
        ExplanationLevel.INTERN,
        description="Explanation complexity level"
    ),
    file: UploadFile = File(...)
):
    ...
```

- `...` (Ellipsis) = required parameter
- Default value = optional parameter
- `Query(min_length=3, max_length=50)` adds validation

### 4c. Request Body

Use Pydantic models to define and validate JSON request bodies.

```python
class ChatMessage(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None
    explanation_level: ExplanationLevel = Field(
        default=ExplanationLevel.INTERN
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatMessage):
    result = await process_chat_message(
        request.user_id,
        request.message,
        request.conversation_id,
        request.explanation_level
    )
    return ChatResponse(**result)
```

### 4d. Response Models

`response_model` tells FastAPI what shape the response takes. It:
- Filters out extra fields (security -- don't leak internal data)
- Validates the response
- Documents the response in OpenAPI

```python
@app.post("/api/career/match", response_model=CareerMatchResponse)
async def career_match(request: CareerMatchRequest):
    ...
    return CareerMatchResponse(results=cards)
```

### 4e. Status Codes

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    ...

@app.delete("/items/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(id: int):
    ...
```

| Code | Meaning | When to Use |
|---|---|---|
| `200` | OK | Successful GET, PUT, PATCH |
| `201` | Created | Successful POST that created a resource |
| `204` | No Content | Successful DELETE |
| `400` | Bad Request | Client sent invalid data |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Authenticated but not authorized |
| `404` | Not Found | Resource doesn't exist |
| `422` | Unprocessable Entity | Validation error (Pydantic default) |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Unhandled server exception |

### 4f. Dependency Injection

FastAPI's DI system lets you declare shared logic (auth, DB sessions, config) that gets injected into endpoints.

```python
from fastapi import Depends

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

async def get_db():
    db = SessionLocal()
    try:
        yield db  # yield makes it a context manager
    finally:
        db.close()

@app.get("/me")
async def read_profile(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Profile).filter_by(user_id=user.id).first()
```

**Why it matters:**
- Avoids code duplication
- Makes testing easy (override dependencies with mocks)
- Handles setup/teardown (DB connections, file handles)
- Dependencies can depend on other dependencies (chain them)

### 4g. Middleware

Middleware processes every request/response. It wraps around your endpoints.

```
Client ──► Middleware A ──► Middleware B ──► Endpoint
         ◄──           ◄──             ◄──
```

```python
from fastapi.middleware.cors import CORSMiddleware
import time

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = str(elapsed)
    return response
```

### 4h. Background Tasks

Execute work after the response is sent. Useful for logging, email, cache warming.

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(f"{message}\n")

@app.post("/send-email/")
async def send_email(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_log, f"Email sent to {email}")
    return {"message": "Email will be sent"}
```

### 4i. File Uploads (UploadFile)

```python
from fastapi import UploadFile, File

@app.post("/api/chat/upload", response_model=ChatResponse)
async def upload_file_for_chat(
    user_id: str = Query(...),
    conversation_id: Optional[str] = Query(None),
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "PDF only")

    content = await file.read()  # async read
    # file.filename, file.size, file.content_type available
    ...
```

`UploadFile` advantages over raw `bytes`:
- Spooled to disk for large files (memory efficient)
- Has `.filename`, `.content_type`, `.size` metadata
- Async `.read()`, `.write()`, `.seek()` methods

---

## 5. Async Support

### Why Async Matters

Traditional synchronous Python:

```
Request 1: ──── DB call (200ms waiting) ──── Response
Request 2:                                   ──── DB call (200ms) ──── Response
Request 3:                                                            ──── ...
```

Asynchronous Python:

```
Request 1: ── await DB call ─────────── Response
Request 2: ── await DB call ───────── Response      (started during R1's wait)
Request 3: ── await API call ────── Response         (started during R1+R2's wait)
```

With `async`/`await`, when one request waits for I/O (DB, HTTP, file), the event loop picks up another request. A single process can handle thousands of concurrent connections.

### Event Loop Basics

```
┌──────────────────────────────────────────┐
│              Event Loop                   │
│                                          │
│  1. Pick next coroutine from queue       │
│  2. Run it until it hits 'await'         │
│  3. Register the I/O callback            │
│  4. Move to next coroutine               │
│  5. When I/O completes, resume coroutine │
│  6. Repeat                               │
└──────────────────────────────────────────┘
```

Python's `asyncio` event loop is single-threaded. It achieves concurrency (not parallelism) by context-switching at `await` points.

### When to Use Async vs Sync

| Scenario | Use | Reason |
|---|---|---|
| HTTP calls to external APIs | `async def` + `httpx.AsyncClient` | I/O bound, high latency |
| Database queries (async driver) | `async def` + `asyncpg`/`databases` | I/O bound |
| File I/O | `async def` + `aiofiles` | I/O bound |
| CPU-heavy computation (ML inference) | `def` (sync) | CPU bound -- let FastAPI run in threadpool |
| Simple dict lookup, math | Either | Negligible difference |
| Supabase / external SDK (sync) | `def` (sync) or wrap with `asyncio.to_thread` | SDK is blocking |

**FastAPI handles both:**
- `async def` endpoints run on the event loop
- `def` (sync) endpoints run in a threadpool automatically

### Concurrent Async Patterns

```python
import asyncio

# Run independent tasks in parallel
summary_task = asyncio.create_task(map_reduce_summary(chunks))
embeds_task = asyncio.create_task(embed_chunks(chunks))
summary, embeds = await asyncio.gather(summary_task, embeds_task)

# Semaphore to limit concurrent calls
_LLM_SEMAPHORE = asyncio.Semaphore(2)

async def call_llm(prompt):
    async with _LLM_SEMAPHORE:  # max 2 concurrent LLM calls
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.post(url, json=payload)
            return res.json()
```

---

## 6. Pydantic Models (Schemas)

### 6a. BaseModel for Request/Response

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class ExplanationLevel(str, Enum):
    FIVE_YEAR_OLD = "5_year_old"
    INTERN = "intern"
    SENIOR = "senior"

class Framework(str, Enum):
    FASTAPI = "fastapi"
    PYTHON = "python"
    REACT = "react"
    MACHINE_LEARNING = "machine_learning"
    GENERIC = "generic"

class ChatMessage(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None
    explanation_level: ExplanationLevel = Field(
        default=ExplanationLevel.INTERN,
        description="Explanation complexity level"
    )
```

### 6b. Field Validation

```python
class QuizAnswer(BaseModel):
    qid: int
    value: int = Field(ge=1, le=5, description="1=Strongly dislike to 5=Love it")

class CareerMatchRequest(BaseModel):
    owner_id: str
    answers: List[int] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Exactly 10 quiz answers (1-5)"
    )
    user_profile: Optional[Dict] = Field(default=None)

class LessonCompletion(BaseModel):
    lesson_id: int
    user_id: str
    completed_at: str
    progress_percentage: float = Field(..., ge=0, le=100)
```

### 6c. Nested Models

```python
class RoadmapLevel(BaseModel):
    title: str
    description: str
    skills: List[str]
    duration: str
    salary_range: str
    responsibilities: List[str]
    learning_objectives: List[str]
    recommended_lessons: List[Dict] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)

class DynamicRoadmap(BaseModel):
    foundational: RoadmapLevel
    intermediate: RoadmapLevel
    advanced: RoadmapLevel

class RoadmapResponse(BaseModel):
    target_role: str
    roadmap: DynamicRoadmap      # nested model
    user_experience: str
```

### 6d. Complex Response Models

```python
class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    file_processed: Optional[bool] = None
    lesson_id: Optional[int] = None
    pdf_name: Optional[str] = None
    summary: Optional[str] = None
    type: Optional[str] = Field(
        default=None,
        description="Content type: chat|lesson|quiz|flashcards|workflow|summary"
    )
    quiz_data: Optional[Dict[str, Any]] = None
    flashcard_data: Optional[Dict[str, Any]] = None
    lesson_data: Optional[Dict[str, Any]] = None
```

### 6e. Pydantic V2 (Current)

Pydantic V2 (powered by Rust via `pydantic-core`) is 5-50x faster than V1.

```python
# V1 syntax (deprecated)
class Item(BaseModel):
    class Config:
        schema_extra = {"example": {"name": "Foo"}}

# V2 syntax (current)
class Item(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"name": "Foo"}})
    name: str

# Serialization
item.model_dump()          # replaces .dict()
item.model_dump_json()     # replaces .json()
Item.model_validate(data)  # replaces .parse_obj()
```

---

## 7. CORS

### What Is CORS?

**Cross-Origin Resource Sharing** is a browser security mechanism. When a frontend at `https://myapp.vercel.app` tries to call an API at `https://api.railway.app`, the browser blocks it unless the API explicitly allows it.

```
Browser at https://myapp.vercel.app
    │
    ├─ Preflight: OPTIONS /api/chat
    │   Headers: Origin: https://myapp.vercel.app
    │
    ▼
API at https://api.railway.app
    │
    ├─ Response: Access-Control-Allow-Origin: https://myapp.vercel.app
    │            Access-Control-Allow-Methods: POST, GET, OPTIONS
    │            Access-Control-Allow-Headers: Content-Type, Authorization
    │
    ▼
Browser: "OK, the API allows this origin" → sends actual request
```

### Why It's Needed

- **Security**: Prevents malicious sites from making API calls on behalf of users
- **Same-Origin Policy**: Browsers block cross-origin requests by default
- **APIs and SPAs**: Modern apps almost always have separate frontend/backend origins

### How to Configure in FastAPI

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://v0-frontend-opal-nine.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,   # allow cookies/auth headers
    allow_methods=["*"],      # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],      # Content-Type, Authorization, etc.
)
```

**Production Best Practice:** Never use `allow_origins=["*"]` with `allow_credentials=True` in production. Whitelist specific origins.

---

## 8. Authentication

### 8a. OAuth2 with JWT Tokens

The most common pattern for API authentication:

```
Client                          Server
  │                                │
  ├─ POST /auth/login ────────────►│
  │  {username, password}          │
  │                                ├─ Verify credentials
  │◄─────── {access_token} ───────┤  Generate JWT
  │                                │
  ├─ GET /api/data ───────────────►│
  │  Header: Authorization:        │
  │  Bearer <token>                ├─ Decode JWT
  │                                ├─ Extract user_id
  │◄─────── {data} ──────────────┤  Return data
```

```python
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from datetime import datetime, timedelta

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    token = create_access_token(data={"sub": user.id})
    return {"access_token": token, "token_type": "bearer"}
```

### 8b. API Key Authentication

Simpler approach for service-to-service communication:

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.get("/secure-data")
async def get_secure_data(api_key: str = Depends(verify_api_key)):
    return {"data": "sensitive information"}
```

### 8c. Comparison

| Method | Use Case | Pros | Cons |
|---|---|---|---|
| **JWT Bearer** | User auth in SPAs/mobile | Stateless, scalable | Token can't be revoked easily |
| **API Key** | Service-to-service | Simple, easy to rotate | No user identity, easy to leak |
| **OAuth2 + Scopes** | Third-party integrations | Fine-grained permissions | Complex setup |
| **Session/Cookie** | Traditional web apps | Server-controlled revocation | Requires session storage |

---

## 9. Database Integration

### 9a. SQLAlchemy (Sync)

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = "postgresql://user:pass@localhost/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
```

### 9b. SQLAlchemy (Async)

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/mydb"
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession)

async def get_db():
    async with async_session() as session:
        yield session

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

### 9c. Supabase Integration (as used in PathWise)

Supabase provides a PostgreSQL database with a REST API and a Python client:

```python
import os
import supabase

SUPA = None
try:
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    if supabase_url and supabase_key:
        SUPA = supabase.create_client(supabase_url, supabase_key)
except Exception as e:
    logger.warning(f"Supabase init failed: {e}. Running in test mode.")

def insert_lesson(owner_id, title, summary, framework, explanation_level):
    if not SUPA:
        return generate_local_id()  # graceful fallback
    
    owner_uuid = _normalize_uuid(owner_id)
    row = {
        "owner": owner_uuid,
        "title": title[:255],
        "summary": summary[:5000],
        "framework": framework.value,
        "explanation_level": explanation_level.value,
    }
    res = SUPA.table("lessons").insert(row).execute()
    return res.data[0]["id"]
```

**Key pattern:** Graceful fallback to local mode when Supabase is unavailable.

### 9d. Connection Pooling

```python
# SQLAlchemy connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_size=5,           # maintain 5 connections
    max_overflow=10,       # allow up to 10 extra under load
    pool_timeout=30,       # wait 30s for a connection
    pool_recycle=1800,     # recycle connections after 30min
    pool_pre_ping=True,    # test connection health before use
)
```

---

## 10. Caching Strategies

### 10a. In-Memory LRU with TTL (as used in PathWise)

```python
from collections import OrderedDict
from datetime import datetime

lesson_store: OrderedDict[str, Dict] = OrderedDict()
LESSON_CACHE_TTL_SECONDS = 60 * 60 * 2  # 2 hours
LESSON_CACHE_CAPACITY = 50

def _is_expired(record: Dict) -> bool:
    ts = record.get("cached_at")
    if not ts:
        return False
    cached = datetime.fromisoformat(ts)
    age = (datetime.utcnow() - cached).total_seconds()
    return age > LESSON_CACHE_TTL_SECONDS

def set_lesson_cache(lesson_id: str, data: Dict):
    data["cached_at"] = datetime.utcnow().isoformat()
    if lesson_id in lesson_store:
        lesson_store.pop(lesson_id)  # remove old position
    lesson_store[lesson_id] = data   # insert at end (most recent)
    # Evict oldest if over capacity
    while len(lesson_store) > LESSON_CACHE_CAPACITY:
        lesson_store.popitem(last=False)

def get_lesson_cache(lesson_id: str) -> Optional[Dict]:
    rec = lesson_store.get(lesson_id)
    if not rec:
        return None
    if _is_expired(rec):
        lesson_store.pop(lesson_id)
        return None
    # Touch: move to end (LRU)
    lesson_store.pop(lesson_id)
    lesson_store[lesson_id] = rec
    return rec
```

**Why this design:**
- `OrderedDict` preserves insertion order for LRU eviction
- TTL prevents serving stale data (2-hour expiry)
- Capacity limit (50) prevents unbounded memory growth
- No external dependency (no Redis needed for small-scale)

### 10b. Redis Caching

For multi-process or multi-server deployments:

```python
import redis
import json

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def cache_get(key: str) -> Optional[dict]:
    data = redis_client.get(key)
    return json.loads(data) if data else None

def cache_set(key: str, value: dict, ttl: int = 7200):
    redis_client.setex(key, ttl, json.dumps(value))

@app.get("/api/career/roadmap/{title}")
async def get_roadmap(title: str):
    cached = cache_get(f"roadmap:{title}")
    if cached:
        return cached
    roadmap = await generate_roadmap(title)
    cache_set(f"roadmap:{title}", roadmap)
    return roadmap
```

### 10c. When to Cache

| Cache When | Don't Cache When |
|---|---|
| Response is expensive to compute (LLM calls, DB aggregations) | Data changes frequently (real-time feeds) |
| Same request is made repeatedly | Responses are user-specific and unique |
| Data changes infrequently | Security-sensitive data |
| External API calls (rate limits, latency) | Small, fast computations |

### 10d. Python `functools.lru_cache`

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_config(key: str) -> str:
    return load_from_file(key)  # expensive I/O
```

Note: `lru_cache` doesn't support TTL natively. Use `cachetools.TTLCache` for TTL + LRU:

```python
from cachetools import TTLCache
cache = TTLCache(maxsize=100, ttl=3600)
```

---

## 11. Error Handling

### 11a. HTTPException

FastAPI's built-in way to return error responses:

```python
from fastapi import HTTPException

@app.post("/api/distill")
async def distill_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB")
    
    try:
        text = pdf_to_text(file)
        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=422,
                detail="Failed to extract text -- maybe scanned or corrupted?"
            )
    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process: {str(e)}")
```

### 11b. Custom Exception Handlers

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class RateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded", "retry_after": exc.retry_after},
        headers={"Retry-After": str(exc.retry_after)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### 11c. Graceful Degradation

Provide fallback responses when services are down:

```python
@app.get("/api/lesson/{lesson_id}/{action}")
async def lesson_action(lesson_id: int, action: str):
    try:
        # 1) Try in-memory cache
        cached = get_lesson_cache(str(lesson_id))
        if cached and cached.get("bullets"):
            return {"content": cached["bullets"]}
        
        # 2) Try Supabase
        summary = get_lesson_summary(lesson_id)
        if summary:
            return {"content": parse_bullets(summary)}
        
        # 3) Generate on-demand (fallback)
        content = await _generate_summary_on_demand(lesson_id)
        return {"content": content}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lesson action failed: {e}")
        raise HTTPException(500, f"Failed to get {action}")
```

### 11d. Retry Logic with Exponential Backoff

```python
import re

async def _call_groq_with_key(messages, api_key, key_type):
    max_retries = 2
    backoff = 1.5
    attempt = 0
    
    async with _LLM_SEMAPHORE:
        while True:
            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    res = await client.post(url, headers=headers, json=payload)
                    res.raise_for_status()
                    return res.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries:
                    # Parse server-suggested wait time
                    msg = e.response.json().get("error", {}).get("message", "")
                    m = re.search(r"try again in ([0-9.]+)s", msg)
                    wait_s = min(float(m.group(1)) + 0.5, 15.0) if m else backoff
                    
                    await asyncio.sleep(wait_s)
                    attempt += 1
                    backoff *= 2  # exponential backoff
                    continue
                raise
```

**Pattern: Cache → Database → Generate → Fallback.** This 3-tier strategy ensures the API always returns something useful.

---

## 12. Deployment

### 12a. Uvicorn (ASGI Server)

```bash
# Development
uvicorn main:app --reload --port 8000

# Production (single worker)
uvicorn main:app --host 0.0.0.0 --port 8000

# Production (multiple workers via Gunicorn)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (dev only) |
| `--host 0.0.0.0` | Listen on all interfaces |
| `-w 4` | 4 worker processes (Gunicorn) |
| `-k uvicorn.workers.UvicornWorker` | Use Uvicorn's async worker class |

### 12b. Procfile (Railway/Heroku)

```
web: python -m uvicorn main:app --host 0.0.0.0 --port $PORT
```

### 12c. Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - redis
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### 12d. Platform Comparison

| Platform | Strengths | Use Case |
|---|---|---|
| **Railway** | Git push deploy, auto-scaling, PostgreSQL add-on | Full-stack apps, PathWise backend |
| **Vercel** | Edge functions, serverless, great for Next.js | Serverless APIs, frontend |
| **Render** | Free tier, auto-deploy from GitHub | Small projects, staging |
| **AWS ECS/Fargate** | Full control, auto-scaling, production grade | Enterprise, high-traffic |
| **Fly.io** | Edge deployment, low latency | Global APIs |

### 12e. Health Checks

Every production API needs health check endpoints:

```python
@app.get("/")
async def root():
    return {"message": "PathWise API is running", "status": "healthy", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PathWise API", "timestamp": "2024-08-01"}

# Advanced: check dependencies
@app.get("/health/detailed")
async def detailed_health():
    checks = {}
    # Check database
    try:
        SUPA.table("lessons").select("id", count="exact").limit(1).execute()
        checks["database"] = "healthy"
    except:
        checks["database"] = "unhealthy"
    # Check LLM
    checks["llm"] = "healthy" if os.getenv("GROQ_API_KEY") else "missing_key"
    
    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
```

---

## 13. FastAPI vs Flask vs Django

| Feature | FastAPI | Flask | Django |
|---|---|---|---|
| **Type** | Async API framework | Micro framework | Full-stack framework |
| **Speed** | Very fast (Starlette + Uvicorn) | Moderate (WSGI) | Moderate (WSGI) |
| **Async** | Native async/await | Limited (Flask 2.0+) | Limited (Django 4.0+) |
| **Validation** | Automatic (Pydantic) | Manual / Flask-Marshmallow | Django Forms / DRF Serializers |
| **Auto Docs** | Built-in (Swagger + ReDoc) | Flask-Swagger (manual) | DRF has built-in docs |
| **ORM** | BYO (SQLAlchemy, Tortoise) | BYO (SQLAlchemy) | Built-in (Django ORM) |
| **Admin Panel** | None built-in | Flask-Admin | Built-in Django Admin |
| **Auth** | BYO (fastapi-users, JWT) | Flask-Login | Built-in (users, groups, perms) |
| **Learning Curve** | Medium | Easy | Steep |
| **Best For** | APIs, microservices, ML serving | Small APIs, prototypes | Full web apps, CMS, e-commerce |
| **Community** | Growing fast | Large, mature | Very large, mature |
| **Type Hints** | Required, enforced | Optional | Optional |
| **Deployment** | Uvicorn/Gunicorn | Gunicorn/uWSGI | Gunicorn/uWSGI |

### When to Choose Each

- **FastAPI**: Building APIs, microservices, ML model serving, real-time apps, you want type safety and auto-docs
- **Flask**: Quick prototypes, simple APIs, you want maximum flexibility and minimal structure
- **Django**: Full web applications with admin panel, user management, content management, you want batteries included

---

## 14. ML Model Serving with FastAPI

### 14a. Loading Models at Startup

```python
from fastapi import FastAPI
import joblib
import torch

app = FastAPI()
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model.joblib")  # scikit-learn
    # or: model = torch.load("model.pth")
    # or: model = SentenceTransformer("all-MiniLM-L6-v2")
```

### 14b. Inference Endpoint

```python
from pydantic import BaseModel
from typing import List
import numpy as np

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    confidence = float(model.predict_proba(X).max())
    return PredictionResponse(prediction=prediction, confidence=confidence)
```

### 14c. Batching for Throughput

```python
import asyncio
from collections import deque

batch_queue = deque()
BATCH_SIZE = 32
BATCH_TIMEOUT = 0.1  # seconds

async def process_batch():
    while True:
        if len(batch_queue) >= BATCH_SIZE or (batch_queue and time_since_first > BATCH_TIMEOUT):
            batch = [batch_queue.popleft() for _ in range(min(len(batch_queue), BATCH_SIZE))]
            inputs = np.array([item["features"] for item in batch])
            predictions = model.predict(inputs)
            for item, pred in zip(batch, predictions):
                item["future"].set_result(pred)
        await asyncio.sleep(0.01)

@app.post("/predict-batched")
async def predict_batched(request: PredictionRequest):
    future = asyncio.get_event_loop().create_future()
    batch_queue.append({"features": request.features, "future": future})
    result = await future
    return {"prediction": float(result)}
```

### 14d. LLM API Integration (as in PathWise)

```python
async def call_groq(messages: List[Dict]) -> str:
    """Call Groq LLM with fallback API keys"""
    primary_key = os.getenv('GROQ_API_KEY')
    fallback_key = os.getenv('GROQ_API_KEY_2')
    
    # Try primary, then fallback
    for key, label in [(primary_key, "primary"), (fallback_key, "fallback")]:
        if key:
            result = await _call_with_retry(messages, key, label)
            if result:
                return result
    
    return ""  # all keys failed

async def cohere_embed(batch: List[str]) -> List[List[float]]:
    """Generate embeddings using Cohere API"""
    payload = {
        "model": "embed-english-light-v3.0",
        "texts": batch,
        "input_type": "search_document"
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload)
        return res.json()["embeddings"]
```

---

## 15. Testing

### 15a. pytest + TestClient

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_create_item():
    response = client.post("/items/", json={"name": "Test", "price": 9.99})
    assert response.status_code == 201
    assert response.json()["name"] == "Test"

def test_validation_error():
    response = client.post("/api/career/match", json={
        "owner_id": "user1",
        "answers": [1, 2]  # needs 10 answers
    })
    assert response.status_code == 422

def test_file_upload():
    with open("test.pdf", "rb") as f:
        response = client.post(
            "/api/chat/upload?user_id=test_user",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
```

### 15b. Async Tests

```python
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.anyio
async def test_career_match():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/career/match", json={
            "owner_id": "test_user",
            "answers": [3, 4, 2, 5, 1, 4, 3, 2, 5, 1]
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
```

### 15c. Dependency Overrides (Mocking)

```python
def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# In tests:
def test_with_mock_db():
    response = client.get("/users/1")
    assert response.status_code == 200

# Cleanup
app.dependency_overrides.clear()
```

### 15d. Testing Patterns

| Test Type | What to Test | Tool |
|---|---|---|
| **Unit** | Individual functions, Pydantic models | `pytest` |
| **Integration** | Full endpoint request/response cycle | `TestClient` |
| **Async Integration** | Async endpoints with real async behavior | `httpx.AsyncClient` |
| **Load Testing** | Performance under concurrent load | `locust`, `k6` |
| **Contract Testing** | API schema compliance | Auto-generated from OpenAPI |

---

## 16. Your Project: PathWise

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PathWise Architecture                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌──────────────────────────────────────────────┐  │
│  │  Next.js     │     │           FastAPI Backend (main.py)          │  │
│  │  Frontend    │────►│                                              │  │
│  │  (Vercel)    │     │  ┌──────────┐  ┌───────────┐  ┌──────────┐ │  │
│  └─────────────┘     │  │  Chat    │  │  Career   │  │  Agent   │ │  │
│                       │  │  Module  │  │  Matcher  │  │  System  │ │  │
│                       │  └────┬─────┘  └─────┬─────┘  └────┬─────┘ │  │
│                       │       │              │              │        │  │
│                       │       ▼              ▼              ▼        │  │
│                       │  ┌────────────────────────────────────────┐ │  │
│                       │  │         Distiller (distiller.py)       │ │  │
│                       │  │  PDF → Chunks → Embed → Summarize     │ │  │
│                       │  └─────────────┬──────────────────────────┘ │  │
│                       │                │                             │  │
│                       └────────────────┼─────────────────────────────┘  │
│                                        │                                │
│                    ┌───────────────────┼───────────────────┐            │
│                    ▼                   ▼                   ▼            │
│              ┌──────────┐      ┌──────────────┐    ┌──────────────┐   │
│              │  Groq    │      │   Cohere     │    │   Supabase   │   │
│              │  LLM     │      │   Embeddings │    │   (Postgres) │   │
│              │ llama-3.3│      │   v3.0       │    │              │   │
│              └──────────┘      └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown (10,687 total lines)

| Module | Lines | Responsibility |
|---|---|---|
| `main.py` | 2,292 | FastAPI app, all endpoints, CORS, middleware |
| `distiller.py` | 1,791 | PDF processing, LLM calls, embeddings, caching, map-reduce |
| `unified_career_system.py` | 1,346 | AI-powered career roadmaps and planning |
| `career_matcher.py` | 875 | Quiz-based career matching with cosine similarity |
| `repairs.py` | 644 | Auto-repair malformed LLM JSON outputs |
| `study_agent.py` | 584 | Agentic AI: SummarizerAgent, DiagnosticAgent, Router |
| `dashboard.py` | 584 | Dashboard recommendations and coaching |
| `schemas.py` | 497 | 50+ Pydantic models for request/response validation |
| `validators.py` | 494 | QA validation for flashcards and quizzes |
| `supabase_helper.py` | 428 | Database operations with graceful fallback |
| `process_dataset.py` | 392 | Data preprocessing for career matching |
| `mastery.py` | 381 | User mastery score tracking |
| `learn_tools.py` | 379 | PDF ingestion, flashcard/quiz generation tools |

### Key Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat/upload` | POST | Upload PDF, process with LLM, start conversation |
| `/api/chat` | POST | Chat with AI about uploaded content |
| `/api/distill` | POST | PDF → chunks → embeddings → summary → flashcards → quiz |
| `/api/career/match` | POST | 10-question quiz → AI career matching |
| `/api/career/roadmap/unified` | POST | Comprehensive career roadmap generation |
| `/api/lesson/{id}/{action}` | GET | Get summary/quiz/flashcards/workflow for a lesson |
| `/api/agent/summary` | POST | Agentic AI structured summarization |
| `/api/agent/diagnostic` | POST | Diagnostic quiz with mastery tracking |
| `/health` | GET | Health check for deployment monitoring |

### Map-Reduce Pattern for PDF Summarization

```
                     PDF Document
                          │
                          ▼
                    pdf_to_text()
                          │
                          ▼
                   chunk_text(text)
                    (400 words each, 50 overlap)
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         chunk_1      chunk_2      chunk_3     ← MAP phase
        "Summarize    "Summarize   "Summarize
         into 3       into 3       into 3
         bullets"     bullets"     bullets"
              │           │           │
              ▼           ▼           ▼
         bullets_1    bullets_2   bullets_3
              │           │           │
              └───────────┼───────────┘
                          ▼
                    REDUCE phase                ← REDUCE phase
              "Merge into max 10
               key bullets"
                          │
                          ▼
                   Final Summary
```

```python
async def map_reduce_summary(chunks, explanation_level):
    # MAP: summarize each chunk in parallel
    mapped = await asyncio.gather(*[summarize_chunk(ch) for ch in chunks])
    
    # REDUCE: merge all bullet groups into max 10 key bullets
    reduce_prompt = f"Merge these bullet groups into max 10 key bullets:\n" + "\n".join(mapped)
    final = await call_groq([{"role": "user", "content": reduce_prompt}])
    return final
```

### TTL Caching with 2-Hour Expiry

```
Request Flow:
                          ┌──────────────┐
                          │   Request     │
                          └──────┬───────┘
                                 │
                          ┌──────▼───────┐
                          │ In-Memory    │ ──── HIT ──► Return cached
                          │ LRU Cache    │
                          │ (TTL: 2hr,   │
                          │  Cap: 50)    │
                          └──────┬───────┘
                              MISS │
                          ┌──────▼───────┐
                          │  Supabase    │ ──── HIT ──► Return + cache
                          │  (Postgres)  │
                          └──────┬───────┘
                              MISS │
                          ┌──────▼───────┐
                          │  Generate    │ ──── Generate + cache + return
                          │  On-Demand   │
                          │  (Groq LLM)  │
                          └──────────────┘
```

### Rate Limiting with Exponential Backoff

```python
# Semaphore limits concurrent LLM calls to 2
_LLM_SEMAPHORE = asyncio.Semaphore(2)

# Dual API key fallback
async def call_groq(messages):
    for key in [primary_key, fallback_key]:
        result = await _call_groq_with_key(messages, key)
        if result:
            return result
    return ""

# Retry with exponential backoff on 429 (rate limit)
async def _call_groq_with_key(messages, api_key):
    max_retries = 2
    backoff = 1.5
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                # Parse server-suggested retry time
                wait = parse_retry_after(e) or backoff
                await asyncio.sleep(wait)
                backoff *= 2  # double the wait each retry
                continue
            raise
```

### Content Deduplication

```python
import hashlib

content_hash_to_lesson_id: Dict[str, int] = {}

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

# Before processing a new PDF, check if we've already processed identical content
hash_val = _hash_text(extracted_text)
if hash_val in content_hash_to_lesson_id:
    return existing_lesson  # skip duplicate processing
```

---

## 17. Common Interview Questions

### Q1: "Why FastAPI over Flask?"

> **Answer:** "I chose FastAPI for PathWise because it offers three critical advantages for our use case. First, **native async support** -- our backend makes concurrent calls to Groq LLM and Cohere embeddings, and async lets us process these in parallel instead of sequentially. Second, **automatic Pydantic validation** -- with 50+ request/response schemas, manual validation in Flask would've been error-prone and verbose. FastAPI validates everything from path parameters to nested JSON bodies automatically. Third, **auto-generated docs** -- with 40+ endpoints, having interactive Swagger docs at `/docs` saved our frontend team hours of back-and-forth. The performance is comparable to Node.js, which matters when you're making multiple LLM API calls per request."

### Q2: "How does Pydantic validation work under the hood?"

> **Answer:** "Pydantic uses Python's type hints as a runtime validation DSL. When FastAPI receives a request, it passes the raw JSON to the Pydantic model's `model_validate()`. Pydantic V2 uses a Rust-based core (`pydantic-core`) that's 5-50x faster than V1. For each field, it: (1) checks if the value exists (required vs optional), (2) attempts type coercion (string `"42"` to int `42`), (3) runs validators like `Field(ge=0, le=100)`, and (4) validates nested models recursively. If any validation fails, it collects all errors and returns them as a structured 422 response. In PathWise, for example, our `CareerMatchRequest` requires exactly 10 answers between 0 and 5 -- Pydantic enforces both constraints in a single pass."

### Q3: "Explain async/await in Python."

> **Answer:** "Async/await enables cooperative multitasking within a single thread. When you `await` an I/O operation, the function yields control back to the event loop, which can then run other coroutines. In PathWise, this was critical -- our `distill_pdf` endpoint makes parallel calls to the Groq LLM (map-reduce summarization) and Cohere (embedding generation) using `asyncio.gather()`. Without async, these would be sequential, roughly doubling the response time. The key mental model: `async def` creates a coroutine, `await` is a suspension point, and the event loop is the scheduler. We also use `asyncio.Semaphore(2)` to limit concurrent LLM calls and avoid overwhelming the API."

### Q4: "How would you serve an ML model with FastAPI?"

> **Answer:** "There's a three-step pattern. First, load the model at startup using `@app.on_event('startup')` so you don't reload it per request. Second, create a Pydantic schema for input validation and a response schema for the prediction output. Third, build the inference endpoint that takes validated input, runs the model, and returns structured output. For production, you'd add: request batching with a queue and timeout, health checks that verify the model is loaded, caching for repeated predictions, and monitoring for inference latency. In PathWise, we used this pattern for our career matcher -- we load the career dataset and precompute embeddings at startup, then the matching endpoint uses cosine similarity for real-time career recommendations."

### Q5: "What is CORS and why is it needed?"

> **Answer:** "CORS is a browser security mechanism that restricts cross-origin HTTP requests. When our React frontend at `v0-frontend-opal-nine.vercel.app` calls our API at `pathwise.railway.app`, the browser first sends an OPTIONS preflight request. Our API must respond with the appropriate `Access-Control-Allow-Origin` header listing the frontend's origin, or the browser blocks the request. In PathWise, I configured CORS middleware to whitelist our Vercel frontend, localhost ports for development, and set `allow_credentials=True` for authenticated requests. In production, you should never use `allow_origins=['*']` with credentials -- always whitelist specific origins."

### Q6: "How do you handle errors in production APIs?"

> **Answer:** "I use a layered approach. At the endpoint level, I catch specific exceptions and raise `HTTPException` with appropriate status codes and clear messages. For external service failures, I implement graceful degradation -- in PathWise, if the Groq API is down, we fall back to cached responses, then Supabase, then generate intelligent fallback content. For rate limits, I use exponential backoff with server-suggested retry times. For unhandled exceptions, a global exception handler catches everything, logs the full traceback, and returns a generic 500. We also use dual API keys -- if the primary Groq key fails, we automatically try the fallback key. The pattern is: cache first, database second, generate third, fallback last."

### Q7: "How do you handle concurrent requests and rate limiting?"

> **Answer:** "In PathWise, I use three concurrency control mechanisms. First, `asyncio.Semaphore(2)` limits concurrent LLM API calls to avoid overwhelming the Groq rate limit. Second, dual API keys with automatic failover -- if the primary key hits a rate limit, we seamlessly switch to the backup. Third, exponential backoff with server-suggested retry times -- we parse the `try again in Xs` from 429 responses and wait accordingly, doubling the backoff on each retry. For the map-reduce summarization, I use `asyncio.gather()` to process chunks in parallel while the semaphore ensures we don't exceed 2 concurrent LLM calls at any time."

### Q8: "Walk me through a request lifecycle in your PathWise API."

> **Answer:** "Let me trace a PDF upload through `/api/chat/upload`. First, the request hits CORS middleware -- the browser's preflight OPTIONS is handled automatically. Then FastAPI validates the query parameters (`user_id`, `explanation_level`) via Pydantic and the file via `UploadFile`. The endpoint writes the PDF to a temp file, then the distiller extracts text with PyMuPDF, chunks it into 400-word segments with 50-word overlap. Next, two async tasks run in parallel via `asyncio.gather`: map-reduce summarization (each chunk summarized by Groq LLM, then merged into 10 bullets) and Cohere embedding generation. The results are cached in our LRU cache with 2-hour TTL, saved to Supabase, and a conversation is initialized. The response goes through `response_model=ChatResponse` validation, which strips any extra fields, and returns to the client. If anything fails, we have fallback content and the temp file is cleaned up in a `finally` block."

### Q9: "How does dependency injection work in FastAPI?"

> **Answer:** "FastAPI's DI system uses the `Depends()` function to declare dependencies that are resolved at request time. You can think of it as automatic argument injection. When an endpoint declares `db: Session = Depends(get_db)`, FastAPI calls `get_db()` before the endpoint, passes the result as `db`, and if `get_db` is a generator (uses `yield`), it runs cleanup after the response. Dependencies can be chained -- a `get_current_user` dependency can itself depend on `get_db`. For testing, you can override any dependency with `app.dependency_overrides[get_db] = mock_db`. This makes it trivial to swap a real database for an in-memory one in tests."

### Q10: "How would you scale this API for high traffic?"

> **Answer:** "I'd approach it in tiers. First, horizontal scaling: run multiple Uvicorn workers behind Gunicorn (`-w 4 -k uvicorn.workers.UvicornWorker`). Second, move the in-memory LRU cache to Redis for shared caching across workers. Third, add a CDN for static content and consider edge caching for read-heavy endpoints. Fourth, for the LLM calls which are the bottleneck, implement request queuing and batch processing. Fifth, set up auto-scaling on Railway or Kubernetes based on CPU/memory metrics. The architecture already supports this -- the stateless endpoint design means any instance can handle any request, and the Supabase database is centralized."

---

## 18. Key Takeaways

1. **FastAPI = Starlette (web) + Pydantic (validation)** -- understand both pillars
2. **Async is not magic** -- it helps with I/O-bound work (DB, API calls), not CPU-bound work (model inference runs in threadpool)
3. **Pydantic models are your contract** -- they validate input, document your API, and filter output
4. **CORS is a browser feature** -- your API permits cross-origin requests, it doesn't enforce them
5. **Cache aggressively for LLM calls** -- they're slow and expensive; use LRU + TTL
6. **Graceful degradation > hard failures** -- always have fallback content
7. **Rate limit yourself before others limit you** -- use semaphores and backoff
8. **Health checks are non-negotiable** -- every production API needs `/health`
9. **Dependency injection makes testing trivial** -- override dependencies to mock services
10. **Map-reduce works for summarization** -- map chunks in parallel, reduce into final output

### Your Differentiator as a Data Scientist

> "I didn't just build models -- I shipped a full production API. PathWise has 10,687 lines of FastAPI code with 50+ Pydantic schemas, async LLM integration, map-reduce summarization, LRU caching with TTL, rate limiting with exponential backoff, dual API key failover, Supabase integration with graceful fallback, and agentic AI patterns. I understand the full stack from model to deployment."

---

*Document prepared for Rahul Sharma -- FastAPI & Backend Development Interview Preparation*
