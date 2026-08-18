# CLAUDE.md — AI PM & Co-Developer Backend Guidelines

## Project Overview
An ultra-fast, context-aware WhatsApp AI Product Manager and Co-Developer backend. It ingests voice memos, text, and PRDs/FRDs to maintain deep project state, manage tasks, enforce deadlines, and provide technical guidance for multi-project developers.

---

## Technical Stack & Architecture
- **Framework:** Python 3.11+, FastAPI (Async throughout)
- **Database:** PostgreSQL with `pgvector` via Supabase / Async SQLAlchemy 2.0
- **AI / Inference:** Tool-calling LLM (Gemini 1.5 Flash / Groq / OpenAI) + Whisper API
- **Embeddings:** 1536-dimensional vectors (`text-embedding-3-small` / `text-embedding-004`)
- **Integration:** Meta WhatsApp Cloud API (Graph API v20.0+)

---

## Architectural Laws & Constraints

### 1. Webhook Non-Blocking Invariant
- **Rule:** The webhook route `POST /api/v1/webhook` must ALWAYS respond with `HTTP 200 OK` within 50ms.
- **Implementation:** Never `await` transcriptions, LLM inference, embeddings, or DB writes inside the route controller. Dispatch all processing to `FastAPI.BackgroundTasks` or the async worker queue immediately.

### 2. No Raw SQL from LLM
- **Rule:** The LLM is strictly prohibited from generating raw SQL.
- **Implementation:** All database queries and state mutations MUST occur through structured Pydantic tool definitions (`app/tools/definitions.py`).

### 3. Multi-Message Human Pacing
- **Rule:** The assistant must never send massive monolithic messages on WhatsApp.
- **Implementation:** 
  - System prompt enforces thought separation using `|||`.
  - The WhatsApp service must split by `|||`, trigger the `typing...` indicator via Graph API, apply a short natural delay (400ms–800ms), and dispatch messages as sequential bubbles.

### 4. 3-Tier Context Assembly (Memory Strategy)
Every prompt to the LLM must assemble:
1. **Tier 1 (Short-Term Buffer):** Last 10–15 messages from `conversation_logs`.
2. **Tier 2 (Relational Context):** Active project profile, active tasks, and upcoming deadlines.
3. **Tier 3 (Vector Semantic Context):** Top-3 `project_knowledge` chunks retrieved via cosine similarity (`embedding <=> query_embedding`) for the active project.

### 5. Silent Background Extraction
- Any conversation turn where new architectural decisions, client quirks, or technical constraints are discussed must trigger an asynchronous, non-blocking extraction task (`app/services/passive_extractor.py`) to populate `project_knowledge`.

---

## Development & Testing Commands

### Setup & Migrations
```bash
# Virtual environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run migrations / create tables
python -m app.db.init_db