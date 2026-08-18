# Orbit Backend 🪐

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat&logo=postgresql&logoColor=white)](https://supabase.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.0_Flash-8E75B2?style=flat&logo=google&logoColor=white)](https://aistudio.google.com)
[![Groq](https://img.shields.io/badge/Groq-Whisper_v3-F55036?style=flat)](https://groq.com)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen?style=flat)]()

An asynchronous, sub-second latency backend service in Python (FastAPI) integrated with **Meta's WhatsApp Cloud API**. 

Orbit acts as an intelligent, conversational **Product Manager and Co-Developer** ("an old friend who happens to be a seasoned engineering PM") for developers and freelancers managing multiple client projects.

---

## 🌟 Key Features

- **⚡ Sub-50ms Non-Blocking Webhook Ingestion**: Acknowledges WhatsApp webhooks immediately with `HTTP 200 OK`, offloading audio transcription, document parsing, LLM inference, tool execution, and message dispatch to async background workers.
- **🛡️ Strict Zero-Raw-SQL Tool Calling**: LLM never generates raw SQL. All state changes occur via deterministic, validated Pydantic tool schemas (`create_project`, `create_task`, `update_task_status`, `list_tasks`, etc.).
- **🧠 3-Tier Memory & Knowledge Architecture**:
  - **Tier 1 (Short-Term Buffer)**: Sliding window of recent conversation turns from `conversation_logs`.
  - **Tier 2 (Relational Context)**: Real-time project profiles, active tasks, blockers, and overdue alerts.
  - **Tier 3 (Semantic pgvector RAG)**: Top-$k$ similarity search (`<=>` cosine distance) over `project_knowledge`.
- **🎙️ Voice Note Transcription**: Automatic download and transcription of WhatsApp voice notes (`.ogg`/`opus`) using Groq's high-speed Whisper API (`whisper-large-v3`).
- **📄 Document Ingestion (PRDs & Specs)**: Ingests PDFs and Markdown files uploaded over WhatsApp, chunks them into ~500-token sections, generates embeddings, and saves them directly to the project's vector knowledge base.
- **💬 Human-Like Multi-Message Delivery**: Responses formatted with `|||` delimiters are split into sequential chat bubbles with natural typing status simulation and micro-delays (400–800ms).
- **🕵️ Silent Passive Fact Extraction**: Non-blocking background worker that inspects conversations and auto-extracts technical decisions, client quirks, and gotchas into long-term memory.
- **🔁 Webhook Deduplication Guard**: In-memory thread-safe LRU cache preventing Meta's retry storms from duplicate processing.

---

## 🏗️ Architecture Flow

```mermaid
flowchart TD
    WA["WhatsApp Cloud API"] -->|"Webhook POST"| WH["webhook.py"]
    WH -->|"200 OK (<50ms)"| WA
    WH -->|"BackgroundTasks"| PL["pipeline.py"]
    
    PL -->|"Voice Note"| AU["audio.py (Groq Whisper)"]
    PL -->|"PDF / Markdown"| DP["document_parser.py"]
    DP --> EM["embeddings.py (Gemini 768-dim)"]
    EM --> KB[("pgvector: project_knowledge")]
    
    PL -->|"Text / Transcribed"| AG["agent.py (Gemini Flash)"]
    AG -->|"Assemble 3-Tier Context"| MM["memory.py"]
    MM --> T1["Tier 1: Short-term logs"]
    MM --> T2["Tier 2: Relational projects & tasks"]
    MM --> T3["Tier 3: pgvector Semantic RAG"]
    
    AG -->|"Deterministic Tool Calls"| TD["definitions.py"]
    TD -->|"SQLAlchemy 2.0 ORM"| DB[("PostgreSQL (Supabase)")]
    
    AG -->|"Delimited Response (|||)"| WS["whatsapp.py"]
    WS -->|"Typing Indicator + Delay"| WA
    
    PL -->|"Fire & Forget"| PE["passive_extractor.py"]
    PE --> KB
```

---

## 📁 Project Structure

```
orbit_backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── webhook.py          # Meta verification (GET) & message ingestion (POST)
│   ├── config.py                   # Pydantic Settings & environment validation
│   ├── db/
│   │   ├── migrations/
│   │   │   └── 001_initial.sql     # PostgreSQL + pgvector schema & enum definitions
│   │   ├── models.py               # SQLAlchemy 2.0 async models (User, Project, Task, etc.)
│   │   └── session.py              # Async engine & sessionmaker (Supabase-ready)
│   ├── main.py                     # FastAPI entrypoint, CORS & lifespan management
│   ├── services/
│   │   ├── agent.py                # LLM orchestrator, persona prompt & tool loop
│   │   ├── audio.py                # Media downloader & Groq Whisper transcription
│   │   ├── document_parser.py      # PDF & Markdown extractor with token-aware chunker
│   │   ├── embeddings.py           # Gemini text-embedding-004 vector generator (768-dim)
│   │   ├── memory.py               # 3-Tier context assembly (Buffer + State + RAG)
│   │   ├── passive_extractor.py    # Silent background fact & gotcha extraction
│   │   └── whatsapp.py             # Meta Graph API client (text, typing, multi-bubble)
│   ├── tools/
│   │   └── definitions.py          # Pydantic tool schemas & async DB handlers
│   └── workers/
│       └── pipeline.py             # Non-blocking background worker orchestrator
├── test_flow.py                    # Comprehensive mock-based end-to-end test suite
├── requirements.txt                # Project dependencies
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git ignore patterns
└── README.md
```

---

## ⚙️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Fully Async)
- **Database**: PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) (Supabase-ready) via `SQLAlchemy 2.0 (async)` and `asyncpg`
- **LLM / Tool Calling**: [Google Gemini 2.0 Flash](https://ai.google.dev/) via `google-genai` SDK
- **Embeddings**: `models/text-embedding-004` (768 dimensions)
- **Audio Transcription**: [Groq Whisper API](https://console.groq.com/) (`whisper-large-v3`)
- **Messaging**: Meta WhatsApp Cloud API (Graph API v20.0+)
- **Testing**: `pytest` & `pytest-asyncio` with mocked HTTP/API layers

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com/) project (or local PostgreSQL with `pgvector` enabled)
- [Google AI Studio API Key](https://aistudio.google.com/) (Free tier)
- [Groq API Key](https://console.groq.com/) (Free tier)
- [Meta Developer Account](https://developers.facebook.com/) with WhatsApp Cloud API configured

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/umaryun/orbit_backend.git
cd orbit_backend

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the sample environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# ─── Supabase / PostgreSQL ───
DATABASE_URL=postgresql+asyncpg://postgres.[YOUR-PROJECT]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres

# ─── WhatsApp Cloud API ───
WHATSAPP_TOKEN=EAAxxxxxxx
WHATSAPP_PHONE_NUMBER_ID=123456789012345
WHATSAPP_VERIFY_TOKEN=your_custom_webhook_verify_token

# ─── Google Gemini (LLM & Embeddings) ───
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_LLM_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004

# ─── Groq (Whisper Transcription) ───
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ─── App Settings ───
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### 4. Database Setup

Run the SQL migration in your Supabase SQL Editor (or via `psql`):

```bash
# Execute the SQL script found at:
app/db/migrations/001_initial.sql
```

*(Alternatively, running the FastAPI app in development automatically creates the tables on startup via `init_db()`)*.

### 5. Running the Application

Start the development server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API Base URL: `http://localhost:8000`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

---

## 🧪 Testing

Run the automated test suite with full test coverage over webhooks, message deduping, payload parsing, semantic chunking, and multi-bubble delivery:

```bash
python -m pytest test_flow.py -v
```

All 19 tests run with mocked external APIs without needing live WhatsApp or AI credentials.

---

## 🔗 Meta WhatsApp Cloud API Setup

1. In the **Meta App Dashboard**, go to **WhatsApp > Configuration**.
2. Set **Callback URL** to: `https://your-domain.com/api/v1/webhook` (use [ngrok](https://ngrok.com) or [localtunnel](https://localtunnel.github.io/www/) for local development).
3. Set **Verify Token** to the value of `WHATSAPP_VERIFY_TOKEN` in your `.env`.
4. Subscribe to the **`messages`** webhook field.

---

## 🛠️ Tool Registry Reference

The LLM interacts with the database exclusively through the following deterministic functions:

| Tool | Purpose |
|---|---|
| `create_project` | Creates a new project and initial project profile |
| `update_project_profile` | Updates tech stack, repository URL, target deployment, or client quirks |
| `create_task` | Adds a task with priority and optional due date |
| `update_task_status` | Updates task status (`todo`, `in_progress`, `blocked`, `done`) and blocker reasons |
| `list_tasks` | Retrieves tasks filtered by project, status, or timeframe (`today`, `this_week`, `overdue`) |
| `list_projects` | Lists user projects filtered by status (`active`, `paused`, `completed`, `archived`) |
| `query_project_knowledge` | Performs semantic cosine similarity search across project documentation and extracted facts |

---

## 📄 License

MIT License. Feel free to use and extend Orbit for your own engineering workflows.
