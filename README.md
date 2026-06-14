# Deep Research Agent API

A production-ready FastAPI backend for the multi-agent deep research system built with **CrewAI**.

## Architecture

```
POST /api/v1/research/          ← submit query
         │
         ▼  (BackgroundTask)
  ┌──────────────────────────────────────────┐
  │              CrewAI Pipeline             │
  │                                          │
  │  1. Research Planner  (plans topics)     │
  │         ↓                                │
  │  2. Internet Researcher (EXA + scrape)   │
  │         ↓                                │
  │  3. Fact Checker (EXA + scrape)          │
  │         ↓                                │
  │  4. Report Writer (markdown report)      │
  └──────────────────────────────────────────┘
         │
         ▼
    job_store (in-memory, thread-safe)
         │
         ▼
GET  /api/v1/jobs/{id}          ← poll status + result
GET  /api/v1/research/{id}/stream ← SSE live progress
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/research/` | Submit research query → returns `job_id` |
| `GET` | `/api/v1/research/{job_id}/stream` | SSE live progress stream |
| `GET` | `/api/v1/jobs/` | List all jobs (paginated) |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status + result |
| `DELETE` | `/api/v1/jobs/{job_id}` | Delete a job |

## SSE Events

The stream endpoint emits these event types:

| Event | When |
|-------|------|
| `stage_started` | A new agent stage begins |
| `stage_completed` | A stage finishes (includes 300-char summary) |
| `completed` | Final report ready (includes full markdown) |
| `error` | Job failed |

Example event:
```
event: stage_started
data: {"event":"stage_started","job_id":"abc-123","stage":"researching","message":"Stage started: researching"}

event: completed
data: {"event":"completed","job_id":"abc-123","stage":"done","result":"# Research Report\n...","message":"Research completed successfully."}
```

## Quickstart

### 1. Local (bare Python)

```bash
# Clone / copy project
cp .env.example .env
# Edit .env → add OPENAI_API_KEY and EXA_API_KEY

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Docker Compose

```bash
cp .env.example .env
# Fill in API keys in .env

docker compose up --build
```

### 3. Submit a job + stream results

```bash
# Submit
curl -X POST http://localhost:8000/api/v1/research/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the top 5 vector databases in 2025?"}'

# → {"job_id":"<uuid>","status":"pending","message":"..."}

# Stream (replace <uuid> with real job_id)
curl -N http://localhost:8000/api/v1/research/<uuid>/stream

# Poll status
curl http://localhost:8000/api/v1/jobs/<uuid>
```

## Configuration

All settings via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **required** | OpenAI key |
| `EXA_API_KEY` | **required** | Exa search key |
| `MODEL` | `gpt-4o-mini` | OpenAI model |
| `EXA_BASE_URL` | `https://api.exa.ai` | Exa endpoint |
| `AGENT_MAX_RPM` | `150` | Max requests/min per agent |
| `AGENT_MAX_ITER` | `15` | Max iterations per agent |
| `MAX_JOBS_IN_MEMORY` | `100` | Job store eviction threshold |
| `ALLOWED_ORIGINS` | `["*"]` | CORS origins list |

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

## Project Structure

```
deep_research_api/
├── app/
│   ├── main.py                   ← FastAPI app + CORS + router registration
│   ├── agents/
│   │   └── research_agents.py    ← 4 CrewAI agents
│   ├── tasks/
│   │   └── research_tasks.py     ← 4 CrewAI tasks
│   ├── crews/
│   │   └── research_crew.py      ← Crew builder + async job runner
│   ├── api/
│   │   ├── research.py           ← POST /research + SSE stream
│   │   ├── jobs.py               ← Job CRUD
│   │   └── health.py             ← /health
│   └── core/
│       ├── config.py             ← Pydantic settings
│       ├── models.py             ← Pydantic schemas
│       └── job_store.py          ← Thread-safe in-memory store
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Notes

- **One worker only**: CrewAI's `kickoff()` is blocking and CPU/IO bound. Run with `--workers 1` and scale horizontally via multiple containers if needed. For true horizontal scaling, replace `job_store` with Redis.
- **No persistence**: Jobs are lost on restart. For production, back `job_store` with Redis or a DB.
- **Interactive docs**: `http://localhost:8000/docs` (Swagger) or `/redoc`
