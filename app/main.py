"""
Deep Research Agent - FastAPI Backend
Built on CrewAI with async job management and SSE streaming
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.job_store import job_store
from app.api.research import router as research_router
from app.api.jobs import router as jobs_router
from app.api.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    print("🚀 Deep Research API starting up...")
    yield
    print("🛑 Deep Research API shutting down...")
    job_store.clear()


app = FastAPI(
    title="Deep Research Agent API",
    description=(
        "A multi-agent deep research backend powered by CrewAI. "
        "Agents: Research Planner → Internet Researcher → Fact Checker → Report Writer."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router, tags=["Health"])
app.include_router(research_router, prefix="/api/v1/research", tags=["Research"])
app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["Jobs"])
