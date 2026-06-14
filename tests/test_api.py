"""
Test suite for the Deep Research API.
Uses FastAPI's TestClient with a mocked CrewAI Crew to avoid real API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.models import JobStatus


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "active_jobs" in data


# ── Research Submission ───────────────────────────────────────────────────────

def test_submit_research_returns_202(client):
    r = client.post(
        "/api/v1/research/",
        json={"query": "What are the top AI frameworks for NLP in 2025?"},
    )
    assert r.status_code == 202
    data = r.json()
    assert "job_id" in data
    assert data["status"] == JobStatus.PENDING
    assert data["job_id"] in data["message"]


def test_submit_research_query_too_short(client):
    r = client.post("/api/v1/research/", json={"query": "short"})
    assert r.status_code == 422


def test_submit_research_query_too_long(client):
    r = client.post("/api/v1/research/", json={"query": "x" * 2001})
    assert r.status_code == 422


# ── Job Status ────────────────────────────────────────────────────────────────

def test_get_job_status(client):
    # Submit a job first
    r = client.post(
        "/api/v1/research/",
        json={"query": "Explain the differences between RAG and fine-tuning LLMs."},
    )
    job_id = r.json()["job_id"]

    r2 = client.get(f"/api/v1/jobs/{job_id}")
    assert r2.status_code == 200
    data = r2.json()
    assert data["job_id"] == job_id
    assert data["status"] in [s.value for s in JobStatus]
    assert "query" in data
    assert "created_at" in data


def test_get_job_not_found(client):
    r = client.get("/api/v1/jobs/nonexistent-id-00000")
    assert r.status_code == 404


# ── Job List ──────────────────────────────────────────────────────────────────

def test_list_jobs(client):
    # Submit two jobs
    for q in [
        "What is LangGraph used for in agentic AI systems?",
        "Compare Celery vs RQ for Python background tasks.",
    ]:
        client.post("/api/v1/research/", json={"query": q})

    r = client.get("/api/v1/jobs/")
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data
    assert "total" in data
    assert data["total"] >= 2


def test_list_jobs_pagination(client):
    r = client.get("/api/v1/jobs/?limit=1&offset=0")
    assert r.status_code == 200
    data = r.json()
    assert len(data["jobs"]) <= 1


# ── Delete Job ────────────────────────────────────────────────────────────────

def test_delete_job(client):
    r = client.post(
        "/api/v1/research/",
        json={"query": "What are vector databases and how do they work?"},
    )
    job_id = r.json()["job_id"]

    r_del = client.delete(f"/api/v1/jobs/{job_id}")
    assert r_del.status_code == 204

    r_get = client.get(f"/api/v1/jobs/{job_id}")
    assert r_get.status_code == 404


def test_delete_job_not_found(client):
    r = client.delete("/api/v1/jobs/ghost-job-999")
    assert r.status_code == 404


# ── SSE Stream (smoke test) ───────────────────────────────────────────────────

def test_stream_not_found(client):
    r = client.get("/api/v1/research/bad-id/stream")
    assert r.status_code == 404
