"""
Health check endpoint — used by load balancers and monitoring.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.job_store import job_store

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    active_jobs: int


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        active_jobs=await job_store.count(),
    )
