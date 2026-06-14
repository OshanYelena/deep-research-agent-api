"""
Jobs management endpoints.

GET    /api/v1/jobs/              → list all jobs (paginated)
GET    /api/v1/jobs/{job_id}      → get a specific job's status + result
DELETE /api/v1/jobs/{job_id}      → delete a job from the store
"""

from fastapi import APIRouter, HTTPException, Query

from app.core.job_store import job_store
from app.core.models import JobListResponse, JobStatusResponse

router = APIRouter()


@router.get(
    "/",
    response_model=JobListResponse,
    summary="List all research jobs",
)
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> JobListResponse:
    jobs = await job_store.list_jobs(limit=limit, offset=offset)
    total = await job_store.count()
    return JobListResponse(
        total=total,
        jobs=[JobStatusResponse(**job.model_dump()) for job in jobs],
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get research job status and result",
)
async def get_job(job_id: str) -> JobStatusResponse:
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatusResponse(**job.model_dump())


@router.delete(
    "/{job_id}",
    summary="Delete a research job",
    status_code=204,
)
async def delete_job(job_id: str) -> None:
    deleted = await job_store.delete(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
