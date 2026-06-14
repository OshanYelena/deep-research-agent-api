"""
Research API endpoints.

POST /api/v1/research/          → submit a new research job
GET  /api/v1/research/{job_id}/stream → SSE stream of job progress + final result
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.core.job_store import job_store
from app.core.models import (
    AgentStage,
    JobStatus,
    ResearchRequest,
    ResearchResponse,
    SSEEvent,
)
from app.crews.research_crew import run_research_job

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Submit Job ────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ResearchResponse,
    summary="Submit a new research job",
    description=(
        "Submits a research query. The crew runs asynchronously in the background. "
        "Poll `/api/v1/jobs/{job_id}` for status or subscribe to "
        "`/api/v1/research/{job_id}/stream` for live SSE updates."
    ),
    status_code=202,
)
async def submit_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    job = await job_store.create(query=request.query)

    background_tasks.add_task(
        run_research_job,
        job_id=job.job_id,
        query=request.query,
    )

    logger.info(f"Research job {job.job_id} queued for query: {request.query[:80]}...")

    return ResearchResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        message=(
            f"Research job accepted. Track progress at "
            f"/api/v1/jobs/{job.job_id} or stream events at "
            f"/api/v1/research/{job.job_id}/stream"
        ),
    )


# ── SSE Stream ────────────────────────────────────────────────────────────────

@router.get(
    "/{job_id}/stream",
    summary="Stream job progress via Server-Sent Events",
    description=(
        "Opens an SSE connection. Events emitted: `stage_started`, `stage_completed`, "
        "`completed`, `error`. The stream closes automatically when the job finishes."
    ),
)
async def stream_job_progress(job_id: str) -> StreamingResponse:
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return StreamingResponse(
        _sse_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",          # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


async def _sse_generator(job_id: str):
    """
    Async generator that polls job_store and yields SSE-formatted events.
    Emits a heartbeat every 15 s so proxies don't close idle connections.
    """
    POLL_INTERVAL = 1.0        # seconds between status polls
    HEARTBEAT_INTERVAL = 15.0  # seconds between keep-alive pings
    last_heartbeat = asyncio.get_event_loop().time()
    last_stage_seen: set[AgentStage] = set()
    final_sent = False

    def _format_sse(event: SSEEvent) -> str:
        data = event.model_dump(exclude_none=True)
        return f"event: {event.event}\ndata: {json.dumps(data)}\n\n"

    def _heartbeat() -> str:
        return f": heartbeat {datetime.now(timezone.utc).isoformat()}\n\n"

    while True:
        now = asyncio.get_event_loop().time()

        # Heartbeat
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            yield _heartbeat()
            last_heartbeat = now

        job = await job_store.get(job_id)
        if not job:
            yield _format_sse(SSEEvent(
                event="error",
                job_id=job_id,
                message="Job not found.",
            ))
            return

        # Emit any new stage events
        for stage_progress in job.stages:
            stage = stage_progress.stage

            # stage_started
            stage_key_start = f"start:{stage}"
            if stage_key_start not in last_stage_seen and stage_progress.started_at:
                last_stage_seen.add(stage_key_start)
                yield _format_sse(SSEEvent(
                    event="stage_started",
                    job_id=job_id,
                    stage=stage,
                    message=f"Stage started: {stage}",
                ))

            # stage_completed
            stage_key_done = f"done:{stage}"
            if stage_key_done not in last_stage_seen and stage_progress.completed_at:
                last_stage_seen.add(stage_key_done)
                yield _format_sse(SSEEvent(
                    event="stage_completed",
                    job_id=job_id,
                    stage=stage,
                    message=stage_progress.output_summary,
                ))

        # Terminal states
        if job.status == JobStatus.COMPLETED and not final_sent:
            final_sent = True
            yield _format_sse(SSEEvent(
                event="completed",
                job_id=job_id,
                stage=AgentStage.DONE,
                result=job.result,
                message="Research completed successfully.",
            ))
            return

        if job.status == JobStatus.FAILED and not final_sent:
            final_sent = True
            yield _format_sse(SSEEvent(
                event="error",
                job_id=job_id,
                message=job.error or "Research job failed.",
            ))
            return

        await asyncio.sleep(POLL_INTERVAL)
