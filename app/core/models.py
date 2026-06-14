"""
Pydantic models — request/response schemas and internal job state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStage(str, Enum):
    PLANNING = "planning"
    RESEARCHING = "researching"
    FACT_CHECKING = "fact_checking"
    WRITING = "writing"
    DONE = "done"


# ── Internal Job State ─────────────────────────────────────────────────────────

class StageProgress(BaseModel):
    stage: AgentStage
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_summary: Optional[str] = None


class Job(BaseModel):
    job_id: str
    query: str
    status: JobStatus = JobStatus.PENDING
    current_stage: Optional[AgentStage] = None
    stages: List[StageProgress] = Field(default_factory=list)
    result: Optional[str] = None          # final markdown report
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


# ── API Request / Response Schemas ─────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The research question to investigate.",
        examples=[
            "Evaluate the top five emerging AI tools for automating competitive "
            "market analysis, including their features, limitations, costs, and "
            "ideal use cases for a mid-sized marketing firm."
        ],
    )


class ResearchResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    query: str
    status: JobStatus
    current_stage: Optional[AgentStage] = None
    stages: List[StageProgress] = []
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    total: int
    jobs: List[JobStatusResponse]


# ── SSE Event Models ───────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    event: str          # "stage_started" | "stage_completed" | "completed" | "error"
    job_id: str
    stage: Optional[AgentStage] = None
    message: Optional[str] = None
    result: Optional[str] = None
