"""
In-memory job store.
Thread-safe using asyncio.Lock. Evicts oldest jobs when MAX_JOBS_IN_MEMORY is hit.
"""

import asyncio
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

from app.core.models import Job, JobStatus
from app.core.config import settings


class JobStore:
    def __init__(self):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = asyncio.Lock()

    async def create(self, query: str) -> Job:
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._jobs) >= settings.MAX_JOBS_IN_MEMORY:
                self._jobs.popitem(last=False)

            job = Job(
                job_id=str(uuid.uuid4()),
                query=query,
                status=JobStatus.PENDING,
                created_at=datetime.now(timezone.utc),
            )
            self._jobs[job.job_id] = job
            return job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs) -> Optional[Job]:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in kwargs.items():
                setattr(job, key, value)
            return job

    async def list_jobs(self, limit: int = 20, offset: int = 0) -> list[Job]:
        async with self._lock:
            jobs = list(reversed(list(self._jobs.values())))
            return jobs[offset : offset + limit]

    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    def clear(self):
        self._jobs.clear()

    async def count(self) -> int:
        async with self._lock:
            return len(self._jobs)


# Singleton instance
job_store = JobStore()
