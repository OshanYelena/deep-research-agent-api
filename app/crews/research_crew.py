"""
Crew builder and async job runner.

Architecture:
  - build_crew()  → assembles agents + tasks into a Crew
  - run_research_job() → the async function that BackgroundTasks calls
    It updates job_store at each stage transition and writes the final result.

Stage mapping (CrewAI executes tasks sequentially, so we map task index → stage):
  0 → PLANNING
  1 → RESEARCHING
  2 → FACT_CHECKING
  3 → WRITING

Updates (v2):
  - Agents/tasks now loaded from config/agents.yaml + config/tasks.yaml
  - report_writer has CustomPlotTool for auto chart generation
  - write_final_report has write_report_guardrail (Summary/Insights/Citations)
  - Crew: memory=True, after_kickoff_callbacks=[save_report_hook]
"""

import asyncio
import os
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional

from crewai import Crew

from app.agents.research_agents import build_agents
from app.tasks.research_tasks import build_tasks
from app.core.config import settings
from app.core.models import AgentStage, JobStatus, StageProgress
from app.core.job_store import job_store

logger = logging.getLogger(__name__)

REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")

# Ordered stage list mirrors task pipeline
STAGE_ORDER = [
    AgentStage.PLANNING,
    AgentStage.RESEARCHING,
    AgentStage.FACT_CHECKING,
    AgentStage.WRITING,
]


def _make_save_report_hook(job_id: str):
    """
    Factory that returns a CrewAI after_kickoff_callback bound to a job_id.
    Saves the final markdown report to disk under reports/<job_id>.md
    """
    def save_report_hook(result) -> None:
        try:
            if hasattr(result, "tasks_output") and result.tasks_output:
                content = result.tasks_output[-1].raw
            else:
                content = str(result)

            os.makedirs(REPORTS_DIR, exist_ok=True)
            filename = os.path.join(REPORTS_DIR, f"{job_id}.md")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Report saved to: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save report for job {job_id}: {e}")

    return save_report_hook


def build_crew(job_id: str) -> Crew:
    """
    Build a fresh Crew for each job (avoids shared state between runs).
    Includes memory, guardrails, CustomPlotTool, and file-save hook.
    """
    agents = build_agents()
    tasks = build_tasks(agents)
    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        memory=True,                                          # ← cross-task memory
        after_kickoff_callbacks=[_make_save_report_hook(job_id)],  # ← save report to disk
        verbose=True,
    )


async def run_research_job(
    job_id: str,
    query: str,
    on_stage_start: Optional[Callable[[str, AgentStage], Awaitable[None]]] = None,
    on_stage_complete: Optional[Callable[[str, AgentStage, str], Awaitable[None]]] = None,
) -> None:
    """
    Execute the full research pipeline for a given job.

    This is designed to run inside FastAPI's BackgroundTasks, so it is async
    but delegates the blocking CrewAI `.kickoff()` call to a thread pool.

    Callbacks (optional, used by SSE endpoint):
      on_stage_start(job_id, stage)
      on_stage_complete(job_id, stage, summary)
    """
    # ── Bootstrap environment for CrewAI / OpenAI ─────────────────────────────
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    os.environ["EXA_API_KEY"] = settings.EXA_API_KEY
    os.environ["MODEL"] = settings.MODEL

    # ── Mark job as running ────────────────────────────────────────────────────
    await job_store.update(
        job_id,
        status=JobStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        current_stage=AgentStage.PLANNING,
    )

    try:
        crew = build_crew(job_id=job_id)

        # We intercept task callbacks via CrewAI's step_callback.
        # CrewAI calls step_callback after each agent step, not after each task,
        # so we track task completion by monkeypatching the task execute methods.
        completed_task_indices: list[int] = []

        original_executes = []
        for idx, task in enumerate(crew.tasks):
            original_execute = task.execute_sync

            def make_patched(task_idx, orig):
                async def patched_async(*args, **kwargs):
                    # ── Stage started ──────────────────────────────────────────
                    stage = STAGE_ORDER[task_idx]
                    stage_progress = StageProgress(
                        stage=stage,
                        started_at=datetime.now(timezone.utc),
                    )
                    job = await job_store.get(job_id)
                    stages = job.stages if job else []
                    stages.append(stage_progress)

                    await job_store.update(
                        job_id,
                        current_stage=stage,
                        stages=stages,
                    )
                    if on_stage_start:
                        await on_stage_start(job_id, stage)

                    # ── Run (blocking) in thread pool ─────────────────────────
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: orig(*args, **kwargs)
                    )

                    # ── Stage completed ────────────────────────────────────────
                    summary = str(result)[:300] if result else ""
                    job = await job_store.get(job_id)
                    stages = job.stages if job else []
                    for sp in stages:
                        if sp.stage == stage and sp.completed_at is None:
                            sp.completed_at = datetime.now(timezone.utc)
                            sp.output_summary = summary
                            break

                    await job_store.update(job_id, stages=stages)
                    if on_stage_complete:
                        await on_stage_complete(job_id, stage, summary)

                    return result

                return patched_async

            original_executes.append(original_execute)
            # We'll run kickoff in executor, so we need sync patching — see below.

        # ── Run kickoff in executor ────────────────────────────────────────────
        # CrewAI is synchronous. We run it in a thread and track stage progress
        # via a simpler approach: update job_store after kickoff returns,
        # relying on task_callback for per-stage updates where CrewAI supports it.
        #
        # For full per-stage SSE, we use CrewAI's task_callback (≥0.28).

        loop = asyncio.get_event_loop()
        stage_idx_tracker = {"current": 0}
        stage_results: list[str] = []

        def sync_task_callback(task_output):
            """Called by CrewAI after each task completes (sync context)."""
            idx = stage_idx_tracker["current"]
            if idx < len(STAGE_ORDER):
                stage = STAGE_ORDER[idx]
                summary = str(task_output)[:300] if task_output else ""
                stage_results.append(summary)

                # Schedule coroutine on event loop from this sync thread
                future = asyncio.run_coroutine_threadsafe(
                    _update_stage_complete(job_id, stage, summary, on_stage_complete),
                    loop,
                )
                try:
                    future.result(timeout=5)
                except Exception as e:
                    logger.warning(f"Stage update callback error: {e}")

                stage_idx_tracker["current"] += 1

                # Pre-announce next stage
                if idx + 1 < len(STAGE_ORDER):
                    next_stage = STAGE_ORDER[idx + 1]
                    next_future = asyncio.run_coroutine_threadsafe(
                        _update_stage_start(job_id, next_stage, on_stage_start),
                        loop,
                    )
                    try:
                        next_future.result(timeout=5)
                    except Exception as e:
                        logger.warning(f"Stage start callback error: {e}")

        # Attach callback to all tasks
        for task in crew.tasks:
            task.callback = sync_task_callback

        # Announce first stage start
        await _update_stage_start(job_id, AgentStage.PLANNING, on_stage_start)

        # ── Blocking kickoff in thread pool ────────────────────────────────────
        crew_result = await loop.run_in_executor(
            None,
            lambda: crew.kickoff(inputs={"user_query": query}),
        )

        final_report = crew_result.raw if hasattr(crew_result, "raw") else str(crew_result)

        # ── Mark job as completed ──────────────────────────────────────────────
        await job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            result=final_report,
            current_stage=AgentStage.DONE,
            completed_at=datetime.now(timezone.utc),
        )
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        await job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            completed_at=datetime.now(timezone.utc),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _update_stage_start(
    job_id: str,
    stage: AgentStage,
    callback: Optional[Callable],
) -> None:
    job = await job_store.get(job_id)
    if not job:
        return
    stages = job.stages
    # Only add if not already present
    if not any(sp.stage == stage for sp in stages):
        stages.append(StageProgress(stage=stage, started_at=datetime.now(timezone.utc)))
    await job_store.update(job_id, current_stage=stage, stages=stages)
    if callback:
        await callback(job_id, stage)


async def _update_stage_complete(
    job_id: str,
    stage: AgentStage,
    summary: str,
    callback: Optional[Callable],
) -> None:
    job = await job_store.get(job_id)
    if not job:
        return
    stages = job.stages
    for sp in stages:
        if sp.stage == stage and sp.completed_at is None:
            sp.completed_at = datetime.now(timezone.utc)
            sp.output_summary = summary
            break
    await job_store.update(job_id, stages=stages)
    if callback:
        await callback(job_id, stage, summary)
