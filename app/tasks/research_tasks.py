"""
CrewAI Task definitions — loaded from config/tasks.yaml.

The write_final_report task now has:
  - write_report_guardrail: enforces Summary / Insights / Citations sections
"""

import os
import yaml
from crewai import Task, Agent

from app.crews.guardrails import write_report_guardrail

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")


def _load_task_config() -> dict:
    config_path = os.path.join(CONFIG_DIR, "tasks.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_tasks(agents: dict[str, Agent]) -> list[Task]:
    """
    Build all four research tasks from YAML config in pipeline order:
    plan → gather → verify → write
    """
    task_config = _load_task_config()

    create_research_plan = Task(
        config=task_config["create_research_plan"],
        agent=agents["research_planner"],
    )

    gather_research_data = Task(
        config=task_config["gather_research_data"],
        agent=agents["researcher"],
    )

    verify_information_quality = Task(
        config=task_config["verify_information_quality"],
        agent=agents["fact_checker"],
    )

    write_final_report = Task(
        config=task_config["write_final_report"],
        agent=agents["report_writer"],
        guardrails=[write_report_guardrail],   # ← new: structural validation
    )

    return [
        create_research_plan,
        gather_research_data,
        verify_information_quality,
        write_final_report,
    ]
