"""
CrewAI Agent definitions — loaded from config/agents.yaml.

Agents are instantiated fresh per-job to avoid state bleed between runs.
The report_writer agent now includes the CustomPlotTool.
"""

import os
import yaml
from crewai import Agent
from crewai_tools import EXASearchTool, ScrapeWebsiteTool

from app.core.config import settings
from app.tools.plot_tool import CustomPlotTool

# Path to config dir relative to project root (where uvicorn is run from)
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")


def _load_agent_config() -> dict:
    config_path = os.path.join(CONFIG_DIR, "agents.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _make_search_tools() -> tuple[EXASearchTool, ScrapeWebsiteTool]:
    """Create tool instances. Called per-crew so each job gets its own instances."""
    exa = EXASearchTool(base_url=settings.EXA_BASE_URL)
    scraper = ScrapeWebsiteTool()
    return exa, scraper


def build_agents() -> dict[str, Agent]:
    """
    Build and return all four research agents from YAML config.
    Returns a dict keyed by config slug.
    """
    agent_config = _load_agent_config()
    exa_tool, scrape_tool = _make_search_tools()

    research_planner = Agent(
        config=agent_config["research_planner"],
        verbose=True,
        max_rpm=settings.AGENT_MAX_RPM,
        max_iter=settings.AGENT_MAX_ITER,
    )

    researcher = Agent(
        config=agent_config["internet_researcher"],
        tools=[exa_tool, scrape_tool],
        verbose=True,
        max_rpm=settings.AGENT_MAX_RPM,
        max_iter=settings.AGENT_MAX_ITER,
    )

    fact_checker = Agent(
        config=agent_config["fact_checker"],
        tools=[exa_tool, scrape_tool],
        verbose=True,
        max_rpm=settings.AGENT_MAX_RPM,
        max_iter=settings.AGENT_MAX_ITER,
    )

    report_writer = Agent(
        config=agent_config["report_writer"],
        tools=[CustomPlotTool()],          # ← new: auto-chart generation
        verbose=True,
        max_rpm=settings.AGENT_MAX_RPM,
        max_iter=settings.AGENT_MAX_ITER,
    )

    return {
        "research_planner": research_planner,
        "researcher": researcher,
        "fact_checker": fact_checker,
        "report_writer": report_writer,
    }
