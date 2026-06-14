"""
CrewAI Agent definitions.
All agents are instantiated fresh per-job to avoid state bleed between runs.
"""

from crewai import Agent
from crewai_tools import EXASearchTool, ScrapeWebsiteTool

from app.core.config import settings


def _make_search_tools() -> tuple[EXASearchTool, ScrapeWebsiteTool]:
    """Create tool instances. Called per-crew so each job gets its own instances."""
    exa = EXASearchTool(base_url=settings.EXA_BASE_URL)
    scraper = ScrapeWebsiteTool()
    return exa, scraper


def build_agents() -> dict[str, Agent]:
    """
    Build and return all four research agents.
    Returns a dict keyed by role slug for easy reference.
    """
    exa_tool, scrape_tool = _make_search_tools()

    research_planner = Agent(
        role="Research Planner",
        goal="Analyze queries and break them down into smaller, specific research topics.",
        backstory=(
            "You are a research strategist who excels at breaking down complex questions "
            "into manageable research components. You identify what needs to be researched "
            "and create clear research objectives."
        ),
        verbose=True,
        max_rpm=settings.AGENT_MAX_RPM,
        max_iter=settings.AGENT_MAX_ITER,
    )

    researcher = Agent(
        role="Internet Researcher",
        goal="Research thoroughly all assigned topics",
        backstory=(
            "You are a skilled researcher with experience in online investigation "
            "and data collection. You know how to find reliable sources, extract relevant "
            "information, and always verify facts across multiple sources to avoid "
            "misinformation or hallucination. You never invent facts and always trace "
            "information to its origin."
        ),
        tools=[exa_tool, scrape_tool],
        verbose=True,
        max_rpm=settings.AGENT_MAX_RPM,
        max_iter=settings.AGENT_MAX_ITER,
    )

    fact_checker = Agent(
        role="Fact Checker",
        goal=(
            "Verify data for accuracy, identify inconsistencies, "
            "and flag potential misinformation"
        ),
        backstory=(
            "You are a quality assurance specialist with expertise in fact-checking "
            "and identifying misinformation and hallucinations. You cross-reference "
            "information, spot inconsistencies, and ensure all data meets high accuracy "
            "standards. You rigorously check for hallucinated or invented content and "
            "require that all facts be supported by evidence."
        ),
        tools=[exa_tool, scrape_tool],
        verbose=True,
        max_rpm=settings.AGENT_MAX_RPM,
        max_iter=settings.AGENT_MAX_ITER,
    )

    report_writer = Agent(
        role="Report Writer",
        goal="Write clear, concise, and well-structured reports based on gathered information",
        backstory=(
            "You are an expert writer who specializes in creating clear, well-structured "
            "research reports. You synthesize complex information into readable formats and "
            "always include proper citations and sources."
        ),
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
